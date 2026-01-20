import argparse
import numpy as np
import time
import math
import cmnd_benders_utils_single_cut
from gurobipy import GRB
import gurobipy as gp
import pandas as pd
import logging
import os
from tabulate import tabulate

"""
This file does benders decomposition for the network
design instances using vanilla Benders (no information reuse).
"""


class Benders(cmnd_benders_utils_single_cut.CMNDinst):

    def __init__(
        self,
        algorithm_details,
        problemfile,
        n_scen,
    ):
        """Intializes and creates the instance data"""

        super().__init__(problemfile, n_scen)

        solve_ip, dual_lookup, init = algorithm_details
        self.solve_ip = solve_ip

        # Initialize lp_active to store active cuts from LP phase (for transfer to IP)
        self.lp_active = {}

        self.master_lp_times = []

        self.subproblem_lp_times = []
        self.subproblem_ip_times = []

        self.cut_time_lp = []
        self.cut_time_ip = []

        self.subproblem_lp_cuts = []
        self.subproblem_ip_cuts = []

        self.subproblem_ip_counts = []
        self.subproblem_counts_lp = []

        self.initial_constraint_counter = []
        self.x_feas_counter_lp = []
        self.x_feas_counter_ip = []

        self.lp_iterations = []

        self.lp_first_stage_sols = []
        self.lp_optimal_first_stage_sols = []
        self.ip_first_stage_sols = []
        self.ip_first_stage_optimal_sols = []

        self.lp_relaxation_time = []
        self.ip_time = []

        self.total_times = []

        self.lp_final_cons = []
        self.lp_cons_to_ip = []
        self.ip_gap = []
        self.ip_nodes = []

        self.root_node_gap = []
        self.tol = 1e-5

        # Hash set to track unique IP solutions
        self.int_hash_set = set()

    def benders(self, saa_iteration):
        """
        Main Benders decomposition algorithm implementation for CMND.

        Performs either LP relaxation only or full LP+IP solve depending on configuration.
        Uses vanilla Benders with no information reuse across SAA iterations.

        Args:
            saa_iteration (int): Current SAA iteration number (1-indexed)

        Returns:
            int: 0 on successful completion
        """
        # Build master problem with LP relaxation for initial phase
        self.master = self.build_master(relaxation=True)
        self.master.setParam("OutputFlag", False)

        # Initialize cut tracking structures for current SAA iteration - SINGLE CUT VERSION
        # For single-cut, we track sets of dual solution tuples (one dual per scenario)
        self.cuts_to_add_to_ip = set()  # Set of tuples, each tuple has one dual ID per scenario
        self.cut_history = []  # List of dual ID lists used in aggregated cuts

        # ==================== LP RELAXATION PHASE ====================
        problem_start = time.perf_counter()
        self.solve_lp_relaxation()
        self.lp_relaxation_time.append(round(time.perf_counter() - problem_start, 3))
        self.x_feas_counter_lp.append(len(self.lp_first_stage_sols))

        # If configured for LP-only, terminate here
        if not self.solve_ip:
            self.lp_cons_to_ip.append(0)
            return 0

        # ==================== IP PHASE PREPARATION ====================
        # Initialize solution value containers
        self.first_stage_values = {}  # First-stage design variables
        self.second_stage_values = {}  # Second-stage cost variable

        # Rebuild master problem for IP phase (with binary variables and active LP cuts)
        self.ip_initialize()

        # Initialize IP phase performance counters
        self.subproblem_ip_solve_time = 0
        self.subproblem_ip_cuts_generated = 0
        self.subproblem_ip_solve_count = 0

        def benders_callback(model, where):
            """
            Gurobi callback function for lazy constraint generation during IP solve.
            """

            # Track root node performance metrics
            if where == GRB.Callback.MIPNODE:
                depth = model.cbGet(GRB.Callback.MIPNODE_NODCNT)

                # Capture root node statistics when it's solved to optimality
                if depth == 0:
                    if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.Status.OPTIMAL:
                        objbnd = model.cbGet(GRB.Callback.MIPNODE_OBJBND)
                        objval = model.cbGet(GRB.Callback.MIPNODE_OBJBST)
                        model._rootgap = abs(objbnd - objval) * 100 / abs(objval)

            # Generate Benders cuts for integer solutions
            if where == GRB.Callback.MIPSOL:
                # Extract current integer solution
                self.first_stage_values = model.cbGetSolution(self.x)
                # For single-cut, z is a scalar variable
                self.second_stage_values = model.cbGetSolution(self.z)

                # Track unique first-stage solutions encountered during IP solve
                x_order = tuple(int(self.first_stage_values[arc]) for arc in self.arcs)
                int_hash = hash(x_order)

                if int_hash not in self.int_hash_set:
                    self.int_hash_set.add(int_hash)
                    self.ip_first_stage_sols.append(x_order)

                # Generate cuts by solving subproblems
                start = time.perf_counter()
                cutadded, zup = self.add_single_cut(model, lazy=True)
                model.cbSetSolution(self.x, self.first_stage_values)
                model.cbSetSolution(self.z, zup)
                self.subproblem_ip_solve_time += time.perf_counter() - start
                self.subproblem_ip_cuts_generated += cutadded
                self.subproblem_ip_solve_count += 1

        # ==================== IP MASTER PROBLEM SOLVE ====================
        # Configure Gurobi for lazy constraint generation
        self.master.setParam("OutputFlag", True)
        self.master.setParam("MIPGap", 1e-3)
        self.master.Params.lazyConstraints = 1
        self.master._rootgap = 1000

        # Solve IP master problem with Benders callback
        self.master.optimize(benders_callback)

        # Record root node performance
        self.root_node_gap.append(self.master._rootgap)

        self.extract_solution_values()
        self.dual_update()

        self.ip_time.append(round(self.master.Runtime, 3))
        logger.info(f"IP bound: {self.master.ObjBound:.3f}")

        print(f"IP bound: {self.master.ObjBound:.3f}")
        print()
        self.initial_constraint_counter.append(self.master.NumConstrs)

        self.ip_gap.append(round(self.master.MIPGap * 100, 4))
        self.ip_nodes.append(self.master.NodeCount)

        x_order = tuple(self.first_stage_values[arc] for arc in self.arcs)

        if x_order not in self.ip_first_stage_optimal_sols:
            self.ip_first_stage_optimal_sols.append(x_order)

        if x_order not in self.ip_first_stage_sols:
            self.ip_first_stage_sols.append(x_order)

        self.x_feas_counter_ip.append(len(self.ip_first_stage_sols))

        self.subproblem_ip_times.append(round(self.subproblem_ip_solve_time, 2))
        self.cut_time_ip.append(round(self.subproblem_ip_solve_time, 2))
        self.subproblem_ip_cuts.append(self.subproblem_ip_cuts_generated)
        self.subproblem_ip_counts.append(self.subproblem_ip_solve_count)

        return 0


    def solve_lp_relaxation(self):
        """
        Solve LP relaxation of master problem using iterative vanilla Benders decomposition.
        """
        # Initialize performance tracking variables
        subproblem_lp_solve_time = 0
        subproblem_lp_cuts_generated = 0
        master_lp_solve_time = 0
        lp_iterations = 0
        upper_bound = np.inf
        subproblem_lp_solve_count = 0

        # ==================== INITIAL MASTER SOLVE ====================
        start = time.perf_counter()
        self.master.optimize()
        master_lp_solve_time += time.perf_counter() - start
        lp_iterations += 1

        # Verify master problem solved successfully
        status = self.master.status
        if status != 2:
            raise Exception(f"Master problem status - {status}")

        # Build subproblem structure and extract initial solution
        self.subproblem = self.build_SP()
        self.extract_solution_values(problem="LP")

        # Initialize upper bound tracking
        self.scenario_upper_bounds = {
            scenario: math.inf for scenario in self.scenario
        }
        upper_bound = np.inf

        # Initialize bounds for convergence check
        lower_bound = self.master.ObjBound
        optimality_gap = upper_bound - lower_bound
        num_cuts_added = 1  # Force initial iteration

        # ==================== MAIN BENDERS LOOP ====================
        while True:
            num_cuts_added = 0

            # Solve subproblems to generate single aggregated cut
            start = time.perf_counter()
            cuts, self.zup = self.add_single_cut(self.master, lazy=False)
            ub = self.calculate_upper_bound_from_subproblems()
            upper_bound = min(ub, upper_bound)

            num_cuts_added = cuts
            subproblem_lp_solve_time += time.perf_counter() - start
            subproblem_lp_cuts_generated += cuts
            subproblem_lp_solve_count += 1

            # Handle numerical issues with zero lower bound
            if lower_bound == 0:
                lower_bound = 0.1

            # Check convergence criteria: no cuts added or optimality gap sufficiently small
            if num_cuts_added == 0 or (optimality_gap < self.gaplimit):
                ub = self.calculate_upper_bound_from_subproblems()
                upper_bound = min(ub, upper_bound)
                logger.info(
                    f"Iteration {lp_iterations}: LP UB: {upper_bound:.3f}, LB: {lower_bound:.3f}"
                )
                print(
                    f"Iteration {lp_iterations}: LP UB: {upper_bound:.3f}, LB: {lower_bound:.3f}"
                )
                break  # LP relaxation converged

            # Calculate percentage optimality gap and continue iterating
            optimality_gap = (upper_bound - lower_bound) / lower_bound

            # Resolve master problem with new cuts
            start = time.perf_counter()
            self.master.optimize()
            master_lp_solve_time += time.perf_counter() - start
            print(
                f"Iteration {lp_iterations}: LP UB: {upper_bound:.3f}, LB: {lower_bound:.3f}"
            )

            # Check master problem status
            status = self.master.status
            if status != 2:
                raise Exception(f"Master problem status - {status}")

            # Update bounds and solution tracking
            lower_bound = self.master.ObjBound

            self.extract_solution_values(problem="LP")
            x_order = np.array([self.first_stage_values[arc] for arc in self.arcs])
            self.lp_first_stage_sols.append(x_order)
            lp_iterations += 1

        # ==================== LP PHASE COMPLETION ====================
        # Record final constraint count and perform consistency checks
        self.lp_final_cons.append(self.master.NumConstrs)

        # Update dual solution data structures
        self.dual_update()

        # Only identify active cuts if we're solving IP (needed for warm-starting IP)
        if self.solve_ip:
            # Identify which cuts are active (binding) at the LP optimal solution
            # For single-cut, this returns a list of dual ID lists
            self.lp_active = self.identify_active_cuts()

            # Prepare cuts to transfer to IP phase (copy of active cuts)
            # For single-cut, convert list to set of tuples
            self.cuts_to_add_to_ip = set()
            for dual_list in self.lp_active:
                if isinstance(dual_list, (list, tuple)):
                    self.cuts_to_add_to_ip.add(tuple(dual_list))

        # Record performance metrics
        self.lp_iterations.append(lp_iterations)
        x_order = np.array([self.first_stage_values[arc] for arc in self.arcs])
        # Always add to optimal solutions (this represents the final optimal LP solution)
        self.lp_optimal_first_stage_sols.append(x_order)

        self.subproblem_lp_times.append(round(subproblem_lp_solve_time, 2))
        self.cut_time_lp.append(round(subproblem_lp_solve_time, 2))
        self.subproblem_lp_cuts.append(subproblem_lp_cuts_generated)
        self.master_lp_times.append(round(master_lp_solve_time, 2))
        self.subproblem_counts_lp.append(subproblem_lp_solve_count)

    def add_cuts_to_master(self, initialize_set, check_lp_cuts=True):
        """Add aggregated cuts to master for single-cut initialization

        Args:
            initialize_set: Set of tuples, each tuple containing dual solution indices for all scenarios
        """
        cut_count = 0

        for duals_tuple in initialize_set:
            cut_expr = gp.LinExpr()
            if check_lp_cuts:
                # Convert duals_tuple to list for comparison with cuts_to_add_to_ip
                duals_list = list(duals_tuple)
                if duals_list in self.cuts_to_add_to_ip:
                    continue
            for scen_idx, scenario in enumerate(self.scenario):
                dual_id = duals_tuple[scen_idx]

                # Add this scenario's contribution to the aggregated cut
                cut_expr += gp.quicksum(
                    self.capacity[(i, j)] * self.x[i, j] * self.H[dual_id][arc_idx]
                    for arc_idx, (i, j) in enumerate(self.arcs)
                ) + gp.quicksum(
                    self.demand[scenario][k]
                    * (self.PI[dual_id][k * 2] - self.PI[dual_id][k * 2 + 1])
                    for k in range(self.nK)
                )

            self.master.addConstr(cut_expr <= self.z)
            cut_count += 1

        return cut_count


    def ip_initialize(self):
        """
        Initialize IP master problem with active cuts from LP phase (vanilla Benders).

        This method:
        1. Completely rebuilds the master problem with binary variables
        2. Adds back the active Benders cuts from LP phase
        """
        # Completely rebuild master problem with binary variables
        self.master = self.build_master(relaxation=False)
        self.master.setParam("TimeLimit", 3600)

        # Rebuild subproblem at the start of IP phase
        self.subproblem = self.build_SP()

        # Add active Benders cuts from LP phase to IP master
        lp_cons = self.add_cuts_to_master(self.cuts_to_add_to_ip, check_lp_cuts=False)
        self.lp_cons_to_ip.append(lp_cons)  # Track cuts transferred from LP to IP
        self.master.update()  # Force Gurobi to update internal state

    def dual_update(self):
        """
        Update dual solution data structures.

        This method prepares dual arrays from the current SAA iteration's dual storage.
        """
        self.H = np.array(self.H_)
        self.PI = np.array(self.PI_)
        self.dual_obj_random = np.matmul(
            self.PI, self.repeated_demand.T
        )

    def identify_active_cuts(self):
        """
        Identify active cuts for single-cut Benders decomposition.

        In single-cut, each cut aggregates across all scenarios, so we need to check
        which cuts from cut_history have their LHS equal to RHS (within tolerance).

        Returns:
            list: Indices of duals defining active cuts from cut_history
        """
        if not hasattr(self, "cut_history") or not self.cut_history:
            return []

        active_cuts = []
        active_tol = 1e-10

        # Current second-stage cost (RHS of cuts)
        current_z = (
            self.second_stage_values
            if hasattr(self, "second_stage_values")
            else self.z.X
        )

        dualctx = np.array(
            [
                self.capacity[(i, j)] * self.first_stage_values[(i, j)]
                for (i, j) in self.arcs
            ]
        )
        s1 = np.matmul(self.H, dualctx)
        s1 = s1.reshape(-1, 1)

        subproblem_evaluations = s1 + self.dual_obj_random
        assert subproblem_evaluations.shape == (len(self.H), self.nS)

        # Vectorized evaluation of all cuts from history
        for cut_dual_ids in self.cut_history:
            # Use precomputed subproblem_evaluations to get LHS efficiently
            # Sum across scenarios for this aggregated cut
            cut_lhs = sum(
                subproblem_evaluations[dual_id, scenario_idx]
                for scenario_idx, dual_id in enumerate(cut_dual_ids)
            )

            # Check if cut is active (LHS approximately equals RHS)
            if abs(cut_lhs - current_z) <= active_tol:
                active_cuts.append(cut_dual_ids)

        return active_cuts


    def solve_saa_iteration(self, saa_iteration):

        start = time.perf_counter()

        self.master = self.build_master(relaxation=True)
        if saa_iteration == 1:
            self.master.setParam("TimeLimit", 36000)  # 10 hours for first SAA
        else:
            self.master.setParam("TimeLimit", 3600)  # 1 hour for subsequent SAAs
        self.master.setParam("OutputFlag", False)
        self.master.setParam("LogToConsole", 0)

        self.benders(saa_iteration)

        del self.master
        stop = time.perf_counter()
        self.total_times.append(round(stop - start, 2))

        logger.info(f"SAA {saa_iteration} done!")


def setup_logger(fname):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(fname, mode="a")
    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger




if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="CMND Benders Decomposition - Vanilla (No Information Reuse) - SINGLE CUT"
    )
    parser.add_argument("algorithm_details", metavar="N", type=int, nargs=3,
                       help="Three integers: solve_ip dual_lookup init (e.g., 1 0 0)")
    parser.add_argument("data", metavar="N", type=str, nargs=3,
                       help="Three strings: problemfile nscen nsaa")
    args = parser.parse_args()

    solve_ip, dual_lookup, init = args.algorithm_details
    problemfile, nscen, nsaa = args.data

    # Algorithm name (single-cut vanilla is always NoReuse)
    algorithm_name = "single_NoReuse"

    # Create data string for filenames
    data = "_".join(args.data)
    data = data.replace("instances-cmnd/", "")
    scenario_filename = problemfile
    scenario_filename = scenario_filename.replace("instances-cmnd/", "")
    scenario_filename = scenario_filename.replace(".dow", "")

    if solve_ip:
        fname = f"detailed-results/cmnd/IP/single_{data}_{algorithm_name}.op"
    else:
        fname = f"detailed-results/cmnd/LP/single_{data}_{algorithm_name}.op"

    print()
    print("Algorithm: Single-Cut Vanilla Benders (No Information Reuse)")

    logger = setup_logger(fname)
    logger.info(f"data: {args.data}")
    logger.info(f"Algorithm: Vanilla Benders (No Information Reuse) - SINGLE CUT")
    logger.info(f"algorithm_details: {args.algorithm_details}")
    logger.info(f"solve_ip: {solve_ip}")

    np.random.seed(3)
    bend = Benders(args.algorithm_details, problemfile, int(nscen))

    n_saa = int(nsaa)
    n_scen = int(nscen)

    saa_solved = []

    # Solve all SAA iterations (vanilla Benders has no information reuse)
    for saa_iteration in range(1, n_saa + 1):
        # Read scenarios from file (same format as solve_cmnd.py)
        if n_scen < 100:
            filename = (
                "instances-cmnd/scenarios/" + scenario_filename + f"_200_{saa_iteration}"
            )
        else:
            filename = (
                "instances-cmnd/scenarios/"
                + scenario_filename
                + f"_{n_scen}_{saa_iteration}"
            )
        bend.read_scenarios(n_scen, filename)
        bend.solve_saa_iteration(saa_iteration)
        saa_solved.append(saa_iteration)

    data_dict = {"Instance": data, "Scenarios": n_scen, "Method": algorithm_name}

    if solve_ip:
        columns = [
            "Total times",
            "IP time",
            "LP relaxation time",
            "IP nodes",
            "Root node gap",
            "subproblem ip counts",
            "cut time IP",
            "x feas counter ip",
            "initial constraint counter",
            "Subproblem IP times",
        ]
    else:
        columns = [
            "Total times",
            "LP iterations",
            "x feas counter lp",
            "LP final cons",
            "subproblem counts lp",
            "subproblem lp cuts",
            "subproblem_lp_times",
            "cut time LP",
        ]

    for col in columns:
        col_data = getattr(bend, col.replace(" ", "_").lower())
        if len(col_data) > 0:
            data_dict[f"{col} SAA 0"] = round(col_data[0], 3)
            if len(col_data) > 1:
                # For NoReuse method, average should include all runs
                data_dict[f"{col} average"] = round(np.mean(col_data), 3)

    df = pd.DataFrame([data_dict])

    df.to_csv(
        "results_single_cmnd.csv",
        index=False,
        quoting=3,
        sep=",",
        escapechar=",",
        mode="a",
        header=not os.path.exists("results_single_cmnd.csv"),
    )

    logger.info("")

    saa_list = [f"SAA {i}" for i in saa_solved]
    logger.info("=" * 80)
    logger.info("DETAILED RESULTS SUMMARY")
    logger.info("=" * 80)

    # Log constraint information
    constraint_data = [
        ["SP cuts - LP", *bend.subproblem_lp_cuts],
        ["SP count - LP", *bend.subproblem_counts_lp],
        ["Final cons LP", *bend.lp_final_cons],
    ]
    logger.info(tabulate(constraint_data, headers=["LP Constraints", *saa_list]))
    logger.info("")

    if solve_ip:
        ip_constraint_data = [
            ["Benders cuts from LP to IP", *bend.lp_cons_to_ip],
            ["SP cuts - IP", *bend.subproblem_ip_cuts],
            ["SP count - IP", *bend.subproblem_ip_counts],
        ]
        logger.info(tabulate(ip_constraint_data, headers=["IP Constraints", *saa_list]))
        logger.info("")

    # Time information
    time_data = [
        ["LP time", *bend.lp_relaxation_time],
        ["IP time", *bend.ip_time],
        ["Total time", *bend.total_times],
    ]
    logger.info(tabulate(time_data, headers=["Time Info", *saa_list]))
    logger.info("")

    # LP time breakdown
    lp_time_data = [
        ["Master time", *bend.master_lp_times],
        ["SP time", *bend.subproblem_lp_times],
        ["Solutions", *bend.x_feas_counter_lp],
    ]
    logger.info(tabulate(lp_time_data, headers=["LP Time", *saa_list]))
    logger.info("")

    if solve_ip:
        # IP time breakdown
        ip_time_data = [
            ["SP time", *bend.subproblem_ip_times],
            ["Root gap", *bend.root_node_gap],
        ]
        logger.info(tabulate(ip_time_data, headers=["IP Time", *saa_list]))
        logger.info("")

        ip_info_data = [
            ["First stage solutions IP", *bend.x_feas_counter_ip],
            ["IP gap %", *bend.ip_gap],
            ["IP nodes", *bend.ip_nodes],
        ]
        logger.info(tabulate(ip_info_data, headers=["IP Info", *saa_list]))
        logger.info("")

    # Summary information
    summary_data = [
        ["LP Iterations", bend.lp_iterations],
    ]
    logger.info(tabulate(summary_data, headers=[]))
    logger.info("")
