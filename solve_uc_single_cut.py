import argparse
import numpy as np
import time
import math
import uc_benders_utils_single_cut as uc_benders_utils  # Single-cut version of utils
from gurobipy import GRB
import gurobipy as gp
import pandas as pd
import logging
import os
from tabulate import tabulate


class AlgorithmDetailsError(Exception):
    """Exception raised for invalid algorithm configuration."""

    pass


"""
Benders Decomposition Implementation for Unit Commitment Problem (UC) - SINGLE CUT VERSION

This implementation features:
1. Sample Average Approximation (SAA) for stochastic programming
2. Dual Solution Pooling (DSP) for accelerated Benders decomposition
3. Multiple initialization techniques (Static/Adaptive/Boosted)
4. Both LP relaxation and IP solving capabilities
5. Comprehensive performance tracking and logging
6. SINGLE-CUT FORMULATION: One aggregated cut across all scenarios instead of per-scenario cuts

The algorithm decomposes the UC problem into:
- Master Problem: First-stage commitment decisions (x variables) with temporal constraints
- Subproblems: Second-stage economic dispatch decisions for each scenario

Key Features:
- DSP: Reuses previously computed dual solutions to avoid redundant subproblem solves
- Curated DSP: Intelligently selects most promising dual solutions
- Static/Adaptive Init: Warm-starts master problem with heuristically selected cuts
- Temporal Constraints: Handles minimum up/down time constraints in master problem
- Single-Cut: Aggregates all scenarios into one cut for tighter LP relaxation
"""


class Benders(uc_benders_utils.UCinst):
    """
    Benders Decomposition solver for Unit Commitment Problem.

    Inherits from UCinst which provides instance data management and basic
    optimization model building capabilities for unit commitment.

    The class implements multiple algorithmic variants:
    - Standard Benders vs. Dual Solution Pooling (DSP)
    - LP relaxation only vs. full IP solving
    - Various initialization techniques for master problem warm-starting
    """

    def __init__(
        self,
        solve_ip,
        instance_file_or_data,
        nscen,
        max_periods=None,
    ):
        """
        Initialize Benders decomposition solver for unit commitment.

        Args:
            solve_ip (int): 0=LP only, 1=solve IP after LP
            instance_file_or_data: Path to UC instance JSON file or instance data dict
            nscen: Number of scenarios to generate
            max_periods: If specified, only use the first max_periods time periods
        """

        super().__init__(nscen)
        self.load_instance(instance_file_or_data, max_periods=max_periods)

        # Build mapping from (generator, time_period) to index in flattened solution vector
        self.gt_index_map = {}
        idx = 0
        for g in self.thermal_gens:
            for t in self.periods:
                self.gt_index_map[(g, t)] = idx
                idx += 1

        # ==================== ALGORITHM CONFIGURATION ====================
        # Core Algorithm Settings - Vanilla Benders (no information reuse across SAAs)
        self.solve_ip = solve_ip

        # Initialize lp_active to store active cuts from LP phase (for transfer to IP)
        self.lp_active = {}

        # ==================== PERFORMANCE TRACKING ====================
        # Timing metrics for different algorithm components
        self.master_lp_times = []
        self.subproblem_lp_times = []
        self.subproblem_ip_times = []
        self.cut_time_lp = []
        self.cut_time_ip = []

        # Work unit tracking for Gurobi computational effort
        self.master_lp_work = []
        self.subproblem_lp_work = []
        self.subproblem_ip_work = []
        self.master_ip_work = []
        self.total_work_lp = []
        self.total_work_ip = []

        # Cut generation statistics
        self.subproblem_lp_cuts = []
        self.subproblem_ip_cuts = []

        # Algorithm iteration and solution tracking
        self.subproblem_ip_counts = []
        self.subproblem_counts_lp = []

        # Solution and constraint tracking
        self.initial_constraint_counter = []
        self.x_feas_counter_lp = []
        self.x_feas_counter_ip = []

        self.lp_iterations = []

        self.lp_first_stage_sols = []
        self.lp_optimal_first_stage_sols = []
        self.ip_first_stage_sols = []
        self.ip_first_stage_optimal_sols = []
        self.previous_saa_solution = None

        self.lp_relaxation_time = []
        self.ip_time = []

        self.total_times = []

        self.lp_final_cons = []
        self.lp_cons_to_ip = []
        self.ip_gap = []
        self.ip_nodes = []

        self.root_node_gap = []

        # UC-specific tracking
        self.int_hash_set = set()
        self.lp_hash_set = set()  # Track unique LP solutions to avoid duplicates

    def benders(self, saa_iteration):
        """
        Main Benders decomposition algorithm implementation for UC.

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
        self.cuts_to_add_to_ip = (
            set()
        )  # Set of tuples, each tuple has one dual ID per scenario
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
        self.first_stage_values = {}  # First-stage commitment variables
        self.second_stage_values = {}  # Second-stage cost variables

        # Rebuild master problem for IP phase (with binary variables and active LP cuts)
        self.ip_initialize()

        # Initialize IP phase performance counters
        self.subproblem_ip_solve_time = 0
        self.subproblem_ip_cuts_generated = 0
        self.subproblem_ip_solve_count = 0
        self.subproblem_ip_work_total = 0

        def benders_callback(model, where):
            """
            Gurobi callback function for lazy constraint generation during IP solve.

            This callback is triggered at MIP nodes and integer solutions to:
            1. Track root node performance (gap)
            2. Generate Benders cuts for integer solutions by solving subproblems
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
                x_order = tuple(
                    int(self.first_stage_values[g, t])
                    for g in self.thermal_gens
                    for t in self.periods
                )
                int_hash = hash(x_order)

                if int_hash not in self.int_hash_set:
                    self.int_hash_set.add(int_hash)
                    self.ip_first_stage_sols.append(x_order)

                # Generate cuts by solving subproblems
                start = time.perf_counter()
                num_cuts_added, z_value, subproblem_work = self.add_single_cut(
                    model, lazy=True
                )
                # Accumulate subproblem work units for IP phase
                self.subproblem_ip_work_total += subproblem_work
                # Update the master problem's z-variable solution with actual value
                # For single-cut, z is a scalar
                model.cbSetSolution(self.x, self.first_stage_values)
                model.cbSetSolution(self.z, z_value)
                self.subproblem_ip_solve_time += time.perf_counter() - start
                self.subproblem_ip_cuts_generated += num_cuts_added
                self.subproblem_ip_solve_count += 1

        # ==================== IP MASTER PROBLEM SOLVE ====================
        # Configure Gurobi for lazy constraint generation
        self.master.setParam("OutputFlag", True)
        self.master.setParam("MIPGap", self.tol)
        self.master.Params.lazyConstraints = 1
        self.master._rootgap = 1000

        # Solve IP master problem with Benders callback
        self.master.optimize(benders_callback)

        # Record root node performance
        self.root_node_gap.append(self.master._rootgap)

        # Record master IP work units
        master_ip_work_total = self.master.getAttr("Work")

        self.extract_solution_values()

        self.ip_time.append(round(self.master.Runtime, 3))
        logger.info(f"IP bound: {self.master.ObjBound:.3f}")

        self.initial_constraint_counter.append(self.master.NumConstrs)

        self.ip_gap.append(round(self.master.MIPGap * 100, 4))
        self.ip_nodes.append(self.master.NodeCount)

        x_order = tuple(
            self.first_stage_values[g, t]
            for g in self.thermal_gens
            for t in self.periods
        )

        if self.previous_saa_solution:
            logger.info(
                f"\n--- Solution Changes per Time Period from Previous SAA Iteration ---"
            )
            total_global_changes = 0
            total_global_decisions = 0

            for t in self.periods:
                period_changes = 0
                period_decisions = 0

                for g in self.thermal_gens:
                    current_val = self.first_stage_values[(g, t)]
                    previous_val = self.previous_saa_solution[self.gt_index_map[(g, t)]]

                    if (
                        abs(current_val - previous_val) > 0.5
                    ):  # For binary, Manhattan distance is 1 if different
                        period_changes += 1

                    period_decisions += 1

                total_global_changes += period_changes
                total_global_decisions += period_decisions

            if total_global_decisions > 0:
                total_global_change_percentage = (
                    total_global_changes / total_global_decisions
                ) * 100
                logger.info(
                    f"Overall changes: {total_global_changes} out of {total_global_decisions} total decisions ({total_global_change_percentage:.2f}%)"
                )
            logger.info(f"----------------------------------------------------\n")

        self.previous_saa_solution = x_order

        if x_order not in self.ip_first_stage_optimal_sols:
            self.ip_first_stage_optimal_sols.append(x_order)

        if x_order not in self.ip_first_stage_sols:
            self.ip_first_stage_sols.append(x_order)

        self.x_feas_counter_ip.append(len(self.ip_first_stage_sols))

        self.subproblem_ip_times.append(round(self.subproblem_ip_solve_time, 2))
        self.cut_time_ip.append(round(self.subproblem_ip_solve_time, 2))
        self.subproblem_ip_cuts.append(self.subproblem_ip_cuts_generated)
        self.subproblem_ip_counts.append(self.subproblem_ip_solve_count)

        # Store work unit totals for IP phase
        self.master_ip_work.append(master_ip_work_total)
        self.subproblem_ip_work.append(self.subproblem_ip_work_total)
        self.total_work_ip.append(master_ip_work_total + self.subproblem_ip_work_total)

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

        # Initialize work unit tracking
        master_lp_work_total = 0
        subproblem_lp_work_total = 0

        # ==================== INITIAL MASTER SOLVE ====================
        start = time.perf_counter()
        self.master.optimize()
        iter_master_time = time.perf_counter() - start
        master_lp_solve_time += iter_master_time
        master_lp_work_total += self.master.getAttr("Work")
        lp_iterations += 1

        # Verify master problem solved successfully
        status = self.master.status
        if status != 2:
            raise Exception(f"Master problem status - {status}")

        # Build subproblem structure and extract initial solution
        self.subproblem = self.build_SP()
        self.extract_solution_values(problem="LP")

        # Track subproblem rebuild counter
        iterations_since_sp_rebuild = 0

        # Initialize upper bound tracking
        self.scenario_upper_bounds = {scenario: math.inf for scenario in self.scenario}
        upper_bound = np.inf

        # Initialize bounds for convergence check
        lower_bound = self.master.ObjBound
        optimality_gap = upper_bound - lower_bound
        num_cuts_added = 1  # Force initial iteration

        # ==================== MAIN BENDERS LOOP ====================
        while True:
            num_cuts_added = 0
            iter_sp_time = 0

            # Solve subproblems to generate single aggregated cut
            start = time.perf_counter()

            num_cuts_generated, aggregated_z_value, subproblem_work = (
                self.add_single_cut(self.master)
            )
            subproblem_lp_work_total += subproblem_work
            calculated_upper_bound = self.calculate_upper_bound_from_subproblems()
            upper_bound = min(calculated_upper_bound, upper_bound)

            num_cuts_added = num_cuts_generated
            iter_sp_time = time.perf_counter() - start
            subproblem_lp_solve_time += iter_sp_time
            subproblem_lp_cuts_generated += num_cuts_generated
            subproblem_lp_solve_count += 1

            # Handle numerical issues with zero lower bound
            if lower_bound == 0:
                lower_bound = 0.1

            # Check convergence criteria: no cuts added or optimality gap sufficiently small
            if num_cuts_added == 0 or (optimality_gap < self.gaplimit):
                calculated_upper_bound = self.calculate_upper_bound_from_subproblems()
                upper_bound = min(calculated_upper_bound, upper_bound)
                logger.info(
                    f"Iteration {lp_iterations}: LP UB: {upper_bound:.3f}, LB: {lower_bound:.3f}"
                )
                print(
                    f"Iteration {lp_iterations}: LP UB: {upper_bound:.3f}, LB: {lower_bound:.3f}"
                )
                break  # LP relaxation converged

            # Calculate percentage optimality gap and continue iterating
            optimality_gap = (upper_bound - lower_bound) / lower_bound

            print(
                f"Iteration {lp_iterations}: LP UB: {upper_bound:.3f}, LB: {lower_bound:.3f}"
            )

            # Rebuild subproblem every 10 iterations
            iterations_since_sp_rebuild += 1
            if iterations_since_sp_rebuild >= 10:
                print(f"  → Rebuilding subproblem at iteration {lp_iterations}")
                self.subproblem = self.build_SP()
                iterations_since_sp_rebuild = 0

            # Resolve master problem with new cuts
            start = time.perf_counter()
            self.master.optimize()
            iter_master_time = time.perf_counter() - start
            master_lp_solve_time += iter_master_time
            master_lp_work_total += self.master.getAttr("Work")

            # Check master problem status
            status = self.master.status
            if status != 2:
                raise Exception(f"Master problem status - {status}")

            # Update bounds and solution tracking
            lower_bound = self.master.ObjBound
            self.extract_solution_values(problem="LP")
            x_order = np.array(
                [
                    self.first_stage_values[g, t]
                    for g in self.thermal_gens
                    for t in self.periods
                ]
            )
            # Check for duplicates before adding LP solution
            lp_hash = hash(tuple(x_order))
            if lp_hash not in self.lp_hash_set:
                self.lp_hash_set.add(lp_hash)
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
        x_order = np.array(
            [
                self.first_stage_values[g, t]
                for g in self.thermal_gens
                for t in self.periods
            ]
        )
        # Always add to optimal solutions (this represents the final optimal LP solution)
        self.lp_optimal_first_stage_sols.append(x_order)

        self.subproblem_lp_times.append(round(subproblem_lp_solve_time, 2))
        self.cut_time_lp.append(round(subproblem_lp_solve_time, 2))
        self.subproblem_lp_cuts.append(subproblem_lp_cuts_generated)
        self.master_lp_times.append(round(master_lp_solve_time, 2))
        self.subproblem_counts_lp.append(subproblem_lp_solve_count)

        # Store work unit totals for LP phase
        self.master_lp_work.append(master_lp_work_total)
        self.subproblem_lp_work.append(subproblem_lp_work_total)
        self.total_work_lp.append(master_lp_work_total + subproblem_lp_work_total)

    def add_cuts_to_master(self, initialize_set, check_lp_cuts=True):
        """
        Add aggregated Benders optimality cuts to master problem - SINGLE CUT VERSION.

        For each tuple of dual solutions (one per scenario), constructs and adds one
        aggregated cut with probability weighting across all scenarios.

        Args:
            initialize_set (set): Set of tuples, each tuple containing dual solution indices for all scenarios
            check_lp_cuts (bool): If True, skip cuts already added to IP from LP phase

        Returns:
            int: Number of cuts added to master problem
        """
        cut_count = 0

        for duals_tuple in initialize_set:
            # Skip cuts that were already transferred from LP phase
            if check_lp_cuts:
                # Convert duals_tuple to tuple for comparison with cuts_to_add_to_ip
                if tuple(duals_tuple) in self.cuts_to_add_to_ip:
                    continue

            # Build aggregated cut expression
            cut_expr = gp.LinExpr()

            for scenario_idx, scenario in enumerate(self.scenario):
                dual_id = duals_tuple[scenario_idx]

                # Get dual solution vectors
                generation_duals = self.generation_duals_array[dual_id, :]
                demand_duals = self.demand_duals_array[dual_id, :]

                # Add this scenario's contribution to the aggregated cut (with probability weighting)
                commitment_idx = 0
                for g in self.thermal_gens:
                    for t in self.periods:
                        # Generation duals stored as [gen_min_duals + gen_max_duals]
                        gen_min_dual = generation_duals[commitment_idx]
                        gen_max_dual = generation_duals[
                            commitment_idx + self.nG * self.T
                        ]

                        dual_coeff = (
                            self.min_power[g] * gen_min_dual
                            + self.max_power[g] * gen_max_dual
                        )
                        cut_expr += (
                            self.probability[scenario] * dual_coeff * self.x[g, t]
                        )
                        commitment_idx += 1

                # Add demand contribution for this scenario
                cut_expr += self.probability[scenario] * sum(
                    self.demand_scenarios[scenario][t] * demand_duals[t - 1]
                    for t in self.periods
                )

            # Add single aggregated cut: z >= cut_expr
            self.master.addConstr(cut_expr <= self.z)
            cut_count += 1

        return cut_count

    def dual_update(self):
        """
        Update dual solution data structures using AtomicDualStorage.

        This method prepares dual arrays from the current SAA iteration's dual storage.
        """
        # Ensure dual storage is initialized
        self.ensure_dual_storage_initialized()

        if len(self.dual_storage) == 0:
            return

        # Prepare arrays - this is very fast with AtomicDualStorage
        self.prepare_dual_arrays()

    def identify_active_cuts(self):
        """
        Identify active (binding) cuts for single-cut Benders - UC VERSION.

        Evaluates the LHS of each stored aggregated cut at the final LP solution (x*)
        and marks it active if it approximately equals the final value of z (within tolerance).

        Returns:
            list: List of dual-id lists (one dual id per scenario) for active cuts
        """
        if not hasattr(self, "cut_history") or not self.cut_history:
            return []

        # Calculate commitment-dependent dual contributions
        commitment_solution = np.array(
            [
                self.first_stage_values[g, t]
                for g in self.thermal_gens
                for t in self.periods
            ]
        )
        commitment_weighted_solution = self.get_commitment_weighted_solution(
            commitment_solution
        )

        # Evaluate all dual solutions at current first-stage solution
        commitment_dual_product = np.matmul(
            self.generation_duals_array, commitment_weighted_solution
        )
        commitment_dual_product = commitment_dual_product.reshape(-1, 1)

        # Compute subproblem objective values for all (dual, scenario) combinations
        subproblem_evaluations = commitment_dual_product + self.dual_obj_random
        assert subproblem_evaluations.shape == (
            len(self.generation_duals_array),
            self.nS,
        )

        # Use the final scalar value of z
        active_tol = 1e-5
        z_val = float(self.second_stage_values)  # scalar z value

        active_cuts = []

        # Evaluate each previously added aggregated cut
        for cut_dual_ids in self.cut_history:
            # Calculate LHS of this aggregated cut using probability weighting
            cut_lhs = sum(
                self.probability[scenario_idx]
                * subproblem_evaluations[dual_id, scenario_idx]
                for scenario_idx, dual_id in enumerate(cut_dual_ids)
            )

            # Check if cut is active (LHS approximately equals RHS)
            if abs(cut_lhs - z_val) <= active_tol:
                active_cuts.append(cut_dual_ids)

        print(f"Total active cuts to transfer to IP: {len(active_cuts)}")

        return active_cuts

    def cache_dsp(self, dual_list, lazy=True):
        """
        Apply dual solution pooling to find violated cuts quickly - SINGLE CUT VERSION.

        This method evaluates a list of dual solutions to find the best dual for each scenario,
        then aggregates them into a single cut with probability weighting.

        Args:
            dual_list (list): List of dual solution indices to evaluate
            lazy (bool): If True, add cuts as lazy constraints

        Returns:
            int: Number of cuts added (0 or 1 for single-cut version)
        """
        # Calculate commitment-dependent dual contributions
        commitment_solution = np.array(
            [
                self.first_stage_values[g, t]
                for g in self.thermal_gens
                for t in self.periods
            ]
        )
        commitment_weighted_solution = self.get_commitment_weighted_solution(
            commitment_solution
        )

        # Calculate dual contributions for selected dual solutions
        commitment_dual_product = np.matmul(
            self.generation_duals_array[dual_list], commitment_weighted_solution
        )
        commitment_dual_product = np.squeeze(commitment_dual_product)

        # Use numba-optimized function to find best dual for each scenario
        subproblem_optimal_values, optimal_dual_indices = (
            uc_benders_utils.find_largest_index_numba_uc(
                commitment_dual_product,
                self.dual_obj_random[np.ix_(dual_list, self.scenario)],
            )
        )

        # Convert relative indices back to original dual solution IDs
        optimal_duals = [dual_list[idx] for idx in optimal_dual_indices]

        # Calculate aggregated z value with probability weighting
        aggregated_z_value = sum(
            self.probability[scenario] * subproblem_optimal_values[scenario]
            for scenario in self.scenario
        )

        # Check if aggregated cut should be added
        if aggregated_z_value - self.second_stage_values > max(
            self.tol, 0.00001 * abs(self.second_stage_values)
        ):
            # Build aggregated cut expression
            aggregated_cut_expr = gp.LinExpr()

            for scenario in self.scenario:
                dual_idx = optimal_duals[scenario]

                # Get dual values from storage
                generation_duals = self.generation_duals_array[dual_idx, :]
                demand_duals = self.demand_duals_array[dual_idx, :]

                # Add this scenario's contribution to the aggregated cut (with probability weighting)
                commitment_idx = 0
                for g in self.thermal_gens:
                    for t in self.periods:
                        gen_min_dual = generation_duals[commitment_idx]
                        gen_max_dual = generation_duals[
                            commitment_idx + self.nG * self.T
                        ]

                        dual_coeff = (
                            self.min_power[g] * gen_min_dual
                            + self.max_power[g] * gen_max_dual
                        )
                        aggregated_cut_expr += (
                            self.probability[scenario] * dual_coeff * self.x[g, t]
                        )
                        commitment_idx += 1

                # Add demand contribution for this scenario
                aggregated_cut_expr += self.probability[scenario] * sum(
                    self.demand_scenarios[scenario][t] * demand_duals[t - 1]
                    for t in self.periods
                )

                # Update dual solution counter
                if dual_idx in self.dual_soln_optimal_counter:
                    self.dual_soln_optimal_counter[dual_idx] += 1

            # Add single aggregated cut: z >= aggregated_cut_expr
            if lazy:
                self.master.cbLazy(self.z >= aggregated_cut_expr)
            else:
                self.master.addConstr(
                    self.z >= aggregated_cut_expr,
                    name=f"single_dsp_cut_{len(self.cut_history)}",
                )

            # Track which dual solutions contributed to this aggregated cut
            self.cut_history.append(optimal_duals.copy())
            return 1

        return 0

    def ip_initialize(self):
        """
        Initialize IP master problem with active cuts from LP phase (vanilla Benders).

        This method:
        1. Completely rebuilds the master problem with binary variables and all structural constraints
        2. Rebuilds the subproblem for IP phase
        3. Adds back the active Benders cuts from LP phase
        """
        # Completely rebuild master problem with binary variables and all structural UC constraints
        self.master = self.build_master(relaxation=False)
        self.master.setParam("TimeLimit", 3600)
        self.master.setParam("WorkLimit", 3600)

        # Rebuild subproblem at the start of IP phase
        print(f"  → Rebuilding subproblem at start of IP phase")
        self.subproblem = self.build_SP()

        # Add active Benders cuts from LP phase to IP master
        lp_cons = self.add_cuts_to_master(self.cuts_to_add_to_ip, check_lp_cuts=False)
        self.lp_cons_to_ip.append(lp_cons)  # Track cuts transferred from LP to IP
        self.master.update()  # Force Gurobi to update internal state

    def solve_saa_iteration(self, saa_iteration):
        """Solve a single SAA iteration."""
        start = time.perf_counter()

        self.master = self.build_master(relaxation=True)
        self.master.setParam("TimeLimit", 3600)
        self.master.setParam("WorkLimit", 3600)
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

    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()

    # File handler for logging to file
    file_handler = logging.FileHandler(fname, mode="a")
    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler for logging to stdout
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def log_and_print(logger, message):
    """Helper function to ensure output goes to both log file and console"""
    logger.info(message)
    # Logger already handles both file and console output, no need for separate print


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Unit Commitment Benders Decomposition - Single Cut Version"
    )
    parser.add_argument(
        "algorithm_details",
        metavar="N",
        type=int,
        nargs=3,
        help="Three integers: solve_ip dual_lookup initialization_method (only solve_ip is used for single-cut)",
    )
    parser.add_argument(
        "data", metavar="N", type=str, nargs=4
    )  # instance_file, nscen, std_dev, nsaa
    parser.add_argument(
        "--extensive",
        action="store_true",
        help="Solve using extensive form instead of Benders decomposition",
    )
    parser.add_argument(
        "--periods",
        type=int,
        help="Number of time periods to use (creates reduced instance if less than original)",
    )
    parser.add_argument(
        "--num-generators",
        type=int,
        help="Number of thermal generators to keep (creates reduced instance with scaled demand)",
    )
    parser.add_argument(
        "--generate",
        type=int,
        nargs="+",
        metavar="N",
        help="Generate instance on-the-fly: --generate N_GENS [N_DAYS] [DIFFICULTY]. "
        "First argument (instance_file) will be used as output filename. "
        "Example: --generate 10 1 2 generates 10 generators, 1 day, difficulty 2. "
        "Defaults: n_days=1, difficulty=1",
    )
    parser.add_argument(
        "--timelimit",
        type=float,
        default=3600,
        help="Time limit in seconds for Gurobi solver (default: 3600)",
    )
    args = parser.parse_args()

    solve_ip, dual_lookup, initialization_method = args.algorithm_details
    instance_file, nscen, std_dev, nsaa = args.data

    # Generate instance on-the-fly if --generate flag is used
    if args.generate:
        n_gens = args.generate[0]
        n_days = args.generate[1] if len(args.generate) > 1 else 1
        difficulty = args.generate[2] if len(args.generate) > 2 else 1

        print(f"\n{'='*70}")
        print("GENERATING INSTANCE ON-THE-FLY")
        print(f"{'='*70}")
        print(f"  Generators: {n_gens}")
        print(f"  Days: {n_days}")
        print(f"  Difficulty: {difficulty}")
        print(f"  Output file: {instance_file}")
        print(f"  Seed: 42 (fixed)")
        print(f"{'='*70}\n")

        instance_file = uc_benders_utils.UCinst.generate_instance(
            n_generators=n_gens,
            n_days=n_days,
            periods_per_day=24,
            difficulty=difficulty,
            output_file=instance_file,
        )
        print(f"\n✓ Instance generated successfully: {instance_file}\n")

    # Create reduced generator instance data if specified
    instance_data = None
    if args.num_generators:
        print(f"Creating reduced instance with {args.num_generators} generators...")
        instance_data = uc_benders_utils.UCinst.create_reduced_generator_instance(
            input_filename_or_data=instance_file,
            num_generators=args.num_generators,
        )
        print(f"Using reduced instance with {args.num_generators} generators")

    # Clean the instance filename for safe file naming
    instance_name = (
        os.path.basename(instance_file)
        .replace(".json", "")
        .replace("/", "_")
        .replace("\\", "_")
    )

    # Include periods in data string if specified
    if args.periods:
        data_string = f"{instance_name}_{nscen}_{std_dev}_{nsaa}_{args.periods}"
    else:
        data_string = f"{instance_name}_{nscen}_{std_dev}_{nsaa}"
    algorithm_string = "_".join(str(i) for i in args.algorithm_details)

    if args.extensive:
        # Extensive form mode - LP relaxation quality analysis
        algorithm_name = "Extensive_LPAnalysis"
        results_filename = (
            f"detailed-results/uc/LP/UC_Extensive_LPAnalysis_{data_string}.op"
        )
        print("Using Extensive Form - LP Relaxation Quality Analysis")

        logger = setup_logger(results_filename)
        logger.info(f"data: {args.data}")
        logger.info(f"Algorithm: Extensive Form - LP Relaxation Quality Analysis")

        np.random.seed(3)
        # Create UC instance directly (no need for Benders class)
        uc_instance = uc_benders_utils.UCinst(int(nscen))
        # Use reduced instance data if available, otherwise use file
        instance_to_load = instance_data if instance_data is not None else instance_file
        uc_instance.load_instance(instance_to_load, max_periods=args.periods)

        num_saa_iterations = int(nsaa)
        num_scenarios = int(nscen)
        demand_std_dev = float(std_dev)

        extensive_results = []

        for saa_iteration in range(1, num_saa_iterations + 1):
            print(f"\n{'#'*70}")
            print(f"### SAA Iteration {saa_iteration}")
            print(f"{'#'*70}")
            uc_instance.generate_demand_scenarios(num_scenarios, demand_std_dev)
            # No renewable scenario generation needed - renewables are decision variables

            # Analyze LP relaxation quality (solves both LP and IP)
            start_time = time.perf_counter()
            analysis_results = uc_instance.analyze_lp_relaxation_quality(
                time_limit=3600,
            )
            total_time = time.perf_counter() - start_time

            extensive_results.append(
                {
                    "saa_iteration": saa_iteration,
                    "lp_objective": analysis_results["lp_objective"],
                    "ip_objective": analysis_results["ip_objective"],
                    "lp_solve_time": analysis_results["lp_solve_time"],
                    "ip_solve_time": analysis_results["ip_solve_time"],
                    "total_time": total_time,
                    "lp_x_zeros": analysis_results["lp_x_zeros"],
                    "lp_x_ones": analysis_results["lp_x_ones"],
                    "lp_x_fractional": analysis_results["lp_x_fractional"],
                    "lp_x_total": analysis_results["lp_x_total"],
                    "lp_integrality_ratio": analysis_results["lp_integrality_ratio"],
                    "lp_ip_gap_percent": analysis_results.get(
                        "lp_ip_gap_percent", None
                    ),
                    "lp_ip_x_differences": analysis_results.get(
                        "lp_ip_x_differences", None
                    ),
                    "ip_x_zeros": analysis_results["ip_x_zeros"],
                    "ip_x_ones": analysis_results["ip_x_ones"],
                    "ip_gap": analysis_results["ip_gap"],
                    "ip_nodes": analysis_results["ip_nodes"],
                }
            )

            # Log results
            logger.info(f"\n=== SAA {saa_iteration} Results ===")
            logger.info(f"LP Objective: {analysis_results['lp_objective']:.4f}")
            logger.info(f"IP Objective: {analysis_results['ip_objective']:.4f}")
            logger.info(
                f"LP-IP Gap: {analysis_results.get('lp_ip_gap_percent', 0):.4f}%"
            )
            logger.info(
                f"LP x=0: {analysis_results['lp_x_zeros']}, x=1: {analysis_results['lp_x_ones']}, fractional: {analysis_results['lp_x_fractional']}"
            )
            logger.info(
                f"LP Integrality Ratio: {100*analysis_results['lp_integrality_ratio']:.1f}%"
            )
            logger.info(
                f"LP Solve Time: {analysis_results['lp_solve_time']:.2f}s, IP Solve Time: {analysis_results['ip_solve_time']:.2f}s"
            )

        # Create results dataframe for extensive form LP analysis
        data_dict = {
            "Instance": data_string,
            "Scenarios": num_scenarios,
            "Method": "Extensive_LPAnalysis",
            "Avg LP Objective": np.mean(
                [
                    r["lp_objective"]
                    for r in extensive_results
                    if r["lp_objective"] is not None
                ]
            ),
            "Avg IP Objective": np.mean(
                [
                    r["ip_objective"]
                    for r in extensive_results
                    if r["ip_objective"] is not None
                ]
            ),
            "Avg LP-IP Gap %": np.mean(
                [
                    r["lp_ip_gap_percent"]
                    for r in extensive_results
                    if r["lp_ip_gap_percent"] is not None
                ]
            ),
            "Avg LP Integrality %": np.mean(
                [100 * r["lp_integrality_ratio"] for r in extensive_results]
            ),
            "Avg LP x=0": np.mean([r["lp_x_zeros"] for r in extensive_results]),
            "Avg LP x=1": np.mean([r["lp_x_ones"] for r in extensive_results]),
            "Avg LP fractional": np.mean(
                [r["lp_x_fractional"] for r in extensive_results]
            ),
            "Total x vars": extensive_results[0]["lp_x_total"],
            "Avg LP-IP x differences": np.mean(
                [
                    r["lp_ip_x_differences"]
                    for r in extensive_results
                    if r["lp_ip_x_differences"] is not None
                ]
            ),
            "Avg LP Solve Time": np.mean(
                [r["lp_solve_time"] for r in extensive_results]
            ),
            "Avg IP Solve Time": np.mean(
                [r["ip_solve_time"] for r in extensive_results]
            ),
            "Avg Total Time": np.mean([r["total_time"] for r in extensive_results]),
        }

        # Print summary table
        print("\n" + "=" * 70)
        print("SUMMARY ACROSS ALL SAA ITERATIONS")
        print("=" * 70)
        summary_data = [
            ["LP Objective", [r["lp_objective"] for r in extensive_results]],
            ["IP Objective", [r["ip_objective"] for r in extensive_results]],
            ["LP-IP Gap %", [r["lp_ip_gap_percent"] for r in extensive_results]],
            ["LP x=0", [r["lp_x_zeros"] for r in extensive_results]],
            ["LP x=1", [r["lp_x_ones"] for r in extensive_results]],
            ["LP fractional", [r["lp_x_fractional"] for r in extensive_results]],
            [
                "LP Integrality %",
                [100 * r["lp_integrality_ratio"] for r in extensive_results],
            ],
            ["LP-IP x diffs", [r["lp_ip_x_differences"] for r in extensive_results]],
        ]
        saa_headers = (
            ["Metric"]
            + [f"SAA {i}" for i in range(1, num_saa_iterations + 1)]
            + ["Avg"]
        )
        table_data = []
        for row in summary_data:
            metric, values = row
            avg_val = (
                np.mean([v for v in values if v is not None])
                if any(v is not None for v in values)
                else None
            )
            formatted_values = [f"{v:.2f}" if v is not None else "N/A" for v in values]
            formatted_avg = f"{avg_val:.2f}" if avg_val is not None else "N/A"
            table_data.append([metric] + formatted_values + [formatted_avg])

        logger.info(tabulate(table_data, headers=saa_headers))
        print(tabulate(table_data, headers=saa_headers))

        solved_saa_iterations = list(range(1, num_saa_iterations + 1))

    else:
        # Single-cut Benders mode
        algorithm_name = f"single_{algorithm_string}"
        if solve_ip:
            results_filename = (
                f"detailed-results/uc/IP/single_UC_IP_{data_string}_{algorithm_name}.op"
            )
        else:
            results_filename = (
                f"detailed-results/uc/LP/single_UC_LP_{data_string}_{algorithm_name}.op"
            )

        print(f"Algorithm: Single-Cut Benders ({algorithm_string})")

        logger = setup_logger(results_filename)
        logger.info(f"data: {args.data}")
        logger.info(f"algorithm_details: {args.algorithm_details}")
        logger.info(f"Algorithm: Single-Cut Benders ({algorithm_string})")
        logger.info(f"solve_ip: {solve_ip}")

        np.random.seed(3)
        # Use reduced instance data if available, otherwise use file
        instance_to_load = instance_data if instance_data is not None else instance_file
        benders_solver = Benders(
            solve_ip,
            instance_to_load,
            int(nscen),
            max_periods=args.periods,
        )

        num_saa_iterations = int(nsaa)
        num_scenarios = int(nscen)
        demand_std_dev = float(std_dev)

        solved_saa_iterations = []

        # Solve all SAA iterations (vanilla Benders has no information reuse)
        for saa_iteration in range(1, num_saa_iterations + 1):
            benders_solver.generate_demand_scenarios(num_scenarios, demand_std_dev)
            # No renewable scenario generation needed - renewables are decision variables

            benders_solver.solve_saa_iteration(saa_iteration)
            solved_saa_iterations.append(saa_iteration)

    if not args.extensive:
        # Original Benders results processing
        data_dict = {
            "Instance": data_string,
            "Scenarios": num_scenarios,
            "Method": algorithm_name,
        }

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
                "master lp work",
                "subproblem lp work",
                "total work lp",
                "master ip work",
                "subproblem ip work",
                "total work ip",
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
                "master lp work",
                "subproblem lp work",
                "total work lp",
            ]

        for col in columns:
            col_data = getattr(benders_solver, col.replace(" ", "_").lower())
            if len(col_data) > 0:
                data_dict[f"{col} SAA 0"] = round(col_data[0], 3)
                if len(col_data) > 1:
                    # For NoReuse method, average should include the first run
                    if algorithm_name == "NoReuse":
                        data_dict[f"{col} average"] = round(np.mean(col_data), 3)
                    else:
                        data_dict[f"{col} average"] = round(np.mean(col_data[1:]), 3)

        # Single-cut vanilla Benders doesn't have initialization or dual pool tracking
        # These attributes are not used in vanilla Benders

    # data_dict is already created for extensive form above

    df = pd.DataFrame([data_dict])

    # Save to instance-specific CSV file (includes algorithm parameters)
    instance_csv = f"results_{data_string}_{algorithm_name}.csv"
    df.to_csv(
        instance_csv,
        index=False,
        quoting=3,
        sep=",",
        escapechar=",",
        mode="a",
        header=not os.path.exists(instance_csv),
    )

    # Also append to global results file (separate for single-cut)
    df.to_csv(
        "results_single_uc.csv",
        index=False,
        quoting=3,
        sep=",",
        escapechar=",",
        mode="a",
        header=not os.path.exists("results_single_uc.csv"),
    )

    logger.info("")

    if not args.extensive:
        saa_iteration_list = [f"SAA {i}" for i in solved_saa_iterations]
        logger.info("=" * 80)
        logger.info("DETAILED RESULTS SUMMARY")
        logger.info("=" * 80)
        # logger.info("")

        # Log constraint information (single-cut vanilla Benders)
        constraint_data = [
            ["SP cuts - LP", *benders_solver.subproblem_lp_cuts],
            ["SP count - LP", *benders_solver.subproblem_counts_lp],
            ["Final cons LP", *benders_solver.lp_final_cons],
        ]
        logger.info(
            tabulate(constraint_data, headers=["LP Constraints", *saa_iteration_list])
        )
        logger.info("")

        if solve_ip:
            ip_constraint_data = [
                ["Benders cuts from LP to IP", *benders_solver.lp_cons_to_ip],
                ["SP cuts - IP", *benders_solver.subproblem_ip_cuts],
                ["SP count - IP", *benders_solver.subproblem_ip_counts],
            ]
            logger.info(
                tabulate(
                    ip_constraint_data,
                    headers=["IP Constraints", *saa_iteration_list],
                )
            )
            logger.info("")

        # Time information
        time_data = [
            ["LP time", *benders_solver.lp_relaxation_time],
            ["IP time", *benders_solver.ip_time],
            ["Total time", *benders_solver.total_times],
        ]

        logger.info(tabulate(time_data, headers=["Time Info", *saa_iteration_list]))
        logger.info("")

        # LP time breakdown (single-cut vanilla Benders)
        lp_time_data = [
            ["Master time", *benders_solver.master_lp_times],
            ["SP time", *benders_solver.subproblem_lp_times],
            ["Solutions", *benders_solver.x_feas_counter_lp],
        ]
        logger.info(tabulate(lp_time_data, headers=["LP Time", *saa_iteration_list]))
        logger.info("")

        # LP work unit breakdown
        lp_work_data = [
            ["Master work", *benders_solver.master_lp_work],
            ["SP work", *benders_solver.subproblem_lp_work],
            ["Total work", *benders_solver.total_work_lp],
        ]
        logger.info(
            tabulate(lp_work_data, headers=["LP Work Units", *saa_iteration_list])
        )
        logger.info("")

        if solve_ip:
            # IP time breakdown (single-cut vanilla Benders)
            ip_time_data = [
                ["SP time", *benders_solver.subproblem_ip_times],
                ["Root gap", *benders_solver.root_node_gap],
            ]
            logger.info(
                tabulate(ip_time_data, headers=["IP Time", *saa_iteration_list])
            )
            logger.info("")

            # IP work unit breakdown
            ip_work_data = [
                ["Master work", *benders_solver.master_ip_work],
                ["SP work", *benders_solver.subproblem_ip_work],
                ["Total work", *benders_solver.total_work_ip],
            ]
            logger.info(
                tabulate(ip_work_data, headers=["IP Work Units", *saa_iteration_list])
            )
            logger.info("")

            ip_info_data = [
                ["First stage solutions IP", *benders_solver.x_feas_counter_ip],
                ["IP gap %", *benders_solver.ip_gap],
                ["IP nodes", *benders_solver.ip_nodes],
            ]
            logger.info(
                tabulate(ip_info_data, headers=["IP Info", *saa_iteration_list])
            )
            logger.info("")

        # Summary information (single-cut vanilla Benders)
        summary_data = [
            ["LP Iterations", benders_solver.lp_iterations],
        ]

        logger.info(tabulate(summary_data, headers=[]))
        logger.info("")

        # Work unit summary
        work_summary_data = [
            ["Total LP work units", [sum(benders_solver.total_work_lp)]],
            [
                "Avg LP work per SAA",
                [
                    (
                        np.mean(benders_solver.total_work_lp)
                        if benders_solver.total_work_lp
                        else 0
                    )
                ],
            ],
        ]
        if solve_ip and benders_solver.total_work_ip:
            work_summary_data.extend(
                [
                    ["Total IP work units", [sum(benders_solver.total_work_ip)]],
                    ["Avg IP work per SAA", [np.mean(benders_solver.total_work_ip)]],
                    [
                        "Total work units (LP+IP)",
                        [
                            sum(benders_solver.total_work_lp)
                            + sum(benders_solver.total_work_ip)
                        ],
                    ],
                ]
            )

        logger.info(tabulate(work_summary_data, headers=["Work Unit Summary", ""]))
        logger.info("")

    # Clean up generated JSON file if it was created with --generate flag
    if args.generate and os.path.exists(instance_file):
        try:
            os.remove(instance_file)
            print(f"\n✓ Cleaned up generated instance file: {instance_file}")
        except Exception as e:
            print(f"\n⚠ Warning: Could not remove generated file {instance_file}: {e}")
