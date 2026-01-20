import argparse
import numpy as np
import time
import math
import uc_benders_utils
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
Benders Decomposition Implementation for Unit Commitment Problem (UC)

This implementation features:
1. Sample Average Approximation (SAA) for stochastic programming
2. Dual Solution Pooling (DSP) for accelerated Benders decomposition  
3. Multiple initialization techniques (Static/Adaptive/Boosted)
4. Both LP relaxation and IP solving capabilities
5. Comprehensive performance tracking and logging

The algorithm decomposes the UC problem into:
- Master Problem: First-stage commitment decisions (x variables) with temporal constraints
- Subproblems: Second-stage economic dispatch decisions for each scenario

Key Features:
- DSP: Reuses previously computed dual solutions to avoid redundant subproblem solves
- Curated DSP: Intelligently selects most promising dual solutions
- Static/Adaptive Init: Warm-starts master problem with heuristically selected cuts
- Temporal Constraints: Handles minimum up/down time constraints in master problem
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
        algorithm_details,
        instance_file_or_data,
        nscen,
        max_periods=None,
        log_filename=None,
        time_limit=3600,
    ):
        """
        Initialize Benders decomposition solver for unit commitment.

        Args:
            algorithm_details: Tuple (solve_ip, dual_lookup, init) where:
                - solve_ip (int): 0=LP only, 1=solve IP after LP
                - dual_lookup (int): 0=no DSP, 1=basic DSP, 2=curated DSP, 3=random DSP
                - init (int): 0=vanilla, 1=static, 2=adaptive, 3=boosted static
            instance_file_or_data: Path to UC instance JSON file or instance data dict
            nscen: Number of scenarios to generate
            max_periods: If specified, only use the first max_periods time periods
            log_filename: Filename for Gurobi logging
            time_limit: Time limit in seconds for Gurobi solver (default: 3600)
        """

        super().__init__(nscen)
        self.load_instance(instance_file_or_data, max_periods=max_periods)

        # Store log filename for Gurobi logging
        self.log_filename = log_filename

        # Store Gurobi solver limits
        self.time_limit = time_limit

        solve_ip, dual_lookup, initialization_method = algorithm_details

        # Build mapping from (generator, time_period) to index in flattened solution vector
        self.gt_index_map = {}
        idx = 0
        for g in self.thermal_gens:
            for t in self.periods:
                self.gt_index_map[(g, t)] = idx
                idx += 1

        # ==================== ALGORITHM CONFIGURATION ====================
        # Dual Solution Pooling (DSP) Configuration
        self.dual_lookup_lp = True if dual_lookup >= 1 else False
        self.dual_lookup_ip = True if dual_lookup >= 1 else False
        self.split_dual = True if dual_lookup == 2 or dual_lookup == 3 else False

        # Core Algorithm Settings
        self.solve_ip = solve_ip

        # Initialization Method Mapping
        init_methods = {
            0: "vanilla",
            1: "tech_1",
            2: "tech_2",
            3: "tech_1_boosted",
        }
        self.init_method = init_methods[initialization_method]

        # Whether to transfer active cuts from LP relaxation to IP master problem
        self.lp_active_cuts = True if solve_ip == 1 else False

        # Initialize lp_active to store active cuts from previous SAA iteration
        self.lp_active = {}

        # ==================== PERFORMANCE TRACKING ====================
        # Timing metrics for different algorithm components
        self.master_lp_times = []
        self.subproblem_lp_times = []
        self.subproblem_ip_times = []
        self.dual_lookup_lp_times = []
        self.dual_lookup_ip_times = []
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
        self.dual_lookup_lp_cuts = []
        self.dual_lookup_ip_cuts = []

        # Algorithm iteration and solution tracking
        self.dual_lookup_lp_counts = []
        self.dual_lookup_ip_counts = []
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
        self.tech_1_lp_cons = []
        self.tech_1_ip_cons = []
        self.tech_2_lp_cons = []
        self.tech_2_ip_cons = []

        self.lp_cons_to_ip = []
        self.ip_gap = []
        self.ip_nodes = []

        self.dual_soln_optimal_counter = {}  # dual id : how many times cuts off soln

        self.dual_pool_size = []
        self.dual_pool_size_final = []
        self.primary_pool_size = [0]
        self.root_node_gap = []
        self.lp_initialization_time = [0]
        self.ip_initialization_time = [0]

        # Timelimit experiment tracking
        self.timelimit = None
        self.lp_timeout = False
        self.ip_timeout = False
        self.final_upper_bound = None
        self.final_lower_bound = None
        self.final_gap = None

        # UC-specific tracking
        self.int_hash_set = set()
        self.lp_hash_set = set()  # Track unique LP solutions to avoid duplicates

        # Additional data structures for DSP integration
        self.generation_duals_array = None  # Numpy array for fast dual evaluation
        self.demand_duals_array = None  # Numpy array for demand constraint duals
        self.renewable_duals_array = None  # Numpy array for renewable constraint duals

    def benders(self, saa_iteration):
        """
        Main Benders decomposition algorithm implementation for UC.

        Performs either LP relaxation only or full LP+IP solve depending on configuration.
        For SAA problems after the first one, leverages dual solution pooling and
        initialization techniques for acceleration.

        Args:
            saa_iteration (int): Current SAA iteration number (1-indexed)

        Returns:
            int: 0 on successful completion
        """
        # Determine if this is the first SAA problem (no prior dual solutions available)
        is_first_saa = 1 if saa_iteration == 1 else 0

        # Build master problem with LP relaxation for initial phase
        self.master = self.build_master(relaxation=True)
        self.master.setParam("OutputFlag", False)

        # Initialize cut tracking structures for current SAA iteration
        self.cuts_to_add_to_ip = {scenario: set() for scenario in self.scenario}
        self.lp_cuts = {scenario: set() for scenario in self.scenario}

        # For subsequent SAA problems: Configure dual solution pooling
        if not is_first_saa:
            # Curated DSP: Intelligent selection of dual solutions for primary pool
            if self.split_dual:
                self.primary_pool = []

                # Configuration flags for different dual solution sources
                add_previous_dsp_cuts = True  # Include duals that generated cuts before
                add_active = True  # Include duals active at previous LP solution
                add_previous_saa = True  # Include duals from previous SAA iterations

                # Add dual solutions that have been effective in generating cuts
                if add_previous_dsp_cuts:
                    self.primary_pool = [
                        k for k, v in self.dual_soln_optimal_counter.items() if v > 0
                    ]

                # Add dual solutions that were active (binding) at the previous LP optimal solution
                # Only add active cuts if we're solving IPs (not just LPs)
                if add_active and self.solve_ip:
                    for scenario in self.scenario:
                        for dual in self.lp_active[scenario]:
                            if dual not in self.primary_pool:
                                self.primary_pool.append(dual)

                # Add dual solutions generated in the previous SAA iteration
                if add_previous_saa:
                    if len(self.dual_pool_size_final) == 1:
                        # First time adding from previous SAA: include all previous duals
                        more_duals = list(range(self.dual_pool_size_final[-1]))
                    else:
                        # Add only the newly generated duals from last SAA iteration
                        more_duals = list(
                            range(
                                self.dual_pool_size_final[-2],
                                self.dual_pool_size_final[-1],
                            )
                        )
                    self.primary_pool.extend(more_duals)

                # Remove duplicates while preserving order
                self.primary_pool = list(dict.fromkeys(self.primary_pool))

                # Random DSP: Replace curated selection with random selection of same size
                if dual_lookup == 3:
                    num_duals = len(self.primary_pool)
                    self.primary_pool = list(
                        np.random.choice(
                            self.dual_pool_size_final[-1], num_duals, replace=False
                        )
                    )
                    assert len(self.primary_pool) == num_duals
            else:
                # Basic DSP: Use all available dual solutions
                self.primary_pool = list(range(len(self.dual_storage)))

            # Track the size of primary dual solution pool
            self.primary_pool_size.append(len(self.primary_pool))

            # Prepare dual arrays for fast evaluation
            self.prepare_dual_arrays()

            # Perform master problem initialization using selected technique
            init_time = time.perf_counter()
            self.lp_initialize(self.init_method)
            self.lp_initialization_time.append(
                round(time.perf_counter() - init_time, 3)
            )
        else:
            # First SAA iteration: no initialization
            self.primary_pool = []
            self.tech_1_lp_cons.append(0)
            self.tech_2_lp_cons.append(0)

        # ==================== LP RELAXATION PHASE ====================
        problem_start = time.perf_counter()
        self.solve_lp_relaxation(is_first_saa, problem_start_time=problem_start)
        self.lp_relaxation_time.append(round(time.perf_counter() - problem_start, 3))
        self.x_feas_counter_lp.append(len(self.lp_first_stage_sols))

        # If configured for LP-only, terminate here or if LP timed out
        if not self.solve_ip or self.lp_timeout:
            # Add tracking for no IP phase
            self.tech_1_ip_cons.append(0)
            self.tech_2_ip_cons.append(0)
            self.lp_cons_to_ip.append(0)
            return 0

        # ==================== IP PHASE PREPARATION ====================
        # Initialize solution value containers
        self.first_stage_values = {}  # First-stage commitment variables
        self.second_stage_values = {}  # Second-stage cost variables

        # if is_first_saa:
        #     # First SAA: Simply convert LP relaxation to IP by changing variable types
        #     self.master.setAttr("vType", self.x, GRB.BINARY)
        #     self.master.setParam("LogToConsole", 1)
        #     # Add tracking for first SAA IP phase
        #     self.tech_1_ip_cons.append(0)
        #     self.tech_2_ip_cons.append(0)
        # else:
        #     # Subsequent SAAs: Reinitialize master problem with cuts and warm-start
        #     init_time = time.perf_counter()
        #     self.ip_initialize(self.init_method)
        #     self.ip_initialization_time.append(
        #         round(time.perf_counter() - init_time, 3)
        #     )
        init_time = time.perf_counter()
        self.ip_initialize(self.init_method)
        self.ip_initialization_time.append(round(time.perf_counter() - init_time, 3))

        # Initialize IP phase performance counters
        self.subproblem_ip_solve_time = 0
        self.subproblem_ip_cuts_generated = 0
        self.subproblem_ip_solve_count = 0
        self.dual_lookup_ip_time = 0
        self.dual_lookup_ip_cuts_generated = 0
        self.dual_lookup_ip_call_count = 0
        self.subproblem_ip_work_total = 0

        def benders_callback(model, where):
            """
            Gurobi callback function for lazy constraint generation during IP solve.

            This callback is triggered at MIP nodes and integer solutions to:
            1. Track root node performance (gap)
            2. Generate Benders cuts for integer solutions using DSP and/or subproblem solves
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

                # Two-tier cut generation strategy
                num_cuts_added = 0

                # Tier 1: Try Dual Solution Pooling (DSP) first - fast cut generation
                if not is_first_saa and self.dual_lookup_ip:
                    start = time.perf_counter()
                    num_cuts_added = self.apply_dual_solution_pooling(
                        self.primary_pool, lazy=True
                    )
                    self.dual_lookup_ip_time += time.perf_counter() - start
                    self.dual_lookup_ip_cuts_generated += num_cuts_added
                    self.dual_lookup_ip_call_count += 1

                # Tier 2: If no cuts from DSP, solve subproblems to generate new cuts
                if num_cuts_added == 0:
                    start = time.perf_counter()
                    num_cuts_added, subproblem_objectives, subproblem_work = (
                        self.generate_benders_cuts(model, lazy=True, first=is_first_saa)
                    )
                    # Accumulate subproblem work units for IP phase
                    self.subproblem_ip_work_total += subproblem_work
                    # Update the master problem's z-variable solution with actual subproblem values
                    scenario_values = {
                        scenario: subproblem_objectives[scenario]
                        for scenario in self.scenario
                    }
                    model.cbSetSolution(self.x, self.first_stage_values)
                    model.cbSetSolution(self.z, scenario_values)
                    self.subproblem_ip_solve_time += time.perf_counter() - start
                    self.subproblem_ip_cuts_generated += num_cuts_added
                    self.subproblem_ip_solve_count += 1

        # ==================== IP MASTER PROBLEM SOLVE ====================
        # Configure Gurobi for lazy constraint generation
        self.master.setParam("OutputFlag", 1)
        self.master.setParam("LogToConsole", 1)
        if self.log_filename:
            self.master.setParam("LogFile", self.log_filename)
        self.master.setParam("MIPGap", self.tol)
        self.master.Params.lazyConstraints = 1
        self.master._rootgap = 1000

        # Solve IP master problem with Benders callback
        self.master.optimize(benders_callback)

        # Check if IP timed out
        if self.master.status == GRB.TIME_LIMIT:
            self.ip_timeout = True
            # Report IP gap at timeout and store bounds
            if (
                hasattr(self.master, "MIPGap")
                and hasattr(self.master, "ObjBound")
                and hasattr(self.master, "ObjVal")
            ):
                ip_gap = self.master.MIPGap * 100
                self.final_lower_bound = self.master.ObjBound
                self.final_upper_bound = self.master.ObjVal
                self.final_gap = ip_gap
                logger.info(f"IP phase timed out. Final MIP gap: {ip_gap:.2f}%")
                print(f"IP phase timed out. Final MIP gap: {ip_gap:.2f}%")
            else:
                logger.info(f"IP phase timed out. Status: {self.master.status}")
                print(f"IP phase timed out. Status: {self.master.status}")
        elif self.timelimit is not None and self.master.status == GRB.OPTIMAL:
            # Store final bounds for successful IP completion with timelimit
            self.final_lower_bound = self.master.ObjBound
            self.final_upper_bound = self.master.ObjVal
            self.final_gap = 0.0  # Optimal solution found

        # Record root node performance
        # If only 1 node was solved, use the final MIPGap instead
        if self.master.NodeCount == 1:
            self.root_node_gap.append(round(self.master.MIPGap * 100, 4))
        else:
            self.root_node_gap.append(self.master._rootgap)

        # Record master IP work units
        master_ip_work_total = self.master.getAttr("Work")

        self.extract_solution_values()

        # logger.info(f"\n--- Final IP Solution (SAA Iteration {saa_iteration}) ---")
        # for t in self.periods:
        #     on_generators = [g for g in self.thermal_gens if self.first_stage_values[(g, t)] > 0.5]
        #     if on_generators:
        #         logger.info(f"Time Period {t}: {', '.join(on_generators)} are ON")
        #     else:
        #         logger.info(f"Time Period {t}: No thermal generators are ON")
        # logger.info(f"----------------------------------------------------\n")
        self.dual_update()

        self.ip_time.append(round(self.master.Runtime, 3))
        logger.info(f"IP bound: {self.master.ObjBound:.3f}")
        logger.info(f"IP objective: {self.master.ObjVal:.3f}")
        logger.info(f"IP gap: {self.master.MIPGap * 100:.4f}%")
        logger.info(f"IP nodes: {self.master.NodeCount}")
        logger.info(f"IP runtime: {self.master.Runtime:.3f}s")

        self.initial_constraint_counter.append(self.master.NumConstrs)
        self.dual_pool_size.append(len(self.dual_storage))
        self.dual_pool_size_final.append(len(self.dual_storage))

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
        self.dual_lookup_ip_times.append(round(self.dual_lookup_ip_time, 2))
        self.cut_time_ip.append(
            round(self.dual_lookup_ip_time + self.subproblem_ip_solve_time, 2)
        )
        self.subproblem_ip_cuts.append(self.subproblem_ip_cuts_generated)
        self.subproblem_ip_counts.append(self.subproblem_ip_solve_count)
        self.dual_lookup_ip_cuts.append(self.dual_lookup_ip_cuts_generated)
        self.dual_lookup_ip_counts.append(self.dual_lookup_ip_call_count)

        # Store work unit totals for IP phase
        self.master_ip_work.append(master_ip_work_total)
        self.subproblem_ip_work.append(self.subproblem_ip_work_total)
        self.total_work_ip.append(master_ip_work_total + self.subproblem_ip_work_total)

        return 0

    def lp_initialize(self, method):
        """
        Initialize LP master problem with cuts using various heuristic techniques.

        Args:
            method (str): Initialization technique - 'vanilla', 'tech_1', 'tech_2', 'tech_1_boosted'
        """
        if method == "tech_1" or method == "tech_1_boosted":
            # Static Initialization: Add highest binding cuts for best historical solutions
            selected_dict = self.select_highest_binding_cuts(
                self.lp_optimal_first_stage_sols[:2], self.primary_pool
            )

        elif method == "tech_2":
            # Adaptive Initialization: Heuristic selection based on solution quality evaluation
            selected_dict, best_x, best_V = self.adaptive_cut_selection_for_lp(
                self.lp_optimal_first_stage_sols,
                self.lp_first_stage_sols,
                self.primary_pool,
                all=False,
            )

            # Warm-start master problem with the best solution found by heuristic
            V, _, best_z = self.compute_value_function_approximation(
                best_x, include_scenario_details=True
            )

            # Initialize scenario upper bounds for tech_2 (needed for upper bound tracking)
            self.scenario_upper_bounds = {s: best_z[s] for s in self.scenario}

            # Set warm-start values for master problem variables
            temp_x = {}
            idx = 0
            for g in self.thermal_gens:
                for t in self.periods:
                    temp_x[g, t] = best_x[idx]
                    idx += 1

            for g in self.thermal_gens:
                temp_x[g, 0] = 0.0

            temp_z = {s: best_z[s] for s in self.scenario}
            self.master.setAttr("start", self.x, temp_x)
            self.master.setAttr("start", self.z, temp_z)

        else:
            # Vanilla initialization: No cuts added, start with basic master problem
            assert method == "vanilla"
            if method == "vanilla":
                self.tech_1_lp_cons.append(0)
                self.tech_2_lp_cons.append(0)
            return 0

        # Add selected cuts to master problem and track them
        for scenario in self.scenario:
            self.lp_cuts[scenario].update(selected_dict[scenario])

        # Add the selected cuts to the master problem
        n_cons = self.add_cuts_to_master(selected_dict, check_lp_cuts=False)

        if method == "tech_1" or method == "tech_1_boosted":
            self.tech_1_lp_cons.append(n_cons)
            self.tech_2_lp_cons.append(0)
        elif method == "tech_2":
            self.tech_1_lp_cons.append(0)
            self.tech_2_lp_cons.append(n_cons)

    def solve_lp_relaxation(self, is_first_saa, problem_start_time=None):
        """
        Solve LP relaxation of master problem using iterative Benders decomposition.

        Args:
            is_first_saa (bool): True if this is the first SAA problem (no DSP available)
            problem_start_time (float): Start time for timeout checking
        """
        # Initialize performance tracking variables
        subproblem_lp_solve_time = 0
        dual_lookup_lp_time = 0
        subproblem_lp_cuts_generated = 0
        dual_lookup_lp_cuts_generated = 0
        master_lp_solve_time = 0
        lp_iterations = 0
        upper_bound = np.inf
        subproblem_lp_solve_count = 0
        dual_lookup_lp_call_count = 0

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

        # Initialize upper bound tracking
        if self.init_method != "tech_2" or is_first_saa:
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
            # Check for timeout
            if self.timelimit is not None and problem_start_time is not None:
                elapsed_time = time.perf_counter() - problem_start_time
                if elapsed_time >= self.timelimit:
                    self.lp_timeout = True
                    # Calculate final gap when timeout occurs
                    if lower_bound > 0:
                        final_gap = (upper_bound - lower_bound) / lower_bound * 100
                    else:
                        final_gap = float("inf")

                    # Store final bounds and gap for CSV output
                    self.final_upper_bound = upper_bound
                    self.final_lower_bound = lower_bound
                    self.final_gap = final_gap

                    logger.info(
                        f"LP phase timed out after {elapsed_time:.2f} seconds. Final gap: {final_gap:.2f}%"
                    )
                    break

            num_cuts_added = 0
            iter_sp_time = 0

            # Phase 1: Try Dual Solution Pooling (DSP) for cut generation
            if not is_first_saa and self.dual_lookup_lp:
                start = time.perf_counter()
                num_cuts_added = self.apply_dual_solution_pooling(
                    self.primary_pool, lazy=False
                )
                dual_lookup_lp_cuts_generated += num_cuts_added
                dual_lookup_lp_time += time.perf_counter() - start
                dual_lookup_lp_call_count += 1

            # Phase 2: If DSP found no cuts, solve subproblems to generate cuts
            if num_cuts_added == 0:
                start = time.perf_counter()

                num_cuts_generated, subproblem_objectives, subproblem_work = (
                    self.generate_benders_cuts(self.master, first=is_first_saa)
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

                # Store final bounds and gap for CSV output (normal completion)
                if self.timelimit is not None:
                    if lower_bound > 0:
                        final_gap = (upper_bound - lower_bound) / lower_bound * 100
                    else:
                        final_gap = float("inf")
                    self.final_upper_bound = upper_bound
                    self.final_lower_bound = lower_bound
                    self.final_gap = final_gap

                logger.info(
                    f"Iteration {lp_iterations}: LP UB: {upper_bound:.3f}, LB: {lower_bound:.3f}"
                )
                break  # LP relaxation converged

            # Calculate percentage optimality gap and continue iterating
            optimality_gap = (upper_bound - lower_bound) / lower_bound

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
            self.lp_active = self.identify_active_cuts()

            # Prepare cuts to transfer to IP phase (copy of active cuts)
            self.cuts_to_add_to_ip = {k: set(v) for k, v in self.lp_active.items()}

        # Record performance metrics
        self.lp_iterations.append(lp_iterations)
        x_order = np.array(
            [
                self.first_stage_values[g, t]
                for g in self.thermal_gens
                for t in self.periods
            ]
        )
        # Check for duplicates before adding final LP solution
        # lp_hash = hash(tuple(x_order))
        # if lp_hash not in self.lp_hash_set:
        #     self.lp_hash_set.add(lp_hash)
        #     self.lp_first_stage_sols.append(x_order)
        # Always add to optimal solutions (this represents the final optimal LP solution)
        self.lp_optimal_first_stage_sols.append(x_order)
        self.dual_pool_size.append(len(self.dual_storage))

        # For LP-only runs, also record final dual pool size
        if not self.solve_ip:
            self.dual_pool_size_final.append(len(self.dual_storage))

        self.subproblem_lp_times.append(round(subproblem_lp_solve_time, 2))
        self.dual_lookup_lp_times.append(round(dual_lookup_lp_time, 2))
        self.cut_time_lp.append(
            round(subproblem_lp_solve_time + dual_lookup_lp_time, 2)
        )
        self.subproblem_lp_cuts.append(subproblem_lp_cuts_generated)
        self.dual_lookup_lp_cuts.append(dual_lookup_lp_cuts_generated)
        self.master_lp_times.append(round(master_lp_solve_time, 2))
        self.subproblem_counts_lp.append(subproblem_lp_solve_count)
        self.dual_lookup_lp_counts.append(dual_lookup_lp_call_count)

        # Store work unit totals for LP phase
        self.master_lp_work.append(master_lp_work_total)
        self.subproblem_lp_work.append(subproblem_lp_work_total)
        self.total_work_lp.append(master_lp_work_total + subproblem_lp_work_total)

    def add_cuts_to_master(self, initialize_dict, check_lp_cuts=True):
        """
        Add Benders optimality cuts to master problem from selected dual solutions.

        For each scenario and each selected dual solution, constructs and adds the
        UC Benders cut based on generation and demand constraint duals.

        Args:
            initialize_dict (dict): {scenario: set(dual_indices)} - dual solutions to add cuts for
            check_lp_cuts (bool): If True, skip cuts already added to IP from LP phase

        Returns:
            int: Number of cuts added to master problem
        """
        LHS = {}  # Left-hand side expressions for cuts
        RHS = {}  # Right-hand side expressions for cuts
        n_cons = 0

        for s in self.scenario:
            for dual_idx in set(initialize_dict[s]):
                # Skip cuts that were already transferred from LP phase
                if check_lp_cuts:
                    if dual_idx in self.cuts_to_add_to_ip[s]:
                        continue

                # Get dual solution vectors
                generation_duals = self.generation_duals_array[dual_idx, :]
                demand_duals = self.demand_duals_array[dual_idx, :]

                # Construct Benders cut coefficients for commitment variables
                # Cut form: ∑_{g,t} [min_power[g] * gen_min_dual + max_power[g] * gen_max_dual] * x[g,t]
                #           + ∑_t demand[s,t] * demand_dual[t] <= z[s]

                coeffs = []
                vars_list = []
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
                        coeffs.append(dual_coeff)
                        vars_list.append(self.x[g, t])
                        commitment_idx += 1

                # Constant term: demand^T * demand_duals
                demand_constant = sum(
                    self.demand_scenarios[s][t] * demand_duals[t - 1]
                    for t in self.periods
                )

                # Build linear expression
                LHS[n_cons] = gp.LinExpr(demand_constant)  # Start with constant term
                LHS[n_cons].addTerms(coeffs, vars_list)  # Add variable terms
                RHS[n_cons] = gp.LinExpr(self.z[s])  # RHS is z_s variable
                n_cons += 1

        # Add all constructed cuts to master problem
        self.master.addConstrs(LHS[c] <= RHS[c] for c in range(n_cons))

        return n_cons

    # @profile
    def dual_update(self):
        """
        Update dual solution data structures using AtomicDualStorage.

        This method is now much simpler and faster since AtomicDualStorage
        handles all array management automatically.
        """
        # Ensure dual storage is initialized
        self.ensure_dual_storage_initialized()

        if len(self.dual_storage) == 0:
            return

        # Prepare arrays - this is now very fast with AtomicDualStorage
        self.prepare_dual_arrays()

        # For basic DSP (not curated), use all available dual solutions
        if not hasattr(self, "split_dual") or not self.split_dual:
            self.primary_pool = list(range(len(self.dual_storage)))

    def dual_update_incremental(self, old_size):
        """
        Incrementally update dual solution data structures for newly added duals only.
        Much faster than dual_update() when only a few duals are added.

        Args:
            old_size (int): Previous size of dual storage before new duals were added
        """
        # Ensure dual storage is initialized
        self.ensure_dual_storage_initialized()

        if len(self.dual_storage) == 0 or old_size >= len(self.dual_storage):
            return

        # Incrementally prepare arrays for new duals only
        self.prepare_dual_arrays_incremental(old_size)

        # For basic DSP (not curated), use all available dual solutions
        if not hasattr(self, "split_dual") or not self.split_dual:
            self.primary_pool = list(range(len(self.dual_storage)))

    def identify_active_cuts(self):
        """
        Identify which dual solutions generate active cuts at current LP solution.

        A cut is considered active if its violation is greater than the tolerance.
        Only considers dual solutions that were actually added as cuts during LP phase.

        Returns:
            dict: {scenario: set(dual_indices)} - active dual solutions per scenario
        """
        # Calculate commitment-dependent dual contributions using optimized method
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

        # Calculate cut violations: LHS - RHS for each cut
        scenario_cost_array = np.array(
            [self.second_stage_values[scenario] for scenario in self.scenario]
        )
        violation_values = subproblem_evaluations - scenario_cost_array

        # Identify active cuts (violations > tolerance) that were actually added
        active_duals_by_scenario = {}
        activity_tolerance = -1e-6  # Numerical tolerance for activity

        for scenario_idx in range(violation_values.shape[1]):
            # Find dual solutions with positive violation
            violating_dual_indices = np.where(
                violation_values[:, scenario_idx] > activity_tolerance
            )[0]
            # Only consider duals that were added as cuts during LP phase
            active_cuts = set.intersection(
                set(violating_dual_indices), self.lp_cuts[scenario_idx]
            )
            active_duals_by_scenario[scenario_idx] = active_cuts

        total_active_cuts = sum(len(cuts) for cuts in active_duals_by_scenario.values())
        print(f"Total active cuts to transfer to IP: {total_active_cuts}")

        return active_duals_by_scenario

    def apply_dual_solution_pooling(self, dual_list, lazy=True):
        """
        Apply dual solution pooling to find violated cuts quickly for UC problem.

        Args:
            dual_list (list): List of dual solution indices to evaluate
            lazy (bool): If True, add cuts as lazy constraints

        Returns:
            int: Number of cuts added
        """
        # Calculate commitment-dependent dual contributions using optimized method
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

        # Get current second-stage cost estimates
        scenario_cost_array = np.array(
            [self.second_stage_values[scenario] for scenario in self.scenario]
        )

        # Use numba-optimized function to find best dual for each scenario
        subproblem_optimal_values, optimal_dual_indices = (
            uc_benders_utils.find_largest_index_numba_uc(
                commitment_dual_product,
                self.dual_obj_random[np.ix_(dual_list, self.scenario)],
            )
        )

        # Convert relative indices back to original dual solution IDs
        optimal_duals = [dual_list[idx] for idx in optimal_dual_indices]

        # Calculate cut violations
        dual_violation_values = subproblem_optimal_values - scenario_cost_array

        # Precompute cut coefficients and constants for reuse using direct indexing
        cut_coefficients = {}
        cut_constants = {}

        for scenario in self.scenario:
            if dual_violation_values[scenario] > self.tol:
                dual_idx = optimal_duals[scenario]

                # Precompute variable coefficients (commitment-dependent terms)
                generation_duals = self.generation_duals_array[dual_idx, :]
                coeffs = []
                vars_list = []
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
                        coeffs.append(dual_coeff)
                        vars_list.append(self.x[g, t])
                        commitment_idx += 1

                cut_coefficients[scenario] = (coeffs, vars_list)

                # Precompute constant terms
                demand_duals = self.demand_duals_array[dual_idx, :]
                cut_constants[scenario] = sum(
                    self.demand_scenarios[scenario][t] * demand_duals[t - 1]
                    for t in self.periods
                )

        # Add cuts from dual pool with precomputed coefficients
        num_cuts_added = self.add_cuts_from_dual_pool_optimized(
            dual_violation_values,
            optimal_duals,
            cut_coefficients,
            cut_constants,
            self.master,
            self.scenario,
            lazy,
        )

        return num_cuts_added

    def add_cuts_from_dual_pool_optimized(
        self,
        dual_violation_values,
        optimal_duals,
        cut_coefficients,
        cut_constants,
        model,
        scenario_list,
        lazy=False,
    ):
        """
        Add cuts from dual solution pool with precomputed coefficients and constants.

        Args:
            dual_violation_values (np.array): Violation values for each scenario
            optimal_duals (list): Optimal dual solution indices for each scenario
            cut_coefficients (dict): Precomputed variable coefficients per scenario
            cut_constants (dict): Precomputed constant terms per scenario
            model: Master model to add cuts to
            scenario_list (list): List of scenarios
            lazy (bool): If True, add cuts as lazy constraints

        Returns:
            int: Number of cuts added
        """
        num_cuts_added = 0

        for scenario in self.scenario:
            # Check if violation is significant enough to add cut
            if (
                dual_violation_values[scenario] > self.tol
                and scenario in cut_coefficients
            ):
                dual_idx = optimal_duals[scenario]

                # Use precomputed coefficients and constants
                coeffs, vars_list = cut_coefficients[scenario]
                total_constant = cut_constants[scenario]

                # Build cut expression directly from precomputed components
                cut_expr = gp.LinExpr(total_constant)
                cut_expr.addTerms(coeffs, vars_list)
                cut_expr = self.z[scenario] >= cut_expr

                # Add the cut to the model
                if lazy:
                    model.cbLazy(cut_expr)
                else:
                    model.addConstr(
                        cut_expr,
                        name=f"benders_cut_{scenario}_{len(self.dual_storage)}",
                    )
                    self.lp_cuts[scenario].add(dual_idx)

                num_cuts_added += 1
                if dual_idx in self.dual_soln_optimal_counter:
                    self.dual_soln_optimal_counter[dual_idx] += 1

        return num_cuts_added

    def ip_initialize(self, method):
        """
        Initialize IP master problem with cuts and warm-start solution.

        This method:
        1. Completely rebuilds the master problem with binary variables and all structural constraints
        2. Rebuilds the subproblem for IP phase
        3. Adds back the active Benders cuts from LP phase
        4. Applies the selected initialization technique
        5. Sets up warm-start solution

        Args:
            method (str): Initialization method ('vanilla', 'tech_1', 'tech_2', 'tech_1_boosted')
        """
        # Completely rebuild master problem with binary variables and all structural UC constraints
        self.master = self.build_master(relaxation=False)
        self.master.setParam("TimeLimit", self.time_limit)
        base_constrs = self.master.NumConstrs

        # Rebuild subproblem at the start of IP phase
        print(f"  → Rebuilding subproblem at start of IP phase")
        self.subproblem = self.build_SP()

        # Add active Benders cuts from LP phase to IP master
        lp_cons = self.add_cuts_to_master(self.cuts_to_add_to_ip, check_lp_cuts=False)
        self.lp_cons_to_ip.append(lp_cons)  # Track cuts transferred from LP to IP
        self.master.update()  # Force Gurobi to update internal state
        after_lp_constrs = self.master.NumConstrs

        # For first SAA iteration: no previous IP solutions to use for initialization
        # Only carry over active LP cuts (already done above)
        if len(self.ip_first_stage_optimal_sols) == 0:
            self.tech_1_ip_cons.append(0)
            self.tech_2_ip_cons.append(0)
            return

        if method == "vanilla" or method == "tech_1" or method == "tech_1_boosted":
            # Use the most recent IP optimal solution for warm-start
            if len(self.ip_first_stage_optimal_sols) >= 1:
                best_x = self.ip_first_stage_optimal_sols[-1]
                self.master.setAttr(
                    "vType", self.x, GRB.BINARY
                )  # Convert to binary variables

                # Evaluate the best solution to get proper z-values
                _, zva = self.optimal_value_duals(best_x, get_duals=False)
                temp_z = {s: zva[s] for s in self.scenario}
                temp_x = {}
                idx = 0
                for g in self.thermal_gens:
                    for t in self.periods:
                        temp_x[g, t] = best_x[idx]
                        idx += 1

                for g in self.thermal_gens:
                    temp_x[g, 0] = 0.0

                # Set warm-start values
                self.master.setAttr("start", self.x, temp_x)
                self.master.setAttr("start", self.z, temp_z)

            if method == "tech_1":
                # Static Initialization: Add highest binding cuts for best IP solutions
                selected_dict = self.select_highest_binding_cuts(
                    self.ip_first_stage_optimal_sols[:2], self.primary_pool
                )
                n_cons = self.add_cuts_to_master(selected_dict, check_lp_cuts=True)
                self.tech_1_ip_cons.append(n_cons)

            elif method == "tech_1_boosted":
                # Boosted Static Initialization: Enhanced version with additional cuts
                n_cons = 0

                # First, determine how many cuts adaptive_cut_init would add
                selza, _, _ = self.select_cuts_for_ip_initialization(
                    self.ip_first_stage_optimal_sols,
                    self.ip_first_stage_sols,
                    self.primary_pool,
                )

                # Count cuts that would be new (not already from LP)
                for s in self.scenario:
                    for duals in set(selza[s]):
                        if duals in self.cuts_to_add_to_ip[s]:
                            continue
                        n_cons += 1

                # Add approximately the same number of cuts using best_new_sols heuristic
                target_num = max(1, n_cons // self.nS)  # Cuts per scenario
                selected_dict = self.select_best_dual_solutions(
                    self.ip_first_stage_optimal_sols[:2], self.primary_pool, target_num
                )
                n_cons = self.add_cuts_to_master(selected_dict, check_lp_cuts=True)
                self.tech_1_ip_cons.append(n_cons)

        elif method == "tech_2":
            # Adaptive Initialization: Use adaptive heuristic for intelligent cut selection
            self.master.setAttr("vType", self.x, GRB.BINARY)

            selected_dict, best_x, best_V = self.select_cuts_for_ip_initialization(
                self.ip_first_stage_optimal_sols,  # Previously optimal IP solutions
                self.ip_first_stage_sols,  # All previously generated IP solutions
                self.primary_pool,  # Available dual solutions
                all_adaptive=False,
            )

            # Update dual data structures and compute warm-start solution
            self.dual_update()
            V, _, best_z = self.compute_value_function_approximation(
                best_x, include_scenario_details=True
            )

            # Verify heuristic solution value matches exact evaluation
            assert (
                abs(V - best_V) < 1e-3
            ), f"Heuristic mismatch: V={V} vs best_V={best_V}"

            print(f"  → Adaptive Init IP bound: {best_V:.2f}")

            # Set warm-start solution
            temp_x = {}
            idx = 0
            for g in self.thermal_gens:
                for t in self.periods:
                    temp_x[g, t] = best_x[idx]
                    idx += 1

            for g in self.thermal_gens:
                temp_x[g, 0] = 0.0

            temp_z = {s: best_z[s] for s in self.scenario}
            self.master.setAttr("start", self.x, temp_x)
            self.master.setAttr("start", self.z, temp_z)

            # Add cuts selected by adaptive heuristic
            before_tech2_constrs = self.master.NumConstrs
            tech_2_cons = self.add_cuts_to_master(selected_dict, check_lp_cuts=True)
            self.master.update()  # Force Gurobi to update internal state
            after_tech2_constrs = self.master.NumConstrs
            self.tech_2_ip_cons.append(tech_2_cons)

        # Add appropriate tracking for methods that don't add additional cuts
        if method == "vanilla":
            self.tech_1_ip_cons.append(0)
            self.tech_2_ip_cons.append(0)
        elif method == "tech_1":
            self.tech_2_ip_cons.append(0)
        elif method == "tech_1_boosted":
            self.tech_2_ip_cons.append(0)
        elif method == "tech_2":
            self.tech_1_ip_cons.append(0)

    def select_highest_binding_cuts(self, first_solution_list, dual_list):
        """
        Select highest binding cuts for static initialization (tech_1).

        For each solution and scenario, finds the dual solution that gives the
        highest objective value (most binding cut).

        Args:
            first_solution_list (list): List of first-stage solutions to evaluate
            dual_list (list): List of dual solution indices to consider

        Returns:
            dict: {scenario: [dual_indices]} - selected dual solutions per scenario
        """
        if not first_solution_list:
            return {s: set() for s in self.scenario}

        final_dual_selections = []

        for sol in first_solution_list:
            # Use optimized commitment-weighted solution computation
            commitment_weighted_solution = self.get_commitment_weighted_solution(sol)

            # Calculate dual contributions for selected dual solutions
            commitment_dual_contributions = np.matmul(
                self.generation_duals_array[dual_list], commitment_weighted_solution
            )
            commitment_dual_contributions = np.squeeze(commitment_dual_contributions)

            # Use numba-optimized function to find best dual for each scenario
            _, optimal_dual_indices = uc_benders_utils.find_largest_index_numba_uc(
                commitment_dual_contributions,
                self.dual_obj_random[np.ix_(dual_list, self.scenario)],
            )

            # Convert relative indices back to original dual solution IDs
            optimal_duals = [dual_list[idx] for idx in optimal_dual_indices]
            final_dual_selections.append(optimal_duals)

        # Transpose to get dual selections by scenario across all solutions
        dual_selections_by_scenario = [list(row) for row in zip(*final_dual_selections)]

        # Convert to dictionary format expected by caller
        result_dict = {}
        for scenario_idx, scenario in enumerate(self.scenario):
            result_dict[scenario] = set(dual_selections_by_scenario[scenario_idx])

        return result_dict

    def adaptive_cut_selection_for_lp(
        self, phase_one_sols, final_sols, dual_list, one_optimal=False, all=False
    ):
        """
        Adaptive cut selection heuristic for LP initialization (tech_2).

        Uses heuristic evaluation to find the best solution from phase_one_sols
        and selects cuts that are most effective for that solution.

        Args:
            phase_one_sols (list): Previously optimal solutions to evaluate
            final_sols (list): All previously generated solutions
            dual_list (list): Available dual solutions
            one_optimal (bool): Solve only one best to optimality
            all (bool): Add highest binding cuts for all phase one sols

        Returns:
            tuple: (selected_cuts_dict, best_solution, best_value)
        """
        selected_dict = {s: set() for s in self.scenario}

        if not phase_one_sols:
            return selected_dict, [], 0

        best_V = np.inf
        best_x_vals = None
        best_lp_active = None

        # Find best solution from optimal solutions using dual evaluation
        for count, sol in enumerate(phase_one_sols):
            # Evaluate using available dual solutions
            sp_optimal, duals = self.evaluate_subproblems_with_dual_list(sol, dual_list)
            startup_cost = self.calculate_startup_cost(commitment_values=sol)
            val = startup_cost + sum(
                self.probability[s] * sp_optimal[s] for s in range(len(sp_optimal))
            )

            if val < best_V:
                best_V = val
                best_x_vals = sol
                best_lp_active = duals

            if all:
                for s in self.scenario:
                    selected_dict[s].add(duals[s])

        # Add cuts active at the best solution
        for s in self.scenario:
            selected_dict[s].add(best_lp_active[s])

        # Find the index of best solution in final_sols
        best_x_idx = 0
        for count, sol in enumerate(final_sols):
            if np.array_equal(best_x_vals, sol):
                best_x_idx = count
                break

        V_sel = dict((k, 0) for k in range(len(final_sols)))
        del V_sel[best_x_idx]  # deleting best_x to save computation

        # Precompute dual evaluations for efficiency using optimized method
        commitment_solutions = [
            self.get_commitment_weighted_solution(sol) for sol in final_sols
        ]
        dualx = np.array(commitment_solutions) @ self.generation_duals_array.T
        startup_costs = [
            self.calculate_startup_cost(commitment_values=sol) for sol in final_sols
        ]

        dual_evaluated = []
        if all:
            for first_sol in phase_one_sols:
                for count, sol in enumerate(final_sols):
                    if np.array_equal(first_sol, sol):
                        dual_evaluated.append(count)

        phase_two_iterations = 0

        while True:
            phase_two_iterations += 1
            if phase_two_iterations == 1:
                vals = {
                    solnum: self.evaluate_subproblems_with_selected_duals(
                        final_sols[solnum], selected_dict, return_optimal_duals=False
                    )
                    for solnum in V_sel.keys()
                }
                vals = {solnum: np.maximum(vals[solnum], 0) for solnum in V_sel.keys()}
            else:
                v = {
                    solnum: self.evaluate_subproblems_fast(
                        dualx[solnum], duals, add_cuts_for_scenarios
                    )
                    for solnum in V_sel.keys()
                }
                vals = {
                    solnum: np.maximum(vals[solnum], v[solnum])
                    for solnum in V_sel.keys()
                }

            V_sel = {
                solnum: startup_costs[solnum]
                + sum(self.probability[s] * vals[solnum][s] for s in self.scenario)
                for solnum in V_sel.keys()
            }
            V_sel = {k: v for (k, v) in V_sel.items() if v < best_V}

            if not V_sel:
                return selected_dict, final_sols[best_x_idx], best_V

            new_best = min(V_sel, key=V_sel.get)

            vals_sel = vals[new_best]
            min_V_sel = V_sel[new_best]

            if new_best in dual_evaluated:
                if one_optimal:
                    sp_solve_start = time.perf_counter()
                    V_sel[new_best], duals, _ = self.optimal_value_duals(
                        final_sols[new_best], get_duals=True
                    )

                    add_cuts_for_scenarios = self.scenario.copy()
                    for scen in self.scenario:
                        selected_dict[scen].add(duals[scen])

                return selected_dict, final_sols[new_best], V_sel[new_best]
            else:
                vals_hat, duals = self.evaluate_subproblems_with_dual_list(
                    final_sols[new_best], dual_list
                )
                V_sel[new_best] = startup_costs[new_best] + sum(
                    self.probability[s] * vals_hat[s] for s in self.scenario
                )

                dual_evaluated.append(new_best)

            vals_diff = np.round(vals_hat - vals_sel, 5)
            assert np.min(vals_diff) >= 0

            if all:
                # Add cuts for all scenarios when all_adaptive is enabled
                add_cuts_for_scenarios = self.scenario.copy()
            else:
                # Use original threshold-based selection
                d = dict(enumerate(vals_diff, 0))
                d_order = sorted(d.items(), key=lambda t: t[1], reverse=True)

                threshold = best_V - min_V_sel

                add_cuts_for_scenarios = []
                sum_val = 0

                for item in d_order:
                    scen, val = item
                    sum_val += val * self.probability[scen]
                    add_cuts_for_scenarios.append(scen)
                    if sum_val > threshold:
                        break

            for scen in add_cuts_for_scenarios:
                selected_dict[scen].add(duals[scen])

            if min(V_sel.values()) > best_V:
                return selected_dict, final_sols[best_x_idx], best_V

    def select_best_dual_solutions(self, first_solution_list, dual_list, n):
        """
        Select the best n dual solutions for each scenario based on objective values.

        Args:
            first_solution_list (list): List of first-stage solutions to evaluate
            dual_list (list): List of dual solution indices to consider
            n (int): Number of dual solutions to select per scenario

        Returns:
            dict: {scenario: [dual_indices]} - selected dual solutions per scenario
        """
        if not first_solution_list:
            return {s: set() for s in self.scenario}

        all_selections = []

        for sol in first_solution_list:
            # Use optimized commitment-weighted solution computation
            commitment_weighted_solution = self.get_commitment_weighted_solution(sol)

            # Calculate dual contributions for all dual solutions
            s1 = np.matmul(
                self.generation_duals_array[dual_list], commitment_weighted_solution
            )
            s1 = s1.reshape(len(dual_list), 1)

            # Add scenario-dependent terms
            final = s1 + self.dual_obj_random[np.ix_(dual_list, self.scenario)]
            finalT = final.T  # scenario * dual

            # Select top n dual solutions for each scenario
            idx = np.argsort(-finalT, axis=1)[:, :n]
            sol_selections = np.vectorize(lambda x: dual_list[x])(idx)
            all_selections.append(sol_selections)

        # Combine selections from all solutions
        if len(all_selections) > 1:
            combined_selections = np.hstack(all_selections)
        else:
            combined_selections = all_selections[0]

        # Create final dictionary with unique selections per scenario
        final_selections = {}
        for scenario_idx in range(combined_selections.shape[0]):
            scenario = self.scenario[scenario_idx]
            # Take unique values and limit to n
            unique_duals = list(dict.fromkeys(combined_selections[scenario_idx]))[:n]
            final_selections[scenario] = set(unique_duals)

        return final_selections

    def find_best_solution_heuristically(self, sols_list, dual_list):
        """
        Find the best solution from a list using heuristic evaluation with dual solution pooling.

        This method iteratively evaluates solutions using available dual solutions and
        progressively refines the evaluation by solving subproblems to optimality when needed.

        Args:
            sols_list (list): List of first-stage solutions to evaluate
            dual_list (list): Available dual solution indices

        Returns:
            tuple: (best_solution_index, best_value, updated_dual_list)
        """
        best_x = -1
        phase_one_iterations = 0
        V_hat = dict((k, 0) for k in range(len(sols_list)))
        startup_costs = [
            self.calculate_startup_cost(commitment_values=sol) for sol in sols_list
        ]

        optimal_evaluated = []
        # Track true optimal values (with startup costs) for solutions that have been exactly evaluated
        optimal_values = {}

        finding_best = True
        # Track dual storage size for incremental updates
        prev_dual_storage_size = len(self.dual_storage)

        while finding_best:
            phase_one_iterations += 1

            # OPTIMIZATION: Use incremental update if new duals were added in previous iteration
            current_dual_storage_size = len(self.dual_storage)
            if phase_one_iterations == 1:
                # First iteration: always do full update
                self.dual_update()
            elif current_dual_storage_size > prev_dual_storage_size:
                # Subsequent iterations with new duals: use incremental update
                self.dual_update_incremental(prev_dual_storage_size)
            # If no new duals, no update needed (arrays already up-to-date)
            prev_dual_storage_size = current_dual_storage_size

            if phase_one_iterations == 1:
                vals = {
                    solnum: self.evaluate_subproblems_with_dual_list(
                        sols_list[solnum], dual_list
                    )[0]
                    for solnum in V_hat.keys()
                }
                vals = {solnum: np.maximum(vals[solnum], 0) for solnum in V_hat.keys()}
            else:
                vals_ = {
                    solnum: self.evaluate_subproblems_with_dual_list(
                        sols_list[solnum], duals
                    )[0]
                    for solnum in V_hat.keys()
                }
                vals = {
                    solnum: np.maximum(vals[solnum], vals_[solnum])
                    for solnum in V_hat.keys()
                }

            V_hat = {
                solnum: startup_costs[solnum]
                + sum(self.probability[s] * vals[solnum][s] for s in self.scenario)
                for solnum in V_hat.keys()
            }

            # Restore true optimal values for solutions that have been exactly evaluated
            # (V_hat computation only uses second-stage costs, but optimal_values includes startup costs)
            for solnum in optimal_evaluated:
                if solnum in V_hat:
                    V_hat[solnum] = optimal_values[solnum]

            new_best = min(V_hat, key=V_hat.get)

            if new_best not in optimal_evaluated:
                start = time.perf_counter()
                optimal_val, duals, _ = self.optimal_value_duals(
                    sols_list[new_best], get_duals=True
                )
                dual_list.extend(duals)
                dual_list = list(dict.fromkeys(dual_list))

                V_hat[new_best] = optimal_val
                optimal_values[new_best] = (
                    optimal_val  # Store true optimal value with startup costs
                )
                optimal_evaluated.append(new_best)

                best_x = new_best  # (current optimal)
                new_best = min(V_hat, key=V_hat.get)

                if new_best == best_x:
                    finding_best = False
                else:
                    V_hat = {k: v for (k, v) in V_hat.items() if v <= optimal_val}
            else:
                finding_best = False

        return new_best, optimal_values[new_best], dual_list

    def evaluate_subproblems_fast(self, dualx, duals, add_cuts_for_scenarios):
        """
        Fast evaluation of subproblems using precomputed dual solution information.

        Args:
            dualx: Precomputed dual evaluations for the current solution (1D array)
            duals: Dual solution indices for each scenario (dict: scenario -> dual_idx)
            add_cuts_for_scenarios: List of scenario indices to evaluate (only evaluate critical scenarios)

        Returns:
            np.array: Subproblem objective values for each scenario
        """
        result = np.zeros(len(self.scenario))

        # Only evaluate scenarios that need cuts (more targeted and efficient)
        for scenario in add_cuts_for_scenarios:
            dual_idx = duals[scenario]
            # Check if dual_idx is valid in the global dual pool
            if dual_idx < self.generation_duals_array.shape[0]:
                result[scenario] = (
                    dualx[dual_idx] + self.dual_obj_random[dual_idx, scenario]
                )

        return result

    def evaluate_subproblems_with_selected_duals(
        self, sol, selected_dict, return_optimal_duals=True
    ):
        """
        Evaluate subproblems using a selected set of dual solutions for each scenario.

        Args:
            sol: First-stage solution to evaluate
            selected_dict: Dictionary mapping scenarios to sets of dual solution indices
            return_optimal_duals: If True, also return the optimal dual for each scenario

        Returns:
            np.array or tuple: Subproblem values, optionally with optimal duals
        """
        # Convert solution to commitment-weighted form using optimized method
        commitment_weighted_solution = self.get_commitment_weighted_solution(sol)

        subproblem_values = np.zeros(len(self.scenario))
        optimal_duals = {} if return_optimal_duals else None

        for scenario in self.scenario:
            if scenario in selected_dict and selected_dict[scenario]:
                dual_list = list(selected_dict[scenario])

                # Evaluate all selected duals for this scenario
                dual_evaluations = []
                for dual_idx in dual_list:
                    dual_contrib = np.dot(
                        self.generation_duals_array[dual_idx],
                        commitment_weighted_solution,
                    )
                    total_value = (
                        dual_contrib + self.dual_obj_random[dual_idx, scenario]
                    )
                    dual_evaluations.append((total_value, dual_idx))

                # Take the maximum (best) evaluation
                best_value, best_dual = max(dual_evaluations)
                subproblem_values[scenario] = best_value

                if return_optimal_duals:
                    optimal_duals[scenario] = best_dual

        if return_optimal_duals:
            return subproblem_values, optimal_duals
        else:
            return subproblem_values

    def select_cuts_for_ip_initialization(
        self, phase_one_sols, final_sols, dual_list, all_adaptive=False
    ):
        """
        Adaptive cut selection for IP initialization using iterative heuristic.

        This method implements the full CFLP/CMND-style adaptive heuristic that:
        1. Finds the best solution from phase_one_sols
        2. Iteratively evaluates remaining solutions to find beneficial cuts
        3. Uses a threshold-based approach to determine which cuts to add

        Args:
            phase_one_sols (list): Previously optimal IP solutions
            final_sols (list): All previously generated IP solutions
            dual_list (list): Available dual solutions

        Returns:
            tuple: (selected_dict, best_x, best_V)
        """
        if len(phase_one_sols) == 1:
            best_V, duals, _ = self.optimal_value_duals(
                phase_one_sols[0], get_duals=True
            )
            dual_list.extend(duals)
            dual_list = list(dict.fromkeys(dual_list))
            best_x = 0
        else:
            best_x, best_V, dual_list = self.find_best_solution_heuristically(
                phase_one_sols, dual_list
            )

        # Arrays are always synchronized with AtomicDualStorage
        self.dual_update()

        # Precompute commitment solutions, dual evaluations, and startup costs ONCE
        # (previously computed 3 times: here, at line 1807, and at line 1925)
        commitment_solutions = [
            self.get_commitment_weighted_solution(sol) for sol in final_sols
        ]
        dualx = np.array(commitment_solutions) @ self.generation_duals_array.T
        startup_costs = [
            self.calculate_startup_cost(commitment_values=sol) for sol in final_sols
        ]

        selected_dict = {s: set() for s in self.scenario}

        for count, sol in enumerate(phase_one_sols):
            _, ip_active = self.evaluate_subproblems_with_dual_list(sol, dual_list)
            for s in self.scenario:
                selected_dict[s].add(ip_active[s])

        x_order = np.array(phase_one_sols[best_x])
        solution_found = False
        for count, sol in enumerate(final_sols):
            if np.array_equal(x_order, sol):
                best_x = count
                solution_found = True
                break

        assert (
            solution_found
        ), f"Solution from phase_one_sols not found in final_sols. phase_one_sols has {len(phase_one_sols)} solutions, final_sols has {len(final_sols)} solutions"

        for scen in self.scenario:
            selected_dict[scen].update(self.cuts_to_add_to_ip[scen])

        for scen in self.scenario:
            dual_list.extend(self.cuts_to_add_to_ip[scen])
        dual_list = list(dict.fromkeys(dual_list))
        # This is necessary because cuts need to be part of dual list otherwise
        # assertion issue.

        V_sel = dict((k, 0) for k in range(len(final_sols)))
        del V_sel[best_x]  # deleting best_x to save computation

        # Note: commitment_solutions, dualx, and startup_costs already computed above
        # (removed duplicate computation that was here)

        phase_two_iterations = 0
        dual_evaluated = []
        optimal_evaluated = []
        # Track true optimal values (with startup costs) for solutions that have been exactly evaluated
        optimal_values = {}

        for first_sol in phase_one_sols:
            for count, sol in enumerate(final_sols):
                if np.array_equal(first_sol, sol):
                    dual_evaluated.append(count)

        while True:
            phase_two_iterations += 1

            if phase_two_iterations == 1:
                vals = {
                    solnum: self.evaluate_subproblems_with_selected_duals(
                        final_sols[solnum], selected_dict, return_optimal_duals=False
                    )
                    for solnum in V_sel.keys()
                }
                vals = {solnum: np.maximum(vals[solnum], 0) for solnum in V_sel.keys()}
            else:
                v = {
                    solnum: self.evaluate_subproblems_fast(
                        dualx[solnum], duals, add_cuts_for_scenarios
                    )
                    for solnum in V_sel.keys()
                }
                vals = {
                    solnum: np.maximum(vals[solnum], v[solnum])
                    for solnum in V_sel.keys()
                }

            V_sel = {
                solnum: startup_costs[solnum]
                + sum(self.probability[s] * vals[solnum][s] for s in self.scenario)
                for solnum in V_sel.keys()
            }

            # Restore true optimal values for solutions that have been exactly evaluated
            # (V_sel computation only uses second-stage costs, but optimal_values includes startup costs)
            for solnum in optimal_evaluated:
                if solnum in V_sel:
                    V_sel[solnum] = optimal_values[solnum]

            before_filter = len(V_sel)
            V_sel = {
                k: v for (k, v) in V_sel.items() if v < best_V
            }  # removes those keys

            if not V_sel:
                return selected_dict, final_sols[best_x], best_V

            new_best = min(V_sel, key=V_sel.get)

            vals_sel = vals[new_best]  # value of every scenario based on selected dict
            min_V_sel = V_sel[new_best]

            best_changed = False

            if new_best in optimal_evaluated:
                assert new_best in dual_evaluated
                return selected_dict, final_sols[new_best], optimal_values[new_best]

            elif new_best in dual_evaluated:
                sp_solve_start = time.perf_counter()

                # Track dual storage size before solving subproblems
                old_dual_storage_size = len(self.dual_storage)

                V_new_best, duals, vals_hat = self.optimal_value_duals(
                    final_sols[new_best], get_duals=True
                )
                V_sel[new_best] = V_new_best
                optimal_values[new_best] = (
                    V_new_best  # Store true optimal value with startup costs
                )

                # Track which duals are NEW (incremental update optimization)
                old_dual_set = set(dual_list)
                dual_list.extend(duals)
                dual_list = list(dict.fromkeys(dual_list))
                new_duals = [d for d in dual_list if d not in old_dual_set]

                optimal_evaluated.append(new_best)

                # INCREMENTAL UPDATE: Only update arrays for new duals if any were added
                if new_duals:
                    # Use incremental update that only processes new duals
                    self.dual_update_incremental(old_dual_storage_size)

                    # OPTIMIZATION: Only compute dualx for NEW dual columns, then append
                    # Old approach: dualx = np.array(commitment_solutions) @ self.generation_duals_array.T
                    # New approach: only compute for new duals
                    new_dualx = (
                        np.array(commitment_solutions)
                        @ self.generation_duals_array[new_duals, :].T
                    )
                    dualx = np.hstack([dualx, new_dualx])

                if V_sel[new_best] < best_V:
                    best_V = V_sel[new_best]
                    best_x = new_best
                    best_changed = True
            else:
                vals_hat, duals = self.evaluate_subproblems_with_dual_list(
                    final_sols[new_best], dual_list
                )
                V_sel[new_best] = startup_costs[new_best] + sum(
                    self.probability[s] * vals_hat[s] for s in self.scenario
                )
                dual_evaluated.append(new_best)

            vals_diff = np.round(vals_hat - vals_sel, 5)

            if (new_best not in optimal_evaluated) or (not best_changed):
                assert np.min(vals_diff) >= 0

                if all_adaptive:
                    # Add cuts for all scenarios when all_adaptive is enabled
                    add_cuts_for_scenarios = self.scenario.copy()
                else:
                    # Use original threshold-based selection
                    d = dict(enumerate(vals_diff, 0))
                    d_order = sorted(d.items(), key=lambda t: t[1], reverse=True)

                    threshold = best_V - min_V_sel

                    add_cuts_for_scenarios = []
                    sum_scen_obj = 0

                    for item in d_order:
                        scen, val = item
                        sum_scen_obj += val * self.probability[scen]
                        add_cuts_for_scenarios.append(scen)
                        if sum_scen_obj > threshold:
                            break

                for scen in add_cuts_for_scenarios:
                    selected_dict[scen].add(duals[scen])
            else:
                # Enter this if optimal changes. in that case we also change the bestv values.
                add_cuts_for_scenarios = self.scenario.copy()
                # Adding all cuts for the guy which is optimal
                for scen in self.scenario:
                    selected_dict[scen].add(duals[scen])

            if min(V_sel.values()) > best_V:
                return selected_dict, final_sols[best_x], best_V

        return selected_dict, final_sols[best_x], best_V

    def solve_saa_iteration(self, saa_iteration):
        """Solve a single SAA iteration."""
        start = time.perf_counter()

        self.master = self.build_master(relaxation=True)
        self.master.setParam("TimeLimit", self.time_limit)
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


def get_combination(details):
    combinations = {
        (0, 0, 0): "NoReuse",
        (0, 1, 0): "DSP",
        (0, 2, 0): "CuratedDSP",
        (0, 2, 1): "StaticInit",
        (0, 2, 2): "AdaptiveInit",
        (0, 3, 0): "RandomDSP",
        (1, 0, 0): "NoReuse",
        (1, 1, 0): "DSP",
        (1, 2, 0): "CuratedDSP",
        (1, 2, 1): "StaticInit",
        (1, 2, 2): "AdaptiveInit",
        (1, 2, 3): "BoostedStaticInit",
        (1, 3, 0): "RandomDSP",
    }
    if details not in combinations:
        raise AlgorithmDetailsError("Error: Input is not correct.")
    return combinations[details]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Unit Commitment Benders Decomposition"
    )
    parser.add_argument("algorithm_details", metavar="N", type=int, nargs=3)
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
    parser.add_argument(
        "--twice",
        action="store_true",
        help="Run for 2 times the timelimit",
    )
    args = parser.parse_args()

    solve_ip, dual_lookup, initialization_method = args.algorithm_details
    instance_file, nscen, std_dev, nsaa = args.data

    # Generate instance on-the-fly if --generate flag is used
    if args.generate:
        n_gens = args.generate[0]
        n_days = args.generate[1] if len(args.generate) > 1 else 1
        difficulty = args.generate[2] if len(args.generate) > 2 else 1

        instance_file = uc_benders_utils.UCinst.generate_instance(
            n_generators=n_gens,
            n_days=n_days,
            periods_per_day=24,
            difficulty=difficulty,
            output_file=instance_file,
        )

    # Create reduced generator instance data if specified
    instance_data = None
    if args.num_generators:
        instance_data = uc_benders_utils.UCinst.create_reduced_generator_instance(
            input_filename_or_data=instance_file,
            num_generators=args.num_generators,
        )

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
        logger = setup_logger(results_filename)
        logger.info("Using Extensive Form - LP Relaxation Quality Analysis")
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
            logger.info(f"\n{'#'*70}")
            logger.info(f"### SAA Iteration {saa_iteration}")
            logger.info(f"{'#'*70}")
            uc_instance.generate_demand_scenarios(num_scenarios, demand_std_dev)
            # No renewable scenario generation needed - renewables are decision variables

            # Analyze LP relaxation quality (solves both LP and IP)
            start_time = time.perf_counter()
            analysis_results = uc_instance.analyze_lp_relaxation_quality(
                time_limit=args.timelimit,
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

        # Log summary table
        logger.info("\n" + "=" * 70)
        logger.info("SUMMARY ACROSS ALL SAA ITERATIONS")
        logger.info("=" * 70)
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

        solved_saa_iterations = list(range(1, num_saa_iterations + 1))

    else:
        # Original Benders mode (multi-cut)
        algorithm_name = get_combination((solve_ip, dual_lookup, initialization_method))
        multi_algorithm_name = f"multi_{algorithm_name}"
        if solve_ip:
            results_filename = (
                f"detailed-results/uc/IP/multi_UC_IP_{data_string}_{algorithm_name}.op"
            )
        else:
            results_filename = (
                f"detailed-results/uc/LP/multi_UC_LP_{data_string}_{algorithm_name}.op"
            )


        logger = setup_logger(results_filename)
        logger.info(f"data: {args.data}")
        logger.info(f"Algorithm combination: {algorithm_name}")

        np.random.seed(3)
        # Use reduced instance data if available, otherwise use file
        instance_to_load = instance_data if instance_data is not None else instance_file

        # Apply twice modifier if specified
        effective_timelimit = args.timelimit
        if args.twice:
            effective_timelimit = args.timelimit * 2
            logger.info(
                f"Using --twice flag: effective timelimit = {effective_timelimit:.2f} seconds"
            )

        benders_solver = Benders(
            args.algorithm_details,
            instance_to_load,
            int(nscen),
            max_periods=args.periods,
            log_filename=results_filename,
            time_limit=effective_timelimit,
        )

        # Set timelimit for timeout detection
        benders_solver.timelimit = effective_timelimit

        num_saa_iterations = int(nsaa)
        num_scenarios = int(nscen)
        demand_std_dev = float(std_dev)

        # When --twice flag is used (timelimit experiments), only run 1 SAA iteration
        if args.twice:
            logger.info("--twice flag specified: running only 1 SAA iteration")
            saa_iterations_to_run = [1]
        else:
            no_reuse_iterations = [
                1,
                2,
                int(num_saa_iterations / 2 + 1),
                num_saa_iterations,
            ]
            saa_iterations_to_run = []
            for saa_iteration in range(1, num_saa_iterations + 1):
                if saa_iteration in no_reuse_iterations or dual_lookup != 0:
                    saa_iterations_to_run.append(saa_iteration)

        solved_saa_iterations = []
        for saa_iteration in saa_iterations_to_run:
            benders_solver.generate_demand_scenarios(num_scenarios, demand_std_dev)
            # No renewable scenario generation needed - renewables are decision variables
            benders_solver.solve_saa_iteration(saa_iteration)
            solved_saa_iterations.append(saa_iteration)

    if not args.extensive:
        # Original Benders results processing
        data_dict = {
            "Instance": data_string,
            "Scenarios": num_scenarios,
            "Method": multi_algorithm_name,
        }

        # When timelimit is specified (and not the default 3600), use simplified CSV output
        if args.timelimit is not None and args.timelimit != 3600:
            data_dict["Timelimit"] = args.timelimit
            data_dict["Upper_Bound"] = (
                benders_solver.final_upper_bound
                if benders_solver.final_upper_bound is not None
                else "N/A"
            )
            data_dict["Lower_Bound"] = (
                benders_solver.final_lower_bound
                if benders_solver.final_lower_bound is not None
                else "N/A"
            )
            data_dict["Gap"] = (
                benders_solver.final_gap
                if benders_solver.final_gap is not None
                else "N/A"
            )
            data_dict["LP_Timeout"] = benders_solver.lp_timeout
            data_dict["IP_Timeout"] = benders_solver.ip_timeout
        else:
            if solve_ip:
                columns = [
                    "Total times",
                    "IP time",
                    "LP relaxation time",
                    "IP nodes",
                    "Root node gap",
                    "dual lookup ip counts",
                    "subproblem ip counts",
                    "cut time IP",
                    "primary pool size",
                    "x feas counter ip",
                    "initial constraint counter",
                    "dual lookup IP times",
                    "Subproblem IP times",
                    "IP initialization time",
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
                    "primary pool size",
                    "x feas counter lp",
                    "LP final cons",
                    "subproblem counts lp",
                    "dual lookup lp cuts",
                    "subproblem lp cuts",
                    "subproblem_lp_times",
                    "dual_lookup_lp_times",
                    "lp initialization time",
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
                        # Average excludes the first SAA iteration (index 0) for all methods
                        data_dict[f"{col} average"] = round(np.mean(col_data[1:]), 3)

        # Average LP heuristic time excludes the first SAA iteration for all methods
        data_dict["avg LP heuristic time"] = (
            f"{np.mean(benders_solver.lp_initialization_time[1:]):.2f}"
            if len(benders_solver.lp_initialization_time) > 1
            else "0.00"
        )
        data_dict["final num of dual solution"] = (
            f"{benders_solver.dual_pool_size[-1]}"
            if benders_solver.dual_pool_size
            else "0"
        )

    # data_dict is already created for extensive form above

    df = pd.DataFrame([data_dict])

    # Also append to global results file (separate for multi-cut)
    df.to_csv(
        "results_multi_uc.csv",
        index=False,
        quoting=3,
        sep=",",
        escapechar=",",
        mode="a",
        header=not os.path.exists("results_multi_uc.csv"),
    )

    logger.info("")

    if not args.extensive:
        saa_iteration_list = [f"SAA {i}" for i in solved_saa_iterations]
        logger.info("=" * 80)
        logger.info("DETAILED RESULTS SUMMARY")
        logger.info("=" * 80)
        # logger.info("")

        # Log constraint information
        constraint_data = [
            ["Tech 1 LP cons", *benders_solver.tech_1_lp_cons],
            ["Tech 2 LP cons", *benders_solver.tech_2_lp_cons],
            ["DSP cuts - LP", *benders_solver.dual_lookup_lp_cuts],
            ["SP cuts - LP", *benders_solver.subproblem_lp_cuts],
            ["DSP count - LP", *benders_solver.dual_lookup_lp_counts],
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
                ["Tech 1 IP cons", *benders_solver.tech_1_ip_cons],
                ["Tech 2 IP cons", *benders_solver.tech_2_ip_cons],
                ["DSP cuts - IP", *benders_solver.dual_lookup_ip_cuts],
                ["SP cuts - IP", *benders_solver.subproblem_ip_cuts],
                ["DSP count - IP", *benders_solver.dual_lookup_ip_counts],
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

        # LP time breakdown
        lp_time_data = [
            ["Master time", *benders_solver.master_lp_times],
            ["DSP time", *benders_solver.dual_lookup_lp_times],
            ["SP time", *benders_solver.subproblem_lp_times],
            ["Init time", *benders_solver.lp_initialization_time],
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
            # IP time breakdown
            ip_time_data = [
                ["DSP time", *benders_solver.dual_lookup_ip_times],
                ["SP time", *benders_solver.subproblem_ip_times],
                ["Init time", *benders_solver.ip_initialization_time],
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

        # Summary information
        summary_data = [
            ["LP Iterations", benders_solver.lp_iterations],
            ["# dual solutions in primary pool", benders_solver.primary_pool_size],
            ["# Total dual solutions after SAA", benders_solver.dual_pool_size_final],
            ["# Total dual solutions", benders_solver.dual_pool_size],
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
        except Exception:
            pass
