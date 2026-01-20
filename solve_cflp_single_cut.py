import argparse
from sys import addaudithook
import numpy as np
import time
import math
import cflp_benders_utils_single_cut
from gurobipy import GRB
import gurobipy as gp
import pandas as pd
import logging
import os
from tabulate import tabulate

"""
This file does benders decomposition for the facility
location instances.
"""


class Benders(cflp_benders_utils_single_cut.CFLPinst):

    def __init__(self, algorithm_details, instance, filename=None):
        """Intializes and creates the instance data"""

        solve_ip, dual_lookup, init = algorithm_details

        # Skip dual collection if dual_lookup is 0 (NoReuse method)
        skip_dual_collection = dual_lookup == 0

        super().__init__(instance, skip_dual_collection=skip_dual_collection)
        if filename:
            self.load_instance(filename)
        else:
            self.createNewInst(*instance[:3])

        self.dual_lookup_lp = True if dual_lookup >= 1 else False
        self.dual_lookup_ip = True if dual_lookup >= 1 else False
        self.split_dual = True if dual_lookup == 2 else False

        self.solve_ip = solve_ip

        init_methods = {0: "vanilla", 1: "tech_1", 2: "tech_2", 3: "tech_1_boosted"}
        self.init_method = init_methods[init]

        self.lp_active_cuts = True if solve_ip == 1 else False

        self.master_lp_times = []

        self.subproblem_lp_times = []
        self.subproblem_ip_times = []

        self.dual_lookup_lp_times = []
        self.dual_lookup_ip_times = []

        self.cut_time_lp = []
        self.cut_time_ip = []

        self.subproblem_lp_cuts = []
        self.subproblem_ip_cuts = []

        self.dual_lookup_lp_cuts = []
        self.dual_lookup_ip_cuts = []

        self.dual_lookup_lp_counts = []
        self.dual_lookup_ip_counts = []

        self.subproblem_ip_counts = []
        self.subproblem_counts_lp = []

        self.initial_constraint_counter = []
        self.x_feas_counter_lp = []
        self.x_feas_counter_ip = []

        self.lp_iterations = []

        self.sol_repeat_counter = {}

        self.lp_first_stage_sols = []
        self.lp_optimal_first_stage_sols = []
        self.ip_first_stage_sols = []
        self.ip_first_stage_optimal_sols = []

        self.lp_relaxation_time = []
        self.ip_time = []

        self.total_times = []

        self.lp_final_cons = []
        self.tech_1_lp_cons = [0]
        self.tech_1_ip_cons = [0]
        self.tech_2_lp_cons = [0]
        self.tech_2_ip_cons = [0]

        self.lp_cons_to_ip = [0]
        self.ip_start_cons = []
        self.ip_gap = []
        self.ip_nodes = []

        self.dual_soln_optimal_counter = {}  # dual id : how many times cuts off soln

        self.dual_pool_size = []
        self.dual_pool_size_final = []
        self.primary_pool_size = [0]
        self.root_node_time = []
        self.root_node_gap = []
        self.tol = 1e-5
        self.lp_gap = 0.0001  # 0.01% gap for LP relaxation
        self.ip_gap_limit = 0.0001  # 0.01% gap for IP
        self.lp_initialization_time = [0]
        self.ip_initialization_time = [0]

    def benders(self, saa_number):

        first = 1 if saa_number == 1 else 0

        self.cuts_to_add_to_ip = set()
        self.cut_history = []

        if not first and not self.skip_dual_collection:
            if self.split_dual:
                self.primary_pool = []

                add_previous_dsp_cuts = True
                add_active = True
                add_previous_saa = True

                if add_previous_dsp_cuts:
                    self.primary_pool = [
                        k for k, v in self.dual_soln_optimal_counter.items() if v > 0
                    ]

                add_active = False
                # change later
                if add_active:
                    for scen in self.scenario:
                        for dual in self.lp_active[scen]:
                            if dual not in self.primary_pool:
                                self.primary_pool.append(dual)

                if add_previous_saa:
                    if len(self.dual_pool_size_final) == 1:
                        more_duals = list(range(self.dual_pool_size_final[-1]))
                    else:
                        more_duals = list(
                            range(
                                self.dual_pool_size_final[-2],
                                self.dual_pool_size_final[-1],
                            )
                        )
                    self.primary_pool.extend(more_duals)

                self.primary_pool = list(dict.fromkeys(self.primary_pool))
            else:
                self.primary_pool = list(range(len(self.capacity_duals_array)))

            self.primary_pool_size.append(len(self.primary_pool))
            self.dual_obj_random = np.matmul(
                self.demand_duals_array, self.Demand_array
            )  # dual * #scenarios Then we will pick best in every row.
            init_time = time.perf_counter()
            self.lp_initialize(self.init_method)
            self.lp_initialization_time.append(
                round(time.perf_counter() - init_time, 3)
            )

        problem_start = time.perf_counter()
        self.solve_lp_relaxation(first)
        self.lp_relaxation_time.append(round(time.perf_counter() - problem_start, 3))
        self.x_feas_counter_lp.append(len(self.lp_first_stage_sols))

        if not self.solve_ip:
            return 0

        self.first_stage_values = {}
        self.second_stage_values = {}

        if first:
            self.master.setAttr("vType", self.x, GRB.BINARY)
        else:
            init_time = time.perf_counter()
            self.ip_initialize(self.init_method)
            self.ip_initialization_time.append(
                round(time.perf_counter() - init_time, 3)
            )

        self.sp_ip_time = 0
        self.sp_ip_cut = 0
        self.sp_ip_count = 0
        self.dual_lookup_ip_time = 0
        self.dual_lookup_ip_cut = 0
        self.dual_lookup_ip_count = 0

        def benders_callback(model, where):

            if where == GRB.Callback.MIPNODE:
                depth = model.cbGet(GRB.Callback.MIPNODE_NODCNT)

                if depth == 0:
                    if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.Status.OPTIMAL:
                        model._runtime = model.cbGet(GRB.Callback.RUNTIME)
                        objbnd = model.cbGet(GRB.Callback.MIPNODE_OBJBND)
                        objval = model.cbGet(GRB.Callback.MIPNODE_OBJBST)
                        model._rootgap = abs(objbnd - objval) * 100 / abs(objval)

            if where == GRB.Callback.MIPSOL:
                self.first_stage_values = model.cbGetSolution(self.x)
                self.second_stage_values = model.cbGetSolution(self.z)

                x_order = tuple(
                    int(self.first_stage_values[ware]) for ware in self.Warehouse
                )
                int_hash = hash(x_order)

                if int_hash not in self.int_hash_set:
                    self.int_hash_set.add(int_hash)
                    self.ip_first_stage_sols.append(x_order)

                cutadded = 0

                if not first and self.dual_lookup_ip:
                    start = time.perf_counter()
                    cutadded = self.cache_dsp(self.primary_pool, lazy=True)
                    self.dual_lookup_ip_time += time.perf_counter() - start
                    self.dual_lookup_ip_cut += cutadded
                    self.dual_lookup_ip_count += 1

                if cutadded == 0:
                    start = time.perf_counter()
                    cutadded, zup = self.add_single_cut(model, lazy=True, first=first)
                    # For single cut, we only have one z variable, so set it to zup
                    model.cbSetSolution(self.x, self.first_stage_values)
                    model.cbSetSolution(self.z, zup)
                    self.sp_ip_time += time.perf_counter() - start
                    self.sp_ip_cut += cutadded
                    self.sp_ip_count += 1

        self.master.setParam("OutputFlag", True)
        self.master.setParam("MIPGap", self.ip_gap_limit)
        self.master.Params.lazyConstraints = 1
        self.master._runtime = 0
        self.master._rootgap = 1000
        self.master.optimize(benders_callback)
        self.root_node_time.append(self.master._runtime)
        self.root_node_gap.append(self.master._rootgap)

        self.extract_solution_values()
        self.dual_update()

        self.ip_time.append(round(self.master.Runtime, 3))
        logger.info(f"IP bound: {self.master.ObjBound:.3f}")

        print(f"IP bound: {self.master.ObjBound:.3f}")
        print()
        self.initial_constraint_counter.append(self.master.NumConstrs)
        if not self.skip_dual_collection:
            self.dual_pool_size.append(len(self.capacity_duals_array))
            self.dual_pool_size_final.append(len(self.capacity_duals_array))
        else:
            self.dual_pool_size.append(0)
            self.dual_pool_size_final.append(0)

        self.ip_gap.append(round(self.master.MIPGap * 100, 4))
        self.ip_nodes.append(self.master.NodeCount)

        x_order = tuple(self.first_stage_values[ware] for ware in self.Warehouse)

        if x_order not in self.ip_first_stage_optimal_sols:
            self.ip_first_stage_optimal_sols.append(x_order)

        if x_order not in self.ip_first_stage_sols:
            self.ip_first_stage_sols.append(x_order)

        self.x_feas_counter_ip.append(len(self.ip_first_stage_sols))

        self.subproblem_ip_times.append(round(self.sp_ip_time, 2))
        self.dual_lookup_ip_times.append(round(self.dual_lookup_ip_time, 2))
        self.cut_time_ip.append(round(self.dual_lookup_ip_time + self.sp_ip_time, 2))
        self.subproblem_ip_cuts.append(self.sp_ip_cut)
        self.subproblem_ip_counts.append(self.sp_ip_count)
        self.dual_lookup_ip_cuts.append(self.dual_lookup_ip_cut)
        self.dual_lookup_ip_counts.append(self.dual_lookup_ip_count)

        return 0

    def lp_initialize(self, method):

        if method == "tech_1" or method == "tech_1_boosted":
            selected_dict = self.select_highest_binding_cuts(
                self.lp_optimal_first_stage_sols[:2], self.primary_pool
            )

        elif method == "tech_2":
            selected_dict, best_x, best_V = self.adaptive_cut_selection_for_lp(
                self.lp_optimal_first_stage_sols,
                self.lp_first_stage_sols,
                self.primary_pool,
            )

            ctx = np.dot(self.setup_np, best_x)
            dualctx = np.multiply(self.capacity_np, best_x)
            s1 = np.matmul(self.capacity_duals_array, dualctx)
            V, _, best_z = self.compute_value_function_approximation(
                ctx, s1, include_scenario_details=True
            )

            temp_x = {ware: best_x[count] for count, ware in enumerate(self.Warehouse)}
            temp_z = sum(self.probab * best_z[s] for s in self.scenario)
            self.zup = temp_z
            self.master.setAttr("start", self.x, temp_x)
            self.z.start = temp_z
        else:
            assert method == "vanilla"
            return 0

        n_cons = self.add_cuts_to_master(selected_dict, check_lp_cuts=False)

        if method == "tech_1":
            self.tech_1_lp_cons.append(n_cons)
        elif method == "tech_2":
            self.tech_2_lp_cons.append(n_cons)

    def solve_lp_relaxation(self, first):

        sp_lp_time = 0
        dual_lookup_lp_time = 0

        sp_lp_cut = 0
        dual_lookup_lp_cut = 0

        master_lp_time = 0
        lp_iterations = 0

        upperbound = np.inf

        sp_count_lp = 0
        dual_lookup_lp_count = 0

        start = time.perf_counter()
        self.master.optimize()

        master_lp_time += time.perf_counter() - start
        lp_iterations += 1

        status = self.master.status
        if status != 2:
            raise Exception(f"Master problem status - {status}")

        self.subproblem = self.build_SP()
        self.extract_solution_values(problem="LP")

        if self.init_method != "tech_2" or first:
            self.zup = {s: math.inf for s in self.scenario}
            upperbound = np.inf

        lowerbound = self.master.ObjBound
        dif = upperbound - lowerbound
        cutadded = 1

        while True:
            cutadded = 0

            if not first and self.dual_lookup_lp:
                start = time.perf_counter()
                cutadded = self.cache_dsp(self.primary_pool, lazy=False)
                dual_lookup_lp_cut += cutadded
                dual_lookup_lp_time += time.perf_counter() - start
                dual_lookup_lp_count += 1

            if cutadded == 0:
                start = time.perf_counter()

                cuts, self.zup = self.add_single_cut(self.master, first=first)
                ub = self.upperbound()
                upperbound = min(ub, upperbound)

                cutadded = cuts
                elapsed = time.perf_counter() - start
                sp_lp_time += elapsed
                sp_lp_cut += cuts
                sp_count_lp += 1

            if lowerbound == 0:
                lowerbound = 0.1

            if cutadded == 0 or (dif < self.lp_gap):
                ub = self.upperbound()
                upperbound = min(ub, upperbound)
                logger.info(
                    f"Iteration {lp_iterations}: LP UB: {upperbound:.3f}, LB: {lowerbound:.3f}"
                )
                print(
                    f"Iteration {lp_iterations}: LP UB: {upperbound:.3f}, LB: {lowerbound:.3f}"
                )
                break

            dif = (upperbound - lowerbound) / lowerbound
            print(f"Iteration {lp_iterations}: LP UB: {upperbound}, LB: {lowerbound}")
            start = time.perf_counter()
            self.master.optimize()
            elapsed = time.perf_counter() - start
            master_lp_time += elapsed

            status = self.master.status
            if status != 2:
                raise Exception(f"Master problem status - {status}")

            lowerbound = self.master.ObjBound

            self.extract_solution_values(problem="LP")
            x_order = np.array(
                [self.first_stage_values[ware] for ware in self.Warehouse]
            )
            self.lp_first_stage_sols.append(x_order)
            lp_iterations += 1

        self.lp_final_cons.append(self.master.NumConstrs)

        self.dual_update()
        self.lp_active = self.identify_active_cuts()

        # For single-cut, carry active cuts from LP to IP initialization
        self.cuts_to_add_to_ip = self.lp_active.copy()
        self.lp_iterations.append(lp_iterations)

        x_order = np.array([self.first_stage_values[ware] for ware in self.Warehouse])
        self.lp_first_stage_sols.append(x_order)  # Track all LP solutions generated
        self.lp_optimal_first_stage_sols.append(x_order)
        if not self.skip_dual_collection:
            self.dual_pool_size.append(len(self.capacity_duals_array))
            if not self.solve_ip:
                self.dual_pool_size_final.append(len(self.capacity_duals_array))
        else:
            self.dual_pool_size.append(0)
            if not self.solve_ip:
                self.dual_pool_size_final.append(0)

        self.subproblem_lp_times.append(round(sp_lp_time, 2))
        self.dual_lookup_lp_times.append(round(dual_lookup_lp_time, 2))
        self.cut_time_lp.append(round(sp_lp_time + dual_lookup_lp_time, 2))
        self.subproblem_lp_cuts.append(sp_lp_cut)
        self.dual_lookup_lp_cuts.append(dual_lookup_lp_cut)
        self.master_lp_times.append(round(master_lp_time, 2))
        self.subproblem_counts_lp.append(sp_count_lp)
        self.dual_lookup_lp_counts.append(dual_lookup_lp_count)

    def add_cuts_to_master(self, initialize_set, check_lp_cuts=True):
        """Add aggregated cuts to master for single-cut initialization

        Args:
            initialize_set: Set of tuples, each tuple containing dual solution indices for all scenarios
            check_lp_cuts: If True, skip cuts already added to IP from LP phase
        """
        cut_count = 0

        for duals_tuple in initialize_set:
            cut_expr = gp.LinExpr()
            if check_lp_cuts:
                # Check if this cut was already added from LP phase
                # cuts_to_add_to_ip contains lists, so we need to check if any match
                skip_cut = False
                for existing_cut in self.cuts_to_add_to_ip:
                    if tuple(existing_cut) == duals_tuple:
                        skip_cut = True
                        break
                if skip_cut:
                    continue
            for scen_idx, scenario in enumerate(self.scenario):
                dual_id = duals_tuple[scen_idx]

                cut_expr += self.probab * (
                    gp.quicksum(
                        self.capacity[i]
                        * self.x[i]
                        * self.capacity_duals_array[dual_id, idx]
                        for idx, i in enumerate(self.Warehouse)
                    )
                    + gp.quicksum(
                        self.Demand[scenario, j] * self.demand_duals_array[dual_id, idx]
                        for idx, j in enumerate(self.Factory)
                    )
                )

            self.master.addConstr(cut_expr <= self.z)
            cut_count += 1

        return cut_count

    def select_best_dual_solutions(self, first_solution_list, dual_list, n):
        """Select the best dual solutions for each solution.

        Args:
            first_solution_list: List of first stage solutions
            dual_list: List of dual solution indices to consider
            n: Number of dual solution patterns to generate

        Returns:
            Set of tuples, each tuple containing dual solution indices for all scenarios
        """
        if not first_solution_list:
            return set()

        selected_patterns = set()

        for sol in first_solution_list:
            dualctx = np.multiply(self.capacity_np, sol)
            s1 = np.matmul(self.capacity_duals_array[dual_list], dualctx)
            s1 = np.squeeze(s1)

            # Get the best dual solutions for all scenarios
            _, indices = cflp_benders_utils_single_cut.find_largest_index_numba(
                s1, self.dual_obj_random[np.ix_(dual_list, self.scenario)]
            )

            # Convert indices to actual dual IDs and create tuple
            best_duals_tuple = tuple(dual_list[idx] for idx in indices)
            selected_patterns.add(best_duals_tuple)

            # Also add some variation by selecting second-best options
            for _ in range(min(n - 1, len(dual_list) - 1)):
                # Create variations by selecting different combinations
                s1_with_random = (
                    s1.reshape(-1, 1)
                    + self.dual_obj_random[np.ix_(dual_list, self.scenario)]
                )

                # For each scenario, sometimes pick second or third best
                variant_indices = []
                for scen_idx in range(len(self.scenario)):
                    scenario_values = s1_with_random[:, scen_idx]
                    sorted_indices = np.argsort(-scenario_values)
                    # Pick from top options with some randomness
                    pick_idx = min(
                        np.random.randint(0, min(3, len(sorted_indices))),
                        len(sorted_indices) - 1,
                    )
                    variant_indices.append(sorted_indices[pick_idx])

                variant_tuple = tuple(dual_list[idx] for idx in variant_indices)
                selected_patterns.add(variant_tuple)

                if len(selected_patterns) >= n:
                    break

        return selected_patterns

    def adaptive_cut_selection_for_lp(
        self,
        phase_one_sols,
        final_sols,
        dual_list,
    ):
        """
        lp master cuts addition given we just want to work with a few duals
        """

        selected_set = set()

        best_V = np.inf
        ctx_vec = np.array(phase_one_sols) @ self.setup_np

        # Find best solution from optimal solutions
        for count, sol in enumerate(phase_one_sols):
            sp_optimal, duals = self.evaluate_subproblems_with_dual_list(sol, dual_list)
            selected_set.add(tuple(duals))
            val = self.probab * np.sum(sp_optimal) + ctx_vec[count]

            if val < best_V:
                best_V = val
                best_x_vals = sol

        logger.info(f"best V from heuristic: {best_V}")

        for count, sol in enumerate(final_sols):
            if np.array_equal(best_x_vals, sol):
                best_x = count
                break

        V_sel = dict((k, 0) for k in range(len(final_sols)))
        del V_sel[best_x]  # deleting best_x to save computation
        ctx_vec = np.array(final_sols) @ self.setup_np
        dualx = (
            np.array(final_sols) * self.capacity_np_T
        ) @ self.capacity_duals_array.T

        dual_evaluated = []
        phase_two_iterations = 0

        while True:
            phase_two_iterations += 1
            if phase_two_iterations == 1:
                theta_vals = self._evaluate_theta_values(V_sel, dualx, selected_set)
            else:
                theta_new_cut = self._evaluate_theta_values(V_sel, dualx, [duals])
                theta_vals = {
                    solnum: np.maximum(theta_vals[solnum], theta_new_cut[solnum])
                    for solnum in V_sel.keys()
                }
            # theta_vals = self._evaluate_theta_values(V_sel, dualx, selected_set)

            V_sel = {
                solnum: (self.probab * theta_vals[solnum] + ctx_vec[solnum])
                for solnum in V_sel.keys()
            }
            V_sel = {k: v for k, v in V_sel.items() if v < best_V}

            if not V_sel:
                return selected_set, final_sols[best_x], best_V

            new_best = min(V_sel, key=V_sel.get)

            if new_best in dual_evaluated:
                return selected_set, final_sols[new_best], V_sel[new_best]
            else:
                vals_hat, duals = self.evaluate_subproblems_with_dual_list(
                    final_sols[new_best], dual_list
                )
                V_sel[new_best] = self.probab * np.sum(vals_hat) + ctx_vec[new_best]
                dual_evaluated.append(new_best)

            # Convert duals to tuple and add to set
            if isinstance(duals, dict):
                dual_tuple = tuple(duals[s] for s in self.scenario)
            elif isinstance(duals, (list, tuple)) and len(duals) == len(self.scenario):
                dual_tuple = tuple(duals)
            else:
                raise ValueError(f"Unexpected dual solution format: {type(duals)}")
            selected_set.add(dual_tuple)

            if min(V_sel.values()) > best_V:
                return selected_set, final_sols[best_x], best_V

    def ip_initialize(self, method):

        # cuts_to_add_to_ip now contains dual solution tuples from identify_active_cuts
        cuts_to_add_as_tuples = set()
        for cut_dual_ids in self.cuts_to_add_to_ip:
            if isinstance(cut_dual_ids, (list, tuple)):
                cuts_to_add_as_tuples.add(tuple(cut_dual_ids))

        self.lp_cons_to_ip.append(len(cuts_to_add_as_tuples))

        # Remove all constraints from LP master problem and add active cuts
        self.master.remove(self.master.getConstrs()[:])
        if cuts_to_add_as_tuples:
            self.add_cuts_to_master(cuts_to_add_as_tuples, check_lp_cuts=False)

        if method == "vanilla" or method == "tech_1" or method == "tech_1_boosted":
            best_x = self.ip_first_stage_optimal_sols[-1]
            self.master.setAttr("vType", self.x, GRB.BINARY)
            _, zva = self.optimal_value_duals(best_x, get_duals=False)
            # For single cut, sum all scenario values with probabilities
            temp_z_val = sum(self.probab * zva[s] for s in self.scenario)
            temp_x = {ware: best_x[count] for count, ware in enumerate(self.Warehouse)}
            self.master.setAttr("start", self.x, temp_x)
            self.z.start = temp_z_val

            if method == "tech_1":
                selected_dict = self.select_highest_binding_cuts(
                    self.ip_first_stage_optimal_sols[:2], self.primary_pool
                )
                n_cons = self.add_cuts_to_master(selected_dict, check_lp_cuts=True)
                self.tech_1_ip_cons.append(n_cons)

            elif method == "tech_1_boosted":
                n_cons = 0
                selza, _, _ = self.select_cuts_for_ip_initialization(
                    self.ip_first_stage_optimal_sols,
                    self.ip_first_stage_sols,
                    self.primary_pool,
                )

                for s in self.scenario:
                    for duals in set(selza[s]):
                        if duals in self.cuts_to_add_to_ip[s]:
                            continue
                        n_cons += 1

                target_num = math.ceil(n_cons / (self.nS))
                selected_dict = self.select_best_dual_solutions(
                    self.ip_first_stage_optimal_sols[:2], self.primary_pool, target_num
                )
                n_cons = self.add_cuts_to_master(selected_dict, check_lp_cuts=True)
                self.tech_1_ip_cons.append(n_cons)

        elif method == "tech_2":

            self.master.setAttr("vType", self.x, GRB.BINARY)
            selected_dict, best_x, best_V = self.select_cuts_for_ip_initialization(
                self.ip_first_stage_optimal_sols,
                self.ip_first_stage_sols,
                self.primary_pool,
            )

            self.dual_update()
            ctx = np.dot(self.setup_np, best_x)
            dualctx = np.multiply(self.capacity_np, best_x)
            s1 = np.matmul(self.capacity_duals_array, dualctx)
            V, _, best_z = self.compute_value_function_approximation(
                ctx, s1, include_scenario_details=True
            )
            assert math.isclose(V, best_V, rel_tol=1e-5)

            temp_x = {ware: best_x[count] for count, ware in enumerate(self.Warehouse)}
            # For single cut, sum all scenario values with probabilities
            temp_z_val = sum(self.probab * best_z[s] for s in self.scenario)
            self.master.setAttr("start", self.x, temp_x)
            self.z.start = temp_z_val
            tech_2_cons = self.add_cuts_to_master(selected_dict, check_lp_cuts=True)

            self.tech_2_ip_cons.append(tech_2_cons)
            if not self.lp_active_cuts:
                assert lp_cons >= self.lp_final_cons[-1]

    def dual_update(self):
        if self.skip_dual_collection:
            return

        self.capacity_duals_array = np.array(self.capacity_duals_list)
        self.demand_duals_array = np.array(self.demand_duals_list)
        self.dual_obj_random = np.matmul(
            self.demand_duals_array, self.Demand_array
        )  # dual * #scenarios Then we will pick best in every row.

        if not self.split_dual:
            self.primary_pool = list(range(len(self.capacity_duals_array)))

    def identify_active_cuts(self):
        """
        Identify active (binding) cuts for single-cut Benders.

        Evaluates the LHS of each stored aggregated cut at the final LP solution (x*)
        and marks it active if it equals the final value of z (within tolerance).

        Returns:
            list: List of dual-id lists (one dual id per scenario) for active cuts
        """
        if self.skip_dual_collection:
            return []

        if not hasattr(self, "cut_history") or not self.cut_history:
            return []

        capacity_weighted_solution = np.array(
            [
                self.capacity[ware] * self.first_stage_values[ware]
                for ware in self.Warehouse
            ]
        )

        # Evaluate all dual solutions at current first-stage solution
        capacity_dual_product = np.matmul(
            self.capacity_duals_array, capacity_weighted_solution
        )  # h^T * (C * x)
        capacity_dual_product = capacity_dual_product.reshape(-1, 1)

        # Compute subproblem objective values for all (dual, scenario) combinations
        subproblem_evaluations = capacity_dual_product + self.dual_obj_random
        assert subproblem_evaluations.shape == (len(self.capacity_duals_array), self.nS)

        # Use the final scalar value of z
        active_tol = 1e-5
        z_val = float(self.z.X)

        active_cuts = []

        # Evaluate each previously added aggregated cut
        for cut_dual_ids in self.cut_history:
            cut_lhs = self.probab * sum(
                subproblem_evaluations[dual_id, scenario_idx]
                for scenario_idx, dual_id in enumerate(cut_dual_ids)
            )

            # Check if cut is active (LHS approximately equals RHS)
            if abs(cut_lhs - z_val) <= active_tol:
                active_cuts.append(cut_dual_ids)

        return active_cuts

    def cache_dsp(self, dual_list, lazy=True):

        # Optimized: Use vectorized operations instead of list comprehension
        first_stage_array = np.array(
            [self.first_stage_values[ware] for ware in self.Warehouse]
        )
        dualctx = self.capacity_np * first_stage_array

        # Convert dual_list to numpy array for better performance
        dual_indices = np.array(dual_list, dtype=np.int64)

        # Use optimized numba kernel that avoids intermediate array creation
        sp_optimal, best_duals = (
            cflp_benders_utils_single_cut.evaluate_dual_subset_numba(
                dualctx, dual_indices, self.capacity_duals_array, self.dual_obj_random
            )
        )

        # Create single aggregated cut like in add_single_cut
        cut_expr = gp.LinExpr()
        zup_total = sum(sp_optimal) * self.probab

        if zup_total - self.second_stage_values > max(
            self.tol, 0.001 * abs(self.second_stage_values)
        ):
            for scenario in self.scenario:
                dual_id = best_duals[scenario]

                # Add this scenario's contribution to the aggregated cut
                cut_expr += self.probab * (
                    gp.quicksum(
                        self.capacity[i]
                        * self.x[i]
                        * self.capacity_duals_array[dual_id, idx]
                        for idx, i in enumerate(self.Warehouse)
                    )
                    + gp.quicksum(
                        self.Demand[scenario, j] * self.demand_duals_array[dual_id, idx]
                        for idx, j in enumerate(self.Factory)
                    )
                )

                # Update dual solution counter
                self.dual_soln_optimal_counter[dual_id] += 1

            # Check if we should add the aggregated cut
            if lazy:
                self.master.cbLazy(cut_expr <= self.z)
            else:
                self.master.addConstr(cut_expr <= self.z)

            # Track which dual solutions contributed to this cut
            self.cut_history.append(best_duals.copy())
            return 1

        return 0

    def find_best_solution_heuristically(self, sols_list, dual_list):

        best_x = -1
        phase_one_iterations = 0
        V_hat = dict((k, 0) for k in range(len(sols_list)))

        optimal_evaluated = []
        ctx_vec = np.array(sols_list) @ self.setup_np

        # Track duals that led to highest cuts for each primal solution
        primal_best_duals = {solnum: set() for solnum in range(len(sols_list))}

        finding_best = True

        while finding_best:
            phase_one_iterations += 1

            if len(self.capacity_duals_array) != len(self.capacity_duals_list):
                self.dual_update()

            if phase_one_iterations == 1:
                # Store both vals and duals for each primal solution
                vals_and_duals = {
                    solnum: self.evaluate_subproblems_with_dual_list(
                        sols_list[solnum], dual_list
                    )
                    for solnum in V_hat.keys()
                }
                vals = {solnum: vals_and_duals[solnum][0] for solnum in V_hat.keys()}

                # Initialize best duals for each primal solution
                for solnum in V_hat.keys():
                    # Convert duals dict to tuple and add to set
                    duals_dict = vals_and_duals[solnum][1]
                    dual_tuple = tuple(duals_dict[s] for s in self.scenario)
                    primal_best_duals[solnum].add(dual_tuple)

                vals = {solnum: np.maximum(vals[solnum], 0) for solnum in V_hat.keys()}
            else:
                vals_and_duals_ = {
                    solnum: self.evaluate_subproblems_with_dual_list(
                        sols_list[solnum], duals
                    )
                    for solnum in V_hat.keys()
                }
                vals_ = {solnum: vals_and_duals_[solnum][0] for solnum in V_hat.keys()}

                # Update vals and corresponding best duals when vals improve
                for solnum in V_hat.keys():
                    old_vals = (
                        vals[solnum] if solnum in vals else np.zeros(len(self.scenario))
                    )
                    new_vals = vals_[solnum]
                    improved_vals = np.maximum(old_vals, new_vals)

                    # Replace old dual solution with new one if any scenario improved
                    if np.any(new_vals > old_vals):
                        # Clear old dual solutions and add the new improved one
                        primal_best_duals[solnum].clear()
                        duals_dict = vals_and_duals_[solnum][1]
                        dual_tuple = tuple(duals_dict[s] for s in self.scenario)
                        primal_best_duals[solnum].add(dual_tuple)

                vals = {solnum: improved_vals for solnum in V_hat.keys()}

            V_hat = {
                solnum: (ctx_vec[solnum] + self.probab * np.sum(vals[solnum]))
                for solnum in V_hat.keys()
            }
            new_best = min(V_hat, key=V_hat.get)

            if new_best not in optimal_evaluated:
                start = time.perf_counter()
                optimal_val, duals, _ = self.optimal_value_duals(
                    sols_list[new_best], get_duals=True
                )
                dual_list.extend(duals)
                dual_list = list(dict.fromkeys(dual_list))

                # Replace with optimal solution
                primal_best_duals[new_best].clear()
                dual_tuple = tuple(duals[s] for s in self.scenario)
                primal_best_duals[new_best].add(dual_tuple)

                V_hat[new_best] = optimal_val
                optimal_evaluated.append(new_best)

                best_x = new_best  # (current optimal)
                new_best = min(V_hat, key=V_hat.get)

                if new_best == best_x:
                    finding_best = False
                else:
                    V_hat = {k: v for (k, v) in V_hat.items() if v <= optimal_val}
            else:
                finding_best = False

        # Convert sets of tuples to list of tuples for compatibility
        selected_list = []
        for dual_set in primal_best_duals.values():
            selected_list.extend(list(dual_set))

        return new_best, V_hat[new_best], dual_list, selected_list

    def select_cuts_for_ip_initialization(self, phase_one_sols, final_sols, dual_list):
        """
        Select which cuts to add to IP master problem initialization.

        Returns:
            tuple: (selected_set, best_solution, best_value)
            where selected_set is a set of tuples, each tuple containing dual solutions for all scenarios
        """
        best_x, best_V, selected_set = self._find_initial_best_solution(
            phase_one_sols, dual_list
        )

        # Add all LP cuts to selected set
        if hasattr(self, "cut_history") and len(self.cut_history) > 0:
            for duals in self.cuts_to_add_to_ip:
                dual_tuple = tuple(duals)
                selected_set.add(dual_tuple)

        self._ensure_dual_arrays_updated()

        best_x_in_final = self._map_best_solution_to_final_sols(
            phase_one_sols[best_x], final_sols
        )

        dual_list = list(dict.fromkeys(dual_list))

        selected_set, best_sol, best_val = self._iterative_cut_selection(
            final_sols,
            best_x_in_final,
            best_V,
            selected_set,
            dual_list,
            phase_one_sols,
        )

        return selected_set, best_sol, best_val

    def _find_initial_best_solution(self, phase_one_sols, dual_list):
        """Find the initial best solution and corresponding dual solutions."""
        if len(phase_one_sols) == 1:
            best_V, duals, _ = self.optimal_value_duals(
                phase_one_sols[0], get_duals=True
            )
            dual_list.extend(duals)
            dual_list = list(dict.fromkeys(dual_list))
            best_x = 0
            # Convert duals to tuple and add to set
            dual_tuple = tuple(duals[s] for s in self.scenario)
            selected_set = {dual_tuple}
        else:
            best_x, best_V, dual_list, selected_list = (
                self.find_best_solution_heuristically(phase_one_sols, dual_list)
            )
            # Convert selected_list to selected_set
            selected_set = set()
            for dual_sol in selected_list:
                if isinstance(dual_sol, dict):
                    dual_tuple = tuple(dual_sol[s] for s in self.scenario)
                elif isinstance(dual_sol, (list, tuple)) and len(dual_sol) == len(
                    self.scenario
                ):
                    dual_tuple = tuple(dual_sol)
                else:
                    raise ValueError(
                        f"Unexpected dual solution format: {type(dual_sol)}"
                    )
                selected_set.add(dual_tuple)

        logger.info(f"best V from heuristic: {best_V}")
        return best_x, best_V, selected_set

    def _ensure_dual_arrays_updated(self):
        """Ensure dual arrays are synchronized with dual lists."""
        if len(self.capacity_duals_array) != len(self.capacity_duals_list):
            self.dual_update()

    def _map_best_solution_to_final_sols(self, best_solution, final_sols):
        """Map the best solution from phase_one to final_sols index."""
        x_order = np.array(best_solution)
        for count, sol in enumerate(final_sols):
            if np.array_equal(x_order, sol):
                return count
        raise ValueError("Best solution not found in final solutions")

    def _iterative_cut_selection(
        self, final_sols, best_x, best_V, selected_set, dual_list, phase_one_sols
    ):
        """Iteratively select cuts to improve the bound."""
        ctx_vec = np.array(final_sols) @ self.setup_np
        dualx = (
            np.array(final_sols) * self.capacity_np_T
        ) @ self.capacity_duals_array.T

        V_sel = {k: 0 for k in range(len(final_sols)) if k != best_x}
        dual_evaluated = self._identify_phase_one_solutions_in_final(
            phase_one_sols, final_sols
        )
        optimal_evaluated = []

        phase_two_iterations = 0
        while True:
            phase_two_iterations += 1
            if phase_two_iterations == 1:
                theta_vals = self._evaluate_theta_values(V_sel, dualx, selected_set)
            else:
                theta_new_cut = self._evaluate_theta_values(V_sel, dualx, [duals])
                theta_vals = {
                    solnum: np.maximum(theta_vals[solnum], theta_new_cut[solnum])
                    for solnum in V_sel.keys()
                }

            V_sel = {
                solnum: (self.probab * theta_vals[solnum] + ctx_vec[solnum])
                for solnum in V_sel.keys()
            }
            V_sel = {k: v for k, v in V_sel.items() if v < best_V}

            if not V_sel:
                return selected_set, final_sols[best_x], best_V

            new_best = min(V_sel, key=V_sel.get)

            if new_best in optimal_evaluated:
                return selected_set, final_sols[new_best], V_sel[new_best]

            duals, updated_dualx = self._process_new_best_solution(
                new_best,
                final_sols,
                dual_list,
                dual_evaluated,
                optimal_evaluated,
                V_sel,
            )

            # Update dualx if new dual solutions were found
            if updated_dualx is not None:
                dualx = updated_dualx

            if duals is not None:
                # Convert duals to tuple and add to set

                if isinstance(duals, dict):
                    dual_tuple = tuple(duals[s] for s in self.scenario)
                elif isinstance(duals, (list, tuple)) and len(duals) == len(
                    self.scenario
                ):
                    dual_tuple = tuple(duals)
                else:
                    raise ValueError(f"Unexpected dual solution format: {type(duals)}")
                selected_set.add(dual_tuple)

                if V_sel[new_best] < best_V and new_best in optimal_evaluated:
                    best_V = V_sel[new_best]
                    best_x = new_best

            if min(V_sel.values()) > best_V:
                return selected_set, final_sols[best_x], best_V

    def _identify_phase_one_solutions_in_final(self, phase_one_sols, final_sols):
        """Identify which final solutions correspond to phase one solutions."""
        dual_evaluated = []
        for first_sol in phase_one_sols:
            for count, sol in enumerate(final_sols):
                if np.array_equal(first_sol, sol):
                    dual_evaluated.append(count)
        return dual_evaluated

    def _evaluate_theta_values(
        self,
        V_sel,
        dualx,
        selected_set,
    ):
        """Evaluate theta values for all candidate solutions."""
        # Ensure dual arrays are synchronized before evaluation
        self._ensure_dual_arrays_updated()
        theta_vals = {}
        for solnum in V_sel.keys():
            theta_val = self.evaluate_subproblems_fast_on_dual_list(
                dualx[solnum], selected_set
            )
            theta_vals[solnum] = theta_val
        return theta_vals

    def _process_new_best_solution(
        self, new_best, final_sols, dual_list, dual_evaluated, optimal_evaluated, V_sel
    ):
        """Process the new best solution and return corresponding dual solutions and updated dualx if needed."""
        # ctx_vec = np.array(final_sols) @ self.setup_np

        if new_best in dual_evaluated:
            V_sel[new_best], duals, _ = self.optimal_value_duals(
                final_sols[new_best], get_duals=True
            )
            optimal_evaluated.append(new_best)
            dual_list.extend(duals)
            dual_list = list(dict.fromkeys(dual_list))

            self._ensure_dual_arrays_updated()
            # Return updated dualx array after new dual solutions are found
            updated_dualx = (
                np.array(final_sols) * self.capacity_np_T
            ) @ self.capacity_duals_array.T
            return duals, updated_dualx
        else:
            ctx_vec = np.dot(final_sols[new_best], self.setup_np)
            vals_hat, duals = self.evaluate_subproblems_with_dual_list(
                final_sols[new_best], dual_list
            )
            V_sel[new_best] = self.probab * np.sum(vals_hat) + ctx_vec
            dual_evaluated.append(new_best)
            return duals, None

    def select_highest_binding_cuts(self, first_solution_list, dual_list):
        """
        Select the highest binding cuts for each solution and scenario.

        Args:
            first_solution_list: List of first stage solutions
            dual_list: List of dual solution indices to consider

        Returns:
            Set of tuples, each tuple containing dual solution indices for all scenarios
        """
        selected_cuts = set()

        for sol in first_solution_list:
            dualctx = np.multiply(self.capacity_np, sol)
            s1 = np.matmul(self.capacity_duals_array, dualctx)
            s1 = np.squeeze(s1)

            _, indices = cflp_benders_utils_single_cut.find_largest_index_numba(
                s1[dual_list], self.dual_obj_random[np.ix_(dual_list, self.scenario)]
            )

            duals_tuple = tuple(dual_list[idx] for idx in indices)
            selected_cuts.add(duals_tuple)

        return selected_cuts

    def solve_saa_iteration(self, saa_number):

        start = time.perf_counter()

        self.master = self.build_master(relaxation=True)
        if saa_number == 1:
            self.master.setParam("TimeLimit", 36000)  # 10 hours for first SAA
        else:
            self.master.setParam("TimeLimit", 3600)  # 1 hour for subsequent SAAs
        self.master.setParam("OutputFlag", False)  # Suppress LP output
        self.master.setParam("LogToConsole", 0)

        self.benders(saa_number)

        del self.master
        stop = time.perf_counter()
        self.total_times.append(round(stop - start, 2))

        logger.info(f"SAA {saa_number} done!")


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


def get_combination(details):
    combinations = {
        (0, 0, 0): "NoReuse",
        (0, 1, 0): "DSP",
        (0, 2, 0): "CuratedDSP",
        (0, 2, 1): "StaticInit",
        (0, 2, 2): "AdaptiveInit",
        (1, 0, 0): "NoReuse",
        (1, 1, 0): "DSP",
        (1, 2, 0): "CuratedDSP",
        (1, 2, 1): "StaticInit",
        (1, 2, 2): "AdaptiveInit",
        (1, 2, 3): "BoostedStaticInit",
    }
    # combinations = {
    #     (0, 0, 0): "LP_NoDSP_NoInit",
    #     (0, 1, 0): "LP_DSP_NoInit",
    #     (0, 2, 0): "LP_CuratedDSP_NoInit",
    #     (0, 2, 1): "LP_CuratedDSP_StaticInit",
    #     (0, 2, 2): "LP_CuratedDSP_AdaptiveInit",
    #     (1, 0, 0): "IP_NoDSP_NoInit",
    #     (1, 1, 0): "IP_DSP_NoInit",
    #     (1, 2, 0): "IP_CuratedDSP_NoInit",
    #     (1, 2, 1): "IP_CuratedDSP_StaticInit",
    #     (1, 2, 2): "IP_CuratedDSP_AdaptiveInit",
    #     (1, 2, 3): "IP_CuratedDSP_BoostedStaticInit",
    # }
    if details not in combinations:
        raise AlgorithmDetailsError("Error: Input is not correct.")
    return combinations[details]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Give m, n, S")
    parser.add_argument("algorithm_details", metavar="N", type=int, nargs=3)
    parser.add_argument("data", metavar="N", type=str, nargs=5)
    args = parser.parse_args()
    test = args.data
    data = "_".join(str(i) for i in args.data)
    algo = "_".join(str(i) for i in args.algorithm_details)

    test_fl = [float(i) for i in test]
    print(test_fl)
    print()
    solve_ip, dual_lookup, init = args.algorithm_details

    data = "_".join(args.data)
    algo = "_".join(str(i) for i in args.algorithm_details)

    match = get_combination((solve_ip, dual_lookup, init))
    # Add "single_" prefix to algorithm name and filenames
    single_match = f"single_{match}"
    if solve_ip:
        fname = f"detailed-results/cflp/IP/single_{data}_{match}.op"
    else:
        fname = f"detailed-results/cflp/LP/single_{data}_{match}.op"
    print("Algo: Single-Cut", match)

    logger = setup_logger(fname)
    logger.info(f"data: {args.data}")
    logger.info(f"Algorithm combination: {match}")

    test = args.data
    np.random.seed(3)
    bend = Benders(args.algorithm_details, test_fl[:4])

    n_saa = int(test[-1])
    n_scen = int(test[-2])

    no_reuse_numbers = [1, 2, int(n_saa / 2 + 1), n_saa]
    saa_solved = []
    # bend.generate_normal_demand_scenarios(0.1, n_scen)  # , save_filename=filename)
    for saa_number in range(1, n_saa + 1):
        # if n_scen < 100:
        #     filename = (
        #         "instances-cflp/scenarios/"
        #         + "_".join(args.data[:3])
        #         + f"_200_{saa_number}"
        #     )
        # else:
        #     filename = (
        #         "instances-cflp/scenarios/"
        #         + "_".join(args.data[:3])
        #         + f"_{n_scen}_{saa_number}"
        #     )
        bend.generate_normal_demand_scenarios(0.1, n_scen)  # , save_filename=filename)
        # bend.read_scenarios(filename)
        if saa_number in no_reuse_numbers or dual_lookup != 0:
            bend.solve_saa_iteration(saa_number)
            saa_solved.append(saa_number)

    data_dict = {"Instance": data, "Scenarios": n_scen, "Method": single_match}

    if solve_ip:
        columns = [
            "Total times",
            "IP time",
            "LP relaxation time",
            "IP nodes",
            "Root node time",
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
        ]

    for col in columns:
        col_data = getattr(bend, col.replace(" ", "_").lower())
        data_dict[f"{col} SAA 0"] = round(col_data[0], 3)
        # data_dict[f"{col} first"] = round(col_data[1], 3)
        # data_dict[f"{col} last"] = round(col_data[-1], 3)
        data_dict[f"{col} average"] = round(np.mean(col_data[1:]), 3)

    data_dict["avg LP heuristic time"] = (
        f"{np.mean(bend.lp_initialization_time[1:]):.2f}"
    )
    data_dict["final num of dual solution"] = f"{bend.dual_pool_size[-1]}"

    df = pd.DataFrame([data_dict])

    df.to_csv(
        "results_single_cflp.csv",
        index=False,
        quoting=3,
        sep=",",
        escapechar=",",
        mode="a",
        header=not os.path.exists("results_single_cflp.csv"),
    )

    logger.info("")

    saa_list = [f"SAA {i}" for i in saa_solved]
    logger.info("******")
    logger.info("")

    data = [
        ["Tech 1 LP cons", *bend.tech_1_lp_cons],
        ["Tech 2 LP cons ", *bend.tech_2_lp_cons],
        ["dual lookup cuts - LP", *bend.dual_lookup_lp_cuts],
        ["subproblem cuts - LP", *bend.subproblem_lp_cuts],
        ["dual lookup count - LP", *bend.dual_lookup_lp_counts],
        ["sp count - LP", *bend.subproblem_counts_lp],
        ["Final cons in LP", *bend.lp_final_cons],
    ]
    logger.info(tabulate(data, headers=["LP constraint information", *saa_list]))
    logger.info("")

    if solve_ip:
        data = [
            ["Actual LP cuts added", *bend.lp_cons_to_ip],
            ["Tech 1 IP cons ", *bend.tech_1_ip_cons],
            ["Tech 2 IP cons ", *bend.tech_2_ip_cons],
            ["Total initial cons IP", *bend.initial_constraint_counter],
            ["dual lookup cuts - IP", *bend.dual_lookup_ip_cuts],
            ["subproblem cuts - IP", *bend.subproblem_ip_cuts],
        ]
        logger.info(tabulate(data, headers=["IP constraint information", *saa_list]))
        logger.info("")

    # ["Time information ******"]
    data = [
        ["LP time total", *bend.lp_relaxation_time],
        ["IP time total", *bend.ip_time],
        ["Total time taken per SAA", *bend.total_times],
    ]

    logger.info(tabulate(data, headers=["Time information", *saa_list]))
    logger.info("")
    data = [
        ["Master time - LP", *bend.master_lp_times],
        ["dual lookup time - LP", *bend.dual_lookup_lp_times],
        ["subproblem time - LP", *bend.subproblem_lp_times],
        ["First stage solutions LP", *bend.x_feas_counter_lp],
        ["LP init time", *bend.lp_initialization_time],
    ]
    logger.info(tabulate(data, headers=["LP Time", *saa_list]))
    logger.info("")

    if solve_ip:
        data = [
            ["dual lookup time - IP", *bend.dual_lookup_ip_times],
            ["dual lookup count - IP", *bend.dual_lookup_ip_counts],
            ["subproblem time - IP", *bend.subproblem_ip_times],
            ["Total time taken per SAA", *bend.total_times],
            ["IP init time", *bend.ip_initialization_time],
        ]
        logger.info(tabulate(data, headers=["IP Time", *saa_list]))
        logger.info("")

        data = [
            ["First stage solutions IP ", *bend.x_feas_counter_ip],
            ["ip gap %", *bend.ip_gap],
            ["ip nodes ", *bend.ip_nodes],
        ]
        logger.info(tabulate(data, headers=["More IP information", *saa_list]))
        logger.info("")

    data = [
        ["LP Iterations data", bend.lp_iterations],
        ["# dual solutions in primary pool", bend.primary_pool_size],
        ["# Total dual solutions after SAA ", bend.dual_pool_size_final],
        ["# Total dual solutions ", bend.dual_pool_size],
    ]

    logger.info(tabulate(data, headers=[]))
    logger.info("")
