import argparse
import numpy as np
import time
import math
import cmnd_benders_utils
from gurobipy import GRB
import gurobipy as gp
import pandas as pd
import logging
import os
from tabulate import tabulate

"""
This file does benders decomposition for the network
design instances.
"""


class Benders(cmnd_benders_utils.CMNDinst):

    def __init__(self, algorithm_details, problemfile, n_scen):
        """Intializes and creates the instance data"""

        super().__init__(problemfile, n_scen)
        solve_ip, dual_lookup, init = algorithm_details

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

        self.lp_active_len = []

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
        self.lp_initialization_time = [0]
        self.ip_initialization_time = [0]
        self.timelimit = None
        self.lp_timeout = False
        self.ip_timeout = False
        self.final_upper_bound = None
        self.final_lower_bound = None
        self.final_gap = None

    def benders(self, saa_number):

        first = 1 if saa_number == 1 else 0

        self.cuts_to_add_to_ip = {s: set() for s in self.scenario}
        self.lp_cuts = {s: set() for s in self.scenario}

        if not first:
            if self.split_dual:
                self.primary_pool = []

                add_previous_dsp_cuts = True
                add_active = True
                add_previous_saa = True

                if add_previous_dsp_cuts:
                    self.primary_pool = [
                        k for k, v in self.dual_soln_optimal_counter.items() if v > 0
                    ]

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
                self.primary_pool = list(range(len(self.H)))

            self.primary_pool_size.append(len(self.primary_pool))
            self.dual_obj_random = np.matmul(self.PI, self.repeated_demand.T)
            init_time = time.perf_counter()
            self.lp_initialize(self.init_method)
            self.lp_initialization_time.append(
                round(time.perf_counter() - init_time, 3)
            )

        problem_start = time.perf_counter()
        self.solve_lp_relaxation(first, problem_start)
        self.lp_relaxation_time.append(round(time.perf_counter() - problem_start, 3))
        self.x_feas_counter_lp.append(len(self.lp_first_stage_sols))

        if not self.solve_ip or self.lp_timeout:
            return 0

        self.xvals = {}
        self.zvals = {}

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
                self.xvals = model.cbGetSolution(self.x)
                self.zvals = model.cbGetSolution(self.z)

                x_order = tuple(int(self.xvals[arc]) for arc in self.arcs)
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
                    cutadded, sp_objs = self.addcut(model, lazy=True, first=first)
                    zvals = {s: sp_objs[s] for s in self.scenario}
                    model.cbSetSolution(self.x, self.xvals)
                    model.cbSetSolution(self.z, zvals)
                    self.sp_ip_time += time.perf_counter() - start
                    self.sp_ip_cut += cutadded
                    self.sp_ip_count += 1

        self.master.setParam("OutputFlag", True)
        self.master.setParam("MIPGap", 1e-3)
        self.master.Params.lazyConstraints = 1

        # Set IP time limit (remaining time after LP phase)
        if self.timelimit is not None:
            lp_time = time.perf_counter() - problem_start
            remaining_time = max(1, self.timelimit - lp_time)  # At least 1 second
            self.master.setParam("TimeLimit", remaining_time)
            logger.info(f"Setting IP time limit to {remaining_time:.2f} seconds")
            print(f"Setting IP time limit to {remaining_time:.2f} seconds")

        self.master._runtime = 0
        self.master._rootgap = 1000
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
        self.root_node_time.append(self.master._runtime)
        self.root_node_gap.append(self.master._rootgap)

        self.solupdate()
        self.dual_update()

        self.ip_time.append(round(self.master.Runtime, 3))
        logger.info(f"IP bound: {self.master.ObjBound:.3f}")

        print(f"IP bound: {self.master.ObjBound:.3f}")
        print()
        self.initial_constraint_counter.append(self.master.NumConstrs)
        self.dual_pool_size.append(len(self.H))
        self.dual_pool_size_final.append(len(self.H))

        self.ip_gap.append(round(self.master.MIPGap * 100, 4))
        self.ip_nodes.append(self.master.NodeCount)

        x_order = tuple(self.xvals[arc] for arc in self.arcs)

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

    def split_dual_list(self):
        subset_keys = [k for k, v in self.dual_soln_optimal_counter.items() if v > 0]
        return subset_keys

    def lp_initialize(self, method):

        if method == "tech_1" or method == "tech_1_boosted":
            selected_dict = self.technique_one(
                self.lp_optimal_first_stage_sols[:2], self.primary_pool
            )

        elif method == "tech_2":
            selected_dict, best_x, best_V = self.tech_two_lp_duals(
                self.lp_optimal_first_stage_sols,
                self.lp_first_stage_sols,
                self.primary_pool,
            )

            ctx = np.dot(self.fixedcost_np, best_x)
            dualctx = np.multiply(self.capacity_np, best_x)
            s1 = np.matmul(self.H, dualctx)
            V, _, best_z = self.value_func_hat_nb(ctx, s1, warm=True)

            temp_y = {arc: best_x[count] for count, arc in enumerate(self.arcs)}
            temp_z = {s: best_z[s] for s in self.scenario}
            self.zup = {s: best_z[s] for s in self.scenario}
            self.master.setAttr("start", self.x, temp_y)
            self.master.setAttr("start", self.z, temp_z)
        else:
            assert method == "vanilla"
            return 0

        for scenario in self.scenario:
            self.lp_cuts[scenario].update(selected_dict[scenario])

        n_cons = self.add_cuts_to_master(selected_dict, check_lp_cuts=False)

        if method == "tech_1":
            self.tech_1_lp_cons.append(n_cons)
        elif method == "tech_2":
            self.tech_2_lp_cons.append(n_cons)

    def solve_lp_relaxation(self, first, problem_start_time=None):

        sp_lp_time = 0
        dual_lookup_lp_time = 0

        sp_lp_cut = 0
        dual_lookup_lp_cut = 0

        master_lp_time = 0
        iterations_lp = 0

        upperbound = np.inf

        sp_count_lp = 0
        dual_lookup_lp_count = 0

        start = time.perf_counter()
        self.master.optimize()

        master_lp_time += time.perf_counter() - start
        iterations_lp += 1

        status = self.master.status
        if status != 2:
            raise Exception(f"Master problem status - {status}")

        self.subproblem = self.build_SP()
        self.solupdate(problem="LP")

        if self.init_method != "tech_2" or first:
            self.zup = {s: math.inf for s in self.scenario}
            upperbound = np.inf

        lowerbound = self.master.ObjBound
        dif = upperbound - lowerbound
        cutadded = 1

        while True:
            # Check for timeout
            if self.timelimit is not None and problem_start_time is not None:
                elapsed_time = time.perf_counter() - problem_start_time
                if elapsed_time >= self.timelimit:
                    self.lp_timeout = True
                    # Calculate final gap when timeout occurs
                    if lowerbound > 0:
                        final_gap = (upperbound - lowerbound) / lowerbound * 100
                    else:
                        final_gap = float("inf")

                    # Store final bounds and gap for CSV output
                    self.final_upper_bound = upperbound
                    self.final_lower_bound = lowerbound
                    self.final_gap = final_gap

                    logger.info(
                        f"LP phase timed out after {elapsed_time:.2f} seconds. Final gap: {final_gap:.2f}%"
                    )
                    print(
                        f"LP phase timed out after {elapsed_time:.2f} seconds. Final gap: {final_gap:.2f}%"
                    )
                    break

            cutadded = 0

            if not first and self.dual_lookup_lp:
                start = time.perf_counter()
                cutadded = self.cache_dsp(self.primary_pool, lazy=False)
                dual_lookup_lp_cut += cutadded
                dual_lookup_lp_time += time.perf_counter() - start
                dual_lookup_lp_count += 1

            if cutadded == 0:
                start = time.perf_counter()

                cuts, self.zup = self.addcut(self.master, first=first)
                ub = self.upperbound()
                upperbound = min(ub, upperbound)

                cutadded = cuts
                sp_lp_time += time.perf_counter() - start
                sp_lp_cut += cuts
                sp_count_lp += 1

            if lowerbound == 0:
                lowerbound = 0.1

            if cutadded == 0 or (dif < self.gaplimit):
                ub = self.upperbound()
                upperbound = min(ub, upperbound)

                # Store final bounds and gap for CSV output (normal completion)
                if self.timelimit is not None:
                    if lowerbound > 0:
                        final_gap = (upperbound - lowerbound) / lowerbound * 100
                    else:
                        final_gap = float("inf")
                    self.final_upper_bound = upperbound
                    self.final_lower_bound = lowerbound
                    self.final_gap = final_gap

                logger.info(f"LP UB: {upperbound:.3f}, LB: {lowerbound:.3f}")
                print(f"LP UB: {upperbound:.3f}, LB: {lowerbound:.3f}")
                break

            dif = (upperbound - lowerbound) * 100 / lowerbound
            #  print(f"LP UB: {upperbound} , LB: {lowerbound}")
            start = time.perf_counter()
            self.master.optimize()
            master_lp_time += time.perf_counter() - start

            status = self.master.status
            if status != 2:
                raise Exception(f"Master problem status - {status}")

            lowerbound = self.master.ObjBound

            self.solupdate(problem="LP")
            x_order = np.array([self.xvals[arc] for arc in self.arcs])
            self.lp_first_stage_sols.append(x_order)
            iterations_lp += 1

        self.lp_final_cons.append(self.master.NumConstrs)

        lp_cuts_num = 0
        for s in self.scenario:
            lp_cuts_num += len(self.lp_cuts[s])

        assert self.master.NumConstrs <= lp_cuts_num

        self.dual_update()
        self.lp_active = self.active_checker()

        self.cuts_to_add_to_ip = {k: set(v) for k, v in self.lp_active.items()}
        self.lp_iterations.append(iterations_lp)

        x_order = np.array([self.xvals[arc] for arc in self.arcs])
        self.lp_optimal_first_stage_sols.append(x_order)
        self.dual_pool_size.append(len(self.H))
        if not self.solve_ip:
            self.dual_pool_size_final.append(len(self.H))

        self.subproblem_lp_times.append(round(sp_lp_time, 2))
        self.dual_lookup_lp_times.append(round(dual_lookup_lp_time, 2))
        self.cut_time_lp.append(round(sp_lp_time + dual_lookup_lp_time, 2))
        self.subproblem_lp_cuts.append(sp_lp_cut)
        self.dual_lookup_lp_cuts.append(dual_lookup_lp_cut)
        self.master_lp_times.append(round(master_lp_time, 2))
        self.subproblem_counts_lp.append(sp_count_lp)
        self.dual_lookup_lp_counts.append(dual_lookup_lp_count)

    def add_cuts_to_master(self, initialize_dict, check_lp_cuts=True):
        """check_lp_cuts checks if the dual soln was already in the lp"""

        LHS = {}
        RHS = {}
        n_cons = 0
        for s in self.scenario:
            for duals in set(initialize_dict[s]):
                if check_lp_cuts:
                    if duals in self.cuts_to_add_to_ip[s]:
                        continue
                coeffs = np.multiply(
                    self.capacity_np, self.H[duals, :]
                )  # can be sped up ig
                cons = np.dot(self.PI[duals], self.repeated_demand[s].T)
                LHS[n_cons] = gp.LinExpr(cons)
                LHS[n_cons].addTerms(coeffs, self.sorted_vars)
                RHS[n_cons] = gp.LinExpr(self.z[s])
                n_cons += 1

        self.master.addConstrs(LHS[c] <= RHS[c] for c in range(n_cons))

        return n_cons

    def bestnewsols(self, first_solution_list, dual_list, n):
        """This function tells us which dual solutions are within some
        tolerance of the optimal solution. Considering the scenario data
        for the new SAA problem.
        n = number of solutions we want to add per scenario."""

        if not first_solution_list:
            return [[] for _ in range(self.nS)]

        for count, sol in enumerate(first_solution_list):

            dualctx = np.multiply(self.capacity_np, sol)
            s1 = np.repeat(np.matmul(self.H[dual_list], dualctx), self.nS)
            s1 = s1.reshape(len(dual_list), self.nS)

            final = (
                s1 + self.dual_obj_random[np.ix_(dual_list, self.scenario)]
            )  # , dec)  # dual * scenario

            finalT = final.T  # scenario * dual
            if count:
                idx = np.argsort(-finalT, axis=1)[:, :n]
                soli = np.vectorize(lambda x: dual_list[x])(idx)
                sol_idx = np.hstack((sol_idx, soli))
            else:
                sol_idx = np.argsort(-finalT, axis=1)[:, :n]
                sol_idx = np.vectorize(lambda x: dual_list[x])(sol_idx)

        final_idx = []

        for row in sol_idx:
            new_shape = (len(first_solution_list), n)
            reshaped_row = np.reshape(row, new_shape)
            flattened = reshaped_row.flatten(order="F")
            final_idx.append(list(dict.fromkeys(flattened))[:n])

        return final_idx

    def tech_two_lp_duals(
        self, phase_one_sols, final_sols, dual_list, one_optimal=False, all=False
    ):
        """
        lp master cuts addition given we just want to work with a few duals
        all -> add highest binding cuts for all phase one sols?
        one_optimal -> solve only one best to optimality?
        """

        selected_dict = {s: set() for s in self.scenario}

        best_V = np.inf
        ctx_vec = np.array(phase_one_sols) @ self.fixedcost_np

        # Find best solution from optimal solutions
        for count, sol in enumerate(phase_one_sols):
            sp_optimal, duals = self.sp_vals_sel_dual_copy(sol, dual_list)
            val = np.sum(sp_optimal) + ctx_vec[count]

            if val < best_V:
                best_V = val
                best_x_vals = sol
                best_lp_active = duals

            if all:
                for s in self.scenario:
                    selected_dict[s].add(duals[s])

        logger.info(f"best V from heuristic: {best_V}")

        for s in self.scenario:
            selected_dict[s].add(best_lp_active[s])

        for count, sol in enumerate(final_sols):
            if np.array_equal(best_x_vals, sol):
                best_x = count
                break

        V_sel = dict((k, 0) for k in range(len(final_sols)))
        del V_sel[best_x]  # deleting best_x to save computation
        ctx_vec = np.array(final_sols) @ self.fixedcost_np
        dualx = (np.array(final_sols) * self.capacity_np_T) @ self.H.T

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
                    solnum: self.sp_vals_sel(
                        final_sols[solnum], selected_dict, return_duals=False
                    )
                    for solnum in V_sel.keys()
                }
                vals = {solnum: np.maximum(vals[solnum], 0) for solnum in V_sel.keys()}
            else:
                v = {
                    solnum: self.sp_vals_evaluate(
                        dualx[solnum], duals, add_cuts_for_scenarios
                    )
                    for solnum in V_sel.keys()
                }
                vals = {
                    solnum: np.maximum(vals[solnum], v[solnum])
                    for solnum in V_sel.keys()
                }

            V_sel = {
                solnum: (np.sum(vals[solnum]) + ctx_vec[solnum])
                for solnum in V_sel.keys()
            }
            V_sel = {k: v for (k, v) in V_sel.items() if v < best_V}

            if not V_sel:
                return selected_dict, final_sols[best_x], best_V

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
                vals_hat, duals = self.sp_vals_sel_dual_copy(
                    final_sols[new_best], dual_list
                )
                V_sel[new_best] = np.sum(vals_hat) + ctx_vec[new_best]

                dual_evaluated.append(new_best)

            vals_diff = np.round(vals_hat - vals_sel, 10)
            assert np.min(vals_diff) >= 0

            d = dict(enumerate(vals_diff, 0))
            d_order = sorted(d.items(), key=lambda t: t[1], reverse=True)

            threshold = best_V - min_V_sel

            add_cuts_for_scenarios = []
            sum = 0

            for item in d_order:
                scen, val = item
                sum += val
                add_cuts_for_scenarios.append(scen)
                if sum > threshold:
                    break

            for scen in add_cuts_for_scenarios:
                selected_dict[scen].add(duals[scen])

            if min(V_sel.values()) > best_V:
                return selected_dict, final_sols[best_x], best_V

    def ip_initialize(self, method):

        self.master.remove(self.master.getConstrs()[:])
        lp_cons = self.add_cuts_to_master(self.cuts_to_add_to_ip, check_lp_cuts=False)
        self.lp_cons_to_ip.append(lp_cons)  # These are being carried from the LP

        if method == "vanilla" or method == "tech_1" or method == "tech_1_boosted":
            best_x = self.ip_first_stage_optimal_sols[-1]
            self.master.setAttr("vType", self.x, GRB.BINARY)
            _, zva = self.optimal_value_duals(best_x, get_duals=False)
            temp_z = {s: zva[s] for s in self.scenario}
            temp_x = {arc: best_x[count] for count, arc in enumerate(self.arcs)}
            self.master.setAttr("start", self.x, temp_x)
            self.master.setAttr("start", self.z, temp_z)

            if method == "tech_1":
                selected_dict = self.technique_one(
                    self.ip_first_stage_optimal_sols[:2], self.primary_pool
                )
                n_cons = self.add_cuts_to_master(selected_dict, check_lp_cuts=True)
                self.tech_1_ip_cons.append(n_cons)

            elif method == "tech_1_boosted":
                n_cons = 0
                selza, _, _ = self.phase_two(
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
                selected_dict = self.bestnewsols(
                    self.ip_first_stage_optimal_sols[:2], self.primary_pool, target_num
                )
                n_cons = self.add_cuts_to_master(selected_dict, check_lp_cuts=True)
                self.tech_1_ip_cons.append(n_cons)

        elif method == "tech_2":

            self.master.setAttr("vType", self.x, GRB.BINARY)
            selected_dict, best_x, best_V = self.phase_two(
                self.ip_first_stage_optimal_sols,
                self.ip_first_stage_sols,
                self.primary_pool,
            )

            self.dual_update()
            ctx = np.dot(self.fixedcost_np, best_x)
            dualctx = np.multiply(self.capacity_np, best_x)
            s1 = np.matmul(self.H, dualctx)
            V, _, best_z = self.value_func_hat_nb(ctx, s1, warm=True)
            assert math.isclose(V, best_V, rel_tol=1e-5)

            temp_x = {arc: best_x[count] for count, arc in enumerate(self.arcs)}
            temp_z = {s: best_z[s] for s in self.scenario}
            self.master.setAttr("start", self.x, temp_x)
            self.master.setAttr("start", self.z, temp_z)
            tech_2_cons = self.add_cuts_to_master(selected_dict, check_lp_cuts=True)

            self.tech_2_ip_cons.append(tech_2_cons)
            if not self.lp_active_cuts:
                assert lp_cons >= self.lp_final_cons[-1]

    def dual_update(self):
        self.H = np.array(self.H_)
        self.PI = np.array(self.PI_)
        self.dual_obj_random = np.matmul(
            self.PI, self.repeated_demand.T
        )  # dual * #scenarios Then we will pick best in every row.

        if not self.split_dual:
            self.primary_pool = list(range(len(self.H)))

    def active_checker(self):
        "This function gives us the dual solutions which generate active cuts from previous iteration."

        dualctx = np.array(
            [self.capacity[(i, j)] * self.xvals[(i, j)] for (i, j) in self.arcs]
        )
        s1 = np.matmul(self.H, dualctx)
        s1 = s1.reshape(-1, 1)

        subproblem_evaluations = s1 + self.dual_obj_random
        assert subproblem_evaluations.shape == (len(self.H), self.nS)

        z_np_array = np.array([self.zvals[scenario] for scenario in self.scenario])
        final = s1 + self.dual_obj_random - z_np_array
        result = {}
        active_tol = -1e-10
        for i in range(final.shape[1]):
            indices = np.where(final[:, i] > active_tol)[0]
            active_cuts = set.intersection(set(indices), self.lp_cuts[i])
            result[i] = active_cuts

        return result

    def cache_dsp(self, dual_list, lazy=True):

        xvals = np.array([self.xvals[(i, j)] for (i, j) in self.arcs])
        s1 = (xvals * self.capacity_np_T) @ self.H.T
        s1 = np.squeeze(s1)
        z_np_array = np.array([self.zvals[scenario] for scenario in self.scenario])

        sp_optimal, indices = cmnd_benders_utils.find_largest_index_numba(
            s1[dual_list], self.dual_obj_random[np.ix_(dual_list, self.scenario)]
        )
        duals = [dual_list[id] for id in indices]
        dual_violations = sp_optimal - z_np_array
        cutadded = self.dsp_cuts(
            dual_violations, duals, self.master, self.scenario, lazy
        )

        return cutadded

    def phase_one(self, sols_list, dual_list):

        best_x = -1
        phase_one_iterations = 0
        V_hat = dict((k, 0) for k in range(len(sols_list)))

        optimal_evaluated = []
        ctx_vec = np.array(sols_list) @ self.fixedcost_np

        finding_best = True

        while finding_best:
            phase_one_iterations += 1

            if len(self.H) != len(self.H_):
                self.dual_update()

            if phase_one_iterations == 1:
                vals = {
                    solnum: self.sp_vals_sel_dual_copy(sols_list[solnum], dual_list)[0]
                    for solnum in V_hat.keys()
                }
                vals = {solnum: np.maximum(vals[solnum], 0) for solnum in V_hat.keys()}
            else:
                vals_ = {
                    solnum: self.sp_vals_sel_dual_copy(sols_list[solnum], duals)[0]
                    for solnum in V_hat.keys()
                }
                vals = {
                    solnum: np.maximum(vals[solnum], vals_[solnum])
                    for solnum in V_hat.keys()
                }

            V_hat = {
                solnum: (ctx_vec[solnum] + np.sum(vals[solnum]))
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

        return new_best, V_hat[new_best], dual_list

    def phase_two(self, phase_one_sols, final_sols, dual_list):
        """
        In this we decide which cuts to add to ip master from the start.
        This should also include cuts added to the lp relaxation.
        And we initiate this list with cuts active at the best_x
        """
        if len(phase_one_sols) == 1:
            best_V, duals, _ = self.optimal_value_duals(
                phase_one_sols[0], get_duals=True
            )
            dual_list.extend(duals)
            dual_list = list(dict.fromkeys(dual_list))
            best_x = 0
        else:
            best_x, best_V, dual_list = self.phase_one(phase_one_sols, dual_list)

        if len(self.H) != len(self.H_):
            self.dual_update()

        ctx_vec = np.array(phase_one_sols) @ self.fixedcost_np
        dualx = (np.array(phase_one_sols) * self.capacity_np_T) @ self.H.T

        selected_dict = {s: set() for s in self.scenario}

        for count, sol in enumerate(phase_one_sols):
            _, ip_active = self.sp_vals_sel_dual_copy(sol, dual_list)
            for s in self.scenario:
                selected_dict[s].add(ip_active[s])

        logger.info(f"best V from heuristic: {best_V}")

        x_order = np.array(phase_one_sols[best_x])
        for count, sol in enumerate(final_sols):
            if np.array_equal(x_order, sol):
                best_x = count
                break

        for scen in self.scenario:
            selected_dict[scen].update(self.cuts_to_add_to_ip[scen])

        for scen in self.scenario:
            dual_list.extend(self.cuts_to_add_to_ip[scen])
        dual_list = list(dict.fromkeys(dual_list))
        # This is necessary because cuts need to be part of dual list otherwise
        # assertion issue.

        V_sel = dict((k, 0) for k in range(len(final_sols)))
        del V_sel[best_x]  # deleting best_x to save computation

        ctx_vec = np.array(final_sols) @ self.fixedcost_np
        dualx = (np.array(final_sols) * self.capacity_np_T) @ self.H.T

        phase_two_iterations = 0
        dual_evaluated = []
        optimal_evaluated = []

        for first_sol in phase_one_sols:
            for count, sol in enumerate(final_sols):
                if np.array_equal(first_sol, sol):
                    dual_evaluated.append(count)

        while True:
            phase_two_iterations += 1
            if phase_two_iterations == 1:
                vals = {
                    solnum: self.sp_vals_sel(
                        final_sols[solnum], selected_dict, return_duals=False
                    )
                    for solnum in V_sel.keys()
                }
                vals = {solnum: np.maximum(vals[solnum], 0) for solnum in V_sel.keys()}
            else:
                v = {
                    solnum: self.sp_vals_evaluate(
                        dualx[solnum], duals, add_cuts_for_scenarios
                    )
                    for solnum in V_sel.keys()
                }
                vals = {
                    solnum: np.maximum(vals[solnum], v[solnum])
                    for solnum in V_sel.keys()
                }

            V_sel = {
                solnum: (np.sum(vals[solnum]) + ctx_vec[solnum])
                for solnum in V_sel.keys()
            }
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
                return selected_dict, final_sols[new_best], V_sel[new_best]

            elif new_best in dual_evaluated:
                sp_solve_start = time.perf_counter()
                V_sel[new_best], duals, vals_hat = self.optimal_value_duals(
                    final_sols[new_best], get_duals=True
                )
                dual_list.extend(duals)
                dual_list = list(dict.fromkeys(dual_list))
                optimal_evaluated.append(new_best)

                if len(self.H) != len(self.H_):
                    self.dual_update()
                    dualx = (np.array(final_sols) * self.capacity_np_T) @ self.H.T

                if V_sel[new_best] < best_V:
                    best_V = V_sel[new_best]
                    best_x = new_best
                    best_changed = True
            else:
                vals_hat, duals = self.sp_vals_sel_dual_copy(
                    final_sols[new_best], dual_list
                )
                V_sel[new_best] = np.sum(vals_hat) + ctx_vec[new_best]
                dual_evaluated.append(new_best)

            vals_diff = np.round(vals_hat - vals_sel, 10)
            if (new_best not in optimal_evaluated) or (not best_changed):
                assert np.min(vals_diff) >= 0

                d = dict(enumerate(vals_diff, 0))
                d_order = sorted(d.items(), key=lambda t: t[1], reverse=True)

                threshold = best_V - min_V_sel

                add_cuts_for_scenarios = []
                sum_scen_obj = 0

                for item in d_order:
                    scen, val = item
                    sum_scen_obj += val
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

    def dsp_cuts(self, dual_violations, best_duals, model, scenario_list, lazy=False):
        """Index set is the set of dual solutions you are considering in this iteration"""

        s1_vec = [self.capacity[(i, j)] * self.xvals[i, j] for (i, j) in self.arcs]

        num_cuts = 0
        LHS = {}
        RHS = {}
        hi = 0

        for scenario in scenario_list:
            s2_vec = [self.demand[scenario][k] for k in range(self.nK)]
            normal = np.linalg.norm(s1_vec + s2_vec + s2_vec)  # , ord=np.inf))
            if dual_violations[scenario] > self.tol * max(normal, 1.0):

                "Turned the exception off for ips as it is possible that the same cut is added\
                twice in branch and cut as it can be used to cutoff that solution. "
                coeffs = np.multiply(
                    self.capacity_np, self.H[best_duals[scenario], :]
                )  # can be sped up ig
                cons = np.dot(
                    self.PI[best_duals[scenario]], self.repeated_demand[scenario].T
                )
                LHS[num_cuts] = gp.LinExpr(cons)
                LHS[num_cuts].addTerms(coeffs, self.sorted_vars)
                RHS[num_cuts] = gp.LinExpr(self.z[scenario])

                if lazy:
                    model.cbLazy(LHS[num_cuts] <= RHS[num_cuts])

                num_cuts += 1
                if not lazy:
                    self.lp_cuts[scenario].add(best_duals[scenario])

                self.dual_soln_optimal_counter[best_duals[scenario]] += 1

        if not lazy:
            model.addConstrs(LHS[c] <= RHS[c] for c in range(num_cuts))

        return num_cuts

    def technique_one(self, first_solution_list, dual_list):
        """
        In technique one, we just add the highest cuts
        for every solution and every scenario.
        So, number of cuts will be cardinality of
        first_solution_list
        """

        final_idx = []

        for sol in first_solution_list:
            s1 = (sol * self.capacity_np_T) @ self.H.T
            s1 = np.squeeze(s1)
            _, indices = cmnd_benders_utils.find_largest_index_numba(
                s1[dual_list], self.dual_obj_random[np.ix_(dual_list, self.scenario)]
            )
            duals = [dual_list[id] for id in indices]
            final_idx.append(duals)

        final_idx = [list(row) for row in zip(*final_idx)]

        return final_idx

    def MIPsolve(self, saa_number):

        start = time.perf_counter()

        self.master = self.build_master(relaxation=True)
        self.master.setParam("TimeLimit", 3600)
        self.master.setParam("OutputFlag", False)
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
    if details not in combinations:
        raise AlgorithmDetailsError("Error: Input is not correct.")
    return combinations[details]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Give datafile, nS, std-dev")
    parser.add_argument("algorithm_details", metavar="N", type=int, nargs=3)
    parser.add_argument("data", metavar="N", type=str, nargs=3)
    parser.add_argument(
        "--timelimit", type=float, default=None, help="Time limit in seconds"
    )
    parser.add_argument(
        "--twice", action="store_true", help="Run for 2 times the timelimit"
    )
    args = parser.parse_args()

    solve_ip, dual_lookup, init = args.algorithm_details

    data = "_".join(args.data)
    algo = "_".join(str(i) for i in args.algorithm_details)

    match = get_combination((solve_ip, dual_lookup, init))

    data = data.replace("instances-cmnd/", "")
    scenario_filename = args.data[0]
    scenario_filename = scenario_filename.replace("instances-cmnd/", "")
    scenario_filename = scenario_filename.replace(".dow", "")

    # Add "multi_" prefix to distinguish from single-cut version
    multi_match = f"multi_{match}"
    if solve_ip:
        fname = f"detailed-results/cmnd/IP/multi_{data}_{match}.op"
    else:
        fname = f"detailed-results/cmnd/LP/multi_{data}_{match}.op"
    print()
    print("Algo: Multi-Cut", match)

    logger = setup_logger(fname)
    logger.info(f"data: {args.data}")
    logger.info(f"Algorithm combination: {match}")

    test = args.data
    np.random.seed(3)
    bend = Benders(args.algorithm_details, *test[:2])

    # Apply twice modifier if specified
    effective_timelimit = args.timelimit
    if args.twice and args.timelimit is not None:
        effective_timelimit = args.timelimit * 2
        logger.info(
            f"Using --twice flag: effective timelimit = {effective_timelimit:.2f} seconds"
        )
        print(
            f"Using --twice flag: effective timelimit = {effective_timelimit:.2f} seconds"
        )

    bend.timelimit = effective_timelimit

    n_saa = int(test[-1])
    n_scen = int(test[1])

    # When timelimit is specified, only run 1 SAA iteration
    if effective_timelimit is not None:
        logger.info("Time limit specified: running only 1 SAA iteration")
        print("Time limit specified: running only 1 SAA iteration")
        saa_iterations_to_run = [1]
    else:
        no_reuse_numbers = [1, 2, int(n_saa / 2 + 1), n_saa]
        saa_iterations_to_run = []
        for saa_number in range(1, n_saa + 1):
            if saa_number in no_reuse_numbers or dual_lookup != 0:
                saa_iterations_to_run.append(saa_number)

    saa_solved = []
    for saa_number in saa_iterations_to_run:
        if n_scen < 100:
            filename = (
                "instances-cmnd/scenarios/" + scenario_filename + f"_200_{saa_number}"
            )
        else:
            filename = (
                "instances-cmnd/scenarios/"
                + scenario_filename
                + f"_{n_scen}_{saa_number}"
            )
        # filename = 'instances-cmnd/scenarios/' + scenario_filename + f'_{n_scen}_{saa_number}'
        bend.read_scenarios(n_scen, filename)
        bend.MIPsolve(saa_number)
        saa_solved.append(saa_number)

    data_dict = {"Instance": data, "Scenarios": n_scen, "Method": multi_match}

    # When timelimit is specified, use simplified CSV output
    if effective_timelimit is not None:
        data_dict["Timelimit"] = effective_timelimit
        data_dict["Upper_Bound"] = (
            bend.final_upper_bound if bend.final_upper_bound is not None else "N/A"
        )
        data_dict["Lower_Bound"] = (
            bend.final_lower_bound if bend.final_lower_bound is not None else "N/A"
        )
        data_dict["Gap"] = bend.final_gap if bend.final_gap is not None else "N/A"
        data_dict["LP_Timeout"] = bend.lp_timeout
        data_dict["IP_Timeout"] = bend.ip_timeout
    else:
        # Original detailed CSV output for non-timelimit experiments
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
        "results_multi_cmnd.csv",
        index=False,
        quoting=3,
        sep=",",
        escapechar=",",
        mode="a",
        header=not os.path.exists("results_multi_cmnd.csv"),
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
