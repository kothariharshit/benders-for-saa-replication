import numpy as np
import math
import gurobipy as gp
from gurobipy import GRB
from numba import njit, prange
import numba as nb

"""
We are trying to multiply the objective by #S.
This might help in better conditioning.
"""


class CMNDinst:

    def __init__(self, problemfile, nscen):
        self.nN = None  # Nodes
        self.nA = None  # arcs
        self.nK = None  # Commodities
        self.nS = int(nscen)  # Scenarios

        self.reading_problemdata(problemfile)

        self.gaplimit = 1e-4
        self.last_sol = None
        self.hash_set = set()
        self.hash_list = []
        self.int_hash_set = set()
        self.PI_ = []
        self.H_ = []

    def reading_problemdata(self, problemfile):
        """
        Reads and initializes the problem data from a specified file.
        This function loads various attributes like nodes, arcs,
        commodities, costs, and capacities from the file.

        Parameters:
        - problemfile (str): The path to the file containing the problem data.

        Returns:
        - int: Returns 0 if the function executes successfully.
        """

        f = open(problemfile)
        line = f.readline()
        line = f.readline()
        self.nN, self.nA, self.nK = list(map(int, line.strip().split()))
        self.demand_data = {}
        self.mu = {}
        self.flowcost = {}
        self.capacity = {}
        self.fixedcost = {}
        self.commodities = []
        self.nodes = []

        for i in range(self.nK):
            comm = "".join(("K", str(i)))
            self.commodities = self.commodities + [comm]

        for i in range(1, self.nN + 1):
            node = "".join(("N", str(i)))
            self.nodes = self.nodes + [node]

        self.scale_o = None

        fixedcostsum = 0
        for _ in range(self.nA):
            line = f.readline()
            tail, head, flowcost, capacity, fixedcost, _, _ = map(
                int, line.strip().split()
            )
            tail = "".join(("N", str(tail)))  # denoting node
            head = "".join(("N", str(head)))  # denoting node
            fixedcostsum += fixedcost

            if self.scale_o == None:
                self.scale_o = 10 ** int(math.log10(fixedcost) + 1)
            self.flowcost[(tail, head)] = flowcost / self.scale_o
            self.capacity[(tail, head)] = capacity
            self.fixedcost[(tail, head)] = (self.nS * fixedcost) / self.scale_o

        self.arcs = list(self.flowcost.keys())

        self.fixedcost_np = np.array([self.fixedcost[arc] for arc in self.arcs])
        self.capacity_np = np.array([self.capacity[arc] for arc in self.arcs])
        self.capacity_np_T = self.capacity_np.reshape(1, -1)

        for k in self.commodities:
            line = f.readline()
            tail, head, mu = list(map(int, line.strip().split()))
            tail = "".join(("N", str(tail)))  # denoting node
            head = "".join(("N", str(head)))  # denoting node
            self.demand_data[k] = tail, head
            self.mu[k] = mu

        self.unmetcost = fixedcostsum / (self.nA * self.scale_o)
        f.close()

        self.arcs = gp.tuplelist(self.arcs)
        self.commodities = gp.tuplelist(self.commodities)
        self.nodes = gp.tuplelist(self.nodes)

        return 0

    def write_scenarios_to_file(self, nscen, filename):

        demandav = np.array(list(self.mu.values()))
        self.demand = np.round(
            np.random.normal(
                loc=demandav, scale=self.nDev * demandav, size=(nscen, self.nK)
            ),
            decimals=2,
        )
        self.demand = np.maximum(self.demand, 0)
        np.savetxt(filename, self.demand, fmt="%.2f", comments="")

    def read_scenarios(self, nscen, filename):
        """
        Reads demand data from a file and generates a specified number of scenarios
        for commodity transportation demands.

        Parameters:
        - nscen (int): Number of scenarios to read.
        - filename: File containing the scenarios
        """

        self.scenario = list(range(nscen))
        self.demand = np.loadtxt(filename)
        self.demand = self.demand[:nscen]
        assert len(self.demand) == nscen

        self.repeated_demand = np.repeat(self.demand, repeats=2, axis=1)
        multiplier = [+1.0 if i % 2 == 0 else -1.0 for i in range(2 * self.nK)]
        self.repeated_demand = self.repeated_demand * np.array(multiplier)

        assert (nscen, self.nK) == self.demand.shape

        self.scen_sp_data = {scenario: {} for scenario in self.scenario}

        for scenario in self.scenario:
            for count, k in enumerate(self.commodities):
                origin, destination = self.demand_data[k]
                self.scen_sp_data[scenario][k, origin] = self.demand[scenario][count]
                self.scen_sp_data[scenario][k, destination] = -self.demand[scenario][
                    count
                ]

    def build_SP(self):
        """
        Here we build the general subproblem model.
        When we need to add look for cuts, this model
        is updated using the functions update_x and update_scen,
        to provide it with the first stage solution value and the
        scenario data.
        """

        SP = gp.Model("Sub-problem")
        x = SP.addVars(self.commodities, self.arcs, vtype=GRB.CONTINUOUS, name="x")
        alpha = SP.addVars(self.commodities, vtype=GRB.CONTINUOUS, name="alpha")
        obj = self.unmetcost * alpha.sum() + gp.quicksum(
            x.sum("*", i, j) * self.flowcost[(i, j)] for i, j in self.arcs
        )
        self.h = SP.addConstrs(
            (x.sum("*", i, j) <= 0.0 for i, j in self.arcs), name="cap"
        )
        self.pi = {}
        for k in self.commodities:
            origin, destination = self.demand_data[k]
            for i in self.nodes:
                if i in self.demand_data[k]:
                    c = 1 if i == origin else -1 if i == destination else 0
                    self.pi[k, i] = SP.addConstr(
                        x.sum(k, i, "*") - x.sum(k, "*", i) + c * alpha[k] == 0.0
                    )
                else:
                    SP.addConstr(x.sum(k, i, "*") - x.sum(k, "*", i) == 0.0)

        SP.setObjective(obj, GRB.MINIMIZE)
        return SP

    def update_x(self, model, xvals):

        temp = {arc: self.capacity[arc] * xvals[arc] for arc in self.arcs}
        model.setAttr("RHS", self.h, temp)

    def update_x_diff(self, xvals):

        for count, arc in enumerate(self.arcs):
            self.h[arc].rhs = max(self.capacity[arc] * xvals[count], 0)

    def update_scen(self, model, scenario):
        model.setAttr("RHS", self.pi, self.scen_sp_data[scenario])

    def build_master(self, relaxation=False):
        """
        Here we build the master problem
        for benders decomposition
        """
        Master = gp.Model("Master")
        if relaxation:
            self.x = Master.addVars(self.arcs, ub=1.0, obj=self.fixedcost, name="y")
        else:
            self.x = Master.addVars(
                self.arcs, vtype=GRB.BINARY, obj=self.fixedcost, name="y"
            )

        self.sorted_vars = list(map(self.x.get, self.arcs))
        self.z = Master.addVars(self.scenario, obj=1.0, name="z")

        Master.modelSense = GRB.MINIMIZE

        return Master

    def addcut(self, model, lazy=False, first=False):
        """
        Adds a cut to the master problem by iterating over all scenarios and checking if the
        subproblem's objective value exceeds a threshold. If it does, it constructs and adds a cut
        to the master problem either as a lazy constraint (during the solve) or as a regular constraint.

        Parameters:
        - model : The master optimization model to which cuts are added.
        - lazy (bool): Flag to indicate whether the cut should be added as a lazy constraint.
        - first (bool): Flag used to handle the first addition differently if required.

        Returns:
        - int: Number of cuts added to the model.
        - list: List of subproblem objective values across all scenarios.

        """
        cutadded = 0
        sp_vals = []
        self.update_x(self.subproblem, self.xvals)

        for scenario in self.scenario:
            self.update_scen(self.subproblem, scenario)
            self.subproblem.setParam("OutputFlag", False)
            self.subproblem.optimize()
            status = self.subproblem.status

            if status != 2:
                raise Exception("Subproblem status - {}".format(status))

            sobj = self.subproblem.ObjVal
            sp_vals.append(sobj)

            if sobj - self.zvals[scenario] > self.tol:

                hi = self.subproblem.getAttr("pi", self.h)
                pii = self.subproblem.getAttr("pi", self.pi)
                hi, pii = self.feas(hi, pii)

                s = scenario
                s1_vec = [
                    self.capacity[(i, j)] * self.xvals[i, j] for (i, j) in self.arcs
                ]
                s2_vec = [self.demand[s][k] for k in range(self.nK)]
                s1 = sum(
                    self.capacity[(i, j)] * self.xvals[i, j] * hi[(i, j)]
                    for (i, j) in self.arcs
                )
                s2 = sum(
                    self.demand[s][count]
                    * (pii[k, self.demand_data[k][0]] - pii[k, self.demand_data[k][1]])
                    for count, k in enumerate(self.commodities)
                )

                constant = [
                    1,
                    self.zvals[scenario],
                ]  # 1 is for the z[scenario] and zvals is for the constant of the cut.
                normal = np.linalg.norm(
                    s1_vec + s2_vec + s2_vec + constant
                )  # , ord=np.inf))

                dualsol = hi | pii
                sol_hash = hash(frozenset(dualsol.items()))

                not_in_list = False

                if sol_hash not in self.hash_list:
                    pii_order = [
                        pii[k, i] for k in self.commodities for i in self.demand_data[k]
                    ]
                    hi_order = [hi[a] for a in self.arcs]
                    self.H_.append(hi_order)
                    self.PI_.append(pii_order)
                    self.hash_list.append(sol_hash)
                    not_in_list = True
                    self.dual_soln_optimal_counter[len(self.hash_list) - 1] = 0
                else:
                    if not first:
                        dual_id = self.hash_list.index(sol_hash)
                        if dual_id < self.dual_pool_size[-1]:
                            self.dual_soln_optimal_counter[
                                self.hash_list.index(sol_hash)
                            ] += 1

                if s1 + s2 - self.zvals[scenario] > self.tol * max(normal, 1.0):
                    cutadded += 1
                    if lazy:
                        model.cbLazy(
                            (
                                gp.quicksum(
                                    self.capacity[(i, j)] * self.x[i, j] * hi[(i, j)]
                                    for (i, j) in self.arcs
                                )
                                + gp.quicksum(
                                    self.demand[s][count]
                                    * (
                                        pii[k, self.demand_data[k][0]]
                                        - pii[k, self.demand_data[k][1]]
                                    )
                                    for count, k in enumerate(self.commodities)
                                )
                                <= self.z[scenario]
                            )
                        )  # , name)
                    else:
                        model.addConstr(
                            (
                                gp.quicksum(
                                    self.capacity[(i, j)] * self.x[i, j] * hi[(i, j)]
                                    for (i, j) in self.arcs
                                )
                                + gp.quicksum(
                                    self.demand[s][count]
                                    * (
                                        pii[k, self.demand_data[k][0]]
                                        - pii[k, self.demand_data[k][1]]
                                    )
                                    for count, k in enumerate(self.commodities)
                                )
                                <= self.z[scenario]
                            )
                        )  # , name)
                    if not_in_list:
                        if not lazy:
                            self.lp_cuts[scenario].add(len(self.H_) - 1)
                        # index of solution! hence, -1
                    else:
                        if not lazy:
                            self.lp_cuts[scenario].add(self.hash_list.index(sol_hash))

        return cutadded, sp_vals

    def upperbound(self):
        """
        Calculates the upper bound of the solution given
        the first stage solution and then scenario subproblem values.

        Returns:
        - float: objective value of the solution
        """

        ctx = sum(self.fixedcost[(i, j)] * self.xvals[(i, j)] for (i, j) in self.arcs)
        scenario_ctx = sum(self.zup[s] for s in self.scenario)
        upperbound = ctx + scenario_ctx
        return upperbound

    def feas(self, hi, pii):
        """
        Ensures that the dual solution obtained from solving the subproblem is
        feasible to the dual. This is needed to ensure the numerical stability
        of the Benders decomposition algorithm.
        """
        for k in self.commodities:
            origin, destination = self.demand_data[k]
            pii[k, origin] = min(pii[k, destination] + self.unmetcost, pii[k, origin])
            i = origin
            j = destination
            if (i, j) in self.arcs:
                hi[(i, j)] = min(
                    hi[(i, j)],
                    self.flowcost[(i, j)] - pii[k, origin] + pii[k, destination],
                )

        for i, j in self.arcs:
            hi[(i, j)] = min(hi[(i, j)], 0.0)

        return hi, pii

    def solupdate(self, problem="IP"):
        """
        Given a solution, this function extracts the values of x and z from the solution
        and stores them in the dictionary.
        """

        self.xvals = self.master.getAttr("x", self.x)
        if problem == "IP":
            for k, v in self.xvals.items():
                if v > 0.5:
                    self.xvals[k] = 1.0
                else:
                    self.xvals[k] = 0.0
        else:
            self.xvals = {k: max(v, 0.0) for k, v in self.xvals.items()}
        self.zvals = self.master.getAttr("x", self.z)

    def optimal_value_duals(self, xvals, get_duals=False):
        """
        So this function takes as input the first stage solution
        It evalautes V(x) and outputs it
        It also gives the optimal subproblem objectives (zvals) as output
        """

        objective = np.dot(self.fixedcost_np, xvals)
        self.update_x_diff(xvals)

        if get_duals:
            duals = []
        sp_vals = []

        for scenario in self.scenario:
            self.update_scen(self.subproblem, scenario)
            self.subproblem.setParam("OutputFlag", False)
            self.subproblem.optimize()
            status = self.subproblem.status

            if status != 2:
                raise Exception("Subproblem status - {}".format(status))

            sobj = self.subproblem.ObjVal
            objective += sobj
            sp_vals.append(sobj)
            hi = self.subproblem.getAttr("pi", self.h)
            pii = self.subproblem.getAttr("pi", self.pi)
            hi, pii = self.feas(hi, pii)

            dualsol = hi | pii
            sol_hash = hash(frozenset(dualsol.items()))

            if sol_hash not in self.hash_list:
                pii_order = [
                    pii[k, i] for k in self.commodities for i in self.demand_data[k]
                ]
                hi_order = [hi[a] for a in self.arcs]
                self.H_.append(hi_order)
                self.PI_.append(pii_order)
                self.hash_list.append(sol_hash)
                self.dual_soln_optimal_counter[len(self.hash_list) - 1] = 0
                if get_duals:
                    duals.append(len(self.hash_list) - 1)
            elif get_duals:
                self.dual_soln_optimal_counter[self.hash_list.index(sol_hash)] += 1
                duals.append(self.hash_list.index(sol_hash))

        if get_duals:
            return objective, np.array(duals), sp_vals
        else:
            return objective, sp_vals

    def sp_vals_sel_dual_copy(self, xvals, dual_list):
        """
        Note that dual list in this function can't be scenario dependent.
        """
        s1 = (xvals * self.capacity_np_T) @ self.H[dual_list].T
        s1 = np.squeeze(s1)

        sp_optimal, indices = find_largest_index_numba(
            s1, self.dual_obj_random[dual_list, :]
        )
        duals = [dual_list[id] for id in indices]

        return sp_optimal, duals

    def sp_vals_sel(self, xvals, selected_dict, return_duals=False):
        """
        This function takes as input the first stage solution and the
        selected list of dual solutions given by the dictionary.
        It returns the evaluations over that selected list of
        solutions.
        No broadcasting is used in this code.
        """
        assert type(selected_dict) is dict
        s1 = (xvals * self.capacity_np_T) @ self.H.T
        s1 = np.squeeze(s1)
        max_indices = []
        ub = []

        for scen in self.scenario:
            subproblem_evaluations = (
                s1[list(selected_dict[scen])]
                + self.dual_obj_random[list(selected_dict[scen]), scen]
            )
            max_index = np.argmax(subproblem_evaluations)
            id = list(selected_dict[scen])[max_index]
            max_indices.append(id)
            ub.append(subproblem_evaluations[max_index])
        if return_duals:
            return ub, max_indices
        else:
            return ub

    def sp_vals_evaluate(self, s1, duals, scenario_subset):
        """
        This function just evaluates the solution on the new duals.
        It is very fast because it just sums the values and no max
        is involved.
        duals is one dual solution for every scenario.
        Also, here I don't do it for all scenarios.
        I just do this calclulation for a few scenarios.
        Because we may not add cuts for all scenarios.
        """
        ub = []
        for scen in self.scenario:
            if scen in scenario_subset:
                ub.append(s1[duals[scen]] + self.dual_obj_random[duals[scen], scen])
            else:
                ub.append(0.0)
        return ub

    def value_func_hat_nb(self, ctx, s1, warm=False):
        """
        So this function takes as input the first stage solution and the list of dual solutions
        to evaluate Q_sel. As output, it gives the V_sel and also the dual solutions which it used
        to obtain this bound for every scenario.
        """
        s1 = np.squeeze(s1)
        sp_optimal, duals = find_largest_index_numba(s1, self.dual_obj_random)
        upperbound = ctx + np.sum(sp_optimal)

        if warm:
            return upperbound, duals, sp_optimal
        else:
            return upperbound, duals


# @njit(parallel=True)#(nb.float64[:], nb.float64[:])
# @njit(nb.types.Tuple((nb.float64[:], nb.int64[:]))(nb.float64[:], nb.float64[:,:]), parallel=True, fastmath=True, nogil=True)
@njit(
    nb.types.Tuple((nb.float64[:], nb.int64[:]))(nb.float64[:], nb.float64[:, :]),
    parallel=True,
    fastmath=True,
)
def find_largest_index_numba(list1, list2):
    """
    Returns the index of the largest element in the sum of list1 and list2 using numba.
    """
    m, n = list2.shape

    indices = np.empty(n, dtype=np.int64)
    max_elements = np.empty(n, dtype=np.float64)

    for scen in prange(n):
        max_sum = np.finfo(np.float64).min
        max_index = -1
        for dual in range(len(list1)):
            sum_value = list1[dual] + list2[dual, scen]
            if sum_value >= max_sum:
                max_sum = sum_value
                max_index = dual
        indices[scen] = max_index
        max_elements[scen] = max_sum

    return max_elements, indices
