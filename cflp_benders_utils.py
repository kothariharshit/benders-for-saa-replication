import numpy as np
from scipy.spatial.distance import cdist
import gurobipy as gp
from gurobipy import GRB
from numba import njit, prange
import numba as nb
import json

""" 
This creates random instances or this can read an instance from a file.
It returns the extensive form of the model.
"""


class CFLPinst:

    def __init__(self, instance):
        self.Nfv = None
        self.m = None
        self.Nn = None
        # self.instance = instance
        # self.reading_capdata(instance)
        # self.scenario = []
        self.setup = {}
        self.capacity = {}
        self.cost = {}
        self.Pair = []
        self.Factory = []
        self.Warehouse = []
        self.mu = []
        self.recourse_cost = 0
        self.lostsalecost = {}

        self.gaplimit = 1e-5
        self.last_sol = None
        self.hash_set = set()
        self.hash_list = []
        self.int_hash_set = set()
        self.PI_ = []
        self.H_ = []

    def createNewInst(self, w, f, seed):

        rng = np.random.RandomState()
        rng.seed(int(seed))

        self.nW = int(w)
        self.nF = int(f)
        print(self.nW)
        print(self.nF)
        print(seed)

        # W is n_facilities
        # f is customers

        inst = {}

        inst["c_x"] = rng.rand(self.nF)
        inst["c_y"] = rng.rand(self.nF)

        inst["f_x"] = rng.rand(self.nW)
        inst["f_y"] = rng.rand(self.nW)

        inst["demands"] = rng.randint(5, 35 + 1, size=self.nF)
        inst["capacities"] = rng.randint(10, 160 + 1, size=self.nW)
        inst["fixed_costs"] = rng.randint(100, 110 + 1, size=self.nW) * np.sqrt(
            inst["capacities"]
        ) + rng.randint(90 + 1, size=self.nW)
        inst["fixed_costs"] = inst["fixed_costs"].astype(int)
        inst["ratio"] = 2.0
        inst["total_demand"] = inst["demands"].sum()
        inst["total_capacity"] = inst["capacities"].sum()

        # adjust capacities according to ratio
        inst["capacities"] = (
            inst["capacities"]
            * inst["ratio"]
            * inst["total_demand"]
            / inst["total_capacity"]
        )
        inst["capacities"] = inst["capacities"].astype(int)
        inst["total_capacity"] = inst["capacities"].sum()

        # transportation costs
        inst["trans_costs"] = (
            np.sqrt(
                (inst["c_x"].reshape((-1, 1)) - inst["f_x"].reshape((1, -1))) ** 2
                + (inst["c_y"].reshape((-1, 1)) - inst["f_y"].reshape((1, -1))) ** 2
            )
            * 10
            * inst["demands"].reshape((-1, 1))
        )
        inst["trans_costs"] = inst["trans_costs"].transpose()
        inst["trans_costs"] = inst["trans_costs"] / np.average(inst["demands"])
        inst["recourse_cost"] = 2 * np.max(
            [
                np.max(inst["fixed_costs"]),
                np.max(inst["trans_costs"]) / np.average(inst["demands"]),
            ]
        )

        # print('remove this bs')
        # self.demand_av = np.average(inst['demands'])

        self.setup = {}
        self.capacity = {}
        self.Pair = gp.tuplelist()
        self.cost = gp.tupledict()
        self.Factory = gp.tuplelist()
        self.Warehouse = gp.tuplelist()
        self.mu = []
        self.lostsalecost = {}

        for i in range(1, self.nW + 1):
            ware = "".join(("W", str(i)))
            self.Warehouse = self.Warehouse + [ware]
            self.setup[ware] = inst["fixed_costs"][i - 1]
            self.capacity[ware] = inst["capacities"][i - 1]

        print("***** Fixed capacity can be changed *****")
        self.recourse_cost = 2 * np.max(
            [
                np.max(inst["fixed_costs"]),
                np.max(inst["trans_costs"]) / np.average(inst["demands"]),
            ]
        )

        for j in range(1, self.nF + 1):
            fac = "".join(("F", str(j)))
            self.Factory = self.Factory + [fac]

            for i in range(1, self.nW + 1):
                ware = "".join(("W", str(i)))
                self.Pair = self.Pair + [(ware, fac)]
                self.cost[(ware, fac)] = inst["trans_costs"][i - 1][j - 1]

            self.lostsalecost[fac] = self.recourse_cost

        self.setup_np = np.array([self.setup[ware] for ware in self.Warehouse])
        self.capacity_np = np.array([self.capacity[ware] for ware in self.Warehouse])
        self.capacity_np_T = self.capacity_np.reshape(1, -1)

        self.mu = inst["demands"]
        return 0

    def save_instance_to_file(self, w, f, seed, filename):

        rng = np.random.RandomState(int(seed))
        self.nW = int(w)
        self.nF = int(f)

        # Generate instance data
        inst = {
            "c_x": rng.rand(self.nF),
            "c_y": rng.rand(self.nF),
            "f_x": rng.rand(self.nW),
            "f_y": rng.rand(self.nW),
            "demands": rng.randint(5, 36, size=self.nF),
            "capacities": rng.randint(10, 161, size=self.nW),
        }
        inst["fixed_costs"] = rng.randint(100, 110 + 1, size=self.nW) * np.sqrt(
            inst["capacities"]
        ) + rng.randint(90 + 1, size=self.nW)
        inst["fixed_costs"] = inst["fixed_costs"].astype(int)

        # Calculate total demand and capacity
        inst["total_demand"] = inst["demands"].sum()
        inst["total_capacity"] = inst["capacities"].sum()

        # Adjust capacities according to total demand and total capacity
        inst["capacities"] = (
            inst["capacities"] * 2.0 * inst["total_demand"] / inst["total_capacity"]
        ).astype(int)
        inst["total_capacity"] = inst["capacities"].sum()

        # Compute transportation costs
        inst["trans_costs"] = (
            np.sqrt(
                (inst["c_x"].reshape((-1, 1)) - inst["f_x"].reshape((1, -1))) ** 2
                + (inst["c_y"].reshape((-1, 1)) - inst["f_y"].reshape((1, -1))) ** 2
            )
            * 10
            * inst["demands"].reshape((-1, 1))
        )

        inst["trans_costs"] = inst["trans_costs"].T / np.average(inst["demands"])
        inst["recourse_cost"] = 2 * np.max(
            [
                np.max(inst["fixed_costs"]),
                np.max(inst["trans_costs"]) / np.average(inst["demands"]),
            ]
        )

        data_to_save = {
            "trans_costs": inst["trans_costs"].tolist(),
            "recourse_cost": inst["recourse_cost"],
            "fixed_costs": inst["fixed_costs"].tolist(),
            "demands": inst["demands"].tolist(),
            "capacities": inst["capacities"].tolist(),
        }

        with open(filename, "w") as file:
            json.dump(data_to_save, file)

    def load_instance(self, filename):

        name = filename.replace("instances-cflp/", "")
        data = name.split("_")
        self.nW = int(data[0])
        self.nF = int(data[1])

        with open(filename, "r") as file:
            inst = json.load(file)

        # Convert lists back to numpy arrays where appropriate

        for i in range(1, self.nW + 1):
            ware = "".join(("W", str(i)))
            self.Warehouse = self.Warehouse + [ware]
            self.setup[ware] = inst["fixed_costs"][i - 1]
            self.capacity[ware] = inst["capacities"][i - 1]
        self.recourse_cost = inst["recourse_cost"]

        # Setup derived attributes for operations

        for j in range(1, self.nF + 1):
            fac = "".join(("F", str(j)))
            self.Factory = self.Factory + [fac]

            for i in range(1, self.nW + 1):
                ware = "".join(("W", str(i)))
                self.Pair = self.Pair + [(ware, fac)]
                self.cost[(ware, fac)] = inst["trans_costs"][i - 1][j - 1]

            self.lostsalecost[fac] = self.recourse_cost

        self.setup_np = np.array([self.setup[ware] for ware in self.Warehouse])
        self.capacity_np = np.array([self.capacity[ware] for ware in self.Warehouse])
        self.capacity_np_T = self.capacity_np.reshape(1, -1)
        self.mu = inst["demands"]

    def write_scenarios_to_file(self, filename):
        """
        Write the generated scenarios to a file.

        Parameters:
        - filename (str): The file where scenarios will be written.
        """
        # Flatten the Demand array for saving
        demands = self.Demand_array.T  # Transpose to shape (nS, nF)
        np.savetxt(filename, demands, fmt="%.6f")

    def read_scenarios(self, filename):
        """
        Read scenarios from a file and populate the corresponding data structures.

        Parameters:
        - filename (str): The file containing the scenarios.
        """
        # Load the demand data from the file
        demand = np.loadtxt(filename)

        # Update the number of scenarios
        self.nS = demand.shape[0]
        self.probab = 1 / self.nS  # probability of each scenario
        self.scenario = list(range(self.nS))
        self.probability = {s: self.probab for s in self.scenario}

        # Check if the loaded data shape matches the number of factories
        if demand.shape[1] != len(self.Factory):
            raise ValueError(
                "The number of columns in the file does not match the number of factories."
            )

        # Populate Demand dictionary
        self.Demand = {}
        for count, fac in enumerate(self.Factory):
            for s in self.scenario:
                self.Demand[s, fac] = demand[s, count]

        # Create the Demand array
        self.Demand_array = demand.T  # Transpose to shape (nF, nS)

        # Populate scenario-specific data structure
        self.scen_sp_data = {scenario: {} for scenario in self.scenario}
        for scenario in self.scenario:
            for j in self.Factory:
                self.scen_sp_data[scenario][j] = self.Demand[scenario, j]

    def sce_ge_normal(self, scale_sigma, nS, save_filename=None):
        """
        Generate scenarios with a normal distribution and optionally save them to a file.

        Parameters:
        - scale_sigma (float): The scaling factor for the standard deviation.
        - nS (int): Number of scenarios to generate.
        - save_filename (str, optional): If provided, scenarios will be saved to this file.
        """
        self.nS = int(nS)
        self.probab = 1 / self.nS  # probability of each scenario
        self.scenario = list(range(self.nS))
        self.probability = {s: 1 / self.nS for s in self.scenario}

        self.Demand = {}

        # Generate scenarios
        for count, fac in enumerate(self.Factory):
            samples = np.random.normal(
                loc=self.mu[count], scale=scale_sigma * self.mu[count], size=self.nS
            )
            for s in self.scenario:
                self.Demand[s, fac] = samples[s]

        # Convert Demand dictionary to an array
        self.Demand_array = np.array(list(self.Demand.values())).reshape(
            self.nF, self.nS
        )
        self.scen_sp_data = {scenario: {} for scenario in self.scenario}

        for scenario in self.scenario:
            for j in self.Factory:
                self.scen_sp_data[scenario][j] = self.Demand[scenario, j]

        # Save scenarios to a file if a filename is provided
        if save_filename is not None:
            self.write_scenarios_to_file(save_filename)

        return 0

    def build_SP(self):
        """Building scenario subproblems"""

        SP = gp.Model("Sub-problem")

        beta = SP.addVars(
            self.Factory, obj=self.lostsalecost, vtype=GRB.CONTINUOUS, name="beta"
        )  # lostsale
        y = SP.addVars(self.Pair, obj=self.cost, vtype=GRB.CONTINUOUS, name="y")
        self.pi = SP.addConstrs(
            (0.0 <= y.sum("*", j) + beta[j] for j in self.Factory), "lostsale"
        )
        self.h = SP.addConstrs(
            (y.sum(i, "*") <= 0.0 for i in self.Warehouse), "Cap_Cons"
        )
        SP._vars = beta

        SP.modelSense = GRB.MINIMIZE
        return SP

    def update_x(self, model, xvals):

        temp = {i: self.capacity[i] * xvals[i] for i in self.Warehouse}
        model.setAttr("RHS", self.h, temp)

    def update_x_diff(self, xvals):

        for count, ware in enumerate(self.Warehouse):
            self.h[ware].rhs = max(self.capacity[ware] * xvals[count], 0)

    def update_scen(self, model, s):

        model.setAttr("RHS", self.pi, self.scen_sp_data[s])

    def build_master(self, relaxation=False):
        Master = gp.Model("Master")
        if relaxation:
            self.x = Master.addVars(self.Warehouse, ub=1.0, obj=self.setup, name="x")
        else:
            self.x = Master.addVars(
                self.Warehouse, vtype=GRB.BINARY, obj=self.setup, name="x"
            )

        self.z = Master.addVars(self.scenario, obj=self.probability, name="z")

        Master.modelSense = GRB.MINIMIZE
        self.sorted_vars = list(map(self.x.get, self.Warehouse))

        return Master

    def addcut(self, model, lazy=False, first=False):
        """this function adds a cut to the master problem looping over all scenarios"""

        cutadded = 0
        sp_vals = []
        self.update_x(self.subproblem, self.xvals)

        for scenario in self.scenario:
            self.update_scen(self.subproblem, scenario)
            self.subproblem.setParam("OutputFlag", False)
            self.subproblem.optimize()
            status = self.subproblem.status

            if status != 2:
                self.subproblem.write("sp.lp")
                raise Exception("Subproblem status - {}".format(status))

            sobj = self.subproblem.ObjVal
            self.zup[scenario] = sobj
            sp_vals.append(sobj)

            if sobj - self.zvals[scenario] > self.tol:

                hi = self.subproblem.getAttr("Pi", self.h)
                pii = self.subproblem.getAttr("Pi", self.pi)
                """ Can be made more efficient """
                s1_vec = [self.capacity[i] * self.xvals[i] for i in self.Warehouse]
                s2_vec = [self.Demand[scenario, j] for j in self.Factory]
                s1 = sum(
                    self.capacity[i] * self.xvals[i] * hi[i] for i in self.Warehouse
                )
                s2 = sum(self.Demand[scenario, j] * pii[j] for j in self.Factory)
                normal = np.linalg.norm(s1_vec + s2_vec)  # , ord=np.inf))
                # hi, pii = self.feas(hi, pii)
                dualsol = hi | pii
                sol_hash = hash(frozenset(dualsol.items()))

                not_in_list = False

                if sol_hash not in self.hash_list:
                    pii_order = [pii[j] for j in self.Factory]
                    hi_order = [hi[i] for i in self.Warehouse]
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
                                    self.capacity[i] * self.x[i] * hi[i]
                                    for i in self.Warehouse
                                )
                                + gp.quicksum(
                                    self.Demand[scenario, j] * pii[j]
                                    for j in self.Factory
                                )
                                <= self.z[scenario]
                            )
                        )  # , name)
                    else:
                        model.addConstr(
                            (
                                gp.quicksum(
                                    self.capacity[i] * self.x[i] * hi[i]
                                    for i in self.Warehouse
                                )
                                + gp.quicksum(
                                    self.Demand[scenario, j] * pii[j]
                                    for j in self.Factory
                                )
                                <= self.z[scenario]
                            )
                        )  # , name)
                    if not_in_list:
                        if not lazy:
                            self.lp_cuts[scenario].add(len(self.H_) - 1)
                    else:
                        if not lazy:
                            self.lp_cuts[scenario].add(self.hash_list.index(sol_hash))

        return cutadded, sp_vals

    def upperbound(self):
        # ctx = self.xvals.prod(self.setup)
        ctx = sum(self.setup[i] * self.xvals[i] for i in self.Warehouse)
        scenario_ctx = sum(self.zup[s] * self.probability[s] for s in self.scenario)
        upperbound = ctx + scenario_ctx
        return upperbound

    def feas(self, hi, pii):

        num_tol = 6
        for j in self.Factory:
            pii[j] = max(pii[j], 0)
            pii[j] = round(min(pii[j], self.lostsalecost[j]), num_tol)

        for i in self.Warehouse:
            for j in self.Factory:
                hi[i] = round(min(self.cost[i, j] - pii[j], hi[i]), num_tol)

        return hi, pii

    def solupdate(self, problem="IP"):
        """Given a solution, this function extracts the values of x and z from the solution
        and stores them in the dictionary."""

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
        """

        objective = np.dot(self.setup_np, xvals)
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
            objective += self.probab * sobj
            sp_vals.append(sobj)
            hi = self.subproblem.getAttr("pi", self.h)
            pii = self.subproblem.getAttr("pi", self.pi)
            # hi, pii = self.feas(hi, pii)
            # vals = sum([self.capacity[i]*xvals[count]*hi[i] for count, i in enumerate(self.Warehouse)])
            # vals1 = sum([self.Demand[scenario,j] * pii[j]  for j in self.Factory])

            dualsol = hi | pii
            sol_hash = hash(frozenset(dualsol.items()))

            if sol_hash not in self.hash_list:
                pii_order = [pii[j] for j in self.Factory]
                hi_order = [hi[i] for i in self.Warehouse]
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
        So this function takes as input the first stage solution and the list of dual solutions
        To evaluate Q_sel. It returns that entire array.
        In this we scan over the entire list of dual soln.
        So this is hard to optimize unless we change the type of the array to float32
        """
        s1 = (xvals * self.capacity_np) @ self.H[dual_list].T
        s1 = np.squeeze(s1)
        # ub, _ = find_both_numba(s1, self.dual_obj_random)
        # ub, _ = find_index_numba(s1, self.dual_obj_random)
        sp_optimal, indices = find_largest_index_numba(
            s1, self.dual_obj_random[dual_list, :]
        )
        duals = [dual_list[id] for id in indices]

        return sp_optimal, duals

    def sp_vals_sel(self, xvals, selected_dict, return_duals=False):
        """
        So this function takes as input the first stage solution and the list of dual solutions
        To evaluate Q_sel. It returns that entire array.
        """
        assert type(selected_dict) is dict
        dualctx = np.multiply(self.capacity_np, xvals)
        s1 = np.matmul(self.H, dualctx)
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
        To evaluate Q_sel. As output, it gives the V_sel and also the dual solutions which it used
        to obtain this bound for every scenario.
        """
        s1 = np.squeeze(s1)
        sp_optimal, duals = find_largest_index_numba(s1, self.dual_obj_random)
        upperbound = ctx + self.probab * np.sum(sp_optimal)

        if warm:
            return upperbound, duals, sp_optimal
        else:
            return upperbound, duals


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
            if sum_value > max_sum:
                max_sum = sum_value
                max_index = dual
        indices[scen] = max_index
        max_elements[scen] = max_sum

    return max_elements, indices
