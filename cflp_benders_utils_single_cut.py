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

    def __init__(self, instance, skip_dual_collection=False):
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

        self.gaplimit = 1e-4
        self.last_sol = None
        self.hash_set = set()
        self.hash_list = []
        self.hash_dict = {}  # Maps hash to index for O(1) lookup
        self.int_hash_set = set()
        self.demand_duals_list = []
        self.capacity_duals_list = []
        self.cut_history = []
        self.skip_dual_collection = skip_dual_collection

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

    def generate_normal_demand_scenarios(self, scale_sigma, nS, save_filename=None):
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

        # Cache ordered constraint lists for fast dual extraction
        self.h_list = [self.h[w] for w in self.Warehouse]
        self.pi_list = [self.pi[f] for f in self.Factory]

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

        self.z = Master.addVar(obj=1.0, name="z")

        Master.modelSense = GRB.MINIMIZE
        self.sorted_vars = list(map(self.x.get, self.Warehouse))

        return Master

    def add_single_cut(self, model, lazy=False, first=False):
        """this function adds a cut to the master problem looping over all scenarios"""

        cutadded = 0
        sp_vals = []
        self.update_x(self.subproblem, self.first_stage_values)

        self.zup = 0
        cut_expr = gp.LinExpr()
        cut_dual_ids = []

        for scenario in self.scenario:
            self.update_scen(self.subproblem, scenario)
            self.subproblem.setParam("OutputFlag", False)
            self.subproblem.optimize()
            status = self.subproblem.status

            if status != 2:
                self.subproblem.write("sp.lp")
                raise Exception("Subproblem status - {}".format(status))

            sobj = self.subproblem.ObjVal
            self.zup += sobj
            sp_vals.append(sobj)

            if not self.skip_dual_collection:
                # Extract duals using cached constraint lists (fast!)
                hi_order = tuple(c.Pi for c in self.h_list)
                pii_order = tuple(c.Pi for c in self.pi_list)

                # Hash tuple directly (much faster than frozenset of dict items)
                sol_hash = hash(hi_order + pii_order)

                if sol_hash not in self.hash_dict:
                    dual_id = len(self.capacity_duals_list)
                    self.capacity_duals_list.append(list(hi_order))
                    self.demand_duals_list.append(list(pii_order))
                    self.hash_list.append(sol_hash)
                    self.hash_dict[sol_hash] = dual_id
                    self.dual_soln_optimal_counter[dual_id] = 0
                else:
                    dual_id = self.hash_dict[sol_hash]
                    if not first:
                        if dual_id < self.dual_pool_size[-1]:
                            self.dual_soln_optimal_counter[dual_id] += 1

                cut_dual_ids.append(dual_id)

                # Create dict for cut expression (needed below)
                hi = {
                    self.Warehouse[i]: hi_order[i] for i in range(len(self.Warehouse))
                }
                pii = {self.Factory[j]: pii_order[j] for j in range(len(self.Factory))}
            else:
                hi = self.subproblem.getAttr("Pi", self.h)
                pii = self.subproblem.getAttr("Pi", self.pi)

            cut_expr += self.probab * (
                gp.quicksum(
                    self.capacity[i] * self.x[i] * hi[i] for i in self.Warehouse
                )
                + gp.quicksum(self.Demand[scenario, j] * pii[j] for j in self.Factory)
            )

        self.zup = self.zup * self.probab
        if self.zup - self.second_stage_values > max(
            self.tol, 0.001 * abs(self.second_stage_values)
        ):
            # if self.zup - self.second_stage_values > max(self.tol, 0.001 * abs(self.second_stage_values)):
            cutadded += 1
            # cut_expr *= self.probab
            if lazy:
                model.cbLazy(cut_expr <= self.z)
            else:
                model.addConstr(cut_expr <= self.z)

            # Track which dual solutions contributed to this cut (only if collecting duals)
            if not self.skip_dual_collection:
                self.cut_history.append(cut_dual_ids.copy())

        return cutadded, self.zup

    def upperbound(self):
        # ctx = self.first_stage_values.prod(self.setup)
        ctx = sum(self.setup[i] * self.first_stage_values[i] for i in self.Warehouse)
        scenario_ctx = self.zup
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

    def extract_solution_values(self, problem="IP"):
        """Given a solution, this function extracts the values of x and z from the solution
        and stores them in the dictionary."""

        self.first_stage_values = self.master.getAttr("x", self.x)
        if problem == "IP":
            for k, v in self.first_stage_values.items():
                if v > 0.5:
                    self.first_stage_values[k] = 1.0
                else:
                    self.first_stage_values[k] = 0.0
        else:
            self.first_stage_values = {
                k: max(v, 0.0) for k, v in self.first_stage_values.items()
            }
        self.second_stage_values = self.z.X
        # self.zvals = self.master.getAttr("x", self.z)

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

            if not self.skip_dual_collection:
                # Extract duals using cached constraint lists (fast!)
                hi_order = tuple(c.pi for c in self.h_list)
                pii_order = tuple(c.pi for c in self.pi_list)

                # Hash tuple directly (much faster than frozenset of dict items)
                sol_hash = hash(hi_order + pii_order)

                if sol_hash not in self.hash_dict:
                    dual_id = len(self.capacity_duals_list)
                    self.capacity_duals_list.append(list(hi_order))
                    self.demand_duals_list.append(list(pii_order))
                    self.hash_list.append(sol_hash)
                    self.hash_dict[sol_hash] = dual_id
                    self.dual_soln_optimal_counter[dual_id] = 0
                    if get_duals:
                        duals.append(dual_id)
                elif get_duals:
                    dual_id = self.hash_dict[sol_hash]
                    self.dual_soln_optimal_counter[dual_id] += 1
                    duals.append(dual_id)

        if get_duals:
            return objective, duals, sp_vals
        else:
            return objective, sp_vals

    def evaluate_subproblems_with_dual_list(self, xvals, dual_list):
        """
        So this function takes as input the first stage solution and the list of dual solutions
        To evaluate Q_sel. It returns that entire array.
        In this we scan over the entire list of dual soln.
        So this is hard to optimize unless we change the type of the array to float32
        """
        # Convert dual_list to numpy array once
        dual_indices = np.array(dual_list, dtype=np.int64)

        s1 = (xvals * self.capacity_np) @ self.capacity_duals_array[dual_indices].T
        s1 = np.squeeze(s1)

        # Ensure s1 is 1D array for numba function
        if s1.ndim == 0:
            s1 = s1.reshape(1)

        # Use optimized version that avoids creating intermediate array
        sp_optimal, local_indices = find_largest_index_with_subset_numba(
            s1, self.dual_obj_random, dual_indices
        )
        duals = [dual_list[id] for id in local_indices]

        return sp_optimal, duals

    def evaluate_subproblems_with_selected_duals(
        self, xvals, selected_dict, return_optimal_duals=False
    ):
        """
        So this function takes as input the first stage solution and the list of dual solutions
        To evaluate Q_sel. It returns that entire array.
        """
        assert type(selected_dict) is dict
        dualctx = np.multiply(self.capacity_np, xvals)
        s1 = np.matmul(self.capacity_duals_array, dualctx)
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
        if return_optimal_duals:
            return ub, max_indices
        else:
            return ub

    def evaluate_subproblems_fast_on_dual_list(self, s1, duals_list):
        """
        This function just evaluates the solution on the new duals.
        It is very fast because it just sums the values and no max
        is involved.
        duals is one dual solution for every scenario.
        """
        max_theta = 0
        for duals in duals_list:
            theta = 0
            for scen in self.scenario:
                sp_val = s1[duals[scen]] + self.dual_obj_random[duals[scen], scen]
                theta += max(sp_val, 0)
            max_theta = max(max_theta, theta)
        return max_theta

    def evaluate_subproblems_fast(self, s1, duals, scenario_subset):
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

    def compute_value_function_approximation(
        self, ctx, s1, include_scenario_details=False
    ):
        """
        So this function takes as input the first stage solution and the list of dual solutions
        To evaluate Q_sel. As output, it gives the V_sel and also the dual solutions which it used
        to obtain this bound for every scenario.
        """
        s1 = np.squeeze(s1)
        sp_optimal, duals = find_largest_index_numba(s1, self.dual_obj_random)
        upperbound = ctx + self.probab * np.sum(sp_optimal)

        if include_scenario_details:
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


@njit(
    nb.types.Tuple((nb.float64[:], nb.int64[:]))(
        nb.float64[:], nb.float64[:, :], nb.int64[:]
    ),
    parallel=True,
    fastmath=True,
)
def find_largest_index_with_subset_numba(s1, dual_obj_random, dual_indices):
    """
    Optimized version that takes dual indices and accesses dual_obj_random directly
    instead of creating a subset array.

    Args:
        s1: 1D array of capacity dual values
        dual_obj_random: Full 2D array of dual objective values (n_duals x n_scenarios)
        dual_indices: Indices of duals to consider

    Returns:
        max_elements: Maximum value for each scenario
        local_indices: Index within dual_indices for each scenario
    """
    n_duals = len(dual_indices)
    n_scenarios = dual_obj_random.shape[1]

    indices = np.empty(n_scenarios, dtype=np.int64)
    max_elements = np.empty(n_scenarios, dtype=np.float64)

    for scen in prange(n_scenarios):
        max_sum = np.finfo(np.float64).min
        max_index = -1
        for local_idx in range(n_duals):
            global_idx = dual_indices[local_idx]
            sum_value = s1[local_idx] + dual_obj_random[global_idx, scen]
            if sum_value > max_sum:
                max_sum = sum_value
                max_index = local_idx
        indices[scen] = max_index
        max_elements[scen] = max_sum

    return max_elements, indices


@njit(
    nb.types.Tuple((nb.float64[:], nb.int64[:]))(
        nb.float64[:], nb.int64[:], nb.float64[:, :], nb.float64[:, :]
    ),
    parallel=True,
    fastmath=True,
    cache=True,
)
def evaluate_dual_subset_numba(
    weighted_capacity, dual_indices, capacity_duals_array, dual_obj_random
):
    """
    Optimized kernel that avoids creating intermediate array subsets.

    Directly computes capacity contributions and finds optimal dual for each scenario.

    Args:
        weighted_capacity: x * capacity (shape: n_warehouses)
        dual_indices: Indices of duals to evaluate (shape: n_duals_subset)
        capacity_duals_array: Full capacity duals (shape: n_all_duals x n_warehouses)
        dual_obj_random: Full demand contributions (shape: n_all_duals x n_scenarios)

    Returns:
        max_values: Best objective for each scenario (shape: n_scenarios)
        optimal_dual_ids: Global dual ID for each scenario (shape: n_scenarios)
    """
    n_scenarios = dual_obj_random.shape[1]
    n_duals = len(dual_indices)
    n_warehouses = len(weighted_capacity)

    max_values = np.empty(n_scenarios, dtype=np.float64)
    optimal_dual_ids = np.empty(n_scenarios, dtype=np.int64)

    # Parallel over scenarios
    for scen in prange(n_scenarios):
        max_obj = np.finfo(np.float64).min
        best_dual_id = -1

        # Loop over subset of duals
        for i in range(n_duals):
            global_dual_idx = dual_indices[i]

            # Compute h^T * (C * x) - dot product manually
            capacity_contrib = 0.0
            for w in range(n_warehouses):
                capacity_contrib += (
                    capacity_duals_array[global_dual_idx, w] * weighted_capacity[w]
                )

            # Add π^T * d_s
            obj_value = capacity_contrib + dual_obj_random[global_dual_idx, scen]

            if obj_value > max_obj:
                max_obj = obj_value
                best_dual_id = global_dual_idx

        max_values[scen] = max_obj
        optimal_dual_ids[scen] = best_dual_id

    return max_values, optimal_dual_ids
