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

        self.gaplimit = 1e-3
        self.last_sol = None
        self.hash_set = set()
        self.hash_list = []
        self.int_hash_set = set()
        self.PI_ = []
        self.H_ = []
        self.nDev = 0.1
        self.probab = 1 / float(nscen)
        self.cut_history = []
        self.dual_soln_optimal_counter = {}
        self.dual_pool_size = []

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

    def sce_ge_normal(self, nscen):
        """
        Generates nscen scenarios
        """

        self.scenario = list(range(nscen))
        demandav = np.array(list(self.mu.values()))
        self.demand = np.round(
            np.random.normal(
                loc=demandav, scale=self.nDev * demandav, size=(nscen, self.nK)
            ),
            decimals=2,
        )
        self.demand = np.maximum(self.demand, np.zeros((nscen, self.nK)))

        for i in range(nscen):
            for k in range(self.nK):
                if self.demand[i][k] < 0.0:
                    print(k, self.demand[i][k])
                    print("h")

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

    def update_x_diff(self, first_stage_values):

        for count, arc in enumerate(self.arcs):
            self.h[arc].rhs = max(self.capacity[arc] * first_stage_values[count], 0)

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
        self.z = Master.addVar(obj=1.0, name="z")

        Master.modelSense = GRB.MINIMIZE

        return Master

    def add_single_cut(self, model, lazy=False, first=False):
        """
        Generate Benders optimality cuts by solving subproblems for all scenarios.

        Iterates over all scenarios and checks if the subproblem's objective value exceeds
        the current second-stage approximation. If it does, constructs and adds an optimality
        cut to the master problem either as a lazy constraint (during branch-and-bound) or
        as a regular constraint (during LP relaxation).

        Args:
            model: The master optimization model to which cuts are added
            lazy (bool): If True, add cuts as lazy constraints; if False, add as regular constraints
            first (bool): Flag used to handle the first iteration differently if required

        Returns:
            tuple: (num_cuts_added, subproblem_objective_values) where:
                - num_cuts_added: Number of cuts added to the model
                - subproblem_objective_values: List of subproblem objective values across all scenarios
        """
        cutadded = 0
        subproblem_objective_values = []
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
                raise Exception("Subproblem status - {}".format(status))

            subproblem_objective = self.subproblem.ObjVal
            self.zup += subproblem_objective
            subproblem_objective_values.append(subproblem_objective)

            hi = self.subproblem.getAttr("pi", self.h)
            pii = self.subproblem.getAttr("pi", self.pi)
            hi, pii = self.feas(hi, pii)

            dualsol = hi | pii
            sol_hash = hash(frozenset(dualsol.items()))

            # Always add dual solution (don't check for duplicates)
            if sol_hash not in self.hash_list:
                pii_order = [
                    pii[k, i] for k in self.commodities for i in self.demand_data[k]
                ]
                hi_order = [hi[a] for a in self.arcs]
                self.H_.append(hi_order)
                self.PI_.append(pii_order)
                self.hash_list.append(sol_hash)
                dual_id = len(self.hash_list) - 1
            else:
                dual_id = self.hash_list.index(sol_hash)

            cut_dual_ids.append(dual_id)
            cut_expr += gp.quicksum(
                self.capacity[(i, j)] * self.x[i, j] * hi[(i, j)]
                for (i, j) in self.arcs
            ) + gp.quicksum(
                self.demand[scenario][count]
                * (pii[k, self.demand_data[k][0]] - pii[k, self.demand_data[k][1]])
                for count, k in enumerate(self.commodities)
            )

        if self.zup - self.second_stage_values > self.tol:
            # ):  # , 0.001 * abs(self.second_stage_values)):
            # if self.zup - self.second_stage_values > max(self.tol, 0.001 * abs(self.second_stage_values)):
            cutadded += 1
            # cut_expr *= self.probab
            if lazy:
                model.cbLazy(cut_expr <= self.z)
            else:
                model.addConstr(cut_expr <= self.z)

            # Track which dual solutions contributed to this cut
            self.cut_history.append(cut_dual_ids.copy())

        return cutadded, self.zup

    def calculate_upper_bound_from_subproblems(self):
        """
        Calculate the upper bound using current first-stage solution and scenario subproblem values.

        Computes the total objective value by summing the first-stage fixed costs with the
        expected second-stage costs from all scenario subproblems.

        Returns:
            float: Total upper bound objective value
        """

        first_stage_cost = sum(
            self.fixedcost[(i, j)] * self.first_stage_values[(i, j)]
            for (i, j) in self.arcs
        )
        scenario_ctx = self.zup
        upperbound = first_stage_cost + scenario_ctx
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

    def extract_solution_values(self, problem="IP"):
        """
        Extract first-stage and second-stage variable values from the master problem solution.

        Retrieves the values of x (first-stage facility decisions) and z (second-stage cost variables)
        from the master problem solution. For integer problems, rounds x values to 0 or 1.

        Args:
            problem (str): Problem type - "IP" for integer programming, "LP" for linear programming
        """

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

    def optimal_value_duals(self, first_stage_values, get_duals=False):
        """
        So this function takes as input the first stage solution
        It evalautes V(x) and outputs it
        It also gives the optimal subproblem objectives (zvals) as output
        """

        objective = np.dot(self.fixedcost_np, first_stage_values)
        self.update_x_diff(first_stage_values)

        if get_duals:
            optimal_dual_solutions = []
        subproblem_objective_values = []

        for scenario in self.scenario:
            self.update_scen(self.subproblem, scenario)
            self.subproblem.setParam("OutputFlag", False)
            self.subproblem.optimize()
            status = self.subproblem.status

            if status != 2:
                raise Exception("Subproblem status - {}".format(status))

            subproblem_objective = self.subproblem.ObjVal
            objective += subproblem_objective
            subproblem_objective_values.append(subproblem_objective)
            hi = self.subproblem.getAttr("pi", self.h)
            pii = self.subproblem.getAttr("pi", self.pi)
            hi, pii = self.feas(hi, pii)

            dualsol = hi | pii
            sol_hash = hash(frozenset(dualsol.items()))

            # Always add dual solution (don't check for duplicates)
            if sol_hash not in self.hash_list:
                pii_order = [
                    pii[k, i] for k in self.commodities for i in self.demand_data[k]
                ]
                hi_order = [hi[a] for a in self.arcs]
                self.H_.append(hi_order)
                self.PI_.append(pii_order)
                self.hash_list.append(sol_hash)
                if get_duals:
                    optimal_dual_solutions.append(len(self.hash_list) - 1)
            elif get_duals:
                optimal_dual_solutions.append(self.hash_list.index(sol_hash))

        if get_duals:
            return (
                objective,
                optimal_dual_solutions,
                subproblem_objective_values,
            )
        else:
            return objective, subproblem_objective_values

    def evaluate_subproblems_with_dual_list(
        self, first_stage_solution, dual_solution_indices
    ):
        """
        Evaluate subproblem upper bounds using a specified list of dual solutions across all scenarios.

        This function uses a uniform list of dual solution indices (same for all scenarios)
        and finds the optimal dual solution for each scenario from this restricted set.
        Uses high-performance numba-compiled function for fast evaluation.

        Args:
            first_stage_solution (np.array): First-stage facility location decisions
            dual_solution_indices (list): List of dual solution indices to evaluate (uniform across scenarios)

        Returns:
            tuple: (scenario_optimal_values, optimal_dual_ids) where:
                - scenario_optimal_values: Array of optimal subproblem values for each scenario
                - optimal_dual_ids: List of optimal dual solution IDs for each scenario
        """
        weighted_capacity_dot_duals = (
            first_stage_solution * self.capacity_np_T
        ) @ self.H[dual_solution_indices].T
        weighted_capacity_dot_duals = np.squeeze(weighted_capacity_dot_duals)

        scenario_optimal_values, relative_dual_indices = find_largest_index_numba(
            weighted_capacity_dot_duals, self.dual_obj_random[dual_solution_indices, :]
        )
        optimal_dual_ids = [
            dual_solution_indices[relative_idx]
            for relative_idx in relative_dual_indices
        ]

        return scenario_optimal_values, optimal_dual_ids

    def evaluate_subproblems_with_selected_duals(
        self, first_stage_solution, scenario_dual_selection, return_duals=False
    ):
        """
        Evaluate subproblem upper bounds using scenario-specific dual solution selections.

        This function implements selective dual evaluation where each scenario uses its own
        curated subset of dual solutions. Provides a balance between computational efficiency
        and solution quality by avoiding evaluation of the full dual pool.

        Args:
            first_stage_solution (np.array): First-stage facility location decisions
            scenario_dual_selection (dict): Dictionary mapping each scenario to its selected dual solution indices
                                           Format: {scenario_id: [dual_id1, dual_id2, ...]}
            return_duals (bool): If True, returns the optimal dual indices along with bounds

        Returns:
            list or tuple: If return_duals=False, returns list of upper bounds for each scenario.
                          If return_duals=True, returns (scenario_upper_bounds, optimal_dual_indices)
        """
        assert isinstance(
            scenario_dual_selection, dict
        ), "scenario_dual_selection must be a dictionary"
        capacity_dual_contributions = (
            first_stage_solution * self.capacity_np_T
        ) @ self.H.T
        capacity_dual_contributions = np.squeeze(capacity_dual_contributions)
        optimal_dual_indices = []
        scenario_upper_bounds = []

        for scenario_id in self.scenario:
            selected_dual_ids = list(scenario_dual_selection[scenario_id])
            dual_objective_values = (
                capacity_dual_contributions[selected_dual_ids]
                + self.dual_obj_random[selected_dual_ids, scenario_id]
            )
            best_dual_position = np.argmax(dual_objective_values)
            optimal_dual_id = selected_dual_ids[best_dual_position]
            optimal_dual_indices.append(optimal_dual_id)
            scenario_upper_bounds.append(dual_objective_values[best_dual_position])
        if return_duals:
            return scenario_upper_bounds, optimal_dual_indices
        else:
            return scenario_upper_bounds

    def evaluate_subproblems_fast(
        self, capacity_dual_contributions, scenario_dual_mapping, active_scenarios
    ):
        """
        Fast evaluation of subproblem values using pre-specified dual solutions for each scenario.

        This function provides direct evaluation without optimization, making it very fast.
        No argmax operations are performed - it simply looks up and sums pre-computed values.
        Only evaluates scenarios in the active set, setting others to 0.0.

        Args:
            capacity_dual_contributions (np.array): Pre-computed capacity dual contributions
            scenario_dual_mapping (dict): Mapping from scenario ID to the dual solution ID to use
            active_scenarios (set or list): Subset of scenarios to evaluate

        Returns:
            list: Subproblem objective values for each scenario (active scenarios get computed values, others get 0.0)
        """
        subproblem_values = []
        for scenario_id in self.scenario:
            if scenario_id in active_scenarios:
                assigned_dual_id = scenario_dual_mapping[scenario_id]
                objective_value = (
                    capacity_dual_contributions[assigned_dual_id]
                    + self.dual_obj_random[assigned_dual_id, scenario_id]
                )
                subproblem_values.append(objective_value)
            else:
                subproblem_values.append(0.0)
        return subproblem_values

    def compute_value_function_approximation(
        self,
        first_stage_cost,
        capacity_dual_contributions,
        include_scenario_details=False,
    ):
        """
        Compute the complete value function approximation using all available dual solutions.

        This function evaluates the full Dual Solution Pooling (DSP) approximation by finding
        the optimal dual solution for each scenario across the entire dual pool. Returns
        the total expected value and identifies which dual solutions were optimal.

        Args:
            first_stage_cost (float): Objective value of first-stage solution
            capacity_dual_contributions (np.array): Pre-computed capacity dual contributions
            include_scenario_details (bool): If True, returns individual scenario values and optimal duals

        Returns:
            tuple: If include_scenario_details=False, returns (total_value_function, optimal_dual_indices)
                   If include_scenario_details=True, returns (total_value_function, optimal_dual_indices, scenario_optimal_values)
        """
        capacity_dual_contributions = np.squeeze(capacity_dual_contributions)
        scenario_optimal_values, optimal_dual_indices = find_largest_index_numba(
            capacity_dual_contributions, self.dual_obj_random
        )
        total_value_function = first_stage_cost + np.sum(scenario_optimal_values)

        if include_scenario_details:
            return total_value_function, optimal_dual_indices, scenario_optimal_values
        else:
            return total_value_function, optimal_dual_indices


# @njit(parallel=True)#(nb.float64[:], nb.float64[:])
# @njit(nb.types.Tuple((nb.float64[:], nb.int64[:]))(nb.float64[:], nb.float64[:,:]), parallel=True, fastmath=True, nogil=True)
@njit(
    nb.types.Tuple((nb.float64[:], nb.int64[:]))(nb.float64[:], nb.float64[:, :]),
    parallel=True,
    fastmath=True,
)
def find_largest_index_numba(capacity_terms, demand_terms):
    """
    High-performance function to find optimal dual solutions for each scenario.

    For each scenario (column), finds the dual solution (row) that maximizes
    the sum capacity_terms[dual] + demand_terms[dual, scenario]. This is the core computational
    kernel for Dual Solution Pooling (DSP).

    Uses Numba JIT compilation with parallel execution and fast math optimizations
    for maximum performance on large dual solution pools.

    Args:
        capacity_terms (np.array): Capacity constraint contributions for each dual
        demand_terms (np.array): Demand constraint contributions for (dual, scenario) pairs

    Returns:
        tuple: (max_elements, indices) where:
            - max_elements: Maximum dual objective value for each scenario
            - indices: Index of optimal dual solution for each scenario
    """
    num_duals, num_scenarios = demand_terms.shape

    optimal_dual_indices = np.empty(num_scenarios, dtype=np.int64)
    max_objective_values = np.empty(num_scenarios, dtype=np.float64)

    for scenario_idx in prange(num_scenarios):
        max_objective = np.finfo(np.float64).min
        best_dual_index = -1
        for dual_idx in range(len(capacity_terms)):
            objective_value = (
                capacity_terms[dual_idx] + demand_terms[dual_idx, scenario_idx]
            )
            if objective_value >= max_objective:
                max_objective = objective_value
                best_dual_index = dual_idx
        optimal_dual_indices[scenario_idx] = best_dual_index
        max_objective_values[scenario_idx] = max_objective

    return max_objective_values, optimal_dual_indices
