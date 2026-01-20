import numpy as np
import json
import gurobipy as gp
from gurobipy import GRB
from numba import njit, prange
import numba as nb
import tempfile
import os

# Import UCIG for instance generation
try:
    from ucig import UCIG
except ImportError:
    # If ucig is not in path, try relative import
    try:
        from .ucig import UCIG
    except ImportError:
        UCIG = None  # UCIG not available, instance generation will be disabled


class AtomicDualStorage:
    """
    Thread-safe atomic storage for dual solutions with guaranteed indexing consistency.

    This class ensures that all dual solution arrays (generation, demand)
    remain perfectly synchronized and eliminates all possible indexing errors.
    """

    def __init__(self, nG, T, nS):
        """
        Initialize atomic dual storage with fixed dimensions.

        Args:
            nG (int): Number of thermal generators
            T (int): Number of time periods
            nS (int): Number of scenarios
        """
        # Store dimensions for validation
        self.nG, self.T, self.nS = nG, T, nS

        # Fixed widths - prevents shape mismatches
        self.gen_width = 2 * nG * T  # min + max duals
        self.dem_width = T

        # Single growth counter - perfect synchronization
        self.capacity = 1000
        self.size = 0

        # Pre-allocated contiguous memory blocks
        self.generation_duals = np.empty(
            (self.capacity, self.gen_width), dtype=np.float64
        )
        self.demand_duals = np.empty((self.capacity, self.dem_width), dtype=np.float64)

        # Hash deduplication with guaranteed 1:1 mapping
        self.hash_to_index = {}
        self.index_to_hash = {}

        # Precomputed scenario-dependent dual objectives
        self.dual_obj_random = np.empty((self.capacity, nS), dtype=np.float64)

    def add_dual_solution(self, gen_duals, dem_duals, dual_hash):
        """
        Atomically add a dual solution with full validation.

        Args:
            gen_duals (list): Generation constraint duals [min_duals + max_duals]
            dem_duals (list): Demand constraint duals
            dual_hash: Hash of the dual solution for deduplication

        Returns:
            int: Index of the dual solution (existing or newly added)
        """
        # Atomic deduplication check
        if dual_hash in self.hash_to_index:
            return self.hash_to_index[dual_hash]

        # Dimension validation - prevents all shape errors
        if len(gen_duals) != self.gen_width:
            raise ValueError(
                f"Generation duals width mismatch: {len(gen_duals)} != {self.gen_width}"
            )
        if len(dem_duals) != self.dem_width:
            raise ValueError(
                f"Demand duals width mismatch: {len(dem_duals)} != {self.dem_width}"
            )

        # Atomic capacity check and growth
        if self.size >= self.capacity:
            self._atomic_grow()

        # Atomic insertion
        idx = self.size
        self.generation_duals[idx] = gen_duals
        self.demand_duals[idx] = dem_duals

        # Atomic index registration - maintains perfect hash-to-index mapping
        self.hash_to_index[dual_hash] = idx
        self.index_to_hash[idx] = dual_hash
        self.size += 1

        return idx

    def _atomic_grow(self):
        """Atomically double the storage capacity."""
        new_capacity = 2 * self.capacity

        # Allocate new arrays with doubled capacity
        new_gen = np.empty((new_capacity, self.gen_width), dtype=np.float64)
        new_dem = np.empty((new_capacity, self.dem_width), dtype=np.float64)
        new_obj = np.empty((new_capacity, self.nS), dtype=np.float64)

        # Copy existing data atomically
        new_gen[: self.size] = self.generation_duals[: self.size]
        new_dem[: self.size] = self.demand_duals[: self.size]
        new_obj[: self.size] = self.dual_obj_random[: self.size]

        # Atomic swap - all arrays updated simultaneously
        self.generation_duals = new_gen
        self.demand_duals = new_dem
        self.dual_obj_random = new_obj
        self.capacity = new_capacity

    def get_arrays(self):
        """
        Get views of active dual solution arrays.

        Returns:
            tuple: (generation_duals_array, demand_duals_array)
        """
        return (
            self.generation_duals[: self.size],
            self.demand_duals[: self.size],
        )

    def get_dual_obj_random(self):
        """Get view of precomputed dual objective values."""
        return self.dual_obj_random[: self.size]

    def update_dual_obj_random(self, demand_scenarios_matrix):
        """
        Update precomputed dual objective values using vectorized operations.

        Args:
            demand_scenarios_matrix (np.array): Shape (nS, T)
        """
        if self.size == 0:
            return

        # Reset dual objectives
        self.dual_obj_random[: self.size, :] = 0.0

        # Vectorized demand contribution calculation
        if demand_scenarios_matrix is not None:
            # Matrix multiplication: (size, T) @ (T, nS) = (size, nS)
            demand_contributions = (
                self.demand_duals[: self.size] @ demand_scenarios_matrix.T
            )
            self.dual_obj_random[: self.size] += demand_contributions

    def update_dual_obj_random_incremental(self, demand_scenarios_matrix, start_idx):
        """
        Incrementally update precomputed dual objective values for newly added duals only.

        Args:
            demand_scenarios_matrix (np.array): Shape (nS, T)
            start_idx (int): Index of first new dual solution to update
        """
        if start_idx >= self.size:
            return

        # Reset dual objectives for new duals only
        self.dual_obj_random[start_idx : self.size, :] = 0.0

        # Vectorized demand contribution calculation for new duals only
        if demand_scenarios_matrix is not None:
            # Matrix multiplication: (num_new_duals, T) @ (T, nS) = (num_new_duals, nS)
            demand_contributions = (
                self.demand_duals[start_idx : self.size] @ demand_scenarios_matrix.T
            )
            self.dual_obj_random[start_idx : self.size] += demand_contributions

    def __len__(self):
        """Return number of stored dual solutions."""
        return self.size

    def get_hash_by_index(self, index):
        """Get hash for a given index."""
        return self.index_to_hash.get(index)

    def get_index_by_hash(self, dual_hash):
        """Get index for a given hash."""
        return self.hash_to_index.get(dual_hash)


class UCinst:

    def __init__(self, nscen=None):
        self.T = 0  # Time periods
        self.nG = None  # Number of thermal generators
        self.nS = int(nscen) if nscen else None  # Number of scenarios

        # Generator data structures
        self.thermal_gens = []
        self.periods = []

        # Generator parameters
        self.min_up_time = {}
        self.min_down_time = {}
        self.min_power = {}
        self.max_power = {}
        self.unit_cost = {}
        self.startup_cost = {}

        # Demand data
        self.demand_base = []
        self.demand_scenarios = {}

        # Scenario management
        self.scenario = []
        self.probability = {}
        self.scen_sp_data = {}

        # Benders decomposition attributes
        self.gaplimit = 0.0001
        self.tol = 1e-3
        self.time_limit = 3600  # Default time limit in seconds
        self.dual_soln_optimal_counter = {}

        # Dual solution storage - will be initialized after dimensions are known
        self.dual_storage = None

        # Performance optimization arrays
        self.dual_obj_random = None
        self.capacity_duals_array = None
        self.demand_duals_array = None
        self.demand_scenarios_matrix = (
            None  # Shape: (nS, T) - precomputed demand scenario matrix
        )

        # Precomputed matrices for fast dual evaluation
        self.min_power_matrix = None  # Shape: (nG*T,) - min power coefficients
        self.max_power_matrix = None  # Shape: (nG*T,) - max power coefficients
        self.commitment_solution_cache = {}  # Cache for commitment-weighted solutions
        self.dual_matrix_cache = {}  # Cache for dual matrix products

        # Solution tracking
        self.scenario_upper_bounds = {}
        self.first_stage_values = {}
        self.second_stage_values = {}
        self.lp_cuts = {}
        self.dual_pool_size = []

        # Cost scaling factor for numerical stability
        self.cost_scale = 1.0
        self.penalty_cost = 1000.0  # Penalty for unmet demand

    @staticmethod
    def generate_instance(
        n_generators=10,
        n_days=1,
        periods_per_day=24,
        difficulty=1,
        output_file=None,
        a_min=0.00001,
        a_max=0.1,
        csc=0,
        **kwargs,
    ):
        """
        Generate a Unit Commitment instance and save to JSON file.

        This is a convenient wrapper around the UCIG instance generator that
        creates instances compatible with uc_benders_utils.py.

        Args:
            n_generators (int): Number of thermal generators [10]
            n_days (int): Number of days in planning horizon [1]
            periods_per_day (int): Number of time periods per day [24]
            difficulty (int): Difficulty level (1=easy, 2=medium, 3=hard) [1]
            output_file (str): Output JSON filename. If None, creates temp file
            a_min (float): Minimum quadratic cost coefficient [0.00001]
            a_max (float): Maximum quadratic cost coefficient [0.1]
            csc (int): 1 for constant startup cost, 0 for variable [0]
            **kwargs: Additional parameters passed to UCIG

        Returns:
            str: Path to the generated instance JSON file

        Example:
            >>> # Generate a simple test instance
            >>> filename = UCinst.generate_instance(n_generators=5, n_days=1)
            >>> uc = UCinst(nscen=10)
            >>> uc.load_instance(filename)

            >>> # Generate a challenging instance
            >>> filename = UCinst.generate_instance(
            ...     n_generators=50, n_days=7, difficulty=3,
            ...     output_file="large_instance.json"
            ... )
        """
        if UCIG is None:
            raise ImportError(
                "UCIG module not available. Ensure ucig.py is in the same directory "
                "or in your Python path."
            )

        # Fixed seed for reproducibility
        seed = 42

        # Create output filename if not provided
        if output_file is None:
            # Create a temporary file with a descriptive name
            output_file = f"uc_instance_g{n_generators}_d{n_days}_s{seed}.json"

        # Initialize UCIG instance generator
        ucig = UCIG(
            gg=n_days,
            breaks=periods_per_day,
            gmax=n_generators,
            a_min=a_min,
            a_max=a_max,
            seed=seed,
            csc=csc,
            difficulty=difficulty,
        )

        # Generate the instance data
        ucig.init_data()

        # Convert to JSON format
        instance_data = ucig.to_json()

        # Save to file
        with open(output_file, "w") as f:
            json.dump(instance_data, f, indent=2)

        print(f"Generated UC instance saved to: {output_file}")
        print(f"  - Generators: {n_generators}")
        print(f"  - Time periods: {n_days * periods_per_day}")
        print(f"  - Difficulty: {difficulty}")
        print(f"  - Seed: 42 (fixed)")

        return output_file

    @classmethod
    def from_generated(
        cls,
        nscen=None,
        n_generators=10,
        n_days=1,
        periods_per_day=24,
        difficulty=1,
        max_periods=None,
        **kwargs,
    ):
        """
        Generate a UC instance and load it in one step.

        This convenience method generates an instance using UCIG and immediately
        loads it into a UCinst object, ready for use with Benders decomposition.

        Args:
            nscen (int): Number of scenarios to generate (required)
            n_generators (int): Number of thermal generators [10]
            n_days (int): Number of days in planning horizon [1]
            periods_per_day (int): Number of time periods per day [24]
            difficulty (int): Difficulty level (1=easy, 2=medium, 3=hard) [1]
            max_periods (int): If specified, only use first max_periods [None]
            **kwargs: Additional parameters passed to generate_instance()

        Returns:
            UCinst: Initialized UCinst object with loaded instance

        Example:
            >>> # Generate and load in one step
            >>> uc = UCinst.from_generated(
            ...     nscen=20, n_generators=20, n_days=7,
            ...     difficulty=2
            ... )
            >>> # Generate scenarios
            >>> uc.generate_demand_scenarios(nS=20, demand_std_dev=0.05)
            >>> # Ready to use with Benders
        """
        if nscen is None:
            raise ValueError("nscen parameter is required for from_generated()")

        # Generate the instance
        instance_file = cls.generate_instance(
            n_generators=n_generators,
            n_days=n_days,
            periods_per_day=periods_per_day,
            difficulty=difficulty,
            output_file=None,  # Use auto-generated filename
            **kwargs,
        )

        # Create UCinst object and load the instance
        uc = cls(nscen=nscen)
        uc.load_instance(
            instance_file, max_periods=max_periods
        )

        # Clean up temporary file (optional - comment out if you want to keep it)
        # os.remove(instance_file)

        return uc

    def load_instance(self, filename_or_data, max_periods=None):
        """
        Load unit commitment instance data from JSON file or dict.

        Args:
            filename_or_data (str or dict): Path to JSON file or data dictionary
            max_periods (int): If specified, only use the first max_periods time periods
        """
        if isinstance(filename_or_data, dict):
            data = filename_or_data
        else:
            with open(filename_or_data, "r") as f:
                data = json.load(f)

        original_periods = data["time_periods"]

        # Apply period limit if specified
        if max_periods is not None:
            if max_periods > original_periods:
                print(
                    f"Warning: Requested periods ({max_periods}) > original periods ({original_periods}). Using all periods."
                )
                self.T = original_periods
            elif max_periods <= 0:
                raise ValueError("max_periods must be positive")
            else:
                self.T = max_periods
                print(
                    f"Using first {self.T} periods out of {original_periods} available periods"
                )
        else:
            self.T = original_periods

        self.periods = list(range(1, self.T + 1))
        self.demand_base = data["demand"][
            : self.T
        ]  # Truncate demand to first T periods

        # Load thermal generator data
        self.thermal_gens = list(data["thermal_generators"].keys())
        self.nG = len(self.thermal_gens)

        for g, gen_data in data["thermal_generators"].items():
            self.min_up_time[g] = gen_data["time_up_minimum"]
            self.min_down_time[g] = gen_data["time_down_minimum"]
            self.min_power[g] = gen_data["power_output_minimum"]
            self.max_power[g] = gen_data["power_output_maximum"]
            self.unit_cost[g] = gen_data["piecewise_production"][0]["cost"]
            # Load startup cost from metadata if available
            if "metadata" in gen_data and "startup_cost" in gen_data["metadata"]:
                self.startup_cost[g] = gen_data["metadata"]["startup_cost"]
            else:
                self.startup_cost[g] = 0.0

        # Calculate penalty cost as multiple of maximum generation cost
        max_gen_cost = max(self.unit_cost.values()) if self.unit_cost else 100.0
        self.penalty_cost = 10 * max_gen_cost

        print(f"Loaded UC instance: {self.nG} thermal gens, {self.T} periods")

        # Initialize precomputed matrices for fast dual evaluation
        self._initialize_coefficient_matrices()

        # Initialize dual storage now that dimensions are known
        self._initialize_dual_storage()

        return 0

    @staticmethod
    def create_reduced_period_instance(input_filename, output_filename, new_periods):
        """
        Create a new instance file with reduced number of time periods.

        Args:
            input_filename (str): Path to original JSON instance file
            output_filename (str): Path for new reduced instance file
            new_periods (int): Number of periods to keep (from the beginning)
        """
        import json
        import os

        # Load original instance
        with open(input_filename, "r") as f:
            data = json.load(f)

        original_periods = data["time_periods"]

        if new_periods >= original_periods:
            print(
                f"Warning: Requested periods ({new_periods}) >= original periods ({original_periods}). No reduction needed."
            )
            return input_filename

        if new_periods <= 0:
            raise ValueError("Number of periods must be positive")

        print(f"Reducing instance from {original_periods} to {new_periods} periods")

        # Create new data structure
        new_data = data.copy()

        # Update time periods
        new_data["time_periods"] = new_periods

        # Truncate demand data
        new_data["demand"] = data["demand"][:new_periods]

        # Truncate reserves data if present
        if "reserves" in data:
            new_data["reserves"] = data["reserves"][:new_periods]

        # Handle renewable generators if present
        if "renewable_generators" in data:
            new_renewable_gens = {}
            for gen_name, gen_data in data["renewable_generators"].items():
                new_gen_data = gen_data.copy()
                # Truncate time-series data
                if "power_output_minimum" in gen_data:
                    new_gen_data["power_output_minimum"] = gen_data[
                        "power_output_minimum"
                    ][:new_periods]
                if "power_output_maximum" in gen_data:
                    new_gen_data["power_output_maximum"] = gen_data[
                        "power_output_maximum"
                    ][:new_periods]
                new_renewable_gens[gen_name] = new_gen_data
            new_data["renewable_generators"] = new_renewable_gens

        # Thermal generators don't need period-specific changes since they have static parameters

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)

        # Save reduced instance
        with open(output_filename, "w") as f:
            json.dump(new_data, f, indent=2)

        print(f"Reduced instance saved to: {output_filename}")
        return output_filename

    @staticmethod
    def create_reduced_generator_instance(
        input_filename_or_data,
        num_generators,
    ):
        """
        Create reduced instance data with fewer thermal generators and scaled demand.

        Randomly selects a subset of thermal generators to keep (using fixed seed 42),
        scales demand proportionally to the lost thermal capacity, and keeps all
        renewable generators unchanged.

        Args:
            input_filename_or_data (str or dict): Path to JSON file or data dictionary
            num_generators (int): Number of thermal generators to keep

        Returns:
            dict: Reduced instance data
        """
        import json
        import random

        # Load original instance
        if isinstance(input_filename_or_data, dict):
            data = input_filename_or_data
        else:
            with open(input_filename_or_data, "r") as f:
                data = json.load(f)

        # Get thermal generator info
        thermal_gens = list(data["thermal_generators"].keys())
        original_num_gens = len(thermal_gens)

        if num_generators >= original_num_gens:
            print(
                f"Warning: Requested generators ({num_generators}) >= original generators ({original_num_gens}). No reduction needed."
            )
            return data

        if num_generators <= 0:
            raise ValueError("num_generators must be positive")

        print(
            f"Reducing from {original_num_gens} to {num_generators} thermal generators"
        )

        # Calculate original total thermal capacity
        original_capacity = sum(
            gen_data["power_output_maximum"]
            for gen_data in data["thermal_generators"].values()
        )

        # Randomly select generators to keep (fixed seed for reproducibility)
        random.seed(42)
        generators_to_keep = random.sample(thermal_gens, num_generators)
        generators_to_keep_set = set(generators_to_keep)

        print(f"Keeping generators: {sorted(generators_to_keep)}")

        # Calculate new thermal capacity
        new_capacity = sum(
            gen_data["power_output_maximum"]
            for gen_name, gen_data in data["thermal_generators"].items()
            if gen_name in generators_to_keep_set
        )

        # Calculate demand scaling factor
        scaling_factor = new_capacity / original_capacity

        print(
            f"Original thermal capacity: {original_capacity:.2f}, "
            f"New thermal capacity: {new_capacity:.2f}, "
            f"Scaling factor: {scaling_factor:.4f}"
        )

        # Verify feasibility
        peak_demand = max(data["demand"])
        scaled_peak_demand = peak_demand * scaling_factor
        if new_capacity < scaled_peak_demand:
            print(
                f"Warning: New capacity ({new_capacity:.2f}) may be insufficient "
                f"for scaled peak demand ({scaled_peak_demand:.2f})"
            )

        # Create new data structure
        new_data = data.copy()

        # Filter thermal generators
        new_thermal_gens = {
            gen_name: gen_data
            for gen_name, gen_data in data["thermal_generators"].items()
            if gen_name in generators_to_keep_set
        }
        new_data["thermal_generators"] = new_thermal_gens

        # Scale demand
        new_data["demand"] = [d * scaling_factor for d in data["demand"]]

        # Scale reserves if present
        if "reserves" in data:
            new_data["reserves"] = [r * scaling_factor for r in data["reserves"]]

        # Keep renewable generators unchanged (if present)
        if "renewable_generators" in data:
            new_data["renewable_generators"] = data["renewable_generators"]
            print(
                f"Keeping all {len(data['renewable_generators'])} renewable generators unchanged"
            )

        return new_data

    def _initialize_coefficient_matrices(self):
        """Initialize precomputed coefficient matrices for fast dual evaluation."""
        # Precompute min and max power coefficient vectors
        self.min_power_matrix = np.array(
            [self.min_power[g] for g in self.thermal_gens for t in self.periods],
            dtype=np.float64,
        )

        self.max_power_matrix = np.array(
            [self.max_power[g] for g in self.thermal_gens for t in self.periods],
            dtype=np.float64,
        )

        # Combined coefficient matrix for vectorized operations
        self.power_coefficient_matrix = np.concatenate(
            [self.min_power_matrix, self.max_power_matrix]
        )

    def _initialize_dual_storage(self):
        """Initialize atomic dual storage after dimensions are known."""
        if self.nS is None:
            # nS will be set later when scenarios are generated
            print("Warning: nS not set yet, dual storage will be initialized later")
            return

        print(f"Initializing dual storage: nG={self.nG}, T={self.T}, nS={self.nS}")
        self.dual_storage = AtomicDualStorage(self.nG, self.T, self.nS)

        # Initialize legacy compatibility properties
        self._dual_obj_random_needs_update = True

    def ensure_dual_storage_initialized(self):
        """Ensure dual storage is initialized (call this after scenarios are generated)."""
        if self.dual_storage is None and self.nS is not None:
            self._initialize_dual_storage()

    def get_commitment_weighted_solution(self, commitment_solution):
        """
        Efficiently compute commitment-weighted solution vector for dual evaluation.

        Args:
            commitment_solution (np.array): Commitment decisions as flat array

        Returns:
            np.array: Weighted solution vector [min_weighted + max_weighted]
        """
        # Convert to numpy array if needed
        if not isinstance(commitment_solution, np.ndarray):
            commitment_solution = np.array(commitment_solution)

        # Convert to tuple for hashing
        cache_key = tuple(commitment_solution)

        # Check cache first
        # if cache_key in self.commitment_solution_cache:
        #     return self.commitment_solution_cache[cache_key]

        # Compute weighted solution using vectorized operations
        min_weighted = self.min_power_matrix * commitment_solution
        max_weighted = self.max_power_matrix * commitment_solution
        weighted_solution = np.concatenate([min_weighted, max_weighted])

        # Cache the result
        self.commitment_solution_cache[cache_key] = weighted_solution
        return weighted_solution

    def clear_caches(self):
        """Clear all caches to free memory. Call when switching SAA iterations."""
        self.commitment_solution_cache.clear()
        self.dual_matrix_cache.clear()

    def generate_demand_scenarios(self, nS, demand_std_dev=0.05, save_filename=None):
        """
        Generate demand scenarios using normal distribution around base demand.
        If nS=1, uses base demand directly (deterministic case).

        Args:
            nS (int): Number of scenarios to generate
            demand_std_dev (float): Standard deviation as fraction of base demand
            save_filename (str, optional): File to save scenarios
        """
        self.nS = int(nS)
        self.scenario = list(range(self.nS))
        self.probability = {s: 1.0 / self.nS for s in self.scenario}

        # Generate demand scenarios
        self.demand_scenarios = {}

        if self.nS == 1:
            # Deterministic case: use base demand directly
            self.demand_scenarios[0] = {}
            for t in self.periods:
                self.demand_scenarios[0][t] = self.demand_base[t - 1]
            print("Generated 1 scenario using base demand (deterministic)")
        else:
            # Stochastic case: generate random scenarios
            for s in self.scenario:
                self.demand_scenarios[s] = {}
                for t in self.periods:
                    base_demand = self.demand_base[t - 1]
                    scenario_demand = np.random.normal(
                        base_demand, demand_std_dev * base_demand
                    )
                    self.demand_scenarios[s][t] = max(
                        scenario_demand, 0.1 * base_demand
                    )
            print(f"Generated {nS} demand scenarios with std dev {demand_std_dev}")

        # Initialize scenario-specific subproblem data
        self.scen_sp_data = {s: {} for s in self.scenario}
        for s in self.scenario:
            for t in self.periods:
                self.scen_sp_data[s][t] = self.demand_scenarios[s][t]

        # Create precomputed demand scenarios matrix
        self._create_demand_scenarios_matrix()

        # Initialize dual storage now that nS is known
        self.ensure_dual_storage_initialized()

        # Save scenarios if filename provided
        if save_filename:
            self.write_scenarios_to_file(save_filename)

        return 0

    def write_scenarios_to_file(self, filename):
        """Write generated scenarios to file for persistence."""
        scenario_data = {
            "demand_scenarios": self.demand_scenarios,
        }

        with open(filename, "w") as f:
            json.dump(scenario_data, f, indent=2)

    def read_scenarios(self, filename):
        """Read scenarios from file."""
        with open(filename, "r") as f:
            scenario_data = json.load(f)

        self.demand_scenarios = scenario_data["demand_scenarios"]
        self.nS = len(self.demand_scenarios)
        self.scenario = list(range(self.nS))
        self.probability = {s: 1.0 / self.nS for s in self.scenario}

        # Initialize scenario-specific subproblem data
        self.scen_sp_data = {s: {} for s in self.scenario}
        for s in self.scenario:
            for t in self.periods:
                self.scen_sp_data[s][t] = self.demand_scenarios[s][t]

        print(f"Loaded {self.nS} scenarios from {filename}")

        # Create precomputed demand scenarios matrix
        self._create_demand_scenarios_matrix()

        # Initialize dual storage now that nS is known
        self.ensure_dual_storage_initialized()

        return 0

    def _create_demand_scenarios_matrix(self):
        """
        Create precomputed demand scenarios matrix for fast vectorized operations.

        Creates a matrix of shape (nS, T) where entry [s, t-1] contains
        the demand for scenario s at time period t.
        """
        if not self.demand_scenarios or not self.periods:
            self.demand_scenarios_matrix = None
            return

        # Create matrix: scenarios x time periods
        self.demand_scenarios_matrix = np.zeros((self.nS, self.T), dtype=np.float64)

        for s in self.scenario:
            for t_idx, t in enumerate(self.periods):
                self.demand_scenarios_matrix[s, t_idx] = self.demand_scenarios[s][t]

        print(
            f"Created demand scenarios matrix: shape {self.demand_scenarios_matrix.shape}"
        )

    def build_master(self, relaxation=False):
        """
        Build the master problem for Benders decomposition.

        Master problem contains:
        - Binary commitment variables x[g,t]
        - Minimum up/down time constraints
        - Second-stage approximation variables z[s]

        Args:
            relaxation (bool): If True, relax binary variables to continuous [0,1]

        Returns:
            gp.Model: Master problem model
        """
        master = gp.Model("UC_Master")

        # Commitment variables - include period 0 for initial conditions
        if relaxation:
            self.x = master.addVars(
                self.thermal_gens,
                range(0, self.T + 1),
                lb=0.0,
                ub=1.0,
                obj=0.0,  # Startup/shutdown costs would go here
                name="commitment",
            )
        else:
            self.x = master.addVars(
                self.thermal_gens,
                range(0, self.T + 1),
                vtype=GRB.BINARY,
                obj=0.0,
                name="commitment",
            )

        # Fix initial conditions (generators start offline)
        for g in self.thermal_gens:
            master.addConstr(self.x[g, 0] == 0, name=f"init_{g}")

        # Start-up and shut-down variables for state transitions (with startup costs)
        self.w = master.addVars(
            self.thermal_gens,
            self.periods,
            vtype=GRB.BINARY if not relaxation else GRB.CONTINUOUS,
            obj={
                (g, t): self.startup_cost[g]
                for g in self.thermal_gens
                for t in self.periods
            },
            name="startup",
        )

        self.v = master.addVars(
            self.thermal_gens,
            self.periods,
            vtype=GRB.BINARY if not relaxation else GRB.CONTINUOUS,
            obj=0.0,  # No shutdown cost
            name="shutdown",
        )

        # Second-stage approximation variables
        self.z = master.addVars(
            self.scenario,
            obj={s: self.probability[s] for s in self.scenario},
            name="second_stage_cost",
        )

        # State transition constraints: x[g,t] = x[g,t-1] + w[g,t] - v[g,t]
        for g in self.thermal_gens:
            for t in self.periods:
                master.addConstr(
                    self.x[g, t] == self.x[g, t - 1] + self.w[g, t] - self.v[g, t],
                    name=f"state_{g}_{t}",
                )

        # Minimum up time constraints
        for g in self.thermal_gens:
            min_up = self.min_up_time[g]
            for t in self.periods:
                if min_up > 1:
                    start_period = max(1, t - min_up + 1)
                    master.addConstr(
                        gp.quicksum(self.w[g, s] for s in range(start_period, t + 1))
                        <= self.x[g, t],
                        name=f"min_up_{g}_{t}",
                    )

        # Minimum down time constraints
        for g in self.thermal_gens:
            min_down = self.min_down_time[g]
            for t in self.periods:
                if min_down > 1:
                    start_period = max(1, t - min_down + 1)
                    master.addConstr(
                        gp.quicksum(self.v[g, s] for s in range(start_period, t + 1))
                        <= 1 - self.x[g, t],
                        name=f"min_down_{g}_{t}",
                    )

        master.modelSense = GRB.MINIMIZE

        # Store variable ordering for solution extraction
        self.sorted_commitment_vars = []
        for g in self.thermal_gens:
            for t in self.periods:
                self.sorted_commitment_vars.append(self.x[g, t])

        # Initialize cut tracking
        self.lp_cuts = {s: set() for s in self.scenario}

        return master

    def build_SP(self):
        """
        Build the subproblem for economic dispatch given commitments.

        Subproblem contains:
        - Power output variables p[g,t] for thermal generators
        - Demand shortfall variables slack[t]
        - Generation bounds based on commitment decisions
        - Demand satisfaction constraints

        Returns:
            gp.Model: Subproblem model
        """
        subproblem = gp.Model("UC_Subproblem")

        # Thermal generation variables
        self.p = subproblem.addVars(
            self.thermal_gens,
            self.periods,
            lb=0.0,
            obj={
                (g, t): self.unit_cost[g] * self.cost_scale
                for g in self.thermal_gens
                for t in self.periods
            },
            name="thermal_power",
        )

        # Demand shortfall variables (penalty for unmet demand)
        self.slack = subproblem.addVars(
            self.periods,
            lb=0.0,
            obj={t: self.penalty_cost * self.cost_scale for t in self.periods},
            name="demand_shortfall",
        )

        # Generation limit constraints (will be updated with commitment values)
        self.gen_min_constrs = {}
        self.gen_max_constrs = {}

        for g in self.thermal_gens:
            for t in self.periods:
                # Minimum generation when committed
                self.gen_min_constrs[g, t] = subproblem.addConstr(
                    self.p[g, t] >= 0.0,  # Will be updated to min_power[g] * x[g,t]
                    name=f"gen_min_{g}_{t}",
                )

                # Maximum generation when committed
                self.gen_max_constrs[g, t] = subproblem.addConstr(
                    self.p[g, t] <= 0.0,  # Will be updated to max_power[g] * x[g,t]
                    name=f"gen_max_{g}_{t}",
                )

        # Demand satisfaction constraints (will be updated with scenario data)
        self.demand_constrs = {}
        for t in self.periods:
            lhs = gp.quicksum(self.p[g, t] for g in self.thermal_gens) + self.slack[t]

            self.demand_constrs[t] = subproblem.addConstr(
                lhs >= 0.0, name=f"demand_{t}"  # Will be updated with scenario demand
            )

        subproblem.modelSense = GRB.MINIMIZE
        return subproblem

    def build_full_subproblem(self, scenario_idx, commitment_values):
        """
        Build a complete subproblem for a specific scenario with all constraints.

        Args:
            scenario_idx (int): Scenario index
            commitment_values (dict): Commitment decisions {(g,t): value}

        Returns:
            gp.Model: Complete subproblem model for the scenario
        """
        subproblem = gp.Model(f"UC_Subproblem_Scenario_{scenario_idx}")

        # Thermal generation variables
        p = subproblem.addVars(
            self.thermal_gens,
            self.periods,
            lb=0.0,
            obj={
                (g, t): self.unit_cost[g] * self.cost_scale
                for g in self.thermal_gens
                for t in self.periods
            },
            name="thermal_power",
        )

        # Demand shortfall variables (penalty for unmet demand)
        slack = subproblem.addVars(
            self.periods,
            lb=0.0,
            obj={t: self.penalty_cost * self.cost_scale for t in self.periods},
            name="demand_shortfall",
        )

        self.gen_min_constrs = {}
        self.gen_max_constrs = {}
        # Generation limit constraints based on commitment values
        for g in self.thermal_gens:
            for t in self.periods:
                x_val = commitment_values.get((g, t), 0.0)

                # Minimum generation when committed
                self.gen_min_constrs[g, t] = subproblem.addConstr(
                    p[g, t] >= self.min_power[g] * x_val,
                    name=f"gen_min_{g}_{t}",
                )

                # Maximum generation when committed
                self.gen_max_constrs[g, t] = subproblem.addConstr(
                    p[g, t] <= self.max_power[g] * x_val,
                    name=f"gen_max_{g}_{t}",
                )

        # Demand satisfaction constraints for the specific scenario
        self.demand_constrs = {}
        for t in self.periods:
            lhs = gp.quicksum(p[g, t] for g in self.thermal_gens) + slack[t]

            self.demand_constrs[t] = subproblem.addConstr(
                lhs >= self.demand_scenarios[scenario_idx][t], name=f"demand_{t}"
            )

        subproblem.modelSense = GRB.MINIMIZE

        # Store variable references for easy access
        subproblem._thermal_power = p
        subproblem._demand_shortfall = slack

        return subproblem

    def update_commitment(self, model, commitment_values):
        """
        Update subproblem generation bounds based on commitment decisions.

        Args:
            model: Subproblem model to update
            commitment_values: Dictionary of commitment values {(g,t): value}
        """
        # Update generation bounds based on commitments
        for g in self.thermal_gens:
            for t in self.periods:
                x_val = commitment_values.get((g, t), 0.0)

                # Update minimum generation constraint
                model.setAttr(
                    "RHS", self.gen_min_constrs[g, t], self.min_power[g] * x_val
                )

                # Update maximum generation constraint
                model.setAttr(
                    "RHS", self.gen_max_constrs[g, t], self.max_power[g] * x_val
                )

        # Update model to apply constraint changes
        model.update()

    def update_scenario(self, model, scenario_idx):
        """
        Update subproblem with scenario-specific data.

        Args:
            model: Subproblem model to update
            scenario_idx: Scenario index
        """
        # Update demand constraints
        for t in self.periods:
            model.setAttr(
                "RHS", self.demand_constrs[t], self.demand_scenarios[scenario_idx][t]
            )

        # Update model to apply changes
        model.update()

    def extract_solution_values(self, problem="IP"):
        """
        Extract first-stage and second-stage variable values from master solution.

        Args:
            problem (str): "IP" for integer solution, "LP" for linear relaxation
        """
        # Extract commitment values
        self.first_stage_values = {}
        for g in self.thermal_gens:
            for t in self.periods:
                val = self.master.getVarByName(f"commitment[{g},{t}]").x
                if problem == "IP":
                    self.first_stage_values[g, t] = 1.0 if val > 0.5 else 0.0
                else:
                    self.first_stage_values[g, t] = max(val, 0.0)

        # Extract startup variable values (w[g,t])
        self.startup_values = {}
        for g in self.thermal_gens:
            for t in self.periods:
                val = self.master.getVarByName(f"startup[{g},{t}]").x
                if problem == "IP":
                    self.startup_values[g, t] = 1.0 if val > 0.5 else 0.0
                else:
                    self.startup_values[g, t] = max(val, 0.0)

        # Extract second-stage approximation values
        self.second_stage_values = {}
        for s in self.scenario:
            self.second_stage_values[s] = self.master.getVarByName(
                f"second_stage_cost[{s}]"
            ).x

    def calculate_startup_cost(self, startup_values=None, commitment_values=None):
        """
        Calculate total startup cost for a given solution.

        Args:
            startup_values: Dictionary {(g,t): value} of startup variable values (preferred)
            commitment_values: Dictionary {(g,t): value} or array of commitment values (fallback)

        Returns:
            float: Total startup cost
        """
        # Prefer using startup values directly (from w variables) - more accurate
        if startup_values is not None:
            if isinstance(startup_values, dict):
                return sum(
                    self.startup_cost[g] * startup_values.get((g, t), 0.0)
                    for g in self.thermal_gens
                    for t in self.periods
                )
            else:
                # Array format
                total_startup_cost = 0.0
                idx = 0
                for g in self.thermal_gens:
                    for t in self.periods:
                        total_startup_cost += self.startup_cost[g] * startup_values[idx]
                        idx += 1
                return total_startup_cost

        # Fallback: compute startups from commitment transitions
        if commitment_values is not None:
            if isinstance(commitment_values, dict):
                total_startup_cost = 0.0
                for g in self.thermal_gens:
                    for t in self.periods:
                        x_curr = commitment_values.get((g, t), 0.0)
                        x_prev = (
                            commitment_values.get((g, t - 1), 0.0) if t > 1 else 0.0
                        )
                        # Startup occurs when x goes from 0 to 1
                        startup = max(0, x_curr - x_prev)
                        total_startup_cost += self.startup_cost[g] * startup
                return total_startup_cost
            else:
                # Array format: convert to dict first
                commitment_dict = {}
                idx = 0
                for g in self.thermal_gens:
                    for t in self.periods:
                        commitment_dict[g, t] = commitment_values[idx]
                        idx += 1
                return self.calculate_startup_cost(commitment_values=commitment_dict)

        return 0.0

    def calculate_upper_bound_from_subproblems(self):
        """
        Calculate upper bound using current first-stage solution and subproblem values.

        Returns:
            float: Total upper bound objective value
        """
        # First-stage startup costs - use startup_values (w variables) if available
        if hasattr(self, "startup_values") and self.startup_values:
            first_stage_cost = self.calculate_startup_cost(
                startup_values=self.startup_values
            )
        else:
            first_stage_cost = self.calculate_startup_cost(
                commitment_values=self.first_stage_values
            )

        # Expected second-stage cost
        expected_second_stage_cost = sum(
            self.scenario_upper_bounds[s] * self.probability[s] for s in self.scenario
        )

        return first_stage_cost + expected_second_stage_cost

    def generate_benders_cuts(self, model, lazy=False, first=False):
        """
        Generate Benders optimality cuts by solving subproblems for all scenarios.

        Args:
            model: Master optimization model to add cuts to
            lazy (bool): If True, add cuts as lazy constraints
            first (bool): Flag for first iteration handling

        Returns:
            tuple: (num_cuts_added, subproblem_objective_values, total_work_units)
        """
        num_cuts_added = 0
        subproblem_objective_values = []
        total_work_units = 0

        # Update subproblem with current first-stage solution
        # self.subproblem = self.build_SP()
        self.update_commitment(self.subproblem, self.first_stage_values)

        for scenario in self.scenario:
            # Update scenario-specific data
            # self.subproblem = self.build_full_subproblem(
            #     scenario, self.first_stage_values
            # )
            self.update_scenario(self.subproblem, scenario)
            self.subproblem.setParam("OutputFlag", False)
            self.subproblem.optimize()

            # Track work units for this subproblem solve
            total_work_units += self.subproblem.getAttr("Work")

            status = self.subproblem.status
            if status != 2:
                raise Exception(f"Subproblem status - {status}")

            subproblem_objective = self.subproblem.ObjVal
            self.scenario_upper_bounds[scenario] = subproblem_objective
            subproblem_objective_values.append(subproblem_objective)

            # Check if cut should be added
            if subproblem_objective - self.second_stage_values[scenario] > max(
                self.tol, 0.00001 * abs(self.second_stage_values[scenario])
            ):

                # Extract dual values as returned by Gurobi for the given constraint senses
                gen_min_duals = self.subproblem.getAttr("Pi", self.gen_min_constrs)
                gen_max_duals = self.subproblem.getAttr("Pi", self.gen_max_constrs)
                demand_duals = self.subproblem.getAttr("Pi", self.demand_constrs)

                # Create hash directly from ordered values (faster than dict with tuple keys)
                # Order: gen_min, gen_max, demand (consistent ordering ensures uniqueness)
                dual_values = (
                    tuple(
                        gen_min_duals[g, t]
                        for g in self.thermal_gens
                        for t in self.periods
                    )
                    + tuple(
                        gen_max_duals[g, t]
                        for g in self.thermal_gens
                        for t in self.periods
                    )
                    + tuple(demand_duals[t] for t in self.periods)
                )
                dual_hash = hash(dual_values)
                is_new_dual = False

                # Check if this dual solution is new using AtomicDualStorage
                if self.dual_storage.get_index_by_hash(dual_hash) is None:
                    # Store new dual solution
                    gen_min_ordered = [
                        gen_min_duals[g, t]
                        for g in self.thermal_gens
                        for t in self.periods
                    ]
                    gen_max_ordered = [
                        gen_max_duals[g, t]
                        for g in self.thermal_gens
                        for t in self.periods
                    ]
                    demand_ordered = [demand_duals[t] for t in self.periods]

                    # Ensure dual storage is initialized
                    self.ensure_dual_storage_initialized()

                    # Add dual solution to atomic storage
                    dual_id = self.dual_storage.add_dual_solution(
                        gen_min_ordered + gen_max_ordered,
                        demand_ordered,
                        dual_hash,
                    )
                    is_new_dual = True
                    self.dual_soln_optimal_counter[dual_id] = 0
                else:
                    # Get existing dual ID from storage
                    dual_id = self.dual_storage.get_index_by_hash(dual_hash)
                    if (
                        not first
                        and self.dual_pool_size
                        and dual_id < self.dual_pool_size[-1]
                    ):
                        self.dual_soln_optimal_counter[dual_id] += 1

                # Calculate cut violation and add if significant
                if self._construct_uc_benders_cut(
                    scenario,
                    gen_min_duals,
                    gen_max_duals,
                    demand_duals,
                    model,
                    lazy,
                    dual_id if not lazy else None,
                ):
                    num_cuts_added += 1

        return num_cuts_added, subproblem_objective_values, total_work_units

    def _construct_uc_benders_cut(
        self,
        scenario,
        gen_min_duals,
        gen_max_duals,
        demand_duals,
        model,
        lazy=False,
        dual_id=None,
    ):
        """
        Construct and add a Benders optimality cut for unit commitment.

        The cut (from LP duality) has the form:
        z[s] >= ∑_{g,t} [min_power[g] * (-Pi_min[g,t]) - max_power[g] * Pi_max[g,t]] * x[g,t]
                + ∑_t demand[s,t] * (-Pi_demand[t])
        where Pi_min is the dual for p >= min*x (≤ 0),
              Pi_max is the dual for p <= max*x (≥ 0),
              Pi_demand is the dual for supply >= demand (≤ 0).

        Args:
            scenario: Scenario index
            gen_min_duals, gen_max_duals, demand_duals: Dual values (Pi) for constraints
            model: Master model to add cut to
            lazy: Whether to add as lazy constraint
            dual_id: Dual solution identifier

        Returns:
            bool: True if cut was added
        """
        # Calculate cut violation to determine if cut should be added
        generation_contribution = 0.0
        for g in self.thermal_gens:
            for t in self.periods:
                x_val = self.first_stage_values.get((g, t), 0.0)
                dual_coeff = (
                    self.min_power[g] * gen_min_duals[g, t]
                    + self.max_power[g] * gen_max_duals[g, t]
                )
                generation_contribution += dual_coeff * x_val

        demand_contribution = sum(
            self.demand_scenarios[scenario][t] * demand_duals[t] for t in self.periods
        )

        total_contribution = generation_contribution + demand_contribution
        violation = total_contribution - self.second_stage_values[scenario]

        # Only add cut if violation is significant
        if violation <= self.tol:
            return False

        # Construct the cut expression
        cut_expr = gp.quicksum(
            (
                self.min_power[g] * gen_min_duals[g, t]
                + self.max_power[g] * gen_max_duals[g, t]
            )
            * self.x[g, t]
            for g in self.thermal_gens
            for t in self.periods
        ) + gp.quicksum(
            self.demand_scenarios[scenario][t] * demand_duals[t] for t in self.periods
        )

        cut_expr = self.z[scenario] >= cut_expr

        # Add the cut to the model
        if lazy:
            model.cbLazy(cut_expr)
        else:
            model.addConstr(
                cut_expr, name=f"benders_cut_{scenario}_{len(self.dual_storage)}"
            )
            if dual_id is not None:
                self.lp_cuts[scenario].add(dual_id)

        return True

    def optimal_value_duals(self, commitment_values, get_duals=False):
        """
        Evaluate value function V(x) for given first-stage commitment solution.

        Args:
            commitment_values (array or dict): First-stage commitment decisions
            get_duals (bool): If True, return optimal dual solution indices

        Returns:
            tuple: (objective_value, [dual_indices], [subproblem_objectives])
        """
        if isinstance(commitment_values, dict):
            # Convert dict to array format for consistency
            commitment_array = []
            for g in self.thermal_gens:
                for t in self.periods:
                    commitment_array.append(commitment_values[g, t])
            commitment_values = np.array(commitment_array)

        # Build commitment dict for startup cost calculation
        commitment_dict = {}
        idx = 0
        for g in self.thermal_gens:
            for t in self.periods:
                commitment_dict[g, t] = commitment_values[idx]
                idx += 1

        # First-stage startup costs computed from commitment transitions
        objective_value = self.calculate_startup_cost(commitment_values=commitment_dict)

        if get_duals:
            optimal_dual_solutions = []
        subproblem_objective_values = []
        subproblem_bounds = []
        total_work_units = 0

        # Update subproblem with current commitment values (reuse subproblem like CFLP)
        self.update_commitment(self.subproblem, commitment_dict)

        for scenario in self.scenario:
            # Update scenario data (reuse subproblem like CFLP)
            self.update_scenario(self.subproblem, scenario)
            self.subproblem.setParam("OutputFlag", False)
            self.subproblem.optimize()

            # Track work units for this subproblem solve
            total_work_units += self.subproblem.getAttr("Work")

            status = self.subproblem.status
            if status != 2:
                raise Exception(f"Subproblem status - {status}")

            subproblem_objective = self.subproblem.ObjVal
            subproblem_bound = self.subproblem.ObjBound
            objective_value += self.probability[scenario] * subproblem_objective
            subproblem_objective_values.append(subproblem_objective)
            subproblem_bounds.append(subproblem_bound)

            if get_duals:
                # Extract and store dual solution
                gen_min_duals = self.subproblem.getAttr("Pi", self.gen_min_constrs)
                gen_max_duals = self.subproblem.getAttr("Pi", self.gen_max_constrs)
                demand_duals = self.subproblem.getAttr("Pi", self.demand_constrs)

                # Create hash directly from ordered values (faster than dict with tuple keys)
                dual_values = (
                    tuple(
                        gen_min_duals[g, t]
                        for g in self.thermal_gens
                        for t in self.periods
                    )
                    + tuple(
                        gen_max_duals[g, t]
                        for g in self.thermal_gens
                        for t in self.periods
                    )
                    + tuple(demand_duals[t] for t in self.periods)
                )
                dual_hash = hash(dual_values)

                # Calculate and print the left-hand side of the Benders cut for debugging
                generation_contribution = 0.0
                for g in self.thermal_gens:
                    for t in self.periods:
                        x_val = commitment_dict[g, t]
                        dual_coeff = (
                            self.min_power[g] * gen_min_duals[g, t]
                            + self.max_power[g] * gen_max_duals[g, t]
                        )
                        generation_contribution += dual_coeff * x_val

                demand_contribution = sum(
                    self.demand_scenarios[scenario][t] * demand_duals[t]
                    for t in self.periods
                )

                cut_lhs_value = generation_contribution + demand_contribution

                # Check if this dual solution is new using AtomicDualStorage
                if self.dual_storage.get_index_by_hash(dual_hash) is None:
                    # Store new dual solution
                    gen_min_ordered = [
                        gen_min_duals[g, t]
                        for g in self.thermal_gens
                        for t in self.periods
                    ]
                    gen_max_ordered = [
                        gen_max_duals[g, t]
                        for g in self.thermal_gens
                        for t in self.periods
                    ]
                    demand_ordered = [demand_duals[t] for t in self.periods]

                    # Ensure dual storage is initialized
                    self.ensure_dual_storage_initialized()

                    # Add dual solution to atomic storage
                    dual_id = self.dual_storage.add_dual_solution(
                        gen_min_ordered + gen_max_ordered,
                        demand_ordered,
                        dual_hash,
                    )
                    self.dual_soln_optimal_counter[dual_id] = 0
                    optimal_dual_solutions.append(dual_id)
                elif get_duals:
                    # Get existing dual ID from storage
                    dual_idx = self.dual_storage.get_index_by_hash(dual_hash)
                    self.dual_soln_optimal_counter[dual_idx] += 1
                    optimal_dual_solutions.append(dual_idx)

        if get_duals:
            return (
                objective_value,
                np.array(optimal_dual_solutions),
                subproblem_objective_values,
            )
        else:
            return objective_value, subproblem_objective_values

    def prepare_dual_arrays(self):
        """
        Prepare numpy arrays for fast dual solution pooling evaluation using AtomicDualStorage.
        Should be called after collecting sufficient dual solutions.
        """
        # Ensure dual storage is initialized
        self.ensure_dual_storage_initialized()

        if len(self.dual_storage) == 0:
            return

        # Get arrays from atomic storage
        (
            self.generation_duals_array,
            self.demand_duals_array,
        ) = self.dual_storage.get_arrays()

        # Update precomputed dual objective values
        self.dual_storage.update_dual_obj_random(self.demand_scenarios_matrix)

        # Get the updated dual objectives
        self.dual_obj_random = self.dual_storage.get_dual_obj_random()

        print(f"Prepared dual arrays: {len(self.dual_storage)} dual solutions")

    def prepare_dual_arrays_incremental(self, old_size):
        """
        Incrementally update dual arrays for newly added dual solutions only.
        This is much faster than prepare_dual_arrays when only a few duals are added.

        Args:
            old_size (int): Previous size of dual storage before new duals were added
        """
        # Ensure dual storage is initialized
        self.ensure_dual_storage_initialized()

        if len(self.dual_storage) == 0 or old_size >= len(self.dual_storage):
            return

        # Get arrays from atomic storage (this is a fast view operation)
        (
            self.generation_duals_array,
            self.demand_duals_array,
        ) = self.dual_storage.get_arrays()

        # INCREMENTAL: Only update dual objectives for NEW duals (from old_size to current size)
        self.dual_storage.update_dual_obj_random_incremental(
            self.demand_scenarios_matrix, old_size
        )

        # Get the updated dual objectives (view includes all duals)
        self.dual_obj_random = self.dual_storage.get_dual_obj_random()

        num_new_duals = len(self.dual_storage) - old_size
        print(
            f"Incrementally updated dual arrays: added {num_new_duals} new duals (total: {len(self.dual_storage)})"
        )

    def evaluate_subproblems_with_dual_list(
        self, commitment_solution, dual_solution_indices
    ):
        """
        Fast evaluation using specified dual solutions across all scenarios.

        Args:
            commitment_solution (np.array): First-stage commitment decisions
            dual_solution_indices (list): List of dual solution indices to evaluate

        Returns:
            tuple: (scenario_optimal_values, optimal_dual_ids)
        """
        if self.dual_obj_random is None:
            self.prepare_dual_arrays()

        # Use cached commitment-weighted solution for efficiency
        commitment_weighted = self.get_commitment_weighted_solution(commitment_solution)

        # Calculate commitment-dependent dual contributions using vectorized operations
        selected_generation_duals = self.generation_duals_array[dual_solution_indices]
        commitment_dual_contributions = np.dot(
            selected_generation_duals, commitment_weighted
        )

        # Use numba-optimized function to find best dual for each scenario
        scenario_optimal_values, relative_dual_indices = find_largest_index_numba_uc(
            commitment_dual_contributions,
            self.dual_obj_random[dual_solution_indices, :],
        )

        # Convert relative indices back to original dual solution IDs
        optimal_dual_ids = [
            dual_solution_indices[relative_idx]
            for relative_idx in relative_dual_indices
        ]

        return scenario_optimal_values, optimal_dual_ids

    def evaluate_subproblems_with_selected_duals(
        self, commitment_solution, scenario_dual_selection, return_optimal_duals=False
    ):
        """
        Evaluate subproblems using scenario-specific dual solution selections.

        Args:
            commitment_solution (np.array): First-stage commitment decisions
            scenario_dual_selection (dict): {scenario_id: [dual_id1, dual_id2, ...]}
            return_optimal_duals (bool): If True, return optimal dual indices

        Returns:
            list or tuple: Upper bounds [and optimal dual indices]
        """
        if self.dual_obj_random is None:
            self.prepare_dual_arrays()

        assert isinstance(
            scenario_dual_selection, dict
        ), "scenario_dual_selection must be a dictionary"

        # Precompute all commitment-dual contributions using vectorized operations
        all_dual_ids = set()
        for dual_list in scenario_dual_selection.values():
            all_dual_ids.update(dual_list)
        all_dual_ids = list(all_dual_ids)

        # Use cached commitment-weighted solution
        commitment_weighted = self.get_commitment_weighted_solution(commitment_solution)

        # Vectorized computation of all dual contributions
        selected_generation_duals = self.generation_duals_array[all_dual_ids]
        all_contributions = np.dot(selected_generation_duals, commitment_weighted)

        # Create dictionary mapping dual_idx to contribution
        commitment_dual_contributions = {
            dual_idx: all_contributions[i] for i, dual_idx in enumerate(all_dual_ids)
        }

        optimal_dual_indices = []
        scenario_upper_bounds = []

        for scenario_idx, scenario in enumerate(self.scenario):
            selected_dual_ids = list(scenario_dual_selection[scenario])

            # Evaluate dual objective for selected duals
            dual_objective_values = [
                commitment_dual_contributions[dual_id]
                + self.dual_obj_random[dual_id, scenario_idx]
                for dual_id in selected_dual_ids
            ]

            best_dual_position = np.argmax(dual_objective_values)
            optimal_dual_id = selected_dual_ids[best_dual_position]

            optimal_dual_indices.append(optimal_dual_id)
            scenario_upper_bounds.append(dual_objective_values[best_dual_position])

        if return_optimal_duals:
            return scenario_upper_bounds, optimal_dual_indices
        else:
            return scenario_upper_bounds

    def compute_value_function_approximation(
        self, commitment_solution, include_scenario_details=False
    ):
        """
        Compute complete value function approximation using all dual solutions.

        Args:
            commitment_solution (np.array): Commitment decisions
            include_scenario_details (bool): If True, return scenario details

        Returns:
            tuple: (total_value, optimal_dual_indices, [scenario_values])
        """
        if self.dual_obj_random is None:
            self.prepare_dual_arrays()

        # Compute commitment-dual contributions for all dual solutions using vectorized operations
        commitment_weighted = self.get_commitment_weighted_solution(commitment_solution)
        commitment_dual_contributions = np.dot(
            self.generation_duals_array, commitment_weighted
        )

        # Find optimal dual for each scenario
        scenario_optimal_values, optimal_dual_indices = find_largest_index_numba_uc(
            commitment_dual_contributions, self.dual_obj_random
        )

        # Compute total value function including first-stage startup costs
        first_stage_cost = self.calculate_startup_cost(
            commitment_values=commitment_solution
        )
        expected_second_stage_cost = sum(
            self.probability[scenario] * scenario_optimal_values[scenario_idx]
            for scenario_idx, scenario in enumerate(self.scenario)
        )
        total_value_function = first_stage_cost + expected_second_stage_cost

        if include_scenario_details:
            return total_value_function, optimal_dual_indices, scenario_optimal_values
        else:
            return total_value_function, optimal_dual_indices

    def build_extensive_form(self, relaxation=False):
        print("relaxation", relaxation)
        """
        Build the extensive form formulation of the stochastic unit commitment problem.

        This creates a single monolithic model that includes all scenarios without
        decomposition. The model contains:
        - First-stage commitment variables x[g,t]
        - Second-stage power generation variables p[g,t,s] for each scenario
        - Renewable generation variables pr[g,t,s] if applicable
        - Demand shortfall variables slack[t,s] for each scenario

        Args:
            relaxation (bool): If True, relax binary variables to continuous [0,1]

        Returns:
            gp.Model: Extensive form model
        """
        extensive = gp.Model("UC_Extensive")

        # First-stage commitment variables (same as master problem)
        if relaxation:
            self.x_ext = extensive.addVars(
                self.thermal_gens,
                range(0, self.T + 1),
                lb=0.0,
                ub=1.0,
                name="commitment",
            )
        else:
            self.x_ext = extensive.addVars(
                self.thermal_gens,
                range(0, self.T + 1),
                vtype=GRB.BINARY,
                name="commitment",
            )

        # Fix initial conditions (generators start offline)
        for g in self.thermal_gens:
            extensive.addConstr(self.x_ext[g, 0] == 0, name=f"init_{g}")

        # Start-up and shut-down variables for state transitions
        # Startup costs are added via setObjective, not here
        self.w_ext = extensive.addVars(
            self.thermal_gens,
            self.periods,
            vtype=GRB.BINARY if not relaxation else GRB.CONTINUOUS,
            obj=0.0,
            name="startup",
        )

        self.v_ext = extensive.addVars(
            self.thermal_gens,
            self.periods,
            vtype=GRB.BINARY if not relaxation else GRB.CONTINUOUS,
            obj=0.0,  # No shutdown cost
            name="shutdown",
        )

        # State transition constraints
        for g in self.thermal_gens:
            for t in self.periods:
                extensive.addConstr(
                    self.x_ext[g, t]
                    == self.x_ext[g, t - 1] + self.w_ext[g, t] - self.v_ext[g, t],
                    name=f"state_{g}_{t}",
                )

        # Minimum up time constraints
        for g in self.thermal_gens:
            min_up = self.min_up_time[g]
            for t in self.periods:
                if min_up > 1:
                    start_period = max(1, t - min_up + 1)
                    extensive.addConstr(
                        gp.quicksum(
                            self.w_ext[g, s] for s in range(start_period, t + 1)
                        )
                        <= self.x_ext[g, t],
                        name=f"min_up_{g}_{t}",
                    )

        # Minimum down time constraints
        for g in self.thermal_gens:
            min_down = self.min_down_time[g]
            for t in self.periods:
                if min_down > 1:
                    start_period = max(1, t - min_down + 1)
                    extensive.addConstr(
                        gp.quicksum(
                            self.v_ext[g, s] for s in range(start_period, t + 1)
                        )
                        <= 1 - self.x_ext[g, t],
                        name=f"min_down_{g}_{t}",
                    )

        # Second-stage variables for each scenario
        self.p_ext = extensive.addVars(
            self.thermal_gens,
            self.periods,
            self.scenario,
            lb=0.0,
            name="thermal_power",
        )

        # Renewable generation variables (if any)
        if self.nR > 0:
            self.pr_ext = extensive.addVars(
                self.renewable_gens,
                self.periods,
                self.scenario,
                lb=0.0,
                name="renewable_power",
            )

            # Set renewable bounds based on capacity limits (decision variables, not fixed)
            for g in self.renewable_gens:
                for t in self.periods:
                    for s in self.scenario:
                        # Renewable is a decision variable within capacity bounds
                        self.pr_ext[g, t, s].lb = self.renewable_min[g, t]
                        self.pr_ext[g, t, s].ub = self.renewable_max[g, t]

        # Demand shortfall variables
        self.slack_ext = extensive.addVars(
            self.periods,
            self.scenario,
            lb=0.0,
            name="demand_shortfall",
        )

        # Generation limit constraints for each scenario
        for g in self.thermal_gens:
            for t in self.periods:
                for s in self.scenario:
                    # Minimum generation when committed
                    extensive.addConstr(
                        self.p_ext[g, t, s] >= self.min_power[g] * self.x_ext[g, t],
                        name=f"gen_min_{g}_{t}_{s}",
                    )

                    # Maximum generation when committed
                    extensive.addConstr(
                        self.p_ext[g, t, s] <= self.max_power[g] * self.x_ext[g, t],
                        name=f"gen_max_{g}_{t}_{s}",
                    )

        # Demand satisfaction constraints for each scenario
        for t in self.periods:
            for s in self.scenario:
                lhs = gp.quicksum(self.p_ext[g, t, s] for g in self.thermal_gens)
                if self.nR > 0:
                    lhs += gp.quicksum(
                        self.pr_ext[g, t, s] for g in self.renewable_gens
                    )
                lhs += self.slack_ext[t, s]

                extensive.addConstr(
                    lhs >= self.demand_scenarios[s][t], name=f"demand_{t}_{s}"
                )

        # Objective function (first-stage startup costs + second-stage costs)
        objective = 0.0

        # First-stage startup costs
        for g in self.thermal_gens:
            for t in self.periods:
                objective += self.startup_cost[g] * self.w_ext[g, t]

        # Second-stage costs
        for s in self.scenario:
            prob = self.probability[s]
            # Generation costs
            for g in self.thermal_gens:
                for t in self.periods:
                    objective += prob * self.unit_cost[g] * self.p_ext[g, t, s]
            # Penalty costs for unmet demand
            for t in self.periods:
                objective += prob * self.penalty_cost * self.slack_ext[t, s]

        extensive.setObjective(objective, GRB.MINIMIZE)

        print(
            f"Built extensive form: {len(self.thermal_gens)} thermal gens, {self.T} periods, {self.nS} scenarios"
        )
        print(
            f"Total variables: {extensive.NumVars}, Total constraints: {extensive.NumConstrs}"
        )

        return extensive

    def solve_extensive_form(self, relaxation=False, time_limit=None):
        """
        Solve the extensive form formulation.

        Args:
            relaxation (bool): If True, solve LP relaxation
            time_limit (float): Time limit in seconds

        Returns:
            dict: Solution information including objective value and solve time
        """
        extensive_model = self.build_extensive_form(relaxation=relaxation)

        if time_limit:
            extensive_model.setParam("TimeLimit", time_limit)
            extensive_model.setParam("WorkLimit", time_limit)

        extensive_model.setParam("OutputFlag", True)
        extensive_model.optimize()

        solution_info = {
            "status": extensive_model.status,
            "objective": (
                extensive_model.ObjVal if extensive_model.status == 2 else None
            ),
            "solve_time": extensive_model.Runtime,
            "gap": (
                extensive_model.MIPGap
                if not relaxation and extensive_model.status == 2
                else 0.0
            ),
            "num_variables": extensive_model.NumVars,
            "num_constraints": extensive_model.NumConstrs,
        }

        if extensive_model.status == 2:
            # Extract first-stage solution
            self.extensive_first_stage = {}
            for g in self.thermal_gens:
                for t in self.periods:
                    self.extensive_first_stage[g, t] = self.x_ext[g, t].x

            # Print startup variables (w_ext) that are 1
            print("\nStartup variables (w_ext) with value 1:")
            startup_count = 0
            for g in self.thermal_gens:
                for t in self.periods:
                    w_val = self.w_ext[g, t].x
                    if w_val > 0.5:  # Consider as 1 if > 0.5
                        print(f"  w[{g}, {t}] = {w_val:.4f}")
                        startup_count += 1
            if startup_count == 0:
                print("  (none)")
            else:
                print(f"  Total startups: {startup_count}")

            # Extract second-stage solutions for each scenario
            self.extensive_second_stage = {}
            for s in self.scenario:
                self.extensive_second_stage[s] = {}
                for g in self.thermal_gens:
                    for t in self.periods:
                        self.extensive_second_stage[s][g, t] = self.p_ext[g, t, s].x

        return solution_info

    def analyze_lp_relaxation_quality(self, time_limit=3600):
        """
        Analyze LP relaxation quality by comparing LP and IP solutions.

        Solves both LP relaxation and IP of the extensive form, then analyzes
        how many first-stage (commitment) variables are integer in the LP solution.

        Args:
            time_limit (float): Time limit in seconds for each solve

        Returns:
            dict: Analysis results including variable counts and integrality info
        """
        results = {}

        # ============== SOLVE LP RELAXATION ==============
        print("\n" + "=" * 70)
        print("SOLVING LP RELAXATION")
        print("=" * 70)

        lp_model = self.build_extensive_form(relaxation=True)
        if time_limit:
            lp_model.setParam("TimeLimit", time_limit)
            lp_model.setParam("WorkLimit", time_limit)
        lp_model.setParam("OutputFlag", False)
        lp_model.optimize()

        results["lp_status"] = lp_model.status
        results["lp_objective"] = lp_model.ObjVal if lp_model.status == 2 else None
        results["lp_solve_time"] = lp_model.Runtime

        # Analyze first-stage variables in LP solution
        lp_x_values = {}
        lp_zeros = 0
        lp_ones = 0
        lp_fractional = 0
        lp_fractional_values = []

        int_tolerance = 1e-5  # Tolerance for considering a value as integer

        if lp_model.status == 2:
            for g in self.thermal_gens:
                for t in self.periods:
                    val = self.x_ext[g, t].x
                    lp_x_values[g, t] = val

                    if val < int_tolerance:
                        lp_zeros += 1
                    elif val > 1 - int_tolerance:
                        lp_ones += 1
                    else:
                        lp_fractional += 1
                        lp_fractional_values.append((g, t, val))

        total_x_vars = len(self.thermal_gens) * len(self.periods)
        lp_integer_count = lp_zeros + lp_ones

        results["lp_x_zeros"] = lp_zeros
        results["lp_x_ones"] = lp_ones
        results["lp_x_fractional"] = lp_fractional
        results["lp_x_integer_count"] = lp_integer_count
        results["lp_x_total"] = total_x_vars
        results["lp_integrality_ratio"] = (
            lp_integer_count / total_x_vars if total_x_vars > 0 else 0
        )
        results["lp_fractional_details"] = lp_fractional_values

        print("\n" + "-" * 50)
        print("LP RELAXATION FIRST-STAGE VARIABLE ANALYSIS")
        print("-" * 50)
        print(f"Total first-stage (x) variables: {total_x_vars}")
        print(
            f"  Variables = 0:        {lp_zeros:5d} ({100*lp_zeros/total_x_vars:.1f}%)"
        )
        print(f"  Variables = 1:        {lp_ones:5d} ({100*lp_ones/total_x_vars:.1f}%)")
        print(
            f"  Variables fractional: {lp_fractional:5d} ({100*lp_fractional/total_x_vars:.1f}%)"
        )
        print(
            f"  Integer variables:    {lp_integer_count:5d} ({100*lp_integer_count/total_x_vars:.1f}%)"
        )
        print(f"LP Objective: {results['lp_objective']:.4f}")

        if lp_fractional_values:
            print(f"\nFractional variables (showing up to 20):")
            for g, t, val in lp_fractional_values[:20]:
                print(f"  x[{g}, {t}] = {val:.6f}")
            if len(lp_fractional_values) > 20:
                print(f"  ... and {len(lp_fractional_values) - 20} more")

        # ============== SOLVE IP ==============
        print("\n" + "=" * 70)
        print("SOLVING INTEGER PROGRAM")
        print("=" * 70)

        ip_model = self.build_extensive_form(relaxation=False)
        if time_limit:
            ip_model.setParam("TimeLimit", time_limit)
            ip_model.setParam("WorkLimit", time_limit)
        ip_model.setParam("OutputFlag", False)
        ip_model.optimize()

        results["ip_status"] = ip_model.status
        results["ip_objective"] = ip_model.ObjVal if ip_model.status in [2, 9] else None
        results["ip_solve_time"] = ip_model.Runtime
        results["ip_gap"] = ip_model.MIPGap if ip_model.status in [2, 9] else None
        results["ip_nodes"] = ip_model.NodeCount

        # Analyze first-stage variables in IP solution
        ip_x_values = {}
        ip_zeros = 0
        ip_ones = 0

        if ip_model.status in [2, 9]:  # Optimal or time limit with solution
            for g in self.thermal_gens:
                for t in self.periods:
                    val = self.x_ext[g, t].x
                    ip_x_values[g, t] = val

                    if val < int_tolerance:
                        ip_zeros += 1
                    else:
                        ip_ones += 1

            # Print startup variables (w_ext) that are 1
            print("\nStartup variables (w_ext) with value 1:")
            startup_count = 0
            for g in self.thermal_gens:
                for t in self.periods:
                    w_val = self.w_ext[g, t].x
                    if w_val > 0.5:  # Consider as 1 if > 0.5
                        print(f"  w[{g}, {t}] = {w_val:.4f}")
                        startup_count += 1
            if startup_count == 0:
                print("  (none)")
            else:
                print(f"  Total startups: {startup_count}")

        results["ip_x_zeros"] = ip_zeros
        results["ip_x_ones"] = ip_ones

        print("\n" + "-" * 50)
        print("IP FIRST-STAGE VARIABLE ANALYSIS")
        print("-" * 50)
        print(f"Total first-stage (x) variables: {total_x_vars}")
        print(f"  Variables = 0: {ip_zeros:5d} ({100*ip_zeros/total_x_vars:.1f}%)")
        print(f"  Variables = 1: {ip_ones:5d} ({100*ip_ones/total_x_vars:.1f}%)")
        print(f"IP Objective: {results['ip_objective']:.4f}")
        if results["ip_gap"] is not None:
            print(f"IP Gap: {100*results['ip_gap']:.4f}%")

        # ============== COMPARISON ==============
        print("\n" + "=" * 70)
        print("LP vs IP COMPARISON")
        print("=" * 70)

        if results["lp_objective"] and results["ip_objective"]:
            lp_ip_gap = (
                (results["ip_objective"] - results["lp_objective"])
                / results["ip_objective"]
                * 100
            )
            results["lp_ip_gap_percent"] = lp_ip_gap
            print(f"LP Objective:  {results['lp_objective']:.4f}")
            print(f"IP Objective:  {results['ip_objective']:.4f}")
            print(f"LP-IP Gap:     {lp_ip_gap:.4f}%")
            print(
                f"LP Integrality: {100*results['lp_integrality_ratio']:.1f}% of x variables are integer"
            )

            # Count how many variables changed from LP to IP
            if lp_x_values and ip_x_values:
                changes = 0
                for g in self.thermal_gens:
                    for t in self.periods:
                        lp_val = lp_x_values[g, t]
                        ip_val = ip_x_values[g, t]
                        # Round LP to nearest integer for comparison
                        lp_rounded = round(lp_val)
                        if abs(lp_rounded - ip_val) > int_tolerance:
                            changes += 1

                results["lp_ip_x_differences"] = changes
                print(
                    f"Variables that changed (LP rounded vs IP): {changes} ({100*changes/total_x_vars:.1f}%)"
                )

        print("=" * 70 + "\n")

        return results


@njit(
    nb.types.Tuple((nb.float64[:], nb.int64[:]))(nb.float64[:], nb.float64[:, :]),
    parallel=True,
    fastmath=True,
)
def find_largest_index_numba_uc(commitment_terms, scenario_terms):
    """
    High-performance function to find optimal dual solutions for each scenario in UC.

    For each scenario (column), finds the dual solution (row) that maximizes
    commitment_terms[dual] + scenario_terms[dual, scenario].

    Args:
        commitment_terms (np.array): Commitment-dependent dual contributions
        scenario_terms (np.array): Scenario-dependent dual contributions (dual, scenario)

    Returns:
        tuple: (max_objective_values, optimal_dual_indices)
    """
    num_duals, num_scenarios = scenario_terms.shape

    optimal_dual_indices = np.empty(num_scenarios, dtype=np.int64)
    max_objective_values = np.empty(num_scenarios, dtype=np.float64)

    for scenario_idx in prange(num_scenarios):
        max_objective = np.finfo(np.float64).min
        best_dual_index = -1

        for dual_idx in range(len(commitment_terms)):
            objective_value = (
                commitment_terms[dual_idx] + scenario_terms[dual_idx, scenario_idx]
            )
            if objective_value > max_objective:
                max_objective = objective_value
                best_dual_index = dual_idx

        optimal_dual_indices[scenario_idx] = best_dual_index
        max_objective_values[scenario_idx] = max_objective

    return max_objective_values, optimal_dual_indices
