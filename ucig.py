#!/usr/bin/env python3
"""
ucig.py
Unit Commitment Instance Generator (Simplified for JSON output)
Python translation by Claude Code
Original C++ implementation by Fabrizio Lacalandra and Luigi Poderico

Simplified version that:
- Removes hydro and hydro cascade components
- Generates only thermal generators
- Outputs in JSON format compatible with uc_benders_utils.py
"""

import random
import argparse
import sys
import json


def uniform(min_val, max_val):
    """Generate uniform random number between min_val and max_val"""
    return random.uniform(min_val, max_val)


class ThermalData:
    """Thermal power unit data"""

    def __init__(self):
        # Power units data
        self.horizon_len = 0

        # Cost coefficients
        self.a = 0.0  # Quadratic coefficient
        self.b = 0.0  # Linear coefficient
        self.c = 0.0  # Constant coefficient

        # Power limits
        self.inf = 0.0  # Minimum power
        self.sup = 0.0  # Maximum power

        # Time constraints
        self.t_on = 0   # Minimum up-time
        self.t_off = 0  # Minimum down-time

        # Cost parameters
        self.costo = 0.0
        self.costof = 0.0
        self.costoc = 0.0
        self.tau = 0.0
        self.tau_max = 0.0
        self.succ = 0.0

        # Initial conditions
        self.storia = []
        self.p0 = 0.0
        self.comb = 0.0

        # Ramping (not used but available)
        self.rampa_up = 0.0
        self.rampa_dwn = 0.0

        # Unit size counters
        self.small = 0
        self.medium = 0
        self.big = 0

    def init_power_data(self, k, horizon_len, gmax, csc, a_min, a_max, difficulty):
        """Initialize thermal unit data"""
        self.storia = [0] * (horizon_len + 1)
        self.a = uniform(a_min, a_max)

        self.small = 0
        self.medium = 0
        self.big = 0

        diff = (a_max - a_min) / 3

        a_min_small = a_max - diff
        a_min_medium = a_min_small - diff
        a_min_big = a_min_medium - diff

        if difficulty >= 2:
            # Forces a balance with respect to the big/medium/small unit number
            if self.small < gmax / 3:
                self.a = uniform(a_min_small, a_max)
            if self.medium < gmax / 3:
                self.a = uniform(a_min_medium, a_min_small)
            if self.big < gmax / 4:
                self.a = uniform(a_min, a_min_medium)

        if self.a >= a_min_small:
            # Small unit
            if difficulty == 1:
                self.t_on = int(uniform(1, 1))
                self.t_off = int(uniform(1, 1))
                self.inf = uniform(30, 50)
                self.sup = uniform(100, 130)
            elif difficulty == 2:
                self.t_on = int(uniform(1, 2))
                self.t_off = int(uniform(1, 2))
                self.inf = uniform(30, 50)
                self.sup = uniform(100, 130)
            else:  # difficulty == 3
                self.t_on = int(uniform(2, 3))
                self.t_off = int(uniform(2, 3))
                self.inf = uniform(30, 50)
                self.sup = uniform(100, 130)
            self.b = uniform(4, 5)
            self.c = uniform(100, 150)
            self.small += 1

        elif self.a >= a_min_medium and self.a <= a_min_small:
            # Medium unit
            if difficulty == 1:
                self.t_on = int(uniform(2, 3))
                self.t_off = int(uniform(3, 3))
                self.inf = uniform(50, 70)
                self.sup = uniform(150, 200)
            elif difficulty == 2:
                self.t_on = int(uniform(2, 3))
                self.t_off = int(uniform(3, 4))
                self.inf = uniform(50, 70)
                self.sup = uniform(150, 200)
            else:  # difficulty == 3
                self.t_on = int(uniform(3, 4))
                self.t_off = int(uniform(3, 4))
                self.inf = uniform(50, 70)
                self.sup = uniform(150, 200)
            self.b = uniform(6.7, 7.4)
            self.c = uniform(190, 350)
            self.medium += 1

        else:
            # Big unit
            if difficulty == 1:
                self.t_on = int(uniform(6, 7))
                self.t_off = int(uniform(6, 7))
                self.inf = uniform(70, 100)
                self.sup = uniform(200, 330)
            elif difficulty == 2:
                self.t_on = int(uniform(7, 8))
                self.t_off = int(uniform(7, 8))
                self.inf = uniform(70, 100)
                self.sup = uniform(200, 330)
            else:  # difficulty == 3
                self.t_on = int(uniform(9, 10))
                self.t_off = int(uniform(9, 10))
                self.inf = uniform(70, 100)
                self.sup = uniform(200, 330)
            self.b = uniform(7, 8.5)
            self.c = uniform(400, 550)
            self.big += 1

        self.storia[0] = int(uniform(-5, 5))
        if -1 < self.storia[0] < 1:  # storia[0] == 0
            self.storia[0] = int(pow(-1.0, k) * self.t_on)

        if csc == 1:
            self.costof = 0
            self.costoc = 0
        else:
            self.costof = uniform(200, 250)
            self.costoc = uniform(150, 200)

        self.costo = uniform(1.3 * self.sup, 1.6 * self.sup)
        self.comb = 1  # uniform(1, 1.1)
        self.tau = uniform(1, 2)
        self.p0 = uniform(1.2 * self.inf, 0.9 * self.sup) if self.storia[0] > 0 else 0
        self.tau_max = 5
        self.succ = self.costo

        for i in range(horizon_len):
            if self.storia[i] >= 1:
                self.storia[i + 1] = self.storia[i] + 1
            else:
                self.storia[i + 1] = -1 if (-1 < self.storia[i] - 1) else self.storia[i] - 1

    def get_min_power(self):
        return self.inf

    def get_max_power(self):
        return self.sup

    def get_storia(self, i):
        return self.storia[i]

    def get_t_off(self):
        return self.t_off

    def get_t_on(self):
        return self.t_on

    def to_json_dict(self, gen_name):
        """Convert to JSON dictionary format compatible with uc_benders_utils.py"""
        # Simplified cost model - use linear cost coefficient
        # In uc_benders_utils.py, only the first piecewise cost is used (line 320)
        unit_cost = self.b * self.comb

        return {
            "time_up_minimum": self.t_on,
            "time_down_minimum": self.t_off,
            "power_output_minimum": self.inf,
            "power_output_maximum": self.sup,
            "piecewise_production": [{"cost": unit_cost}],
            # Store additional parameters as metadata (optional)
            "metadata": {
                "quadratic_cost": self.a * self.comb,
                "linear_cost": self.b * self.comb,
                "constant_cost": self.c * self.comb,
                "startup_cost": self.costof * self.comb,
                "shutdown_cost": self.costoc * self.comb,
                "initial_status": self.storia[0],
                "initial_power": self.p0
            }
        }

    def __str__(self):
        """String representation for output"""
        return (f"{self.a * self.comb}\t{self.b * self.comb}\t{self.c * self.comb}\t"
                f"{self.inf}\t{self.sup}\t{self.storia[0]}\t{self.t_on}\t{self.t_off}\t"
                f"{self.costof * self.comb}\t{self.costoc * self.comb}\t{self.tau}\t"
                f"{self.tau_max}\t{self.costoc * self.costo}\t{self.succ * self.costo}\t"
                f"{self.p0}")


class Load:
    """Load demand data"""

    def __init__(self):
        self.carico_max = 0.0
        self.breaks = 0
        self.carico = []
        self.perc_demand = []
        self.rr_perc = []

    def read_perc_demand(self, horizon_len):
        """Read demand profile from perc.dat file"""
        try:
            with open("perc.dat", "r") as f:
                lines = f.readlines()
        except IOError:
            print('Error opening file "perc.dat".', file=sys.stderr)
            sys.exit(1)

        breaks_demand = [0.0] * self.breaks
        self.perc_demand = [0.0] * horizon_len
        self.rr_perc = [0.0] * self.breaks

        for i in range(self.breaks):
            parts = lines[i].split()
            ai = int(parts[0])
            breaks_demand[i] = float(parts[1])
            self.rr_perc[i] = uniform(0.06, 0.11)

        for j in range(horizon_len):
            i = j % self.breaks
            self.perc_demand[j] = breaks_demand[i] * uniform(0.9, 1)

    def init_load(self, horizon_len, breaks, thermal_data):
        """Initialize load data (simplified without hydro)"""
        self.carico = [0.0] * horizon_len
        num_thermal = len(thermal_data)
        self.breaks = breaks

        self.read_perc_demand(horizon_len)

        # Compute carico_max based only on thermal capacity
        total_max_power = 0.0

        for i in range(num_thermal):
            if (thermal_data[i].get_storia(0) <= -thermal_data[i].get_t_off() or
                thermal_data[i].get_storia(0) >= 1):
                total_max_power += thermal_data[i].get_max_power()

        # Set max load to be between 50% and 70-90% of available thermal capacity
        self.carico_max = uniform(
            total_max_power / 2,
            total_max_power * uniform(0.7, 0.9)
        )

        # Compute carico for each time period
        for h in range(horizon_len):
            self.carico[h] = uniform(0.950, 1) * self.perc_demand[h] * self.carico_max

            total_max_power = 0.0
            total_min_power = 0.0

            for i in range(num_thermal):
                if (thermal_data[i].get_storia(h) <= -thermal_data[i].get_t_off() or
                    thermal_data[i].get_storia(h) >= 1):
                    total_max_power += thermal_data[i].get_max_power()

                if (thermal_data[i].get_storia(h) >= 1 and
                    thermal_data[i].get_storia(h) <= thermal_data[i].get_t_on()):
                    total_min_power += thermal_data[i].get_min_power()

            # Adjust demand to be feasible with available thermal capacity
            if 1.11 * self.carico[h] > total_max_power:
                self.carico[h] = (uniform(0.9, 0.95) * total_max_power) / 1.09

            if self.carico[h] < total_min_power:
                self.carico[h] = uniform(1.15, 1.25) * total_min_power

    def __str__(self):
        """String representation for output"""
        lines = []

        # Loads section
        num_days = len(self.carico) // self.breaks
        lines.append(f"Loads\t{num_days}\t{self.breaks}")

        for i in range(0, len(self.carico), self.breaks):
            line = "\t".join(str(self.carico[j]) for j in range(i, min(i + self.breaks, len(self.carico))))
            lines.append(line)

        # Spinning reserve section
        lines.append(f"SpinningReserve\t{self.breaks}")
        line = "\t".join(str(x) for x in self.rr_perc)
        lines.append(line)

        return "\n".join(lines)


class UCIG:
    """Unit Commitment Instance Generator (Simplified)"""

    def __init__(self, gg, breaks, gmax, a_min, a_max, seed, csc, difficulty):
        self.gg = gg
        self.breaks = breaks
        self.gmax = gmax
        self.a_min = a_min
        self.a_max = a_max
        self.horizon_len = gg * breaks
        self.seed = seed
        self.csc = csc
        self.difficulty = difficulty

        self.min_power = 0.0
        self.max_power = 0.0
        self.max_thermal = 0.0

        self.thermal_data = []
        self.load = Load()

    def init_data(self):
        """Initialize all data (simplified without hydro)"""
        self.thermal_data = []
        self.min_power = 0.0
        self.max_power = 0.0
        self.max_thermal = 0.0

        random.seed(self.seed)

        # Initialize thermal units
        for k in range(self.gmax):
            thermal = ThermalData()
            thermal.init_power_data(k, self.horizon_len, self.gmax, self.csc,
                                   self.a_min, self.a_max, self.difficulty)
            self.thermal_data.append(thermal)
            self.min_power += thermal.get_min_power()
            self.max_power += thermal.get_max_power()
            self.max_thermal += thermal.get_max_power()

        # Initialize load
        self.load.init_load(self.horizon_len, self.breaks, self.thermal_data)

    def next_seed(self):
        """Update the random generator seed"""
        self.seed += 1

    def to_json(self):
        """Export instance to JSON format compatible with uc_benders_utils.py"""
        # Build thermal generators dictionary
        thermal_generators = {}
        for k in range(self.gmax):
            gen_name = f"g{k+1}"
            thermal_generators[gen_name] = self.thermal_data[k].to_json_dict(gen_name)

        # Build demand list
        demand = self.load.carico

        # Create JSON structure
        instance = {
            "time_periods": self.horizon_len,
            "demand": demand,
            "thermal_generators": thermal_generators,
            # Add metadata
            "metadata": {
                "instance_id": self.seed,
                "num_days": self.gg,
                "breaks_per_day": self.breaks,
                "num_thermal_generators": self.gmax,
                "difficulty": self.difficulty,
                "a_min": self.a_min,
                "a_max": self.a_max,
                "min_system_capacity": self.min_power,
                "max_system_capacity": self.max_power,
                "max_thermal_capacity": self.max_thermal
            }
        }

        return instance

    def write_general_information(self):
        """Write general information (legacy text format)"""
        print(f"ProblemNum\t{self.seed}")
        print(f"HorizonLen\t{self.horizon_len}")
        print(f"NumThermal\t{self.gmax}")

    def write_load_curve(self):
        """Write load curve"""
        print("LoadCurve")
        print(f"MinSystemCapacity\t{self.min_power}")
        print(f"MaxSystemCapacity\t{self.max_power}")
        print(f"MaxThermalCapacity\t{self.max_thermal}")
        print(self.load)

    def write_thermal_data(self):
        """Write thermal data (legacy text format)"""
        print("ThermalSection")
        for k in range(self.gmax):
            print(f"{k}\t{self.thermal_data[k]}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Unit Commitment Instance Generator (Simplified)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Simplified UC Instance Generator - Thermal units only

Parameters:
  Ic: Number of instances to generate (0 for unlimited) [1]
  Gg: Number of days in planning horizon [1]
  Breaks: Number of time periods per day [24]
  Gmax: Number of thermal units [10]
  Seed: Random number generator seed [0]
  CSC: 1 for constant startup cost, 0 otherwise [0]
  AMin: Minimum quadratic cost coefficient (typical: 0.0001-0.0005) [0.00001]
  AMax: Maximum quadratic cost coefficient (typical: 0.001-0.05) [0.1]
  Difficulty: Instance difficulty level (1, 2, or 3) [1]

Output formats:
  --json: Output in JSON format compatible with uc_benders_utils.py
  --output: Output filename (use .json extension for JSON format)
"""
    )

    parser.add_argument("--Ic", type=int, default=1,
                       help="Number of instances to generate")
    parser.add_argument("--Gg", type=int, default=1,
                       help="Number of days in horizon")
    parser.add_argument("--Breaks", type=int, default=24,
                       help="Number of time periods per day")
    parser.add_argument("--Gmax", type=int, default=10,
                       help="Number of thermal units")
    parser.add_argument("--Seed", type=int, default=0,
                       help="Random number generator seed")
    parser.add_argument("--CSC", type=int, default=0,
                       help="1 for constant startup cost, 0 otherwise")
    parser.add_argument("--AMin", type=float, default=0.00001,
                       help="Minimum quadratic coefficient")
    parser.add_argument("--AMax", type=float, default=0.1,
                       help="Maximum quadratic coefficient")
    parser.add_argument("--Difficulty", type=int, default=1,
                       help="Difficulty level (1, 2, or 3)")
    parser.add_argument("--json", action="store_true",
                       help="Output in JSON format")
    parser.add_argument("--output", type=str, default=None,
                       help="Output filename (stdout if not specified)")

    args = parser.parse_args()

    # Validate parameters
    ic = args.Ic
    gg = args.Gg
    breaks = args.Breaks
    gmax = args.Gmax
    a_min = args.AMin
    a_max = args.AMax
    seed = args.Seed
    csc = args.CSC
    difficulty = args.Difficulty

    if a_min <= 1e-12 or a_max <= 1e-12:
        print("\nERROR: a_min AND a_max MUST BE STRICTLY POSITIVE", file=sys.stderr)
        print(f"a_min= {a_min} a_max= {a_max}", file=sys.stderr)
        sys.exit(1)

    if a_min >= a_max:
        print("\nERROR: cannot set: a_min>=a_max", file=sys.stderr)
        sys.exit(1)

    if a_min < 0.00001 or a_min >= 0.01:
        print(f"\nERROR: a_min= {a_min} is not a realistic value!!", file=sys.stderr)
        print("typical range for a_min is [0.0001,0.0005]", file=sys.stderr)
        sys.exit(1)

    if a_max < 0.1 or a_max >= 0.5:
        print(f"\nERROR: a_max= {a_max} is not a realistic value!!", file=sys.stderr)
        print("typical range for a_max is [0.001,0.05]", file=sys.stderr)
        sys.exit(1)

    if difficulty not in [1, 2, 3]:
        print("\nERROR: difficulty can only be set to 1 or 2 or 3", file=sys.stderr)
        sys.exit(1)

    # Create UCIG instance
    ucig = UCIG(gg, breaks, gmax, a_min, a_max, seed, csc, difficulty)

    if ic == 0:
        ic = sys.maxsize

    # Generate instances
    for instance_num in range(ic):
        ucig.init_data()

        if args.json:
            # JSON output format
            instance_data = ucig.to_json()

            if args.output:
                # Write to file
                with open(args.output, 'w') as f:
                    json.dump(instance_data, f, indent=2)
                print(f"Generated instance saved to {args.output}", file=sys.stderr)
            else:
                # Write to stdout
                print(json.dumps(instance_data, indent=2))
        else:
            # Legacy text format
            ucig.write_general_information()
            ucig.write_load_curve()
            ucig.write_thermal_data()

        ucig.next_seed()

        # Only generate one instance if output file is specified
        if args.output and ic > 1:
            print(f"Warning: Only first instance saved to {args.output}", file=sys.stderr)
            break


if __name__ == "__main__":
    main()
