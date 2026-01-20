import subprocess
import matplotlib.pyplot as plt
import random
from scipy.stats import gmean
import pandas as pd
import os
import warnings
import seaborn as sns
import argparse

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# Constants
NSAA = 3
SEED = 2
DEFAULT_NSCENARIOS = 2
random.seed(SEED)

# Parameter combinations for the experiments
PARAMETER_COMBINATIONS = [
    ("0", "0"),  # NoReuse
    ("1", "0"),  # DSP
    ("2", "0"),  # CuratedDSP
    ("2", "1"),  # StaticInit
    ("2", "2"),  # AdaptiveInit
]


def compute_gmean(group):
    """Compute geometric mean for numeric columns."""
    numeric_columns = group.select_dtypes(include=["number"]).columns
    return group[numeric_columns].apply(lambda x: round(gmean(x + 1) - 1, 2))


def run_cflp_or_cmnd_experiments(
    problem_type, single_cut, run_type, lp_or_ip, num_instances
):
    """Run experiments for CFLP or CMND problems."""
    # File selection based on problem type and LP or IP
    cut_prefix = "single" if single_cut else "multi"
    if problem_type == "cflp":
        filename = (
            "instances-cflp/cflp_ip_instances.txt"
            if lp_or_ip
            else "instances-cflp/cflp_lp_instances.txt"
        )
        result_filename = f"results_{cut_prefix}_cflp.csv"
    else:  # cmnd
        filename = (
            "instances-cmnd/cmnd_ip_instances.txt"
            if lp_or_ip
            else "instances-cmnd/cmnd_lp_instances.txt"
        )
        result_filename = f"results_{cut_prefix}_cmnd.csv"

    # Remove existing result file if it exists
    if os.path.exists(result_filename):
        os.remove(result_filename)

    # Read all instances from the file
    with open(filename, "r") as file:
        lines = [line.strip() for line in file if line.strip()]

    # Validate the number of instances
    if num_instances > len(lines):
        print(
            "The number of instances requested exceeds the number available. Running all available instances."
        )
        num_instances = len(lines)

    # Select a random subset of instances
    selected_instances = random.sample(lines, num_instances)

    # Get the solver script
    suffix = "_single_cut" if single_cut else ""
    solver_script = f"solve_{problem_type}{suffix}.py"

    # Processing instances based on the selected type
    for line in selected_instances:
        line_numbers = line.split()
        print(f"Processing instance: {line}")

        if run_type == 2:  # Boosted Static Init
            subprocess.run(
                ["python3", solver_script, "1", "2", "2"]
                + line_numbers
                + [str(DEFAULT_NSCENARIOS), str(NSAA)]
            )
            subprocess.run(
                ["python3", solver_script, "1", "2", "3"]
                + line_numbers
                + [str(DEFAULT_NSCENARIOS), str(NSAA)]
            )
        elif run_type == 3:  # Scenario Variation
            nscen_list = [200, 400, 800]
            if problem_type == "cflp":
                line_numbers = ["35", "105", "5"]
            else:
                line_numbers = ["instances-cmnd/r03.1.dow"]
            for nscen in nscen_list:
                print(f"Processing scenario: {nscen}")
                for dsp_technique, init_technique in PARAMETER_COMBINATIONS:
                    subprocess.run(
                        [
                            "python3",
                            solver_script,
                            str(lp_or_ip),
                            dsp_technique,
                            init_technique,
                        ]
                        + line_numbers
                        + [str(nscen), str(NSAA)]
                    )
        else:  # Normal Run
            assert run_type == 1
            for dsp_technique, init_technique in PARAMETER_COMBINATIONS:
                subprocess.run(
                    [
                        "python3",
                        solver_script,
                        str(lp_or_ip),
                        dsp_technique,
                        init_technique,
                    ]
                    + line_numbers
                    + [str(DEFAULT_NSCENARIOS), str(NSAA)]
                )

    return result_filename, cut_prefix


def run_uc_experiments(single_cut, run_type, lp_or_ip, num_instances):
    """Run experiments for UC (Unit Commitment) problems."""
    cut_prefix = "single" if single_cut else "multi"
    result_filename = f"results_{cut_prefix}_uc.csv"

    # Remove existing result file if it exists
    if os.path.exists(result_filename):
        os.remove(result_filename)

    # UC instances are generated with varied parameters
    variance = 0.1
    ndays = 2

    # Create list of UC instance specifications with varied parameters
    selected_instances = []
    generator_options = [10, 20, 30, 40]
    difficulty_options = [2, 3]

    for i in range(num_instances):
        ngen = random.choice(generator_options)
        difficulty = random.choice(difficulty_options)
        selected_instances.append(f"{ngen} {ndays} {difficulty} {variance}")

    print(f"\nGenerated {num_instances} UC instances:")
    for i, inst in enumerate(selected_instances, 1):
        print(f"  {i}. {inst}")

    # Get the solver script
    suffix = "_single_cut" if single_cut else ""
    solver_script = f"solve_uc{suffix}.py"

    # Processing instances
    for line in selected_instances:
        line_numbers = line.split()
        print(f"Processing instance: {line}")

        ngen, ndays, diff, var = line_numbers
        instance_file = "dummy"  # Placeholder, will use --generate flag

        if run_type == 2:  # Boosted Static Init
            for init_val in ["2", "3"]:
                cmd = [
                    "python3",
                    solver_script,
                    "1",
                    "2",
                    init_val,
                    instance_file,
                    str(DEFAULT_NSCENARIOS),
                    var,
                    str(NSAA),
                    "--generate",
                    ngen,
                    ndays,
                    diff,
                ]
                subprocess.run(cmd)
        elif run_type == 3:  # Scenario Variation
            nscen_list = [200, 400, 800]
            for nscen in nscen_list:
                print(f"Processing scenario: {nscen}")
                for dsp_technique, init_technique in PARAMETER_COMBINATIONS:
                    cmd = [
                        "python3",
                        solver_script,
                        str(lp_or_ip),
                        dsp_technique,
                        init_technique,
                        instance_file,
                        str(nscen),
                        var,
                        str(NSAA),
                        "--generate",
                        ngen,
                        ndays,
                        diff,
                    ]
                    subprocess.run(cmd)
        else:  # Normal Run
            assert run_type == 1
            for dsp_technique, init_technique in PARAMETER_COMBINATIONS:
                cmd = [
                    "python3",
                    solver_script,
                    str(lp_or_ip),
                    dsp_technique,
                    init_technique,
                    instance_file,
                    str(DEFAULT_NSCENARIOS),
                    var,
                    str(NSAA),
                    "--generate",
                    ngen,
                    ndays,
                    diff,
                ]
                subprocess.run(cmd)

    return result_filename, cut_prefix


def process_results(result_filename, cut_prefix, problem_type, run_type, lp_or_ip):
    """Process and rename results, compute geometric means, and generate plots."""
    # Set the instance type for file naming
    instance_type = "lp" if lp_or_ip == 0 else "ip"
    new_file_name = None

    # Generate the new result file name based on the run type
    if run_type == 1:
        new_file_name = (
            f"results_{cut_prefix}_{problem_type}_normal_{instance_type}.csv"
        )
    elif run_type == 2:
        new_file_name = (
            f"results_{cut_prefix}_{problem_type}_boosted_{instance_type}.csv"
        )
    elif run_type == 3:
        new_file_name = (
            f"results_{cut_prefix}_{problem_type}_scenario_{instance_type}.csv"
        )

    # Rename the result file if it exists
    if os.path.exists(result_filename):
        os.rename(result_filename, new_file_name)
    else:
        print(f"Warning: Result file {result_filename} not found.")
        return

    # Compute and save geometric means if it's a normal run
    if run_type == 1:
        problem = "ip" if lp_or_ip else "lp"
        data = pd.read_csv(new_file_name)
        grouped = data.groupby("Method")

        geometric_means = grouped.apply(compute_gmean)
        geometric_means.reset_index(inplace=True)

        output_path = f"geometric_means_{cut_prefix}_{problem_type}_{problem}.csv"
        geometric_means.to_csv(output_path, index=False)
        print(f"Geometric means saved to {output_path}.")

    elif run_type == 3:
        problem = "ip" if lp_or_ip else "lp"
        data = pd.read_csv(new_file_name)
        data["Scenarios-Method"] = (
            data["Scenarios"].astype(str) + "-" + data["Method"].astype(str)
        )
        data = data.drop(columns=["Scenarios"])
        geometric_means = data.groupby("Scenarios-Method").apply(compute_gmean)
        geometric_means = geometric_means.reset_index()
        output_path = f"geometric_means_{cut_prefix}_{problem_type}_{problem}.csv"
        geometric_means.to_csv(output_path, index=False)
        print(f"Geometric means saved to {output_path}.")

    # Plotting ECDF of total times for normal run
    if run_type == 1:
        df = pd.read_csv(new_file_name, index_col=False)
        df.columns = [c.replace(" ", "_") for c in df.columns]
        # Handle both "multi_" and "single_" prefixed method names
        df["Method"] = df["Method"].str.replace(r"^(multi_|single_)", "", regex=True)
        df["Method"] = df["Method"].replace(["NoReuse"], "No reuse")
        df["Method"] = df["Method"].replace(["AdaptiveInit"], "Adaptive init")
        df["Method"] = df["Method"].replace(["DSP"], "DSP")
        df["Method"] = df["Method"].replace(["CuratedDSP"], "Curated DSP")
        df["Method"] = df["Method"].replace(["StaticInit"], "Static init")

        df_new = df[
            df.Method.isin(
                ("No reuse", "Adaptive init", "Static init", "DSP", "Curated DSP")
            )
        ]

        # Only include methods that are actually present in the data
        all_methods = ["DSP", "Curated DSP", "Static init", "Adaptive init"]
        legend_order = [m for m in all_methods if m in df_new["Method"].unique()]

        # Determine x-axis column based on problem type
        if "Total_times_average" in df_new.columns:
            x_col = "Total_times_average"
            x_label = "Total time average"
        elif "Gap" in df_new.columns:
            x_col = "Gap"
            x_label = "Gap"
        else:
            print("Warning: No suitable column found for ECDF plot.")
            return

        # Plot ECDF
        if len(df_new) > 0 and len(legend_order) > 0:
            sns.ecdfplot(
                data=df_new,
                x=x_col,
                hue="Method",
                hue_order=legend_order,
            ).set(
                xlabel=x_label,
                ylabel="Proportion of instances",
            )

            plt.savefig(f"{problem_type}_cdf_{instance_type}_{cut_prefix}.png")
            print(f"Plot saved to {problem_type}_cdf_{instance_type}_{cut_prefix}.png")
            plt.close()
        else:
            print("Warning: No data available for ECDF plot.")


def main():
    """Main function to run experiments based on user input."""
    parser = argparse.ArgumentParser(
        description="Run Benders decomposition experiments for CFLP, CMND, or UC problems"
    )
    parser.add_argument(
        "--problem-type",
        type=str,
        choices=["cflp", "cmnd", "uc"],
        required=True,
        help="Problem type: 'cflp', 'cmnd', or 'uc'",
    )
    parser.add_argument(
        "--single-cut",
        action="store_true",
        help="Run single cut experiments (default: multi-cut)",
    )
    parser.add_argument(
        "--run-type",
        type=int,
        choices=[1, 2, 3],
        required=True,
        help="1=Normal Run, 2=Boosted Static Init, 3=Scenario Variation",
    )
    parser.add_argument(
        "--lp-or-ip",
        type=int,
        choices=[0, 1],
        required=True,
        help="0=LP instances, 1=IP instances",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        required=True,
        help="Number of instances to run",
    )

    args = parser.parse_args()

    cut_type_str = "Single-cut" if args.single_cut else "Multi-cut"
    print(f"\n{'='*60}")
    print(f"Running {args.problem_type.upper()} Experiments")
    print(f"{'='*60}")
    print(f"Problem Type: {args.problem_type}")
    print(f"Cut Type: {cut_type_str}")
    print(
        f"Run Type: {['', 'Normal', 'Boosted Static Init', 'Scenario Variation'][args.run_type]}"
    )
    print(f"Instance Type: {'LP' if args.lp_or_ip == 0 else 'IP'}")
    print(f"Number of Instances: {args.num_instances}")
    print(f"{'='*60}\n")

    # Run experiments based on problem type
    if args.problem_type in ["cflp", "cmnd"]:
        result_filename, cut_prefix = run_cflp_or_cmnd_experiments(
            args.problem_type,
            args.single_cut,
            args.run_type,
            args.lp_or_ip,
            args.num_instances,
        )
    else:  # uc
        result_filename, cut_prefix = run_uc_experiments(
            args.single_cut, args.run_type, args.lp_or_ip, args.num_instances
        )

    # Process results
    process_results(
        result_filename, cut_prefix, args.problem_type, args.run_type, args.lp_or_ip
    )

    print(f"\n{'='*60}")
    print("Experiments completed successfully!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
