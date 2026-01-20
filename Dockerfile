# Use Gurobi's official Docker image as the base
FROM gurobi/python:10.0.1_3.9

# Install additional Python dependencies (if required)
RUN pip install numpy pandas matplotlib scipy numba tabulate seaborn

# Create a working directory
WORKDIR /app

# Define default command (adjust based on how your scripts are run)
CMD ["python3", "run_experiments.py"]
