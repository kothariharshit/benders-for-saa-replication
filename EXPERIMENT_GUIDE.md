# Experiment Runner Guide

This guide explains how to use `run_experiments.py` to run both single-cut and multi-cut experiments for CFLP, CMND, and UC problems.

## Usage

### Basic Command Structure
```bash
python3 run_experiments.py \
    --problem-type {cflp,cmnd,uc} \
    [--single-cut] \
    --run-type {1,2,3} \
    --lp-or-ip {0,1} \
    --num-instances N
```

### Arguments

- `--problem-type`: Problem to solve
  - `cflp`: Facility Location Problem
  - `cmnd`: Multicommodity Network Design
  - `uc`: Unit Commitment

- `--single-cut`: Optional flag to run single-cut variant (default is multi-cut)
  - **Note:** Single-cut experiments are only supported for CFLP with `--run-type 1` (Normal Run)
  - Single-cut is too slow for CMND and UC problems. Therefore, we do not test information reuse methods for it. No reuse method can still be run.

- `--run-type`: Type of experiment
  - `1`: Normal Run (all 5 algorithm variants)
  - `2`: Boosted Static Init (only boosted variants) - **multi-cut only**
  - `3`: Scenario Variation (test with 200, 400, 800 scenarios) - **multi-cut only**

- `--lp-or-ip`: Instance type
  - `0`: LP instances (relaxed)
  - `1`: IP instances (integer)

- `--num-instances`: Number of instances to run

## Examples

### Example 1: CFLP Multi-cut Normal Run
Run 5 CFLP LP instances with multi-cut Benders:
```bash
python3 run_experiments.py \
    --problem-type cflp \
    --run-type 1 \
    --lp-or-ip 0 \
    --num-instances 5
```

### Example 2: CFLP Single-cut Normal Run
Run 5 CFLP LP instances with single-cut Benders:
```bash
python3 run_experiments.py \
    --problem-type cflp \
    --single-cut \
    --run-type 1 \
    --lp-or-ip 0 \
    --num-instances 5
```

### Example 3: UC Multi-cut Normal Run
Run 4 UC LP instances (randomly generated):
```bash
python3 run_experiments.py \
    --problem-type uc \
    --run-type 1 \
    --lp-or-ip 0 \
    --num-instances 4
```

## Algorithm Configuration

All solver scripts use the three-integer format: `[solve_ip] [dual_lookup] [init]`

- **solve_ip**: `0` for LP relaxation, `1` for IP
- **dual_lookup**: `0` = NoReuse, `1` = DSP, `2` = CuratedDSP
- **init**: `0` = No init, `1` = Static init, `2` = Adaptive init

**5 Algorithm Variants** (tested in `--run-type 1`):
1. **NoReuse**: `1 0 0` - No dual solution reuse
2. **DSP**: `1 1 0` - Basic DSP
3. **CuratedDSP**: `1 2 0` - Curated DSP
4. **StaticInit**: `1 2 1` - Curated DSP with Static Initialization
5. **AdaptiveInit**: `1 2 2` - Curated DSP with Adaptive Initialization

## Solver Scripts & Output Files

### Multi-Cut Solvers
- **Scripts**: `solve_cflp.py`, `solve_cmnd.py`, `solve_uc.py`
- **Algorithms**: All 5 variants (NoReuse, DSP, CuratedDSP, StaticInit, AdaptiveInit)
- **CSV output**: `results_multi_{cflp|cmnd|uc}.csv`
- **Logs**: `detailed-results/{problem}/{IP|LP}/multi_...{algorithm}.op`

### Single-Cut Solvers
- **Scripts**: `solve_cflp_single_cut.py`, `solve_cmnd_single_cut.py`, `solve_uc_single_cut.py`
- **Algorithms**: Vanilla NoReuse only (no information reuse for cmnd and uc)
- **CSV output**: `results_single_{cflp|cmnd|uc}.csv`
- **Logs**: `detailed-results/{problem}/{IP|LP}/single_...{algorithm}.op`

## Time Limit Experiments

Compare NoReuse vs AdaptiveInit with equal computational time. Must use individual solver scripts (not `run_experiments.py`).

### Workflow Example

```bash
# 1. Run AdaptiveInit to get baseline time (e.g., output shows: 45.3 sec average)
python3 solve_cmnd.py 1 2 2 instances-cmnd/r06.2.dow 200 26

# 2. Run NoReuse with equal time limit
python3 solve_cmnd.py 1 0 0 instances-cmnd/r06.2.dow 200 26 --timelimit 45.3

# 3. Run NoReuse with 2x time limit
python3 solve_cmnd.py 1 0 0 instances-cmnd/r06.2.dow 200 26 --timelimit 45.3 --twice
```

Compare upper/lower bounds and gaps to evaluate whether AdaptiveInit provides better solutions with equal time budgets.
