# Minimizing Mission Time Project Overview

This project simulates UAV-assisted data collection from multiple Ground Nodes
(GNs). Its main goal is to minimize the overall Mission Completion Time (MCT).
The simulation first creates a wireless environment, generates GN locations,
uses a Genetic Algorithm to decide the UAV assignment and visiting order, and
then compares several trajectory planning methods on the same GN sequence.

## Main Workflow

The main entry point is `main.py`.

```text
parameters.py
  -> TrajectoryOptimizer computes communication radius, data rates, and FM capacity
  -> SimulationEnvironment generates GN coordinates
  -> MissionAllocationGA assigns GNs and determines the visiting order
  -> Each UAV route is evaluated by multiple methods
       V-Shaped
       Convex
       CMC
       BOB-V
       BOB-F
       BOB-F_Center
  -> visualizer.py saves trajectory figures
  -> reporter.py saves flight log CSV files
  -> analyze_results.py analyzes batch simulation logs
```

## File Structure

### `main.py`

The orchestration script for the full simulation.

Responsibilities:

- Create the `TrajectoryOptimizer`.
- Create the `SimulationEnvironment`.
- Run `MissionAllocationGA`.
- Execute all trajectory planners for each UAV route.
- Print mission time and path length summaries.
- Save trajectory figures and CSV reports.
- Run multiple simulations in parallel.

The V-Shaped / JOFC-like search logic is currently implemented directly inside
`main.py`, so this file has more responsibility than the other modules.

### `parameters.py`

Central configuration file for the simulation.

It defines:

- Area size: `AREA_WIDTH`, `AREA_HEIGHT`
- Number of GNs and UAVs: `NUM_GNS`, `NUM_UAVS`
- Data Center position: `DATA_CENTER_POS`
- UAV altitude, speed, and power parameters
- Wireless channel parameters
- Genetic Algorithm parameters
- Numerical integration and search settings

Most experiment-level changes should start here.

### `models.py`

Scientific and mathematical model library.

It contains:

- Line-of-Sight probability model
- Path loss model
- SNR calculation
- Shannon transmission rate calculation
- UAV flight power model
- Initial mission cost heuristic for GA fitness evaluation

This module is mostly stateless and is used by the optimizer and planners.

### `utility.py`

Common unit conversion helpers.

It contains:

- dBm to Watt
- Watt to dBm
- dB to linear scale
- linear scale to dB

### `environment.py`

Environment generation module.

`SimulationEnvironment` generates GN coordinates based on the communication
radius and scenario parameters. The generator enforces these constraints:

- GNs are not too close to the simulation boundary.
- GN communication regions do not cover the Data Center.
- GNs keep a minimum distance from each other.

### `mission_allocation_ga.py`

Genetic Algorithm for assignment and routing.

The chromosome design is based on a GN permutation. For multiple UAVs, divider
genes are inserted to split the chromosome into different UAV routes. For a
single UAV, the chromosome is simply a permutation of all GN indices.

The fitness objective is min-max style: it attempts to reduce the largest route
cost among all UAVs.

### `trajectory_optimizer.py`

Core trajectory and communication optimizer.

Responsibilities:

- Compute the maximum horizontal communication radius `comm_radius_d`.
- Compute hover data rate above a GN.
- Compute data rate at the communication boundary.
- Numerically integrate collected data along a line segment.
- Compute the maximum data capacity in Flying Mode (FM).
- Find the optimal hovering/turning point (OH) for FM collection.

`find_optimal_fm_trajectory()` selects different logic depending on geometry:

- Non-overlapping case: mid-perpendicular search.
- Overlapping/general case: ellipse-based search.

This module is the shared low-level dependency for most planners.

### `convex_trajectory_planner.py`

Convex shortest-path baseline.

For a fixed GN sequence, this planner uses CVXPY to find the shortest path that
passes through each GN communication disk. It returns:

- Full path coordinates
- Total path length
- Collection segments for each GN, represented by So/Eo points

### `cmc_planner.py`

CMC stands for Convex-Maximal-Collection.

Workflow:

1. Use `ConvexTrajectoryPlanner` to compute the shortest path.
2. Find intersections between the path and each GN communication disk.
3. Determine valid collection periods according to the GN sequence.
4. Compute the amount of data collected during flight.
5. If collected data is insufficient, add hover time at the best available point.

It returns total time, flight time, hover time, path length, and plotting points.

### `bob_planner.py`

BOB-V planner.

The main idea is to use Convex/CMC information as anchors, then perform a
JOFC-like local search for each GN.

For each GN, the local objective includes:

```text
fly-in time + collection time + fly-out time to the next anchor
```

This lets the planner consider not only the current GN but also the direction of
the next target.

### `bob_overlap.py`

BOB-F planner.

This method is designed for chains of overlapping GN communication regions. It
uses Dynamic Programming over a layered graph.

Core steps:

1. Detect continuous overlapping GN groups in the sequence.
2. Build a layered graph for each group.
3. Layers include SP, FIP candidates, flexible handover candidates, FOP
   candidates, and a global anchor.
4. Run a forward DP pass to compute the minimum cost.
5. Backtrack to recover the best trajectory.

This planner is useful when several GN disks overlap and flexible handover
points can reduce mission time.

### `bob_overlap_center.py`

Center-anchor variant of BOB-F.

Its structure is very similar to `bob_overlap.py`. The main difference is the
choice of global anchor between overlapping groups. This version tends to use
the next GN center as the anchor, instead of the CMC-based entry anchor.

### `visualizer.py`

Plotting module.

It can generate:

- Initial UAV route figures
- Final trajectory comparison figures
- Convex path detail figures
- Performance curves

Some plotting logic for BOB-V and CMC markers is currently kept as commented
code for later use.

### `reporter.py`

CSV report generator for a single simulation run.

It currently focuses on:

- V-Shaped
- Convex

The report includes:

- Method
- Sequence
- GN index
- Fly-in time and distance
- Collection time and distance
- Fly-back time and distance

### `analyze_results.py`

Batch result analysis tool.

It parses `*_log.txt` files inside a simulation result directory and extracts
mission time and path length for all methods. It then generates:

- Statistical summary
- MCT boxplot
- Path length boxplot
- Improvement barplot
- Head-to-head pie charts

Usage:

```bash
python analyze_results.py simulation_results/run_YYYY-MM-DD_HH-MM-SS
```

If no directory is provided, the script attempts to select the latest run
directory automatically.

## Output Directory

Simulation outputs are saved under:

```text
simulation_results/run_YYYY-MM-DD_HH-MM-SS/
```

Common output files:

- `run_x_log.txt`
- `run_x_initial_routes.png`
- `run_x_final_trajectories.png`
- `run_x_UAV_0_flight_log_report.csv`
- `summary_analyzing.txt`
- Summary plots generated by `analyze_results.py`

## How to Run

From the `Minimizing Mission Time` directory:

```bash
python main.py
```

Current batch settings in `main.py`:

```python
NUMBER_OF_RUNS = 200
MAX_WORKERS = 10
```

For debugging, it is usually better to reduce them first:

```python
NUMBER_OF_RUNS = 1
MAX_WORKERS = 1
```

This makes the log easier to inspect and avoids creating many result files.

## Dependencies

Main Python packages:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `cvxpy`

`convex_trajectory_planner.py` currently solves the convex problem with:

```python
problem.solve(solver=cp.ECOS, verbose=False)
```

Therefore, the Python environment needs a working ECOS solver.

## Current Architecture Notes

1. `main.py` is responsible for many things.
   It handles batch execution, logging, orchestration, the V-Shaped algorithm,
   and final result aggregation. A future refactor could move the V-Shaped
   logic into a dedicated planner class.

2. Geometry helper functions are duplicated.
   `cmc_planner.py`, `bob_planner.py`, and `bob_overlap.py` contain similar
   line-circle intersection and closest-point logic. A shared `geometry_utils.py`
   module would reduce duplication.

3. `bob_overlap.py` and `bob_overlap_center.py` are very similar.
   They could be merged into one planner with an `anchor_strategy` option, such
   as `cmc` or `center`.

4. Possible typo in `trajectory_optimizer.py`.
   In `_get_closest_point_on_ellipse()`, this line should be reviewed:

   ```python
   angle = np.arctan2(fop[1] - fip[1], fop[0] - fop[0])
   ```

   The second argument is always zero. It may have been intended to be:

   ```python
   angle = np.arctan2(fop[1] - fip[1], fop[0] - fip[0])
   ```

5. Possible missing import in `analyze_results.py`.
   The script uses `traceback.print_exc()` in the error handler, but
   `traceback` is not imported at the top of the file.

## Suggested Refactor Roadmap

Recommended order:

1. Extract a dedicated `VShapedPlanner`.
2. Add a shared `geometry_utils.py`.
3. Merge BOB-F and BOB-F Center with an anchor strategy option.
4. Standardize return formats across all planners.
5. Extend `reporter.py` to support all methods.
6. Move batch settings to command-line arguments.

These changes would make it easier to add new planners and compare methods
without making `main.py` larger.
