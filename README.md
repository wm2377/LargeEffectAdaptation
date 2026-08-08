# Adaptation from large-effect alleles

Simulation and analytic code for the manuscript figures.

## The pipeline

Everything is driven by a single `Snakefile`. It has three layers:

1. **Simulation rules** run forward-in-time Wright-Fisher simulations and pickle the
   per-replicate output, one pickle per parameter combination. `run_simulation` is the
   multi-allele sweep over the mutational input 2NU; `run_single_site*` follow a single
   large-effect allele; `run_grid_*` cover the phase-space grids. Replicates run across a
   rule's `threads:` and are checkpointed as they finish, so an interrupted job resumes
   from the replicates it already completed.
2. **Analytic rules** evaluate the closed-form expectations over the same parameter grid
   (`generate_analytic_curves`, `generate_fixation_probability`,
   `generate_trajectory_class_boundaries`, ...), again one pickle per combination.
   `generate_min_shift_lookup` precomputes a table that the analytic curves interpolate
   instead of re-solving a bisection per shift.
3. **Plotting rules** (`plot_figure_*`) read those pickles and write a PNG. Each is a thin
   wrapper around one `Figure_*.py` script.

Parameters live at the top of the `Snakefile` (`SDISTS`, `N`, `N_REPLICATES`, the sweep
grids and the per-figure constants); the plotting rules pass them through as `params:`, so
the figure scripts hardcode nothing that the pipeline controls.

The simulation core is C++ (`cpp/sim_core.cpp`, loaded by `simulation_cpp.py`), built on
first use and ~200x faster than the Python reference implementation in
`simulation_classes.py`. Set `SIM_ENGINE=python` to use the reference implementation;
`validate_sim_core.py` checks the two agree.

To rebuild the figures without Snakemake — which may want to rebuild the simulations —
run `python regenerate_figures.py`. It reads the `Snakefile`'s parameter header directly
and writes publication-format TIFFs to `results/plots_submission/`.

## Figures

| Figure | Script | Plotting rule | Upstream rules |
|---|---|---|---|
| 1 | `Figure_1.py` | `plot_figure_1` | — (schematic, no data) |
| 2 | `Figure_2.py` | `plot_figure_2` | — (schematic, no data) |
| 3 | `Figure_3.py` | `plot_figure_3` | `run_single_site`, `run_single_site_with_trajectories`, `generate_fixation_probability` |
| 4 | `Figure_4.py` | `plot_figure_4` | `run_simulation`, `generate_analytic_curves` |
| 5 | `Figure_5.py` | `plot_figure_5` | `run_simulation_full_output` |
| 6 | `Figure_6_S13_S16_S17_S18.py` | `plot_figure_6` | `run_grid_*` → `aggregate_phase_space_data` |
| S3 | `Figure_S3.py` | `plot_figure_S3` | — (closed form, no data) |
| S4 | `Figure_S4.py` | `plot_figure_S4` | `generate_fixation_probability` |
| S5 | `Figure_S5.py` | `plot_figure_S5` | `generate_trajectory_class_boundaries` |
| S6 | `Figure_S6.py` | `plot_figure_S6` | `run_simulation`, `generate_analytic_curves_with_fixed_effect_size_distribution` |
| S7 | `Figure_S7.py` | `plot_figure_S7` | `run_figure_S7_simulation`, `generate_analytic_curves` |
| S8 | `Figure_S8.py` | `plot_figure_S8` | `run_simulation` |
| S9 | `Figure_S9.py` | `plot_figure_S9` | `run_simulation`, `generate_analytic_curves_with_fixed_effect_size_distribution` |
| S10 | `Figure_S10.py` | `plot_figure_S10` | `run_simulation`, `generate_analytic_curves_with_fixed_effect_size_distribution` |
| S11 | `Figure_S11.py` | `plot_figure_S11` | `run_simulation`, `generate_analytic_curves_by_effect_size_window` |
| S12 | `Figure_S12.py` | `plot_figure_S12` | `run_single_site_vs_S`, `generate_fixation_probability`, `calculate_hayward_fixation_probabilities` |
| S13 | `Figure_6_S13_S16_S17_S18.py` | `plot_figure_S13` | `run_grid_*` → `aggregate_phase_space_data` |
| S14 | `Figure_S14.py` | `plot_figure_S14` | `run_grid_*` → `aggregate_phase_space_data` |
| S15 | `Figure_S15.py` | `plot_figure_S15` | — (reads `Simons_2022_SSD_dfe.mat`) |
| S16 | `Figure_6_S13_S16_S17_S18.py` | `plot_figure_S16` | `run_grid_*` → `aggregate_phase_space_data` |
| S17 | `Figure_6_S13_S16_S17_S18.py` | `plot_figure_S17` | `run_grid_*` → `aggregate_phase_space_data` |
| S18 | `Figure_6_S13_S16_S17_S18.py` | `plot_figure_S18` | `run_grid_*` → `aggregate_phase_space_data` |

Figures 6 and S13/S16/S17/S18 are the same four-panel phase space drawn from one script;
they differ only in parameter set, plotted quantity (fixation probability vs adaptive
contribution), smoothing and layout, all passed as `params:`. S14 is the analytic
companion and reads the same `phase_space` pickles.

## Rules behind the figures

| Rule | Script | Writes |
|---|---|---|
| `run_simulation` | `simulation_run.py` | `simulations/` |
| `run_figure_S7_simulation` | `simulation_run.py` | `simulations_figure_S7/` |
| `run_simulation_full_output` | `simulation_run.py` | `simulations_full_output/` |
| `run_single_site` | `single_site_simulation_run.py` | `single_site/` |
| `run_single_site_with_trajectories` | `single_site_simulation_run.py` | `single_site/*_with_trajectories.pkl` |
| `run_single_site_vs_S` | `single_site_simulation_run.py` | `single_site/*_Ssweep.pkl` |
| `run_grid_*` | `run_grid_batch.py` | `simulations_grid/` |
| `phase_space_null_simulations` | `generate_null_phase_space_simulation.py` | `simulations_grid/` (out-of-scope cells) |
| `aggregate_phase_space_data` | `aggregate_phase_space_data.py` | `phase_space/` |
| `generate_analytic_curves` | `calculate_analytic_curves.py` | `analytic_curves/` |
| `generate_analytic_curves_with_fixed_effect_size_distribution` | `calculate_analytic_curves.py` | `analytic_curves/*__with_fixed_effect_size_distribution.pkl` |
| `generate_analytic_curves_by_effect_size_window` | `calculate_analytic_curves.py` | `analytic_curves/*__by_effect_size_window.pkl` |
| `generate_fixation_probability` | `calculate_analytic_curves.py` | `fixation_probability/` |
| `calculate_hayward_fixation_probabilities` | `calculate_hayward_fixation_probability.py` | `hayward_fixation_probability/` |
| `generate_trajectory_class_boundaries` | `calculate_trajectory_class_boundaries.py` | `trajectory_class_boundaries/` |
| `generate_min_shift_lookup` | `build_min_shift_lookup.py` | `lookup_tables/` |

## Supporting modules

| Module | Role |
|---|---|
| `analytic_functions.py` | fixation / establishment expectations, and the distribution of fixed effect sizes |
| `simulation_classes.py` | Python reference implementation of the multi-allele simulation |
| `simulation_cpp.py`, `cpp/sim_core.cpp` | the C++ core and its ctypes bridge |
| `single_site_simulation_classes.py` | single-allele simulation (segregating and new) |
| `classify_trajectory.py` | deterministic trajectory classes and their shift boundaries |
| `calculate_example_trajectories.py` | deterministic and realized example trajectories, imported by Figures 3 and S5 |
| `generate_segregating_mutations.py` | samples the standing variation at the shift from the MSDB steady state, for `simulation_classes.py` |
| `phase_space_grids.py` | the phase-space grid definitions, shared by the Snakefile and the aggregator |
| `common_functions.py` | figure styling, `CurvedText`, figure-sizing helpers |
| `mixture_distribution.py` | mixture effect-size distribution used by S11 and S18 |
| `validate_sim_core.py` | statistical comparison of the C++ and Python engines |
| `cpp/rng.hpp` | xoshiro256** RNG plus the binomial and Poisson samplers the recursion needs |
| `cpp/sdist.hpp` | effect-size distributions and the MSDB steady state, on the C++ side |
| `cpp/Makefile` | builds `libsimcore.so`; invoked automatically on first use |
