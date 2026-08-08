# Large-effect alleles: individual-based and all-allele simulations

Simulation code for Supplementary Figures S1 and S2.

## The pipeline

Two simulation models are implemented, and Figure S1 exists to compare them:

1. **All-allele** (`cpp/all_allele_core.cpp`) tracks allele frequencies under a diffusion
   approximation: each generation an allele's expected frequency change is computed in
   closed form and the realized change drawn from a binomial. Cheap — a full replicate is
   about a second — so it carries the whole parameter grid.
2. **Individual-based** (`cpp/individual_core.cpp`) tracks explicit diploid genotypes:
   2N parents are drawn in proportion to `exp(-(z - optimum)² / 2V_S)`, each contributes a
   gamete under free recombination, and gametes pair into offspring. Drift and selection
   emerge from sampling individuals rather than from a diffusion limit, which is what makes
   the comparison worth drawing — and what makes it far more expensive.

Both cores are C++ behind a plain-C interface, loaded through `ctypes`, built on first use.
`all_allele_model.py` is a numpy implementation of the all-allele model, kept as a readable
reference for what the C++ core does.

A run is: seed the population from the steady-state sojourn density, burn in for `10N`
generations, shift the optimum by `Λ`, record `4N` generations. The C++ cores accumulate
the per-generation summaries the figures need (per-effect-size-bin moments, skew
trajectories, fixation records) as they go, so the per-generation allele records are never
materialised.

Both figures are driven by `Snakefile`, one job per parameter cell; the grid,
replicate count and threads live at the top of it, and `config.yaml` beside it carries only
SLURM submission settings. Finished replicates are skipped, so an interrupted cell resumes
from what it completed.

Every replicate's seed derives from `(σ², 2NU, Λ, replicate)`, so any replicate reproduces
standalone and a partial sweep can be resumed without changing results.

## Figures

| Figure | Script | Rule | Upstream |
|---|---|---|---|
| S1 | `make_figure_S1.py` | `figure_S1` | `combination`, `individual_combination` |
| S2 | `make_figure_S2.py` | `figure_S2` | `combination` |

Figure S1 plots the mean number of aligned large-effect fixations after the shift against
the mutational input, individual-based (`+`) against all-allele (`o`), at σ² = 4. Figure S2
plots the relative additive variance and skew of the small-effect background over time,
split by mutational input, from the all-allele model alone.

## Rules and runners

| Rule / runner | Script | Writes |
|---|---|---|
| `combination` | `run_combination.py` | `data/sigma2_*/LargeN2U_*/shift_*/` |
| `individual_combination` | `run_individual_combination.py` | `data_individual/sigma2_*/LargeN2U_*/shift_*/` |
| `figure_S1` | `make_figure_S1.py` | `figures/Figure_S1.png` |
| `figure_S2` | `make_figure_S2.py` | `figures/Figure_S2.png` |
| — | `run_individual_replicate.py` | one replicate, for running a cell by hand |

Per cell, `run_combination.py` writes `processed_iteration_{n}.pkl` per replicate, then
pools them into `all_processed_results_with_mutation_counts.pkl` (per-generation moments for
both effect-size bins, plus each replicate's fixed effect sizes) and `skew_results.pkl`
(pooled skew trajectories for both bins). `run_individual_combination.py` writes
`individual_iteration_{n}.pkl` per replicate and pools them into `individual_summary.pkl`.

## Supporting modules

| Module | Role |
|---|---|
| `all_allele_cpp.py`, `cpp/all_allele_core.cpp` | the all-allele C++ core and its ctypes bridge |
| `individual_cpp.py`, `cpp/individual_core.cpp` | the individual-based C++ core and its bridge |
| `all_allele_model.py` | numpy implementation of the all-allele model (reference) |
| `cpp/bench_individual.cpp` | timing harness for the individual core, run without Python |

## Running

    snakemake --profile .        # both figures, on the cluster
    snakemake --cores 4          # locally
    snakemake --profile . figures/Figure_S2.png    # one figure only
