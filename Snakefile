import os
from scipy import stats
import numpy as np

from mixture_distribution import MixtureDistribution
# (lo, hi] bounds on the squared effect size S = a^2 for Figure S11's three columns.
# Imported from the figure so the analytic curves are integrated over exactly the windows
# the simulated fixations are binned into.
from Figure_S11 import WINDOW_BOUNDS as EFFECT_SIZE_WINDOWS

# directory holding this Snakefile, run_simulations scripts, and the results
base = "/insomnia001/depts/pas_lab/projects/zoonomia/04232026/misha"
RESULTS = os.path.join(base, "results")

# Named effect-size (S) distributions. The chosen name appears in the output
# filename; add entries here to sweep more distributions. (Keep names free of
# underscores so they don't collide with the "_N2U"/"_shift" filename delimiters.)
SDISTS = {
    "expon": stats.expon(loc=100, scale=400),
    # Figure S11: 50% the large-effect exponential, 50% uniform on [10, 100].
    "mixexpunif": MixtureDistribution(
        components=[stats.expon(loc=100, scale=400), stats.uniform(loc=10, scale=90)],
        weights=[0.5, 0.5],
    ),
    # Figure S18's alternative effect-size distribution: uniform on S in [100, 1000].
    # Same lower bound as "expon" but bounded above and with no tail, so it isolates the
    # effect of the tail rather than of the large-effect scale.
    "unif100to1000": stats.uniform(loc=100, scale=900),
}

# Parameter grid + fixed parameters for the sweep
N           = 5000        # Wright-Fisher population sizes
SEED        = 0             # base seed (per-replicate seeds are spawned from it)
STOP_TIME   = int(4*N)           # minimum generations per replicate
N_REPLICATES = 10000           # independent replicates per parameter combination

# Replicates per cell of the Figure 6 / S13 / S16-S18 phase-space grids
PHASE_SPACE_N_REPLICATES = 1024

# These multipliers raise the replicate count for the parameter combinations
# when fixations are rare, so that we can accurately estimate the distribution of fixed effect sizes
FIXATION_POOR_BOOST = {(50, 80): 50, (80, 80): 5}


def n_replicates_for(n2u, shift, sigma2):
    """Replicates for one figure-sweep (2NU, shift, sigma2) simulation.

    A pickle's name carries no replicate count, so the count has to be decided from the
    parameters alone. 
    """
    boost = FIXATION_POOR_BOOST.get((int(shift), int(sigma2)), 1)
    if n2u > 1:
        boost = 1
    elif n2u > 0.1:
        boost = min(boost, 10)
    return N_REPLICATES * boost

# When True, generate_analytic_curves reads min_shift_new from the precomputed
# lookup table (generate_min_shift_lookup), interpolating linearly in S, instead
# of re-solving the per-S bisection for every shift.
USE_MIN_SHIFT_LOOKUP = True

# When True, generate_analytic_curves additionally computes the distribution of
# effect sizes among fixed alleles. OFF by default
COMPUTE_FIXED_EFFECT_SIZE_DISTRIBUTION = False

# Filename-stable numeric tokens. str(np.float64) is not stable across numpy versions
def fnum(x):
    return str(float(format(float(x), ".12g")))

# Population-scaled mutation input (2NU) for the large-effect alleles
N2U_GRID  = [fnum(x) for x in np.logspace(-3, 3, 31)]
N2U_SWEEP = N2U_GRID

# Mutational inputs Figures 4 and S7 are drawn at:
FIGURE_MAIN_N2U = fnum(0.01)
FIGURE_ALT_N2U  = fnum(0.02)

# number of replicates for Figure S7
FIGURE_S7_N_REPLICATES = 100000

# Single-site sweep: one large-effect allele (squared effect size S, sign +1/-1) on
# the infinitesimal background, run via single_site_simulation_run.py. 
SINGLE_SITE_MODES  = ["segregating", "new"]  
SINGLE_SITE_SIGNS  = ["pos"]                 
SINGLE_SITE_S      = [200]                   
SINGLE_SITE_SHIFT  = list(np.linspace(0, 100, 41))  
SINGLE_SITE_SIGMA2 = [40]                   
# when True, every replicate saves its full (x, D) trajectory; when False, only fix/loss outcomes are stored
SINGLE_SITE_RECORD_TRAJECTORIES = False

# replicates per parameter combination
SINGLE_SITE_N_REPLICATES = {"segregating": 16000, "new": 64000}

# Single-site fixation-probability-vs-S 
SINGLE_SITE_S_SWEEP = [fnum(x) for x in np.logspace(-3, 3, 26)]
SINGLE_SITE_S_SWEEP_SCENARIOS = [(80, 80), (50, 80), (80, 40)]   # (shift, sigma2)
SINGLE_SITE_S_SWEEP_N_REPLICATES = 5000

# Figure 3, panels D/E
FIGURE_3_ANALYTIC_SHIFT = list(np.linspace(0, 100, 101))
FIGURE_3_SIGMA2_LIGHT   = 100
# Figure 3, panels A-C
FIGURE_3_TRAJECTORY_SHIFT = [30, 100]

# Figure 5
FIGURE_5_SDIST        = "expon"
FIGURE_5_N2U          = 10
FIGURE_5_SHIFT        = 80
FIGURE_5_SIGMA2       = 40
FIGURE_5_N_REPLICATES = 100

# Figure S5
FIGURE_S5_S      = 200.0
FIGURE_S5_SIGMA2 = [fnum(x) for x in np.unique(np.concatenate([
    np.logspace(np.log10(0.01), np.log10(10), 40),
    np.linspace(10, 100, 40),
]))]

# Figure 6 + S13/S16/S17/S18
PHASE_SPACE_PARAMSETS = {
    "default":  dict(chosen_shiftA=80, chosen_sigma2=80, chosen_p=0.5, chosen_shiftC=80,
                     sdist="expon"),
    "alt":      dict(chosen_shiftA=50, chosen_sigma2=40, chosen_p=0.1, chosen_shiftC=50,
                     sdist="expon"),
    # Figure S18. Identical to "default" except for the effect-size distribution, so the
    # two figures differ in exactly one variable.
    "altsdist": dict(chosen_shiftA=80, chosen_sigma2=80, chosen_p=0.5, chosen_shiftC=80,
                     sdist="unif100to1000"),
}

# The exact (2NU, sigma2, shift) grids each phase-space panel is simulated on live in
# phase_space_grids.py (shared with aggregate_phase_space_data.py so file names and panel
# placement cannot drift): panel A over (2NU x sigma2), B over (2NU x shift), C over
# (p x total variance V_A), D over (V_A x shift/sqrt(V_A)) -- with C/D's simulation
# parameters derived per cell. Panel D cells whose ABSOLUTE shift exceeds the scope cutoff
# (Lambda > sqrt(V_S)) are not simulated; they get a null placeholder (simulations_null/).
import phase_space_grids as psg


def phase_space_input_files(paramset):
    """Every simulation / null pickle rule aggregate_phase_space_data needs for one
    parameter set (absolute paths)."""
    ps = PHASE_SPACE_PARAMSETS[paramset]
    return [os.path.join(RESULTS, rel) for rel in psg.all_input_relpaths(ps)]


# Every wildcard may only ever contain alphanumeric characters or '.', never '_'. 
wildcard_constraints:
    sdist  = r"[A-Za-z0-9.]+",
    N2U    = r"[A-Za-z0-9.]+",
    shift  = r"[A-Za-z0-9.]+",
    sigma2 = r"[A-Za-z0-9.]+",
    # single-site wildcards. 'sign' is encoded pos/neg (not +1/-1) because '-' is
    # neither in the delimiter-safe alphabet nor a legal wildcard character.
    mode   = r"segregating|new",
    sign   = r"pos|neg",
    S      = r"[A-Za-z0-9.]+",
    # phase-space (Figure 6 / S13 / S16-18) parameter-set name, e.g. "default" / "alt"
    paramset = r"[A-Za-z0-9]+",


# Run the Figure S4 plotting job on the local/head node instead of submitting it as a
# cluster job (it is a quick, light plot that just assembles the fixation_probability
# pickles).
localrules: plot_figure_S4, plot_figure_S11, plot_figure_S12,
    plot_figure_6, plot_figure_S13, plot_figure_S14, plot_figure_S16,
    plot_figure_S17, plot_figure_S18, phase_space_null_simulations


rule all:
    input:
        os.path.join(RESULTS, "plots/Figure_1.png"),
        os.path.join(RESULTS, "plots/Figure_2.png"),
        os.path.join(RESULTS, "plots/Figure_3.png"),
        expand(os.path.join(RESULTS, "plots/Figure_4__{sdist}_N2U{N2U}_shift{shift}_sigma2{sigma2}.png"),sdist=['expon'],N2U=[FIGURE_MAIN_N2U],shift=[80],sigma2=[40],),
        os.path.join(RESULTS, "plots/Figure_5.png"),
        os.path.join(RESULTS, "plots/Figure_6.png"),
        os.path.join(RESULTS, "plots/Figure_S3.png"),
        expand(os.path.join(RESULTS, "plots/Figure_S4__S{S}_sigma2{sigma2}.png"),S=[200],sigma2=[40],),
        os.path.join(RESULTS, "plots/Figure_S5.png"),
        expand(os.path.join(RESULTS, "plots/Figure_S6__{sdist}.png"),sdist=['expon']),
        expand(os.path.join(RESULTS, "plots/Figure_S7__{sdist}_{N2U}_{sigma2}.png"),sdist=['expon'],N2U=[FIGURE_MAIN_N2U],sigma2=[40],),
        expand(os.path.join(RESULTS, "plots/Figure_S8__{sdist}.png"),sdist=['expon']),
        expand(os.path.join(RESULTS, "plots/Figure_S9__{sdist}.png"),sdist=['expon']),
        expand(os.path.join(RESULTS, "plots/Figure_S10__{sdist}.png"),sdist=['expon']),
        expand(os.path.join(RESULTS, "plots/Figure_S11__{sdist}.png"), sdist = ['mixexpunif']),
        os.path.join(RESULTS, "plots/Figure_S12.png"),
        os.path.join(RESULTS, "plots/Figure_S13.png"),
        os.path.join(RESULTS, "plots/Figure_S14.png"),
        os.path.join(RESULTS, "plots/Figure_S15.png"),
        os.path.join(RESULTS, "plots/Figure_S16.png"),
        os.path.join(RESULTS, "plots/Figure_S17.png"),
        os.path.join(RESULTS, "plots/Figure_S18.png"),
        
        
        
rule run_simulation:
    # Replicates run in the C++ core (cpp/sim_core.cpp), which simulation_run.py builds
    # on first use and drives through simulation_cpp.py
    output:
        update(os.path.join(RESULTS, "simulations/sdist{sdist}_N2U{N2U}_shift{shift}_sigma2{sigma2}.pkl"))
    params:
        N              = N,
        sdist          = lambda wc: SDISTS[wc.sdist],
        N2U            = lambda wc: float(wc.N2U),
        sigma2         = lambda wc: float(wc.sigma2),
        shift          = lambda wc: float(wc.shift),
        n_replicates   = lambda wc: n_replicates_for(float(wc.N2U), float(wc.shift),
                                                     float(wc.sigma2)),
        stop_time      = STOP_TIME,
        record_moments = False,
        seed           = SEED,
    threads: 16
    resources:
        mem_mb    = lambda wc, threads: 512 * threads,
        time      = 720,
        partition = "short",
    script:
        "simulation_run.py"


rule run_figure_S7_simulation:
    # Exactly the simulation run_simulation performs -- same engine, same parameters, same
    # script -- but at FIGURE_S7_N_REPLICATES instead of n_replicates_for(), and written to
    # its own directory so the two counts cannot collide in one filename (a pickle's name
    # carries no replicate count). Figure S7 is the only consumer.
    output:
        update(os.path.join(
            RESULTS, "simulations_figure_S7/sdist{sdist}_N2U{N2U}_shift{shift}_sigma2{sigma2}.pkl"))
    params:
        N              = N,
        sdist          = lambda wc: SDISTS[wc.sdist],
        N2U            = lambda wc: float(wc.N2U),
        sigma2         = lambda wc: float(wc.sigma2),
        shift          = lambda wc: float(wc.shift),
        n_replicates   = FIGURE_S7_N_REPLICATES,
        stop_time      = STOP_TIME,
        record_moments = False,
        seed           = SEED,
    threads: 16
    resources:
        mem_mb    = lambda wc, threads: 512 * threads,
        time      = 720,
        partition = "short",
    script:
        "simulation_run.py"


# --------------------------------------------------------------------------------------- #
# Phase-space grid simulations (Figure 6 / S13 / S16-S18), run in batches
# --------------------------------------------------------------------------------------- #

# The grid deliberately does NOT reuse run_simulation: its pickles go to simulations_grid/,
# so the coarse grid cells cannot collide with the (much more precise) figure-sweep pickles
# in simulations/, and every cell runs at a flat PHASE_SPACE_N_REPLICATES.

GRID_BATCH_THREADS             = 16
GRID_BATCH_BUDGET_CORE_SECONDS = 4 * 3600 * GRID_BATCH_THREADS   # ~4 h per job on 16 cores

GRID_BATCHES = psg.grid_batches(PHASE_SPACE_PARAMSETS, PHASE_SPACE_N_REPLICATES,
                                GRID_BATCH_BUDGET_CORE_SECONDS)

# Each batch holds cells for a single (parameter set, panel) group -- so a batch is one
# effect-size distribution and one figure panel -- and is named run_grid_<paramset>_<panel>_<k>
# (e.g. run_grid_default_A_00; shared cells get a joined owner label). See grid_batches().
for _batch in GRID_BATCHES:
    rule:
        name:
            f"run_grid_{_batch['name']}"
        output:
            [update(os.path.join(RESULTS, _rp)) for _rp, _c in _batch["cells"]]
        params:
            # one entry per cell, in run order; run_grid_batch.py loops over them
            cells = [dict(path=os.path.join(RESULTS, _rp), sdist=SDISTS[_c["sdist"]],
                          N2U=_c["N2U"], sigma2=_c["sigma2"], shift=_c["shift"])
                     for _rp, _c in _batch["cells"]],
            N              = N,
            n_replicates   = PHASE_SPACE_N_REPLICATES,
            stop_time      = STOP_TIME,
            record_moments = False,
            seed           = SEED,
        threads: GRID_BATCH_THREADS
        resources:
            # the figure sweeps at 2NU = 1000 peaked at ~1.6 GB on 16 cores; a batch runs
            # its cells one at a time, so its peak is one cell's peak
            mem_mb    = lambda wc, threads: 192 * threads,
            time      = 720,
            partition = "short",
        script:
            "run_grid_batch.py"

# Convenience aggregation targets so a subset of the grid can be run on its own without
# naming individual batches: one per effect-size distribution (grid_expon, grid_mixexpunif),
# per parameter set (grid_default, grid_alt, grid_altsdist), and per panel
# (grid_default_A, grid_default_B, ...). Each expands to just that group's cell pickles.
#   e.g.  snakemake grid_expon      # every expon batch
#         snakemake grid_default_A  # only Figure 6 panel A
for _tgt, _relpaths in psg.grid_group_targets(GRID_BATCHES).items():
    rule:
        name:
            _tgt
        input:
            [os.path.join(RESULTS, _rp) for _rp in _relpaths]


rule run_simulation_full_output:
    # Same simulation as run_simulation, but with full_output=True,
    output:
        update(os.path.join(
            RESULTS, "simulations_full_output/sdist{sdist}_N2U{N2U}_shift{shift}_sigma2{sigma2}.pkl"))
    params:
        N                = N,
        sdist            = lambda wc: SDISTS[wc.sdist],
        N2U              = lambda wc: float(wc.N2U),
        sigma2           = lambda wc: float(wc.sigma2),
        shift            = lambda wc: float(wc.shift),
        n_replicates     = FIGURE_5_N_REPLICATES,
        stop_time        = STOP_TIME,
        record_moments   = False,
        full_output      = True,
        checkpoint_every = 25,
        seed             = SEED,
    threads: 16
    resources:
        mem_mb    = lambda wc, threads: 1024 * threads,
        time      = 720,
        partition = "short",
    script:
        "simulation_run.py"


rule run_single_site:
    output:
        update(os.path.join(
            RESULTS, "single_site/mode{mode}_S{S}_sign{sign}_shift{shift}_sigma2{sigma2}.pkl"))
    params:
        mode         = lambda wc: wc.mode,
        N            = N,
        S            = lambda wc: float(wc.S),
        # decode the delimiter-safe sign token back to +1 / -1
        sign         = lambda wc: 1 if wc.sign == "pos" else -1,
        sigma2       = lambda wc: float(wc.sigma2),
        shift        = lambda wc: float(wc.shift),
        n_replicates = lambda wc: SINGLE_SITE_N_REPLICATES[wc.mode],
        record_trajectories = False,
        seed         = SEED,
    threads: 16
    resources:
        mem_mb    = lambda wc, threads: 512 * threads,
        time      = 720,
        partition = "short",
    script:
        "single_site_simulation_run.py"

rule run_single_site_with_trajectories:
    output:
        update(os.path.join(
            RESULTS, "single_site/mode{mode}_S{S}_sign{sign}_shift{shift}_sigma2{sigma2}_with_trajectories.pkl"))
    params:
        mode         = lambda wc: wc.mode,
        N            = N,
        S            = lambda wc: float(wc.S),
        sign         = lambda wc: 1 if wc.sign == "pos" else -1,
        sigma2       = lambda wc: float(wc.sigma2),
        shift        = lambda wc: float(wc.shift),
        x0           = lambda wc: 1.0 / (2.0 * float(wc.S)) if wc.mode == "segregating" else None,
        t0           = lambda wc: 0 if wc.mode == "new" else None,
        n_replicates = 1000,
        record_trajectories = True,
        seed         = SEED,
    threads: 16
    resources:
        mem_mb    = lambda wc, threads: 512 * threads,
        time      = 720,
        partition = "short",
    script:
        "single_site_simulation_run.py"


rule run_single_site_vs_S:
    output:
        update(os.path.join(
            RESULTS, "single_site/mode{mode}_S{S}_sign{sign}_shift{shift}_sigma2{sigma2}_Ssweep.pkl"))
    params:
        mode         = lambda wc: wc.mode,
        N            = N,
        S            = lambda wc: float(wc.S),
        sign         = lambda wc: 1 if wc.sign == "pos" else -1,
        sigma2       = lambda wc: float(wc.sigma2),
        shift        = lambda wc: float(wc.shift),
        n_replicates = SINGLE_SITE_S_SWEEP_N_REPLICATES,
        record_trajectories = False,
        seed         = SEED,
    threads: 16
    resources:
        mem_mb    = lambda wc, threads: 512 * threads,
        time      = 720,
        partition = "short",
    script:
        "single_site_simulation_run.py"


rule generate_min_shift_lookup:
    output:
        os.path.join(RESULTS, "lookup_tables/sdist{sdist}_N%d_sigma2{sigma2}.npz" % N)
    params:
        N      = N,
        sdist  = lambda wc: SDISTS[wc.sdist],
        sigma2 = lambda wc: float(wc.sigma2),
        n_grid = 200,
        grid_scale = 'quantile',
    threads: 16
    resources:
        mem_mb    = lambda wc, threads: 512 * threads,
        time      = 720,
        partition = "short",
    script:
        "build_min_shift_lookup.py"


rule generate_analytic_curves:
    # When USE_MIN_SHIFT_LOOKUP is set, depend on (and consume) the precomputed
    # min_shift table so the new-allele curve interpolates the threshold in S
    # instead of re-solving min_shift_new per shift.
    input:
        **({"min_shift_lookup": os.path.join(
                RESULTS, "lookup_tables/sdist{sdist}_N%d_sigma2{sigma2}.npz" % N)}
           if USE_MIN_SHIFT_LOOKUP else {})
    output:
        os.path.join(RESULTS, "analytic_curves/sdist{sdist}_shift{shift}_sigma2{sigma2}.pkl")
    params:
        N                    = N,
        sdist                = lambda wc: SDISTS[wc.sdist],
        sigma2               = lambda wc: float(wc.sigma2),
        shift                = lambda wc: float(wc.shift),
        use_min_shift_lookup = USE_MIN_SHIFT_LOOKUP,
        compute_fixed_effect_size_distribution = False,
    threads: 1
    resources:
        mem_mb    = lambda wc, threads: 512 * threads,
        time      = 720,
        partition = "short",
    script:
        "calculate_analytic_curves.py"

rule generate_analytic_curves_with_fixed_effect_size_distribution:
    # When USE_MIN_SHIFT_LOOKUP is set, depend on (and consume) the precomputed
    # min_shift table so the new-allele curve interpolates the threshold in S
    # instead of re-solving min_shift_new per shift.
    input:
        **({"min_shift_lookup": os.path.join(
                RESULTS, "lookup_tables/sdist{sdist}_N%d_sigma2{sigma2}.npz" % N)}
           if USE_MIN_SHIFT_LOOKUP else {})
    output:
        os.path.join(RESULTS, "analytic_curves/sdist{sdist}_shift{shift}_sigma2{sigma2}__with_fixed_effect_size_distribution.pkl")
    params:
        N                    = N,
        sdist                = lambda wc: SDISTS[wc.sdist],
        sigma2               = lambda wc: float(wc.sigma2),
        shift                = lambda wc: float(wc.shift),
        use_min_shift_lookup = USE_MIN_SHIFT_LOOKUP,
        compute_fixed_effect_size_distribution = True,
    threads: 1
    resources:
        mem_mb    = lambda wc, threads: 512 * threads,
        time      = 720,
        partition = "short",
    script:
        "calculate_analytic_curves.py"


rule generate_analytic_curves_by_effect_size_window:
    # Figure S11's analytic curves: the same expectations as generate_analytic_curves, but
    # with the integral over the effect-size distribution restricted to each of the three
    # EFFECT_SIZE_WINDOWS in turn (S > 100, 30 < S <= 100, 10 < S <= 30), so every column
    # of the figure gets its own curve. Output is keyed by window name, each holding the
    # usual {'segregating', 'new', 'all'} split, per unit of the TOTAL mutational input.
    input:
        **({"min_shift_lookup": os.path.join(
                RESULTS, "lookup_tables/sdist{sdist}_N%d_sigma2{sigma2}.npz" % N)}
           if USE_MIN_SHIFT_LOOKUP else {})
    output:
        os.path.join(RESULTS, "analytic_curves/sdist{sdist}_shift{shift}_sigma2{sigma2}__by_effect_size_window.pkl")
    params:
        N                    = N,
        sdist                = lambda wc: SDISTS[wc.sdist],
        sigma2               = lambda wc: float(wc.sigma2),
        shift                = lambda wc: float(wc.shift),
        windows              = EFFECT_SIZE_WINDOWS,
        use_min_shift_lookup = USE_MIN_SHIFT_LOOKUP,
    threads: 1
    resources:
        mem_mb    = lambda wc, threads: 512 * threads,
        time      = 720,
        partition = "short",
    script:
        "calculate_analytic_curves.py"


rule generate_fixation_probability:
    # Fixation probability of a large-effect allele of FIXED squared effect size S = a^2
    # (not marginalized over a distribution), split by source, as a function of the
    # optimum-shift size, using the deterministic double recursion for the fixation
    # cutoffs. Reuses calculate_analytic_curves.py, which switches to its fixed-effect-size
    # fixation-probability mode when given an 'S' param instead of 'sdist'
    output:
        os.path.join(RESULTS, "fixation_probability/S{S}_shift{shift}_sigma2{sigma2}.pkl")
    params:
        N      = N,
        S      = lambda wc: float(wc.S),
        sigma2 = lambda wc: float(wc.sigma2),
        shift  = lambda wc: float(wc.shift),
    threads: 1
    resources:
        mem_mb    = lambda wc, threads: 512 * threads,
        time      = 720,
        partition = "short",
    script:
        "calculate_analytic_curves.py"


rule generate_trajectory_class_boundaries:
    # Figure S5: for a fixed large effect S = a^2 and initial frequency x0 = 1/a^2,
    # find the four optimum-shift sizes at which the deterministic trajectory class changes 
    output:
        os.path.join(RESULTS, "trajectory_class_boundaries/S{S}_sigma2{sigma2}.pkl")
    params:
        N      = N,
        S      = lambda wc: float(wc.S),
        sigma2 = lambda wc: float(wc.sigma2),
    threads: 1
    resources:
        mem_mb    = 1024,
        time      = 720,
        partition = "short",
    script:
        "calculate_trajectory_class_boundaries.py"

rule calculate_hayward_fixation_probabilities:
    # Figure S12, fixation probabilities calculated using the non-linear lande equation from Hayward et al. 2023.
    # One pickle per parameter combination
    output:
        os.path.join(RESULTS, "hayward_fixation_probability/S{S}_sign{sign}_shift{shift}_sigma2{sigma2}_Hayward.pkl")
    params:
        N      = N,
        S      = lambda wc: float(wc.S),
        sign   = lambda wc: 1 if wc.sign == "pos" else -1,
        Va = lambda wc: float(wc.sigma2),
        shift  = lambda wc: float(wc.shift),
    threads: 1
    resources:
        mem_mb    = 1024,
        time      = 720,
        partition = "short",
    script:
        "calculate_hayward_fixation_probability.py"


rule plot_figure_1:
    output:
        figure_1 = os.path.join(RESULTS, "plots/Figure_1.png"),
    resources:
        mem_mb    = 1024,
        time      = 720,
        partition = "short",
    script:
        "Figure_1.py"


rule plot_figure_2:
    output:
        figure_2 = os.path.join(RESULTS, "plots/Figure_2.png"),
    resources:
        mem_mb    = 1024,
        time      = 720,
        partition = "short",
    script:
        "Figure_2.py"

rule plot_figure_3:
    input:
        simulation_results =
        expand(
            os.path.join(
                RESULTS, "single_site/mode{mode}_S{S}_sign{sign}_shift{shift}_sigma2{sigma2}.pkl"),
            mode=SINGLE_SITE_MODES,
            sign=SINGLE_SITE_SIGNS,
            S=SINGLE_SITE_S,
            shift=SINGLE_SITE_SHIFT,
            sigma2=SINGLE_SITE_SIGMA2,
        ),
        trajectory_results = expand(
            os.path.join(
                RESULTS,
                "single_site/mode{mode}_S{S}_sign{sign}_shift{shift}_sigma2{sigma2}_with_trajectories.pkl"),
            mode=["segregating"],
            sign=["pos"],
            S=[200],
            shift=FIGURE_3_TRAJECTORY_SHIFT,
            sigma2=[40],
        ),
        # double-recursion fixation probability vs shift, at the simulations' sigma2 ...
        fixation_probability = expand(
            os.path.join(RESULTS, "fixation_probability/S{S}_shift{shift}_sigma2{sigma2}.pkl"),
            S=SINGLE_SITE_S,
            shift=FIGURE_3_ANALYTIC_SHIFT,
            sigma2=SINGLE_SITE_SIGMA2,
        ),
        # ... and at the larger background variance (the grey line in panel D)
        fixation_probability_light = expand(
            os.path.join(RESULTS, "fixation_probability/S{S}_shift{shift}_sigma2{sigma2}.pkl"),
            S=SINGLE_SITE_S,
            shift=FIGURE_3_ANALYTIC_SHIFT,
            sigma2=[FIGURE_3_SIGMA2_LIGHT],
        ),
    output:
        figure_3 = os.path.join(RESULTS, "plots/Figure_3.png"),
    params:
        N      = N,
        S      = SINGLE_SITE_S[0],
        sigma2 = SINGLE_SITE_SIGMA2[0],
    resources:
        mem_mb    = 1024,
        time      = 720,
        partition = "short",
    script:
        "Figure_3.py"

rule plot_figure_S3:
    output:
        figure_s3 = os.path.join(RESULTS, "plots/Figure_S3.png"),
    resources:
        mem_mb    = 1024,
        time      = 720,
        partition = "short",
    script:
        "Figure_S3.py"


rule plot_figure_S4:
    input:
        fixation_probability = expand(
            os.path.join(RESULTS, "fixation_probability/S{{S}}_shift{shift}_sigma2{{sigma2}}.pkl"),
            shift=np.linspace(0, 100, 101),
        ),
    output:
        figure_s4 = os.path.join(RESULTS, "plots/Figure_S4__S{S}_sigma2{sigma2}.png"),
    params:
        N      = N,
        S      = lambda wc: float(wc.S),
        sigma2 = lambda wc: float(wc.sigma2),
    resources:
        mem_mb    = 1024,
        time      = 720,
        partition = "short",
    script:
        "Figure_S4.py"


rule plot_figure_S5:
    input:
        boundaries = expand(
            os.path.join(RESULTS, "trajectory_class_boundaries/S{S}_sigma2{sigma2}.pkl"),
            S=[fnum(FIGURE_S5_S)],
            sigma2=FIGURE_S5_SIGMA2,
        ),
    output:
        figure_s5 = os.path.join(RESULTS, "plots/Figure_S5.png"),
    params:
        N = N,
        S = FIGURE_S5_S,
    resources:
        mem_mb    = 2048,
        time      = 720,
        partition = "short",
    script:
        "Figure_S5.py"


rule plot_figure_4:
    input:
        simulations_A = expand(
            os.path.join(RESULTS, "simulations/sdist{{sdist}}_N2U{N2U}_shift{{shift}}_sigma2{{sigma2}}.pkl"),
            N2U = N2U_SWEEP
        ),

        analytic = expand(
            os.path.join(RESULTS, "analytic_curves/sdist{{sdist}}_shift{shift}_sigma2{{sigma2}}.pkl"),
            shift=np.linspace(0,100,101),
        ),
    output:
        figure_4    = os.path.join(RESULTS, "plots/Figure_4__{sdist}_N2U{N2U}_shift{shift}_sigma2{sigma2}.png"),
    params:
        sdist          = lambda wc: SDISTS[wc.sdist],
        shift          = lambda wc: float(wc.shift),
        sigma2         = lambda wc: float(wc.sigma2),
        N2U            = lambda wc: float(wc.N2U),
    resources:
        mem_mb    = 1024,
        time      = 720,
        partition = "short",
    script:
        "Figure_4.py"

rule plot_figure_5:
    input:
        simulations = os.path.join(
            RESULTS,
            "simulations_full_output/sdist{sdist}_N2U{N2U}_shift{shift}_sigma2{sigma2}.pkl".format(
                sdist=FIGURE_5_SDIST, N2U=FIGURE_5_N2U,
                shift=FIGURE_5_SHIFT, sigma2=FIGURE_5_SIGMA2)),
    output:
        figure_5 = os.path.join(RESULTS, "plots/Figure_5.png"),
    resources:
        mem_mb    = 4096,
        time      = 720,
        partition = "short",
    script:
        "Figure_5.py"


rule plot_figure_S6:
    input:
        simulations_A = (expand(
            os.path.join(RESULTS, "simulations/sdist{{sdist}}_N2U{N2U}_shift{shift}_sigma2{sigma2}.pkl"),
            N2U = N2U_SWEEP,
            shift = [80],
            sigma2 = [40,80],
        ) +
        expand(
            os.path.join(RESULTS, "simulations/sdist{{sdist}}_N2U{N2U}_shift{shift}_sigma2{sigma2}.pkl"),
            N2U = N2U_SWEEP,
            shift = [50],
            sigma2 = [80],
        )),

        analytic = (expand(
            os.path.join(RESULTS, "analytic_curves/sdist{{sdist}}_shift{shift}_sigma2{sigma2}__with_fixed_effect_size_distribution.pkl"),
            shift=[80],
            sigma2 = [40,80]
        ) +
        expand(
            os.path.join(RESULTS, "analytic_curves/sdist{{sdist}}_shift{shift}_sigma2{sigma2}__with_fixed_effect_size_distribution.pkl"),
            shift=[50],
            sigma2 = [80]
        )
        ),
    output:
        figure_s6    = os.path.join(RESULTS, "plots/Figure_S6__{sdist}.png"),
    params:
        sdist          = lambda wc: SDISTS[wc.sdist],
    resources:
        mem_mb    = 5120,
        time      = 720,
        partition = "short",
    script:
        "Figure_S6.py"
        
        
rule plot_figure_S7:
    input:
        simulations = expand(
            os.path.join(RESULTS, "simulations_figure_S7/sdist{{sdist}}_N2U{{N2U}}_shift{shift}_sigma2{{sigma2}}.pkl"),
            shift=np.linspace(0,100,42),
        ),
        analytic = expand(
            os.path.join(RESULTS, "analytic_curves/sdist{{sdist}}_shift{shift}_sigma2{{sigma2}}.pkl"),
            shift=np.linspace(0,100,101),
        ),
    output:
        fixations    = os.path.join(RESULTS, "plots/Figure_S7__{sdist}_{N2U}_{sigma2}.png"),
        contribution = os.path.join(RESULTS, "plots/Figure_S7_contribution__{sdist}_{N2U}_{sigma2}.png"),
    params:
        sdist          = lambda wc: SDISTS[wc.sdist],
        N2U            = lambda wc: float(wc.N2U),
        sigma2         = lambda wc: float(wc.sigma2),
    resources:
        mem_mb    = 1024,
        time      = 720,
        partition = "short",
    script:
        "Figure_S7.py"

rule plot_figure_S8:
    input:
        simulations = (
            expand(
                os.path.join(RESULTS, "simulations/sdist{{sdist}}_N2U{N2U}_shift{shift}_sigma2{sigma2}.pkl"),
                N2U = N2U_SWEEP,
                shift = [80],
                sigma2 = [40],
            ) +
            expand(
                os.path.join(RESULTS, "simulations/sdist{{sdist}}_N2U{N2U}_shift{shift}_sigma2{sigma2}.pkl"),
                N2U = N2U_SWEEP,
                shift = [80],
                sigma2 = [80],
            ) +
            expand(
                os.path.join(RESULTS, "simulations/sdist{{sdist}}_N2U{N2U}_shift{shift}_sigma2{sigma2}.pkl"),
                N2U = N2U_SWEEP,
                shift = [50],
                sigma2 = [80],
            )
        )
    output:
        figure_s8 = os.path.join(RESULTS, "plots/Figure_S8__{sdist}.png"),
    params:
        # the (shift, sigma2) combinations are fixed by the input expand above; Figure_S8.py
        # reads the actual shift / sigma2 from each pickle, so no per-wildcard params needed.
        sdist          = lambda wc: SDISTS[wc.sdist],
        min_a          = lambda wc: np.sqrt(SDISTS[wc.sdist].ppf(0.0001)),
    resources:
        mem_mb    = 5120,
        time      = 720,
        partition = "short",
    script:
        "Figure_S8.py"

rule plot_figure_S9:
    input:
        # 2NU = 0.01 simulation pickle per (shift, sigma2) panel (empirical red CDF)
        simulations = (
            expand(os.path.join(RESULTS, "simulations/sdist{{sdist}}_N2U{N2U}_shift{shift}_sigma2{sigma2}.pkl"),
                   N2U = [0.01], shift = [80], sigma2 = [40])
            + expand(os.path.join(RESULTS, "simulations/sdist{{sdist}}_N2U{N2U}_shift{shift}_sigma2{sigma2}.pkl"),
                   N2U = [0.01], shift = [80], sigma2 = [80])
            + expand(os.path.join(RESULTS, "simulations/sdist{{sdist}}_N2U{N2U}_shift{shift}_sigma2{sigma2}.pkl"),
                   N2U = [0.01], shift = [50], sigma2 = [80])
        ),
        # analytic fixed-effect-size distribution per (shift, sigma2) panel (black CDF)
        analytic = (
            expand(os.path.join(RESULTS, "analytic_curves/sdist{{sdist}}_shift{shift}_sigma2{sigma2}__with_fixed_effect_size_distribution.pkl"),
                   shift = [80], sigma2 = [40])
            + expand(os.path.join(RESULTS, "analytic_curves/sdist{{sdist}}_shift{shift}_sigma2{sigma2}__with_fixed_effect_size_distribution.pkl"),
                   shift = [80], sigma2 = [80])
            + expand(os.path.join(RESULTS, "analytic_curves/sdist{{sdist}}_shift{shift}_sigma2{sigma2}__with_fixed_effect_size_distribution.pkl"),
                   shift = [50], sigma2 = [80])
        ),
    output:
        figure_s9 = os.path.join(RESULTS, "plots/Figure_S9__{sdist}.png"),
    params:
        # sdist is the new-mutation distribution g (grey CDF line); shift / sigma2 are read
        # from the pickles, so no per-wildcard params are needed.
        sdist          = lambda wc: SDISTS[wc.sdist],
    resources:
        mem_mb    = 5120,
        time      = 720,
        partition = "short",
    script:
        "Figure_S9.py"

rule plot_figure_S10:
    input:
        simulations = (
            expand(
                os.path.join(RESULTS, "simulations/sdist{{sdist}}_N2U{N2U}_shift{shift}_sigma2{sigma2}.pkl"),
                N2U = N2U_SWEEP, shift = [80], sigma2 = [40],
            )
            + expand(
                os.path.join(RESULTS, "simulations/sdist{{sdist}}_N2U{N2U}_shift{shift}_sigma2{sigma2}.pkl"),
                N2U = N2U_SWEEP, shift = [80], sigma2 = [80],
            )
            + expand(
                os.path.join(RESULTS, "simulations/sdist{{sdist}}_N2U{N2U}_shift{shift}_sigma2{sigma2}.pkl"),
                N2U = N2U_SWEEP, shift = [50], sigma2 = [80],
            )
        ),
        # one analytic pickle per panel (shift, sigma2); each supplies that panel's analytic
        # fixed-effect-size distribution g_f(a).
        analytic = (
            expand(os.path.join(RESULTS, "analytic_curves/sdist{{sdist}}_shift{shift}_sigma2{sigma2}__with_fixed_effect_size_distribution.pkl"),
                   shift = [80], sigma2 = [40])
            + expand(os.path.join(RESULTS, "analytic_curves/sdist{{sdist}}_shift{shift}_sigma2{sigma2}__with_fixed_effect_size_distribution.pkl"),
                   shift = [80], sigma2 = [80])
            + expand(os.path.join(RESULTS, "analytic_curves/sdist{{sdist}}_shift{shift}_sigma2{sigma2}__with_fixed_effect_size_distribution.pkl"),
                   shift = [50], sigma2 = [80])
        ),
    output:
        figure_s10 = os.path.join(RESULTS, "plots/Figure_S10__{sdist}.png"),
    params:
        # sdist is the fallback distribution for the g_f(a) box when an analytic pickle
        # lacks fixed_effect_size_distribution; shift / sigma2 are read from the pickles.
        sdist          = lambda wc: SDISTS[wc.sdist],
    resources:
        mem_mb    = 5120,
        time      = 720,
        partition = "short",
    script:
        "Figure_S10.py"


# Figure S11 requests the mixexpunif simulation pickles for N2U_SWEEP x these (shift,
# sigma2) combos.
FIGURE_S11_COMBOS = [(80, 40), (80, 80), (50, 80)]

def figure_s11_simulation_inputs(wildcards):
    # Request every FIGURE_S11_COMBOS x N2U_SWEEP pickle unconditionally, so Snakemake
    # will build any that are missing as dependencies of the figure 
    files = []
    for shift, sigma2 in FIGURE_S11_COMBOS:
        for n2u in N2U_SWEEP:
            files.append(os.path.join(
                RESULTS,
                f"simulations/sdist{wildcards.sdist}_N2U{n2u}_shift{shift}_sigma2{sigma2}.pkl"))
    return files


rule plot_figure_S11:
    input:
        simulations = figure_s11_simulation_inputs,
        analytic = [
            os.path.join(
                RESULTS,
                "analytic_curves/sdist{{sdist}}_shift{shift}_sigma2{sigma2}__by_effect_size_window.pkl"
            ).format(shift=shift, sigma2=sigma2)
            for shift, sigma2 in FIGURE_S11_COMBOS
        ],
    output:
        figure_s11 = os.path.join(RESULTS, "plots/Figure_S11__{sdist}.png"),
    params:
        # sdist is used to weight each window's x-axis (its mutational-input fraction);
        # shift / sigma2 are read from the pickles, so no per-wildcard params are needed.
        sdist          = lambda wc: SDISTS[wc.sdist],
    resources:
        mem_mb    = 5120,
        time      = 720,
        partition = "short",
    script:
        "Figure_S11.py"


rule plot_figure_S12:
    input:
        simulation_pickles = expand(
            os.path.join(RESULTS, "single_site/mode{mode}_S{S}_sign{sign}_shift{shift}_sigma2{sigma2}_Ssweep.pkl"),
            shift = [80, 50],
            sigma2 = [40, 80],
            S = np.logspace(-3,3,31),
            sign = ["pos"],
            mode = ["segregating"]
        ),
        hayward_pickles = expand(
            os.path.join(RESULTS, "hayward_fixation_probability/S{S}_sign{sign}_shift{shift}_sigma2{sigma2}_Hayward.pkl"),
            shift = [80, 50],
            sigma2 = [40, 80],
            S = np.logspace(-3,3,200),
            sign = ["pos"],
        ),
        double_recursion_pickles = expand(
                os.path.join(RESULTS, "fixation_probability/S{S}_shift{shift}_sigma2{sigma2}.pkl"),
                shift = [80, 50],
                sigma2 = [40, 80],
                S = np.logspace(-3,3,200)
            ),
    output:
        figure_s12 = os.path.join(RESULTS, "plots/Figure_S12.png"),
    script:
        "Figure_S12.py"


rule aggregate_phase_space_data:
    input:
        lambda wc: phase_space_input_files(wc.paramset)
    output:
        os.path.join(RESULTS, "phase_space/paramset{paramset}.pkl"),
    params:
        chosen_shiftA = lambda wc: PHASE_SPACE_PARAMSETS[wc.paramset]["chosen_shiftA"],
        chosen_sigma2 = lambda wc: PHASE_SPACE_PARAMSETS[wc.paramset]["chosen_sigma2"],
        chosen_p      = lambda wc: PHASE_SPACE_PARAMSETS[wc.paramset]["chosen_p"],
        chosen_shiftC = lambda wc: PHASE_SPACE_PARAMSETS[wc.paramset]["chosen_shiftC"],
        sdist         = lambda wc: PHASE_SPACE_PARAMSETS[wc.paramset]["sdist"],
        results_dir   = RESULTS,
    resources:
        mem_mb    = 8192,
        time      = 720,
        partition = "short",
    script:
        "aggregate_phase_space_data.py"


# Null placeholders for the phase-space cells that are out of the model's scope (panel D,
# absolute shift > sqrt(V_S)): no simulation is run, a zero-replicate result is written so
# the aggregation can leave those grid positions blank.
PHASE_SPACE_NULL_CELLS = psg.null_cells(PHASE_SPACE_PARAMSETS)


rule phase_space_null_simulations:
    output:
        [os.path.join(RESULTS, _rp) for _rp, _c in PHASE_SPACE_NULL_CELLS],
    params:
        N     = N,
        cells = [dict(path=os.path.join(RESULTS, _rp), N2U=_c["N2U"],
                      shift=_c["shift"], sigma2=_c["sigma2"])
                 for _rp, _c in PHASE_SPACE_NULL_CELLS],
    resources:
        mem_mb    = 512,
        time      = 60,
        partition = "short",
    script:
        "generate_null_phase_space_simulation.py"


rule plot_figure_6:
    input:
        data = os.path.join(RESULTS, "phase_space", "paramsetdefault.pkl"),
    output:
        figure = os.path.join(RESULTS, "plots/Figure_6.png"),
    params:
        mode     = "fixations",
        smoothed = True,
        layout   = "standard",
    resources:
        mem_mb    = 2048,
        time      = 720,
        partition = "short",
    script:
        "Figure_6_S13_S16_S17_S18.py"


rule plot_figure_S13:
    input:
        data = os.path.join(RESULTS, "phase_space", "paramsetdefault.pkl"),
    output:
        figure = os.path.join(RESULTS, "plots/Figure_S13.png"),
    params:
        mode     = "fixations",
        smoothed = False,
        layout   = "standard",
    resources:
        mem_mb    = 2048,
        time      = 720,
        partition = "short",
    script:
        "Figure_6_S13_S16_S17_S18.py"


rule plot_figure_S16:
    input:
        data = os.path.join(RESULTS, "phase_space", "paramsetdefault.pkl"),
    output:
        figure = os.path.join(RESULTS, "plots/Figure_S16.png"),
    params:
        mode     = "adaptation",
        smoothed = True,
        layout   = "standard",
    resources:
        mem_mb    = 2048,
        time      = 720,
        partition = "short",
    script:
        "Figure_6_S13_S16_S17_S18.py"


rule plot_figure_S17:
    input:
        data = os.path.join(RESULTS, "phase_space", "paramsetalt.pkl"),
    output:
        figure = os.path.join(RESULTS, "plots/Figure_S17.png"),
    params:
        mode     = "fixations",
        smoothed = True,
        layout   = "standard",
    resources:
        mem_mb    = 2048,
        time      = 720,
        partition = "short",
    script:
        "Figure_6_S13_S16_S17_S18.py"


rule plot_figure_S18:
    input:
        data = os.path.join(RESULTS, "phase_space", "paramsetaltsdist.pkl"),
    output:
        figure = os.path.join(RESULTS, "plots/Figure_S18.png"),
    params:
        mode     = "fixations",
        smoothed = True,          # ignored by the smoothed_raw layout
        layout   = "smoothed_raw",
    resources:
        mem_mb    = 2048,
        time      = 720,
        partition = "short",
    script:
        "Figure_6_S13_S16_S17_S18.py"


rule plot_figure_S14:
    input:
        data_default = os.path.join(RESULTS, "phase_space", "paramsetdefault.pkl"),
        data_alt     = os.path.join(RESULTS, "phase_space", "paramsetalt.pkl"),
    output:
        figure = os.path.join(RESULTS, "plots/Figure_S14.png"),
    resources:
        mem_mb    = 1024,
        time      = 720,
        partition = "short",
    script:
        "Figure_S14.py"



rule plot_figure_S15:
    # Figure S15: GWAS power to detect a single large-effect fixation vs divergence time --
    # panel A the sample size for 90% power, panel B the variance explained at n=200 -- for
    # three effect-size distributions (exponential mixture, log-uniform, and the empirical
    # Simons et al. SSD) and two background variances. The empirical SSD is read from
    # Simons_2022_SSD_dfe.mat; the rest is computed in closed form. See Figure_S15.py.
    input:
        ssd = os.path.join(base, "Simons_2022_SSD_dfe.mat"),
    output:
        figure_s15 = os.path.join(RESULTS, "plots/Figure_S15.png"),
    resources:
        mem_mb    = 1024,
        time      = 720,
        partition = "short",
    script:
        "Figure_S15.py"

