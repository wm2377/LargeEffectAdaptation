"""
Snakemake script: aggregate the multiple-allele (2NU) simulations into the phase-space
panel grids that the plotting script (Figure_6_S13_S16_S17_S18.py) consumes.

From each simulation pickle only three quantities are used:
    1. the number of replicates,
    2. the number of fixations per replicate  (results["n_fixations"]),
    3. the effect size a and initial frequency x0 of every allele that fixes
       (replicate["fixed_effect_sizes"] / ["fixed_initial_frequencies"]).

From those, per grid cell:
    fixation probability = fraction of replicates with >= 1 large-effect fixation,
    adaptive contribution = mean per replicate of sum(2 a (1 - x0)) over the aligned
        (a > 0) fixed alleles -- the long-term phenotypic distance closed by fixed
        large-effect alleles

Each panel has its own grid, defined by phase_space_grids.py. Cells with no
simulation -- panel C's high-V_A / low-p corner and panel D's out-of-scope shifts --
are left at 0 (white / no fixation).

Input
    the run_simulation / null pickles for one parameter set (declared for the DAG; the
    script re-derives each cell's path via phase_space_grids so placement is exact).
Output
    one pickle per parameter set, {"params": {...}, "A"/"B"/"C"/"D": {"x","y","fixation","adaptation"}}

Run as a Snakemake script, or drive aggregate()/build_panel_from_cells() directly (a test
harness can pass in-memory reduced-simulation dicts via the `results` map).
"""

import os
import sys
import pickle

import numpy as np

# make the sibling phase_space_grids importable when run as a Snakemake script
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
import phase_space_grids as psg

ADAPTATION_MAX = 100.0     # colour-scale ceiling for the adaptive-contribution field


# --------------------------------------------------------------------------- #
# Reducing one simulation pickle down to the three quantities we keep
# --------------------------------------------------------------------------- #
def load_simulation_summary(path_or_obj):
    """Reduce one simulation result to {N2U, shift, sigma2, n_replicates, n_fixations,
    fixed_effect_sizes, fixed_initial_frequencies}.

    Accepts a path to a run_simulation pickle (full format: 'parameters', top-level
    'n_fixations', and per-replicate 'fixed_effect_sizes'/'fixed_initial_frequencies'),
    or an already-reduced dict (same keys as the return value) -- so a test harness can
    pass synthetic cells without writing files.
    """
    r = path_or_obj if isinstance(path_or_obj, dict) else pickle.load(open(path_or_obj, "rb"))

    if "replicates" in r:            # full run_simulation format
        params = r["parameters"]
        reps = r["replicates"]
        n_rep = len(reps)
        n_fix = np.asarray(r["n_fixations"]) if "n_fixations" in r else \
            np.asarray([rep["n_fixations"] for rep in reps])
        if reps:
            a = np.concatenate([np.asarray(rep["fixed_effect_sizes"], dtype=float) for rep in reps])
            x0 = np.concatenate([np.asarray(rep["fixed_initial_frequencies"], dtype=float) for rep in reps])
        else:
            a = np.array([]); x0 = np.array([])
        N2U, shift, sigma2 = params["N2U"], params["shift"], params["sigma2"]
    else:                            # reduced / null format (flat, or under 'parameters')
        params = r.get("parameters", r)
        n_rep = int(r["n_replicates"])
        n_fix = np.asarray(r["n_fixations"])
        a = np.asarray(r["fixed_effect_sizes"], dtype=float)
        x0 = np.asarray(r["fixed_initial_frequencies"], dtype=float)
        N2U = r.get("N2U", params["N2U"])
        shift = r.get("shift", params["shift"])
        sigma2 = r.get("sigma2", params["sigma2"])
    return dict(N2U=float(N2U), shift=float(shift), sigma2=float(sigma2),
                n_replicates=n_rep, n_fixations=n_fix,
                fixed_effect_sizes=a, fixed_initial_frequencies=x0)


def cell_fixation_probability(s):
    """Fraction of replicates with at least one large-effect fixation."""
    if s["n_replicates"] == 0:
        return np.nan
    return float(np.mean(np.asarray(s["n_fixations"]) >= 1))


def cell_adaptation(s):
    """Mean per replicate of the phenotypic distance closed by aligned fixed alleles,
    sum(2 a (1 - x0)) over a > 0, clipped to [0, ADAPTATION_MAX]."""
    if s["n_replicates"] == 0:
        return np.nan
    a = np.asarray(s["fixed_effect_sizes"], dtype=float)
    x0 = np.asarray(s["fixed_initial_frequencies"], dtype=float)
    aligned = a > 0
    contribution = float(np.sum(2.0 * a[aligned] * (1.0 - x0[aligned])))
    return min(contribution / s["n_replicates"], ADAPTATION_MAX)


# --------------------------------------------------------------------------- #
# Panel builder: place each cell's statistic at its (x, y)
# --------------------------------------------------------------------------- #
def _load_cell(cell, results_dir, sdist, results):
    rel = psg.cell_relpath(cell, sdist)
    if results is not None and rel in results:
        return load_simulation_summary(results[rel])
    return load_simulation_summary(os.path.join(results_dir, rel))


def build_panel_from_cells(cells, results_dir, sdist, results=None):
    xs = sorted({round(c["x"], 9) for c in cells})
    ys = sorted({round(c["y"], 9) for c in cells})
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: j for j, v in enumerate(ys)}
    fix = np.zeros((len(ys), len(xs)))
    ad = np.zeros_like(fix)
    for c in cells:
        s = _load_cell(c, results_dir, sdist, results)
        pf, a = cell_fixation_probability(s), cell_adaptation(s)
        i, j = xi[round(c["x"], 9)], yi[round(c["y"], 9)]
        fix[j, i] = 0.0 if np.isnan(pf) else pf     # null cell -> 0
        ad[j, i] = 0.0 if np.isnan(a) else a
    return dict(x=np.asarray(xs, dtype=float), y=np.asarray(ys, dtype=float),
                fixation=fix, adaptation=ad)


def aggregate(params, results_dir, sdist, results=None):
    cells = psg.panel_cells(params)
    return {"params": dict(params),
            "A": build_panel_from_cells(cells["A"], results_dir, sdist, results),
            "B": build_panel_from_cells(cells["B"], results_dir, sdist, results),
            "C": build_panel_from_cells(cells["C"], results_dir, sdist, results),
            "D": build_panel_from_cells(cells["D"], results_dir, sdist, results)}


def make_phase_space_data(out_path, params, results_dir, sdist, results=None):
    data = aggregate(params, results_dir, sdist, results)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump(data, fh)
    return data


if "snakemake" in globals():
    par = snakemake.params      # noqa: F821
    make_phase_space_data(
        snakemake.output[0],    # noqa: F821
        params=dict(chosen_shiftA=float(par.chosen_shiftA), chosen_sigma2=float(par.chosen_sigma2),
                    chosen_p=float(par.chosen_p), chosen_shiftC=float(par.chosen_shiftC)),
        results_dir=par.results_dir, sdist=par.sdist)
elif __name__ == "__main__":
    print(__doc__)
    print("This module is driven by the Snakefile (rule aggregate_phase_space_data) or "
          "imported; it has no standalone CLI because it needs a grid of simulation inputs.")
    sys.exit(0)
