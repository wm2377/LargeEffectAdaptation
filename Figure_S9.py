"""
Snakemake script: Figure S9, three panels of the cumulative distribution of squared
effect sizes (S = a^2) of *fixed* large-effect alleles at a low mutational input
(2NU = 0.01). It is the CDF companion of the Figure S10 boxplots and shares its three
(shift, sigma2) panels:

    A : Lambda = 80, sigma2 = 40      (permissive)
    B : Lambda = 80, sigma2 = 80      (permissive)
    C : Lambda = 50, sigma2 = 80      (restrictive)

Each panel overlays three CDFs over S:
    1) analytic CDF of fixed effect sizes  -- black solid, from the analytic pickle's
       'fixed_effect_size_distribution'. Independent of 2NU.
    2) CDF of new-mutation effect sizes g  -- grey solid (sdist.cdf).
    3) empirical CDF from simulations      -- red dotted, pooled over all fixed alleles,
       with a red band for the bootstrap spread.

The black line is omitted (with a stderr note) when the analytic pickle lacks the
fixed-effect-size distribution; the red line/band when no matching simulation is given.

Input
    simulations : 2NU = 0.01 simulation pickles, one per (shift, sigma2) panel
    analytic    : one analytic_curves pickle per (shift, sigma2) panel
Params
    sdist : the new-mutation effect-size (S) distribution g (grey CDF line)
Output
    figure_s9 : the three-panel PNG
"""

import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.legend_handler import HandlerTuple

import common_functions as cf


FONTSIZE = 12
N2U_TARGET = 0.01              # the mutational input this figure is drawn at
N_BOOTSTRAP = 10               # bootstrap resamples for the empirical-CDF error band

# panels match Figure S10 (permissive A,B then restrictive C)
PANELS = [
    {"shift": 80, "sigma2": 40, "letter": "A"},
    {"shift": 80, "sigma2": 80, "letter": "B"},
    {"shift": 50, "sigma2": 80, "letter": "C"},
]

# squared-effect-size (S = a^2) plotting range and grid (log x, ticks only at 1e2/1e3)
X_MIN, X_MAX = 100.0, 3000.0
GRID = np.geomspace(X_MIN, X_MAX, 300)

ANALYTIC_COLOR = "k"           # analytic fixed-effect CDF
NEW_COLOR = "0.6"              # new-mutation (sdist) CDF
SIM_COLOR = "red"             # empirical (simulated) fixed-effect CDF


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def analytic_cdf(paths):
    """{(shift, sigma2): (S_grid, cdf)} of the analytic fixed-effect-size CDF.

    Pickles whose 'fixed_effect_size_distribution' is absent are skipped.
    """
    out = {}
    for p in paths:
        with open(p, "rb") as f:
            res = pickle.load(f)
        prm = res["parameters"]
        dist = res.get("fixed_effect_size_distribution")
        if dist is not None:
            out[(prm["shift"], prm["sigma2"])] = (np.asarray(dist["S_grid"], dtype=float),
                                                  np.asarray(dist["cdf"], dtype=float))
    return out


def empirical_fixed_S(paths, n2u_target=N2U_TARGET):
    """{(shift, sigma2): array of pooled fixed S = a^2} from 2NU = n2u_target sims.

    Pooled across replicates; pickles at a different 2NU are ignored.
    """
    out = {}
    for p in paths:
        with open(p, "rb") as f:
            res = pickle.load(f)
        prm = res["parameters"]
        if not np.isclose(prm["N2U"], n2u_target, rtol=1e-3):
            continue
        arrs = [np.asarray(rep["fixed_effect_sizes"], dtype=float) for rep in res["replicates"]]
        a = np.concatenate(arrs) if arrs else np.array([])
        S = a ** 2
        S = S[S > 0]
        out.setdefault((prm["shift"], prm["sigma2"]), []).append(S)
    return {k: np.concatenate(v) for k, v in out.items() if len(v)}


def _ecdf_on_grid(S_values, grid):
    """Empirical CDF of S_values evaluated at each point of `grid`."""
    S_sorted = np.sort(S_values)
    return np.searchsorted(S_sorted, grid, side="right") / S_sorted.size


def empirical_cdf_with_band(S_values, grid, n_boot=N_BOOTSTRAP, rng=None):
    """Empirical CDF on `grid` plus a bootstrap min/max band.

    The band is the pointwise [min, max] over `n_boot` bootstrap resamples.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    main = _ecdf_on_grid(S_values, grid)
    n = S_values.size
    boots = np.array([_ecdf_on_grid(rng.choice(S_values, size=n, replace=True), grid)
                      for _ in range(n_boot)])
    return main, boots.min(axis=0), boots.max(axis=0)


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def _draw_panel(ax, panel, analytic, emp_S, sdist, is_first, rng):
    key = (panel["shift"], panel["sigma2"])

    # 2) grey: CDF of new-mutation effect sizes g = sdist (same in every panel)
    if sdist is not None:
        ax.plot(GRID, sdist.cdf(GRID), color=NEW_COLOR, lw=1.5, zorder=2)

    # 1) black: analytic CDF of fixed effect sizes
    if key in analytic:
        S_grid, cdf = analytic[key]
        ax.plot(S_grid, cdf, color=ANALYTIC_COLOR, lw=1.5, zorder=3)

    # 3) red dotted: empirical CDF from simulations, with a bootstrap error band
    if key in emp_S and emp_S[key].size:
        main, lo, hi = empirical_cdf_with_band(emp_S[key], GRID, N_BOOTSTRAP, rng)
        ax.fill_between(GRID, lo, hi, color=SIM_COLOR, alpha=0.25, lw=0, zorder=1)
        ax.plot(GRID, main, color=SIM_COLOR, ls=":", lw=1.5, zorder=4)

    # axes: log x (ticks only at 1e2 / 1e3), linear y in [0, 1] (ticks at 0, 1/2, 1)
    ax.set_xscale("log")
    ax.set_xlim([X_MIN, X_MAX])
    ax.set_xticks([1e2, 1e3])
    ax.set_xticklabels([r"$10^2$", r"$10^3$"], fontsize=FONTSIZE)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.set_xlabel("Effect size squared" + "\n" + r"$S=a^2$", fontsize=FONTSIZE, labelpad=0)

    ax.set_ylim([0, 1])
    ax.set_yticks([0, 0.5, 1])
    if is_first:
        ax.set_yticklabels(["0", r"$\frac{1}{2}$", "1"], fontsize=FONTSIZE)
        ax.set_ylabel("CDF value", fontsize=FONTSIZE, labelpad=2)
    else:
        ax.set_yticklabels([])

    ax.set_title(rf"$\bf{{{panel['letter']}.}}$ "
                 rf"$\Lambda = {panel['shift']:g},\ \sigma^2 = {panel['sigma2']:g}$",
                 fontsize=FONTSIZE, loc="left")


def _draw_background(fig):
    """Permissive/Restrictive rounded background boxes + labels."""
    first_xy, first_wh = (-0.072, -0.24), (0.741, 1.42)
    second_xy = (first_xy[0] + first_wh[0], first_xy[1])
    second_wh = (1.01 - second_xy[0], first_wh[1])
    box_kw = dict(boxstyle="Round, pad=0,rounding_size=0.05", mutation_aspect=2,
                  clip_on=False, transform=fig.transFigure, figure=fig, zorder=-1)
    fig.patches.append(FancyBboxPatch(first_xy, *first_wh,
                                      facecolor=cf.colorA_face, edgecolor=cf.colorA_edge, **box_kw))
    fig.patches.append(FancyBboxPatch(second_xy, *second_wh,
                                      facecolor=cf.colorB_face, edgecolor=cf.colorB_edge, **box_kw))
    fig.text((first_xy[0] * 2 + first_wh[0]) / 2, first_wh[1] + first_xy[1] - 0.08,
             "Permissive", fontweight="bold", color=cf.colorA / 2, fontsize=FONTSIZE, ha="center")
    fig.text((second_xy[0] * 2 + second_wh[0]) / 2, second_wh[1] + second_xy[1] - 0.08,
             "Restrictive", fontweight="bold", color=cf.colorB / 2, fontsize=FONTSIZE, ha="center")


# --------------------------------------------------------------------------- #
# Assemble the figure
# --------------------------------------------------------------------------- #
def make_figure_S9(out_path, analytic, emp_S, sdist, figsize=(10, 3.5)):
    fig = plt.figure(figsize=figsize, dpi=400)

    # three equal panels; the B|C gap is widened to separate the permissive and
    # restrictive groups
    spacing = 0.02
    extra = 0.02                       # additional space between panels B and C
    width = (1 - spacing * 3 - extra) / 3
    axes = [fig.add_axes([spacing + i * (width + spacing) + (extra if i >= 2 else 0),
                          0.05, width, 0.9]) for i in range(3)]

    rng = np.random.default_rng(0)   # reproducible bootstrap across panels
    for i, (ax, panel) in enumerate(zip(axes, PANELS)):
        _draw_panel(ax, panel, analytic, emp_S, sdist, is_first=(i == 0), rng=rng)

    _draw_background(fig)

    # boxed legend below the figure. The simulation entry combines the bootstrap
    # sleeve and the dotted mean line into one handle via HandlerTuple.
    sim_handle = (Patch(facecolor=SIM_COLOR, alpha=0.25, linewidth=0),
                  Line2D([], [], color=SIM_COLOR, ls=":", lw=1.5))
    handles = [sim_handle,
               Line2D([], [], color=ANALYTIC_COLOR, lw=1.5),
               Line2D([], [], color=NEW_COLOR, lw=1.5)]
    labels = [r"$2NU = 0.01$", r"$g_f(a)$", r"$g(a)$"]
    # y below the Permissive/Restrictive background boxes (which reach y = -0.16)
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.26),
               fontsize=FONTSIZE - 2, edgecolor="k", framealpha=1, handlelength=2.0,
               handler_map={tuple: HandlerTuple(ndivide=1, pad=0)})

    cf.make_figure_set_width(fig, out_path, target_width_inches=6.5, dpi=400)
    plt.close(fig)


def main(snakemake):
    sdist = getattr(snakemake.params, "sdist", None)
    analytic = analytic_cdf(getattr(snakemake.input, "analytic", []) or [])
    emp_S = empirical_fixed_S(getattr(snakemake.input, "simulations", []) or [])

    missing = [(p["shift"], p["sigma2"]) for p in PANELS
               if (p["shift"], p["sigma2"]) not in analytic]
    if missing:
        print(f"Figure_S9: no analytic fixed-effect-size CDF for panels {missing} "
              f"(analytic curves built with COMPUTE_FIXED_EFFECT_SIZE_DISTRIBUTION off?); "
              f"the black line is omitted there.", file=sys.stderr)

    out_path = (getattr(snakemake.output, "figure_s9", None)
                if hasattr(snakemake.output, "figure_s9") else None) or snakemake.output[0]
    make_figure_S9(out_path, analytic, emp_S, sdist)


if "snakemake" in globals():
    main(snakemake)  # noqa: F821
