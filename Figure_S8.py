"""
Snakemake script: Figure S8, a two-panel summary of when the quasi-static phase of
adaptation begins, as a function of the mutational input 2NU. Both panels are
styled like Figure 4 panel A (Figure_4.py): simulation means with ~2-SE error bars
over a log 2NU axis and a grey band marking the low-mutational-input regime (2NU<1).

Panel A: the mean *time* the quasi-static phase begins (convergence_time, in
         generations).
Panel B: the mean *distance* to the optimum when the quasi-static phase begins
         (convergence_D_window). This can be negative -- the transient
         overshooting discussed in the supplement -- so the y-axis is not clipped
         at zero and a reference line marks D_qs = 0.

The quasi-static onset is recorded per replicate by simulation_classes.Simulation:
the first generation at which the 10-generation sliding-window means of D, V and u3
satisfy |[D_w - u3_w/(2 V_w)] / D_w| < 0.05, together with D_w there. Replicates that
never reach the criterion store NaN and are dropped from the mean.

There is one point series per (shift, sigma2) combination, discovered from the
supplied pickles and coloured by common_functions.color_by_both.

Input
    simulations : simulation pickles across 2NU for one or more (shift, sigma2)
                  combinations (and a single sdist)
Params
    sdist  (kept for parity with the other figure rules)
Output
    figure_s8 : the two-panel PNG
"""

import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

from common_functions import color_by_both, make_figure_set_width


INTERNAL_FONTSIZE = 10         # legend text
EXTERNAL_FONTSIZE = 11         # titles, axis labels, tick labels

# the two panels: per-replicate pickle key, panel letter, title, y-axis label
METRICS = ("time", "distance")
REP_KEY = {"time": "convergence_time", "distance": "convergence_D_window"}
PANEL_LETTER = {"time": "A", "distance": "B"}
TITLE = {"time": r"$\bf{A.}$ Quasi-static phase: time at onset", "distance": r"$\bf{B.}$ Quasi-static phase: distance at onset"}
YLABEL = {"time": r"Time after shift (generations)",
          "distance": r"Distance from optimum"}


# --------------------------------------------------------------------------- #
# Simulation data: per-(shift, sigma2) quasi-static onset summaries vs 2NU
# --------------------------------------------------------------------------- #
def sim_summary(path):
    """Return one pickle's NaN-aware mean / ~2-SE for both onset metrics.

    Per replicate, 'time' is convergence_time and 'distance' is
    convergence_D_window; replicates that never reached the quasi-static criterion
    store NaN and are excluded from the mean and standard error.
    """
    with open(path, "rb") as f:
        res = pickle.load(f)
    p = res["parameters"]

    means, sems = {}, {}
    for metric in METRICS:
        v = np.array([rep[REP_KEY[metric]] for rep in res["replicates"]], dtype=float)
        v = v[~np.isnan(v)]
        means[metric] = float(v.mean()) if v.size else np.nan
        sems[metric] = float(2.0 * v.std() / np.sqrt(v.size)) if v.size > 1 else 0.0  # ~2 SE
    return {"n2u": p["N2U"], "shift": p["shift"], "sigma2": p["sigma2"],
            "means": means, "sems": sems}


# --------------------------------------------------------------------------- #
# Panels (each one styled like Figure 4 panel A)
# --------------------------------------------------------------------------- #
def make_panel(ax, metric, groups, min_a=None):
    """Draw one S8 panel: one ~2-SE error-bar series per (shift, sigma2) group."""
    for g in groups:
        ax.errorbar(g["n2u"], g["means"][metric], yerr=g["sems"][metric],
                    marker=".", ls="", color=g["color"], zorder=2)  # default marker size, matching Figure_S6

    # grey band marks the low-mutational-input regime (2NU < 1)
    ax.axvspan(1e-3, 1.0, facecolor=[0.9, 0.9, 0.9], edgecolor="none", zorder=0)
    if metric == "distance":
        ax.axhline(0, color="0.3", lw=1.5, zorder=1)   # D_qs = 0 (overshooting below)
        # analytic quasi-static distance D_qs = min(a)/2, labelled above the line
        if min_a is not None:
            y_qs = min_a / 2
            ax.axhline(y_qs, color="0.3", lw=1.5, ls="--", zorder=1)
            ax.text(0.002, y_qs, r"$D_{qs} = \min(a)/2$", color="0.3",
                    ha="left", va="bottom", fontsize=INTERNAL_FONTSIZE)

    ax.set_title(TITLE[metric], fontsize=EXTERNAL_FONTSIZE, loc="center")
    ax.set_xscale("log")
    ax.set_xlim([1e-3, 1e3])
    # a tick on every decade, but a label only every other one; set_xticks alone would
    # install a FixedLocator and drop the unlabelled decades' ticks entirely
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
    ax.set_xticklabels(["", r"$10^{-2}$", "", r"$1$", "", r"$10^{2}$", ""],
                       fontsize=EXTERNAL_FONTSIZE)
    ax.set_xlabel(r"Mutational input ($2NU$)", fontsize=EXTERNAL_FONTSIZE, labelpad=1)
    ax.set_ylabel(YLABEL[metric], fontsize=EXTERNAL_FONTSIZE, labelpad=2)
    ax.tick_params(labelsize=EXTERNAL_FONTSIZE)
    ax.set_box_aspect(1)                       # square panels, like Figure 4A
    if metric == "time":
        ax.set_yscale("log")                             # set scale before limits
        ax.set_ylim(bottom=1, top=20000)                 # onset time, log scale 1..20000
# --------------------------------------------------------------------------- #
# Assemble the figure
# --------------------------------------------------------------------------- #
def make_figure_S8(out_path, groups, min_a=None, figsize=(7, 3.7)):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize, dpi=400)
    make_panel(axes[0], "time", groups)
    make_panel(axes[1], "distance", groups, min_a=min_a)
    
    fig.tight_layout()

    # shared legend below the panels: one entry per (shift, sigma2) group, plus a note
    # that the points are simulation means with ~2-SE error bars
    def errorbar_handle(color, label):
        return axes[0].errorbar([], [], yerr=[], marker=".", ls="", capsize=2,
                                color=color, label=label)
    handles = [errorbar_handle(g["color"],
                               rf"$\Lambda={g['shift']:g}$, $\sigma^2={g['sigma2']:g}$")
               for g in groups]
    # centre the legend on the panels' horizontal extent, not the canvas: the wide
    # left y-label makes the two differ
    fig.canvas.draw()
    _renderer = fig.canvas.get_renderer()
    _content = Bbox.union([ax.get_tightbbox(_renderer) for ax in axes])
    center_x = 0.5 * (_content.x0 + _content.x1) / fig.bbox.width
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4),
               fontsize=INTERNAL_FONTSIZE, bbox_to_anchor=(center_x, -0.12),
               edgecolor="k", framealpha=1)

    make_figure_set_width(fig, out_path, target_width_inches=6.5, dpi=400)
    plt.close(fig)


def main(snakemake):
    paths = snakemake.input.simulations
    sims = [sim_summary(p) for p in paths]

    # group simulations by the (shift, sigma2) combinations actually present
    keys = sorted({(s["shift"], s["sigma2"]) for s in sims})
    groups = []
    for sh, sg in keys:
        rows = sorted((s for s in sims if s["shift"] == sh and s["sigma2"] == sg),
                      key=lambda r: r["n2u"])
        n2u = np.array([r["n2u"] for r in rows])
        means = {m: np.array([r["means"][m] for r in rows]) for m in METRICS}
        sems = {m: np.array([r["sems"][m] for r in rows]) for m in METRICS}
        groups.append({
            "shift": sh, "sigma2": sg, "n2u": n2u, "means": means, "sems": sems,
            "color": color_by_both(sg, sh),   # colour by (sigma2, shift)
        })

    out_path = (getattr(snakemake.output, "figure_s8", None)
                if hasattr(snakemake.output, "figure_s8") else None) or snakemake.output[0]
    min_a = getattr(snakemake.params, "min_a", None)
    make_figure_S8(out_path, groups, min_a=min_a)


if "snakemake" in globals():
    main(snakemake)  # noqa: F821
