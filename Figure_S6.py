"""
Snakemake script: Figure S6, a six-panel (2 rows x 3 columns) sweep of adaptation
from large-effect alleles vs the mutational input 2NU. Every panel is styled like
Figure 4 panel A (Figure_4.py): simulation means with ~2-SE error bars over a log
2NU axis, a grey band marking the low-mutational-input regime (2NU < 1), and the
analytic approximation drawn as a dashed line over that same 2NU <= 1 range.

Rows  -> metric:  top = mean number of fixations; bottom = mean contribution to
         adaptation, as a fraction of the shift size.
Columns -> source of the fixed allele:
    left   (all)         : every fixed large-effect allele
    middle (segregating) : standing variation present at the shift (arrival t == 0)
    right  (new)         : de novo mutations that arose after the shift (t > 0)

Panel D reproduces Figure 4A. The metric (row) and source (column) are carried by
the coloured rounded label boxes drawn behind the grid rather than by panel titles.

Per-source simulation summaries follow Figure_S7.summarize and the analytic curves
follow Figure_S7._analytic_curves, reformulated to sweep N2U at a fixed shift.
Points and their dashed analytic lines are coloured by (shift, sigma2); any number
of combinations supplied as pickles are overlaid automatically.

Input
    simulations_A : simulation pickles across 2NU (fixed shift / sigma2 / sdist)
    analytic      : analytic-curve pickle(s) at the matching shift / sigma2
Params
    shift, sigma2, sdist, N2U  (kept for parity with Figure_4 / the Snakefile rule)
Output
    figure_s6 : the six-panel PNG
"""

import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from common_functions import color_by_both, make_figure_set_width


FONTSIZE = 11                  # titles, axis/tick labels, label boxes, legend

SOURCES = ("all", "segregating", "new")

# the two metrics, one per row
METRICS = ("fixations", "contribution")
ROW_OF_METRIC = {"fixations": 0, "contribution": 1}
SCALE_BY_SHIFT = {"fixations": False, "contribution": True}   # contribution as a fraction of Lambda

# panel letters laid out over the (metric, source) grid
PANEL_LETTER = {
    ("fixations", "all"): "A", ("fixations", "segregating"): "B", ("fixations", "new"): "C",
    ("contribution", "all"): "D", ("contribution", "segregating"): "E", ("contribution", "new"): "F",
}

# classify each fixed allele from its arrival generation (cf. Figure_S7)
SOURCE_MASK = {
    "all":         lambda t: np.ones(t.shape, dtype=bool),
    "segregating": lambda t: t == 0,     # standing variation at the shift
    "new":         lambda t: t > 0,      # de novo after the shift
}

# analytic approximation is only valid / drawn over the low-mutational-input regime
N2U_GRID = np.logspace(-3, 0, 100)

# figure size converged by the notebook's square-panel / target-width loops; using
# it directly keeps panel A square and reproduces the hand-tuned label-box layout.
FIGSIZE = (7.344632768361582, 4.813864456866591)


# --------------------------------------------------------------------------- #
# Simulation data: per-source fixations / contribution for one 2NU pickle
# --------------------------------------------------------------------------- #
def sim_summary(path):
    """Per-source means/sems of both metrics for one simulation pickle.

    Mirrors Figure_S7.summarize but keyed for the 2NU sweep. Per replicate the
    'fixations' value is the number of fixed alleles of a source and the
    'contribution' value is sum_i 2*a_i*(1-x_0_i) over them, using each allele's
    recorded initial frequency x_0 (fixed_initial_frequencies; a = signed
    fixed_effect_sizes). Older pickles fall back to x_0 = 1/(2N). The mean and a
    ~2-SE error are taken over replicates.
    """
    with open(path, "rb") as f:
        res = pickle.load(f)
    p = res["parameters"]
    N = p["N"]

    vals = {m: {s: [] for s in SOURCES} for m in METRICS}
    for rep in res["replicates"]:
        t = np.asarray(rep["fixed_arrival_times"])
        a = np.asarray(rep["fixed_effect_sizes"])     # signed a = +/- sqrt(S)
        if "fixed_initial_frequencies" in rep:
            x0 = np.asarray(rep["fixed_initial_frequencies"])
        else:
            x0 = np.full(a.shape, 1.0 / (2.0 * N))     # older pickles
        contrib = 2.0 * a * (1.0 - x0)
        for s in SOURCES:
            m = SOURCE_MASK[s](t)
            vals["fixations"][s].append(int(np.count_nonzero(m)))
            vals["contribution"][s].append(float(np.sum(contrib[m])))

    means = {m: {} for m in METRICS}
    sems = {m: {} for m in METRICS}
    for m in METRICS:
        for s in SOURCES:
            v = np.asarray(vals[m][s], dtype=float)
            means[m][s] = v.mean() if v.size else np.nan
            sems[m][s] = (2.0 * v.std() / np.sqrt(v.size)) if v.size > 1 else 0.0  # ~2 SE
    return {"n2u": p["N2U"], "shift": p["shift"], "sigma2": p["sigma2"],
            "means": means, "sems": sems}


# --------------------------------------------------------------------------- #
# Analytic data: per-source curves vs 2NU at a fixed shift
# --------------------------------------------------------------------------- #
def analytic_summary(path):
    """(shift, sigma2, analytic) for one analytic-curve pickle.

    analytic maps metric -> {source: per-allele expectation}; 'contribution' is
    None for older pickles that predate expected_contribution.
    """
    with open(path, "rb") as f:
        res = pickle.load(f)
    p = res["parameters"]
    analytic = {
        "fixations": res.get("expected_fixations"),
        "contribution": res.get("expected_contribution"),
    }
    return p["shift"], p["sigma2"], analytic


def analytic_curve(analytic, metric, n2u):
    """Per-source analytic `metric` vs 2NU at fixed shift, bounded by gap-closing.

    The first large-effect substitution to fix closes the phenotypic gap, so at most
    one large-effect allele fixes. The expected number of fixations from a source is
    therefore the *probability* that at least one such allele fixes -- which saturates
    at 1 rather than growing linearly in 2NU (the linear form is only the small-2NU
    limit used vs shift in Figure_S7):

        P_seg = 1 - exp(-aseg)                       (segregating)
        P_new = exp(-aseg) * (1 - exp(-mu_new))      (new, only if seg did not fix first)
        all   = P_seg + P_new = 1 - exp(-(aseg + mu_new))   (<= 1)

    where aseg = (per-2NU segregating fixation rate) * 2NU and mu_new likewise for new.
    The contribution to adaptation is the per-fixation contribution times that fixation
    probability. Returns {source: array over n2u}, or None if the metric is absent.
    """
    if analytic[metric] is None:
        return None
    n2u = np.asarray(n2u, dtype=float)

    fix_seg = analytic["fixations"]["segregating"]
    fix_new = analytic["fixations"]["new"]
    p_seg = -np.expm1(-fix_seg * n2u)                         # 1 - exp(-aseg)
    p_new = np.exp(-fix_seg * n2u) * -np.expm1(-fix_new * n2u)  # gap not closed by seg first

    if metric == "fixations":
        seg, new = p_seg, p_new
    else:  # contribution = (contribution per fixation) * (probability of fixation)
        # fix_seg / fix_new are scalars, so guard with a plain conditional. np.where would
        # evaluate the division eagerly and raise ZeroDivisionError when a rate is 0 (e.g.
        # no new-allele fixations expected, as at shift=50/sigma2=80), and np.errstate does
        # not suppress Python scalar division-by-zero.
        per_seg = analytic["contribution"]["segregating"] / fix_seg if fix_seg > 0 else 0.0
        per_new = analytic["contribution"]["new"] / fix_new if fix_new > 0 else 0.0
        seg, new = per_seg * p_seg, per_new * p_new
    return {"segregating": seg, "new": new, "all": seg + new}


# --------------------------------------------------------------------------- #
# Panels (each one styled like Figure 4 panel A)
# --------------------------------------------------------------------------- #
def make_panel(ax, metric, source, groups):
    """Draw one S6 panel: simulation points (~2-SE error bars) + analytic dashed line.

    `groups` is the list of per-(shift, sigma2) bundles built in main(); contribution
    panels divide by the shift so the y-axis is a fraction of Lambda, matching Figure 4A.
    """
    scale = SCALE_BY_SHIFT[metric]
    for g in groups:
        s = g["shift"] if scale else 1.0
        ax.errorbar(g["n2u"], g["means"][metric][source] / s,
                    yerr=g["sems"][metric][source] / s,
                    marker=".", ls="", color=g["color"], markersize=5,zorder=2)
        curve = g["curves"][metric]
        if curve is not None:
            line, = ax.plot(N2U_GRID, curve[source] / s, ls="--", color=g["color"], zorder=10)
            line.set_dashes([3, 2])

    # grey band marks the low-mutational-input regime (2NU < 1)
    ax.axvspan(1e-4, 1.0, facecolor="gray", edgecolor="none", alpha=0.15, zorder=1)

    ax.set_title(f"{PANEL_LETTER[(metric, source)]}.",
                 size=FONTSIZE, fontweight="bold", loc="left", pad=0)
    for sp in ax.spines.values():
        sp.set_color("k")

    ax.set_xscale("log")
    ax.set_xlim([1e-3 / 1.2, 500 * 1.2])
    ax.set_xticks([1e-2, 1, 1e2])

    if metric == "contribution":          # bottom row: x labels + Lambda-fraction y axis
        ax.set_xticklabels([r"$10^{-2}$", r"$1$", r"$10^2$"], size=FONTSIZE)
        ax.set_xlabel(r"Mutational input ($2NU$)", size=FONTSIZE, labelpad=0)
        ax.set_ylim([0, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(["0", r"$\frac{1}{2}$", "1"], size=FONTSIZE)
    else:                                  # top row: hide x labels, fixation-count y axis
        ax.tick_params(labelbottom=False)
        ax.set_ylim([0, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["0", "1", "2"], size=FONTSIZE)


# --------------------------------------------------------------------------- #
# Label boxes drawn behind the grid (ported verbatim from Figure_S6.ipynb)
# --------------------------------------------------------------------------- #
def _draw_label_boxes(fig):
    """Coloured rounded boxes + bold labels for the metric rows and source columns."""
    box_kw = dict(boxstyle="Round, pad=0,rounding_size=0.025", mutation_aspect=1,
                  clip_on=False, transform=fig.transFigure, figure=fig, zorder=-1)

    # local colour scheme (face = pastel fill, edge = darker text/outline)
    color1 = np.array([187, 221, 255]) / 256   # All        (blue)
    color2 = np.array([255, 229, 187]) / 256   # Segregating (tan)
    color3 = np.array([206, 255, 187]) / 256   # New        (green)
    colorA = np.array([0.8, 0.8, 0.8])         # contribution row (lighter grey)
    colorB = np.array([0.6, 0.6, 0.6])         # fixations row    (darker grey)
    faces = {k: c / 2 + 1 / 2 for k, c in
             dict(c1=color1, c2=color2, c3=color3, cA=colorA, cB=colorB).items()}
    edges = {k: c / 2 for k, c in
             dict(c1=color1, c2=color2, c3=color3, cA=colorA, cB=colorB).items()}

    # left-hand row-label boxes (bottom = contribution, top = fixations)
    initial_x, initial_x2, width, initial_y = 0.03, 0.092, 0.1, 0.07
    height = (0.93 - initial_y) / 2
    fig.patches.append(FancyBboxPatch((initial_x, initial_y), width, height,
                                      facecolor=faces["cA"], edgecolor=edges["cA"], **box_kw))
    fig.patches.append(FancyBboxPatch((initial_x, initial_y + height), width, height,
                                      facecolor=faces["cB"], edgecolor=edges["cB"], **box_kw))
    fig.text((initial_x + initial_x2) / 2, initial_y + height / 2,
             r"$\bf{Adaptive\ contribution}$" + "\n" + "relative to shift",
             fontsize=FONTSIZE, color=edges["cA"], ha="center", va="center", rotation=90)
    fig.text((initial_x + initial_x2) / 2, initial_y + height * 3 / 2,
             "Fixations", fontweight="bold",
             fontsize=FONTSIZE, color=edges["cB"], ha="center", va="center", rotation=90)

    # top column-header boxes (full-height, behind All / Segregating / New)
    width = (0.85 - initial_x) / 3
    initial_x = initial_x2
    height0 = height * 2 + initial_y
    initial_y, height = 0, 0.98
    for i, key in enumerate(("c1", "c2", "c3")):
        fig.patches.append(FancyBboxPatch((initial_x + width * i, initial_y), width, height,
                                          facecolor=faces[key], edgecolor=edges[key], **box_kw))
    y_text = (height0 + height + initial_y) / 2
    fig.text((initial_x * 2 + width) / 2, y_text, "All", fontweight="bold",
             fontsize=FONTSIZE, color=edges["c1"], ha="center", va="center")
    fig.text((initial_x * 2 + width * 3) / 2, y_text, "Segregating", fontweight="bold",
             fontsize=FONTSIZE, color=edges["c2"], ha="center", va="center")
    fig.text((initial_x * 2 + width * 5) / 2, y_text, "New", fontweight="bold",
             fontsize=FONTSIZE, color=edges["c3"], ha="center", va="center")


# --------------------------------------------------------------------------- #
# Assemble the figure
# --------------------------------------------------------------------------- #
def make_figure_S6(out_path, groups, figsize=FIGSIZE):
    fig, axes = plt.subplots(nrows=2, ncols=3, dpi=400, figsize=figsize)

    for metric in METRICS:
        r = ROW_OF_METRIC[metric]
        for col, source in enumerate(SOURCES):
            make_panel(axes[r, col], metric, source, groups)

    plt.subplots_adjust(wspace=0.17, hspace=0.175)
    _draw_label_boxes(fig)

    # combined legend below the bottom-middle panel (E): one entry per (shift, sigma2)
    # group plus the simulation/analytic style key, matching the notebook placement.
    # simulation entries are drawn as error bars (marker + vertical up/down bar with caps,
    # no connecting line) to match how the means are plotted; the analytic entry is a line.
    def errorbar_handle(color, label):
        return axes[1, 1].errorbar([], [], yerr=[], marker=".", ls="", markersize=5,
                                   capsize=2, color=color, label=label)
    group_handles = [errorbar_handle(g["color"],
                                     rf"$\Lambda={g['shift']:g}$, $\sigma^2={g['sigma2']:g}$")
                     for g in groups]
    desc_handles = [
        errorbar_handle("gray", r"Mean $\pm 2$ SEs"),
        Line2D([], [], ls="--", color="gray", label="Double recursion"),
    ]
    # matplotlib fills legends column-major, so use one column per group and interleave
    # (group, description) within each column: this keeps every parameter combination on
    # the first row and the description entries on the second. Pad with a blank handle so
    # the (group, description) pairs line up.
    ncols = len(group_handles)
    blank = Line2D([], [], ls="", marker="", label="")
    desc_padded = desc_handles + [blank] * (ncols - len(desc_handles))
    handles = [h for pair in zip(group_handles, desc_padded) for h in pair]
    axes[1, 1].legend(handles=handles, ncols=ncols, loc="center", handlelength=1,
                      fontsize=FONTSIZE, bbox_to_anchor=(0.5, -0.5),
                      edgecolor="k", framealpha=1)

    make_figure_set_width(fig, out_path, target_width_inches=6.5, dpi=300)
    plt.close(fig)


def main(snakemake):
    sims = [sim_summary(p) for p in snakemake.input.simulations_A]
    analytics = [analytic_summary(p) for p in snakemake.input.analytic]
    # (shift, sigma2) -> analytic dict, rounded so float keys match the simulations
    amap = {(round(sh, 6), round(sg, 6)): an for sh, sg, an in analytics}

    # group simulations by (shift, sigma2) so each combination is one coloured series
    keys = sorted({(s["shift"], s["sigma2"]) for s in sims})
    groups = []
    for sh, sg in keys:
        rows = sorted((s for s in sims if s["shift"] == sh and s["sigma2"] == sg),
                      key=lambda r: r["n2u"])
        n2u = np.array([r["n2u"] for r in rows])
        means = {m: {s: np.array([r["means"][m][s] for r in rows]) for s in SOURCES} for m in METRICS}
        sems = {m: {s: np.array([r["sems"][m][s] for r in rows]) for s in SOURCES} for m in METRICS}

        an = amap.get((round(sh, 6), round(sg, 6)))
        curves = {m: (analytic_curve(an, m, N2U_GRID) if an is not None else None) for m in METRICS}

        groups.append({
            "shift": sh, "sigma2": sg, "n2u": n2u, "means": means, "sems": sems,
            "curves": curves,
            "color": color_by_both(sg, sh),   # colour by (sigma2, shift)
        })

    out_path = (getattr(snakemake.output, "figure_s6", None)
                if hasattr(snakemake.output, "figure_s6") else None) or snakemake.output[0]
    make_figure_S6(out_path, groups)


if "snakemake" in globals():
    main(snakemake)  # noqa: F821
