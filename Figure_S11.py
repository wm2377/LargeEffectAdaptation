"""
Snakemake script: Figure S11, a six-panel (2 rows x 3 columns) sweep of fixation /
adaptation vs the mutational input, split by the effect size of the fixed allele. It
is the effect-size-window counterpart of Figure S6, with panels styled like Figure 4A.

Rows  -> metric:  top = mean number of fixations; bottom = mean contribution to
         adaptation, as a fraction of the shift size.
Columns -> the squared-effect-size (S = a^2) window the fixed allele falls in:
    left   (large)          : S > 100
    middle ('high' midrange): 30 < S <= 100
    right  ('low' midrange) : 10 < S <= 30

Built for the SDISTS["mixexpunif"] mixture. Each window receives only the fraction `p`
of the total mutational input that the effect-size distribution places in it, so each
panel's x-axis is the window-specific input p*2NU and its grey band marks total
2NU < 1. `p` is read off the supplied sdist, falling back to DEFAULT_WEIGHT.

Points are coloured by the (shift, sigma2) combination; any number of combinations
supplied as pickles are overlaid automatically. Each also gets the matching analytic
curve as a dashed line over the low-mutational-input regime, from the per-window
analytic pickles. Larger-effect alleles fix first and close the phenotypic gap, so the
windows form a precedence chain (see analytic_curve).

Input
    simulations : simulation pickles across 2NU for one or more (shift, sigma2)
                  combinations, run with the mixexpunif effect-size distribution
    analytic    : per-window analytic-curve pickles at the matching shift / sigma2
                  (optional: the figure drops the dashed lines if none are supplied)
Params
    sdist : the effect-size (S) distribution; used to weight each window's x-axis
Output
    figure_s11 : the six-panel PNG
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

# the three effect-size windows, one per column (left -> right)
WINDOWS = ("large", "large_midrange", "small_midrange")

# Fixation precedence: bigger-effect alleles fix first and close the phenotypic gap,
# so each window is drawn conditional on no allele from an earlier one having fixed.
PRECEDES = {w: WINDOWS[:i] for i, w in enumerate(WINDOWS)}

# (lo, hi] bounds on S = a^2 defining each window; hi = inf is the open large window
WINDOW_BOUNDS = {
    "large":          (100.0, np.inf),
    "large_midrange": (30.0, 100.0),
    "small_midrange": (10.0, 30.0),
}

# fall-back window weights for the 50/50 expon+U[10,100] mixture, used only when no
# sdist is supplied
DEFAULT_WEIGHT = {
    "large":          0.5,
    "large_midrange": 0.5 * (100 - 30) / (100 - 10),
    "small_midrange": 0.5 * (30 - 10) / (100 - 10),
}

# Column header text, drawn in the label boxes behind the grid. Plain strings,
# emboldened by fontweight= where they are drawn -- not mathtext, which would render
# the hyphen as a spaced minus sign.
WINDOW_TITLE = {
    "large":          "Large-effect alleles",
    "large_midrange": "'High' midrange\neffect alleles",
    "small_midrange": "'Low' midrange\neffect alleles",
}

# per-column x-axis annotation appended under the bottom-row x label
WINDOW_XLABEL_EXTRA = {
    "large":          r"$a^2 > 100$",
    "large_midrange": r"$30 < a^2 \leq 100$",
    "small_midrange": r"$10 < a^2 \leq 30$",
}

# the two metrics, one per row
METRICS = ("fixations", "contribution")
ROW_OF_METRIC = {"fixations": 0, "contribution": 1}
SCALE_BY_SHIFT = {"fixations": False, "contribution": True}   # contribution as a fraction of Lambda

# panel letters laid out over the (metric, window) grid
PANEL_LETTER = {
    ("fixations", "large"): "A", ("fixations", "large_midrange"): "B", ("fixations", "small_midrange"): "C",
    ("contribution", "large"): "D", ("contribution", "large_midrange"): "E", ("contribution", "small_midrange"): "F",
}

# y-axis extent. The contribution row is always [0, 1] (a fraction of Lambda); the
# fixations row is set per panel from the data by panel_ymax().
CONTRIBUTION_YMAX = 1.0
FIXATION_YMAX_FLOOR = 1.0      # never draw a fixations panel shorter than 0-1

# the analytic approximation is only drawn over the low-mutational-input regime
N2U_GRID = np.logspace(-3, 0, 100)

# reused from Figure_S6, whose 2x3 grid + label-box layout is identical
FIGSIZE = (7.344632768361582, 4.813864456866591)


# --------------------------------------------------------------------------- #
# Window weight p: fraction of the total mutational input that lands in a window
# --------------------------------------------------------------------------- #
def window_weight(sdist, window):
    """Probability mass the effect-size (S) distribution puts in `window`.

    Read off sdist.cdf, falling back to DEFAULT_WEIGHT when no sdist is given.
    """
    if sdist is None:
        return DEFAULT_WEIGHT[window]
    lo, hi = WINDOW_BOUNDS[window]
    hi_cdf = 1.0 if np.isinf(hi) else float(sdist.cdf(hi))
    return hi_cdf - float(sdist.cdf(lo))


# --------------------------------------------------------------------------- #
# Simulation data: per-window fixations / contribution for one 2NU pickle
# --------------------------------------------------------------------------- #
def sim_summary(path):
    """Per-window means/sems of both metrics for one simulation pickle.

    Per replicate the 'fixations' value is the number of fixed alleles whose squared
    effect size falls in a window, and the 'contribution' value is sum_i 2*a_i*(1-x_0_i)
    over those alleles. The mean and a ~2-SE error are taken over replicates.
    """
    with open(path, "rb") as f:
        res = pickle.load(f)
    p = res["parameters"]
    N = p["N"]

    vals = {m: {w: [] for w in WINDOWS} for m in METRICS}
    for rep in res["replicates"]:
        a = np.asarray(rep["fixed_effect_sizes"], dtype=float)   # signed a = +/- sqrt(S)
        if "fixed_initial_frequencies" in rep:
            x0 = np.asarray(rep["fixed_initial_frequencies"])
        else:
            x0 = np.full(a.shape, 1.0 / (2.0 * N))               # older pickles
        S = a ** 2
        contrib = 2.0 * a * (1.0 - x0)
        for w in WINDOWS:
            lo, hi = WINDOW_BOUNDS[w]
            m = (S > lo) & (S <= hi)
            vals["fixations"][w].append(int(np.count_nonzero(m)))
            vals["contribution"][w].append(float(np.sum(contrib[m])))

    means = {m: {} for m in METRICS}
    sems = {m: {} for m in METRICS}
    for m in METRICS:
        for w in WINDOWS:
            v = np.asarray(vals[m][w], dtype=float)
            means[m][w] = v.mean() if v.size else np.nan
            sems[m][w] = (2.0 * v.std() / np.sqrt(v.size)) if v.size > 1 else 0.0  # ~2 SE
    return {"n2u": p["N2U"], "shift": p["shift"], "sigma2": p["sigma2"],
            "means": means, "sems": sems}


# --------------------------------------------------------------------------- #
# Analytic data: per-window curves vs 2NU at a fixed shift
# --------------------------------------------------------------------------- #
def analytic_summary(path):
    """(shift, sigma2, analytic) for one per-window analytic-curve pickle.

    analytic maps metric -> {window: {'segregating', 'new', 'all'}}, each a per-unit-2NU
    expectation restricted to that effect-size window.
    """
    with open(path, "rb") as f:
        res = pickle.load(f)
    p = res["parameters"]
    analytic = {
        "fixations": res.get("expected_fixations"),
        "contribution": res.get("expected_contribution"),
    }
    return p["shift"], p["sigma2"], analytic


def analytic_curve(analytic, metric, window, n2u):
    """Analytic `metric` for one effect-size window vs the TOTAL 2NU.

    Within a window (as in Figure_S6.analytic_curve) the first fixation closes the
    phenotypic gap, so the new-allele term is conditioned on the segregating one not
    having closed it first:

        P_seg = 1 - exp(-aseg)
        P_new = exp(-aseg) * (1 - exp(-mu_new))

    Across windows, a curve is additionally conditioned on no allele from any
    larger-effect window having fixed. The contribution is the contribution per
    fixation times that probability. Returns None if the metric is absent.
    """
    if analytic[metric] is None:
        return None
    n2u = np.asarray(n2u, dtype=float)

    fix_seg = analytic["fixations"][window]["segregating"]
    fix_new = analytic["fixations"][window]["new"]
    p_seg = -np.expm1(-fix_seg * n2u)                            # 1 - exp(-aseg)
    p_new = np.exp(-fix_seg * n2u) * -np.expm1(-fix_new * n2u)   # gap not closed by seg first

    # conditioned on no allele from any larger-effect window having fixed first
    rate_above = sum(analytic["fixations"][w]["all"] for w in PRECEDES[window])
    if rate_above > 0:
        suppression = np.exp(-rate_above * n2u)
        p_seg, p_new = p_seg * suppression, p_new * suppression

    if metric == "fixations":
        return p_seg + p_new
    # the rates are scalars, so guard the division with a conditional rather than
    # np.where, which would evaluate it eagerly and raise on a window with no fixations
    per_seg = analytic["contribution"][window]["segregating"] / fix_seg if fix_seg > 0 else 0.0
    per_new = analytic["contribution"][window]["new"] / fix_new if fix_new > 0 else 0.0
    return per_seg * p_seg + per_new * p_new


# --------------------------------------------------------------------------- #
# Panels (each one styled like Figure 4 panel A)
# --------------------------------------------------------------------------- #
def panel_ymax(metric, window, groups):
    """Upper y limit for one panel.

    Contribution panels are fixed at 1 (a fraction of Lambda). Fixation panels take the
    ceiling of the largest plotted simulation mean, floored at FIXATION_YMAX_FLOOR.
    """
    if SCALE_BY_SHIFT[metric]:
        return CONTRIBUTION_YMAX
    vals = np.concatenate([np.asarray(g["means"][metric][window], dtype=float)
                           for g in groups]) if groups else np.array([])
    vals = vals[np.isfinite(vals)]
    peak = vals.max() if vals.size else 0.0
    return max(FIXATION_YMAX_FLOOR, float(np.ceil(peak)))


def fixation_yticks(ymax):
    """Evenly spaced integer ticks up to `ymax`, at most 5 of them."""
    step = int(np.ceil(ymax / 4.0))
    return list(range(0, int(ymax) + 1, step))


def make_panel(ax, metric, window, groups, weight):
    """Draw one S11 panel: simulation points with ~2-SE error bars + analytic dashed line.

    `groups` is the list of per-(shift, sigma2) bundles built in main(). The x-axis is
    the window-specific mutational input p*2NU, with `p` = `weight`.
    """
    p = weight
    scale = SCALE_BY_SHIFT[metric]
    for g in groups:
        s = g["shift"] if scale else 1.0
        ax.errorbar(p * g["n2u"], g["means"][metric][window] / s,
                    yerr=g["sems"][metric][window] / s,
                    marker=".", ls="", markersize=5, color=g["color"], zorder=2)
        # the analytic expectations are per unit of the total 2NU, so the curve is
        # evaluated on N2U_GRID and plotted against this window's share of it
        curve = g["curves"][metric][window] if g["curves"] is not None else None
        if curve is not None:
            line, = ax.plot(p * N2U_GRID, curve / s, ls="--", color=g["color"], zorder=10)
            line.set_dashes([3, 2])

    # grey band marks the low-mutational-input regime: total 2NU < 1  <=>  p*2NU < p
    ymax = panel_ymax(metric, window, groups)
    ax.fill_between([1e-4, p], [0, 0], [ymax, ymax], color=[0.9, 0.9, 0.9],
                    edgecolor="none", zorder=0)

    ax.set_title(f"{PANEL_LETTER[(metric, window)]}.",
                 size=FONTSIZE, fontweight="bold", loc="left", pad=0)
    for sp in ax.spines.values():
        sp.set_color("k")

    ax.set_xscale("log")
    ax.set_xticks([1e-4, 1e-2, 1, 1e2])
    ax.set_xticklabels([r"$10^{-4}$", r"$10^{-2}$", r"$1$", r"$10^2$"], size=FONTSIZE)
        
    if metric == "contribution":          # bottom row
        ax.set_xlabel(r"Mutational input ($2NU$)" + "\nfor " + WINDOW_XLABEL_EXTRA[window],
                      size=FONTSIZE, labelpad=0)
        ax.set_ylim([0, ymax])
        ax.set_yticks([0, ymax / 2, ymax])
        ax.set_yticklabels(["0", r"$\frac{1}{2}$", "1"], size=FONTSIZE)
    else:                                  # top row
        ax.tick_params(labelbottom=True)
        ax.set_ylim([0, ymax])
        ticks = fixation_yticks(ymax)
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(t) for t in ticks], size=FONTSIZE)

    # window-specific x-limits, applied last: set_xticks would otherwise re-expand the
    # lower bound to the smallest tick
    ax.set_xlim([1e-3 * p, 1e3 * p])


# --------------------------------------------------------------------------- #
# Label boxes drawn behind the grid
# --------------------------------------------------------------------------- #
def _draw_label_boxes(fig):
    """Coloured rounded boxes + bold labels for the metric rows and window columns."""
    box_kw = dict(boxstyle="Round, pad=0,rounding_size=0.025", mutation_aspect=1,
                  clip_on=False, transform=fig.transFigure, figure=fig, zorder=-1)

    # face = pastel fill, edge = darker text/outline
    color1 = np.array([187, 221, 255]) / 256   # large        (blue)
    color2 = np.array([255, 229, 187]) / 256   # large-mid    (tan)
    color3 = np.array([206, 255, 187]) / 256   # small-mid    (green)
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

    # top column-header boxes (full-height, behind the three effect-size windows)
    width = (0.85 - initial_x) / 3
    initial_x = initial_x2
    height0 = height * 2 + initial_y
    # extended above the panels so the two-line window titles fit inside them
    initial_y, height = -0.02, 1.035
    right_box_extra = 0.01      # widens only the rightmost box's right edge
    for i, key in enumerate(("c1", "c2", "c3")):
        w = width + (right_box_extra if key == "c3" else 0.0)
        fig.patches.append(FancyBboxPatch((initial_x + width * i, initial_y), w, height,
                                          facecolor=faces[key], edgecolor=edges[key], **box_kw))
    y_text = (height0 + height + initial_y) / 2.01
    for i, (key, window) in enumerate(zip(("c1", "c2", "c3"), WINDOWS)):
        fig.text(initial_x + width * (i + 0.5), y_text, WINDOW_TITLE[window],
                 fontsize=FONTSIZE, fontweight="bold", color=edges[key],
                 ha="center", va="center")


# --------------------------------------------------------------------------- #
# Assemble the figure
# --------------------------------------------------------------------------- #
def make_figure_S11(out_path, groups, weights, figsize=FIGSIZE):
    fig, axes = plt.subplots(nrows=2, ncols=3, dpi=400, figsize=figsize)

    for metric in METRICS:
        r = ROW_OF_METRIC[metric]
        for col, window in enumerate(WINDOWS):
            make_panel(axes[r, col], metric, window, groups, weights[window])

    # hspace is generous because the top row keeps its own x tick labels
    plt.subplots_adjust(wspace=0.17, hspace=0.45)
    _draw_label_boxes(fig)

    # legend below panel E as a 2 x 3 grid: the (shift, sigma2) combinations on top,
    # the line/marker keys below. matplotlib fills columns top-to-bottom.
    combo_groups = sorted(groups, key=lambda g: (g["sigma2"], -g["shift"]))
    combo_handles = [
        axes[1, 1].errorbar([np.nan], [np.nan], yerr=[np.nan], marker=".", ls="",
                            markersize=5, color=g["color"],
                            label=rf"$\Lambda={g['shift']:g}$, $\sigma^2={g['sigma2']:g}$")
        for g in combo_groups
    ]
    double_recursion = Line2D([], [], ls="--", color="k", label="Double recursion")
    mean_handle = axes[1, 1].errorbar([np.nan], [np.nan], yerr=[np.nan], marker=".",
                                      ls="", markersize=5, color="k",
                                      label=r"Mean $\pm$ 2SEs")
    blank = Line2D([], [], ls="", marker="", label="")

    handles = [combo_handles[0], double_recursion,
               combo_handles[1], blank,
               combo_handles[2], mean_handle]
    axes[1, 1].legend(handles=handles, ncols=3, loc="center", handlelength=1.8,
                      fontsize=FONTSIZE, bbox_to_anchor=(0.5, -0.7),
                      edgecolor="k", framealpha=1)

    make_figure_set_width(fig, out_path, target_width_inches=6.5, dpi=300)
    plt.close(fig)


def main(snakemake):
    sims = [sim_summary(p) for p in snakemake.input.simulations]
    sdist = getattr(snakemake.params, "sdist", None)
    weights = {w: window_weight(sdist, w) for w in WINDOWS}

    # per-window analytic curves, keyed by (shift, sigma2) rounded so the float keys match
    # the simulations. Optional: without them the figure is just the simulation points.
    analytics = [analytic_summary(p) for p in getattr(snakemake.input, "analytic", [])]
    amap = {(round(sh, 6), round(sg, 6)): an for sh, sg, an in analytics}

    # group simulations by (shift, sigma2) so each combination is one coloured series
    keys = sorted({(s["shift"], s["sigma2"]) for s in sims})
    groups = []
    for sh, sg in keys:
        rows = sorted((s for s in sims if s["shift"] == sh and s["sigma2"] == sg),
                      key=lambda r: r["n2u"])
        n2u = np.array([r["n2u"] for r in rows])
        means = {m: {w: np.array([r["means"][m][w] for r in rows]) for w in WINDOWS} for m in METRICS}
        sems = {m: {w: np.array([r["sems"][m][w] for r in rows]) for w in WINDOWS} for m in METRICS}
        an = amap.get((round(sh, 6), round(sg, 6)))
        curves = (None if an is None else
                  {m: {w: analytic_curve(an, m, w, N2U_GRID) for w in WINDOWS} for m in METRICS})
        groups.append({
            "shift": sh, "sigma2": sg, "n2u": n2u, "means": means, "sems": sems,
            "curves": curves,
            "color": color_by_both(sg, sh),   # colour by (sigma2, shift)
        })

    out_path = (getattr(snakemake.output, "figure_s11", None)
                if hasattr(snakemake.output, "figure_s11") else None) or snakemake.output[0]
    make_figure_S11(out_path, groups, weights)


if "snakemake" in globals():
    main(snakemake)  # noqa: F821
