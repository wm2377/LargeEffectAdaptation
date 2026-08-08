"""
Snakemake script: Figure S10, three panels of boxplots showing the distribution of
squared effect sizes (S = a^2) of *fixed* large-effect alleles as a function of the
mutational input 2NU. Each panel is one (shift, sigma2) combination:

    A : Lambda = 80, sigma2 = 40      (permissive)
    B : Lambda = 80, sigma2 = 80      (permissive)
    C : Lambda = 50, sigma2 = 80      (restrictive)

For each panel and each 2NU value, all fixed alleles are pooled across replicates and
their squared effect sizes (fixed_effect_sizes**2) are summarized as one boxplot
(box = IQR, whiskers = 2.5/97.5 percentiles, black line = median, white line = mean).
Boxes are coloured by common_functions.color_by_both(sigma2, shift), consistent with
the other figures.

A grey box at the far left of each panel marks the analytic no-interference DFE of
fixed alleles, g_f(a) = g(a) X(a) / X, read from the matching analytic pickle's
'fixed_effect_size_distribution'. When that field is absent the panel falls back to a
placeholder box drawn from the prior sdist (dummy_dfe_box).

Input
    simulations : simulation pickles across 2NU for the three (shift, sigma2)
                  combinations (and a single sdist)
    analytic    : one analytic_curves pickle per (shift, sigma2) panel, supplying the
                  analytic fixed-effect-size distribution g_f(a)
Params
    sdist : the effect-size (S) distribution, the fallback g_f(a) box
Output
    figure_s10 : the three-panel PNG
"""

import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import common_functions as cf
from common_functions import color_by_both


FONTSIZE = 12
MIN_FIXED = 5                  # need more than this many pooled fixed alleles to draw a box

# the two large-shift (permissive) panels first, then the small-shift (restrictive)
PANELS = [
    {"shift": 80, "sigma2": 40, "letter": "A"},
    {"shift": 80, "sigma2": 80, "letter": "B"},
    {"shift": 50, "sigma2": 80, "letter": "C"},
]

GF_BOX_X = 10 ** (-3.2)     # x position of the grey analytic g_f(a) box


# --------------------------------------------------------------------------- #
# Data: squared effect sizes of fixed alleles, pooled across replicates, by 2NU
# --------------------------------------------------------------------------- #
def fixed_S_by_n2u(paths):
    """{(shift, sigma2): {2NU: array of squared effect sizes of fixed alleles}}.

    For each pickle, every fixed allele's signed effect size a (fixed_effect_sizes)
    is pooled across replicates and squared to S = a^2. A 2NU entry is kept only if
    more than MIN_FIXED fixed alleles are available there.
    """
    out = {}
    for p in paths:
        with open(p, "rb") as f:
            res = pickle.load(f)
        prm = res["parameters"]
        arrs = [np.asarray(rep["fixed_effect_sizes"], dtype=float) for rep in res["replicates"]]
        a = np.concatenate(arrs) if arrs else np.array([])
        S = a ** 2
        S = S[S > 0]
        if S.size > MIN_FIXED:
            out.setdefault((prm["shift"], prm["sigma2"]), {})[prm["N2U"]] = S
    return out


def dummy_dfe_box(sdist):
    """Fallback stats for the grey g_f(a) box (a matplotlib bxp item).

    Quantiles of the prior S distribution, so the box still renders sensibly when an
    analytic pickle has no fixed_effect_size_distribution.
    """
    if sdist is not None:
        item = {k: float(sdist.ppf(q)) for k, q in
                (("whislo", 0.025), ("q1", 0.25), ("med", 0.5), ("q3", 0.75), ("whishi", 0.975))}
        item["mean"] = float(sdist.mean())
    else:
        item = {"whislo": 150, "q1": 300, "med": 500, "q3": 800, "whishi": 1500, "mean": 600}
    return item


def analytic_boxes(paths):
    """{(shift, sigma2): bxp item} of the analytic g_f(a) box from analytic pickles.

    Keyed by each pickle's (shift, sigma2). Pickles with no distribution are skipped,
    so the panel falls back to dummy_dfe_box.
    """
    out = {}
    for p in paths:
        with open(p, "rb") as f:
            res = pickle.load(f)
        prm = res["parameters"]
        dist = res.get("fixed_effect_size_distribution")
        if dist is not None:
            out[(prm["shift"], prm["sigma2"])] = dist["boxplot"]
    return out


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #
def _style_boxplot(bplot, facecolor, alpha):
    """Common box/median/mean styling (translucent fill, black median, white mean)."""
    rgba = np.array(mpl.colors.to_rgba(facecolor))
    rgba[-1] = alpha
    for patch in bplot["boxes"]:
        patch.set_facecolor(rgba)
        patch.set_edgecolor([0.7] * 3)
    for mline in bplot["medians"]:
        mline.set_color("k")
        mline.set_linewidth(2)
    for mline in bplot["means"]:
        mline.set_color("w")
        mline.set_linewidth(2)


def _draw_panel(ax, panel, panel_data, box, is_first):
    color = color_by_both(panel["sigma2"], panel["shift"])

    # simulation boxplots, one per 2NU
    if panel_data:
        n2us = sorted(panel_data)
        bplot = ax.boxplot([panel_data[n] for n in n2us], sym="", whis=(2.5, 97.5),
                           patch_artist=True, showmeans=True, meanline=True,
                           positions=n2us, widths=[n / 4 for n in n2us])
        _style_boxplot(bplot, color, alpha=0.3)

    # grey analytic g_f(a) box (the fixed-effect-size distribution for this panel)
    bplot = ax.bxp([box], positions=[GF_BOX_X], showfliers=False,
                   patch_artist=True, showmeans=True, meanline=True, widths=GF_BOX_X / 4)
    _style_boxplot(bplot, "grey", alpha=0.5)
    for patch in bplot["boxes"]:
        patch.set_edgecolor([0.7] * 3)

    # applied after the boxplots, which reset ticks/limits
    ax.set_yscale("log")
    ax.set_ylim([100, 2500])
    ax.set_xscale("log")
    ax.set_xlim([GF_BOX_X / 1.3, 200 * 1.2])
    ax.set_xlabel(r"Mutational input ($2NU$)", fontsize=FONTSIZE, labelpad=0)
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1, 1e1, 1e2])
    ax.set_xticklabels(["", r"$10^{-2}$", "", r"$1$", "", r"$10^2$"], fontsize=FONTSIZE)
    ax.tick_params(bottom=False, which="minor")
    ax.set_title(rf"$\bf{{{panel['letter']}.}}$ "
                 rf"$\Lambda = {panel['shift']:g},\ \sigma^2 = {panel['sigma2']:g}$",
                 fontsize=FONTSIZE, loc="left")

    if is_first:
        ax.set_ylabel(r"Effect size squared ($S=a^2$)", fontsize=FONTSIZE, labelpad=0)
        ax.tick_params(labelsize=FONTSIZE)
    else:
        ax.tick_params(labelleft=False)


def _draw_background(fig, axes):
    """Permissive/Restrictive background boxes, labels, and the g_f(a) arrow."""
    # permissive box (panels A, B) and restrictive box (panel C), in figure coords
    first_xy, first_wh = (-0.075, -0.18), (0.748, 1.36)
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

    # curved arrow + label pointing at the grey g_f(a) box (panel A only)
    arrow = FancyArrowPatch((10 ** (-2.2), 2100), (GF_BOX_X, 1600),
                            connectionstyle="arc3,rad=.5", color="k",
                            arrowstyle="Simple, tail_width=0.5, head_width=5, head_length=8")
    axes[0].add_patch(arrow)
    axes[0].text(10 ** (-2.2), 2000, r"$g_f(a)$", fontsize=FONTSIZE, color="k")


# --------------------------------------------------------------------------- #
# Assemble the figure
# --------------------------------------------------------------------------- #
def make_figure_S10(out_path, data, boxes, sdist, figsize=(10, 3.5)):
    fig = plt.figure(figsize=figsize, dpi=400)

    # three equal panels; the B|C gap is widened to separate the permissive and
    # restrictive groups
    spacing = 0.02
    extra = 0.02                       # additional space between panels B and C
    width = (1 - spacing * 3 - extra) / 3
    axes = [fig.add_axes([spacing + i * (width + spacing) + (extra if i >= 2 else 0),
                          0.05, width, 0.9]) for i in range(3)]

    for i, (ax, panel) in enumerate(zip(axes, PANELS)):
        box = boxes.get((panel["shift"], panel["sigma2"])) or dummy_dfe_box(sdist)
        _draw_panel(ax, panel, data.get((panel["shift"], panel["sigma2"]), {}), box, is_first=(i == 0))

    _draw_background(fig, axes)

    cf.make_figure_set_width(fig, out_path, target_width_inches=6.5, dpi=400)
    plt.close(fig)


def main(snakemake):
    data = fixed_S_by_n2u(snakemake.input.simulations)
    boxes = analytic_boxes(getattr(snakemake.input, "analytic", []) or [])
    sdist = getattr(snakemake.params, "sdist", None)

    missing = [(p["shift"], p["sigma2"]) for p in PANELS if (p["shift"], p["sigma2"]) not in boxes]
    if missing:
        print(f"Figure_S10: no analytic g_f(a) box for panels {missing} "
              f"(analytic curves built with COMPUTE_FIXED_EFFECT_SIZE_DISTRIBUTION off?); "
              f"using the placeholder prior-sdist box there.", file=sys.stderr)

    out_path = (getattr(snakemake.output, "figure_s10", None)
                if hasattr(snakemake.output, "figure_s10") else None) or snakemake.output[0]
    make_figure_S10(out_path, data, boxes, sdist)


if "snakemake" in globals():
    main(snakemake)  # noqa: F821
