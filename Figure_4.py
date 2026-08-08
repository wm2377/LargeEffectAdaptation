"""
Snakemake script: Figure 4, a three-panel summary of adaptation from large-effect
alleles.

Panel A: simulated mean contribution to adaptation relative to the shift size, vs
         the mutational input 2NU.
Panel B: analytic expected number of fixations vs shift, by source.
Panel C: analytic expected number that establish vs shift, by source.

Input
    simulations_A : simulation pickles across N2U (fixed shift / sdist / sigma2)
    analytic      : analytic-curve pickles across shift (fixed sdist / sigma2)
Params
    N2U   : fixed 2NU used to scale the Panel B/C analytic curves
    shift : fixed shift Panel A's contribution is divided by
    sigma2, sdist
Output
    figure_4 : the three-panel PNG
"""

import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from common_functions import CurvedText, make_figure_set_width


INTERNAL_FONTSIZE = 8.5        # in-figure labels (CurvedText)
EXTERNAL_FONTSIZE = 10         # titles, axis labels, tick labels
MAX_Y = 0.35                   # band height and Panel C y-limit
PANEL_B_YMAX = 0.22
# keeps the B/C frame the same size on the page whatever PANEL_B_YMAX is
FRAME_SCALE = PANEL_B_YMAX / 0.4
NEW_COLOR = "red"
SEG_COLOR = "sienna"
TOTAL_COLOR = "k"
FIXED_BOUNDARY = 35            # shift above which fixation dominates
BAND_NO_EST = [0.8, 0.8, 1.0]
BAND_EST = [0.9, 0.8, 0.9]
BAND_FIX = [1.0, 0.8, 0.8]

SOURCES = ("all", "segregating", "new")


# --------------------------------------------------------------------------- #
# Panel A data: simulated contribution to adaptation vs 2NU
# --------------------------------------------------------------------------- #
def sim_contribution(path):
    """Return (N2U, shift, mean, sem) of sum_i 2*a_i*(1-x_0_i) over the fixed alleles."""
    with open(path, "rb") as f:
        res = pickle.load(f)
    N = res["parameters"]["N"]
    vals = []
    for rep in res["replicates"]:
        a = np.asarray(rep["fixed_effect_sizes"])
        if "fixed_initial_frequencies" in rep:
            x0 = np.asarray(rep["fixed_initial_frequencies"])
        else:
            x0 = np.full(a.shape, 1.0 / (2.0 * N))     # older pickles
        vals.append(float(np.sum(2.0 * a * (1.0 - x0))))
    vals = np.asarray(vals, dtype=float)
    mean = float(vals.mean()) if vals.size else np.nan
    sem = float(2.0 * vals.std() / np.sqrt(vals.size)) if vals.size > 1 else 0.0   # ~2 SE
    return res["parameters"]["N2U"], res["parameters"]["shift"], mean, sem


# --------------------------------------------------------------------------- #
# Panels B / C data: analytic curves vs shift
# --------------------------------------------------------------------------- #
def analytic_rows(paths):
    """Sorted list of (shift, results_dict) for the analytic-curve pickles."""
    rows = []
    for p in paths:
        with open(p, "rb") as f:
            res = pickle.load(f)
        rows.append((res["parameters"]["shift"], res))
    return sorted(rows, key=lambda r: r[0])


def _per_source(arows, key):
    """N2U-unscaled per-source arrays {source: array over shift} for results[key]."""
    return {s: np.array([r[1][key][s] for r in arows]) for s in SOURCES}


def fixation_curves(arows, N2U):
    """
    Assumes only one large-effect allele can fix per replicate
    and that the fixation of a segregating allele prevents the fixation of a new allele.
    """
    raw = _per_source(arows, "expected_fixations")
    mu_new = raw["new"] * N2U
    aseg = raw["segregating"] * N2U
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(mu_new > 1e-12, -np.expm1(-mu_new) / mu_new, 1.0)
    new = mu_new * frac * np.exp(-aseg)
    seg = raw["segregating"] * N2U
    return {"segregating": seg, "new": new, "all": seg + new}


def establishment_curves(arows, N2U):
    """Raw N2U-scaled expected establishments per source."""
    raw = _per_source(arows, "expected_establishments")
    return {s: raw[s] * N2U for s in SOURCES}


def establishment_boundary(ashifts, est, threshold=1e-3):
    """Smallest shift at which the total (seg+new) establishment exceeds threshold."""
    total = est["segregating"] + est["new"]
    over = np.where(total > threshold)[0]
    return float(ashifts[over[0]]) if over.size else 0.0


# --------------------------------------------------------------------------- #
# Background regime bands shared by Panels B and C
# --------------------------------------------------------------------------- #
def _draw_boundary_bands(ax, est_boundary):
    # drawn to MAX_Y; each panel's ylim clips them
    ax.fill_between([0, est_boundary], [0, 0], [MAX_Y, MAX_Y], color=BAND_NO_EST, edgecolor="None")
    ax.fill_between([est_boundary, FIXED_BOUNDARY], [0, 0], [MAX_Y, MAX_Y], color=BAND_EST, edgecolor="None")
    ax.fill_between([FIXED_BOUNDARY, 100], [0, 0], [MAX_Y, MAX_Y], color=BAND_FIX, edgecolor="None")


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def make_figure4A(ax, n2u, mean, sem, shift, example_inputs=(0.01, 10)):
    for x, m, e in zip(n2u, mean, sem):
        ax.errorbar(x, m / shift, yerr=e / shift, marker=".", color="k", markersize=3)

    for xin in example_inputs:
        ax.plot([xin, xin], [0, 1], color="green", ls="--", alpha=0.5)

    ax.set_title(r"$\bf{A.}$ Adaptive contribution", fontsize=EXTERNAL_FONTSIZE, loc="left")
    ax.set_xscale("log")
    ax.set_xlim([1e-3, 1e3])
    ax.set_xticks([1e-2, 1e0, 1e2])
    ax.set_xticklabels([r"$10^{-2}$", r"$1$", r"$10^{2}$"], fontsize=EXTERNAL_FONTSIZE)
    ax.set_xlabel(r"Mutational input ($2NU$)", fontsize=EXTERNAL_FONTSIZE, labelpad=1)
    ax.set_ylim([0, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", r"$\frac{\Lambda}{2}$", r"$\Lambda$"], fontsize=EXTERNAL_FONTSIZE)
    ax.set_ylabel(r"Adaptive contribution", fontsize=EXTERNAL_FONTSIZE, labelpad=0)
    ax.fill_between([1e-3, 1], [0, 0], [1, 1], color=[0.9, 0.9, 0.9], edgecolor="None", zorder=0)


def make_figure4B(ax, ashifts, fix, est_boundary):
    _draw_boundary_bands(ax, est_boundary)

    seg, new, total = fix["segregating"], fix["new"], fix["all"]
    ax.plot(ashifts, seg, color=SEG_COLOR)
    ax.plot(ashifts, new, color=NEW_COLOR)
    ax.plot(ashifts, total, color=TOTAL_COLOR)

    # offsets are data-space, so they scale by FRAME_SCALE
    seg_m = ashifts > 51
    CurvedText(x=ashifts[seg_m]+1, y=seg[seg_m] + 0.005 * FRAME_SCALE, text="Segregating",
               va="bottom", color=SEG_COLOR, alpha=1, axes=ax, fontsize=INTERNAL_FONTSIZE)
    new_m = ashifts > 70
    CurvedText(x=ashifts[new_m], y=new[new_m] - 0.01 * FRAME_SCALE, text="New",
               va="top", color=NEW_COLOR, alpha=1, axes=ax, fontsize=INTERNAL_FONTSIZE)
    tot_m = ashifts > 65
    CurvedText(x=ashifts[tot_m], y=total[tot_m] + 0.01 * FRAME_SCALE, text="Total",
               va="bottom", color=TOTAL_COLOR, alpha=1, axes=ax, fontsize=INTERNAL_FONTSIZE)

    ax.set_title(r"$\bf{B.}$ Fixation", fontsize=EXTERNAL_FONTSIZE, loc="left")
    ax.set_xlim([0, 100])
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels([0, 50, 100], fontsize=EXTERNAL_FONTSIZE)
    ax.set_xlabel(r"Shift size ($\Lambda$)", fontsize=EXTERNAL_FONTSIZE, labelpad=1)
    ax.set_ylim([0, PANEL_B_YMAX])
    ax.set_yticks([0, 0.1, 0.2])
    ax.set_yticklabels(["0", "0.1", "0.2"], fontsize=EXTERNAL_FONTSIZE)
    ax.set_ylabel(r"Number", fontsize=EXTERNAL_FONTSIZE, labelpad=0)


def make_figure4C(ax, ashifts, est, est_boundary):
    _draw_boundary_bands(ax, est_boundary)

    seg, new = est["segregating"], est["new"]
    ax.plot(ashifts, seg, color=SEG_COLOR)
    ax.plot(ashifts, new, color=NEW_COLOR)
    ax.plot(ashifts, seg + new, color=TOTAL_COLOR)

    ax.set_title(r"$\bf{C.}$ Establishment", fontsize=EXTERNAL_FONTSIZE, loc="left")
    ax.set_xlim([0, 100])
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels([0, 50, 100], fontsize=EXTERNAL_FONTSIZE)
    ax.set_xlabel(r"Shift size ($\Lambda$)", fontsize=EXTERNAL_FONTSIZE, labelpad=1)
    ax.set_ylim([0, MAX_Y])
    # 3-char labels: wider ones narrow Panel B and clip its "Segregating" label
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_yticklabels(["0", "0.1", "0.2", "0.3"], fontsize=EXTERNAL_FONTSIZE)
    ax.set_ylabel("Number", fontsize=EXTERNAL_FONTSIZE, labelpad=0)


# --------------------------------------------------------------------------- #
# Assemble the figure
# --------------------------------------------------------------------------- #
def make_figure_4(panelA, panelBC, N2U, figsize=(6.5, 2.7)):
    n2u, mean, sem, shift = panelA
    ashifts, fix, est, est_boundary = panelBC

    # 5 axes: A | spacer | B | spacer | C, so B and C sit close together
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=figsize, dpi=400,
                             gridspec_kw={"width_ratios": [25, 4, 25, 1, 25]})
    axes[1].axis("off")
    axes[3].axis("off")
    axes = [axes[0], axes[2], axes[4]]

    make_figure4C(axes[2], ashifts, est, est_boundary)
    make_figure4B(axes[1], ashifts, fix, est_boundary)
    make_figure4A(axes[0], n2u, mean, sem, shift)

    plt.subplots_adjust(wspace=0)
    plt.tight_layout(w_pad=-1.8)
    fig.canvas.draw()

    rounded_rect = FancyBboxPatch((-30, -0.16 * FRAME_SCALE), 270, 0.64 * FRAME_SCALE,
                                  boxstyle="Round, pad=0,rounding_size=5",
                                  mutation_aspect=0.8 / 300 * FRAME_SCALE, color=[0.92, 0.95, 1],
                                  edgecolor=None, clip_on=False, zorder=0)
    axes[1].add_patch(rounded_rect)
    axes[1].text(x=115, y=-0.13 * FRAME_SCALE, s=rf"Low mutational input ($2NU={N2U:g}$)", color="k",
                 fontsize=EXTERNAL_FONTSIZE, rotation=0, va="center", ha="center", clip_on=False)
    fig.canvas.draw()

    bbox = axes[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    return fig, width, height


def _render_square(out_path, panelA, panelBC, N2U, max_iter=20):
    """Nudge the figure height until Panel A is (nearly) square, then save at 6.5in width."""
    width_fig, height_fig = 6.5, 2.7
    fig, width, height = make_figure_4(panelA, panelBC, N2U, figsize=(width_fig, height_fig))
    for _ in range(max_iter):
        if np.isclose(width, height, rtol=0.005):
            break
        plt.close(fig)
        height_fig = height_fig * width / height
        fig, width, height = make_figure_4(panelA, panelBC, N2U, figsize=(width_fig, height_fig))
    make_figure_set_width(fig, out_path, target_width_inches=6.5, dpi=400)
    plt.close(fig)


def main(snakemake):
    a_rows = sorted(sim_contribution(p) for p in snakemake.input.simulations_A)
    n2u = np.array([r[0] for r in a_rows])
    mean = np.array([r[2] for r in a_rows])
    sem = np.array([r[3] for r in a_rows])
    shift = float(snakemake.params.shift)

    arows = analytic_rows(snakemake.input.analytic)
    ashifts = np.array([r[0] for r in arows])
    N2U = float(snakemake.params.N2U)
    fix = fixation_curves(arows, N2U)
    est = establishment_curves(arows, N2U)
    est_boundary = establishment_boundary(ashifts, est)

    out_path = (getattr(snakemake.output, "figure_4", None)
                if hasattr(snakemake.output, "figure_4") else None) or snakemake.output[0]

    _render_square(out_path, (n2u, mean, sem, shift), (ashifts, fix, est, est_boundary), N2U)


if "snakemake" in globals():
    main(snakemake)  # noqa: F821
