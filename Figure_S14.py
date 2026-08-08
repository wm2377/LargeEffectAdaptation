"""
Snakemake script: Figure S14, the analytic-approximation companion to the phase-space
figure (Figure 6). Two panels, each comparing a closed-form threshold (solid) with the
1%-fixation contour read off the corresponding phase-space panel (dashed):

    A  "Maximum variance to fix": the largest initial genetic variance V_A(0) that still
       fixes, vs the proportion p of variance from large-effect alleles, for two shift
       sizes Lambda (80 firebrick, 50 navy). The dashed lines are the 1% contour of
       phase-space panel C at those shifts (default -> Lambda=80, alt -> Lambda=50).
    B  "Minimum shift to fix": the smallest shift (in units of sqrt(V_A(0))) that fixes,
       vs the initial genetic variance V_A(0), for two proportions p (0.1 navy, 0.5
       firebrick). The dashed lines are the 1% contour of phase-space panel D at those p
       (default -> p=0.5, alt -> p=0.1).

The closed forms (a^2 = 100 is the fixed large-effect size):
    max_variance(Lambda, p) = a (Lambda - a) / (ln(a^2) (1 - p))
    min_shift(V_A, p)       = (ln(a^2)/a)(1 - p) sqrt(V_A) + a / sqrt(V_A)

Grey / green shading marks regions outside the model's scope (V_A(0) > V_S/10 and
Lambda > sqrt(V_S)). The dashed lines are contours of the phase-space fields, so this
figure reads the same phase-space pickles Figure 6 does.

Input
    data_default : phase-space pickle for the default parameter set (Lambda=80, p=0.5)
    data_alt     : phase-space pickle for the alt parameter set     (Lambda=50, p=0.1)
Output
    figure : the two-panel PNG

Run as a Snakemake script or directly:
    python Figure_S14.py [data_default.pkl] [data_alt.pkl] [out.png]
"""

import os
import sys
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass

import common_functions  # noqa: F401  -- imported for its mathtext.fontset setting

FONTSIZE = 11
A2 = 100.0                     # fixed squared large-effect size a^2 the curves assume

# default = Lambda 80 / p 0.5 -> firebrick ; alt = Lambda 50 / p 0.1 -> navy
PARAMSET_COLOR = {"default": "firebrick", "alt": "navy"}


# --------------------------------------------------------------------------- #
# Closed-form thresholds
# --------------------------------------------------------------------------- #
def min_shift(Va, p, a2=A2):
    a = np.sqrt(a2)
    return (np.log(a2) / a) * (1 - p) * np.sqrt(Va) + a / np.sqrt(Va)


def max_variance(shift, p, a2=A2):
    a = np.sqrt(a2)
    return a * (shift - a) / (np.log(a2) * (1 - p))


# --------------------------------------------------------------------------- #
# 1%-fixation contour read off a phase-space panel grid
# --------------------------------------------------------------------------- #
def one_percent_contour(grid, level=0.01):
    """Return (x, y) vertices of the ``level`` fixation contour of one phase-space
    panel, smoothed as the phase-space figure smooths it. Picks the longest segment."""
    x, y = np.asarray(grid["x"]), np.asarray(grid["y"])
    Z = ndimage.gaussian_filter(np.asarray(grid["fixation"], dtype=float),
                                sigma=[1.5, 1], order=0, mode="nearest", truncate=1)
    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(x, y, Z, [level])
    segs = cs.allsegs[0]
    plt.close(fig_tmp)
    if not segs:
        return np.array([]), np.array([])
    seg = max(segs, key=len)
    return seg[:, 0], seg[:, 1]


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def panel_A(ax, data_by_set):
    """Maximum variance to fix vs proportion p."""
    p_values = np.linspace(0, 1, 1000)
    p_values[-1] = 0.999
    for shift, color in zip([50, 80], ["navy", "firebrick"]):
        ax.plot(p_values, [max_variance(shift, p) for p in p_values], color=color)

    for name, data in data_by_set.items():
        x, y = one_percent_contour(data["C"])          # panel C: x=p, y=variance
        keep = y < 1000
        ax.plot(x[keep], y[keep], color=PARAMSET_COLOR[name], ls="--")

    ax.set_title(r"$\bf{A.}$ Maximum variance to fix", fontsize=FONTSIZE)
    ax.set_xlabel("Proportion of variance from\n" + r"large-effect alleles ($p$)",
                  fontsize=FONTSIZE, labelpad=0)
    ax.set_ylabel(r"Maximum initial variance ($V_A(0)$)", fontsize=FONTSIZE, labelpad=0)
    ax.text(0.05, 260, r"$\Lambda=80$", color="firebrick", rotation=6)
    ax.text(0.05, 45, r"$\Lambda=50$", color="navy", rotation=6)
    ax.set_yscale("log")
    ax.set_xlim([0, 1]); ax.set_ylim([10, 1e5])
    ax.fill_between([0, 1], [1000, 1000], [1e5, 1e5], color="k", alpha=0.2)
    ax.text(0.05, 1500, r"$V_A(0)> \frac{V_S}{10}$", color="k", fontsize=FONTSIZE)


def panel_B(ax, data_by_set):
    """Minimum shift to fix vs initial genetic variance."""
    var_values = np.logspace(0, 5, 1000)
    for p, color in zip([0.1, 0.5], ["navy", "firebrick"]):
        ax.plot(var_values, [min_shift(Va, p) for Va in var_values], color=color)

    for name, data in data_by_set.items():
        x, y = one_percent_contour(data["D"])          # panel D: x=variance, y=shift/sqrt(Va)
        keep = (x > 5) & (y < np.where(x > 0, 100 / np.sqrt(np.maximum(x, 1e-9)), 0))
        ax.plot(x[keep], y[keep], color=PARAMSET_COLOR[name], ls="--")

    ax.fill_between(var_values, 100 / np.sqrt(var_values), [500] * len(var_values),
                    alpha=0.2, color="green")
    ax.fill_between([1000, 1e6], [1, 1], [500, 500], color="k", alpha=0.2)
    ax.text(6, 130, r"$\Lambda > \sqrt{V_S}$", color="green", fontsize=FONTSIZE)
    ax.text(1500, 24, r"$p=0.1$", color="navy", fontsize=FONTSIZE, rotation=40)
    ax.text(1500, 6.5, r"$p=0.5$", color="firebrick", fontsize=FONTSIZE, rotation=40)
    ax.text(1500, 130, r"$V_A(0)> \frac{V_S}{10}$", color="k", fontsize=FONTSIZE)
    ax.set_title(r"$\bf{B.}$ Minimum shift to fix", fontsize=FONTSIZE)
    ax.set_xlabel(r"Initial genetic variance ($V_A(0)$)", fontsize=FONTSIZE, labelpad=0)
    ax.set_ylabel(r"Minimum shift size ($\Lambda/\sqrt{V_A(0)}$)", fontsize=FONTSIZE, labelpad=0)
    ax.set_yscale("log"); ax.set_xscale("log")
    ax.set_xlim([5, 1e5]); ax.set_ylim([1, 200])
    ax.set_xticks([5, 100, 10000])
    ax.set_xticklabels(["5", r"$10^2$", r"$10^4$"])


def make_figure_S13(data_by_set, out_path):
    """Build the two-panel figure, iterating the height until panel B is square, then save."""
    height = 2.5
    for _ in range(12):
        fig, axes = plt.subplots(ncols=2, figsize=(6.5, height), dpi=400)
        panel_A(axes[0], data_by_set)
        panel_B(axes[1], data_by_set)
        axes[0].plot([], [], color="k", ls="--", label="1% fixation probability")
        axes[0].plot([], [], color="k", ls="-", label="Analytic approximation")
        plt.subplots_adjust(wspace=0.3)

        # one shared legend centred on the pair, below whichever x-label hangs lowest
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        lowest = min(ax.get_tightbbox(r).transformed(inv).y0 for ax in axes)
        centre = (axes[0].get_position().x0 + axes[1].get_position().x1) / 2
        fig.legend(*axes[0].get_legend_handles_labels(), framealpha=1, edgecolor="k",
                   handlelength=1, fontsize=FONTSIZE, ncol=2,
                   loc="upper center", bbox_to_anchor=(centre, lowest - 0.03),
                   bbox_transform=fig.transFigure)

        bbox = axes[1].get_window_extent()
        w_in, h_in = bbox.width / fig.dpi, bbox.height / fig.dpi
        if np.isclose(w_in, h_in, rtol=0.01):
            break
        height = height * w_in / h_in
        plt.close(fig)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def main(data_default, data_alt, out_path):
    make_figure_S13({"default": load(data_default), "alt": load(data_alt)}, out_path)


if "snakemake" in globals():
    inp = snakemake.input       # noqa: F821
    out = snakemake.output      # noqa: F821
    main(getattr(inp, "data_default", None) or inp[0],
         getattr(inp, "data_alt", None) or inp[1],
         getattr(out, "figure", None) or out[0])
elif __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    fp = os.path.join(here, "results", "phase_space")
    d_default = sys.argv[1] if len(sys.argv) > 1 else os.path.join(fp, "paramsetdefault.pkl")
    d_alt = sys.argv[2] if len(sys.argv) > 2 else os.path.join(fp, "paramsetalt.pkl")
    out_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(here, "results", "plots", "Figure_S14.png")
    main(d_default, d_alt, out_path)
