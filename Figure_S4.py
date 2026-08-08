"""
Standalone script: Figure S4, the four supplementary analytic approximations for the
expected number of large-effect *new* alleles that fix following an optimum shift --
equations S26, S27, S28 and S36, as returned by analytic_functions.evaluate_equations --
plotted against the optimum-shift size for a fixed squared effect size S = a^2, background
genetic variance sigma2, and Wright-Fisher population size N.

Each panel also carries a grey "Double recursion" reference line, read from the
fixation_probability pickles (computed on the fly on a standalone run).

    Panel A "Fixation probability: segregating" -- Eqs. S26, S27, S28.
        S26: the exact expression, over the whole shift range.
        S27: the x_c << x_est approximation of S26, drawn only in that regime.
        S28: the complementary exponential approximation, drawn where x_c > x_est.
    Panel B "Fixation probability: new" -- Eq. S36, over the whole shift range.

Run either as a Snakemake script (saves to snakemake.output.figure_s4) or directly:
    python Figure_S4.py [output.png]
Direct runs default to results/plots/Figure_S4__S100_sigma240.png next to this script.
"""

import os
import sys
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common_functions as cf
import analytic_functions as af


# illustrative defaults for a direct run; the Snakemake rule overrides these
S = 100.0
SIGMA2 = 40.0
N = 5000

# Boundary factor between the S27 and S28 regimes: S27 is drawn where x_c <= C*x_est
# and S28 where x_c > C*x_est (C = 1 splits exactly at x_c = x_est).
C = 0.10


def compute_curves(shifts, a, N, sigma2):
    """evaluate_equations over the shift sweep, plus the x_c and x_est arrays.

    Returns (S26, S27, S28, S36, x_c, x_est): the four equation arrays and the two
    frequencies (defined exactly as in evaluate_equations) whose ratio sets the S27 / S28
    regime boundary. A shift at which an equation is undefined (a domain error inside a log,
    a non-finite normalization integral, ...) becomes np.nan so matplotlib leaves a gap.
    """
    n = len(shifts)
    S26, S27, S28, S36 = (np.full(n, np.nan) for _ in range(4))
    for i, shift in enumerate(shifts):
        try:
            vals = af.evaluate_equations(shift=shift, a=a, N=N, sigma2=sigma2)
        except Exception:
            continue
        S26[i], S27[i], S28[i], S36[i] = [v if np.isfinite(v) else np.nan for v in vals]
    # x_c / x_est set the S27 vs S28 regime boundary. shifts > a/2 here, so x_est > 0.
    x_est = 1.0 / ((2 * a) * (shifts - a / 2))
    x_c = np.exp(-a * (shifts - a) / sigma2)
    return S26, S27, S28, S36, x_c, x_est


# vertical fraction of the y-range by which a curve label is lifted above its curve.
_LABEL_DY_FRAC = 0.03


def _label_curve(ax, x, y, text, color, x0, x1, fontsize=11, below=False,steep=False):
    """Write `text` along a curve with CurvedText, in place of a legend.

    The label follows the curve between x0 and x1, offset above it (or below it when
    `below` is True). Does nothing if too few valid samples fall in [x0, x1].
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sel = np.isfinite(y) & (x >= x0) & (x <= x1)
    if sel.sum() < 3:
        return
    ylo, yhi = ax.get_ylim()
    dy = _LABEL_DY_FRAC * (yhi - ylo)
    if steep:
        x -= 0.075
    if below:
        x += 0.35
    cf.CurvedText(x[sel], y[sel] + (-2*dy if below else dy), text, ax, color=color, fontsize=fontsize)


def _load_double_recursion(pickles, N, S, sigma2, n_grid=21):
    """Return (shifts, p_seg, p_new) for the double-recursion fixation-probability lines.

    Read from the fixation_probability pickles, or -- when `pickles` is falsy -- computed
    directly over a coarse shift grid, which is much slower.
    """
    shifts, p_seg, p_new = [], [], []
    if pickles:
        for path in pickles:
            with open(path, "rb") as fh:
                r = pickle.load(fh)
            shifts.append(r["parameters"]["shift"])
            p_seg.append(r["fixation_probability"]["segregating"])
            p_new.append(r["fixation_probability"]["new"])
    else:
        from calculate_analytic_curves import calculate_fixation_probability
        for shift in np.linspace(0, 100, n_grid):
            r = calculate_fixation_probability(N=N, S=S, sigma2=sigma2, shift=float(shift))
            shifts.append(float(shift))
            p_seg.append(r["fixation_probability"]["segregating"])
            p_new.append(r["fixation_probability"]["new"])
    order = np.argsort(shifts)
    return np.asarray(shifts)[order], np.asarray(p_seg)[order], np.asarray(p_new)[order]


def plot_equations(fig_width, fig_height, N, S, sigma2, C=C, double_recursion=None):
    """Draw the two-panel Figure S4 vs the effect-size-relative shift (Lambda / a).

    Panel A (segregating): Eqs. S26/S27/S28, each over its regime of validity. Panel B
    (new): Eq. S36. Both carry the grey "Double recursion" line from `double_recursion`
    = (shift, P_seg, P_new) when supplied, and are labelled in place with CurvedText.
    Returns (fig, fig_width, fig_height, width_inches, height_inches) for
    make_figure_square.
    """
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=300)

    a = np.sqrt(S)
    # start just above a/2, below which x_est flips sign and the approximations are
    # not defined
    shifts = np.linspace(a / 2 + 0.5, 100, 400)
    rel = shifts / a
    S26, S27, S28, S36, x_c, x_est = compute_curves(shifts, a=a, N=N, sigma2=sigma2)
    s27_masked = np.where(x_c <= C * x_est, S27, np.nan)   # only where x_c <= C*x_est
    s28_masked = np.where(x_c > C * x_est, S28, np.nan)    # only where x_c > C*x_est

    dr_rel = dr_seg = dr_new = None
    if double_recursion is not None:
        dr_shift, dr_seg, dr_new = double_recursion
        dr_rel = np.asarray(dr_shift) / a

    axA.plot(rel, S26, color="cornflowerblue", linestyle="-", linewidth=2.5)
    axA.plot(rel, s27_masked, color="firebrick", linestyle="--", linewidth=2.5)
    axA.plot(rel, s28_masked, color="goldenrod", linestyle=":", linewidth=2.5)
    if dr_rel is not None:
        axA.plot(dr_rel, dr_seg, color="gray", linestyle="-", linewidth=2.5, zorder=1)
    axA.set_xlim(left=0,right=7)
    axA.set_ylim(0, 0.7)
    _label_curve(axA, rel, s28_masked, "Eq. S28", "goldenrod", 2.0, 7.0,steep=True)
    _label_curve(axA, rel, S26, "Eq. S26", "cornflowerblue", 3.0, 7.0)
    _label_curve(axA, rel, s27_masked, "Eq. S27", "firebrick", 5.0, 7.0)
    if dr_rel is not None:
        _label_curve(axA, dr_rel, dr_seg, "Double recursion", "gray", 2.8,7.0, below=True)
    axA.set_title(r"$\mathbf{A.}$ Fixation probability: segregating", fontsize=11, loc="left")
    axA.set_xlabel(r"Shift size relative to $a$ ($\Lambda/a$)", fontsize=11)
    axA.set_ylabel("Probability", fontsize=11)

    axB.plot(rel, S36, color="cornflowerblue", linestyle="-", linewidth=2.5)
    if dr_rel is not None:
        axB.plot(dr_rel, dr_new, color="gray", linestyle="-", linewidth=2.5, zorder=1)
    axB.set_xlim(left=0,right=7)
    axB.set_ylim(0, 0.04)
    _label_curve(axB, rel, S36, "Eq. S36", "cornflowerblue", 4.0, 7)
    if dr_rel is not None:
        _label_curve(axB, dr_rel, dr_new, "Double recursion", "gray", 4.0, 7, below=True)
    axB.set_title(r"$\mathbf{B.}$ Fixation probability: new", fontsize=11, loc="left")
    axB.set_xlabel(r"Shift size relative to $a$ ($\Lambda/a$)", fontsize=11)
    axB.set_ylabel("Probability", fontsize=11)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.45)   # widen the gap between the two panels
    bbox = axA.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    return fig, fig_width, fig_height, bbox.width, bbox.height


def main(out_path, N=N, S=S, sigma2=SIGMA2, C=C, dr_pickles=None):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    # loaded once, since make_figure_square re-runs plot_equations
    double_recursion = _load_double_recursion(dr_pickles, N=N, S=S, sigma2=sigma2)
    fig = cf.make_figure_square(
        plot_equations, fig_width=8, fig_height=4.5, N=N, S=S, sigma2=sigma2, C=C,
        double_recursion=double_recursion)[0]
    cf.make_figure_set_width(fig, out_path, target_width_inches=6.5, dpi=300)


if "snakemake" in globals():
    p = snakemake.params      # noqa: F821
    out = snakemake.output    # noqa: F821
    inp = snakemake.input     # noqa: F821
    dr = list(getattr(inp, "fixation_probability", []) or [])
    main(getattr(out, "figure_s4", None) or out[0], N=p.N, S=p.S, sigma2=p.sigma2, dr_pickles=dr)
elif __name__ == "__main__":
    default = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "plots",
        "Figure_S4__S100_sigma240.png",
    )
    main(sys.argv[1] if len(sys.argv) > 1 else default)
