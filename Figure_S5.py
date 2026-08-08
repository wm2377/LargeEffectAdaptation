"""
Standalone / Snakemake script: Figure S5, the phase space of deterministic allelic
trajectory classes after an optimum shift, plus example allele-frequency and
distance trajectories.

Layout (mosaic-like gridspec, panels A-D as a 2x2 block on the left, E on the right):
    A (top-left)    : allele-frequency trajectories, small background variance
    B (top-right)   : allele-frequency trajectories, larger background variance
    C (bottom-left) : distance-to-optimum trajectories, small background variance
    D (bottom-right): distance-to-optimum trajectories, larger background variance
    E (right half)  : phase space of trajectory classes over (background variance,
                      shift size), coloured by class.

The five trajectory classes (see classify_trajectory) map, in order of increasing
shift, to the coloured regions of panel E:
    5 Quick loss                     (grey)   -- no fix, never strong
    4 Establishment and loss         (blue)   -- no fix, strong at some point
    3 Partial sweep and overshooting (green)  -- fix, not always strong
    2 Sweep with overshooting        (orange) -- fix, always strong, D goes negative
    1 Sweep without overshooting     (red)    -- fix, always strong, D never negative
The region boundaries are the four shift sizes from find_class_boundaries, swept over
sigma2, read from the trajectory_class_boundaries pickles or computed on the fly.

Panels A-D show, for two background variances, one representative shift per class: the
deterministic trajectory as a solid line plus a few realized ones faintly.

Run either as a Snakemake script (saves to snakemake.output.figure_s5) or directly:
    python Figure_S5.py [output.png]
Direct runs default to results/plots/Figure_S5.png next to this script.
"""

import os
import sys
import glob
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass

import common_functions as cf
from classify_trajectory import find_class_boundaries
from calculate_example_trajectories import calculate_example_trajectories

# fixed model parameters (global N, Figure S5 effect size)
N = 5000
S = 200.0

# two background variances for the trajectory panels: ~0 and 0.3 a^2
SMALL_SIGMA2 = 1.0
LARGE_SIGMA2 = 0.3 * S          # = 60

# fraction of the way up the class-3 band [b3, b2] used to pick the green
# trajectory's shift, per column
SMALL_SIGMA2_CLASS3_FRAC = 2.0 / 3.0
LARGE_SIGMA2_CLASS3_FRAC = 0.999

# realized trajectories: generate this many candidates per (sigma2, class), then show
# up to N_SHOW that share the deterministic trajectory's fate and timing
N_REALIZED_GENERATE = 300
N_SHOW = 5

# colours per class (1..5)
CLASS_COLORS = {
    1: [0.8, 0.3, 0.3],   # sweep without overshooting  (dark red)
    2: [1.0, 0.7, 0.3],   # sweep with overshooting      (orange)
    3: [0.3, 0.7, 0.3],   # partial sweep + overshooting (green)
    4: [0.3, 0.5, 1.0],   # establishment and loss       (blue)
    5: [0.5, 0.5, 0.5],   # quick loss                   (grey)
}


# --------------------------------------------------------------------------- #
# Panel E: load / compute the class boundaries over the sigma2 sweep
# --------------------------------------------------------------------------- #
def _sigma2_grid():
    """The Figure S5 sigma2 grid: 40 log-spaced in [0.01, 10] + 40 linear in [10, 100]."""
    return np.unique(np.concatenate([
        np.logspace(np.log10(0.01), np.log10(10), 40),
        np.linspace(10, 100, 40),
    ]))


def load_boundaries(pickle_paths=None, a=None, x0=None, N=N):
    """Return (sigma2, {k: boundary_array}) for k in (4, 3, 2, 1), sorted by sigma2.

    From the trajectory_class_boundaries pickles when `pickle_paths` is given; otherwise
    computed on the fly via find_class_boundaries over the sigma2 grid.
    """
    if pickle_paths:
        recs = []
        for path in pickle_paths:
            with open(path, "rb") as fh:
                r = pickle.load(fh)
            recs.append((float(r["parameters"]["sigma2"]), r["boundaries"]))
    else:
        recs = [(float(s), find_class_boundaries(a=a, x0=x0, N=N, sigma2=float(s)))
                for s in _sigma2_grid()]
    recs.sort(key=lambda t: t[0])
    sigma2 = np.array([t[0] for t in recs])
    boundaries = {k: np.array([t[1][k] for t in recs]) for k in (4, 3, 2, 1)}
    return sigma2, boundaries


def create_phase_panel(ax, sigma2, boundaries, a):
    """Panel E: fill the (sigma2/a^2, shift/a) plane with the five class regions."""
    top = 100.0
    x = (sigma2 / a ** 2).copy()
    x[0] = 0.0                                   # anchor the left edge at 0 (as original)
    # boundaries in shift/a units, clipped to the plotted shift range [0, 100]
    b4, b3, b2, b1 = (np.clip(boundaries[k], 0, top) / a for k in (4, 3, 2, 1))
    lo = np.zeros_like(x)
    hi = np.full_like(x, top / a)

    ax.fill_between(x, lo, b4, color=CLASS_COLORS[5])   # class 5: quick loss
    ax.fill_between(x, b4, b3, color=CLASS_COLORS[4])   # class 4: establishment and loss
    ax.fill_between(x, b3, b2, color=CLASS_COLORS[3])   # class 3: partial sweep
    ax.fill_between(x, b2, b1, color=CLASS_COLORS[2])   # class 2: sweep with overshooting
    ax.fill_between(x, b1, hi, color=CLASS_COLORS[1])   # class 1: sweep without overshooting

    ax.set_xlim(0, x[-1])
    ax.set_ylim(0, top / a)
    ax.set_xlabel(r"Background variance (units of $a^2$)")
    ax.set_ylabel(r"Shift size (units of $a$)", labelpad=0)
    # bold letter + regular words in one Text, which needs mathtext
    ax.set_title(r"$\bf{E.}$ Phase space of allelic trajectories")

    # curved region labels on a uniform x-grid, so the letters spread evenly despite
    # the log-dense sigma2 sampling; started in from the left edge so they are not cut off
    xu = np.linspace(0.025, x[-1], 100)
    def interp(b):
        return np.interp(xu, x, b)
    cf.CurvedText(x=xu, y=interp(hi) - 0.4,  text="Sweep without overshooting",
                  axes=ax, color=list(np.array(CLASS_COLORS[1]) * 0.5), fontsize=11)
    cf.CurvedText(x=xu, y=interp(b1) + 0.15, text="Sweep with overshooting",
                  axes=ax, color=list(CLASS_COLORS[2]), fontsize=11)
    cf.CurvedText(x=xu, y=interp(b2) + 0.15, text="Partial sweep and overshooting",
                  axes=ax, color=list(np.array(CLASS_COLORS[3]) * 0.8), fontsize=11)
    cf.CurvedText(x=xu, y=interp(b3) - 0.4,  text="Establishment and loss",
                  axes=ax, color=list(np.array(CLASS_COLORS[4]) * 0.5), fontsize=11)
    cf.CurvedText(x=xu, y=interp(b4) - 0.4,  text="Quick loss",
                  axes=ax, color="k", fontsize=11)


# --------------------------------------------------------------------------- #
# Panels A-D: expected + realized trajectories
# --------------------------------------------------------------------------- #
def _scaled_time(n, sigma2):
    """Generation index 0..n-1 in units of V_S/sigma2 = 2N/sigma2."""
    return np.arange(n) * sigma2 / (2.0 * N)


def _extend_distance(D, sigma2, target_scaled=12.0, npts=100):
    """Append the deterministic post-absorption decay of D so the line reaches the
    right edge of the panel. Returns (scaled_time, D) covering >= target_scaled units."""
    n = len(D)
    t = np.arange(n, dtype=float)
    ext_gen = max(1.0, target_scaled * 2.0 * N / sigma2 - n)
    t_ext = np.linspace(0.0, ext_gen, npts)
    D_ext = D[-1] * np.exp(-sigma2 / (2.0 * N) * t_ext)
    D_full = np.concatenate([D, D_ext[1:]])
    t_full = np.concatenate([t, t[-1] + t_ext[1:]]) * sigma2 / (2.0 * N)
    return t_full, D_full


def _representative_shifts(boundaries, class3_frac=2.0 / 3.0):
    """One shift per class from the four boundaries, matching the original weighting.

    boundaries: {4,3,2,1} = min shift producing class 4/3/2/1 (b4<=b3<=b2<=b1).
    class3_frac places the class-3 shift that fraction of the way up its (often narrow)
    band [b3, b2] from the fixation boundary b3 to the strong-selection boundary b2; a
    larger value picks a slightly larger, faster-fixing shift (default 2/3 = original).
    """
    b1, b2, b3, b4 = boundaries[1], boundaries[2], boundaries[3], boundaries[4]
    return {
        1: (100.0 + 5.0 * b1) / 6.0,       # above the overshoot boundary
        2: (b1 + b2) / 2.0,                # between strong-selection and overshoot
        3: b3 + class3_frac * (b2 - b3),   # within the fixation .. strong-selection band
        4: (b3 + b4) / 2.0,                # between ever-strong and fixation
        5: b4 * 0.8,                       # below the ever-strong boundary
    }


def plot_trajectory_panels(ax_x, ax_d, a, x0, sigma2, seed=0, class3_frac=2.0 / 3.0):
    """Draw one (allele, distance) column for a single sigma2 over all five classes."""
    boundaries = find_class_boundaries(a=a, x0=x0, N=N, sigma2=sigma2)
    shifts = _representative_shifts(boundaries, class3_frac=class3_frac)

    for cls in (1, 2, 3, 4, 5):
        shift = shifts[cls]
        color = CLASS_COLORS[cls]
        res = calculate_example_trajectories(
            N=N, a=a, x0=x0, sigma2=sigma2, shift=shift,
            n_realized=N_REALIZED_GENERATE, seed=seed + cls,
        )
        exp = res["expected"]

        # expected trajectory: allele (skip class 5, a trivial quick loss) + distance
        if cls != 5:
            ax_x.plot(_scaled_time(len(exp["x"]), sigma2), exp["x"], color=color, lw=2)
        td, Dd = _extend_distance(exp["D"], sigma2)
        ax_d.plot(td, Dd / shift, color=color, lw=2)

        # keep only trajectories that share the deterministic one's fate and reach it
        # within 20% of its time
        det_fixed = bool(exp["fixed"])
        det_absorbed = bool(exp["fixed"] or exp["lost"])
        det_time = len(exp["x"]) - 1
        shown = 0
        for r in res["realized"]:
            if shown >= N_SHOW:
                break
            if r["fixed"] != det_fixed:                                  # same fate
                continue
            if det_absorbed and abs(r["n_generations"] - det_time) > 0.2 * det_time:
                continue                                                 # similar timing
            if cls != 5:
                ax_x.plot(_scaled_time(len(r["x"]), sigma2), r["x"],
                          color=color, lw=1.5, alpha=0.25)
            tdr, Ddr = _extend_distance(r["D"], sigma2)
            ax_d.plot(tdr, Ddr / shift, color=color, lw=1.5, alpha=0.25)
            shown += 1

    # small and large sigma2 use different time windows
    small = sigma2 < 4
    for ax in (ax_x, ax_d):
        if small:
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.5, 1])
            ax.set_xticklabels(["0", "0.5", "1"])
        else:
            ax.set_xlim(0, 10)
            ax.set_xticks([0, 4, 8, 12])
            ax.set_xticklabels(["0", "4", "8", "12"])
        ax.set_xlabel(r"Time (units of $V_S/\sigma^2$)")
    ax_x.set_ylim(0, 1)
    ax_x.set_yticks([0, 0.5, 1])
    ax_x.set_yticklabels(["0", r"$\frac{1}{2}$", "1"])
    ax_x.set_ylabel("Allele frequency", labelpad=0)
    ax_d.set_ylim(-0.4, 1)
    ax_d.set_yticks([0, 0.5, 1])
    ax_d.set_yticklabels(["0", r"$\frac{1}{2}$", "1"])
    ax_d.set_ylabel(r"Distance ($D / \Lambda$)", labelpad=0)


# --------------------------------------------------------------------------- #
# Figure assembly
# --------------------------------------------------------------------------- #
def make_figure_s5(out_path, boundary_pickles=None, seed=116):
    a = np.sqrt(S)
    x0 = 1.0 / S

    mpl.rcParams["figure.dpi"] = 300
    mpl.rcParams["font.size"] = 11

    fig = plt.figure(constrained_layout=True, figsize=(8, 4))
    gs = fig.add_gridspec(2, 22)
    axE = fig.add_subplot(gs[:, 10:22])          # phase space (right half)
    axA = fig.add_subplot(gs[0, 0:5])            # allele, small sigma2
    axB = fig.add_subplot(gs[0, 5:10])           # allele, large sigma2
    axC = fig.add_subplot(gs[1, 0:5])            # distance, small sigma2
    axD = fig.add_subplot(gs[1, 5:10])           # distance, large sigma2

    # Panel E
    sigma2_grid, boundaries = load_boundaries(pickle_paths=boundary_pickles, a=a, x0=x0)
    create_phase_panel(axE, sigma2_grid, boundaries, a)

    # Panels A/C (small sigma2) and B/D (large sigma2)
    np.random.seed(seed)
    plot_trajectory_panels(axA, axC, a=a, x0=x0, sigma2=SMALL_SIGMA2, seed=seed,
                           class3_frac=SMALL_SIGMA2_CLASS3_FRAC)
    plot_trajectory_panels(axB, axD, a=a, x0=x0, sigma2=LARGE_SIGMA2, seed=seed + 100,
                           class3_frac=LARGE_SIGMA2_CLASS3_FRAC)

    axA.set_title("A.", loc="left", fontweight="bold")
    axB.set_title("B.", loc="left", fontweight="bold")
    axC.set_title("C.", loc="left", fontweight="bold")
    axD.set_title("D.", loc="left", fontweight="bold")

    # background boxes + column/row labels, in figure-fraction coords
    dark_orange = [0.7, 0.4, 0.1]
    col = 5.5 / 22
    fig.patches.extend([
        FancyBboxPatch((-0.03, -0.025), col, 0.5, boxstyle="Round, pad=0,rounding_size=0.025",
                       mutation_aspect=1, facecolor=[0.7, 0.7, 0.7], edgecolor="None",
                       clip_on=False, transform=fig.transFigure, figure=fig, zorder=-1),
        FancyBboxPatch((-0.03, 0.475), col, 0.5, boxstyle="Round, pad=0,rounding_size=0.025",
                       mutation_aspect=1, facecolor=[0.9, 0.9, 0.9], edgecolor="None",
                       clip_on=False, transform=fig.transFigure, figure=fig, zorder=-1),
        FancyBboxPatch((0, -0.025), col, 1.09, boxstyle="Round, pad=0,rounding_size=0.025",
                       mutation_aspect=1, facecolor=[0.95, 0.85, 0.75], edgecolor="None",
                       clip_on=False, transform=fig.transFigure, figure=fig, zorder=-1),
        FancyBboxPatch((col - 0.007, -0.025), col, 1.09, boxstyle="Round, pad=0,rounding_size=0.025",
                       mutation_aspect=1, facecolor=[0.85, 0.85, 1], edgecolor="None",
                       clip_on=False, transform=fig.transFigure, figure=fig, zorder=-1),
    ])
    fig.text(col / 2 * 3, 1.0, r"$\sigma^2=0.3a^2$", ha="center", va="bottom",
             fontsize=11, color="navy")
    fig.text(col / 2, 1.0, r"$\sigma^2 \approx 0$", ha="center", va="bottom",
             fontsize=11, color=dark_orange)
    fig.text(-0.027, 0.475 + 0.5 / 2, "Allelic traj.", fontweight="bold",
             ha="left", va="center", fontsize=11, rotation=90, color=[0.5, 0.5, 0.5])
    fig.text(-0.027, -0.025 + 0.5 / 2, "Phenotypic traj.", fontweight="bold",
             ha="left", va="center", fontsize=11, rotation=90, color="k")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(out_path, boundary_pickles=None):
    make_figure_s5(out_path, boundary_pickles=boundary_pickles)


if "snakemake" in globals():
    out = snakemake.output      # noqa: F821
    inp = snakemake.input       # noqa: F821
    pickles = list(getattr(inp, "boundaries", []) or [])
    main(getattr(out, "figure_s5", None) or out[0], boundary_pickles=pickles)
elif __name__ == "__main__":
    default = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "plots", "Figure_S5.png")
    out_path = sys.argv[1] if len(sys.argv) > 1 else default
    # use precomputed boundary pickles if present, else compute on the fly
    found = sorted(glob.glob(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results", "trajectory_class_boundaries", "S*_sigma2*.pkl")))
    main(out_path, boundary_pickles=found or None)
