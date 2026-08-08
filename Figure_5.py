"""
Snakemake script: Figure 5, the adaptive response when large-effect alleles are
common (2NU = 10).

Everything is drawn from a single full_output simulation pickle -- 100 replicates of
the polygenic-adaptation Simulation at the manuscript parameters.

Panels
    A: mean distance from the optimum vs time, with Lande's infinitesimal prediction
       (dashed). The grey band is the rapid phase, ending at RAPID_PHASE_END.
    B: (inset in A) the phenotype distribution at five snapshot times, pooled over
       replicates after recentring each one.
    C: allele-frequency trajectories from a single replicate: up to N_PER_BIN lost
       alleles per effect-size bin, plus one small-effect allele that fixes (thick).

The dotted lines crossing panels A and C mark the distance D = a/2 at which selection
on a rare allele changes sign, and the time the mean distance falls through it.

Input
    simulations : the full_output simulation pickle (100 replicates)
Output
    figure_5 : the three-panel PNG

Run either as a Snakemake script (saves to snakemake.output.figure_5) or directly:
    python Figure_5.py [simulation.pkl] [output.png]
"""

import os
import sys
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass

from common_functions import make_figure_set_width


EXTERNAL_FONTSIZE = 11         # titles, axis labels
INTERNAL_FONTSIZE = 11         # in-figure annotations
INSET_FONTSIZE = 11            # panel B (the inset)

# Effect-size (a^2) bins colouring the panel C trajectories. The top bin is
# open-ended (the colorbar's "extend" arrow).
BOUNDS = [100, 200, 500, 1500]

# Squared effect sizes whose sign-change distance D = a/2 is marked in panels A and C.
SIGN_CHANGE_S = [1500, 500, 100]

# panel B: individuals sampled per replicate per snapshot, the grid the pooled
# densities are evaluated on, and the snapshots to draw in time order.
N_INDIVIDUALS = 500
PHENOTYPE_GRID = np.linspace(-60, 200, 600)
SNAPSHOT_KEYS = ["initial", "gen20", "quasi_static", "gen300", "final"]

# panel C: trajectories per effect-size bin, the replicate they come from (only a
# fallback; see select_replicate) and the seed for the choice among eligible alleles.
N_PER_BIN = 4
TRAJECTORY_REPLICATE = 0
TRAJECTORY_SEED = 1
FIXED_ALLELE_BIN = 0           # bin the single fixed allele is drawn from

# panel A is wider than panel C by the ratio of their mosaic widths, so give it that
# many more decades and the two share a scale.
PANEL_C_XMAX = 4000
MOSAIC_WIDTHS = [25, 1, 1, 2]
PANEL_A_XMAX = 10 ** (np.log10(PANEL_C_XMAX) * sum(MOSAIC_WIDTHS) / MOSAIC_WIDTHS[0])

# Right edge of the grey "rapid phase" band, in generations; None uses the mean of
# the replicates' convergence_time instead.
RAPID_PHASE_END = 85

PHENOTYPE_SEED = 0


def _truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    return mcolors.LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})", cmap(np.linspace(minval, maxval, n)))


def _mute_color(color, mute=0.85):
    """Wash a colour toward white, keeping it opaque."""
    muted = [c * mute + 1 * (1 - mute) for c in color]
    muted[-1] = 1
    return muted


_GIST = _truncate_colormap(plt.get_cmap("gist_earth_r"), 0.3, 0.9)
BIN_COLORS = [
    _mute_color(mcolors.to_rgba("goldenrod")),   # 100 <= a^2 < 200
    _mute_color(mcolors.to_rgba("teal")),        # 200 <= a^2 < 500
    _mute_color(_GIST(0.9)),                     # 500 <= a^2  (and the extend arrow)
]
CMAP = mcolors.LinearSegmentedColormap.from_list("Custom cmap", BIN_COLORS, len(BIN_COLORS))
CMAP.set_over(BIN_COLORS[-1])
NORM = mcolors.BoundaryNorm(BOUNDS, len(BIN_COLORS))


def bin_index(S):
    """Index of the effect-size bin holding squared effect size S (bins are
    right-open; everything above the top boundary joins the last bin)."""
    return int(np.clip(np.searchsorted(BOUNDS, S, side="right") - 1, 0, len(BIN_COLORS) - 1))


def edge_color(S):
    """Colour for a sign-change line at squared effect size S.

    Values on a bin edge take the bin below it, so each line is drawn in the colour
    of the alleles it bounds from above.
    """
    return BIN_COLORS[int(np.clip(np.searchsorted(BOUNDS, S, side="left") - 1,
                                  0, len(BIN_COLORS) - 1))]


# --------------------------------------------------------------------------- #
# Data reduction
# --------------------------------------------------------------------------- #
def load_simulation(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def mean_distance_trajectory(replicates):
    """Distance from the optimum each generation, averaged over replicates.

    Replicates stop at different generations (the run continues past stop_time while
    |d| > 1), so they are truncated to the shortest one before averaging.
    """
    n = min(len(rep["d_trajectory"]) for rep in replicates)
    return np.mean([rep["d_trajectory"][:n] for rep in replicates], axis=0)


def mean_quasi_static_time(replicates):
    """Mean generation at which the quasi-static phase begins (NaNs dropped)."""
    times = np.array([rep["convergence_time"] for rep in replicates], dtype=float)
    return float(np.nanmean(times))


def sign_change_points(D_mean, S_values):
    """For each S: (S, D, t) at which selection on a rare allele changes sign.

    D = sqrt(S)/2, and t is the interpolated generation at which the mean distance
    trajectory first falls through it.
    """
    points = []
    for S in S_values:
        D_crit = np.sqrt(S) / 2
        below = np.nonzero(D_mean <= D_crit)[0]
        if len(below) == 0:                     # never gets that close to the optimum
            points.append((S, D_crit, float(len(D_mean) - 1)))
            continue
        i = int(below[0])
        if i == 0:
            time = 0.0
        else:
            time = i - 1 + (D_mean[i - 1] - D_crit) / (D_mean[i - 1] - D_mean[i])
        points.append((S, D_crit, float(time)))
    return points


def sample_phenotypes(snapshot, sigma2, n_individuals, rng):
    """Sample `n_individuals` phenotypes from one replicate's allele-state snapshot.

    A phenotype is the sum of the large-effect genotypes (copies drawn Binomial(2, x))
    plus a normal deviate for the infinitesimal background. Fixed alleles are a common
    constant, dropped by the recentring in phenotype_densities().
    """
    a = np.asarray(snapshot["segregating"]["a"], dtype=float)
    x = np.asarray(snapshot["segregating"]["x"], dtype=float)
    genotypes = rng.binomial(2, x, size=(n_individuals, len(x)))
    background = rng.normal(0.0, np.sqrt(sigma2), n_individuals)
    return genotypes @ a + background


def phenotype_densities(replicates, shift, sigma2, seed=PHENOTYPE_SEED):
    """Pooled phenotype density at each snapshot time, as {key: (density, mean)}.

    Each replicate is recentred on zero and the pooled sample placed at the
    across-replicate mean phenotype, so the curves show the within-population spread
    rather than the scatter of replicate means.
    """
    rng = np.random.default_rng(seed)
    densities = {}
    for key in SNAPSHOT_KEYS:
        reps = [rep for rep in replicates if key in rep["snapshots"]]
        if not reps:
            continue
        mean_phenotype = shift - np.mean([rep["snapshots"][key]["D"] for rep in reps])
        pooled = []
        for rep in reps:
            phenotypes = sample_phenotypes(rep["snapshots"][key], sigma2, N_INDIVIDUALS, rng)
            pooled.append(phenotypes - phenotypes.mean())
        pooled = np.concatenate(pooled) + mean_phenotype
        densities[key] = (gaussian_kde(pooled)(PHENOTYPE_GRID), mean_phenotype)
    return densities


def select_replicate(replicates, default=TRAJECTORY_REPLICATE):
    """Index of the replicate panel C draws its trajectories from..
    """
    for i, rep in enumerate(replicates):
        if any(r["a"] > 0 and r["t_initial"] <= 1 and r["fate"] == "fixed"
               and bin_index(r["a"] ** 2) == FIXED_ALLELE_BIN
               for r in rep["recorded_trajectories"]):
            return i
    return default


def select_trajectories(replicate, n_per_bin=N_PER_BIN, seed=TRAJECTORY_SEED):
    """Pick the allele trajectories drawn in panel C from one replicate.

    Eligible alleles are aligned with the direction of adaptation (a > 0) and present
    at the shift. Among those, up to `n_per_bin` eventually-lost alleles are chosen at
    random per effect-size bin, plus one allele that fixes (from FIXED_ALLELE_BIN).
    """
    eligible = [r for r in replicate["recorded_trajectories"]
                if r["a"] > 0 and r["t_initial"] <= 1]
    rng = np.random.default_rng(seed)

    lost = [r for r in eligible if r["fate"] == "extinct"]
    per_bin = {i: [] for i in range(len(BIN_COLORS))}
    for i in rng.permutation(len(lost)):
        chosen = per_bin[bin_index(lost[i]["a"] ** 2)]
        if len(chosen) < n_per_bin:
            chosen.append(lost[i])

    # one fixed allele, preferably small-effect and standing at the shift; widen the
    # search only as far as needed
    fixed = [r for r in eligible if r["fate"] == "fixed"]
    small = [r for r in fixed if bin_index(r["a"] ** 2) == FIXED_ALLELE_BIN]
    if not (small or fixed):
        fixed = [r for r in replicate["recorded_trajectories"] if r["fate"] == "fixed"]
        small = [r for r in fixed if bin_index(r["a"] ** 2) == FIXED_ALLELE_BIN]
    candidates = small or fixed
    fixed_allele = candidates[int(rng.integers(len(candidates)))] if candidates else None

    return [r for bin_ in per_bin.values() for r in bin_], fixed_allele


# --------------------------------------------------------------------------- #
# Panel A: mean distance from the optimum
# --------------------------------------------------------------------------- #
def panel_A(fig, ax, D_mean, t_qs, points, shift, sigma2, N2U, N):
    times = np.arange(len(D_mean), dtype=float)
    ax.plot(times[1:], D_mean[1:], color="k")

    # Lande: D decays at rate V_A / (2N), with V_A the background sigma2 plus the
    # 4*2NU contributed by the large-effect alleles at MSDB.
    Va = sigma2 + 4 * N2U
    lande_time = np.logspace(0, 4, 1000)
    ax.plot(lande_time, shift * np.exp(-lande_time * Va / (2 * N)), color="k", linestyle="--")

    ax.set_xscale("log")
    ax.set_xlim([1, PANEL_A_XMAX])
    ax.set_ylim([0, shift])
    ax.set_xlabel("Time since shift (generations)", fontsize=EXTERNAL_FONTSIZE)
    ax.set_ylabel("Distance from optimum", fontsize=EXTERNAL_FONTSIZE)
    ax.set_title(r"$\bf{A.}$ Trajectory of mean distance from optimum",
                 fontsize=EXTERNAL_FONTSIZE, loc="left")

    ax.fill_between([0, t_qs], [0, 0], [shift * 1.125] * 2, color="k", alpha=0.1, edgecolor=None)

    ax.text(x=70, y=60, s="Lande", va="bottom", ha="center", color="k", fontsize=INTERNAL_FONTSIZE)
    ax.annotate("", xy=(70, 1.025 * shift * np.exp(-70 * Va / (2 * N))), xytext=(70, 60),
                arrowprops=dict(arrowstyle="->", color="k"))
    ax.text(x=10, y=55, s="Simulated mean", va="center", ha="right", color="k",
            fontsize=INTERNAL_FONTSIZE)
    ax.annotate("", xy=(17, 55), xytext=(10, 55), arrowprops=dict(arrowstyle="->", color="k"))

    # sign-change lines: horizontal at D = a/2 (broken around its own label) and vertical
    # down to the time axis at the crossing generation
    fig.canvas.draw()       # realize a renderer, so the label extents below are valid
    x_text = 3.2
    for S, value, time in points:
        color = edge_color(S)
        ax.plot([time, time], [0, value], color=color, linestyle=":", lw=3)

        label = ax.text(x=x_text, y=value, s=r"$a^2=$" + f"{S:g}", va="center", ha="left",
                        color=color, fontsize=INTERNAL_FONTSIZE)
        bbox = label.get_window_extent(renderer=fig.canvas.get_renderer())
        inv = ax.transData.inverted()
        x_min, _ = inv.transform(bbox.corners()[0])
        x_max, _ = inv.transform(bbox.corners()[2])
        ax.plot([0, x_min * 0.925], [value, value], color=color, linestyle=":", lw=3)
        if time > x_max * 1.2:    # else the label reaches past the crossing already
            ax.plot([x_max * 1.2, time], [value, value], color=color, linestyle=":", lw=3)

    S_top, value_top, time_top = points[int(np.argmax([p[0] for p in points]))]
    ax.text(x=time_top * 0.1575, y=value_top + 5, s="Selection reverses\nfor rare alleles",
            va="bottom", ha="center", color="k", fontsize=INTERNAL_FONTSIZE)


# --------------------------------------------------------------------------- #
# Panel B: the phenotype distribution through time (inset in panel A)
# --------------------------------------------------------------------------- #
def panel_B(ax, densities, shift):
    cmap = matplotlib.colormaps["Greys"]
    n = len(SNAPSHOT_KEYS)

    for i, key in enumerate(SNAPSHOT_KEYS):
        if key not in densities:
            continue
        density, mean_phenotype = densities[key]
        color = cmap(i / (n - 1) * 0.8 + 0.2)     # light (pre-shift) -> black (equilibrated)
        ax.plot(PHENOTYPE_GRID, density, color=color)
        j = int(np.argmin(np.abs(PHENOTYPE_GRID - mean_phenotype)))
        ax.plot([PHENOTYPE_GRID[j]] * 2, [0, density[j]], color=color, linestyle="--",
                lw=1, alpha=0.5)

    ax.set_title(r"$\bf{B.}$ Phenotypic evolution", fontsize=INSET_FONTSIZE, loc="left")
    ax.set_xlabel("Phenotype", size=INSET_FONTSIZE)
    ax.set_ylabel("Density", size=INSET_FONTSIZE)
    ax.set_xlim([-40, 180])
    ax.set_xticks([0, shift])
    ax.set_xticklabels([r"$0$", r"$\Lambda$"])
    ax.set_yticks([])

    # a "time" arrow spanning old optimum -> new optimum, shaded like the curves
    arrow_height = ax.get_ylim()[1]
    ax.set_ylim([0, arrow_height * 1.1])
    ax.annotate("", xy=(shift, arrow_height), xycoords="data",
                xytext=(10, arrow_height), textcoords="data",
                arrowprops=dict(facecolor="black", shrink=0.0, headlength=5),
                horizontalalignment="center", verticalalignment="bottom")
    for i in range(int(shift) - 5):
        color = cmap(i / shift * 0.8 + 0.2)
        ax.annotate("", xy=(i + 1, arrow_height), xycoords="data",
                    xytext=(i, arrow_height), textcoords="data",
                    arrowprops=dict(facecolor=color, edgecolor=color, width=9, shrink=0.02,
                                    headlength=0.01, headwidth=1),
                    horizontalalignment="center", verticalalignment="bottom")
    ax.text(x=shift / 2, y=arrow_height, s="time", ha="center", va="center",
            size=INSET_FONTSIZE, color="white")


# --------------------------------------------------------------------------- #
# Panel C: allele-frequency trajectories from one replicate
# --------------------------------------------------------------------------- #
def _plot_trajectory(ax, record, lw):
    trajectory = np.asarray(record["trajectory"], dtype=float)
    # generation 0 cannot be drawn on the log time axis
    visible = trajectory[trajectory[:, 0] >= 1]
    ax.plot(visible[:, 0], visible[:, 1], color=CMAP(NORM(record["a"] ** 2)), lw=lw)


def panel_C(ax, ax_colorbar, lost, fixed_allele, t_qs, points):
    for record in lost:
        _plot_trajectory(ax, record, lw=1)
    if fixed_allele is not None:
        _plot_trajectory(ax, fixed_allele, lw=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([1, PANEL_C_XMAX])
    ax.set_ylim([1e-4, 1])
    ax.set_xlabel("Time since shift (generations)", fontsize=EXTERNAL_FONTSIZE)
    ax.set_ylabel("Frequency", fontsize=EXTERNAL_FONTSIZE)
    ax.set_title(r"$\bf{C.}$ Trajectories of allele frequency",
                 fontsize=EXTERNAL_FONTSIZE, loc="left")

    ax.fill_between([0, t_qs], [1e-4, 1e-4], [1, 1], color="k", alpha=0.1, edgecolor=None)
    for S, _value, time in points:
        ax.plot([time, time], [1e-4, 1], color=edge_color(S), linestyle=":", lw=3)

    cbar = mpl.colorbar.Colorbar(ax_colorbar, cmap=CMAP, norm=NORM, orientation="vertical",
                                 extend="max", boundaries=BOUNDS)
    cbar.set_label(r"Effect size squared $(a^2)$", fontsize=EXTERNAL_FONTSIZE)
    cbar.set_ticks(BOUNDS)
    cbar.set_ticklabels([f"{b:g}" for b in BOUNDS])


# --------------------------------------------------------------------------- #
# Figure assembly
# --------------------------------------------------------------------------- #
def make_figure_5(results, out_path):
    parameters = results["parameters"]
    N, shift, sigma2, N2U = (parameters["N"], parameters["shift"],
                             parameters["sigma2"], parameters["N2U"])
    replicates = results["replicates"]

    D_mean = mean_distance_trajectory(replicates)
    # the grey band's right edge; panel B's quasi-static curve comes from the
    # snapshot the simulation took at its own convergence_time
    t_qs = RAPID_PHASE_END if RAPID_PHASE_END is not None else mean_quasi_static_time(replicates)
    points = sign_change_points(D_mean, SIGN_CHANGE_S)
    densities = phenotype_densities(replicates, shift=shift, sigma2=sigma2)
    lost, fixed_allele = select_trajectories(replicates[select_replicate(replicates)])

    # A spans the full width on top; the bottom row is panel C, a spacer, the colorbar
    # and a spacer
    fig, axes = plt.subplot_mosaic("AAAA;BCDE", figsize=(8, 8), dpi=400,
                                   gridspec_kw={"width_ratios": MOSAIC_WIDTHS})
    axes["C"].axis("off")
    axes["E"].axis("off")

    panel_A(fig, axes["A"], D_mean, t_qs, points, shift=shift, sigma2=sigma2, N2U=N2U, N=N)
    panel_B(axes["A"].inset_axes([0.625, 0.3, 0.35, 0.6], zorder=1), densities, shift=shift)
    panel_C(axes["B"], axes["D"], lost, fixed_allele, t_qs=t_qs, points=points)

    plt.subplots_adjust(hspace=0.3, wspace=0)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    make_figure_set_width(fig, out_path, target_width_inches=6.5, dpi=400)
    plt.close(fig)


def main(simulation_pickle, out_path):
    make_figure_5(load_simulation(simulation_pickle), out_path)


if "snakemake" in globals():
    inp = snakemake.input       # noqa: F821
    out = snakemake.output      # noqa: F821
    simulations = getattr(inp, "simulations", None) or inp[0]
    if not isinstance(simulations, str):
        simulations = simulations[0]
    main(simulations, getattr(out, "figure_5", None) or out[0])
elif __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    default_in = os.path.join(
        here, "results", "simulations_full_output",
        "sdistexpon_N2U10_shift80_sigma240.pkl")
    default_out = os.path.join(here, "results", "plots", "Figure_5.png")
    main(sys.argv[1] if len(sys.argv) > 1 else default_in,
         sys.argv[2] if len(sys.argv) > 2 else default_out)
