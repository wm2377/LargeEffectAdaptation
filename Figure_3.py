"""
    Fixation of a single large effect allele.
    The trajectories of the mean distance from the optimum (in A)
    and the trajectories of the large effect allele (in B and C) are
    shown for individual simulations with N=5000, a^2=200 with x_0=1⁄((〖2a〗^2 ) ), σ^2=40, and Λ=30 (purple) and 100 (yellow and black [in C]).
    The probabilities that a segregating allele establishes itself in the population and fixes (shown in D) a
    ssume N=5000, σ^2=40 (and 100 for the gray line in D), a^2=200, and average over the distribution of initial frequencies at MSDB.
    The probabilities for a new allele were calculated for the same parameters,
    assuming that the allele is equally likely to arise (at frequency 1⁄2N) any time between the shift and time t=606 (at which D_L (t)=δ).
    These calculations are described in supplement section 5. Simulation results were averaged over 16,000 and 64,000 replicas in (D) and (E), respectively.

Inputs (supplied by the plot_figure_3 rule; see the Snakefile)
    simulation_results          single_site pickles over the shift grid, per mode -- D/E error bars
    trajectory_results          ..._with_trajectories.pkl for shift 30 and 100 -- panels A/B/C
    fixation_probability        fixation_probability pickles -- the analytic lines in D and E
    fixation_probability_light  the same at sigma2 = SIGMA2_LIGHT -- the grey line in D

The establishment-probability lines are computed here rather than read from a pickle.

Run either as a Snakemake script (saves to snakemake.output.figure_3) or directly:
    python Figure_3.py [output.png]
Direct runs default to results/plots/Figure_3.png and glob the results directory for inputs.
"""

import os
import sys
import glob
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass

import common_functions as cf
import analytic_functions as af
from calculate_example_trajectories import expected_trajectory
from single_site_simulation_classes import SegregatingSimulation

# Model parameters, matching the single-site sweep in the Snakefile.
N = 5000
S = 200.0                  # squared effect size a^2 of the large-effect allele
SIGMA2 = 40.0              # background genetic variance of the simulations
SIGMA2_LIGHT = 100.0       # larger background variance, grey line in panel D

FONTSIZE = 10
TICK_FONTSIZE = 9

# Panels A-C: the two example optimum shifts and the initial frequency every
# segregating replicate starts at.
EXAMPLE_SHIFTS = (30, 100)
EXAMPLE_X0 = 1.0 / (2.0 * S)
SHIFT_COLORS = {30: "purple", 100: "goldenrod"}
EXTINCT_COLOR = "k"
# At shift 100 the allele establishes and fixes; at 30 it establishes but is lost
# anyway, because the optimum is reached before the sweep can finish.
EXAMPLE_ROLES = {100: "fix", 30: "loss"}
EXTINCT_SHIFT = 100        # replicate pool supplying panel C's black trajectory

PANEL_XMAX = 1000
INSET_XMAX = 40
INSET_YMAX = 0.01

SIMULATED_ALPHA = 0.5

# Panel A: (generation, height in units of Lambda, role, label) per arrow. Heights
# are stepped down so the ~200-generation-wide labels do not overlap.
PANEL_A_ARROWS = (
    (150, 0.78, "fix",   "Establish & fix"),
    (300, 0.62, "loss",  "Establish & loss"),
    (380, 0.40, "lande", "Lande"),
)
# Panel C: where "Immediate loss" is written, in axis fractions.
INSET_LOSS_LABEL_XY = (0.97, 0.04)

# Panels D/E: fixation and establishment are told apart by line style, not colour.
COLOR_FIXATION = "k"
COLOR_ESTABLISHMENT = "k"
COLOR_FIXATION_LIGHT = "gray"

# Panels D/E: the four dynamical regimes shaded behind the curves, in order of
# increasing shift.
REGIME_COLORS = (
    [0.30, 0.50, 1.00],   # none establish
    [0.60, 0.30, 0.80],   # establish but not fix
    [0.90, 0.30, 0.30],   # some fix
    [1.00, 0.75, 0.20],   # most fix
)
REGIME_ALPHA = 0.16

# Regime-boundary thresholds (see regime_boundaries). "Some fix" is per panel: a
# new allele's fixation probability peaks an order of magnitude lower.
FIX_THRESHOLD_SOME = {"segregating": 0.01, "new": 0.001}
FIX_THRESHOLD_MOST = 0.1     # P_fix first exceeds this ...
FIX_ESTABLISH_TOL = 0.05     # ... and comes within this fraction of P_est -> most fix

# Panels D/E: where each in-panel curve label starts, and how far left of its curve
# it is slid, both in units of a.
LABEL_START_ESTABLISHMENT = 0.75
LABEL_START_FIXATION = 2.5
LABEL_OFFSET_X = 0.2


# --------------------------------------------------------------------------- #
# Loading the pickles
# --------------------------------------------------------------------------- #
def _load_simulation_results(filepaths):
    """Simulated fixation probability vs shift, keyed by mode ('segregating' / 'new').

    Returns {mode: (shifts, mean_probability, se)}, sorted by shift, se = 2 SE.
    """
    results = {}
    for filepath in filepaths:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        fixed = np.asarray(data["fixed"], dtype=float)
        mode = data["parameters"]["mode"]
        shift = float(data["parameters"]["Lambda"])

        if mode not in results:
            results[mode] = ([], [], [])
        results[mode][0].append(shift)
        results[mode][1].append(fixed.mean())
        results[mode][2].append(2 * fixed.std() / np.sqrt(fixed.size))

    ordered = {}
    for mode, (shifts, means, ses) in results.items():
        order = np.argsort(shifts)
        ordered[mode] = tuple(np.asarray(v)[order] for v in (shifts, means, ses))
    return ordered


def _load_fixation_probability(filepaths):
    """Double-recursion fixation probability vs shift, from the fixation_probability pickles.

    Returns (shifts, p_segregating, p_new), sorted by shift. `p_new` is normalized by
    the time for the distance to relax to 1; panel E rescales it onto the
    establishment window.
    """
    shifts, p_seg, p_new = [], [], []
    for filepath in filepaths:
        with open(filepath, "rb") as f:
            r = pickle.load(f)
        shifts.append(float(r["parameters"]["shift"]))
        p_seg.append(r["fixation_probability"]["segregating"])
        p_new.append(r["fixation_probability"]["new"])
    order = np.argsort(shifts)
    return tuple(np.asarray(v, dtype=float)[order] for v in (shifts, p_seg, p_new))


def _load_trajectory_results(filepaths):
    """{shift: [replicate, ...]} from the ..._with_trajectories.pkl files."""
    trajectories = {}
    for filepath in filepaths:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        trajectories[int(data["parameters"]["Lambda"])] = data["replicates"]
    return trajectories


# --------------------------------------------------------------------------- #
# Establishment probability (computed here; there is no pickle for it)
# --------------------------------------------------------------------------- #
def new_allele_arrival_windows(shifts, N, S, sigma2):
    """The two candidate arrival windows for a new allele, in generations, per shift.

    Under Lande's equation D(t) = shift * exp(-t*sigma2/V_S), so:

    t_lande          the time for D to reach 1; the window simulations draw t0 from.
    t_establishment  the time for D to reach a/2, past which a single copy is no
                     longer beneficial. Panel E reports probabilities over this window.

    Both are 0 where no window exists (shift <= a/2).
    """
    V_S = 2.0 * N
    a = np.sqrt(S)
    shifts = np.asarray(shifts, dtype=float)
    t_lande = np.zeros_like(shifts)
    t_establishment = np.zeros_like(shifts)
    beneficial = shifts > a / 2
    t_lande[beneficial] = V_S / sigma2 * np.log(shifts[beneficial])
    t_establishment[beneficial] = V_S / sigma2 * np.log(shifts[beneficial] / (a / 2))
    return t_lande, t_establishment


def rescale_to_establishment_window(values, shifts, N, S, sigma2):
    """Renormalize a new-allele probability from the D -> 1 window onto the D -> a/2 one.

    An allele arriving after t_establishment never fixes, so it contributes 0 to
    either average and the two differ only by their denominators. Shifts with no
    establishment window give 0.
    """
    t_lande, t_establishment = new_allele_arrival_windows(shifts, N, S, sigma2)
    factor = np.zeros_like(t_lande)
    exists = t_establishment > 0
    factor[exists] = t_lande[exists] / t_establishment[exists]
    return np.asarray(values, dtype=float) * factor


def establishment_probability(shifts, N, S, sigma2):
    """Probability that a segregating / new large-effect allele of effect S establishes.

    Segregating: averaged over the initial frequencies at MSDB.
    New: averaged over arrivals uniform on the establishment window.
    """
    c = af.normalization_constant(S=S, N=N)
    _, t_establishment = new_allele_arrival_windows(shifts, N=N, S=S, sigma2=sigma2)

    p_seg, p_new = np.zeros(len(shifts)), np.zeros(len(shifts))
    for i, shift in enumerate(shifts):
        if shift <= 0:
            continue
        p_seg[i] = c * af.expected_number_of_established_segregating_alleles(
            S=S, N=N, shift=shift, sigma2=sigma2)
        if t_establishment[i] > 0:
            p_new[i] = af.expected_number_of_established_new_alleles(
                S=S, N=N, shift=shift, sigma2=sigma2) / t_establishment[i]
    return p_seg, p_new


# --------------------------------------------------------------------------- #
# The four dynamical regimes shaded behind panels D and E
# --------------------------------------------------------------------------- #
def regime_boundaries(shifts, p_fix, p_est, a, fix_threshold_some):
    """The three shifts at which the dynamics of an allele change.

    Returns (a/2, shift_some, shift_most), splitting the shift axis into four regimes:
    none establish, establish but none fix, some fix, most fix. A boundary that is
    never crossed comes back as np.nan (panel E has no "most fix" regime).
    """
    def first(condition):
        return float(shifts[np.argmax(condition)]) if condition.any() else np.nan

    shift_some = first(p_fix > fix_threshold_some)
    shift_most = first((p_fix > FIX_THRESHOLD_MOST)
                       & (p_fix >= (1 - FIX_ESTABLISH_TOL) * p_est))
    return a / 2, shift_some, shift_most


def shade_regimes(ax, boundaries, a, xmax):
    """Wash the four regimes of `boundaries` across `ax` (x in units of a), left to right.

    A boundary that is never crossed (np.nan) is pinned to the right edge.
    """
    edges = [0.0] + [b / a if np.isfinite(b) else xmax for b in boundaries] + [xmax]
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        lo, hi = np.clip([lo, hi], 0.0, xmax)
        if hi <= lo:
            continue
        ax.axvspan(lo, hi, color=REGIME_COLORS[i], alpha=REGIME_ALPHA, lw=0, zorder=0)


# --------------------------------------------------------------------------- #
# Panels A-C: picking and extending the example trajectories
# --------------------------------------------------------------------------- #
def _pad_to(values, n):
    """`values` truncated or extended to length n, holding its final value."""
    values = np.asarray(values, dtype=float)
    if len(values) >= n:
        return values[:n]
    return np.concatenate([values, np.full(n - len(values), values[-1])])


def _extend_distance(D, n, sigma2, N):
    """`D` extended to length n by the Lande decay D_{t+1} = D_t * (1 - sigma2/V_S).

    The recorded trajectories stop at absorption, short of panel A's window.
    """
    D = np.asarray(D, dtype=float)
    if len(D) >= n:
        return D[:n]
    decay = np.arange(1, n - len(D) + 1)
    return np.concatenate([D, D[-1] * (1.0 - sigma2 / (2.0 * N)) ** decay])


def select_fixed_replicate(replicates, x_deterministic, n_compare):
    """The fixed replicate whose frequency trajectory tracks the deterministic one best.

    Both are held at their final value out to n_compare generations and scored by RMS
    frequency difference.
    """
    target = _pad_to(x_deterministic, n_compare)
    best, best_score = None, np.inf
    for replicate in replicates:
        if not replicate["fixed"]:
            continue
        score = np.sqrt(np.mean((_pad_to(replicate["x_trajectory"], n_compare) - target) ** 2))
        if score < best_score:
            best, best_score = replicate, score
    return best


def select_established_then_lost_replicate(replicates, escape_frequency=INSET_YMAX,
                                           escape_generations=INSET_XMAX,
                                           max_generations=PANEL_XMAX):
    """The trajectory of an allele that escapes drift, rises, and is nonetheless lost.

    Requires a replicate that is lost, clears `escape_frequency` within
    `escape_generations` (visibly escaping drift in panel C), and is absorbed within
    `max_generations` (so panels A/B show the loss). Among those, take the highest
    peak -- the most legible rise and fall. Fallbacks drop one requirement at a time.
    """
    lost = [replicate for replicate in replicates if not replicate["fixed"]]
    def peak(replicate):
        return np.max(replicate["x_trajectory"])
    def peak_in_window(replicate):
        return np.max(np.asarray(replicate["x_trajectory"])[:escape_generations + 1])

    escaped = [r for r in lost if peak_in_window(r) >= escape_frequency]
    if not escaped:
        return max(lost, key=peak_in_window)
    visible = [r for r in escaped if r["n_generations"] <= max_generations]
    return max(visible or escaped, key=peak)


def select_extinct_replicate(replicates, max_generations=INSET_XMAX, max_frequency=INSET_YMAX):
    """The black trajectory of panel C: an allele lost early, without ever rising far.

    The longest-lived lost replicate that fits entirely inside the inset.
    """
    candidates = [
        replicate for replicate in replicates
        if not replicate["fixed"]
        and replicate["n_generations"] <= max_generations
        and np.max(replicate["x_trajectory"]) < max_frequency
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda replicate: replicate["n_generations"])


def build_examples(trajectories, N, S, sigma2):
    """Assemble everything panels A-C draw, per example shift.

    Per shift: the deterministic trajectory, plus one realized replicate playing that
    shift's EXAMPLE_ROLES part, shared by all three panels. EXTINCT_SHIFT additionally
    carries panel C's immediate-loss replicate.
    """
    examples = {}
    for shift in EXAMPLE_SHIFTS:
        replicates = trajectories[shift]
        simulation = SegregatingSimulation(N=N, S=S, sign=1, sigma2=sigma2,
                                           Lambda=float(shift), seed=0)
        deterministic = expected_trajectory(simulation, x0=EXAMPLE_X0, shift=float(shift))

        if EXAMPLE_ROLES[shift] == "fix":
            n_compare = max(len(deterministic["x"]), PANEL_XMAX + 1)
            replicate = select_fixed_replicate(replicates, deterministic["x"], n_compare)
        else:
            replicate = select_established_then_lost_replicate(replicates)

        examples[shift] = {
            "deterministic": deterministic,
            "x": np.asarray(replicate["x_trajectory"], dtype=float),
            "D": np.asarray(replicate["D_trajectory"], dtype=float),
            "extinct": (select_extinct_replicate(replicates)
                        if shift == EXTINCT_SHIFT else None),
        }
    return examples


# --------------------------------------------------------------------------- #
# Panels A-C
# --------------------------------------------------------------------------- #
def plot_distance_trajectories(ax, examples, N, sigma2):
    """Panel A: the mean distance from the optimum, relative to the shift that produced it.

    Per shift, the realized distance (solid) and the deterministic prediction (dashed),
    both continued past absorption by the Lande decay. The black dotted line is the
    Lande approximation, the same curve for both shifts once divided by Lambda.
    """
    time = np.arange(PANEL_XMAX + 1)
    curves = {}
    for shift in EXAMPLE_SHIFTS:
        example = examples[shift]
        color = SHIFT_COLORS[shift]
        deterministic = _extend_distance(example["deterministic"]["D"], len(time), sigma2, N) / shift
        curves[EXAMPLE_ROLES[shift]] = (deterministic, color)
        ax.plot(time, _extend_distance(example["D"], len(time), sigma2, N) / shift,
                color=color, lw=1.5, alpha=SIMULATED_ALPHA)
        ax.plot(time, deterministic, color=color, lw=1.5, ls="--")

    lande = np.exp(-time * sigma2 / (2.0 * N))
    curves["lande"] = (lande, "k")
    ax.plot(time, lande, color="k", lw=1.5, ls=":")

    # drop a vertical arrow onto each deterministic line and write its name above the tail
    for generation, height, role, label in PANEL_A_ARROWS:
        curve, color = curves[role]
        ax.annotate("", xy=(generation, curve[generation]), xytext=(generation, height),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5,
                                    shrinkA=0, shrinkB=0))
        ax.text(generation, height + 0.02, label,
                ha="center", va="bottom", fontsize=FONTSIZE, color=color)

    ax.set_xlim(0, PANEL_XMAX)
    ax.set_xticks(np.arange(0, PANEL_XMAX + 1, 200))
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", r"$\Lambda/2$", r"$\Lambda$"])
    ax.set_xlabel("Time since shift (generations)", fontsize=FONTSIZE, labelpad=1)
    ax.set_ylabel("Distance", fontsize=FONTSIZE)
    ax.set_title(r"$\bf{A.}$ Trajectory of mean distance from the optimum",
                 fontsize=FONTSIZE, loc="left")
    ax.legend(handles=[Line2D([], [], color="k", ls="-", label="Simulated"),
                       Line2D([], [], color="k", ls="--", label="Deterministic")],
              loc="upper right", fontsize=FONTSIZE, frameon=True, fancybox=True,
              edgecolor="0.6", framealpha=1.0)


def plot_frequency_trajectories(ax, examples):
    """Panel B: the large-effect allele's frequency, realized (solid) and deterministic (dashed)."""
    time = np.arange(PANEL_XMAX + 1)
    for shift in EXAMPLE_SHIFTS:
        example = examples[shift]
        color = SHIFT_COLORS[shift]
        ax.plot(time, _pad_to(example["x"], len(time)), color=color, lw=1.5,
                alpha=SIMULATED_ALPHA)
        ax.plot(time, _pad_to(example["deterministic"]["x"], len(time)),
                color=color, lw=1.5, ls="--")

    ax.set_xlim(0, PANEL_XMAX)
    ax.set_xticks(np.arange(0, PANEL_XMAX + 1, 200))
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("Time since shift (generations)", fontsize=FONTSIZE, labelpad=1)
    ax.set_ylabel("Frequency", fontsize=FONTSIZE)
    ax.set_title(r"$\bf{B.}$ Trajectory of large-effect allele", fontsize=FONTSIZE, loc="left")


def plot_establishment_inset(ax, examples):
    """Panel C: the first INSET_XMAX generations, where the allele escapes drift or is lost.

    The coloured trajectories are the opening of the same two replicates panel B follows;
    the black one is a replicate of the EXTINCT_SHIFT run that never establishes.
    """
    for shift in EXAMPLE_SHIFTS:
        x = _pad_to(examples[shift]["x"], INSET_XMAX + 1)
        ax.plot(np.arange(INSET_XMAX + 1), x, color=SHIFT_COLORS[shift], lw=1.5)

    for shift in EXAMPLE_SHIFTS:
        extinct = examples[shift]["extinct"]
        if extinct is not None:
            x = np.asarray(extinct["x_trajectory"], dtype=float)[:INSET_XMAX + 1]
            ax.plot(np.arange(len(x)), x, color=EXTINCT_COLOR, lw=1.5)
            ax.text(*INSET_LOSS_LABEL_XY, "Immediate loss", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=FONTSIZE, color=EXTINCT_COLOR)

    ax.set_xlim(0, INSET_XMAX)
    ax.set_xticks([0, INSET_XMAX])
    ax.set_ylim(0, INSET_YMAX)
    ax.set_yticks([0, INSET_YMAX])
    # the inset sits inside panel B, so ticks, labels and title are pulled in tight
    ax.tick_params(labelsize=TICK_FONTSIZE, pad=1, length=2)
    # negative pad lifts the x label into the gap between the two end tick labels
    ax.set_xlabel("Generations since shift", fontsize=FONTSIZE, labelpad=-9)
    ax.set_ylabel("Frequency", fontsize=FONTSIZE, labelpad=0)
    ax.set_title(r"$\bf{C.}$ Establishment or loss", fontsize=FONTSIZE, loc="left", pad=2)


# --------------------------------------------------------------------------- #
# Panels D and E
# --------------------------------------------------------------------------- #
_LABEL_DY_FRAC = 0.03      # a curve label is lifted this fraction of the y-range above it


def _label_curve(ax, x, y, text, color, x0, x1, fontsize=FONTSIZE, dx=LABEL_OFFSET_X,
                 below=False):
    """Write `text` along a curve with CurvedText, in place of a legend entry.

    The label starts where the curve reaches x0, slid `dx` to the left.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    selected = np.isfinite(y) & (x >= x0) & (x <= x1)
    if selected.sum() < 3:
        return
    ylo, yhi = ax.get_ylim()
    dy = _LABEL_DY_FRAC * (yhi - ylo)
    cf.CurvedText(x[selected] - dx, y[selected] + (-2 * dy if below else dy), text, ax,
                  color=color, fontsize=fontsize)


# Panel D: the "greater background variance" arrow. It is drawn at the probability
# at which the two background-variance fixation curves are compared, and pulled back
# from each curve by VARIANCE_ARROW_GAP (in units of a).
VARIANCE_ARROW_PROBABILITY = 0.2
VARIANCE_ARROW_LABEL = "Greater\nbackground\nvariance"
VARIANCE_ARROW_FONTSIZE = FONTSIZE - 1
VARIANCE_ARROW_GAP = 0.12
VARIANCE_ARROW_LABEL_DX = 0.125     # label offset right of the midpoint, in units of a


def _annotate_background_variance(ax, x_fix, p_fix, x_fix_light, p_fix_light):
    """Panel D: a grey arrow from the sigma2 fixation curve to the SIGMA2_LIGHT one.

    Both curves are read at VARIANCE_ARROW_PROBABILITY, so the arrow measures how much
    further the optimum has to move for the same fixation probability once the
    background variance doubles. Skipped if the grey curve never gets that high.
    """
    if len(x_fix_light) == 0 or np.max(p_fix_light) < VARIANCE_ARROW_PROBABILITY:
        return
    # both curves increase in the shift here, so np.interp inverts them
    x_start = np.interp(VARIANCE_ARROW_PROBABILITY, p_fix, x_fix) + VARIANCE_ARROW_GAP
    x_end = np.interp(VARIANCE_ARROW_PROBABILITY, p_fix_light, x_fix_light) - VARIANCE_ARROW_GAP
    if x_end <= x_start:
        return
    y = VARIANCE_ARROW_PROBABILITY
    ax.annotate("", xy=(x_end, y), xytext=(x_start, y),
                arrowprops=dict(arrowstyle="->", color=COLOR_FIXATION_LIGHT, lw=1.5,
                                shrinkA=0, shrinkB=0), zorder=4)
    ylo, yhi = ax.get_ylim()
    ax.text((x_start + x_end) / 2 + VARIANCE_ARROW_LABEL_DX, y + 0.015 * (yhi - ylo),
            VARIANCE_ARROW_LABEL,
            ha="center", va="bottom", fontsize=VARIANCE_ARROW_FONTSIZE,
            color=COLOR_FIXATION_LIGHT, linespacing=1.1, zorder=4)


def plot_fixation_probabilities_shift_sweep(ax_seg, ax_new, simulation_results,
                                            fixation, fixation_light, N=N, S=S, sigma2=SIGMA2):
    """Panels D and E: probability that a segregating / new allele establishes and fixes.

    Each panel carries the simulated fixation probability (black error bars, 2 SE), the
    double-recursion fixation probability (black solid) and the establishment
    probability (black dashed). Panel D adds the fixation probability at SIGMA2_LIGHT
    (grey). The x-axis is the shift in units of a.
    """
    a = np.sqrt(S)
    xmax = ax_seg.get_xlim()[1]

    shifts, p_fix_seg, p_fix_new = fixation
    p_est_seg, p_est_new = establishment_probability(shifts, N=N, S=S, sigma2=sigma2)
    # the pickles average over arrivals up to the Lande time; panel E reports the
    # average over the establishment window instead
    p_fix_new = rescale_to_establishment_window(p_fix_new, shifts, N=N, S=S, sigma2=sigma2)
    relative_shifts = shifts / a

    for mode, ax, p_fix, p_est in (("segregating", ax_seg, p_fix_seg, p_est_seg),
                                   ("new", ax_new, p_fix_new, p_est_new)):
        boundaries = regime_boundaries(shifts, p_fix, p_est, a,
                                       fix_threshold_some=FIX_THRESHOLD_SOME[mode])
        shade_regimes(ax, boundaries, a, xmax)
        ax.plot(relative_shifts, p_est, color=COLOR_ESTABLISHMENT, lw=2, ls="--", zorder=2)
        ax.plot(relative_shifts, p_fix, color=COLOR_FIXATION, lw=2, zorder=2)

    for mode, ax in (("segregating", ax_seg), ("new", ax_new)):
        sim_shifts, mean_probability, se_probability = simulation_results[mode]
        if mode == "new":
            # the simulations draw t0 over the Lande window too
            mean_probability, se_probability = (
                rescale_to_establishment_window(v, sim_shifts, N=N, S=S, sigma2=sigma2)
                for v in (mean_probability, se_probability))
        ax.errorbar(sim_shifts / a, mean_probability, yerr=se_probability, ls="",
                    marker=".", markersize=8, color=COLOR_FIXATION, zorder=3)

    light_shifts, p_fix_seg_light, _ = fixation_light
    ax_seg.plot(light_shifts / a, p_fix_seg_light, color=COLOR_FIXATION_LIGHT, lw=2, zorder=2)
    _annotate_background_variance(ax_seg, relative_shifts, p_fix_seg,
                                  light_shifts / a, p_fix_seg_light)

    # name the two black lines along the curves, in place of a legend
    _label_curve(ax_seg, relative_shifts, p_est_seg, "Establishment", COLOR_ESTABLISHMENT,
                 LABEL_START_ESTABLISHMENT, xmax)
    _label_curve(ax_seg, relative_shifts, p_fix_seg, "Fixation", COLOR_FIXATION,
                 LABEL_START_FIXATION, xmax)

    for ax in (ax_seg, ax_new):
        ax.set_xlabel("Shift size (in units of $a$)", fontsize=FONTSIZE, labelpad=1)
        ax.set_ylabel("Probability", fontsize=FONTSIZE)
        ax.set_xlim(0, xmax)
    ax_seg.set_title(r"$\bf{D.}$ Fixation probability: segregating", fontsize=FONTSIZE, loc="left")
    ax_new.set_title(r"$\bf{E.}$ Fixation probability: new", fontsize=FONTSIZE, loc="left")


# --------------------------------------------------------------------------- #
# Figure assembly
# --------------------------------------------------------------------------- #
def build_figure_3(fig_width, fig_height, examples, simulation_results, fixation,
                   fixation_light, N=N, S=S, sigma2=SIGMA2):
    """Draw the whole figure at the given canvas size.

    Returns (fig, fig_width, fig_height, width_inches, height_inches), the last two
    measuring panel D, so the caller can iterate the canvas toward a square panel D.
    """
    # tick labels take no fontsize= at their call sites, hence rcParams
    plt.rcParams["font.size"] = FONTSIZE
    plt.rcParams["xtick.labelsize"] = TICK_FONTSIZE
    plt.rcParams["ytick.labelsize"] = TICK_FONTSIZE

    # A and B stacked full-width across the top; D and E side by side below
    fig, axes = plt.subplot_mosaic("AA;BB;DE", figsize=(fig_width, fig_height), dpi=300,
                                   height_ratios=[1, 1, 2])

    plot_distance_trajectories(axes["A"], examples, N=N, sigma2=sigma2)
    plot_frequency_trajectories(axes["B"], examples)
    # panel C sits inside panel B, in the band the trajectories leave empty
    plot_establishment_inset(axes["B"].inset_axes([0.61, 0.27, 0.36, 0.54]), examples)

    # y-limits first: shade_regimes and the curved labels place text as a fraction
    # of the y-range
    axes["D"].set_xlim(0, 7)
    axes["D"].set_ylim(0, 0.6)
    axes["E"].set_xlim(0, 7)
    axes["E"].set_ylim(0, 0.08)
    plot_fixation_probabilities_shift_sweep(
        axes["D"], axes["E"], simulation_results, fixation, fixation_light,
        N=N, S=S, sigma2=sigma2)

    # w_pad is the gap between D and E: just wide enough for panel E's labels
    plt.tight_layout(h_pad=1.5, w_pad=0.4)
    fig.panels = axes          # so the caller can re-measure a panel after a rescale
    bbox = axes["D"].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    return fig, fig_width, fig_height, bbox.width, bbox.height


def make_figure_with_square_panel(build_function, panel, out_path, target_width_inches=6.5,
                                  fig_width=8, fig_height=8, dpi=300, max_iter=15,
                                  rtol=0.002, **kwargs):
    """Save `build_function`'s figure at a fixed tight width with `panel` square.

    Each iteration rebuilds the figure, sets it to the target width, then re-measures
    `panel` and stretches the canvas height to square it up, until it is square to
    `rtol`. The final set_width rescales uniformly, so it does not undo the squaring.
    """
    fig = None
    for _ in range(max_iter):
        if fig is not None:
            plt.close(fig)
        fig = build_function(fig_width=fig_width, fig_height=fig_height, **kwargs)[0]
        cf.make_figure_set_width(fig, None, target_width_inches=target_width_inches,
                                 dpi=dpi, save=False)
        fig.canvas.draw()
        bbox = fig.panels[panel].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        if np.isclose(bbox.width, bbox.height, rtol=rtol):
            break
        fig_width, fig_height = fig.get_size_inches()
        fig_height *= bbox.width / bbox.height

    cf.make_figure_set_width(fig, out_path, target_width_inches=target_width_inches, dpi=dpi)
    return fig


def make_figure_3(out_path, simulation_pickles, trajectory_pickles,
                  fixation_pickles, fixation_light_pickles, N=N, S=S, sigma2=SIGMA2):
    simulation_results = _load_simulation_results(simulation_pickles)
    trajectories = _load_trajectory_results(trajectory_pickles)
    fixation = _load_fixation_probability(fixation_pickles)
    fixation_light = _load_fixation_probability(fixation_light_pickles)
    examples = build_examples(trajectories, N=N, S=S, sigma2=sigma2)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig = make_figure_with_square_panel(
        build_figure_3, panel="D", out_path=out_path, target_width_inches=6.5,
        fig_width=8, fig_height=8, dpi=300, examples=examples,
        simulation_results=simulation_results, fixation=fixation,
        fixation_light=fixation_light, N=N, S=S, sigma2=sigma2)
    plt.close(fig)


def main(out_path, simulation_pickles, trajectory_pickles,
         fixation_pickles, fixation_light_pickles, N=N, S=S, sigma2=SIGMA2):
    make_figure_3(out_path, simulation_pickles, trajectory_pickles,
                  fixation_pickles, fixation_light_pickles, N=N, S=S, sigma2=sigma2)


def _discover_pickles(results_dir, S, sigma2):
    """The four input file lists, for a direct run outside Snakemake.

    The trailing '.pkl' anchors keep the '_Ssweep' and '_with_trajectories' variants out of
    the plain single_site glob.
    """
    s_token, sigma2_token = f"{S:g}", f"{sigma2:g}"
    single_site = os.path.join(results_dir, "single_site")
    fixation_dir = os.path.join(results_dir, "fixation_probability")
    return (
        sorted(glob.glob(os.path.join(
            single_site, f"mode*_S{s_token}_signpos_shift*_sigma2{sigma2_token}.pkl"))),
        [os.path.join(single_site,
                      f"modesegregating_S{s_token}_signpos_shift{shift}"
                      f"_sigma2{sigma2_token}_with_trajectories.pkl")
         for shift in EXAMPLE_SHIFTS],
        sorted(glob.glob(os.path.join(
            fixation_dir, f"S{s_token}_shift*_sigma2{sigma2_token}.pkl"))),
        sorted(glob.glob(os.path.join(
            fixation_dir, f"S{s_token}_shift*_sigma2{SIGMA2_LIGHT:g}.pkl"))),
    )


if "snakemake" in globals():
    p = snakemake.params      # noqa: F821
    inp = snakemake.input     # noqa: F821
    out = snakemake.output    # noqa: F821
    main(getattr(out, "figure_3", None) or out[0],
         simulation_pickles=list(inp.simulation_results),
         trajectory_pickles=list(inp.trajectory_results),
         fixation_pickles=list(inp.fixation_probability),
         fixation_light_pickles=list(inp.fixation_probability_light),
         N=p.N, S=float(p.S), sigma2=float(p.sigma2))
elif __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    default = os.path.join(here, "results", "plots", "Figure_3.png")
    main(sys.argv[1] if len(sys.argv) > 1 else default,
         *_discover_pickles(os.path.join(here, "results"), S=S, sigma2=SIGMA2))
