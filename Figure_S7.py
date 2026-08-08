"""
Snakemake script: plot, vs optimum shift size and split by the source of each
fixed large-effect allele, two summaries as separate figures:

    1. mean number of fixations
    2. mean contribution to adaptation  (sum of 2*a*(1-x_0) over fixed alleles)

Source split (from each fixed allele's arrival generation fixed_arrival_times):
    * all         : every fixed large-effect allele
    * segregating : standing variation present at the shift (t_initial == 0)
    * new         : de novo mutations that arose after the shift (t_initial > 0)

For each shift the mean (and standard error) over that pickle's replicates is
plotted as points; the analytic expectation (calculate_analytic_curves.py) is
overlaid as a matched-color line.

Input
    simulations : one simulation pickle per shift (same sdist / N2U / sigma2)
    analytic    : one analytic-curve pickle per shift
Output
    fixations    : the headline Figure S7 -- mean number of fixations vs shift,
                   styled like Figure 4 panel B.
    contribution : mean contribution to adaptation vs shift, in the generic
                   diagnostic style (default colors, legend, parametric title).
"""

import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from common_functions import CurvedText, make_figure_set_width


INTERNAL_FONTSIZE = 10         # in-figure curve labels (CurvedText)
EXTERNAL_FONTSIZE = 11         # title, axis labels, tick labels
NEW_COLOR = "red"
SEG_COLOR = "sienna"
TOTAL_COLOR = "k"
STYLE_COLORS = {"all": TOTAL_COLOR, "segregating": SEG_COLOR, "new": NEW_COLOR}

# background regime bands, as in Figure 4B
FIXED_BOUNDARY = 35
BAND_NO_EST = [0.8, 0.8, 1.0]
BAND_EST = [0.9, 0.8, 0.9]
BAND_FIX = [1.0, 0.8, 0.8]
BAND_TOP = 0.22                        # drawn up to the panel's y-limit


# how each fixed allele is classified from its arrival generation
SOURCES = {
    "all":         lambda t: np.ones(t.shape, dtype=bool),
    "segregating": lambda t: t == 0,     # standing variation at the shift
    "new":         lambda t: t > 0,      # de novo after the shift
}

LABELS = {
    "all":         "all large-effect alleles",
    "segregating": "segregating (standing variation)",
    "new":         "new (de novo) mutations",
}

# fixed color per source so the simulation points and the analytic curve match
COLORS = {
    "all":         "C0",
    "segregating": "C1",
    "new":         "C2",
}

# per-metric analytic-pickle key, y-axis label and title stem
METRICS = ("fixations", "contribution")
ANALYTIC_KEY = {"fixations": "expected_fixations",
                "contribution": "expected_contribution"}
YLABEL = {"fixations": "mean number of fixations",
          "contribution": "mean contribution to adaptation"}
TITLE = {"fixations": "Fixations",
         "contribution": "Contribution to adaptation"}


def summarize(path):
    """Return (shift, means, sems, parameters) for one simulation pickle.

    means/sems are nested as [metric][source], over replicates of, per replicate:
        'fixations'    : the number of fixed alleles of that source
        'contribution' : sum of 2*a*(1-x_0) over those fixed alleles
    """
    with open(path, "rb") as f:
        res = pickle.load(f)

    N = res["parameters"]["N"]

    vals = {m: {s: [] for s in SOURCES} for m in METRICS}
    for rep in res["replicates"]:
        t = np.asarray(rep["fixed_arrival_times"])
        a = np.asarray(rep["fixed_effect_sizes"])     # signed a = +/- sqrt(S)
        if "fixed_initial_frequencies" in rep:
            x0 = np.asarray(rep["fixed_initial_frequencies"])
        else:
            x0 = np.full(a.shape, 1.0 / (2.0 * N))     # older pickles
        contrib = 2.0 * a * (1.0 - x0)
        for source, mask in SOURCES.items():
            m = mask(t)
            vals["fixations"][source].append(int(np.count_nonzero(m)))
            vals["contribution"][source].append(float(np.sum(contrib[m])))

    means = {m: {} for m in METRICS}
    sems = {m: {} for m in METRICS}
    for metric in METRICS:
        for source in SOURCES:
            v = np.asarray(vals[metric][source], dtype=float)
            means[metric][source] = v.mean() if v.size else np.nan
            sems[metric][source] = (2.0 * v.std() / np.sqrt(v.size)) if v.size > 1 else 0.0

    return res["parameters"]["shift"], means, sems, res["parameters"]


def summarize_analytic(path):
    """Return (shift, analytic) for one analytic-curve pickle.

    analytic maps metric -> {source: expected value}, metric in
    ('fixations', 'contribution'). 'contribution' is None for older pickles that
    predate calculate_analytic_curves' expected_contribution output.
    """
    with open(path, "rb") as f:
        res = pickle.load(f)
    analytic = {
        "fixations": res[ANALYTIC_KEY["fixations"]],
        "contribution": res.get(ANALYTIC_KEY["contribution"]),
        "establishments": res.get("expected_establishments"),
    }
    return res["parameters"]["shift"], analytic


def _establishment_boundary(arows, N2U, threshold=1e-3):
    """Smallest shift at which total (seg+new) establishment exceeds `threshold`.

    Establishment is the N2U-scaled analytic expectation (no competing-risk thinning),
    matching Figure_4.establishment_boundary so the S7 regime bands line up with 4B.
    Returns 0.0 if the analytic pickles predate expected_establishments.
    """
    if any(a["establishments"] is None for _, a in arows):
        return 0.0
    ashifts = np.array([s for s, _ in arows])
    seg = np.array([a["establishments"]["segregating"] for _, a in arows]) * N2U
    new = np.array([a["establishments"]["new"] for _, a in arows]) * N2U
    over = np.where(seg + new > threshold)[0]
    return float(ashifts[over[0]]) if over.size else 0.0


def _analytic_curves(arows, metric, N2U):
    """Per-source analytic `metric` vs shift at fixed 2NU, bounded by gap-closing.

    The first large-effect substitution to fix closes the phenotypic gap, so at most
    one large-effect allele fixes. The expected number of fixations from a source is
    therefore the *probability* that at least one such allele fixes, which saturates
    at 1 rather than growing linearly in 2NU:

        P_seg = 1 - exp(-aseg)                       (segregating)
        P_new = exp(-aseg) * (1 - exp(-mu_new))      (new, only if seg did not fix first)
        all   = P_seg + P_new = 1 - exp(-(aseg + mu_new))   (<= 1)

    where aseg = (per-2NU segregating fixation rate) * 2NU and mu_new likewise for new.
    The contribution to adaptation is the per-fixation contribution times that fixation
    probability. Identical logic to Figure_S6.analytic_curve (here vectorized over shift
    at fixed 2NU). Returns the per-source dict, or None if the metric is absent.
    """
    if any(a[metric] is None for _, a in arows):
        return None
    fix_seg = np.array([a["fixations"]["segregating"] for _, a in arows])
    fix_new = np.array([a["fixations"]["new"] for _, a in arows])
    p_seg = 1-np.exp(-fix_seg * N2U)                          # 1 - exp(-aseg)
    p_new = np.exp(-fix_seg * N2U) * (1-np.exp(-fix_new * N2U))  # gap not closed by seg first

    if metric == "fixations":
        seg, new = p_seg, p_new
    else:  # contribution = (contribution per fixation) * (probability of fixation)
        contrib_seg = np.array([a[metric]["segregating"] for _, a in arows])
        contrib_new = np.array([a[metric]["new"] for _, a in arows])
        with np.errstate(divide="ignore", invalid="ignore"):
            per_seg = np.where(fix_seg > 0, contrib_seg / fix_seg, 0.0)
            per_new = np.where(fix_new > 0, contrib_new / fix_new, 0.0)
        seg, new = per_seg * p_seg, per_new * p_new
    return {"segregating": seg, "new": new, "all": seg + new}


def _plot_panel(ax, rows, arows, metric, params, N2U):
    """Draw one metric panel: simulation points + analytic lines, per source."""
    shifts = np.array([r[0] for r in rows])
    ashifts = np.array([r[0] for r in arows])

    # simulation: points with error bars
    for source in ("all", "segregating", "new"):
        y = np.array([r[1][metric][source] for r in rows])
        yerr = np.array([r[2][metric][source] for r in rows])
        ax.errorbar(shifts, y, yerr=yerr, marker="o", linestyle="none",
                    capsize=3, color=COLORS[source])

    # analytic: matched-color lines, skipped if the metric is missing from the pickles
    curves = _analytic_curves(arows, metric, N2U)
    if curves is not None:
        for source in ("all", "segregating", "new"):
            ax.plot(ashifts, curves[source], color=COLORS[source], linestyle="-")

    ax.set_xlabel("shift size")
    ax.set_ylabel(YLABEL[metric])
    # the output filename is static (no wildcards), so read the parameters
    # recorded in the pickles instead of snakemake.wildcards
    sdist_name = getattr(getattr(params.get("sdist", None), "dist", None), "name", "sdist")
    ax.set_title(
        f"{TITLE[metric]} vs shift  "
        f"(sdist={sdist_name}, N2U={params['N2U']}, "
        f"sigma2={params['sigma2']}, N={params['N']}, "
        f"{params['n_replicates']} reps)"
    )

    # two-part legend: source (color) and data type (simulation points vs analytic line)
    source_handles = [Line2D([], [], color=COLORS[s], marker="o", linestyle="-",
                             label=LABELS[s]) for s in ("all", "segregating", "new")]
    style_handles = [
        Line2D([], [], color="0.3", marker="o", linestyle="none", label="simulation"),
        Line2D([], [], color="0.3", marker="none", linestyle="-", label="analytic"),
    ]
    leg = ax.legend(handles=source_handles, loc="upper left", title="source")
    ax.add_artist(leg)
    ax.legend(handles=style_handles, loc="lower right")
    ax.grid(True, alpha=0.3)


def _plot_fixations(ax, rows, arows, N2U):
    """Draw the fixations panel, styled like Figure 4 panel B.

    Simulation means are points with ~2-SE error bars; the analytic expectation is
    overlaid as a matched-color line, labelled in place with CurvedText.
    """
    shifts = np.array([r[0] for r in rows])
    ashifts = np.array([r[0] for r in arows])

    est_boundary = _establishment_boundary(arows, N2U)
    ax.fill_between([0, est_boundary], 0, BAND_TOP, color=BAND_NO_EST, edgecolor="None", zorder=0)
    ax.fill_between([est_boundary, FIXED_BOUNDARY], 0, BAND_TOP, color=BAND_EST, edgecolor="None", zorder=0)
    ax.fill_between([FIXED_BOUNDARY, 100], 0, BAND_TOP, color=BAND_FIX, edgecolor="None", zorder=0)

    # simulation: points with error bars
    for source in ("all", "segregating", "new"):
        y = np.array([r[1]["fixations"][source] for r in rows])
        yerr = np.array([r[2]["fixations"][source] for r in rows])
        ax.errorbar(shifts, y, yerr=yerr, marker=".", linestyle="none",
                    markersize=3, color=STYLE_COLORS[source])

    # analytic: matched-color lines, labelled over the range where each has separated
    curves = _analytic_curves(arows, "fixations", N2U)
    if curves is not None:
        seg, new, total = curves["segregating"], curves["new"], curves["all"]
        ax.plot(ashifts, seg, color=SEG_COLOR)
        ax.plot(ashifts, new, color=NEW_COLOR)
        ax.plot(ashifts, total, color=TOTAL_COLOR)

        seg_m = ashifts > 53
        CurvedText(x=ashifts[seg_m] + 0.25, y=seg[seg_m] + 0.005, text="Segregating",
                   va="bottom", color=SEG_COLOR, alpha=1, axes=ax, fontsize=INTERNAL_FONTSIZE)
        new_m = ashifts > 70
        CurvedText(x=ashifts[new_m], y=new[new_m] - 0.01, text="New",
                   va="top", color=NEW_COLOR, alpha=1, axes=ax, fontsize=INTERNAL_FONTSIZE)
        tot_m = ashifts > 65
        CurvedText(x=ashifts[tot_m], y=total[tot_m] + 0.01, text="Total",
                   va="bottom", color=TOTAL_COLOR, alpha=1, axes=ax, fontsize=INTERNAL_FONTSIZE)

    # no title; the two curve types are distinguished by the legend
    analytic_handle = Line2D([], [], color="k", linestyle="-")
    mean_handle = ax.errorbar([np.nan], [np.nan], yerr=[np.nan], marker=".",
                              markersize=3, linestyle="none", color="k")
    ax.legend([analytic_handle, mean_handle],
              ["Analytic prediction", r"Mean $\pm$ 2SEs"],
              loc="upper left", fontsize=EXTERNAL_FONTSIZE,
              framealpha=1, facecolor="white", edgecolor="black")

    ax.set_xlim([0, 100])
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels([0, 50, 100], fontsize=EXTERNAL_FONTSIZE)
    ax.set_xlabel(r"Shift size ($\Lambda$)", fontsize=EXTERNAL_FONTSIZE, labelpad=1)
    if N2U == 0.02:
        ax.set_ylim([0, 0.42])
        ax.set_yticks([0, 0.2, 0.4])
        ax.set_yticklabels(["0", "0.2", "0.4"], fontsize=EXTERNAL_FONTSIZE)
    else:
        ax.set_ylim([0, 0.22])
        ax.set_yticks([0, 0.1, 0.2])
        ax.set_yticklabels(["0", "0.1", "0.2"], fontsize=EXTERNAL_FONTSIZE)
    ax.set_ylabel("Number of fixations", fontsize=EXTERNAL_FONTSIZE, labelpad=0)
    ax.set_box_aspect(1)


def main(snakemake):
    rows = sorted((summarize(p) for p in snakemake.input.simulations), key=lambda r: r[0])
    params = rows[0][3]
    arows = sorted((summarize_analytic(p) for p in snakemake.input.analytic), key=lambda r: r[0])
    N2U = snakemake.params.N2U

    # one figure per metric; outputs may be named or positional
    out = snakemake.output
    out_path = {
        "fixations": getattr(out, "fixations", None) or out[0],
        "contribution": getattr(out, "contribution", None) or out[1],
    }

    # fixations: the headline Figure S7
    fig, ax = plt.subplots(figsize=(4, 4), dpi=400)
    _plot_fixations(ax, rows, arows, N2U)
    fig.tight_layout()
    make_figure_set_width(fig, out_path["fixations"], target_width_inches=4, dpi=400)
    plt.close(fig)

    # contribution: secondary output, kept in the generic diagnostic style
    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_panel(ax, rows, arows, "contribution", params, N2U)
    fig.tight_layout()
    fig.savefig(out_path["contribution"], dpi=150)
    plt.close(fig)


if "snakemake" in globals():
    main(snakemake)  # noqa: F821
