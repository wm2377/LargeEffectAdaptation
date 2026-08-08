"""Supplementary Figure S1: large-effect fixations, individual-based vs all-allele.

"""

import argparse
import glob
import os
import pickle
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIGMA2 = 4
SHIFTS = [50, 80]
LARGE_EFFECT_MIN_A = 10.0     # |a| at or above which a fixation counts as large-effect
FONTSIZE = 11

COLOUR = {80: "brown", 50: "forestgreen"}


def _n2u_from_path(path):
    m = re.search(r"LargeN2U_([0-9.eE+-]+)", path)
    return float(m.group(1)) if m else None


def load_individual(root, placeholders=None):
    """{shift: {2NU: (mean, ste)}} from the per-cell summaries."""
    if placeholders is None:
        placeholders = []
    out = {s: {} for s in SHIFTS}
    pattern = os.path.join(root, f"sigma2_{SIGMA2}", "LargeN2U_*", "shift_*",
                           "individual_summary.pkl")
    for path in sorted(glob.glob(pattern)):
        with open(path, "rb") as fh:
            s = pickle.load(fh)
        if s["n_replicates"]:
            out[int(s["shift"])][float(s["large_2NU"])] = (s["mean"], s["ste"])
            if s.get("placeholder"):
                placeholders.append((s["shift"], s["large_2NU"]))
    return out


def load_all_allele(root):
    """{shift: {2NU: (mean, ste)}} of large-effect fixations per replicate.
    """
    out = {s: {} for s in SHIFTS}
    for shift in SHIFTS:
        pattern = os.path.join(root, f"sigma2_{SIGMA2}", "LargeN2U_*", f"shift_{shift}",
                               "all_processed_results_with_mutation_counts.pkl")
        for path in sorted(glob.glob(pattern)):
            with open(path, "rb") as fh:
                _pooled = pickle.load(fh)
                fixations = pickle.load(fh)
            counts = np.array([sum(1 for a in effects if a >= LARGE_EFFECT_MIN_A)
                               for effects in fixations.values()], dtype=float)
            if counts.size == 0:
                continue
            ste = counts.std(ddof=1) / np.sqrt(counts.size) if counts.size > 1 else 0.0
            out[shift][_n2u_from_path(path)] = (counts.mean(), ste)
    return out


def build(out_path, data_root, individual_root, hide_placeholder_note=False):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    placeholders = []
    individual = load_individual(individual_root, placeholders)
    all_allele = load_all_allele(data_root)
    for label, d in (("individual", individual), ("all-allele", all_allele)):
        n = sum(len(v) for v in d.values())
        print(f"  {label}: {n} points "
              f"({', '.join(f'{k}:{len(v)}' for k, v in d.items())})", flush=True)

    fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=400)
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1e3)
    ax.set_ylim(0, 3.1)
    # the low-mutational-input regime, where large-effect alleles are rare
    ax.fill_between([1e-3, 1], [0, 0], [3.1, 3.1], color="k", alpha=0.1, edgecolor="None")

    for shift in SHIFTS:
        colour = COLOUR[shift]
        pts = sorted(all_allele[shift].items())
        if pts:
            x = [p[0] for p in pts]
            y = [p[1][0] for p in pts]
            e = [p[1][1] for p in pts]
            ax.errorbar(x, y, yerr=e, fmt="o", ms=5, alpha=0.25, color=colour, ls="")
        pts = sorted(individual[shift].items())
        if pts:
            x = [p[0] for p in pts]
            y = [p[1][0] for p in pts]
            e = [p[1][1] for p in pts]
            ax.errorbar(x, y, yerr=e, fmt="+", ms=8, mew=1.8, color=colour, ls="")

    ax.set_xlabel(r"Mutational input of large effect alleles ($2NU$)", fontsize=FONTSIZE)
    ax.set_ylabel("Large effect fixations", fontsize=FONTSIZE)
    ax.tick_params(labelsize=FONTSIZE - 1)
    ax.text(0.2, 1.85, r"$\Lambda = 80$", color=COLOUR[80], fontsize=FONTSIZE, ha="center")
    ax.text(0.2, 0.80, r"$\Lambda = 50$", color=COLOUR[50], fontsize=FONTSIZE, ha="center")
    for (series, label), dy in (((all_allele, "All allele"), 0.30),
                                ((individual, "Individual"), -0.22)):
        pts = sorted(series[50].items())
        if not pts:
            continue
        x, (y, _e) = pts[-1]
        ax.annotate(label, xy=(x, y), xytext=(x / 9, y + dy),
                    ha="right", va="center", fontsize=FONTSIZE,
                    arrowprops=dict(arrowstyle="->", color="k", lw=1,
                                    connectionstyle="arc3,rad=0"))

    # A figure resting partly on placeholder cells must say so on its face.
    if placeholders and not hide_placeholder_note:
        ax.text(0.98, 0.02, f"PLACEHOLDER data in {len(placeholders)} cells",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=FONTSIZE - 3, color="firebrick", weight="bold")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=400)
    plt.close(fig)
    if placeholders:
        print(f"  WARNING: {len(placeholders)} individual cell(s) are PLACEHOLDERS, not "
              f"simulated: {sorted({p[1] for p in placeholders})}", flush=True)
    print(f"wrote {out_path}", flush=True)
    return out_path


if "snakemake" in globals():
    build(snakemake.output[0],                                       # noqa: F821
          snakemake.params.data_root,                                # noqa: F821
          snakemake.params.individual_root)                          # noqa: F821
elif __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    base = os.path.dirname(here)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_path")
    ap.add_argument("--data-root", default=os.path.join(base, "data"))
    ap.add_argument("--individual-root", default=os.path.join(base, "data_individual"))
    ap.add_argument("--hide-placeholder-note", action="store_true")
    args = ap.parse_args()
    build(args.out_path, args.data_root, args.individual_root,
          hide_placeholder_note=args.hide_placeholder_note)
