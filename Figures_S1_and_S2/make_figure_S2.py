"""Supplementary Figure S2

Four panels, split by mutational input (top row 2NU >= 1, bottom row 2NU < 1):
    A, C  relative additive variance of the SMALL-effect alleles, V_A(t) / sigma^2 --
          how the large-effect alleles perturb the background as the population adapts
    B, D  relative phenotypic skew of the small-effect alleles, u_3(t) / 2 sigma^2

"""

import argparse
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D

AGGREGATE_NAME = "all_processed_results_with_mutation_counts.pkl"
SKEW_NAME = "skew_results.pkl"

SIGMA2_VALUES = [4, 40, 80]
SHIFT_VALUES = [50, 80]
LARGE_2NU_VALUES = [float(x) for x in np.logspace(-3, 3, 13)[:-1]]
N_GENERATIONS = 20000
FONTSIZE = 11

# S = a^2 bins, matching all_allele_cpp.BIN_LIMITS: the first is the small-effect bin
# (a^2 < 100, i.e. |a| < 10), the second the large-effect bin.
SMALL_BIN, LARGE_BIN = 100, 20000


def colour_for(sigma2, shift):
    return {(80, 80): "cornflowerblue", (80, 50): "lightcoral",
            (40, 80): "purple", (40, 50): "darkorange",
            (4, 80): "brown", (4, 50): "forestgreen"}.get((sigma2, shift), "k")


def cell_dir(data_root, sigma2, large_2NU, shift):
    return os.path.join(data_root, f"sigma2_{sigma2}", f"LargeN2U_{large_2NU}",
                        f"shift_{shift}")


def load(data_root):
    """Every cell's pooled moments and skew, plus a tally of what was absent."""
    moments, skew, missing, found = {}, {}, [], 0
    for sigma2 in SIGMA2_VALUES:
        for shift in SHIFT_VALUES:
            for n2u in LARGE_2NU_VALUES:
                d = cell_dir(data_root, sigma2, n2u, shift)
                agg, skw = os.path.join(d, AGGREGATE_NAME), os.path.join(d, SKEW_NAME)
                if os.path.exists(agg):
                    with open(agg, "rb") as fh:
                        moments[(sigma2, shift, n2u)] = (pickle.load(fh), pickle.load(fh))
                    found += 1
                else:
                    missing.append(agg)
                if os.path.exists(skw):
                    with open(skw, "rb") as fh:
                        skew[(sigma2, shift, n2u)] = pickle.load(fh)
                else:
                    missing.append(skw)
    return moments, skew, missing, found


def _mean_and_error(sum_, sum_sq, count, scale):
    """Across-replicate mean and 2 standard errors, NaN where no replicate contributed."""
    with np.errstate(invalid="ignore", divide="ignore"):
        safe = np.maximum(count, 1)
        mean = np.where(count > 0, sum_ / safe, np.nan)
        var = np.maximum(sum_sq / safe - mean ** 2, 0)
        err = np.where(count > 0, 2 * np.sqrt(var / safe), np.nan)
    return mean / scale, err / scale


def series_for(moments, sigma2, shift, group, bin_value=SMALL_BIN):
    """Pool every 2NU in `group` for one (sigma2, shift); return the two trajectories.

    Both the variance and the skew come from the SAME pooled moments, so they are
    guaranteed to be the same length and to rest on the same replicates -- which matters
    now that the skew panel is normalised by the variance trajectory.

    Returns (V_A(t), V_A error, u3(t), u3 error) or None when the group has no data.
    """
    acc = {m: {k: np.zeros(N_GENERATIONS) for k in ("sum", "sum_sq", "count")}
           for m in ("variance", "skew")}
    found = False
    for (s, sh, n2u), (pooled, _fix) in moments.items():
        if (s, sh) != (sigma2, shift) or (group == "large") != (n2u >= 1):
            continue
        generations = sorted(pooled)[:N_GENERATIONS]
        if not generations:
            continue
        n = len(generations)
        for metric in ("variance", "skew"):
            for key in ("sum", "sum_sq", "count"):
                acc[metric][key][:n] += np.array(
                    [pooled[g][bin_value][metric][key] for g in generations])
        found = True
    if not found:
        return None

    out = []
    for metric in ("variance", "skew"):
        a = acc[metric]
        with np.errstate(invalid="ignore", divide="ignore"):
            safe = np.maximum(a["count"], 1)
            mean = np.where(a["count"] > 0, a["sum"] / safe, np.nan)
            var = np.maximum(a["sum_sq"] / safe - mean ** 2, 0)
            err = np.where(a["count"] > 0, 2 * np.sqrt(var / safe), np.nan)
        out += [mean, err]
    return tuple(out)


def _draw(ax, mean, err, colour):
    t = np.arange(mean.size)
    ax.fill_between(t, mean - err, mean + err, alpha=0.2, color=colour)
    ax.plot(t, mean, color=colour)


def draw_panels(ax_var, ax_skew, moments, group, bin_value=SMALL_BIN):
    """One row: relative variance on the left, skew relative to the variance on the right."""
    drawn = 0
    for sigma2 in SIGMA2_VALUES:
        for shift in SHIFT_VALUES:
            series = series_for(moments, sigma2, shift, group, bin_value)
            if series is None:
                continue
            v_mean, v_err, s_mean, s_err = series
            colour = colour_for(sigma2, shift)

            # LEFT: normalised by the mean variance at t = 0, i.e. the first generation
            # after burn-in, so every curve starts at 1 and the panel shows the relative
            # excursion rather than the absolute level.
            v0 = v_mean[0]
            if np.isfinite(v0) and v0 > 0:
                _draw(ax_var, v_mean / v0, v_err / v0, colour)

            # RIGHT: skew relative to twice the CONTEMPORANEOUS variance, u3(t) / 2 V_A(t)
            with np.errstate(invalid="ignore", divide="ignore"):
                denom = 2 * v_mean
                ok = np.isfinite(denom) & (denom > 0)
                rel = np.where(ok, s_mean / np.where(ok, denom, 1), np.nan)
                rel_err = np.where(ok, s_err / np.where(ok, denom, 1), np.nan)
            _draw(ax_skew, rel, rel_err, colour)
            drawn += 1
    return drawn


def add_label_boxes(fig, axes, row_labels):
    """Rounded column/row boxes behind the grid
    """
    ax1, ax2, ax3, ax4 = axes

    # tight bboxes include tick labels / axis labels / titles, but only after a draw
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    def tight(ax):
        return ax.get_tightbbox(renderer).transformed(inv)

    margin = 0.008            # breathing room between the labels and the box edge
    header_h = 0.034          # strip inside the box, above the panels, for the title

    columns = ((ax1, ax3), (ax2, ax4))
    extents = []
    for col_top, col_bottom in columns:
        b_top, b_bottom = tight(col_top), tight(col_bottom)
        extents.append(dict(x0=min(b_top.x0, b_bottom.x0) - margin,
                            x1=max(b_top.x1, b_bottom.x1) + margin,
                            y_bottom=b_bottom.y0 - margin,
                            y_top=b_top.y1 + margin))

    # both boxes share their vertical edges, so the two columns line up
    box_bottom = min(e["y_bottom"] for e in extents)
    box_top = max(e["y_top"] for e in extents) + header_h
    # the row boxes meet in the gap between the rows
    split_y = (min(tight(ax).y0 for ax in (ax1, ax2)) - margin
               + max(tight(ax).y1 for ax in (ax3, ax4)) + margin) / 2

    style = dict(boxstyle="Round, pad=0, rounding_size=0.02", mutation_aspect=1,
                 clip_on=False, transform=fig.transFigure, figure=fig, zorder=-1)

    for extent, face, edge, text in zip(
            extents,
            ([0.86, 0.92, 1.0], [0.93, 0.85, 0.74]),
            ("tab:blue", "saddlebrown"),
            ("Phenotypic variance", "Phenotypic skew")):
        x0, width = extent["x0"], extent["x1"] - extent["x0"]
        fig.patches.append(FancyBboxPatch((x0, box_bottom), width, box_top - box_bottom,
                                          facecolor=face, edgecolor=edge, lw=2, **style))
        # title inside the box, centred in the header strip
        fig.text(x0 + width / 2, box_top - header_h / 2, text, ha="center", va="center",
                 fontsize=FONTSIZE, weight="bold", color=edge)

    # Row boxes: same vertical extent as the column boxes, split at the inter-row gap, and
    # butted up against the left column box (whose left edge is now measured, not assumed).
    row_w = 0.042
    row_x = extents[0]["x0"] - row_w
    for text, (y0, y1), face in zip(row_labels,
                                    ((split_y, box_top), (box_bottom, split_y)),
                                    ([0.70] * 3, [0.85] * 3)):
        fig.patches.append(FancyBboxPatch((row_x, y0), row_w, y1 - y0,
                                          facecolor=face, edgecolor="0.35", lw=1.5, **style))
        fig.text(row_x + row_w / 2, (y0 + y1) / 2, text, ha="center", va="center",
                 rotation=90, fontsize=FONTSIZE, weight="bold", color="k")

    # the full horizontal extent of the drawn block, for centring the legend beneath it
    return row_x, extents[-1]["x1"]


def add_legend(fig, axes, content_extent=None):
    """Colour key drawn as a GRID -- a column per sigma^2, a row per shift size.

    """
    ax1, ax2, ax3, ax4 = axes

    swatch_w, swatch_h = 0.052, 0.017     # short and thick, so the sleeve is legible
    cell_w = swatch_w + 0.085             # column pitch
    row_label_w = 0.105                   # room for the "Lambda = 80" labels
    row_h = 0.022
    base_y = 0.014                        # bottom of the key, in figure coords

    grid_w = row_label_w + len(SIGMA2_VALUES) * cell_w
    if content_extent is None:
        content_extent = (ax1.get_position().x0, ax2.get_position().x1)
    centre = (content_extent[0] + content_extent[1]) / 2
    grid_left = centre - grid_w / 2 + row_label_w

    shifts = SHIFT_VALUES[::-1]           # Lambda = 80 on top, as in the original
    for j, sigma2 in enumerate(SIGMA2_VALUES):
        cx = grid_left + (j + 0.5) * cell_w
        fig.text(cx, base_y + len(shifts) * row_h + 0.006, rf"$\sigma^2 = {sigma2}$",
                 ha="center", va="bottom", fontsize=FONTSIZE - 1)
    for i, shift in enumerate(shifts):
        cy = base_y + (len(shifts) - 1 - i) * row_h
        fig.text(grid_left - 0.012, cy + swatch_h / 2, rf"$\Lambda = {shift}$",
                 ha="right", va="center", fontsize=FONTSIZE - 1)
        for j, sigma2 in enumerate(SIGMA2_VALUES):
            cx = grid_left + (j + 0.5) * cell_w
            colour = colour_for(sigma2, shift)
            fig.patches.append(plt.Rectangle(
                (cx - swatch_w / 2, cy), swatch_w, swatch_h,
                facecolor=colour, edgecolor="none", alpha=0.2,
                transform=fig.transFigure, figure=fig, zorder=3))
            fig.lines.append(Line2D(
                [cx - swatch_w / 2, cx + swatch_w / 2], [cy + swatch_h / 2] * 2,
                color=colour, lw=1.8, solid_capstyle="butt",
                transform=fig.transFigure, figure=fig, zorder=4))


def build(out_path, data_root, bin_value=SMALL_BIN):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    moments, _skew, missing, found = load(data_root)
    total = len(SIGMA2_VALUES) * len(SHIFT_VALUES) * len(LARGE_2NU_VALUES)
    print(f"loaded {found}/{total} parameter combinations", flush=True)
    if missing:
        print(f"  {len(missing)} input file(s) absent, first: {missing[0]}", flush=True)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(7.0, 8.0), dpi=400)
    axes = (ax1, ax2, ax3, ax4)

    # top row = the group named first here; rows are labelled to match
    groups = [("small", r"$2NU < 1$"), ("large", r"$2NU \geq 1$")]
    drawn = {}
    drawn[id(ax1)] = drawn[id(ax2)] = draw_panels(ax1, ax2, moments, groups[0][0], bin_value)
    drawn[id(ax3)] = drawn[id(ax4)] = draw_panels(ax3, ax4, moments, groups[1][0], bin_value)

    for ax in (ax1, ax3):
        ax.set_ylabel(r"Relative contribution ($V_A(t)/V_A(0)$)",
                      fontsize=FONTSIZE - 1)
    for ax in (ax2, ax4):
        ax.set_ylabel(r"Maximum $D_{qs}$ ($\mu_3(t)/(2V_A(t))$)",
                      fontsize=FONTSIZE - 1)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlim(1, N_GENERATIONS - 10)
        ax.set_xlabel("Time since shift (generations)", fontsize=FONTSIZE - 1)
        ax.tick_params(labelsize=FONTSIZE - 1)
        ax.set_box_aspect(1)   # square panels
    for ax, label in zip(axes, "ABCD"):
        ax.set_title(r"$\bf{%s.}$" % label, fontsize=FONTSIZE, loc="left")
    for ax in axes:
        if drawn[id(ax)] == 0:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center",
                    fontsize=FONTSIZE, color="firebrick", weight="bold")

    # Equal y-range within a column, symmetric about that column's null value (1 for the
    # relative variance, 0 for the relative skew)
    for column, null in (((ax1, ax3), 1.0), ((ax2, ax4), 0.0)):
        reach = max(max(abs(ax.dataLim.ymin - null), abs(ax.dataLim.ymax - null))
                    for ax in column)
        reach *= 1.05                                  # a little headroom
        for ax in column:
            ax.set_ylim(null - reach, null + reach)

    fig.tight_layout(rect=[0.085, 0.085, 0.995, 0.935])
    # extra room between the rows, so the column titles sit clear of the panel letters
    fig.subplots_adjust(hspace=0.1)
    extent = add_label_boxes(fig, axes, [groups[0][1], groups[1][1]])
    add_legend(fig, axes, content_extent=extent)
    fig.savefig(out_path, bbox_inches="tight", dpi=400)
    plt.close(fig)

    print(f"\n==== inventory ====\nseries drawn per panel A/B/C/D: "
          f"{drawn[id(ax1)]}/{drawn[id(ax2)]}/{drawn[id(ax3)]}/{drawn[id(ax4)]}"
          f" -> {out_path}", flush=True)
    if min(drawn.values()) == 0:
        print("WARNING: a panel has no data and is stamped on the figure.", flush=True)
    return out_path


if "snakemake" in globals():
    build(snakemake.output[0],                                    # noqa: F821
          data_root=snakemake.params.data_root)                   # noqa: F821
elif __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_path")
    ap.add_argument("--data-root", default=os.path.join(os.path.dirname(here), "data"))
    ap.add_argument("--bin", choices=["small", "large"], default="small")
    args = ap.parse_args()
    build(args.out_path, args.data_root,
          bin_value=SMALL_BIN if args.bin == "small" else LARGE_BIN)
