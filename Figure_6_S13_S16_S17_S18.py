"""
Snakemake script: the phase-space contour figure -- Figure 6 and its supplementary
variants S13, S16, S17, S18.

All of these are built from the same panel-drawing primitives; a figure is specified by
a (paramset, mode, smoothed, layout) choice:

    * mode      -- "fixations" (fixation probability, 0..1) or "adaptation"
                   (long-term adaptive contribution, 0..100)
    * smoothed  -- filled contours of a Gaussian-smoothed field (True) or the raw grid
                   as a pcolormesh (False)  [ignored by the "smoothed_raw" layout]
    * paramset  -- which data pickle (fixed parameters / effect-size distribution)
    * layout    -- "standard": 2x2, top row 'Dynamic' parameters (A background
                   variance, B shift size), bottom row 'Intuitive' parameters
                   (C total variance vs proportion p, D shift/sqrt(V_A) vs variance).
                   "smoothed_raw": smoothed A/B on top, the same A/B raw below.

The manuscript assignment used by the Snakefile is:

    Figure 6  : default params, fixations,  smoothed, standard
    Fig. S13  : default params, fixations,  raw,      standard
    Fig. S16  : default params, adaptation, smoothed, standard
    Fig. S17  : alt params,     fixations,  smoothed, standard
    Fig. S18  : altsdist params, fixations,           smoothed_raw

Panels sit in rounded background boxes: the left column is "Genetic parameters"
(navy), the right "Ecological parameters" (orange). A shared horizontal colourbar
sits underneath, marked in red at the 10%-fixation (or 20%-adaptation) level.

Input
    data : the phase-space pickle for one parameter set,
           {"params": {...}, "A"/"B"/"C"/"D": {"x", "y", "fixation", "adaptation"}}
Output
    figure : the assembled PNG

Run as a Snakemake script or directly:
    python Figure_6_S13_S16_S17_S18.py <data.pkl> <out.png> [mode] [smoothed] [layout]
        mode     = fixations | adaptation      (default fixations)
        smoothed = smoothed | raw              (default smoothed)
        layout   = standard | smoothed_raw     (default standard)
"""

import os
import sys
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import ndimage

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass

from common_functions import make_figure_set_width

FONTSIZE = 10

COLORBAR_WIDTH_FRAC = 0.6      # colourbar width, as a fraction of the canvas

# Vertical breathing room, in units of text_pad (one eighth of a line of text). The
# panels are kept square by the height solver in make_figure, so raising either of
# these makes the canvas taller rather than squashing the panels.
ROW_GAP_PADS = 4.
TOP_MARGIN_PADS = 4.

# The red reference marks at style["red_level"]. The contour is drawn on contoured
# panels; the cell-edge staircase (_threshold_staircase) is its raw-panel equivalent.
# The colourbar's red tick appears only if some panel shows the level.
DRAW_RED_CONTOUR   = True
DRAW_RED_STAIRCASE = False

# True snaps the raw pcolormesh squares to the colourbar's bins, so they can be read
# off it like a filled contour; False uses the continuous vmin..vmax ramp.
BIN_RAW_SQUARES_TO_COLOURBAR = True

# Panel D's out-of-scope wedge (absolute shift > sqrt(V_S)) holds cells that are never
# simulated and land in the pickle as 0. FILL_OUT_OF_SCOPE_CELLS interpolates them
# before contouring; GREEN_OUT_OF_SCOPE_SQUARES paints them green on the raw panels,
# so no white "zero" squares peek out. Must match phase_space_grids.SHIFT_SCOPE_MAX.
SHIFT_SCOPE_MAX = 100.0
SCOPE_GREEN = [0.8, 0.9, 0.8]
FILL_OUT_OF_SCOPE_CELLS   = True
GREEN_OUT_OF_SCOPE_SQUARES = True

# per-mode drawing constants: contour levels, colour normalisation, the red reference
# level, and where along the colourbar that red mark falls.
MODE_STYLE = {
    "fixations": dict(
        levels=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        bounds=[0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        vmin=0, vmax=1, red_level=0.1, red_frac=2 / 11,
        field="fixation", label="Fixation probability",
    ),
    "adaptation": dict(
        levels=[1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        bounds=[0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        vmin=1, vmax=100, red_level=20, red_frac=3 / 11,
        field="adaptation", label="Long-term contribution to adaptation",
    ),
}


def _cell_edges(v):
    """Cell boundaries for coordinates ``v``, which hold the simulated points at the
    cell centres: each edge sits halfway to the next simulation, and the outer edges
    extend half a step past the end points. Halved in log space when the points are
    log-spaced, so a square looks centred on its simulation as plotted."""
    v = np.asarray(v, dtype=float)
    if v.size == 1:                       # degenerate axis: fall back to a unit cell
        return np.array([v[0] - 0.5, v[0] + 0.5])

    # log-spaced if the point-to-point RATIOS are more uniform than the differences
    log_spaced = (v > 0).all()
    if log_spaced:
        d_lin, d_log = np.diff(v), np.diff(np.log(v))
        spread = lambda d: d.std() / abs(d.mean()) if d.mean() else np.inf
        log_spaced = spread(d_log) < spread(d_lin)

    u = np.log(v) if log_spaced else v
    edges = np.empty(u.size + 1)
    edges[1:-1] = (u[:-1] + u[1:]) / 2                  # halfway to each neighbour
    edges[0]    = u[0]  - (u[1]  - u[0])  / 2           # half a step past the ends
    edges[-1]   = u[-1] + (u[-1] - u[-2]) / 2
    return np.exp(edges) if log_spaced else edges


def _threshold_staircase(xe, ye, Z, level):
    """Cell-edge segments separating cells at or above ``level`` from cells below it.

    Unlike a contour, which interpolates between cell centres, this traces the actual
    boundaries of the drawn squares. ``xe``/``ye`` are the edges from _cell_edges();
    ``Z`` is (len(ye)-1, len(xe)-1). Only interior transitions are traced."""
    xe, ye = np.asarray(xe, float), np.asarray(ye, float)
    hot = np.asarray(Z, dtype=float) >= level
    segs = []

    # vertical edges: neighbours differ across a column boundary
    j, i = np.nonzero(hot[:, :-1] != hot[:, 1:])
    segs += [[(xe[c + 1], ye[r]), (xe[c + 1], ye[r + 1])] for r, c in zip(j, i)]

    # horizontal edges: neighbours differ across a row boundary
    j, i = np.nonzero(hot[:-1, :] != hot[1:, :])
    segs += [[(xe[c], ye[r + 1]), (xe[c + 1], ye[r + 1])] for r, c in zip(j, i)]
    return segs


def _out_of_scope_mask(x, y):
    """Panel D's out-of-scope cells: absolute shift = y * sqrt(V_A) above sqrt(V_S).

    Re-derived from the plotted grid rather than read from the pickle, so it matches
    the green scope patch drawn by _cosmetic_D exactly."""
    X, Y = np.meshgrid(np.asarray(x, float), np.asarray(y, float))
    return Y * np.sqrt(X) > SHIFT_SCOPE_MAX


def _fill_out_of_scope(Z, mask):
    """Give each unsimulated (masked) cell the mean of its left and below neighbours.

    Swept in increasing x and y so a filled value feeds the next cell along, carrying
    the field into the wedge instead of letting it fall off a cliff to 0. A masked cell
    with neither neighbour available keeps its 0."""
    Z = np.array(Z, dtype=float, copy=True)
    ny, nx = Z.shape
    for j in range(ny):
        for i in range(nx):
            if not mask[j, i]:
                continue
            neighbours = []
            if i > 0:
                neighbours.append(Z[j, i - 1])          # left
            if j > 0:
                neighbours.append(Z[j - 1, i])          # below
            if neighbours:
                Z[j, i] = float(np.mean(neighbours))
    return Z


def draw_panel(ax, grid, mode, raw, panel=None):
    """Draw one panel's contour field on ``ax``."""
    style = MODE_STYLE[mode]
    x, y = grid["x"], grid["y"]
    Z = np.asarray(grid[style["field"]], dtype=float)   # shape (len(y), len(x))
    cmap = mpl.colormaps["Greys"]
    norm = mpl.colors.Normalize(vmin=style["vmin"], vmax=style["vmax"])

    # panel D only: the never-simulated wedge above the scope cutoff
    scope_mask = _out_of_scope_mask(x, y) if panel == "D" else None
    if scope_mask is not None and FILL_OUT_OF_SCOPE_CELLS and scope_mask.any():
        Z = _fill_out_of_scope(Z, scope_mask)

    # light Gaussian smoothing (sigma is [y, x]), mostly to round the contour corners
    Z2 = ndimage.gaussian_filter(Z, sigma=[1.5, 1], order=0, mode="nearest", truncate=1)

    if raw:
        # explicit edges + shading="flat": one square per simulation (see _cell_edges)
        xe, ye = _cell_edges(x), _cell_edges(y)
        square_norm = (mpl.colors.BoundaryNorm(style["bounds"], cmap.N)
                       if BIN_RAW_SQUARES_TO_COLOURBAR else norm)
        ax.pcolormesh(xe, ye, Z, cmap=cmap, shading="flat", norm=square_norm)
        # unpainted out-of-scope squares would read as white (zero fixation) cells
        if scope_mask is not None and GREEN_OUT_OF_SCOPE_SQUARES and scope_mask.any():
            ax.pcolormesh(xe, ye, np.where(scope_mask, 1.0, np.nan), shading="flat",
                          cmap=mpl.colors.ListedColormap([SCOPE_GREEN]), vmin=0, vmax=1)
        # red staircase along the edges of the cells that cross the reference level
        if DRAW_RED_STAIRCASE:
            ax.add_collection(mpl.collections.LineCollection(
                _threshold_staircase(xe, ye, Z, style["red_level"]),
                colors="r", linewidths=2, zorder=5))
    else:
        ax.contourf(x, y, Z2, style["levels"], cmap=cmap, norm=norm, alpha=1)
    # a single red reference contour on the contoured panels; raw panels get the
    # cell-edge staircase above instead
    if DRAW_RED_CONTOUR and not raw:
        ax.contour(x, y, Z2, [style["red_level"]], colors="r", linewidths=2)


# ---- per-panel axis cosmetics (limits / ticks / labels) ---------- #
def _cosmetic_A(ax, raw=False):
    ax.set_xscale("log")
    ax.set_xlim([1e-3, 1e3]); ax.set_ylim([0, 250])
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1, 10, 100, 1000])
    ax.set_xticklabels(["$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$10^{0}$",
                        "$10^{1}$", "$10^{2}$", "$10^{3}$"])
    ax.set_yticks([0, 50, 100, 150, 200, 250])
    ax.set_ylabel("Background variance", fontsize=FONTSIZE, labelpad=0)
    ax.set_xlabel("Mutational input", fontsize=FONTSIZE, labelpad=0)


def _cosmetic_B(ax, raw=False):
    ax.set_xscale("log")
    ax.set_xlim([1e-3, 1e3]); ax.set_ylim([0, 100])
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1, 10, 100, 1000])
    ax.set_xticklabels(["$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$10^{0}$",
                        "$10^{1}$", "$10^{2}$", "$10^{3}$"])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("Shift size", fontsize=FONTSIZE, labelpad=0)
    ax.set_xlabel("Mutational input", fontsize=FONTSIZE, labelpad=0)


def _cosmetic_C(ax, raw=False):
    ax.set_xlim([0, 1]); ax.set_ylim([1, 1500])
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 200, 400, 600, 800, 1000, 1200, 1400])
    ax.set_xlabel("Proportion of variance\nfrom large-effect alleles", fontsize=FONTSIZE, labelpad=0)
    ax.set_ylabel("Total phenotypic variance", fontsize=FONTSIZE, labelpad=0)


def _cosmetic_D(ax, raw=False):
    # shift larger than sqrt(V_S) is outside the model's scope; on a raw panel those
    # cells are painted green individually instead (GREEN_OUT_OF_SCOPE_SQUARES)
    if not (raw and GREEN_OUT_OF_SCOPE_SQUARES):
        xv = np.linspace(1, 1000, 1000)
        ax.fill_between(xv, SHIFT_SCOPE_MAX / np.sqrt(xv), 100, color=SCOPE_GREEN,
                        alpha=1, zorder=100)
    ax.text(200, 9, r"$\Lambda > \sqrt{V_S}$", fontsize=FONTSIZE, color="darkgreen",
            ha="center", zorder=101)
    ax.set_xlim([0, 300]); ax.set_ylim([0, 10])
    ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_xlabel("Total phenotypic variance", fontsize=FONTSIZE, labelpad=0)
    ax.set_ylabel(r"Shift size in units of $\sqrt{V_A}$", fontsize=FONTSIZE, labelpad=0)


# A slot is one of the four grid positions: which data panel to draw, its axis
# cosmetics, whether to force raw, and its bold letter.
def _slots(layout, raw):
    """Return the four panel slots and the two row labels for a layout."""
    if layout == "smoothed_raw":
        slots = [
            dict(key="A", cosmetic=_cosmetic_A, raw=False, title="A"),
            dict(key="B", cosmetic=_cosmetic_B, raw=False, title="B"),
            dict(key="A", cosmetic=_cosmetic_A, raw=True,  title="C"),
            dict(key="B", cosmetic=_cosmetic_B, raw=True,  title="D"),
        ]
        return slots, ("Smoothed contours", "Raw data")
    slots = [
        dict(key="A", cosmetic=_cosmetic_A, raw=raw, title="A"),
        dict(key="B", cosmetic=_cosmetic_B, raw=raw, title="B"),
        dict(key="C", cosmetic=_cosmetic_C, raw=raw, title="C"),
        dict(key="D", cosmetic=_cosmetic_D, raw=raw, title="D"),
    ]
    return slots, ("'Dynamic' parameters", "'Intuitive' parameters")


def make_contour_plot_figure(data, mode="fixations", raw=False, layout="standard", height_in=6.5):
    """Assemble the four-panel phase-space figure; returns (fig, top-left axis)."""
    style = MODE_STYLE[mode]
    slots, (row_label_top, row_label_bottom) = _slots(layout, raw)
    plt.rcParams["font.size"] = FONTSIZE
    plt.rcParams["axes.labelsize"] = FONTSIZE
    plt.rcParams["axes.titlesize"] = FONTSIZE
    plt.rcParams["xtick.labelsize"] = FONTSIZE
    plt.rcParams["ytick.labelsize"] = FONTSIZE
    cmap = mpl.colormaps["Greys"]
    facecolor = [0.92, 0.95, 1]

    fig = plt.figure(figsize=(6.5, height_in), dpi=400)

    # measure the height of a line of text (in figure fraction) to use as a padding unit
    z = fig.add_axes(rect=[0, 0, 1, 1]); z.set_xticks([]); z.set_yticks([])
    r = fig.canvas.get_renderer()
    t = z.text(0.5, 0.5, "Figure 6", fontsize=FONTSIZE, ha="center", va="center")
    text_pad = t.get_window_extent(renderer=r).height / fig.dpi / 8
    t.remove(); z.remove()

    space_between_axes = 0.02
    space_between_colorbar = 0.1
    horizontal_space_between_axes = 0.04
    current_y = 0.2 / 8 + 3 * text_pad + space_between_colorbar
    max_y = 1 - TOP_MARGIN_PADS * text_pad
    remaining_height = max_y - current_y
    min_x = 8 * text_pad
    height = (remaining_height - ROW_GAP_PADS * text_pad - space_between_axes) / 2
    dx = height + horizontal_space_between_axes + text_pad * 4   # column-to-column offset

    # bottom-left, bottom-right, then top-left, top-right
    c = fig.add_axes(rect=[min_x, current_y, height, height])
    d = fig.add_axes(rect=[min_x + dx, current_y, height, height])
    current_y += height + space_between_axes + ROW_GAP_PADS * text_pad
    a = fig.add_axes(rect=[min_x, current_y, height, height])
    b = fig.add_axes(rect=[min_x + dx, current_y, height, height])

    # the shared colourbar; it is centred at the end, once the column boxes are placed
    colorbar_width = COLORBAR_WIDTH_FRAC
    colorbar = fig.add_axes(rect=[min_x, 3 * text_pad, colorbar_width, 0.2 / 8])

    for ax, slot in zip([a, b, c, d], slots):
        draw_panel(ax, data[slot["key"]], mode, slot["raw"], panel=slot["key"])
        slot["cosmetic"](ax, slot["raw"])
        ax.set_title("%s." % slot["title"], fontsize=FONTSIZE, fontweight="bold", loc="left")

    # ---- shared horizontal colourbar ---------------------------------------- #
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("Custom cmap", cmaplist, cmap.N)
    norm = mpl.colors.BoundaryNorm(style["bounds"], cmap.N)
    cb = mpl.colorbar.ColorbarBase(ax=colorbar, cmap=cmap, norm=norm,
                                   ticks=style["bounds"], boundaries=style["bounds"],
                                   format="%g", orientation="horizontal")
    cb.set_label(style["label"], size=FONTSIZE)
    cb.set_ticks(style["bounds"]); cb.set_ticklabels(style["bounds"])
    cb.ax.tick_params(labelsize=FONTSIZE)
    # mark the red reference level only if some panel shows it, so an all-raw figure
    # gets no orphaned tick
    shows_red = any((DRAW_RED_STAIRCASE if s["raw"] else DRAW_RED_CONTOUR) for s in slots)
    if shows_red:
        colorbar.plot([style["red_frac"]] * 2, [0, 1], color="r", transform=colorbar.transAxes)

    # ---- rounded background boxes + column/row labels ----------------------- #
    vertical_lower_y = 0.2 / 8 + 1 * text_pad + space_between_colorbar / 2
    vertical_upper_y = 1
    vertical_height = vertical_upper_y - vertical_lower_y
    vertical_left_x = min_x - text_pad * 5 - 0.01
    vertical_left_width = min_x - text_pad * 3 + height + horizontal_space_between_axes / 2 + text_pad / 2
    vertical_right_x = vertical_left_width + vertical_left_x
    vertical_right_max_x = min_x + dx + height + text_pad * 2
    vertical_right_width = vertical_right_max_x - vertical_right_x

    horizontal_lower_y_min = vertical_lower_y
    horizontal_lower_y_max = (vertical_lower_y + height + text_pad * (ROW_GAP_PADS + 3)
                              + space_between_axes / 2 - 0.005)
    horizontal_upper_y_min = horizontal_lower_y_max
    horizontal_upper_y_max = 1 - 2 * text_pad
    box = dict(boxstyle="Round, pad=0, rounding_size=0.05", mutation_aspect=1,
               edgecolor="None", clip_on=False, transform=fig.transFigure, figure=fig, zorder=-1)
    fig.patches.extend([
        FancyBboxPatch((0, horizontal_lower_y_min), 0.3, horizontal_lower_y_max - horizontal_lower_y_min,
                       facecolor=[0.55, 0.55, 0.55], **box),
        FancyBboxPatch((0, horizontal_upper_y_min), 0.3, horizontal_upper_y_max - horizontal_upper_y_min,
                       facecolor=[0.8, 0.8, 0.8], **box),
        FancyBboxPatch((vertical_left_x, vertical_lower_y), vertical_left_width, vertical_height,
                       facecolor=facecolor, **box),
        FancyBboxPatch((vertical_right_x, vertical_lower_y), vertical_right_width, vertical_height,
                       facecolor=[1, 0.95, 0.9], **box),
    ])
    fig.text(vertical_left_x + vertical_left_width / 2, (vertical_upper_y + horizontal_upper_y_max) / 2,
             "Genetic parameters", weight="bold", fontsize=FONTSIZE, ha="center", va="center", color="navy")
    fig.text(vertical_right_x + vertical_right_width / 2, (vertical_upper_y + horizontal_upper_y_max) / 2,
             "Ecological parameters", weight="bold", fontsize=FONTSIZE, ha="center", va="center", color="darkorange")
    fig.text(vertical_left_x / 2, (horizontal_lower_y_min + horizontal_lower_y_max) / 2,
             row_label_bottom, weight="bold", fontsize=FONTSIZE, ha="center", va="center", rotation=90, color="k")
    fig.text(vertical_left_x / 2, (horizontal_upper_y_min + horizontal_upper_y_max) / 2,
             row_label_top, weight="bold", fontsize=FONTSIZE, ha="center", va="center", rotation=90, color="k")

    # centre the colourbar on the two column boxes, not on the panel block, which sits
    # slightly right of them
    pos = colorbar.get_position()
    colorbar.set_position([(vertical_left_x + vertical_right_max_x - pos.width) / 2,
                           pos.y0, pos.width, pos.height])
    return fig, a


def make_figure(data, out_path, mode="fixations", smoothed=True, layout="standard"):
    """Build the figure, iterating the canvas height until the top-left panel is square, then save."""
    raw = not smoothed
    height = 6.5
    for _ in range(12):
        fig, ax = make_contour_plot_figure(data, mode=mode, raw=raw, layout=layout, height_in=height)
        bbox = ax.get_window_extent()
        w_in, h_in = bbox.width / fig.dpi, bbox.height / fig.dpi
        if np.isclose(w_in, h_in, rtol=0.01):
            break
        height = height * w_in / h_in
        plt.close(fig)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    # the panels are sized out of the leftover vertical space, so rescaling to the
    # manuscript's 6.5" width restores them and turns the padding into a taller figure
    make_figure_set_width(fig, out_path, target_width_inches=6.5, dpi=400)
    plt.close(fig)


def load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def main(data_pickle, out_path, mode="fixations", smoothed=True, layout="standard"):
    make_figure(load(data_pickle), out_path, mode=mode, smoothed=smoothed, layout=layout)


if "snakemake" in globals():
    inp = snakemake.input       # noqa: F821
    out = snakemake.output      # noqa: F821
    par = snakemake.params      # noqa: F821
    data = getattr(inp, "data", None) or inp[0]
    main(data, getattr(out, "figure", None) or out[0],
         mode=getattr(par, "mode", "fixations"),
         smoothed=getattr(par, "smoothed", True),
         layout=getattr(par, "layout", "standard"))
elif __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    default_in = os.path.join(here, "results", "phase_space", "paramsetdefault.pkl")
    default_out = os.path.join(here, "results", "plots", "Figure_6.png")
    data_pickle = sys.argv[1] if len(sys.argv) > 1 else default_in
    out_path = sys.argv[2] if len(sys.argv) > 2 else default_out
    mode = sys.argv[3] if len(sys.argv) > 3 else "fixations"
    smoothed = (sys.argv[4].lower() != "raw") if len(sys.argv) > 4 else True
    layout = sys.argv[5] if len(sys.argv) > 5 else "standard"
    main(data_pickle, out_path, mode=mode, smoothed=smoothed, layout=layout)
