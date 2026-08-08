"""
Shared definition of the phase-space simulation grids for Figure 6 (and S13/S16/S17/S18).

Imported by both the Snakefile and aggregate_phase_space_data.py, so the two cannot drift.

Each panel is a grid of "cells" carrying a position (x, y) and the simulation parameters
(N2U, sigma2, shift) producing it. `null` cells are out of scope and get a cheap
placeholder instead of a simulation. The four panels:

    A : x = 2NU (log, 31 in [1e-3, 1e3]),  y = background variance sigma2
        (31 linear in [0, 250] with the 0 replaced by 1); shift fixed at chosen_shiftA.
    B : x = 2NU (same 31),  y = shift (31 linear in [0, 100]); sigma2 fixed at chosen_sigma2.
    C : x = proportion p (39 values, see panel_C_p_values), y = total variance V_A
        (31 linear in [0, 500]; plus 30 more in (500, 1500] where p > 0.6). Per cell the
        simulation has N2U = p V_A / 4, sigma2 = V_A (1 - p), shift = chosen_shiftC.
    D : x = total variance V_A (31 linear in [0, 500]), y = shift in units of sqrt(V_A)
        (31 linear in [0, 10]). Per cell N2U = p V_A / 4, sigma2 = V_A (1 - p) with the
        FIXED p = chosen_p, and shift = y * sqrt(V_A). If that absolute shift exceeds
        SHIFT_SCOPE_MAX the cell is `null` (not simulated).

The counts above are the unrefined axes; GRID_REFINEMENTS interleaves extra points
without moving any of them, so at the default 1 pass every "31" reads 61.

The V_A = 0 point of panels C and D is floored to V_A = 1, as panel A floors its
background variance, since a total variance of 0 has no mutational input.
"""

import numpy as np

SHIFT_SCOPE_MAX = 100.0     # absolute shifts above this are out of scope (panel D nulls)

# Refinement passes per panel axis; each takes an n-point axis to 2n-1 points. Points
# are only ever inserted, and originals are copied bit-for-bit, so already-simulated
# cells keep their exact file names and are not rerun. 0 recovers the original grid.
GRID_REFINEMENTS = 1


def fnum(x):
    """Filename-stable numeric token (identical to the Snakefile's fnum)."""
    return str(float(format(float(x), ".12g")))


def _refine(v, log=False, passes=None):
    """Insert a point midway between each neighbouring pair, `passes` times.

    Originals are written through unchanged, never round-tripped through log/exp, so
    the pickle names fnum() builds from them do not shift."""
    v = np.asarray(v, dtype=float)
    for _ in range(GRID_REFINEMENTS if passes is None else passes):
        if v.size < 2:
            break
        u = np.log(v) if log else v
        mid = (u[:-1] + u[1:]) / 2.0
        out = np.empty(2 * v.size - 1, dtype=float)
        out[0::2] = v                                  # originals, untouched
        out[1::2] = np.exp(mid) if log else mid
        v = out
    return v


# --------------------------------------------------------------------------- #
# Panel axis grids
# --------------------------------------------------------------------------- #
def panel_A_axes():
    n2u = _refine(np.logspace(-3, 3, 31), log=True)
    sigma2 = _refine(np.linspace(0, 250, 31)).copy()
    sigma2[0] = 1.0                       # never use background variance 0; use 1
    return n2u, sigma2


def panel_B_axes():
    return _refine(np.logspace(-3, 3, 31), log=True), _refine(np.linspace(0, 100, 31))


def panel_C_p_values():
    # dense in the interior, log-refined toward p -> 0 and p -> 1. Each of the three
    # stretches is refined in its own space (the tails geometrically in p / 1-p), so the
    # midpoints follow the spacing the stretch was built with.
    return np.append(
        np.append(_refine(np.logspace(-3, np.log10(1 / 30), 5), log=True),
                  _refine(np.linspace(0, 1, 31)[1:-1])),
        1 - _refine(np.logspace(np.log10(1 / 30), -3, 5), log=True))


_PANEL_C_TOTALVAR_LOW = _refine(np.linspace(0, 500, 31)).copy()
_PANEL_C_TOTALVAR_LOW[0] = 1.0                             # never use total variance 0; use 1
_PANEL_C_TOTALVAR_HIGH = _refine(np.linspace(500, 1500, 31))[1:]   # values in (500, 1500]


def panel_C_totalvar_values(p):
    if p > 0.6:
        return np.concatenate([_PANEL_C_TOTALVAR_LOW, _PANEL_C_TOTALVAR_HIGH])
    return _PANEL_C_TOTALVAR_LOW


def panel_D_axes():
    total_var = _refine(np.linspace(0, 500, 31)).copy()
    total_var[0] = 1.0                                    # never use total variance 0; use 1
    return total_var, _refine(np.linspace(0, 10, 31))    # total variance, shift/sqrt(V_A)


# --------------------------------------------------------------------------- #
# Cells (panel position + simulation parameters)
# --------------------------------------------------------------------------- #
def _cell(x, y, N2U, sigma2, shift, null=False):
    return dict(x=float(x), y=float(y), N2U=float(N2U), sigma2=float(sigma2),
                shift=float(shift), null=bool(null))


def _cells_A(ps):
    n2u, sigma2 = panel_A_axes()
    return [_cell(n, s, n, s, ps["chosen_shiftA"]) for n in n2u for s in sigma2]


def _cells_B(ps):
    n2u, shift = panel_B_axes()
    return [_cell(n, sh, n, ps["chosen_sigma2"], sh) for n in n2u for sh in shift]


def _cells_C(ps):
    cells = []
    for p in panel_C_p_values():
        for tv in panel_C_totalvar_values(p):     # tv >= 1 (V_A floored, see panel_C_totalvar_values)
            cells.append(_cell(p, tv, N2U=p * tv / 4.0, sigma2=tv * (1.0 - p),
                               shift=ps["chosen_shiftC"]))
    return cells


def _cells_D(ps):
    tv_axis, std_axis = panel_D_axes()            # tv >= 1 (V_A floored)
    p = ps["chosen_p"]
    cells = []
    for tv in tv_axis:
        for std in std_axis:
            shift = std * np.sqrt(tv)            # absolute shift
            cells.append(_cell(tv, std, N2U=p * tv / 4.0, sigma2=tv * (1.0 - p),
                               shift=shift, null=shift > SHIFT_SCOPE_MAX))
    return cells


def panel_cells(ps):
    """The four panels' cell lists for one parameter-set dict (keys chosen_shiftA /
    chosen_sigma2 / chosen_p / chosen_shiftC)."""
    return dict(A=_cells_A(ps), B=_cells_B(ps), C=_cells_C(ps), D=_cells_D(ps))


# --------------------------------------------------------------------------- #
# Filenames (relative to the results directory)
# --------------------------------------------------------------------------- #
def sim_basename(sdist, N2U, shift, sigma2):
    return "sdist%s_N2U%s_shift%s_sigma2%s.pkl" % (sdist, fnum(N2U), fnum(shift), fnum(sigma2))


def cell_relpath(cell, sdist):
    """Path (relative to RESULTS) of the simulation feeding this cell: a real
    run_grid_simulation pickle, or a null placeholder for an out-of-scope cell.

    Grid cells live in their own directory (simulations_grid/, rule run_grid_simulation)
    rather than alongside the figure sweeps in simulations/: the grid is run at a uniform,
    much lower replicate count (PHASE_SPACE_N_REPLICATES), and identical file names in one
    directory would have made the few cells the two sweeps share ambiguous."""
    sub = "simulations_null" if cell["null"] else "simulations_grid"
    return sub + "/" + sim_basename(sdist, cell["N2U"], cell["shift"], cell["sigma2"])


def null_cells(paramsets):
    """Every null placeholder needed by `paramsets`, as sorted (relpath, cell) pairs,
    de-duplicated across parameter sets and panels.

    Placeholders are written by one job for the whole grid (rule
    phase_space_null_simulations) rather than one job per file: they are zero-replicate
    stubs that cost microseconds each, so hundreds of separate jobs were pure overhead."""
    out = {}
    for name, ps in paramsets.items():
        for panel_cell_list in panel_cells(ps).values():
            for c in panel_cell_list:
                if c["null"]:
                    out.setdefault(cell_relpath(c, ps["sdist"]), c)
    return sorted(out.items())


def all_input_relpaths(ps):
    """Every simulation/null file (relative to RESULTS) the aggregation of one parameter
    set depends on, de-duplicated."""
    seen = set()
    out = []
    for cells in panel_cells(ps).values():
        for c in cells:
            rp = cell_relpath(c, ps["sdist"])
            if rp not in seen:
                seen.add(rp)
                out.append(rp)
    return out


# --------------------------------------------------------------------------- #
# Batching: many grid cells per cluster job
# --------------------------------------------------------------------------- #
# One job per grid cell would be thousands of mostly seconds-long jobs, so cells are
# packed into batches run serially by one job. Cells of equal 2NU stay together, and a
# batch is capped by estimated cost rather than cell count, since per-cell cost spans
# four orders of magnitude. Cost is roughly linear in 2NU, calibrated at 0.0015 s per
# replicate at 2NU = 0.01 and 13-17 core-seconds at 2NU = 1000.
COST_PER_REPLICATE_BASE  = 0.002     # core-seconds, 2NU-independent part
COST_PER_REPLICATE_SLOPE = 0.0173    # core-seconds per unit 2NU


def cell_cost(cell, n_replicates):
    """Estimated cost of simulating one cell, in core-seconds."""
    return n_replicates * (COST_PER_REPLICATE_BASE
                           + COST_PER_REPLICATE_SLOPE * float(cell["N2U"]))


def _pack(items, budget):
    """Greedily pack (key, cost) items, in order, into runs under `budget`. An item
    costlier than the budget becomes a batch of its own; one cell is one simulation."""
    batches, cur, cur_cost = [], [], 0.0
    for key, cost in items:
        if cur and cur_cost + cost > budget:
            batches.append(cur)
            cur, cur_cost = [], 0.0
        cur.append(key)
        cur_cost += cost
    if cur:
        batches.append(cur)
    return batches


def _owner_label(owners):
    """Rule-name-safe label for a set of (paramset, panel) owners, e.g. ('default','A')
    -> 'default_A', and a shared cell -> 'default_A__default_B' (owners joined by '__')."""
    return "__".join("%s_%s" % (name, panel) for name, panel in owners)


def grid_batches(paramsets, n_replicates, budget_core_seconds):
    """Group every simulated grid cell of `paramsets` into batch jobs.

    `paramsets` maps a parameter-set name to its parameter dict. Returns a list of batch
    dicts, each with:

        name   : unique rule-name stem, owner label + index, e.g. 'default_A_00'
        owners : sorted tuple of (paramset_name, panel_letter) whose cells the batch holds
        sdist  : the single effect-size distribution of the batch's cells
        cells  : list of (relpath, cell) pairs (relpath relative to the results dir), in
                 run order

    Null (out-of-scope) cells are excluded: they are placeholders, not simulations.

    Cells are partitioned by which (parameter set, panel) pairs need them, so a batch is
    never a mix of cells wanted by different figures OR different panels OR different
    effect-size distributions -- panel A of Figure 6 batches apart from panel B, and the
    expon grids apart from the mixexpunif ones, so each can be run on its own (see
    grid_group_targets). A cell shared by two panels/parameter sets is de-duplicated to a
    single batch owned by both (so it is still simulated once). Within a partition cells
    are ordered by 2NU (so a batch is cost-homogeneous) and packed up to
    `budget_core_seconds`.
    """
    # relpath -> (cell, set of (paramset, panel) needing it); de-duplicates the cells that
    # several panels or several parameter sets share. The sdist name is part of relpath,
    # so cells sharing one are identical (and always the same sdist).
    cells = {}
    for name, ps in paramsets.items():
        for panel, panel_cell_list in panel_cells(ps).items():
            for c in panel_cell_list:
                if c["null"]:
                    continue
                rp = cell_relpath(c, ps["sdist"])
                cells.setdefault(rp, (dict(c, sdist=ps["sdist"]), set()))[1].add((name, panel))

    partitions = {}
    for rp, (c, owners) in cells.items():
        partitions.setdefault(tuple(sorted(owners)), []).append((rp, c))

    batches = []
    for owners, members in sorted(partitions.items()):
        members.sort(key=lambda rc: (float(rc[1]["N2U"]), rc[0]))
        label = _owner_label(owners)
        sdist = members[0][1]["sdist"]           # single sdist per partition, by construction
        for k, cell_list in enumerate(_pack([(rc, cell_cost(rc[1], n_replicates))
                                             for rc in members], budget_core_seconds)):
            batches.append(dict(name="%s_%02d" % (label, k), owners=owners,
                                sdist=sdist, cells=cell_list))
    return batches


def grid_group_targets(batches):
    """Convenience target -> [relpath] map for running subsets of the grid produced by
    grid_batches(). Targets are emitted per effect-size distribution ('grid_<sdist>'),
    per parameter set ('grid_<paramset>'), per (distribution, panel)
    ('grid_<sdist>_<panel>', e.g. grid_expon_A pools panel A over every expon parameter
    set), and per (parameter set, panel) ('grid_<paramset>_<panel>'). A batch is listed
    under every target it belongs to, so a cell shared by two panels is (correctly) built
    by either panel's target."""
    targets = {}
    for b in batches:
        relpaths = [rp for rp, _c in b["cells"]]
        keys = {"grid_%s" % b["sdist"]}
        for name, panel in b["owners"]:
            keys.add("grid_%s" % name)
            keys.add("grid_%s_%s" % (name, panel))
            keys.add("grid_%s_%s" % (b["sdist"], panel))
        for key in keys:
            targets.setdefault(key, []).extend(relpaths)
    # de-dup while keeping order (a shared batch reaches one paramset target via 2 panels)
    return {t: list(dict.fromkeys(rps)) for t, rps in targets.items()}
