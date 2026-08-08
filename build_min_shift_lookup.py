"""
Precompute a lookup table of `min_shift_new` values over the effect-size (S) axis.

`min_shift_new(S, N, sigma2)` is the smallest optimum shift at which a new allele of
effect S fixes. It dominates the cost of `total_number_of_fixed_new_alleles`, yet
depends only on (S, N, sigma2) -- the `maximum_shift` argument only sets the bisection
bracket -- so sweeping many shifts recomputes the same value inside every `quad`.

This script computes it once on a grid of S spanning the integration support and saves
it; an interpolator built from the table then serves every shift. The table is specific
to a (N, sigma2) pair and to the S-range of `sdist`.

Usage
-----
Build (defaults match the Snakefile: expon(loc=100, scale=400), N=5000, sigma2=40):

    python build_min_shift_lookup.py --output results/min_shift_expon_N5000_sigma240.npz

Consume:

    from build_min_shift_lookup import (
        load_table, make_interpolator, total_number_of_fixed_new_alleles_lookup,
    )
    table  = load_table("results/min_shift_expon_N5000_sigma240.npz")
    interp = make_interpolator(table)
    for shift in np.linspace(0, 100, 101):
        n_new = total_number_of_fixed_new_alleles_lookup(
            sdist=sdist, N=5000, shift=shift, sigma2=40, min_shift_interp=interp)
"""

import argparse
import os
import sys
from functools import partial

import numpy as np
from scipy import stats
from scipy.integrate import quad
from scipy.interpolate import interp1d

# make analytic_functions importable when run from anywhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analytic_functions import (
    ALIGNED_FRACTION,
    _S_integration_limits,
    establishment_prob,
    get_recursion_deterministic,
    mean_t_til_establishment,
    min_shift_new,
)


# --------------------------------------------------------------------------- #
# Building the table
# --------------------------------------------------------------------------- #
def _fixes_new_allele(S, N, sigma2, shift_val):
    """Does a new allele of scaled effect S fix under an optimum shift `shift_val`?

    Replicates the `account_for_time_establishment=True` branch of
    `min_shift_new`, used both to validate the bisection bracket and inside the
    bisection itself (via `min_shift_new`).
    """
    res = mean_t_til_establishment(p=1 / (2 * N), shift=shift_val, a=np.sqrt(S), N=N)
    # mean_t_til_establishment usually returns (t_est, x_est) but returns a bare
    # scalar 0 in the (rare) p > establishment_x edge case; guard both shapes.
    if np.isscalar(res):
        t_est, x_est = 0.0, 1 / (2 * N)
    else:
        t_est, x_est = res
    shift_use = shift_val * np.exp(-t_est * sigma2 / (2 * N))
    return bool(
        get_recursion_deterministic(
            S, x0=x_est, D0=shift_use, sigma2=sigma2, N=N, fix_freq=0.5
        )
    )


def _bracket_max_shift(S, N, sigma2, start=4.0, cap=256.0):
    """Find an upper bracket `m` such that a shift of `m` fixes the allele.

    The true threshold lies in [0, m], so `m` is a valid `maximum_shift` for the
    bisection.  Doubles `start` until fixation is achieved (a large enough shift
    always fixes) or `cap` is exceeded.  Returns (m, fixed): if `fixed` is False
    no fixation was found below `cap` and the threshold is effectively infinite.
    """
    m = start
    while m <= cap:
        if _fixes_new_allele(S, N, sigma2, m):
            return m, True
        m *= 2
    return cap, False


def compute_min_shift(
    S, N, sigma2, tolerance=1e-6, bracket_start=4.0, max_bracket_cap=256.0
):
    """The minimum shift size for which a new allele of effect S fixes.

    Equivalent to `min_shift_new(..., account_for_time_establishment=True)` but
    with an automatically-grown bracket so the result is the true threshold (not
    capped at any particular query `shift + 1`).  Returns `max_bracket_cap` as a
    sentinel "never fixes below the cap" value when no fixation is found.
    """
    m, fixed = _bracket_max_shift(
        S, N, sigma2, start=bracket_start, cap=max_bracket_cap
    )
    if not fixed:
        return float(max_bracket_cap)
    return float(
        min_shift_new(
            S=S,
            N=N,
            sigma2=sigma2,
            minimum_shift=0,
            maximum_shift=m,
            tolerance=tolerance,
            account_for_time_establishment=True,
        )
    )


def build_min_shift_table(
    sdist,
    N,
    sigma2,
    n_grid=200,
    tolerance=1e-6,
    n_jobs=1,
    grid_scale="quantile",
    q_lo=0.0001,
    q_hi=0.9999,
    bracket_start=4.0,
    max_bracket_cap=256.0,
    verbose=True,
):
    """Compute min_shift on a grid of S spanning the integration support of `sdist`.

    The endpoints match the bounds `total_number_of_fixed_new_alleles` uses, so the
    interpolator covers everything the downstream `quad` can sample.

    `grid_scale` controls the interior spacing:
        "quantile" : even samples in probability space, mapped through sdist.ppf;
                     q_lo>0 / q_hi<1 keep the endpoints off the 0/1 quantiles
                     (where ppf may be +/-inf). This concentrates S-points where
                     sdist has the most mass -- i.e. where the sdist.pdf(S) weight
                     in the downstream integral is largest.
        "linear"   : even samples in S between ppf(q_lo) and ppf(q_hi).
        "log"      : geometric samples in S between ppf(q_lo) and ppf(q_hi).

    Returns a dict suitable for `save_table` / `make_interpolator`.
    """
    if grid_scale == "quantile":
        S_grid = np.asarray(sdist.ppf(np.linspace(q_lo, q_hi, n_grid)), dtype=float)
    elif grid_scale == "log":
        S_grid = np.geomspace(float(sdist.ppf(q_lo)), float(sdist.ppf(q_hi)), n_grid)
    else:  # "linear"
        S_grid = np.linspace(float(sdist.ppf(q_lo)), float(sdist.ppf(q_hi)), n_grid)

    worker = partial(
        compute_min_shift,
        N=N,
        sigma2=sigma2,
        tolerance=tolerance,
        bracket_start=bracket_start,
        max_bracket_cap=max_bracket_cap,
    )

    if verbose:
        print(
            f"Building min_shift table: {n_grid} S-points ({grid_scale}) in "
            f"[{S_grid[0]:.4g}, {S_grid[-1]:.4g}], "
            f"N={N}, sigma2={sigma2}, n_jobs={n_jobs}",
            file=sys.stderr,
        )

    if n_jobs == 1:
        min_shift = np.empty(n_grid)
        for i, S in enumerate(S_grid):
            min_shift[i] = worker(S)
            if verbose and (i % 10 == 0 or i == n_grid - 1):
                print(f"  [{i + 1}/{n_grid}] S={S:.4g} -> min_shift={min_shift[i]:.6g}",
                      file=sys.stderr)
    else:
        import multiprocessing as mp

        min_shift = np.empty(n_grid)
        with mp.Pool(processes=n_jobs) as pool:
            for i, val in enumerate(pool.imap(worker, S_grid, chunksize=1)):
                min_shift[i] = val
                if verbose and (i % 10 == 0 or i == n_grid - 1):
                    print(f"  [{i + 1}/{n_grid}] S={S_grid[i]:.4g} -> "
                          f"min_shift={val:.6g}", file=sys.stderr)

    return {
        "S_grid": S_grid,
        "min_shift": min_shift,
        "N": float(N),
        "sigma2": float(sigma2),
        "tolerance": float(tolerance),
        "max_bracket_cap": float(max_bracket_cap),
        "q_lo": float(q_lo),
        "q_hi": float(q_hi),
        # Descriptive metadata only. A frozen scipy distribution carries .dist.name /
        # .args / .kwds; a composite one (MixtureDistribution) carries none of them, so
        # fall back to its class name and empty arguments -- the table itself depends only
        # on the S grid, N and sigma2, so nothing downstream needs these to be exact.
        "dist_name": getattr(getattr(sdist, "dist", None), "name", None) or type(sdist).__name__,
        "dist_args": np.asarray(getattr(sdist, "args", ()), dtype=float),
        "dist_kwds": str(getattr(sdist, "kwds", {})),
    }


# --------------------------------------------------------------------------- #
# Saving / loading / interpolating
# --------------------------------------------------------------------------- #
def save_table(path, table):
    """Save the table to a .npz file (arrays + scalar metadata)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **table)


def load_table(path):
    """Load a table saved by `save_table`; returns a plain dict."""
    with np.load(path, allow_pickle=False) as data:
        out = {}
        for k in data.files:
            arr = data[k]
            # unwrap 0-d arrays back to python scalars / strings
            out[k] = arr.item() if arr.ndim == 0 else arr
        return out


def make_interpolator(table, kind="linear"):
    """Build an S -> min_shift interpolator from a table.

    Linear interpolation (the default) is monotone-safe between grid points and
    will not overshoot; min_shift(S) is smooth, so a moderately fine grid is
    accurate.  Out-of-range S (should not occur, since the grid matches the
    integration bounds) is clamped to the nearest endpoint.
    """
    S = np.asarray(table["S_grid"], dtype=float)
    y = np.asarray(table["min_shift"], dtype=float)
    return interp1d(
        S, y, kind=kind, bounds_error=False, fill_value=(y[0], y[-1])
    )


# --------------------------------------------------------------------------- #
# Lookup-based reimplementation of the new-allele integrals
# --------------------------------------------------------------------------- #
def expected_number_of_fixed_new_alleles_lookup(S, N, shift, sigma2, min_shift_interp):
    """`expected_number_of_fixed_new_alleles` with min_shift read from the table.

    Identical to the original except `min_shift` is interpolated instead of
    recomputed via `min_shift_new`.
    """
    if shift <= np.sqrt(S) / 2:
        return 0.0
    min_shift = float(min_shift_interp(S))
    if min_shift > shift:
        return 0.0
    t_max = max(0, np.log(shift / min_shift) / sigma2 * (2 * N))
    t_unstable = max(0, np.log(shift / (np.sqrt(S) / 2)) / sigma2 * (2 * N))
    return quad(
        lambda t: establishment_prob(
            x=1 / (2 * N), S=S, shift=shift * np.exp(-t * sigma2 / (2 * N)), N=N
        ),
        0,
        t_max,
        points=[t_unstable],
    )[0]


def total_number_of_fixed_new_alleles_lookup(sdist, N, shift, sigma2, min_shift_interp,
                                             S_lo=None, S_hi=None):
    """`total_number_of_fixed_new_alleles` using the precomputed min_shift table.

    Carries the same ALIGNED_FRACTION as the analytic_functions version, so the
    result is per unit of the total mutational input 2NU. S_lo/S_hi restrict the
    integral to an effect-size window, exactly as in analytic_functions.
    """
    limits = _S_integration_limits(sdist, S_lo, S_hi)
    if limits is None:
        return 0.0
    return ALIGNED_FRACTION * quad(
        lambda S: expected_number_of_fixed_new_alleles_lookup(
            S=S, N=N, shift=shift, sigma2=sigma2, min_shift_interp=min_shift_interp
        )
        * sdist.pdf(S),
        *limits,
    )[0]


def expected_contribution_of_fixed_new_alleles_lookup(S, N, shift, sigma2, min_shift_interp):
    """`expected_contribution_of_fixed_new_alleles` with min_shift read from the table.

    A new allele arises as a single copy (x_0 = 1/(2N)), so its contribution to adaptation on
    fixation is the constant 2*sqrt(S)*(1-1/(2N)); scale the lookup-based expected count by it.
    """
    a = np.sqrt(S)
    x0 = 1 / (2 * N)
    return 2 * a * (1 - x0) * expected_number_of_fixed_new_alleles_lookup(
        S=S, N=N, shift=shift, sigma2=sigma2, min_shift_interp=min_shift_interp
    )


def total_contribution_of_fixed_new_alleles_lookup(sdist, N, shift, sigma2, min_shift_interp,
                                                   S_lo=None, S_hi=None):
    """`total_contribution_of_fixed_new_alleles` using the precomputed min_shift table.

    Per unit of the total mutational input 2NU (ALIGNED_FRACTION), as above; S_lo/S_hi
    restrict the integral to an effect-size window.
    """
    limits = _S_integration_limits(sdist, S_lo, S_hi)
    if limits is None:
        return 0.0
    return ALIGNED_FRACTION * quad(
        lambda S: expected_contribution_of_fixed_new_alleles_lookup(
            S=S, N=N, shift=shift, sigma2=sigma2, min_shift_interp=min_shift_interp
        )
        * sdist.pdf(S),
        *limits,
    )[0]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_dist(name, shapes, loc, scale):
    """Construct a frozen scipy.stats distribution from CLI arguments."""
    dist = getattr(stats, name)
    return dist(*shapes, loc=loc, scale=scale)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--N", type=float, default=5000, help="Wright-Fisher population size")
    p.add_argument("--sigma2", type=float, default=40, help="background genetic variance")
    p.add_argument("--dist-name", default="expon",
                   help="scipy.stats distribution name for the S distribution")
    p.add_argument("--dist-shapes", type=float, nargs="*", default=[],
                   help="shape parameters for the distribution (e.g. `a` for gamma)")
    p.add_argument("--loc", type=float, default=100, help="distribution loc")
    p.add_argument("--scale", type=float, default=400, help="distribution scale")
    p.add_argument("--n-grid", type=int, default=200, help="number of S grid points")
    p.add_argument("--grid-scale", choices=["quantile", "linear", "log"],
                   default="quantile",
                   help="S grid spacing: 'quantile' = even in sdist.ppf space "
                        "(default), 'linear'/'log' = even/geometric in S")
    p.add_argument("--tolerance", type=float, default=1e-6,
                   help="bisection tolerance for min_shift (the original uses (1/2)**40)")
    p.add_argument("--n-jobs", type=int, default=1,
                   help="parallel workers across the S grid (S points are independent)")
    p.add_argument("--max-bracket-cap", type=float, default=256,
                   help="largest shift searched; S that do not fix below it get this value")
    p.add_argument("--output", required=True, help="output .npz path")
    args = p.parse_args(argv)

    sdist = _build_dist(args.dist_name, args.dist_shapes, args.loc, args.scale)

    table = build_min_shift_table(
        sdist=sdist,
        N=args.N,
        sigma2=args.sigma2,
        n_grid=args.n_grid,
        tolerance=args.tolerance,
        n_jobs=args.n_jobs,
        grid_scale=args.grid_scale,
        max_bracket_cap=args.max_bracket_cap,
    )
    save_table(args.output, table)
    print(f"Saved min_shift lookup table -> {args.output}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Snakemake entry point
# --------------------------------------------------------------------------- #
def run_from_snakemake(snakemake):
    """Build and save the table from an injected Snakemake `script:` object.

    Required params : N, sdist, sigma2
    Optional params : n_grid, tolerance, grid_scale, max_bracket_cap, n_jobs
                      (n_jobs defaults to snakemake.threads).
    Output          : snakemake.output[0] (.npz path).
    """
    p = snakemake.params

    def _get(name, default):
        try:
            return p[name]
        except (KeyError, IndexError, TypeError, AttributeError):
            return default

    N = p["N"]
    sdist = p["sdist"]
    sigma2 = p["sigma2"]

    table = build_min_shift_table(
        sdist=sdist,
        N=N,
        sigma2=sigma2,
        n_grid=_get("n_grid", 200),
        tolerance=_get("tolerance", 1e-6),
        n_jobs=_get("n_jobs", getattr(snakemake, "threads", 1)),
        grid_scale=_get("grid_scale", "quantile"),
        max_bracket_cap=_get("max_bracket_cap", 256.0),
    )
    save_table(snakemake.output[0], table)
    print(
        f"Saved min_shift lookup table ({table['dist_name']}, N={N}, "
        f"sigma2={sigma2}) -> {snakemake.output[0]}",
        file=sys.stderr,
    )


# Run automatically under Snakemake's `script:` directive (the `snakemake` object
# is injected into globals); otherwise behave as a normal CLI but stay importable.
if "snakemake" in globals():
    run_from_snakemake(snakemake)  # noqa: F821
elif __name__ == "__main__":
    main()
