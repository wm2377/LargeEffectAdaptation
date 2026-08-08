"""
Snakemake script: compute, after an optimum shift and split by source
(segregating / new / all), the analytic expectation of three quantities for
large-effect alleles:

    * expected_fixations      : the number of fixed alleles
    * expected_contribution   : their contribution to adaptation, i.e. the sum of
                                2*a*(1-x_0) over the fixed alleles, where a=sqrt(S)
                                is the (non-squared) effect size and x_0 is the
                                allele's initial frequency.
    * expected_establishments : the number of alleles that establish (escape drift
                                while beneficial) -- the fixation threshold dropped,
                                so >= expected_fixations.

Source split:
    * segregating : standing variation present at the shift
    * new         : de novo mutations arising after the shift
    * all         : segregating + new

The expectation is taken over the effect-size distribution `sdist` and stored per
unit of the total mutational input 2NU, so a figure multiplies by 2NU. One pickle
per (sdist, shift, sigma2), mirroring the per-shift simulation outputs so a curve
can be assembled across shifts at plot time.

The dominant cost is the per-S `min_shift_new` bisection, which is independent of
`shift`. With `use_min_shift_lookup`, it is interpolated from the table built by
build_min_shift_lookup.py instead of re-solved per shift.

Required snakemake.params
    N      : Wright-Fisher population size
    sdist  : scipy.stats distribution of scaled selection coefficients S
    sigma2 : background (infinitesimal) genetic variance
    shift  : size of the optimum shift (initial distance d)

Optional snakemake.params
    windows : {name: (lo, hi)} bounds on S = a^2. When supplied, the script switches to
              effect-size-window mode: the same expectations, but with the integral over
              the effect-size distribution restricted to each window in turn (Figure S11),
              written as {window: {'segregating','new','all'}} under 'expected_fixations'
              and 'expected_contribution'. Establishments and the fixed-effect-size
              distribution are not computed in this mode.
    use_min_shift_lookup : if truthy, read min_shift_new from the lookup table
                           (linear interpolation in S) instead of solving for it.
                           Requires the `min_shift_lookup` input below. Default False.

Snakemake input (only when use_min_shift_lookup is set)
    min_shift_lookup : .npz min_shift lookup table from generate_min_shift_lookup.

Output
    snakemake.output[0] : pickle of the results dict produced by
                          calculate_analytic_curves().
"""

import os
import sys
import pickle

# make sure analytic_functions (in this script's directory) is importable when
# run as a Snakemake script
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass

import numpy as np

from analytic_functions import (
    total_number_of_fixed_segregating_alleles,
    total_number_of_fixed_new_alleles,
    total_contribution_of_fixed_segregating_alleles,
    total_contribution_of_fixed_new_alleles,
    total_number_of_established_segregating_alleles,
    total_number_of_established_new_alleles,
    fixed_effect_size_summary,
    normalization_constant,
    expected_number_of_fixed_segregating_alleles,
    expected_number_of_fixed_new_alleles,
)
from build_min_shift_lookup import (
    load_table,
    make_interpolator,
    total_number_of_fixed_new_alleles_lookup,
    total_contribution_of_fixed_new_alleles_lookup,
)


def calculate_analytic_curves(N, sdist, sigma2, shift,
                              use_min_shift_lookup=False,
                              min_shift_table_path=None,
                              compute_fixed_effect_size_distribution=False):
    """Return the analytic expected fixation counts for one parameter combination.

    `use_min_shift_lookup` interpolates the new-allele `min_shift_new` from the
    table at `min_shift_table_path` instead of re-solving it; the segregating term
    is unaffected. `compute_fixed_effect_size_distribution` additionally computes
    f(a) = g(a) X(a) / X, reusing the same interpolator.

    Returns a dict with:
        'parameters'           : the parameters used
        'expected_fixations'   : keyed by source, marginalized over `sdist`
        'expected_contribution': same keys, the contribution to adaptation
        'expected_establishments': same keys, with the fixation threshold dropped;
                                 does not use the min_shift lookup.
        'fixed_effect_size_distribution': the fixed-effect-size summary dict
                                 (S_grid / pdf / cdf / X / boxplot) when the flag is
                                 set, else None.
    """
    n_segregating = total_number_of_fixed_segregating_alleles(
        sdist=sdist, N=N, shift=shift, sigma2=sigma2)
    c_segregating = total_contribution_of_fixed_segregating_alleles(
        sdist=sdist, N=N, shift=shift, sigma2=sigma2)
    e_segregating = total_number_of_established_segregating_alleles(
        sdist=sdist, N=N, shift=shift, sigma2=sigma2)
    e_new = total_number_of_established_new_alleles(
        sdist=sdist, N=N, shift=shift, sigma2=sigma2)

    # None -> the slow direct path
    min_shift_interp = None
    if use_min_shift_lookup:
        if min_shift_table_path is None:
            raise ValueError(
                "use_min_shift_lookup=True requires min_shift_table_path "
                "(the .npz produced by the generate_min_shift_lookup rule)")
        table = load_table(min_shift_table_path)
        # the table is specific to one (N, sigma2)
        if not np.isclose(table['N'], N) or not np.isclose(table['sigma2'], sigma2):
            raise ValueError(
                f"min_shift lookup table was built for N={table['N']}, "
                f"sigma2={table['sigma2']}, but this curve needs N={N}, "
                f"sigma2={sigma2}")
        min_shift_interp = make_interpolator(table)  # kind='linear'
        n_new = total_number_of_fixed_new_alleles_lookup(
            sdist=sdist, N=N, shift=shift, sigma2=sigma2,
            min_shift_interp=min_shift_interp)
        c_new = total_contribution_of_fixed_new_alleles_lookup(
            sdist=sdist, N=N, shift=shift, sigma2=sigma2,
            min_shift_interp=min_shift_interp)
    else:
        n_new = total_number_of_fixed_new_alleles(
            sdist=sdist, N=N, shift=shift, sigma2=sigma2)
        c_new = total_contribution_of_fixed_new_alleles(
            sdist=sdist, N=N, shift=shift, sigma2=sigma2)

    # Distribution of effect sizes among fixed alleles (off by default; see the flag).
    fixed_effect_size_distribution = None
    if compute_fixed_effect_size_distribution:
        fixed_effect_size_distribution = fixed_effect_size_summary(
            sdist=sdist, N=N, shift=shift, sigma2=sigma2,
            min_shift_interp=min_shift_interp)

    return {
        'parameters': {
            'N': N, 'sdist': sdist, 'sigma2': sigma2, 'shift': shift,
            'use_min_shift_lookup': use_min_shift_lookup,
            'compute_fixed_effect_size_distribution': compute_fixed_effect_size_distribution,
        },
        'expected_fixations': {
            'segregating': n_segregating,
            'new': n_new,
            'all': n_segregating + n_new,
        },
        'expected_contribution': {
            'segregating': c_segregating,
            'new': c_new,
            'all': c_segregating + c_new,
        },
        'expected_establishments': {
            'segregating': e_segregating,
            'new': e_new,
            'all': e_segregating + e_new,
        },
        'fixed_effect_size_distribution': fixed_effect_size_distribution,
    }


def calculate_analytic_curves_by_window(N, sdist, sigma2, shift, windows,
                                        use_min_shift_lookup=False,
                                        min_shift_table_path=None):
    """Analytic expectations split by EFFECT-SIZE WINDOW (Figure S11).

    `windows` maps a window name to its (lo, hi] bounds on the squared effect size
    S = a^2 (hi may be inf, e.g. the open large-effect window S > 100). Each window's
    expectation is the same integral calculate_analytic_curves takes over the whole
    effect-size distribution, restricted to the part of that distribution falling in the
    window (analytic_functions' S_lo / S_hi) -- so the windows partition the totals: a
    set of windows covering the support sums back to 'expected_fixations' /
    'expected_contribution'.

    Like the unrestricted curves everything is per unit of the TOTAL mutational input
    2NU. Figure S11 plots each panel against the window-specific input p*2NU (p = the window's mass under sdist)

    Returns a dict with 'parameters' and, keyed by window name, 'expected_fixations' and
    'expected_contribution' dicts of {'segregating', 'new', 'all'}.
    """
    # the min_shift lookup, if enabled, is built once and reused across every window
    min_shift_interp = None
    if use_min_shift_lookup:
        if min_shift_table_path is None:
            raise ValueError(
                "use_min_shift_lookup=True requires min_shift_table_path "
                "(the .npz produced by the generate_min_shift_lookup rule)")
        table = load_table(min_shift_table_path)
        if not np.isclose(table['N'], N) or not np.isclose(table['sigma2'], sigma2):
            raise ValueError(
                f"min_shift lookup table was built for N={table['N']}, "
                f"sigma2={table['sigma2']}, but these curves need N={N}, "
                f"sigma2={sigma2}")
        min_shift_interp = make_interpolator(table)  # kind='linear'

    fixations, contribution = {}, {}
    for name, (lo, hi) in windows.items():
        n_seg = total_number_of_fixed_segregating_alleles(
            sdist=sdist, N=N, shift=shift, sigma2=sigma2, S_lo=lo, S_hi=hi)
        c_seg = total_contribution_of_fixed_segregating_alleles(
            sdist=sdist, N=N, shift=shift, sigma2=sigma2, S_lo=lo, S_hi=hi)
        if min_shift_interp is not None:
            n_new = total_number_of_fixed_new_alleles_lookup(
                sdist=sdist, N=N, shift=shift, sigma2=sigma2,
                min_shift_interp=min_shift_interp, S_lo=lo, S_hi=hi)
            c_new = total_contribution_of_fixed_new_alleles_lookup(
                sdist=sdist, N=N, shift=shift, sigma2=sigma2,
                min_shift_interp=min_shift_interp, S_lo=lo, S_hi=hi)
        else:
            n_new = total_number_of_fixed_new_alleles(
                sdist=sdist, N=N, shift=shift, sigma2=sigma2, S_lo=lo, S_hi=hi)
            c_new = total_contribution_of_fixed_new_alleles(
                sdist=sdist, N=N, shift=shift, sigma2=sigma2, S_lo=lo, S_hi=hi)
        fixations[name] = {'segregating': n_seg, 'new': n_new, 'all': n_seg + n_new}
        contribution[name] = {'segregating': c_seg, 'new': c_new, 'all': c_seg + c_new}

    return {
        'parameters': {
            'N': N, 'sdist': sdist, 'sigma2': sigma2, 'shift': shift,
            'windows': dict(windows),
            'use_min_shift_lookup': use_min_shift_lookup,
        },
        'expected_fixations': fixations,
        'expected_contribution': contribution,
    }


def calculate_fixation_probability(N, S, sigma2, shift):
    """Fixation probability of a large-effect allele of fixed squared effect size S = a^2,
    split by source, with the deterministic double recursion setting the fixation cutoffs.

    Unlike calculate_analytic_curves (which marginalizes over an effect-size distribution
    sdist), this is a single fixed effect size S, so it calls the per-S analytic_functions
    directly. Returns a dict with:
        'parameters'           : the parameters used (N, S, sigma2, shift)
        'fixation_probability' : dict keyed by source:
            'segregating' : normalization_constant(S,N) *
                            expected_number_of_fixed_segregating_alleles(S,N,shift,sigma2)
                            -- the probability that a standing allele of effect S fixes
                            (the quantity Eqs. S26-S28 approximate); its min-frequency
                            cutoff comes from min_freq_segregating (the double recursion).
            'new'         : expected_number_of_fixed_new_alleles(S,N,shift,sigma2) (the
                            per-S new-fixation count, double recursion via min_shift_new)
                            divided by t_max, the time for the optimum distance to decay
                            from `shift` to 1 under Lande's exponential approach
                            D(t) = shift*exp(-t*sigma2/V_S) with V_S = 2N; i.e.
                            t_max = V_S/sigma2 * ln(shift). This turns the arrival-time-
                            integrated count into a per-generation fixation probability.
                            0 when shift <= 1 (t_max non-positive) or nothing fixes.
    """
    p_segregating = normalization_constant(S=S, N=N) * expected_number_of_fixed_segregating_alleles(
        S=S, N=N, shift=shift, sigma2=sigma2)
    # New alleles: normalize the expected count by t_max = V_S/sigma2 * ln(shift) (V_S = 2N),
    # the Lande time for the distance to decay from `shift` to 1. Guard shift <= 1 (t_max <= 0).
    n_new = expected_number_of_fixed_new_alleles(S=S, N=N, shift=shift, sigma2=sigma2)
    V_S = 2 * N
    t_max = V_S / sigma2 * np.log(shift) if shift > 1 else 0.0
    p_new = n_new / t_max if t_max > 0 else 0.0
    return {
        'parameters': {'N': N, 'S': S, 'sigma2': sigma2, 'shift': shift},
        'fixation_probability': {'segregating': p_segregating, 'new': p_new},
    }


def _require(params, name):
    """Read a required snakemake param, with a clear error if it is missing."""
    try:
        return params[name]
    except (KeyError, IndexError, TypeError, AttributeError):
        raise KeyError(f"required snakemake param '{name}' is missing")


def _optional(params, name, default):
    """Read an optional snakemake param, falling back to `default` if absent."""
    try:
        return params[name]
    except (KeyError, IndexError, TypeError, AttributeError):
        return default


def main(snakemake):
    p = snakemake.params

    # Fixed-effect-size fixation-probability mode: when a squared effect size 'S' is
    # supplied (in place of a distribution 'sdist'), compute the per-S fixation
    # probability split by source rather than the sdist-integrated curves.
    S = _optional(p, 'S', None)
    if S is not None:
        results = calculate_fixation_probability(
            N=_require(p, 'N'), S=float(S),
            sigma2=_require(p, 'sigma2'), shift=_require(p, 'shift'))
        with open(snakemake.output[0], 'wb') as fout:
            pickle.dump(results, fout)
        pr = results['parameters']
        fp = results['fixation_probability']
        print(
            f"calculate_analytic_curves[fixation_probability]: S={pr['S']} "
            f"shift={pr['shift']} sigma2={pr['sigma2']} -> "
            f"P_fix(seg={fp['segregating']:.4g}, new={fp['new']:.4g})",
            file=sys.stderr,
        )
        return

    use_lookup = bool(_optional(p, 'use_min_shift_lookup', False))
    table_path = None
    if use_lookup:
        # min_shift table is declared as the named input 'min_shift_lookup'
        table_path = snakemake.input.min_shift_lookup

    # Effect-size-window mode: when 'windows' is supplied, split the same expectations by
    # window (Figure S11) instead of integrating over the whole effect-size distribution.
    windows = _optional(p, 'windows', None)
    if windows is not None:
        results = calculate_analytic_curves_by_window(
            N=_require(p, 'N'), sdist=_require(p, 'sdist'),
            sigma2=_require(p, 'sigma2'), shift=_require(p, 'shift'),
            windows=windows, use_min_shift_lookup=use_lookup,
            min_shift_table_path=table_path)
        with open(snakemake.output[0], 'wb') as fout:
            pickle.dump(results, fout)
        pr = results['parameters']
        summary = " ".join(
            f"{name}(fix={results['expected_fixations'][name]['all']:.4g}, "
            f"contrib={results['expected_contribution'][name]['all']:.4g})"
            for name in windows)
        print(
            f"calculate_analytic_curves[by_window]: shift={pr['shift']} "
            f"sigma2={pr['sigma2']} min_shift_lookup={use_lookup} -> {summary}",
            file=sys.stderr,
        )
        return

    compute_dfe = bool(_optional(p, 'compute_fixed_effect_size_distribution', False))

    results = calculate_analytic_curves(
        N=_require(p, 'N'),
        sdist=_require(p, 'sdist'),
        sigma2=_require(p, 'sigma2'),
        shift=_require(p, 'shift'),
        use_min_shift_lookup=use_lookup,
        min_shift_table_path=table_path,
        compute_fixed_effect_size_distribution=compute_dfe,
    )

    with open(snakemake.output[0], 'wb') as fout:
        pickle.dump(results, fout)

    ef = results['expected_fixations']
    ec = results['expected_contribution']
    ee = results['expected_establishments']
    print(
        f"calculate_analytic_curves: shift={results['parameters']['shift']} "
        f"sigma2={results['parameters']['sigma2']} "
        f"min_shift_lookup={use_lookup} -> "
        f"fixations(seg={ef['segregating']:.4g}, new={ef['new']:.4g}, "
        f"all={ef['all']:.4g}) "
        f"contribution(seg={ec['segregating']:.4g}, new={ec['new']:.4g}, "
        f"all={ec['all']:.4g}) "
        f"establishments(seg={ee['segregating']:.4g}, new={ee['new']:.4g}, "
        f"all={ee['all']:.4g})",
        file=sys.stderr,
    )


# Under Snakemake's `script:` directive the `snakemake` object is injected into
# globals; run automatically in that case (but stay importable for testing).
if "snakemake" in globals():
    main(snakemake)  # noqa: F821
