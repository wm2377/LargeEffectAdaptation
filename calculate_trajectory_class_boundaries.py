"""
Snakemake script (Figure S5, step 2): find the optimum-shift sizes at which the
deterministic allelic trajectory changes class, for one background variance.

For a fixed S = a^2 and x0 = 1/a^2, the deterministic double recursion produces one
of five trajectory classes, whose label decreases monotonically as the shift grows
(see classify_trajectory). This locates the four switching shifts for one sigma2.

Required snakemake.params
    N      : Wright-Fisher population size
    S      : squared effect size S = a^2 (a = sqrt(S), x0 = 1/S)
    sigma2 : background (infinitesimal) genetic variance

Optional snakemake.params (defaults from classify_trajectory)
    b                : binary-search precision for the boundaries (default 0.01)
    fix_freq         : frequency treated as fixation (default 0.9999)
    strong_threshold : 2*N*s above which selection is "strong" (default 10)

Output
    snakemake.output[0] : pickle of {'parameters': ..., 'boundaries': {4,3,2,1}}
        where boundaries[k] is the minimum shift that produces class k (i.e. the
        lower boundary of class k). np.inf marks a boundary not reached.
"""

import os
import sys
import pickle

# make classify_trajectory (in this script's directory) importable when run as a
# Snakemake script
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass

import numpy as np

from classify_trajectory import find_class_boundaries, CLASS_NAMES


def calculate_trajectory_class_boundaries(N, S, sigma2, b=0.01, fix_freq=0.9999,
                                          strong_threshold=10.0):
    """Return {'parameters', 'boundaries'} for one (N, S, sigma2) combination.

    a = sqrt(S) and x0 = 1/S = 1/a^2. `boundaries` is the dict returned by
    find_class_boundaries: {4: shift_5to4, 3: shift_4to3, 2: shift_3to2,
    1: shift_2to1}, keyed by the class that begins at that shift.
    """
    a = np.sqrt(S)
    x0 = 1.0 / S
    boundaries = find_class_boundaries(
        a=a, x0=x0, N=N, sigma2=sigma2, b=b,
        fix_freq=fix_freq, strong_threshold=strong_threshold,
    )
    return {
        "parameters": {
            "N": N, "S": S, "a": a, "x0": x0, "sigma2": sigma2,
            "b": b, "fix_freq": fix_freq, "strong_threshold": strong_threshold,
        },
        "boundaries": boundaries,
        "class_names": CLASS_NAMES,
    }


def _require(params, name):
    try:
        return params[name]
    except (KeyError, TypeError):
        return getattr(params, name)


def _optional(params, name, default):
    try:
        return _require(params, name)
    except (KeyError, AttributeError):
        return default


def main(snakemake):
    p = snakemake.params
    results = calculate_trajectory_class_boundaries(
        N=_require(p, "N"),
        S=_require(p, "S"),
        sigma2=_require(p, "sigma2"),
        b=_optional(p, "b", 0.01),
        fix_freq=_optional(p, "fix_freq", 0.9999),
        strong_threshold=_optional(p, "strong_threshold", 10.0),
    )
    os.makedirs(os.path.dirname(os.path.abspath(snakemake.output[0])), exist_ok=True)
    with open(snakemake.output[0], "wb") as fout:
        pickle.dump(results, fout)

    par = results["parameters"]
    bnd = results["boundaries"]
    print(
        f"trajectory_class_boundaries: S={par['S']} sigma2={par['sigma2']} "
        f"x0={par['x0']:.4g} -> shift boundaries "
        f"(5->4={bnd[4]:.4g}, 4->3={bnd[3]:.4g}, 3->2={bnd[2]:.4g}, 2->1={bnd[1]:.4g})",
        file=sys.stderr,
    )


# Under Snakemake's `script:` directive the `snakemake` object is injected into
# globals; run automatically in that case (but stay importable / runnable directly).
if "snakemake" in globals():
    main(snakemake)  # noqa: F821
elif __name__ == "__main__":
    # Direct run: compute the boundaries for the Figure S5 defaults (S=200,
    # sigma2 from argv[1] or 60) and print them.
    S = 200.0
    sigma2 = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    res = calculate_trajectory_class_boundaries(N=5000, S=S, sigma2=sigma2)
    print(f"S={S} sigma2={sigma2} x0={res['parameters']['x0']}")
    for k in (4, 3, 2, 1):
        print(f"  min shift -> class {k}: {res['boundaries'][k]:.6g}  ({CLASS_NAMES[k]})")
