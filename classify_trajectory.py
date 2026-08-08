"""
Classify the type of allelic trajectory produced by the deterministic double
recursion (the same recursion as analytic_functions.get_recursion_deterministic).

The deterministic (x, D) trajectory is traced from a signed effect size a, initial
frequency x0, population size N, shift D0 and background variance sigma2, then
assigned to one of five classes by the scaled selection coefficient 2*N*s and by D.

Per generation, with Vs = 2N:
    dx = a/Vs * x*(1-x) * (D - a*(1/2 - x)*(1 - D**2/Vs))
    D  += -sigma2*D/Vs - 2*a*dx
Since dx = s*x*(1-x), the coefficient at (x, D) is
    2*N*s = a * (D - a*(1/2 - x)*(1 - D**2/Vs)),
and selection is "strong" when it exceeds STRONG_THRESHOLD.

The five classes:
  1) Fixes; 2*N*s > 10 at every generation; distance D never goes negative.
  2) Fixes; 2*N*s > 10 at every generation; but D becomes negative at some point.
  3) Fixes; but does NOT experience strong selection at every generation.
  4) Does not fix; but experiences strong selection at some point.
  5) Does not fix; never experiences strong selection.

Run directly, e.g.:
    python classify_trajectory.py --a 10 --x0 0.0001 --N 5000 --shift 40 --sigma2 40
"""

import argparse

import numpy as np

STRONG_THRESHOLD = 10.0

CLASS_NAMES = {
    1: "fix, always strong, distance never negative",
    2: "fix, always strong, distance goes negative",
    3: "fix, not always strong",
    4: "no fix, strong at some point",
    5: "no fix, never strong",
}


def scaled_selection_coefficient(a, x, D, N):
    """The scaled selection coefficient 2*N*s at state (x, D), i.e. the recursion's
    dx / (x*(1-x)) * 2N, whose sign decides whether the allele advances."""
    Vs = 2.0 * N
    return a * (D - a * (0.5 - x) * (1.0 - D ** 2 / Vs))


def trace_trajectory(a, x0, shift, sigma2, N, fix_freq=0.9999,
                     strong_threshold=STRONG_THRESHOLD, stop_below_half=False,
                     max_iter=10_000_000):
    """Trace the deterministic (x, D) trajectory and return diagnostics.

    Returns a dict with:
        fixed             : bool  -- did x reach fix_freq?
        always_strong     : bool  -- was 2*N*s > strong_threshold at every generation?
        ever_strong       : bool  -- was 2*N*s > strong_threshold at any generation?
        distance_negative : bool  -- did D ever become < 0?
        n_steps           : int   -- generations simulated
        min_distance      : float -- smallest D reached
        min_scaled_s      : float -- smallest 2*N*s experienced (over generations run)
        max_scaled_s      : float -- largest 2*N*s experienced
        stalled           : bool  -- stopped because dx <= 0 (allele can't advance)
        hit_max_iter      : bool  -- stopped because max_iter was reached
    """
    Vs = 2.0 * N
    x = float(x0)
    D = float(shift)

    fixed = False
    stalled = False
    hit_max_iter = False
    always_strong = True
    ever_strong = False
    distance_negative = D < 0.0
    min_distance = D
    min_scaled_s = np.inf
    max_scaled_s = -np.inf
    n_steps = 0

    while x < fix_freq:
        # selection experienced at the current state, before advancing
        s2N = scaled_selection_coefficient(a=a, x=x, D=D, N=N)
        strong = s2N > strong_threshold
        always_strong = always_strong and strong
        ever_strong = ever_strong or strong
        min_scaled_s = min(min_scaled_s, s2N)
        max_scaled_s = max(max_scaled_s, s2N)

        dx = a / Vs * x * (1.0 - x) * (D - a * (0.5 - x) * (1.0 - D ** 2 / Vs))
        if dx <= 0.0:
            stalled = True
            break

        x += dx
        D += -sigma2 * D / Vs - 2.0 * a * dx
        n_steps += 1

        min_distance = min(min_distance, D)
        if D < 0.0:
            distance_negative = True
        if stop_below_half and D < 0.5:
            stalled = True
            break
        if n_steps >= max_iter:
            hit_max_iter = True
            break
    else:
        # while-condition x < fix_freq turned False without a break -> fixation
        fixed = True

    return {
        "fixed": fixed,
        "always_strong": always_strong,
        "ever_strong": ever_strong,
        "distance_negative": distance_negative,
        "n_steps": n_steps,
        "min_distance": min_distance,
        "min_scaled_s": float(min_scaled_s),
        "max_scaled_s": float(max_scaled_s),
        "stalled": stalled,
        "hit_max_iter": hit_max_iter,
    }


def classify_trajectory(a, x0, shift, sigma2, N, fix_freq=0.9999,
                        strong_threshold=STRONG_THRESHOLD, stop_below_half=False,
                        max_iter=10_000_000):
    """Return the trajectory class (int 1..5) for the given parameters.

    a is the (signed) effect size sqrt(S); shift is the initial distance to the
    optimum D0. See the module docstring for the class definitions.
    """
    d = trace_trajectory(a=a, x0=x0, shift=shift, sigma2=sigma2, N=N,
                         fix_freq=fix_freq, strong_threshold=strong_threshold,
                         stop_below_half=stop_below_half, max_iter=max_iter)
    if d["fixed"]:
        if d["always_strong"]:
            if d["distance_negative"]:
                return 2
            else:
                return 1
        else:
            return 3
    if d["ever_strong"]:
        return 4
    else:
        return 5

def find_class_boundaries(a, x0, N, sigma2, b=0.01, fix_freq=0.9999,
                          strong_threshold=STRONG_THRESHOLD, stop_below_half=False,
                          max_iter=10_000_000, max_shift=1e5, b_min=1e-9):
    '''The four shift sizes at which the trajectory classification switches.

    The class label is non-increasing in shift (5 -> 4 -> 3 -> 2 -> 1), so
    "classify_trajectory(shift) <= k" is monotone and the minimum shift satisfying it
    is the lower boundary of class k. Each is found by binary search to precision b:
        5 -> 4 : min shift at which selection is ever strong
        4 -> 3 : min shift at which the allele fixes
        3 -> 2 : min shift at which it fixes and is strong every generation
        2 -> 1 : min shift at which fixation no longer drives the distance negative

    A class region can be thinner than b, leaving two boundaries equal; the search is
    then repeated at b/10, b/100, ... until every adjacent pair is distinct.

    Returns {4: shift_5to4, 3: shift_4to3, 2: shift_3to2, 1: shift_2to1}, keyed by the
    class beginning at that shift, nested and np.inf where never reached up to
    max_shift.
    '''
    def cls(shift):
        return classify_trajectory(a=a, x0=x0, shift=shift, sigma2=sigma2, N=N,
                                   fix_freq=fix_freq, strong_threshold=strong_threshold,
                                   stop_below_half=stop_below_half, max_iter=max_iter)

    # minimum shift with cls(shift) <= k, searched in [lo, max_shift] to precision prec.
    # `lo` is a known shift with cls(lo) > k (predicate False); we grow an upper bracket
    # that satisfies the predicate, then bisect until the bracket is narrower than prec.
    def min_shift_for(k, lo, prec):
        if cls(lo) <= k:      # already in class <= k at the lower bound
            return lo
        hi = max(lo * 2.0, 1.0)
        while cls(hi) > k:    # expand until the predicate is satisfied (or give up)
            hi *= 2.0
            if hi > max_shift:
                return np.inf
        while hi - lo > prec:
            mid = 0.5 * (lo + hi)
            if cls(mid) <= k:
                hi = mid
            else:
                lo = mid
        return hi

    # all four boundaries at a given precision. They are nested, so each search starts
    # from the previous (lower) one.
    def compute(prec):
        boundaries = {}
        lo = 0.0
        for k in (4, 3, 2, 1):
            bk = min_shift_for(k, lo, prec)
            boundaries[k] = bk
            lo = bk if np.isfinite(bk) else lo
        return boundaries

    # True once every adjacent (finite) pair is separated by more than prec, i.e. no two
    # boundaries collapsed onto each other at this precision.
    def all_distinct(boundaries, prec):
        vals = [boundaries[k] for k in (4, 3, 2, 1)]
        for lo_v, hi_v in zip(vals[:-1], vals[1:]):
            if np.isfinite(lo_v) and np.isfinite(hi_v) and (hi_v - lo_v) <= prec:
                return False
        return True

    prec = b
    boundaries = compute(prec)
    while prec > b_min and not all_distinct(boundaries, prec):
        prec *= 0.1
        boundaries = compute(prec)
    return boundaries
    
def _main():
    p = argparse.ArgumentParser(description="Classify a deterministic allelic trajectory (1..5).")
    p.add_argument("--a", type=float, required=True, help="effect size a = sqrt(S) (signed)")
    p.add_argument("--x0", type=float, required=True, help="initial allele frequency")
    p.add_argument("--N", type=float, required=True, help="population size")
    p.add_argument("--shift", type=float, required=True, help="optimum-shift size (initial distance D0)")
    p.add_argument("--sigma2", type=float, required=True, help="background genetic variance")
    p.add_argument("--fix-freq", type=float, default=0.5, help="fixation frequency threshold (default 0.5)")
    p.add_argument("--strong-threshold", type=float, default=STRONG_THRESHOLD,
                   help="2*N*s threshold for strong selection (default 10)")
    p.add_argument("--stop-below-half", action="store_true",
                   help="honour the get_recursion_deterministic D < 0.5 'not fixed' rule")
    args = p.parse_args()

    d = trace_trajectory(a=args.a, x0=args.x0, shift=args.shift, sigma2=args.sigma2,
                         N=args.N, fix_freq=args.fix_freq,
                         strong_threshold=args.strong_threshold,
                         stop_below_half=args.stop_below_half)
    label = classify_trajectory(a=args.a, x0=args.x0, shift=args.shift, sigma2=args.sigma2,
                                N=args.N, fix_freq=args.fix_freq,
                                strong_threshold=args.strong_threshold,
                                stop_below_half=args.stop_below_half)
    print(f"class {label}: {CLASS_NAMES[label]}")
    for k, v in d.items():
        print(f"  {k}: {v}")
        
    class_boundaries = find_class_boundaries(a=args.a, x0=args.x0, N=args.N, sigma2=args.sigma2)
    print("class boundaries (min shift for class k):")
    for k in (4, 3, 2, 1):
        print(f"  class {k}: {class_boundaries[k]}")


if __name__ == "__main__":
    _main()
