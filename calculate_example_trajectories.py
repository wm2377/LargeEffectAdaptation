"""
Snakemake script (Figure S5, step 3): for one (a, x0, sigma2, shift) combination,
compute the expected (deterministic) trajectory of the allele frequency x and the
distance-to-optimum D, plus n_realized (default 100) realized trajectories that
include drift.

Both use the single-site dynamics of single_site_simulation_classes.py (V_S = 2N):
    E[dx] = x(1-x)/V_S * (a*D + a^2*(x - 1/2)*(1 - D^2/V_S))
    D    += -sigma2*D/V_S - 2*a*dx          (dx the realized change in x)
The realized trajectories add drift, x_new = Binomial(2N, x + E[dx]) / (2N), and run
to absorption; the expected one iterates the same E[dx] with dx = E[dx] until the
allele is near fixed or lost, it stalls, or max_generations is reached.

Required snakemake.params
    N      : Wright-Fisher population size (V_S = 2N)
    a      : (signed) effect size a = sqrt(S); S = a^2, sign = sign(a)
    x0     : initial allele frequency
    sigma2 : background (infinitesimal) genetic variance
    shift  : optimum-shift size = initial distance D0

Optional snakemake.params
    n_realized      : number of drift trajectories (default 100)
    seed            : base RNG seed for the realized trajectories (default 0)
    max_generations : cap on the expected-trajectory length (default 50000)

Output
    snakemake.output[0] : pickle of {'parameters', 'expected', 'realized'} (see
        calculate_example_trajectories for the structure).
"""

import os
import sys
import pickle

# make the single-site simulation classes (in this script's directory) importable
# when run as a Snakemake script
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass

import numpy as np

from single_site_simulation_classes import SegregatingSimulation


def expected_trajectory(sim, x0, shift, max_generations=50000):
    """Deterministic (drift-free) trajectory of (x, D) from (x0, shift).

    Iterates the simulation's mean dynamics until x is within 1/(2N) of fixation or
    loss, the trajectory stalls (x and D both stop changing), or max_generations is
    reached. Returns {'x', 'D', 'time', 'fixed', 'lost', 'stalled'} with x/D/time as
    equal-length arrays indexed from generation 0.
    """
    V_S, a, sigma2, eps = sim.V_S, sim.a, sim.sigma2, 1.0 / (2.0 * sim.N)
    x, D = float(x0), float(shift)
    xs, Ds = [x], [D]
    fixed = lost = stalled = False

    for _ in range(int(max_generations)):
        if x >= 1.0 - eps:
            fixed = True
            break
        if x <= eps:
            lost = True
            break
        dx = sim.expected_delta_x(x, D)                 # E[dx] at the current state
        x_new = min(max(x + dx, 0.0), 1.0)
        D_new = D - sigma2 * D / V_S - 2.0 * a * (x_new - x)
        if abs(x_new - x) < 1e-15 and abs(D_new - D) < 1e-15:
            stalled = True                              # sitting at a fixed point
            x, D = x_new, D_new
            xs.append(x)
            Ds.append(D)
            break
        x, D = x_new, D_new
        xs.append(x)
        Ds.append(D)

    return {
        "x": np.asarray(xs),
        "D": np.asarray(Ds),
        "time": np.arange(len(xs)),
        "fixed": bool(fixed),
        "lost": bool(lost),
        "stalled": bool(stalled),
    }


def realized_trajectories(sim, x0, n_realized=100):
    """n_realized drift trajectories from x0 (D starting at the shift), to absorption.

    Reuses sim.run(x0=x0, record_trajectory=True); the single Generator advances
    between calls, so the trajectories are independent. Returns a list of dicts
    {'x', 'D', 'time', 'fixed', 'n_generations'}.
    """
    out = []
    for _ in range(int(n_realized)):
        r = sim.run(x0=x0, record_trajectory=True)
        out.append({
            "x": np.asarray(r.x_trajectory),
            "D": np.asarray(r.D_trajectory),
            "time": np.asarray(r.time_trajectory),
            "fixed": bool(r.fixed),
            "n_generations": int(r.n_generations),
        })
    return out


def calculate_example_trajectories(N, a, x0, sigma2, shift, n_realized=100, seed=0,
                                   max_generations=50000):
    """Expected + n_realized trajectories for one (a, x0, sigma2, shift) combination.

    a is the signed effect size (S = a^2, sign = sign(a)). Returns
        {'parameters', 'expected', 'realized'}
    where 'expected' is expected_trajectory(...) and 'realized' is a list of
    realized_trajectories(...) dicts. The expected trajectory is computed first (it
    uses no randomness), then the realized ones from a seeded Generator.
    """
    S = float(a) ** 2
    sign = 1 if a >= 0 else -1
    sim = SegregatingSimulation(N=N, S=S, sign=sign, sigma2=sigma2, Lambda=shift, seed=seed)

    expected = expected_trajectory(sim, x0=x0, shift=shift, max_generations=max_generations)
    realized = realized_trajectories(sim, x0=x0, n_realized=n_realized)

    return {
        "parameters": {
            "N": N, "a": float(a), "S": S, "sign": sign,
            "x0": float(x0), "sigma2": float(sigma2), "shift": float(shift),
            "n_realized": int(n_realized), "seed": int(seed),
            "max_generations": int(max_generations),
        },
        "expected": expected,
        "realized": realized,
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
    results = calculate_example_trajectories(
        N=_require(p, "N"),
        a=_require(p, "a"),
        x0=_require(p, "x0"),
        sigma2=_require(p, "sigma2"),
        shift=_require(p, "shift"),
        n_realized=_optional(p, "n_realized", 100),
        seed=_optional(p, "seed", 0),
        max_generations=_optional(p, "max_generations", 50000),
    )
    os.makedirs(os.path.dirname(os.path.abspath(snakemake.output[0])), exist_ok=True)
    with open(snakemake.output[0], "wb") as fout:
        pickle.dump(results, fout)

    e = results["expected"]
    n_fixed = sum(r["fixed"] for r in results["realized"])
    print(
        f"example_trajectories: a={results['parameters']['a']:.4g} "
        f"x0={results['parameters']['x0']:.4g} shift={results['parameters']['shift']:.4g} "
        f"sigma2={results['parameters']['sigma2']:.4g} -> expected "
        f"({'fixed' if e['fixed'] else 'lost' if e['lost'] else 'stalled/capped'}, "
        f"{len(e['x'])-1} gens); realized {n_fixed}/{len(results['realized'])} fixed",
        file=sys.stderr,
    )


# Under Snakemake's `script:` directive the `snakemake` object is injected into
# globals; run automatically in that case (but stay importable / runnable directly).
if "snakemake" in globals():
    main(snakemake)  # noqa: F821
elif __name__ == "__main__":
    # Direct run with the Figure S5 defaults: S = a^2 = 200, x0 = 1/a^2, N = 5000,
    # shift and sigma2 from argv (default shift=50, sigma2=60).
    N = 5000
    S = 200.0
    a = np.sqrt(S)
    x0 = 1.0 / S
    shift = float(sys.argv[1]) if len(sys.argv) > 1 else 50.0
    sigma2 = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    res = calculate_example_trajectories(N=N, a=a, x0=x0, sigma2=sigma2, shift=shift)
    e = res["expected"]
    n_fixed = sum(r["fixed"] for r in res["realized"])
    print(f"a={a:.4g} x0={x0} shift={shift} sigma2={sigma2} N={N}")
    print(f"  expected: {'fixed' if e['fixed'] else 'lost' if e['lost'] else 'stalled/capped'}, "
          f"{len(e['x'])-1} generations, final x={e['x'][-1]:.4f}, final D={e['D'][-1]:.4f}")
    print(f"  realized: {n_fixed}/{len(res['realized'])} fixed; "
          f"lengths min/median/max = "
          f"{min(len(r['x']) for r in res['realized'])}/"
          f"{int(np.median([len(r['x']) for r in res['realized']]))}/"
          f"{max(len(r['x']) for r in res['realized'])}")
