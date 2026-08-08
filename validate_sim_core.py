"""Check that the C++ simulation core reproduces the Python reference implementation.

The C++ port (cpp/sim_core.cpp, driven by simulation_cpp.py) uses different RNG
algorithms and replaces the standing-variation root-finds with tabulated inverse
cdfs, so its replicates are NOT bit-identical to simulation_classes.Simulation's.
They should be draws from the same distributions. This script checks that, layer
by layer, so a failure points at the layer that broke:

  1. samplers        -- the C++ binomial and Poisson against NumPy's, at the
                        (n, p) and lambda regimes the recursion actually hits
  2. total_n         -- expected number of standing-variation sites, C++ table
                        integral against the scipy nested quadrature
  3. standing        -- the sampled standing variation (frequency x, effect S)
                        against generate_segregating_mutations.generate_alleles,
                        and (3b) against the exact steady-state marginals, which
                        resolves far smaller biases than the slow Python sampler can
  4. replicates      -- whole replicates: fixation counts, fixed effect sizes,
                        quasi-static onset time, and the mean distance trajectory

Usage
    python validate_sim_core.py                 # default: quick config
    python validate_sim_core.py --full          # more replicates, both 2NU values
    python validate_sim_core.py --workers 16

Reported p-values are two-sided KS or chi-square; with this many comparisons an
occasional p < 0.05 is expected, so each check prints its statistic and the
script only fails on p < 0.001 (or on a mean that is outside a 4-SE band).

On precision: the tabulated steady state reproduces the exact per-site variance
contribution 2 S x (1-x) to within about 0.1%. That statistic is heavy-tailed --
a single 400k-draw batch scatters by ~0.3% -- so the check below averages over
independent batches rather than trusting one sample's standard error. The
fixation counts, which are what the figures actually plot, match the existing
10,000-replicate Python pickles across the 2NU sweep to within 2.1 SE.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import simulation_cpp as sc
from generate_segregating_mutations import generate_alleles, folded_sojourn_time
from simulation_classes import Simulation

SDIST = stats.expon(loc=100, scale=400)
N = 5000

# p-value below which a comparison is called a failure (not merely noise)
ALPHA = 1e-3

_failures = []


def check(name, ok, detail):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: {detail}")
    if not ok:
        _failures.append(name)


# --------------------------------------------------------------------------- #
# 1. discrete samplers
# --------------------------------------------------------------------------- #
def _discrete_chisq(a, b):
    """Chi-square contingency test between two samples of small integers."""
    lo, hi = int(min(a.min(), b.min())), int(max(a.max(), b.max()))
    bins = np.arange(lo, hi + 2) - 0.5
    oa, _ = np.histogram(a, bins=bins)
    ob, _ = np.histogram(b, bins=bins)
    keep = (oa + ob) >= 10                      # pool away the sparse tails
    if keep.sum() < 2:
        return None
    return stats.chi2_contingency(np.vstack([oa[keep], ob[keep]]))


def check_samplers(n_draws=200000):
    """C++ binomial / Poisson against NumPy, over the regimes the loop uses."""
    print(f"\n1. discrete samplers (C++ vs NumPy, {n_draws} draws each)")
    rng = np.random.default_rng(0)

    # Binomial: 2N Wright-Fisher trials, at frequencies spanning a new mutation
    # (one copy) up to the middle of the range. p*2N < 30 exercises the inversion
    # branch, above it the BTRS rejection branch.
    for i, p in enumerate((1.0 / (2 * N), 1e-3, 1e-2, 0.05, 0.5)):
        cpp = sc.sample_binomial(2 * N, p, n_draws, seed=100 + i)
        npy = rng.binomial(2 * N, p, size=n_draws).astype(float)
        se = np.sqrt(cpp.var() / n_draws + npy.var() / n_draws)
        z = (cpp.mean() - npy.mean()) / se
        res = _discrete_chisq(cpp, npy)
        pv = res.pvalue if res is not None else 1.0
        check(f"binomial n={2 * N} p={p:g}", abs(z) < 4 and pv > ALPHA,
              f"mean cpp={cpp.mean():.4f} numpy={npy.mean():.4f} (z={z:+.2f}), "
              f"var cpp={cpp.var():.4f} numpy={npy.var():.4f}, chi2 p={pv:.3f}")

    # Poisson: the per-generation mutation influx, over the swept 2NU range.
    # lambda < 10 exercises Knuth, above it PTRS.
    for i, lam in enumerate((0.001, 0.01, 1.0, 10.0, 158.0, 1000.0)):
        cpp = sc.sample_poisson(lam, n_draws, seed=200 + i)
        npy = rng.poisson(lam, size=n_draws).astype(float)
        se = np.sqrt(cpp.var() / n_draws + npy.var() / n_draws)
        z = (cpp.mean() - npy.mean()) / se if se > 0 else 0.0
        res = _discrete_chisq(cpp, npy)
        pv = res.pvalue if res is not None else 1.0
        check(f"poisson lambda={lam:g}", abs(z) < 4 and pv > ALPHA,
              f"mean cpp={cpp.mean():.5f} numpy={npy.mean():.5f} (z={z:+.2f}), "
              f"var cpp={cpp.var():.5f} numpy={npy.var():.5f}, chi2 p={pv:.3f}")


# --------------------------------------------------------------------------- #
# 2. expected number of standing-variation sites
# --------------------------------------------------------------------------- #
def check_total_n(n2u=10.0):
    print("\n2. total_n (expected standing-variation sites)")
    sim = Simulation(N=N, sdist=SDIST, N2U=n2u, sigma2=40.0, shift=80.0, seed=0)
    py = sim.total_n()
    cpp = sc.expected_n_segregating(N, SDIST, n2u)
    rel = abs(cpp - py) / py
    check("total_n", rel < 1e-3, f"python={py:.6f} cpp={cpp:.6f} rel.diff={rel:.2e}")


# --------------------------------------------------------------------------- #
# 3. standing-variation sampler
# --------------------------------------------------------------------------- #
def _py_standing(args):
    n, seed = args
    rng = np.random.default_rng(seed)
    sim = Simulation(N=N, sdist=SDIST, N2U=1.0, sigma2=40.0, shift=80.0, seed=seed)
    from scipy.integrate import quad
    prob_segregating = quad(
        lambda S: quad(lambda x: folded_sojourn_time(S=S, x=x, N=N), 0, 1 / 2,
                       points=[1 / (2 * N)])[0] * SDIST.pdf(S),
        SDIST.ppf(0), SDIST.ppf(0.9999))[0]
    del sim
    muts = generate_alleles(n=n, prob_segregating=prob_segregating, N=N, sdist=SDIST, rng=rng)
    arr = np.asarray(muts, dtype=float)
    return arr[:, 0], arr[:, 1]


def check_standing_exact(n_samples=40000):
    """C++ standing variation against the exact steady-state marginals.

    A sharper test than comparing with the Python sampler, which is too slow to
    run enough times to resolve small biases: here the target is the analytic
    density itself, integrated with scipy. The joint steady state is
    p(S, x) ~ sdist.pdf(S) * folded_sojourn_time(S, x), so the marginals are

        p(x) ~ integral over S of sdist.pdf(S) * T(S, x)
        p(S) ~ sdist.pdf(S) * integral over x of T(S, x)

    over the same effect-size range the Python sampler normalizes on,
    [ppf(0), ppf(0.9999)].
    """
    from scipy.integrate import quad
    print(f"\n3b. standing variation vs the exact marginals ({n_samples} draws)")
    s_lo, s_hi = SDIST.ppf(0), SDIST.ppf(0.9999)
    x, S = sc.sample_standing_variation(N, SDIST, n_samples, seed=42)

    # frequency: grid dense near the 1/(2N) boundary, geometric above it
    xg = np.concatenate([np.linspace(0, 1 / (2 * N), 400),
                         np.exp(np.linspace(np.log(1 / (2 * N)), np.log(0.5), 1200))[1:]])
    fx = np.array([quad(lambda s: SDIST.pdf(s) * folded_sojourn_time(S=s, x=max(v, 1e-12), N=N),
                        s_lo, s_hi, limit=200)[0] for v in xg])
    Fx = np.concatenate([[0.0], np.cumsum(0.5 * (fx[1:] + fx[:-1]) * np.diff(xg))])
    Fx /= Fx[-1]
    ks = stats.ks_1samp(x, lambda v: np.interp(v, xg, Fx))
    check("standing frequency x vs exact cdf", ks.pvalue > ALPHA,
          f"KS D={ks.statistic:.5f} p={ks.pvalue:.3f}")

    # effect size
    Sg = SDIST.ppf(np.linspace(0, 0.9999, 600))
    fS = np.array([SDIST.pdf(v) * quad(lambda y: folded_sojourn_time(S=v, x=y, N=N),
                                       0, 0.5, points=[1 / (2 * N)])[0] for v in Sg])
    FS = np.concatenate([[0.0], np.cumsum(0.5 * (fS[1:] + fS[:-1]) * np.diff(Sg))])
    FS /= FS[-1]
    ks = stats.ks_1samp(S, lambda v: np.interp(v, Sg, FS))
    check("standing effect S vs exact cdf", ks.pvalue > ALPHA,
          f"KS D={ks.statistic:.5f} p={ks.pvalue:.3f} "
          f"(mean cpp={S.mean():.2f} exact={np.trapezoid(Sg * fS, Sg) / np.trapezoid(fS, Sg):.2f})")

    # The moment that actually drives the dynamics: each standing site contributes
    # 2 a^2 x (1-x) = 2 S x (1-x) to the genetic variance at the shift. It is a
    # heavy-tailed statistic, so its uncertainty is estimated across independent
    # batches rather than from a single sample's variance (a single 400k batch
    # scatters by ~0.3%, which is easy to mistake for a bias).
    num = quad(lambda s: SDIST.pdf(s) * quad(
        lambda y: 2 * s * y * (1 - y) * folded_sojourn_time(S=s, x=y, N=N),
        0, 0.5, points=[1 / (2 * N)], limit=200)[0], s_lo, s_hi, limit=200)[0]
    den = quad(lambda s: SDIST.pdf(s) * quad(
        lambda y: folded_sojourn_time(S=s, x=y, N=N),
        0, 0.5, points=[1 / (2 * N)], limit=200)[0], s_lo, s_hi, limit=200)[0]
    exact = num / den
    batch_means = []
    for b in range(10):
        xb, Sb = sc.sample_standing_variation(N, SDIST, 200000, seed=5000 + b)
        batch_means.append((2 * Sb * xb * (1 - xb)).mean())
    batch_means = np.array(batch_means)
    se = batch_means.std(ddof=1) / np.sqrt(len(batch_means))
    bias = (batch_means.mean() - exact) / exact
    # tolerance: 1% on the initial genetic variance, far below the replicate-to-
    # replicate scatter any figure sees
    check("per-site variance 2Sx(1-x) vs exact", abs(bias) < 0.01,
          f"cpp={batch_means.mean():.5f} exact={exact:.5f} bias={100 * bias:+.2f}% "
          f"(SE={100 * se / exact:.2f}%)")


def check_standing(n_samples=600, workers=8):
    """Sampled (x, S) from C++ vs the scipy root-find sampler, by KS test."""
    print(f"\n3. standing variation ({n_samples} alleles per engine)")
    per = max(1, n_samples // workers)
    jobs = [(per, 1000 + i) for i in range(workers)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(_py_standing, jobs))
    py_x = np.concatenate([p[0] for p in parts])
    py_S = np.concatenate([p[1] for p in parts])
    t_py = time.time() - t0

    t0 = time.time()
    cpp_x, cpp_S = sc.sample_standing_variation(N, SDIST, len(py_x), seed=7)
    t_cpp = time.time() - t0
    print(f"     python {len(py_x)} alleles in {t_py:.1f}s; cpp in {t_cpp:.4f}s")

    ks_x = stats.ks_2samp(py_x, cpp_x)
    ks_S = stats.ks_2samp(py_S, cpp_S)
    check("standing frequency x", ks_x.pvalue > ALPHA,
          f"KS D={ks_x.statistic:.4f} p={ks_x.pvalue:.3f} "
          f"(median python={np.median(py_x):.3e} cpp={np.median(cpp_x):.3e})")
    check("standing effect S", ks_S.pvalue > ALPHA,
          f"KS D={ks_S.statistic:.4f} p={ks_S.pvalue:.3f} "
          f"(mean python={py_S.mean():.1f} cpp={cpp_S.mean():.1f})")

    # the quantity the dynamics actually care about: the genetic variance the
    # standing variation contributes at the shift, 2 a^2 x (1-x) summed per site
    v_py = (2 * py_S * py_x * (1 - py_x)).mean()
    v_cpp = (2 * cpp_S * cpp_x * (1 - cpp_x)).mean()
    se = np.sqrt((2 * py_S * py_x * (1 - py_x)).var() / len(py_x) +
                 (2 * cpp_S * cpp_x * (1 - cpp_x)).var() / len(cpp_x))
    check("per-site variance contribution", abs(v_py - v_cpp) < 4 * se,
          f"python={v_py:.3f} cpp={v_cpp:.3f} diff={v_py - v_cpp:+.3f} (4SE={4 * se:.3f})")


# --------------------------------------------------------------------------- #
# 4. whole replicates
# --------------------------------------------------------------------------- #
def _py_replicate(args):
    n2u, sigma2, shift, stop_time, seed = args
    sim = Simulation(N=N, sdist=SDIST, N2U=n2u, sigma2=sigma2, shift=shift,
                     stop_time=stop_time, seed=seed)
    sim.run_simulation()
    return {
        'n_fixations': len(sim.stats['fixations']),
        'fixed_effect_sizes': np.array([m.a for m in sim.stats['fixations']]),
        'convergence_time': sim.convergence_time,
        'convergence_D_window': sim.convergence_D_window,
        'd_trajectory': np.asarray(sim.stats['d_trajectory'], dtype=float),
    }


def _cpp_replicate(args):
    n2u, sigma2, shift, stop_time, seed = args
    rep = sc.run_replicate(N=N, sdist=SDIST, N2U=n2u, sigma2=sigma2, shift=shift,
                           stop_time=stop_time, seed=seed, full_output=True)
    return rep


def check_replicates(n2u, n_reps, stop_time=20000, sigma2=40.0, shift=80.0, workers=8):
    print(f"\n4. replicates at 2NU={n2u} ({n_reps} per engine, stop_time={stop_time:g})")
    py_jobs = [(n2u, sigma2, shift, stop_time, 20000 + i) for i in range(n_reps)]
    cpp_jobs = [(n2u, sigma2, shift, stop_time, 90000 + i) for i in range(n_reps)]

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        py = list(ex.map(_py_replicate, py_jobs))
    t_py = time.time() - t0

    t0 = time.time()
    cpp = [_cpp_replicate(j) for j in cpp_jobs]
    t_cpp = time.time() - t0
    print(f"     python {n_reps} reps in {t_py:.1f}s ({workers} workers); "
          f"cpp in {t_cpp:.2f}s (1 worker) -> {t_py * workers / max(t_cpp, 1e-9):.0f}x per core")

    # fixation counts
    fp = np.array([r['n_fixations'] for r in py], dtype=float)
    fc = np.array([r['n_fixations'] for r in cpp], dtype=float)
    se = np.sqrt(fp.var() / len(fp) + fc.var() / len(fc))
    check("mean fixations", abs(fp.mean() - fc.mean()) < 4 * se + 1e-12,
          f"python={fp.mean():.4f} cpp={fc.mean():.4f} diff={fp.mean() - fc.mean():+.4f} "
          f"(4SE={4 * se:.4f})")

    # distribution of the counts (chi-square over the observed support)
    lo, hi = int(min(fp.min(), fc.min())), int(max(fp.max(), fc.max()))
    if hi > lo:
        bins = np.arange(lo, hi + 2) - 0.5
        op, _ = np.histogram(fp, bins=bins)
        oc, _ = np.histogram(fc, bins=bins)
        keep = (op + oc) >= 5
        if keep.sum() >= 2:
            chi = stats.chi2_contingency(np.vstack([op[keep], oc[keep]]))
            check("fixation-count distribution", chi.pvalue > ALPHA,
                  f"chi2={chi.statistic:.2f} p={chi.pvalue:.3f}")

    # effect sizes of the alleles that fixed
    ep = np.concatenate([r['fixed_effect_sizes'] for r in py]) if fp.sum() else np.array([])
    ec = np.concatenate([r['fixed_effect_sizes'] for r in cpp]) if fc.sum() else np.array([])
    if len(ep) >= 20 and len(ec) >= 20:
        ks = stats.ks_2samp(ep, ec)
        check("fixed effect sizes", ks.pvalue > ALPHA,
              f"KS D={ks.statistic:.4f} p={ks.pvalue:.3f} "
              f"(mean python={ep.mean():.2f} cpp={ec.mean():.2f})")
    else:
        print(f"  [skip] fixed effect sizes: too few fixations "
              f"(python={len(ep)}, cpp={len(ec)})")

    # onset of the quasi-static phase
    tp = np.array([r['convergence_time'] for r in py], dtype=float)
    tc = np.array([r['convergence_time'] for r in cpp], dtype=float)
    fin_p, fin_c = tp[~np.isnan(tp)], tc[~np.isnan(tc)]
    check("quasi-static onset: fraction reached",
          abs(len(fin_p) / len(tp) - len(fin_c) / len(tc)) < 0.1,
          f"python={len(fin_p) / len(tp):.3f} cpp={len(fin_c) / len(tc):.3f}")
    if len(fin_p) >= 20 and len(fin_c) >= 20:
        ks = stats.ks_2samp(fin_p, fin_c)
        check("quasi-static onset time", ks.pvalue > ALPHA,
              f"KS D={ks.statistic:.4f} p={ks.pvalue:.3f} "
              f"(median python={np.median(fin_p):.0f} cpp={np.median(fin_c):.0f})")

        dp = np.array([r['convergence_D_window'] for r in py], dtype=float)
        dc = np.array([r['convergence_D_window'] for r in cpp], dtype=float)
        dp, dc = dp[~np.isnan(dp)], dc[~np.isnan(dc)]
        ks = stats.ks_2samp(dp, dc)
        check("quasi-static onset distance", ks.pvalue > ALPHA,
              f"KS D={ks.statistic:.4f} p={ks.pvalue:.3f} "
              f"(mean python={dp.mean():.3f} cpp={dc.mean():.3f})")

    # mean distance trajectory: the headline dynamic of the whole model
    L = min(min(len(r['d_trajectory']) for r in py),
            min(len(r['d_trajectory']) for r in cpp))
    Dp = np.vstack([r['d_trajectory'][:L] for r in py])
    Dc = np.vstack([r['d_trajectory'][:L] for r in cpp])
    worst_z, worst_t = 0.0, 0
    for t in (1, 5, 10, 20, 50, 100, 300, 1000, 3000, min(10000, L - 1)):
        if t >= L:
            continue
        se_t = np.sqrt(Dp[:, t].var() / len(Dp) + Dc[:, t].var() / len(Dc))
        z = (Dp[:, t].mean() - Dc[:, t].mean()) / se_t if se_t > 0 else 0.0
        if abs(z) > abs(worst_z):
            worst_z, worst_t = z, t
    check("mean distance trajectory", abs(worst_z) < 4,
          f"largest deviation z={worst_z:+.2f} at generation {worst_t} "
          f"(python={Dp[:, worst_t].mean():.3f} cpp={Dc[:, worst_t].mean():.3f})")

    # genetic variance contributed by the standing variation at the shift
    v_cpp = np.array([np.sum(2 * s['segregating']['a'] ** 2 * s['segregating']['x'] *
                             (1 - s['segregating']['x']))
                      for s in (r['snapshots']['initial'] for r in cpp)])
    print(f"     V_A(large) at the shift, cpp mean = {v_cpp.mean():.2f} "
          f"(4*2NU = {4 * n2u:.2f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true",
                    help="more replicates and both mutational inputs (slow: the "
                         "Python engine takes ~30s per 2NU=10 replicate)")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 8))
    args = ap.parse_args()

    print(f"validating the C++ core against simulation_classes.Simulation "
          f"({args.workers} workers)")
    check_samplers()
    check_total_n()
    check_standing(n_samples=600 if args.full else 300, workers=args.workers)
    check_standing_exact(n_samples=40000 if args.full else 20000)
    if args.full:
        check_replicates(n2u=0.01, n_reps=400, workers=args.workers)
        check_replicates(n2u=10.0, n_reps=100, workers=args.workers)
    else:
        check_replicates(n2u=0.01, n_reps=args.workers * 8, workers=args.workers)
        check_replicates(n2u=10.0, n_reps=args.workers * 2, workers=args.workers)

    print()
    if _failures:
        print(f"FAILED: {len(_failures)} check(s): {', '.join(_failures)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
