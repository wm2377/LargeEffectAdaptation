"""
A lightweight finite mixture of scipy.stats frozen distributions.

This lives in its own importable module (rather than inline in the Snakefile) so an
instance can be pickled by Snakemake into a script job and then shipped on to the
ProcessPoolExecutor workers in simulation_run.py -- pickling needs the class to be
importable by a stable module path.

It implements the subset of the scipy distribution interface the simulation and the
standing-variation sampler use -- pdf, cdf, ppf, rvs -- plus mean/var/std/support for
convenience. Sampling (rvs) draws a component per variate (exact, no inversion); the
quantile (ppf) inverts the mixture cdf numerically.

NOTE: the analytic pipeline (build_min_shift_lookup.py) additionally inspects the
scipy-frozen attributes .dist/.args/.kwds, which a mixture does not have. Those code
paths are not needed for the simulations and will be handled separately when the
analytics are wired up for the mixture.
"""

import numpy as np
from scipy.optimize import brentq


class MixtureDistribution:
    """Weighted mixture of frozen scipy.stats distributions `components`."""

    def __init__(self, components, weights):
        if len(components) != len(weights):
            raise ValueError("components and weights must have the same length")
        self.components = list(components)
        w = np.asarray(weights, dtype=float)
        if np.any(w < 0) or w.sum() <= 0:
            raise ValueError("weights must be non-negative and sum to a positive value")
        self.weights = w / w.sum()

    # ── support ────────────────────────────────────────────────────────────── #
    def support(self):
        los = [c.support()[0] for c in self.components]
        his = [c.support()[1] for c in self.components]
        return min(los), max(his)

    # ── density / distribution / survival ──────────────────────────────────── #
    def pdf(self, x):
        return sum(w * c.pdf(x) for w, c in zip(self.weights, self.components))

    def cdf(self, x):
        return sum(w * c.cdf(x) for w, c in zip(self.weights, self.components))

    def sf(self, x):
        return 1.0 - self.cdf(x)

    # ── quantile (numeric inversion of the mixture cdf) ─────────────────────── #
    def ppf(self, q):
        q_arr = np.atleast_1d(np.asarray(q, dtype=float))
        lo, hi_support = self.support()
        out = np.empty_like(q_arr)
        for i, qi in enumerate(q_arr):
            if qi <= 0.0:
                out[i] = lo
            elif qi >= 1.0:
                out[i] = hi_support
            else:
                # the largest component quantile is a valid upper bracket: the mixture
                # cdf there is >= qi (it equals qi for the argmax component and is >= qi
                # for the others), while cdf(lo) == 0 <= qi.
                hi = max(c.ppf(qi) for c in self.components)
                out[i] = lo if hi <= lo else brentq(lambda z: self.cdf(z) - qi, lo, hi,
                                                     xtol=1e-10, rtol=1e-12)
        return out[0] if np.ndim(q) == 0 else out

    # ── sampling ────────────────────────────────────────────────────────────── #
    def rvs(self, size=1, random_state=None):
        rng = (random_state if isinstance(random_state, np.random.Generator)
               else np.random.default_rng(random_state))
        n = int(np.prod(size)) if not np.isscalar(size) else int(size)
        which = rng.choice(len(self.components), size=n, p=self.weights)
        out = np.empty(n, dtype=float)
        for j, c in enumerate(self.components):
            mask = which == j
            m = int(mask.sum())
            if m:
                out[mask] = c.rvs(size=m, random_state=rng)
        return out.reshape(size) if not np.isscalar(size) else out

    # ── moments ──────────────────────────────────────────────────────────────── #
    def mean(self):
        return float(sum(w * c.mean() for w, c in zip(self.weights, self.components)))

    def var(self):
        m = self.mean()
        second = sum(w * (c.var() + c.mean() ** 2) for w, c in zip(self.weights, self.components))
        return float(second - m ** 2)

    def std(self):
        return float(np.sqrt(self.var()))
