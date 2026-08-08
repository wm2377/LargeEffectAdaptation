"""Vectorised all-allele simulations.
"""

import pickle
import sys

import numpy as np
from scipy import stats
from scipy.integrate import quad

# Frequencies below this are treated as 0 / above 1 - this as 1 only through the binomial
# draw itself; no separate epsilon is applied, matching the original's exact == comparisons.
__all__ = [
    "EffectSizeDistribution", "AllelePopulation", "AllAlleleSimulation",
    "variance_star", "sojourn_time", "draw_initial_frequencies",
    "expected_segregating_per_input", "expected_variance_per_input",
    "expected_variance_given_a2", "expected_segregating_given_a2",
]


# --------------------------------------------------------------------------------------
# effect-size distribution
# --------------------------------------------------------------------------------------

class EffectSizeDistribution:
    """Mixture of a small-effect and a large-effect distribution of S = a^2.

    `weight` is the probability a new mutation is drawn from the large-effect component,
    i.e. the large-effect share of the total mutational input.
    """

    def __init__(self, weight, distribution_type="expon"):
        self.distribution_type = distribution_type
        self.weight = weight
        self.small_dist = stats.expon(scale=1)
        if distribution_type == "expon":
            self.large_dist = stats.expon(scale=400, loc=100)
        elif distribution_type == "uniform":
            self.large_dist = stats.uniform(loc=100, scale=900)
        else:
            raise ValueError(f"Unsupported distribution type: {distribution_type!r}")

    def rvs(self, size, rng):
        """`size` draws of S = a^2 from the mixture."""
        is_large = rng.random(size) <= self.weight
        out = np.empty(size, dtype=float)
        n_large = int(is_large.sum())
        if n_large:
            out[is_large] = self.large_dist.rvs(size=n_large, random_state=rng)
        if n_large < size:
            out[~is_large] = self.small_dist.rvs(size=size - n_large, random_state=rng)
        return out

    def pdf(self, a2):
        return (1 - self.weight) * self.small_dist.pdf(a2) + self.weight * self.large_dist.pdf(a2)


# --------------------------------------------------------------------------------------
# steady-state (sojourn time) quantities -- unchanged from all_allele_classes.py
# --------------------------------------------------------------------------------------

def variance_star(a2, x):
    """Phenotypic variance contributed by an allele of squared effect a2 at frequency x."""
    return 2 * a2 * x * (1 - x)


def sojourn_time(N, a2, x):
    """Expected time an allele of squared effect a2 spends at frequency x (unnormalised).
    """
    v = 2 * np.exp(-variance_star(a2, x) / 2) / (1 - x)
    return v * np.where(np.asarray(x) < 1 / (2 * N), 2 * N, 1 / np.maximum(x, 1e-300))


def expected_segregating_given_a2(N, a2):
    """Expected number of segregating alleles of squared effect a2, per unit mutational input."""
    return quad(lambda x: sojourn_time(N=N, a2=a2, x=x), 0, 1 / 2, points=[1 / (2 * N)])[0]


def expected_variance_given_a2(N, a2):
    """Expected phenotypic variance from alleles of squared effect a2, per unit input."""
    return quad(lambda x: variance_star(a2=a2, x=x) * sojourn_time(N=N, a2=a2, x=x),
                0, 1 / 2, points=[1 / (2 * N)])[0]


def _mixture_quad(fn, effect_size_distribution):
    """Integrate fn(a2) * mixture pdf over a2, splitting at the components' tails.
    """
    d = effect_size_distribution
    return quad(lambda a2: fn(a2) * d.pdf(a2), 0, 10000,
                points=[d.small_dist.ppf(0.9999), d.large_dist.ppf(0.001),
                        d.small_dist.ppf(0.9999)])[0]


def expected_segregating_per_input(N, effect_size_distribution):
    return _mixture_quad(lambda a2: expected_segregating_given_a2(N=N, a2=a2),
                         effect_size_distribution)


def expected_variance_per_input(N, effect_size_distribution):
    return _mixture_quad(lambda a2: expected_variance_given_a2(N=N, a2=a2),
                         effect_size_distribution)


def draw_initial_frequencies(N, a2, rng, n_grid=2048):
    """Frequencies for alleles of squared effects `a2`, drawn from the sojourn-time density.
    """
    a2 = np.atleast_1d(np.asarray(a2, dtype=float))
    x_entry = 1 / (2 * N)
    # a few points below the kink, then log-spaced up to 1/2
    grid = np.concatenate([
        np.linspace(0, x_entry, 16, endpoint=False),
        np.geomspace(x_entry, 0.5, n_grid),
    ])
    # density (n_alleles, n_grid); sojourn_time broadcasts over the a2 axis
    dens = sojourn_time(N=N, a2=a2[:, None], x=grid[None, :])
    # cumulative trapezoid along the frequency axis
    widths = np.diff(grid)
    cdf = np.concatenate([
        np.zeros((a2.size, 1)),
        np.cumsum((dens[:, 1:] + dens[:, :-1]) / 2 * widths, axis=1),
    ], axis=1)
    cdf /= cdf[:, -1:]
    p = rng.random(a2.size)
    # per-allele inverse CDF; searchsorted needs a loop over rows but it is O(log n) each
    out = np.empty(a2.size)
    for i in range(a2.size):
        out[i] = np.interp(p[i], cdf[i], grid)
    return out


# --------------------------------------------------------------------------------------
# population
# --------------------------------------------------------------------------------------

class AllelePopulation:
    """Segregating alleles as parallel numpy arrays (signed effect a, squared effect a2, x)."""

    def __init__(self, N, effect_size_distribution, optimum, mutation_rate, rng):
        self.N = int(N)
        self.Vs = 2 * self.N
        self.optimum = float(optimum)
        self.mutation_rate = float(mutation_rate)
        self.effect_size_distribution = effect_size_distribution
        self.rng = rng

        self.a = np.empty(0, dtype=float)
        self.a2 = np.empty(0, dtype=float)
        self.x = np.empty(0, dtype=float)

        self.fixed_background = 0.0
        self.fixations = []        # signed effect size of every allele that ever fixed
        self.new_fixations = []    # fixations since the last generate_output()

    # -- state ---------------------------------------------------------------------
    def mean_phenotype(self):
        return self.fixed_background + float(np.sum(2 * self.a * self.x))

    # -- one generation ------------------------------------------------------------
    def expected_change(self, distance):
        """E[dx] for every segregating allele (the model equation; see module docstring)."""
        return (self.a / self.Vs
                * (distance - self.a * (0.5 - self.x) * (1 - distance ** 2 / self.Vs))
                * self.x * (1 - self.x))

    def update_frequencies(self, distance):
        if self.a.size == 0:
            return
        p = self.x + self.expected_change(distance)
        # The original passed p straight to np.random.binomial, which raises for p outside
        # [0, 1]; clipping is a no-op whenever that would not have happened.
        np.clip(p, 0.0, 1.0, out=p)
        self.x = self.rng.binomial(2 * self.N, p) / (2 * self.N)

    def handle_fixed_or_extinct(self):
        fixed = self.x >= 1.0
        if fixed.any():
            for a in self.a[fixed]:
                self.fixations.append(float(a))
                self.new_fixations.append(float(a))
            self.fixed_background += float(np.sum(2 * self.a[fixed]))
        keep = (~fixed) & (self.x > 0.0)
        if not keep.all():
            self.a, self.a2, self.x = self.a[keep], self.a2[keep], self.x[keep]

    def add_new_mutations(self):
        n_new = self.rng.poisson(2 * self.mutation_rate * self.N)
        if not n_new:
            return
        a2 = self.effect_size_distribution.rvs(n_new, self.rng)
        signs = self.rng.choice([-1.0, 1.0], size=n_new)
        self.a = np.concatenate([self.a, np.sqrt(a2) * signs])
        self.a2 = np.concatenate([self.a2, a2])
        self.x = np.concatenate([self.x, np.full(n_new, 1 / (2 * self.N))])

    def next_generation(self):
        """Advance one generation and return this generation's recorded output."""
        distance = self.optimum - self.mean_phenotype()
        self.update_frequencies(distance)
        self.handle_fixed_or_extinct()
        self.add_new_mutations()
        return self.generate_output()

    def generate_output(self):
        """The per-generation record, in the exact format the processing scripts expect."""
        out = {
            "mutations": list(zip(self.a.tolist(), self.x.tolist())),
            "fixed_background": self.fixed_background,
            "fixations": list(self.new_fixations),
            "mean_phenotype": self.mean_phenotype(),
        }
        self.new_fixations = []
        return out

    def force_minor_alleles(self):
        """Relabel every allele so it is the minor one (x <= 1/2), flipping its sign."""
        major = self.x > 0.5
        if major.any():
            self.x[major] = 1 - self.x[major]
            self.a[major] = -self.a[major]


# --------------------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------------------

class AllAlleleSimulation:
    """Burn in at a stationary optimum, shift the optimum, then record the response."""

    def __init__(self, N, distribution_type, optimum, shift, mutation_rate, burn_time,
                 weight, output_file=None, seed=None, flush_every=10):
        self.N = int(N)
        self.shift = float(shift)
        self.burn_time = int(burn_time)
        self.output_file = output_file
        self.flush_every = int(flush_every)
        self.rng = np.random.default_rng(seed)

        self.effect_size_distribution = EffectSizeDistribution(
            weight=weight, distribution_type=distribution_type)
        self.population = AllelePopulation(
            N=N, effect_size_distribution=self.effect_size_distribution, optimum=optimum,
            mutation_rate=mutation_rate, rng=self.rng)

        self.history = {}
        self.n_seg_per_mutational_input = None
        self.variance_per_mutational_input = None

    # -- setup ---------------------------------------------------------------------
    def calculate_expected_metrics(self):
        self.n_seg_per_mutational_input = expected_segregating_per_input(
            N=self.N, effect_size_distribution=self.effect_size_distribution)
        self.variance_per_mutational_input = expected_variance_per_input(
            N=self.N, effect_size_distribution=self.effect_size_distribution)
        return self.n_seg_per_mutational_input, self.variance_per_mutational_input

    def initialize_population(self):
        """Seed the population at its expected steady state, so burn-in starts close to it."""
        if self.n_seg_per_mutational_input is None:
            self.calculate_expected_metrics()
        pop = self.population
        n_seg = self.n_seg_per_mutational_input * pop.mutation_rate * 2 * pop.N
        n = self.rng.poisson(n_seg)
        if not n:
            return
        a2 = self.effect_size_distribution.rvs(n, self.rng)
        signs = self.rng.choice([-1.0, 1.0], size=n)
        pop.a2 = a2
        pop.a = np.sqrt(a2) * signs
        pop.x = draw_initial_frequencies(N=pop.N, a2=a2, rng=self.rng)

    # -- phases --------------------------------------------------------------------
    def burn(self, progress_every=1000):
        """Run to (approximate) stationarity at the pre-shift optimum, recording nothing."""
        for t in range(-self.burn_time, 1):
            self.population.next_generation()
            if progress_every and t % progress_every == 0:
                print(f"  burn-in generation {t}", flush=True)
        self.history[0] = self.population.generate_output()

    def shift_optimum(self):
        self.population.optimum += self.shift

    def force_minor_alleles(self):
        self.population.force_minor_alleles()

    def run(self, generations, progress_every=2000):
        """Record `generations` generations after the shift, streaming to output_file."""
        for t in range(1, int(generations)):
            self.history[t + 1] = self.population.next_generation()
            if progress_every and t % progress_every == 0:
                print(f"  generation {t} ({self.population.a.size} segregating)", flush=True)
            if len(self.history) > self.flush_every:
                self.flush()
        self.flush()   # the original left the final partial batch unwritten

    def flush(self):
        """Append the buffered generations to output_file (a stream of pickled dicts)."""
        if not self.history:
            return
        if self.output_file is not None:
            with open(self.output_file, "ab") as fh:
                pickle.dump(self.history, fh)
        self.history = {}
