'''
Simulations with a single large-effect allele and an infinitesimal background of constant
magnitude \sigma^2. The allele has squared effect size S and sign +1 or -1.

Two modes:
- segregating: the allele is at frequency x0 when the optimum shifts by \Lambda. x0 is
  either specified or drawn from the stationary distribution under MSDB.
- new: the allele arises at time t0 after the shift, at frequency 1/(2N). t0 is either
  specified or uniform on [0, T], with T the time the distance is expected to reach one
  under Lande's equation.

Each generation the distance updates as \Delta D = -\sigma^2 D/V_S - 2*a*\Delta x, and
the frequency by Wright-Fisher sampling with
E(\Delta x) = x*(1-x)/V_S * (a*D + a^2*(x - 1/2)*(1 - D^2/V_S)) and variance
x*(1-x)/(2*N). V_S = 2*N. Runs end when the allele fixes or is lost.

Recorded: whether the allele fixed or was lost; optionally the frequency and distance
trajectories plus x0 (segregating) or t0 (new).

Sign convention follows simulation_classes.py, so the dynamics are adaptive: with D > 0
and a > 0 the allele is aligned with adaptation, a*D drives x up, and -2*a*dx pulls D
toward the optimum. The a^2*(x - 1/2) term is destabilizing selection against trait
variance.

The (1 - D^2/V_S) factor on that term matches
analytic_functions.get_recursion_deterministic and matters whenever D^2 is an appreciable
fraction of V_S: at N = 5000 and a shift of 100 it is exactly 1, switching the
destabilizing term off at the shift. Omitting it made escape from drift too hard and
pushed the simulated fixation probability below the analytic curve at large shifts.
'''

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import quad


@dataclass
class SimulationResult:
    """Outcome of a single single-site replicate.

    Attributes
    ----------
    fixed : bool
        True if the allele fixed, False if it was lost.
    n_generations : int
        Generations the allele segregated, from when it was present until
        absorption.
    initial_frequency : float
        Frequency when the allele first appeared.
    arise_time : int or None
        For new alleles, the generation t0 after the shift; None for segregating.
    x_trajectory, D_trajectory, time_trajectory : np.ndarray or None
        Frequency, distance and generation index over time, indexed from the shift;
        None unless the run was asked to record them. For new alleles the
        deterministic pre-arrival phase is included, so x_trajectory[t] == 0 for
        t < arise_time.
    """

    fixed: bool
    n_generations: int
    initial_frequency: float
    arise_time: Optional[int] = None
    x_trajectory: Optional[np.ndarray] = None
    D_trajectory: Optional[np.ndarray] = None
    time_trajectory: Optional[np.ndarray] = None

    @property
    def lost(self) -> bool:
        return not self.fixed


class SingleSiteSimulation:
    """Base class: one large-effect allele on a constant infinitesimal background.

    Holds the parameters and the shared Wright-Fisher-with-selection dynamics. The
    two scenarios in the module docstring are the subclasses SegregatingSimulation
    (allele present at the shift) and NewMutationSimulation (allele arises later);
    each supplies its own initial conditions via run().
    """

    def __init__(self, N, S, sign, sigma2, Lambda, seed=None):
        """Set up parameters and the RNG.

        Parameters
        ----------
        N : int
            Wright-Fisher population size; V_S is 2N.
        S : float
            Squared effect size, so the per-copy effect is a = sign * sqrt(S).
        sign : {+1, -1}
            Aligned with (+1) or opposing (-1) adaptation.
        sigma2 : float
            Genetic variance of the infinitesimal background.
        Lambda : float
            Size of the optimum shift, i.e. the initial distance D.
        seed : int or None
            Seed for the NumPy random Generator.
        """
        if sign not in (-1, 1, -1.0, 1.0):
            raise ValueError("sign must be +1 or -1")

        self.N = int(N)
        self.S = float(S)
        self.sign = 1 if sign > 0 else -1
        self.a = self.sign * np.sqrt(self.S)
        self.sigma2 = float(sigma2)
        self.Lambda = float(Lambda)
        self.V_S = 2.0 * self.N

        self.rng = np.random.default_rng(seed)

    def expected_delta_x(self, x, D):
        """Deterministic expected change in allele frequency.

        Directional selection toward the optimum (a*D) plus destabilizing selection
        against trait variance (a^2*(x - 1/2)), the latter weakened far from the
        optimum by (1 - D^2/V_S) and vanishing at D^2 = V_S. Matches
        analytic_functions.get_recursion_deterministic.
        """
        return x * (1.0 - x) / self.V_S * (
            self.a * D + self.a ** 2 * (x - 0.5) * (1.0 - D ** 2 / self.V_S))

    def _wright_fisher_frequency(self, x, D):
        p = x + self.expected_delta_x(x, D)
        p = min(max(p, 0.0), 1.0)
        return self.rng.binomial(2 * self.N, p) / (2.0 * self.N)

    def _update_distance(self, D, delta_x):
        # Lande: D relaxes via the background and is shifted by the focal allele
        return D - self.sigma2 * D / self.V_S - 2.0 * self.a * delta_x

    def _run_until_absorption(self, x0, D0, record=False):
        """Advance (x, D) one generation at a time until the allele fixes or is lost.

        Returns (fixed, n_generations, x_trajectory, D_trajectory); the trajectories
        include the initial state, and are None unless record=True.
        """
        x, D = float(x0), float(D0)
        xs = [x] if record else None
        Ds = [D] if record else None

        n = 0
        while 0.0 < x < 1.0:
            x_new = self._wright_fisher_frequency(x, D)
            D = self._update_distance(D, x_new - x)
            x = x_new
            n += 1
            if record:
                xs.append(x)
                Ds.append(D)

        fixed = x >= 1.0
        if record:
            return fixed, n, np.asarray(xs), np.asarray(Ds)
        return fixed, n, None, None

    # ------------------------------------------------------------------
    # Lande relaxation time used to draw the arise time of new alleles
    # ------------------------------------------------------------------
    def lande_time_to_distance_one(self):
        """Generations for D to decay from Lambda to 1 under the background alone.

        With no large-effect allele the distance follows D_t = Lambda * (1 - sigma2/V_S)^t
        (Lande's equation with delta_x = 0). T is the smallest t with D_t <= 1.
        """
        if self.Lambda <= 1.0:
            return 0
        else:
            t = np.log(self.Lambda) / (self.sigma2/self.V_S)  # exact real-valued solution
        return int(np.ceil(t))  # round up to the first integer generation with D <= 1

    # ------------------------------------------------------------------
    # Convenience: many replicates + summary
    # ------------------------------------------------------------------
    def run(self, record_trajectory=False, **kwargs):
        raise NotImplementedError("use SegregatingSimulation or NewMutationSimulation")

    def run_replicates(self, n_replicates, n_record=0, **run_kwargs):
        """Run n_replicates and return a list of SimulationResult.

        The first n_record replicates record their full trajectories (the "some
        simulations" of the docstring); the rest record only the fix/loss outcome.
        Extra keyword arguments are passed through to run() (e.g. x0 or t0).
        """
        return [
            self.run(record_trajectory=(i < n_record), **run_kwargs)
            for i in range(n_replicates)
        ]

    @staticmethod
    def fixation_probability(results):
        """Fraction of the given results in which the allele fixed."""
        if not results:
            return float("nan")
        return float(np.mean([r.fixed for r in results]))


class SegregatingSimulation(SingleSiteSimulation):
    """Scenario 1: the allele is already segregating when the optimum shifts.

    At the shift the distance jumps to Lambda and the allele sits at frequency x0
    (specified, or drawn from the mutation-selection-drift steady state).
    """

    def run(self, x0=None, record_trajectory=False):
        """Simulate one segregating-allele replicate.

        x0 : float or None
            Initial allele frequency. If None it is drawn from the steady-state
            density via sample_initial_frequency().
        """
        if x0 is None:
            x0 = self.sample_initial_frequency()
        x0 = float(x0)

        # distance starts at the size of the optimum shift
        fixed, n, xs, Ds = self._run_until_absorption(x0, self.Lambda, record_trajectory)
        times = np.arange(n + 1) if record_trajectory else None
        return SimulationResult(
            fixed=fixed,
            n_generations=n,
            initial_frequency=x0,
            arise_time=None,
            x_trajectory=xs,
            D_trajectory=Ds,
            time_trajectory=times,
        )

    # ------------------------------------------------------------------
    # Mutation-selection-drift steady state of the focal allele's frequency
    # ------------------------------------------------------------------
    
    def variance_star(self, x):
        return 2 * self.S * x * (1 - x)

    def folded_sojourn_time(self, x):
        if x < 0:
            raise ValueError
        elif x > 1 / 2:
            raise ValueError
        else:
            value = 2 * np.exp(-self.variance_star(x) / 2) / ((1 - x))
            if x <= 1 / (2 * self.N):
                return 2 * self.N *  value
            else:
                return value / x

    # total (unnormalized) sojourn mass over the folded frequency range [0, 1/2].
    # This is the *integral* of the density over [0, 1/2] (the CDF normalizer), not
    # the density at 1/2. It depends only on S and N, so compute it once and cache it.
    def _total_sojourn_time(self):
        if getattr(self, "_total_sojourn_cache", None) is None:
            self._total_sojourn_cache, _ = quad(
                self.folded_sojourn_time, 0, 1 / 2, points=[1 / (2 * self.N)]
            )
        return self._total_sojourn_cache

    # cumulative distribution function (CDF) of the steady-state (folded) density
    def sojourn_time_cdf(self, x):
        if x < 0:
            return 0.0
        elif x > 1 / 2:
            return 1.0
        else:
            # integral of the folded sojourn time from 0 to x, normalized by the
            # total over [0, 1/2] so the CDF runs from 0 to 1
            integral, _ = quad(self.folded_sojourn_time, 0, x, points=[1 / (2 * self.N)])
            return integral / self._total_sojourn_time()
            
            
    def sample_initial_frequency(self):
        """Draw x0 from the steady-state density
        """
        twoN = 2 * self.N
        
        # draw a random uniform value in [0, 1] to use for inverse transform sampling
        u = self.rng.random()
        
        # use a root-finding method to find the x0 such that CDF(x0) = u
        result = root_scalar(lambda x: self.sojourn_time_cdf(x) - u, bracket=[0, 1 / 2])
        if not result.converged:
            raise RuntimeError("Root finding for initial frequency did not converge.")
        return result.root
        


class NewMutationSimulation(SingleSiteSimulation):
    """Scenario 2: the allele arises t0 generations after the optimum shifts.

    Before t0 the allele is absent (x = 0) and the distance relaxes deterministically
    from Lambda under the background alone. At t0 the allele enters as a single copy
    (x = 1/(2N)) and the stochastic dynamics begin.
    """

    def run(self, t0=None, record_trajectory=False):
        """Simulate one new-allele replicate.

        t0 : int or None
            Generation (after the shift) at which the allele arises. If None it is
            drawn uniformly on [0, T], where T = lande_time_to_distance_one().
        """
        if t0 is None:
            t0 = int(self.rng.integers(0, self.lande_time_to_distance_one() + 1))
        t0 = int(t0)

        # deterministic decay of the distance over the pre-arrival phase
        D0 = self.Lambda * np.exp(-self.sigma2 / self.V_S * t0)
        x0 = 1.0 / (2.0 * self.N)

        fixed, n, xs, Ds = self._run_until_absorption(x0, D0, record_trajectory)

        if record_trajectory:
            # prepend the pre-arrival phase (x = 0, D relaxing from Lambda to D0) so
            # the trajectory is indexed from the shift at generation 0
            pre_x = np.zeros(t0)
            pre_D = self.Lambda * np.exp(-self.sigma2 / self.V_S * np.arange(t0))
            xs = np.concatenate([pre_x, xs])
            Ds = np.concatenate([pre_D, Ds])
            times = np.arange(t0 + n + 1)
        else:
            times = None

        return SimulationResult(
            fixed=fixed,
            n_generations=n,
            initial_frequency=x0,
            arise_time=t0,
            x_trajectory=xs,
            D_trajectory=Ds,
            time_trajectory=times,
        )


if __name__ == "__main__":
    N = 1000
    common = dict(N=N, S=10.0, sign=1, sigma2=1.0, Lambda=5.0, seed=0)

    seg = SegregatingSimulation(**common)
    seg_results = seg.run_replicates(2000, n_record=1)
    print(
        "segregating: P(fix) = {:.3f}, example x0 = {:.4f}".format(
            SingleSiteSimulation.fixation_probability(seg_results),
            seg_results[0].initial_frequency,
        )
    )

    new = NewMutationSimulation(**common)
    print("Lande time T to reach D = 1:", new.lande_time_to_distance_one())
    new_results = new.run_replicates(2000, n_record=1)
    print(
        "new:         P(fix) = {:.3f}, example t0 = {}".format(
            SingleSiteSimulation.fixation_probability(new_results),
            new_results[0].arise_time,
        )
    )
