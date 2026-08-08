"""
Forward-in-time simulation of polygenic adaptation to a shifted phenotypic optimum.

A quantitative trait is under stabilizing selection toward an optimum that is suddenly
displaced at the start of the run. The state variable `d` is the signed distance from the
population mean to the optimum: it begins at `shift` and relaxes toward 0.

Adaptation comes from two sources: large-effect alleles, tracked as vectorized arrays
(frequency `x`, signed effect `a`) evolving under deterministic selection plus binomial
drift in a Wright-Fisher population of size N; and an infinitesimal background summarised
by the decay term -sigma2 * d / (2N).

Effect sizes are a = sqrt(S) with S drawn from `sdist`, then given a random sign. The
deterministic per-generation change is

    dx = a/(2N) * (d - a*(1/2 - x)) * x * (1 - x),

combining directional selection toward the optimum with stabilizing selection against
trait variance. The standing variation at the shift is sampled from the
mutation-selection-drift steady state, whose folded density is `folded_sojourn_time`.

Outputs (`self.stats`)
    'fixations'      : one Mutation record per fixed allele (.a, .t_initial)
    'd_trajectory'   : distance d, every generation
    'second_moment' / 'third_moment' : trait moments, if record_moments
    'recorded_trajectories' / 'snapshots' : per-allele trajectories and allele-state
                       snapshots, if full_output (see Simulation.__init__)

`generate_alleles` is imported from generate_segregating_mutations; it returns (x, S, y0)
tuples for the standing-variation alleles, using precomputed lookup files.
"""

from collections import deque

import numpy as np
from scipy.integrate import quad
from generate_segregating_mutations import generate_alleles


class Simulation:
    """Drives a single replicate of polygenic adaptation to a shifted optimum.

    The trait mean starts a distance `shift` from the optimum (d = shift) and
    relaxes toward 0. Large-effect alleles are tracked explicitly as parallel
    NumPy arrays; the infinitesimal background is a single decay term on d.
    """

    # under full_output, alleles lost below this frequency are discarded
    TRAJECTORY_RECORD_THRESHOLD = 0.01

    def __init__(self, N, sdist, N2U, sigma2, shift, tracking_time = 50, stop_time=1e4, seed=None, record_moments=False, full_output=False):
        """Set up parameters, the empty stat containers, and the RNG.

        Parameters
        ----------
        N : int
            Wright-Fisher population size.
        sdist : scipy.stats distribution
            Distribution of scaled selection coefficients S; an allele's effect
            size is a = sqrt(S) (sign assigned later, per allele).
        N2U : float
            2NU, the population-scaled per-generation mutation influx for the
            explicitly tracked large-effect alleles. This is the TOTAL influx
            over both sign classes: a Poisson(2NU) number of new alleles arrives
            each generation and each is independently aligned with or opposing
            the shift, so the aligned influx is 2NU/2.
        sigma2 : float
            Background (infinitesimal) genetic variance; sets the strength of
            the stabilizing-selection pull on d each generation.
        shift : float
            Size of the sudden optimum displacement, i.e. the initial value of d.
        tracking_time : int
            Generation cutoff defining the "standing variation": alleles that
            arose before this time are the ones whose extinction can be tracked.
        stop_time : float
            Minimum number of generations to run (the loop also keeps going
            while |d| > 1).
        seed : int or None
            Seed for the NumPy random Generator. None gives non-deterministic
            behaviour; pass an int for reproducible runs.
        record_moments : bool
            If True, record the trait's second and third moments on the
            recording schedule. Off by default since fixations are the usual
            output of interest.
        full_output : bool
            If True, additionally record (expensive, for small runs): the full
            frequency trajectory of every allele that fixes or ever reaches
            frequency TRAJECTORY_RECORD_THRESHOLD (alleles lost below it are
            discarded), and -- via snapshots at the shift ('initial'), generations
            20 and 300, the onset of the quasi-static phase, and the end of the run
            ('final') -- the full list of segregating and fixed alleles at those
            times. Requires per-allele identity tracking, so it is off by default.
        """

        # parameters
        self.N = N
        self.sdist = sdist
        self.N2U = N2U
        self.sigma2 = sigma2
        self.shift = shift
        self.stop_time = stop_time
        self.tracking_time = tracking_time
        self.record_moments = record_moments
        self.full_output = full_output
        self.all_segregating_extinct = False
        self.integral_distance = 0

        self.rng = np.random.default_rng(seed)

        # parallel arrays, one entry per segregating allele: frequency, signed
        # effect, birth generation, frequency when the allele first appeared, and a
        # stable id so full_output can follow alleles across the pruning that
        # reorders the arrays.
        self.x = np.empty(0)
        self.a = np.empty(0)
        self.t_initial = np.empty(0)
        self.x_initial = np.empty(0)
        self.ids = np.empty(0, dtype=int)
        self._next_id = 0

        # 'recorded_trajectories' and 'snapshots' are only filled when full_output
        self.stat_names = ['d_trajectory','fixations','second_moment','third_moment',
                           'recorded_trajectories']
        self.stats = {}
        for name in self.stat_names:
            self.stats[name] = []
        self.stats['snapshots'] = {}
        # live trajectories keyed by id (full_output only), finalized into
        # 'recorded_trajectories' when the allele fixes, or when it is lost/the run
        # ends having reached TRAJECTORY_RECORD_THRESHOLD. _reached holds those ids.
        self._traj = {}
        self._reached = set()

        # First generation at which the 10-generation sliding-window means of D, V
        # and u3 satisfy |[D_w - u3_w/(2 V_w)] / D_w| < 0.05, and D_w there.
        self.convergence_time = np.nan
        self.convergence_D_window = np.nan

    def variance_star(self, S, x):
        # scaled per-locus variance contribution, used by the density below
        return 2 * S * x * (1 - x)

    def folded_sojourn_time(self, S, x):
        # Expected time an allele of effect S spends near frequency x at MSDB,
        # folded so x and 1-x are combined; only x in [0, 1/2] is valid.
        if x < 0:
            raise ValueError
        elif x > 1 / 2:
            raise ValueError
        else:
            value = 2 * np.exp(-self.variance_star(S=S, x=x) / 2) / (x * (1 - x))
            # boundary correction keeping the density integrable as x -> 0
            if x <= 1 / (2 * self.N):
                return 2 * self.N * x * value
            else:
                return value

    def total_n(self):
        # Sojourn time over all frequencies for a given S (1/(2N) is flagged to help
        # the quadrature), averaged over sdist.
        a = quad(
            lambda S: quad(lambda x: self.folded_sojourn_time(S=S, x=x), 0, 1 / 2, points=[1 / (2 * self.N)])[
                          0] * self.sdist.pdf(S), self.sdist.ppf(0.0000001), self.sdist.ppf(0.99999))[0]
        # N2U is the influx over both sign classes, so this count covers both
        return a * self.N2U

    # k stable unique ids, so alleles can be followed across pruning
    def _assign_ids(self, k):
        ids = np.arange(self._next_id, self._next_id + k, dtype=int)
        self._next_id += k
        if self.full_output:
            for i in ids:
                self._traj[int(i)] = []
        return ids

    def initiate_mutations(self, n):
        """Seed the standing variation present the moment the optimum shifts.

        `n` is the expected number of segregating sites (from total_n); a Poisson
        count is drawn and each allele sampled from the steady state.
        """
        # The number of alleles is poisson distributed around the expectation; n
        # already covers both sign classes (2NU is the total mutational input),
        # and the aligned/opposing split is made by the random signs below.
        n_realized = self.rng.poisson(n)
        # We need the probability of segregating for the next step of generating the effect sizes
        prob_segregating = quad(lambda S: quad(lambda x: self.folded_sojourn_time(S=S,x=x),0,1/2,points=[1/(2*self.N)])[0]*self.sdist.pdf(S),self.sdist.ppf(0),self.sdist.ppf(0.9999))[0]
        # get the effect sizes. I offloaded all this into a seperate file, because I use some precalculated files
        # Returns a list of (x, S, y0): frequency, scaled effect, and steady-state quantile (y0 is unused here).
        # rng=self.rng keeps the standing-variation sampling tied to the seeded Generator.
        chosen_muts = generate_alleles(n=n_realized,prob_segregating=prob_segregating,N=self.N,sdist=self.sdist,rng=self.rng)

        # load the variants into the state arrays. Effect size a = sqrt(S) with a
        # random +/-1 sign per allele (aligned vs. opposing adaptation).
        chosen = np.asarray(chosen_muts, dtype=float)
        if chosen.size == 0:
            self.x = np.empty(0)
            self.a = np.empty(0)
            self.t_initial = np.empty(0)
            self.x_initial = np.empty(0)
            self.ids = np.empty(0, dtype=int)
            return

        x, S = chosen[:, 0], chosen[:, 1]
        signs = self.rng.choice([-1.0, 1.0], size=len(S))
        self.x = x.copy()
        self.a = np.sqrt(S) * signs
        self.t_initial = np.zeros(len(S))
        # standing variation's initial frequency is its sampled frequency at the shift
        self.x_initial = x.copy()
        self.ids = self._assign_ids(len(S))

    # add new mutations each generation and keep track of how much this changes the mean phenotype.
    # we assume new mutations are poisson distributed around 2NU
    # given that 2NU is parameterized as the total 2NU for large effect alleles
    # (aligned and opposing together; the sign is drawn per allele below)
    def add_new_mutations(self, t):
        """Introduce this generation's new mutations and report their phenotypic effect.

        A Poisson(2NU) number of new alleles each enter as a single copy
        (x = 1/(2N)); 2NU counts both sign classes, so on average half are
        aligned with the shift. Returns the total change in mean phenotype they
        contribute.
        """
        twoN = 2 * self.N
        k = self.rng.poisson(self.N2U)

        # New alleles: effect a = sqrt(S) with a random +/-1 sign, each starting
        # at a single copy. (All array ops below are no-ops when k == 0.)
        # random_state=self.rng keeps sdist sampling tied to the seeded Generator.
        a_new = np.sqrt(self.sdist.rvs(size=k, random_state=self.rng)) * self.rng.choice([-1.0, 1.0], size=k)
        x_new = np.full(k, 1.0 / twoN)

        # Each new allele's contribution to the mean phenotype = 2 * a * (1/2N).
        change_in_d = (2 * a_new * x_new).sum()

        # append the new alleles to the state arrays (new alleles start at 1/(2N))
        self.x = np.concatenate([self.x, x_new])
        self.a = np.concatenate([self.a, a_new])
        self.t_initial = np.concatenate([self.t_initial, np.full(k, t)])
        self.x_initial = np.concatenate([self.x_initial, x_new])
        self.ids = np.concatenate([self.ids, self._assign_ids(k)])

        return change_in_d

    # update the distance based on the change from large effect alleles and from background. Store the distance every generation
    def update_distance_trajectory(self, d, change_in_d):
        # New distance = old distance, minus the mean shift from tracked alleles
        # (change_in_d), minus the infinitesimal background pull toward the
        # optimum (sigma2 * d / 2N). Recorded every generation.
        d += -change_in_d - self.sigma2 * d / (2 * self.N)
        self.stats['d_trajectory'].append(d)
        return d

    def update_mutation_frequencies(self, d, t):
        """Advance every tracked allele one generation and tally the results.

        Applies deterministic selection + binomial drift to the whole frequency
        array at once, returns the total change in the mean phenotype, records any
        fixations, and prunes fixed/lost alleles. `t` timestamps the per-allele
        trajectories under full_output.
        """
        twoN = 2 * self.N

        det_dx = self.a / twoN * (d - self.a * (0.5 - self.x)) * self.x * (1 - self.x)
        exp_x = np.clip(self.x + det_dx, 0.0, 1.0)
        # one Wright-Fisher binomial draw supplies drift for every allele
        new_x = self.rng.binomial(twoN, exp_x) / twoN

        change_in_d = (2 * self.a * (new_x - self.x)).sum()

        self.x = new_x

        # x_initial is kept so the contribution 2*a*(1-x_initial) needs no
        # assumption that x_initial = 1/(2N)
        fixed_idx = np.nonzero(self.x >= 1.0)[0]
        for i in fixed_idx:
            self.stats['fixations'].append(
                Mutation(a=float(self.a[i]), t=float(self.t_initial[i]),
                         x0=float(self.x_initial[i])))

        if self.full_output:
            thr = self.TRAJECTORY_RECORD_THRESHOLD
            for j, allele_id in enumerate(self.ids):
                aid = int(allele_id)
                xj = float(self.x[j])
                self._traj[aid].append((t, xj))
                if xj >= thr:
                    self._reached.add(aid)
            # finalize the trajectory of each allele that fixed (always recorded)
            for i in fixed_idx:
                self._record_trajectory(i, fate='fixed', fixation_time=t)
            # alleles lost this generation: record if they ever reached the threshold,
            # otherwise discard the trajectory to save space
            for i in np.nonzero(self.x <= 0.0)[0]:
                aid = int(self.ids[i])
                if aid in self._reached:
                    self._record_trajectory(i, fate='extinct', fixation_time=np.nan)
                else:
                    self._traj.pop(aid, None)

        # Keep only still-segregating alleles (drop fixed x>=1 and lost x<=0).
        keep = (self.x > 0.0) & (self.x < 1.0)
        self.x = self.x[keep]
        self.a = self.a[keep]
        self.t_initial = self.t_initial[keep]
        self.x_initial = self.x_initial[keep]
        self.ids = self.ids[keep]

        return change_in_d

    # determine if all mutations that arose before the cutoff time have gone extinct
    def check_all_segregating_extinct(self):
        # True once no live allele predates tracking_time (i.e. the original
        # standing variation is fully resolved).
        self.all_segregating_extinct = not np.any(self.t_initial < self.tracking_time)
        return

    # `i` indexes the still un-pruned state arrays
    def _record_trajectory(self, i, fate, fixation_time):
        aid = int(self.ids[i])
        self._reached.discard(aid)
        self.stats['recorded_trajectories'].append({
            'id': aid,
            'a': float(self.a[i]),
            't_initial': float(self.t_initial[i]),
            'x0': float(self.x_initial[i]),
            'fate': fate,
            'fixation_time': fixation_time,
            'trajectory': np.array(self._traj.pop(aid)),
        })

    # `d` is stored alongside so a snapshot needs no lookup into 'd_trajectory'
    def _take_snapshot(self, t, key, d):
        fixations = self.stats['fixations']
        self.stats['snapshots'][key] = {
            'time': t,
            'D': d,
            'segregating': {
                'id': self.ids.copy(),
                'a': self.a.copy(),
                'x': self.x.copy(),
                't_initial': self.t_initial.copy(),
                'x0': self.x_initial.copy(),
            },
            'fixed': {
                'a': np.array([m.a for m in fixations]),
                't_initial': np.array([m.t_initial for m in fixations]),
                'x0': np.array([m.x0 for m in fixations]),
            },
        }

    # main simulation loop: each generation update existing alleles (pruning fixed/lost
    # ones), add new mutations, update the distance d, and periodically record stats.
    # runs until at least stop_time generations have passed AND |d| <= 1.
    def recursion(self):

        d = self.shift
        self.stats['d_trajectory'] = [d]

        # the standing variation at the moment of the shift, before any generation
        # has elapsed (the pre-shift steady state)
        if self.full_output:
            self._take_snapshot(t=0, key='initial', d=d)

        # 10-generation sliding windows of the distance D and the phenotypic
        # moments V (variance, incl. the background sigma2 via second_moment) and
        # u3 (third moment), used to flag the first generation at which
        # |[D_w - u3_w/(2 V_w)] / D_w| < 0.05. Only maintained until the criterion
        # is first met (self.convergence_time still NaN).
        win = 10
        D_buf, V_buf, u3_buf = deque(maxlen=win), deque(maxlen=win), deque(maxlen=win)

        t = 0
        while t < self.stop_time or abs(d) > 1:# or not self.all_segregating_extinct:

            # update allele frequencies and remove mutations that went extinct or fixed.
            # fixations are recorded inside update_mutation_frequencies
            change_in_d = self.update_mutation_frequencies(d=d, t=t)

            change_in_d += self.add_new_mutations(t=t)
            d = self.update_distance_trajectory(d=d, change_in_d=change_in_d)

            # phenotypic-convergence criterion on 10-generation window means
            if np.isnan(self.convergence_time):
                D_buf.append(d)
                V_buf.append(self.second_moment())
                u3_buf.append(self.third_moment())
                if len(D_buf) == win:
                    D_w = np.mean(D_buf)
                    V_w = np.mean(V_buf)
                    u3_w = np.mean(u3_buf)
                    if D_w != 0 and V_w != 0 and abs((D_w - u3_w / (2 * V_w)) / D_w) < 0.05:
                        self.convergence_time = t
                        self.convergence_D_window = float(D_w)
                        if self.full_output:
                            self._take_snapshot(t, 'quasi_static', d)

            if self.full_output:
                if t == 20:
                    self._take_snapshot(t, 'gen20', d)
                elif t == 300:
                    self._take_snapshot(t, 'gen300', d)

            # moments on a schedule that samples densely early, sparsely later
            if self.record_moments and (
                    t < 100 or (t < 500 and t % 5 == 0) or (t < 1000 and t % 10 == 0) or t % 50 == 0):
                self.update_second_moment()
                self.update_third_moment()

            t += 1

        # the run only ends once |d| <= 1, so 'final' is the equilibrated population
        if self.full_output:
            self._take_snapshot(t=t, key='final', d=d)
            for j in range(len(self.ids)):
                if int(self.ids[j]) in self._reached:
                    self._record_trajectory(j, fate='segregating', fixation_time=np.nan)
            self._traj.clear()
            self._reached.clear()

    # Some functions for defining different metrics
    # total phenotypic variance: the genetic variance from the tracked large-effect
    # alleles (sum of 2a^2 x(1-x)) plus the constant background variance sigma2.
    def second_moment(self):
        return np.sum(2 * self.a ** 2 * self.x * (1 - self.x)) + self.sigma2

    # total third moment / skew from all segregating alleles (sum of 2a^3 x(1-x)(1-2x)).
    # The background variance is symmetric, so it contributes nothing to the third moment.
    def third_moment(self):
        return np.sum(2 * self.a ** 3 * self.x * (1 - self.x) * (1 - 2 * self.x))

    def update_second_moment(self):
        self.stats['second_moment'].append(self.second_moment())

    def update_third_moment(self):
        self.stats['third_moment'].append(self.third_moment())

    # This initializes the simulation and then runs it
    def run_simulation(self):
        # expected number of standing-variation sites, then seed them, then run
        n = self.total_n()
        self.initiate_mutations(n=n)
        self.recursion()

# The second class is a lightweight record for a single fixed allele
class Mutation:
    """A fixed allele recorded in stats['fixations'].

    With the per-allele dynamics now vectorized on the Simulation, this class is
    only a small data holder. It captures the things needed to summarise
    fixations: the signed effect size (.a), the generation the allele arose
    (.t_initial), and its initial frequency (.x0). The number of fixations is
    len(stats['fixations']); the effect direction, if needed, is np.sign(.a).
    """

    def __init__(self, a, t, x0):
        self.a = a                  # signed effect size of the fixed allele
        self.t_initial = t          # generation the allele arose
        self.x0 = x0                # frequency when the allele first appeared
