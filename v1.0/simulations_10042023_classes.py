import numpy as np
import sys
import scipy.stats as stats
from scipy.integrate import quad
import pickle
from scipy.special import erf
from scipy.optimize import root
from copy import deepcopy as dc
from generate_segregating_mutations import generate_alleles

# Defines the classes for the simplified all-allele simulations

# The first class is the simulation itself
class Simulation:

    def __init__(self, N, sdist, N2U, sigma2, shift, tracking_time = 50, output_full = True, store_mutation_trajectories=False,stop_time=1e4):

        # parameters
        self.N = N
        self.sdist = sdist
        self.N2U = N2U
        self.sigma2 = sigma2
        self.shift = shift
        self.stop_time = stop_time
        self.tracking_time = tracking_time
        self.all_segregating_extinct = False
        self.integral_distance = 0

        # mandatory storage
        self.mutations = []
        
        self.output_full = output_full
        self.store_mutation_trajectories = store_mutation_trajectories
        self.ranked = False
        self.t1_d = quad(lambda S: np.sqrt(S)*sdist.pdf(S),sdist.ppf(0.001),sdist.ppf(0.999))[0]/2

        self.stat_names = ['d_trajectory','fixations','negative_variants','positive_variants','top_step_mutations','stored_mutations','second_moment','third_moment','aligned_variants','opposing_variants']
        self.stats = {}
        for name in self.stat_names:
            self.stats[name] = []

    # Some functions to define the steady state distribution of allele frequencies
    def variance_star(self, S, x):
        return 2 * S * x * (1 - x)

    def folded_sojourn_time(self, S, x):
        if x < 0:
            raise ValueError
        elif x > 1 / 2:
            raise ValueError
        else:
            value = 2 * np.exp(-self.variance_star(S=S, x=x) / 2) / (x * (1 - x))
            if x <= 1 / (2 * self.N):
                return 2 * self.N * x * value
            else:
                return value

    # Determine how many segregating sites we expect
    # This is the integral over the sojourn time times 2NU
    def total_n(self):
        a = quad(
            lambda S: quad(lambda x: self.folded_sojourn_time(S=S, x=x), 0, 1 / 2, points=[1 / (2 * self.N)])[
                          0] * self.sdist.pdf(S), self.sdist.ppf(0.0000001), self.sdist.ppf(0.99999))[0]
        return a * self.N2U

    ###### I think I don't use this anymore, but I leave it here in case I want to be able to refer to it with just the simulation object
    # Find the initial frequency based on a random starting value.
    # Must be between 0 and 0.5
    def cumulant(self, S, y):
        if y >= 0.5:
            return 1
        elif y <= 0:
            return 0
        top = quad(lambda x: self.folded_sojourn_time(S=S, x=x), 0, y, points=[1 / (2 * self.N)])[0]
        bottom = quad(lambda x: self.folded_sojourn_time(S=S, x=x), 0, 1 / 2, points=[1 / (2 * self.N)])[0]
        return top / bottom

    # Run the root finding algorithm to generate a random initial frequency
    def get_frequency(self, S):
        y0 = np.random.random()
        x = root(lambda y: self.cumulant(S=S, y=y) - y0, 1 / (2 * self.N)).x[0]
        if x < 0:
            y0, x = self.get_frequency(S)
        return y0, x

    # Run the root finding algorithm to get a given initial frequency
    def get_given_frequency(self, S, y0):
        return root(lambda y: self.cumulant(S=S, y=y) - y0, 1 / (2 * self.N)).x[0]

    def generate_segregating_mutations():
        pass
    ########
    
    # Generate the set of alleles segregating at the begining of the shift
    def initiate_mutations(self, n):
        # The number of alleles is poisson distributed around twice the expectation (because of aligned and opposing)
        n_realized = np.random.poisson(2 * n)
        # We need the probability of segregating for the next step of generating the effect sizes
        prob_segregating = quad(lambda S: quad(lambda x: self.folded_sojourn_time(S=S,x=x),0,1/2,points=[1/(2*self.N)])[0]*self.sdist.pdf(S),self.sdist.ppf(0),self.sdist.ppf(0.9999))[0]
        # get the effect sizes. I offloaded all this into a seperate file, because I use some precalculated files
        chosen_muts = generate_alleles(n=n_realized,prob_segregating=prob_segregating,N=self.N,sdist=self.sdist)  

        # add the variants to the list of mutations
        mutations = []
        for (x, S, y0) in chosen_muts:
            mutations.append(Mutation(x=x, a=np.sqrt(S), y0=y0, t=0, store=self.store_mutation_trajectories))

        self.mutations = mutations

    # add new mutations each generation and keep track of how much this changes the mean phenotype.
    # we assume new mutations are poisson distributed around 2NU
    # given that 2NU is parameterized as the 2NU for large effect alleles
    def add_new_mutations(self,t):
        change_in_d = 0
        aligned = 0
        opposing = 0
        positive = 0
        negative = 0
        for _ in range(np.random.poisson(self.N2U * 2)):
            a = (self.sdist.rvs()) ** (0.5)
            new_mutation = Mutation(a=a, x=1 / (2 * self.N), y0=t, t=t, store=False)
            self.mutations.append(new_mutation)

            mut_change_in_d = 2 * new_mutation.a * 1 / (2 * self.N)
            change_in_d += mut_change_in_d
            if new_mutation.a >= 0:
                aligned += mut_change_in_d
            else:
                opposing += mut_change_in_d

            exp_dx = self.mutations[-1].calc_change_in_x(d=self.stats['d_trajectory'][-1],N=self.N,t=0)
            if exp_dx >= 0:
                positive += mut_change_in_d
            else:
                negative += mut_change_in_d

        return change_in_d,aligned,opposing,positive,negative

    # update the distance based on the change from large effect alleles and from background. Store the distance every generation
    def update_distance_trajectory(self,d,change_in_d):
        d += -change_in_d - self.sigma2 * d / (2 * self.N)
        self.stats['d_trajectory'].append(d)
        return d

    # determine if allele is fixed or extinct. Store if it fixed.
    def check_if_extinct_or_fixed(self,mut):
        if mut.x == 1:
            self.stats['fixations'].append(mut)
            fixed_or_extinct = True
        elif mut.x == 0:
            fixed_or_extinct = True
        else:
            fixed_or_extinct = False
        return fixed_or_extinct

    # update mutation frequencies. Keep track of the change in the mean phenotype and
    # which alleles fix or go extinct
    def update_mutation_frequencies(self,d,t):
        remove_mutations = []
        change_in_d = 0
        aligned = 0
        opposing = 0
        negative = 0
        positive = 0
        for mut_index, mut in enumerate(self.mutations):
            mut_change_in_d, net_selection_sign = mut.update_freq(d=d, N=self.N, t=t)

            change_in_d += mut_change_in_d
            if mut.sign > 0:
                aligned += mut_change_in_d
            else:
                opposing += mut_change_in_d
            if net_selection_sign > 0:
                positive += mut_change_in_d
            else:
                negative += mut_change_in_d

            fixed_or_extinct = self.check_if_extinct_or_fixed(mut=mut)
            if fixed_or_extinct:
                remove_mutations.append(mut_index)

        return remove_mutations,change_in_d, aligned, opposing, negative, positive

    # remove mutations that fixed or went extinct. If it's a mutation that we are storing, store it.
    def remove_mutations_that_fixed_or_went_extinct(self,remove_mutations):
        if len(remove_mutations) > 0:
            mut_index_sorted = np.sort(remove_mutations)[::-1]
            for mut_index in mut_index_sorted:
                if self.store_mutation_trajectories and self.mutations[mut_index].store:
                    self.stats['stored_mutations'].append(dc(self.mutations[mut_index]))
                self.mutations.pop(mut_index)

    # determine if all mutations that arose before the cutoff time have gone extinct
    def check_all_segregating_extinct(self):
        for mut in self.mutations:
            if mut.t_initial < self.tracking_time:
                self.all_segregating_extinct = False
                return
        self.all_segregating_extinct = True
        return

    def recursion(self):

        d = self.shift
        self.stats['d_trajectory'] = [d]

        t = 0
        t_interval = 0
        aligned_interval = 0
        opposing_interval = 0
        negative_interval = 0
        positive_interval = 0
        while t < self.stop_time or np.abs(d) > 1:# or not self.all_segregating_extinct:

            # update allele frequencies and remove mutations that went extinct or fixed. Mutations that we want to store
            # or any that fix are stored within these functions
            remove_mutations,change_in_d,aligned,opposing,negative,positive = self.update_mutation_frequencies(d=d, t=t)
            self.remove_mutations_that_fixed_or_went_extinct(remove_mutations)

            # add new mutations each generation
            change_in_d_new,aligned_new,opposing_new,positive_new,negative_new = self.add_new_mutations(t=t)
            change_in_d += change_in_d_new
            
            aligned_interval += aligned + aligned_new
            opposing_interval += opposing+opposing_new
            negative_interval += negative + negative_new
            positive_interval += positive + positive_new
            t_interval += 1

            # update the distance
            d = self.update_distance_trajectory(d=d, change_in_d=change_in_d)
                
            # in-depth output if we want it
            if self.output_full:
                if not self.ranked and d < self.t1_d and 'top_step_mutations' in self.stat_names:
                    self.ranked = True
                    self.rank_mutations()

                if t < 100 or (t < 500 and t % 5 == 0) or (t < 1000 and t % 10 == 0) or t % 50 == 0:
                    if 'second_moment' in self.stat_names:
                        self.update_second_moment()
                    if 'third_moment' in self.stat_names:
                        self.update_third_moment()

                    self.stats['negative_variants'].append(negative_interval/t_interval)
                    self.stats['positive_variants'].append(positive_interval/t_interval)
                    self.stats['aligned_variants'].append(aligned_interval/t_interval)
                    self.stats['opposing_variants'].append(opposing_interval/t_interval)
                    t_interval = 0
                    aligned_interval = 0
                    opposing_interval = 0
                    negative_interval = 0
                    positive_interval = 0


            t += 1

    # Some functions for defining different metrics
    def update_second_moment(self):
        self.stats['second_moment'].append(sum([mut.calculate_second_moment() for mut in self.mutations]))

    def update_third_moment(self):
        self.stats['third_moment'].append(sum([mut.calculate_third_moment() for mut in self.mutations]))

    def rank_mutations(self):

        step_sizes = np.array([mut.get_step_size(D=self.stats['d_trajectory'][-1], N=self.N) for mut in self.mutations])
        ranks = np.flip(np.argsort(step_sizes))
        for rank, mut_index in enumerate(ranks):
            self.mutations[mut_index].rank = rank

            if rank < 5:
                self.stats['top_step_mutations'].append(self.mutations[mut_index])

    def update_large_effect_contributions(self, d,aligned_new,opposing_new):

        negative_contribution, positive_contribution, aligned, opposing  = self.calc_contribution_to_adaptation(d=d)

        self.stats['negative_variants'].append(negative_contribution)
        self.stats['positive_variants'].append(positive_contribution)
        self.stats['aligned_variants'].append(aligned + aligned_new)
        self.stats['opposing_variants'].append(opposing+opposing_new)

    def calc_contribution_to_adaptation(self, d):

        positive = 0
        negative = 0
        aligned = 0
        opposing = 0
        for mut in self.mutations:
            dx = mut.calc_change_in_x(d=d, N=self.N, t=0)
            change = 2 * mut.a * dx
            if dx <= 0:
                negative += change
            else:
                positive += change
            if mut.a > 0:
                aligned += change
            else:
                opposing += change

        return negative, positive, aligned, opposing

    # This initializes the simulation and then runs it
    def run_simulation(self):
        # print('running simulation')
        # sys.stdout.flush()
        n = self.total_n()
        # print('initializing mutations')
        # sys.stdout.flush()
        self.initiate_mutations(n=n)
        # print('doing recursion')
        # sys.stdout.flush()
        self.recursion()

# The second class defines the mutations
class Mutation:

    def __init__(self, x, a, y0, t, store=True):

        self.x = x
        self.store = store
        self.sign = 2 * (np.random.random() > 0.5) - 1
        self.a = a * self.sign
        self.t_initial = t
        self.trajectory = [x]
        self.y0 = y0

        self.rank = -1
        self.step = 0

        self.second_moment = 0
        self.third_moment = 0

    def calculate_second_moment(self):
        self.second_moment = 2 * self.a ** 2 * self.x * (1 - self.x)
        return self.second_moment

    def calculate_third_moment(self):
        self.third_moment = 2 * self.a ** 3 * self.x * (1 - self.x) * (1 - 2 * self.x)
        return self.third_moment

    def calc_change_in_x(self, d, N, t):
        a = self.a
        x = self.x
        change_in_x = a / (2 * N) * (d - a * (1 / 2 - x)) * x * (1 - x)
        return change_in_x

    def update_freq(self, d, N, t):

        a = self.a
        x = self.x
        change_in_x = self.calc_change_in_x(d=d, N=N, t=t)

        if change_in_x >= 0:
            net_selection_sign = 1
        else:
            net_selection_sign = 0
        if x < 0:
            print(x, change_in_x, self.trajectory)

        try:
            exp_x = min(1,max(0,x + change_in_x))
            new_x = np.random.binomial(2 * N, exp_x) / (2 * N)
        except:
            print(change_in_x, x)
            raise EOFError

        change_in_x = new_x - x
        self.x = new_x
        if self.store:
            self.trajectory.append(x)

        return 2 * a * change_in_x, net_selection_sign

    def get_step_size(self, D, N):
        a = self.a
        x = self.x

        change_in_x = a / (2 * N) * (D - a * (1 / 2 - x)) * x * (1 - x)
        self.step = change_in_x
        return self.step
