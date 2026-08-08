import numpy as np
import scipy
from scipy import stats
from scipy.integrate import quad
import pickle
import sys
from scipy.optimize import root

class Individual:
    def __init__(self,effect_size_distribution,mutation_rate):
        self.genotype = {}
        self.phenotype = 0
        self.fitness = 1
        self.mutation_rate = mutation_rate
        self.effect_size_distribution = effect_size_distribution

    
    def calculate_phenotype(self,fixed_background=0):
        """
        Calculate the phenotype based on the genotype and fixed background.
        """
        self.phenotype = fixed_background
        for mut,m in self.genotype.items():
            self.phenotype += mut.a * m
        
    
    def calculate_fitness(self, optimum, Vs):
        """
        Calculate the fitness based on the phenotype and the optimum.
        """
        self.fitness = np.exp(-((self.phenotype - optimum) ** 2) / (2 * Vs))
    
    def create_gamete(self):
        """
        Create a gamete from the individual's genotype, applying mutations.
        """
        gamete = {}
        for mut,m in self.genotype.items():
            if m == 2:
                gamete[mut] = 1
            elif m == 1:
                if np.random.rand() < 0.5:
                    gamete[mut] = 1
        n_mutations = np.random.poisson(self.mutation_rate)
        for _ in range(n_mutations):
            a2 = self.effect_size_distribution.rvs()
            gamete[Mutation(a2=a2)] = 1
        return gamete
    
class Mutation:
    def __init__(self,a2):
        self.a2 = a2  # squared effect size
        self.sign = np.random.choice([-1, 1])  # random sign for the effect size
        self.a = np.sqrt(a2)*self.sign  # effect size

class Population:
    def __init__(self, N, effect_size_distribution, optimum, mutation_rate):
        self.individuals = [Individual(effect_size_distribution, mutation_rate) for _ in range(N)]
        self.N = N
        self.optimum = optimum
        self.Vs = 2*N
        self.effect_size_distribution = effect_size_distribution
        self.mutation_rate = mutation_rate
        
        self.mutation_counts = {}
        self.fixed_background = 0
        self.fixations = []
    
    def calculate_phenotypes(self):
        """
        Calculate phenotypes for all individuals in the population.
        """
        for ind in self.individuals:
            ind.calculate_phenotype(fixed_background=self.fixed_background)
    
    def calculate_fitnesses(self):
        """
        Calculate fitnesses for all individuals in the population after updating phenotypes.
        """
        self.calculate_phenotypes()
        for ind in self.individuals:
            ind.calculate_fitness(optimum=self.optimum, Vs=self.Vs)
    
    def choose_parents(self):
        """
        Choose 2N parents based on fitness, using a weighted random choice.
        """
        self.calculate_fitnesses()
        fitnesses = np.array([ind.fitness for ind in self.individuals])
        probabilities = fitnesses / np.sum(fitnesses)
        parents_indices = np.random.choice(range(self.N), size=2*self.N, p=probabilities)
        return [self.individuals[i] for i in parents_indices]
    
    def create_gametes(self):
        """
        Create gametes for all individuals in the population.
        """
        gametes = []
        
        for ind in self.choose_parents():
            gametes.append(ind.create_gamete())
        return gametes
    
    def reproduce(self):
        """
        Reproduce the population by creating gametes and forming new individuals.
        """
        gametes = self.create_gametes()
        new_individuals = []
        for gameteA,gameteB in zip(gametes[::2], gametes[1::2]):
            new_individual = Individual(self.effect_size_distribution, self.mutation_rate)
            new_individual.genotype = {}
            for mut in gameteA:
                new_individual.genotype[mut] = 1
            for mut in gameteB:
                if mut in new_individual.genotype:
                    new_individual.genotype[mut] += 1
                else:
                    new_individual.genotype[mut] = 1
            new_individuals.append(new_individual)
        self.individuals = new_individuals
        
    def update_mutation_counts(self):
        """
        Update the mutation counts in the population.
        """
        self.mutation_counts = {}
        for ind in self.individuals:
            for mut,m in ind.genotype.items():
                if mut not in self.mutation_counts:
                    self.mutation_counts[mut] = 0
                self.mutation_counts[mut] += m
    
    def calculate_fixed_background(self,t):
        """
        Calculate the fixed background of the population.
        """

        new_fixations = []
        for mut in self.mutation_counts:
            if self.mutation_counts[mut] == 2*self.N:
                self.fixed_background += 2*mut.a
                self.fixations.append((mut,t))
                new_fixations.append(mut)
        return new_fixations
        
    def remove_fixed_mutations(self, new_fixations):
        for mut in new_fixations:
            for ind in self.individuals:
                ind.genotype.pop(mut)
        
    def next_generation(self,t):
        """
        Advance the population to the next generation.
        """
        self.reproduce()
        self.update_mutation_counts()
        new_fixations = self.calculate_fixed_background(t)
        self.remove_fixed_mutations(new_fixations)
        self.update_mutation_counts()
        self.calculate_phenotypes()
        
class Simulation:
    def __init__(self, N, distribution_type, optimum, shift, output_file, mutation_rate, burn_time,weight,stats_to_record):
        self.distribution_type = distribution_type
        self.optimum = optimum
        self.shift = shift
        self.N = N
        self.mutation_rate = mutation_rate
        self.burn_time = burn_time 
        
        self.output_file = output_file
        self.population_output_file = output_file.replace('.pkl', '_population.pkl')
        self.expected_output_file = output_file.replace('.pkl', '_expected.pkl')
        
        self.n_seg_per_mutational_input = 0
        self.variance_per_mutational_input = 0
        self.effect_size_distribution = EffectSizeDistribution(weight=weight, distribution_type=distribution_type)
    
        self.population = Population(N, self.effect_size_distribution, optimum, mutation_rate)
        self.history = {}
        self.stats_to_record = stats_to_record
        
        
    def calculate_expected_metrics(self):
        """
        Calculate expected metrics for the simulation.
        """
        self.n_seg_per_mutational_input = calculate_expected_number_of_segregating_mutations(
            N=self.population.N,
            effect_size_distribution=self.effect_size_distribution
        )
        print(f"Expected number of segregating mutations per mutational input: {self.n_seg_per_mutational_input}")
        
        self.variance_per_mutational_input = calculate_expected_variance(
            N=self.population.N,
            effect_size_distribution=self.effect_size_distribution
        )
        print(f"Expected phenotypic variance per mutational input: {self.variance_per_mutational_input}")
        with open(self.expected_output_file, 'wb') as f:
            pickle.dump((self.n_seg_per_mutational_input, self.variance_per_mutational_input), f)

    # burn in, do not record information but do save the population
    # this is to allow the population to reach a stable state before running the simulation
    def burn(self):
        """
        Burn in the population for a specified number of generations.
        """
        t = -int(self.burn_time)
        while t <= 0:
            self.population.next_generation(t)
            if t % 100 == 0:
                if t % 1000 == 0:
                    i = self.output_file.split('iteration_')[1].split('.pkl')[0] 
                    print(f"Generation {t}, Population {i}")
                    sys.stdout.flush()
                self.output_copy_of_population(t)
            t += 1
        self.history[0]=(self.population.fixed_background, self.population.mutation_counts.copy())
        self.output_copy_of_population(t,name='burned_population.pkl')
        
    # shift the optimum of the population
    def shift_optimum(self):
        """
        Shift the optimum of the population by a specified amount.
        """
        self.population.optimum += self.shift
        
    def calculate_phenotypic_moments(self):
        """
        Calculate the phenotypic moments of the population.
        """
        phenotypes = np.array([ind.phenotype for ind in self.population.individuals])
        mean = np.mean(phenotypes)
        variance = np.var(phenotypes)
        skewness = scipy.stats.skew(phenotypes)
        return mean, variance, skewness
    
    # run the simulation for a specified number of generations
    def run(self, generations, t=1):
        """
        Run the simulation for a specified number of generations.
        """
        
        while t < generations:
            self.population.next_generation(t)
            self.history[t+1] = self.create_history()
            if t % 100 == 0:
                i = self.output_file.split('iteration_')[1].split('.pkl')[0]
                print(f"Generation {t}, Population {i}")
                sys.stdout.flush()
                self.output_copy_of_population(t)
                self.output_history()
            t += 1

    def create_history(self):
        """
        Create a history of the simulation.
        """
        h = {}
        if 'fixed_background' in self.stats_to_record:
            h['fixed_background'] = self.population.fixed_background
        if 'mutation_counts' in self.stats_to_record:
            h['mutation_counts'] = self.population.mutation_counts.copy()
        if 'phenotypic_moments' in self.stats_to_record:
            h['mean'], h['variance'], h['skewness'] = self.calculate_phenotypic_moments()
        if 'fixations' in self.stats_to_record:
            h['fixations'] = self.population.fixations.copy()
        return h
    
    def output_history(self):
        """
        Output the history of the simulation to a file.
        """
        with open(self.output_file, 'ab') as f:
            pickle.dump(self.history, f)
        self.history = {}  # Clear history after saving to file
    
    def output_copy_of_population(self,t,name=None):
        """
        Output a copy of the current population to a file.
        """
        if name is None:
            with open(self.population_output_file, 'wb') as f:
                pickle.dump((self.population,self.history,self.effect_size_distribution,self.distribution_type,t), f)
        else:
            with open(self.population_output_file.replace('.pkl',name), 'wb') as f:
                pickle.dump((self.population,self.history,self.effect_size_distribution,self.distribution_type,t), f)
        
    
    def initialize_population(self):
        """
        Initialize the population with random mutations.
        """
        n_seg = self.n_seg_per_mutational_input * self.population.mutation_rate * 2 * self.population.N
        for _ in range(np.random.poisson(n_seg)):
            a2 = self.effect_size_distribution.rvs()
            mut = Mutation(a2=a2)
            freq = get_random_frequency(N=self.population.N, a2=a2, p=np.random.random())  # Get a random frequency for the mutation
            allele_counts = np.random.binomial(2, freq, size=self.population.N)
            for ind, count in zip(self.population.individuals, allele_counts):
                if count > 0:
                    ind.genotype[mut] = count
        self.population.update_mutation_counts()

def variance_star(a2,x):
    return 2 * a2 * x * (1-x)

def sojourn_time(N, a2, x):
    """
    Calculate the expected sojourn time at frequency x for a mutation with effect size a2 in a population of size N.
    """
    v = 2 * np.exp(-variance_star(a2, x)/2)/(1-x)  # Variance of the effect size distributio
    if x < 1/(2*N):
        v *= 2*N
    else:
        v *= 1/x
    return v

def get_random_frequency(N,a2,p):
    denominator = quad(lambda x: sojourn_time(N=N,a2=a2,x=x), 0, 1/2, points=[1/(2*N)])[0]  # Integrate sojourn time from 0 to 1/2
    numerator = lambda y: quad(lambda x: sojourn_time(N=N,a2=a2,x=x), 0, y, points=[1/(2*N)])[0]  # Integrate sojourn time from 0 to y
    return root(lambda z: numerator(z) - p * denominator, 1/(2*N)).x[0]

def calculate_expected_number_of_segregating_mutations_given_a2(N, a2):
    """
    Calculate the expected number of segregating mutations given a2 per mutational input
    """
    return quad(lambda x: sojourn_time(N=N,a2=a2,x=x), 0, 1/2,points=[1/(2*N)])[0]  # Integrate sojourn time from 0 to 1/2

def calculate_expected_number_of_segregating_mutations(N, effect_size_distribution):
    
    """
    Calculate the expected number of segregating mutations after a given number of generations per mutational input.
    """
    return quad(
        lambda a2: calculate_expected_number_of_segregating_mutations_given_a2(N=N,a2=a2)*effect_size_distribution.pdf(a2), 
        0, 
        10000, 
        points=[effect_size_distribution.small_dist.ppf(0.9999),effect_size_distribution.large_dist.ppf(0.001),effect_size_distribution.small_dist.ppf(0.9999)],
    )[0]

def calculate_expected_variance_given_a2(N, a2):
    """
    Calculate the expected phenotypic variance given a2 per mutational input.
    """
    return quad(lambda x: variance_star(a2=a2,x=x)*sojourn_time(N=N,a2=a2,x=x), 0, 1/2, points=[1/(2*N)])[0]  # Integrate variance from 0 to 1/2
    
def calculate_expected_variance(N, effect_size_distribution):
    """
    Calculate the expected phenotypic variance in the population.
    """
    return quad(
        lambda a2: calculate_expected_variance_given_a2(N=N,a2=a2)*effect_size_distribution.pdf(a2), 
        0, 
        10000, 
        points=[effect_size_distribution.small_dist.ppf(0.9999),effect_size_distribution.large_dist.ppf(0.001),effect_size_distribution.small_dist.ppf(0.9999)],
    )[0]
    
# defines a mixture of two distributions for the effect size of mutations
class EffectSizeDistribution:
    def __init__(self, weight, distribution_type='expon'):
        self.distribution_type = distribution_type
        self.weight = weight
        self.small_dist = stats.expon(scale=1)
        if distribution_type == 'expon':
            self.large_dist = stats.expon(scale=400,loc=100)
        elif distribution_type == 'uniform':
            self.large_dist = stats.uniform(loc=100, scale=900)
        else:
            raise ValueError("Unsupported distribution type")
    
    def rvs(self):
        random = np.random.rand()
        if random > self.weight:
            self.dist = self.small_dist
        else:
            self.dist = self.large_dist
        return self.dist.rvs()
    
    def pdf(self, a2):
        return (1-self.weight) * self.small_dist.pdf(a2) + (self.weight) * self.large_dist.pdf(a2)