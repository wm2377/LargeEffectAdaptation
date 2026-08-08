import numpy as np
import scipy
from scipy import stats
from scipy.integrate import quad
import pickle
import sys
from scipy.optimize import root

class Mutation:
    def __init__(self,a2,N,frequency=0):
        self.a2 = a2  # squared effect size
        self.sign = np.random.choice([-1, 1])  # random sign for the effect size
        self.a = np.sqrt(a2)*self.sign  # effect size
        self.frequency = frequency
        self.N = N
        self.Vs = 2*N  
    
    def expected_change_in_frequency(self,distance):
        """
        Calculate the expected change in frequency of the mutation given its effect size.
        """
        return self.a / self.Vs * (distance - self.a*(1/2-self.frequency)*(1-distance**2/self.Vs)) * self.frequency * (1 - self.frequency) # distance is the distance to the optimum

    def next_generation(self, distance):
        """
        Calculate the change in frequency of the mutation given its effect size and the distance to the optimum.
        """
        expected_change = self.expected_change_in_frequency(distance)
        expected_frequency = self.frequency + expected_change
        realized_frequency = np.random.binomial(2*self.N, expected_frequency) / (2*self.N)  # Binomial sampling to get the realized frequency
        realized_change = realized_frequency - self.frequency
        self.frequency = realized_frequency
        return realized_change
    
    def fixed(self):
        """
        Check if the mutation is fixed in the population.
        """
        return self.frequency == 1
    
    def extinct(self):
        """
        Check if the mutation is extinct in the population.
        """
        return self.frequency == 0
    
    def contribution_to_variance(self):
        """
        Calculate the contribution of the mutation to the phenotypic variance.
        """
        return 2 * self.a2 * self.frequency * (1 - self.frequency)
    
    def contribution_to_phenotype(self):
        """
        Calculate the contribution of the mutation to the phenotype.
        """
        return 2 * self.a * self.frequency
    
    def force_to_be_minor(self):
        """
        Force the mutation to be minor allele (frequency <= 0.5)
        """
        if self.frequency > 0.5:
            self.frequency = 1 - self.frequency
            self.a = -self.a
            self.sign = -self.sign

class Population:
    def __init__(self, N, effect_size_distribution, optimum, mutation_rate,output_file):
        self.mutations = []
        self.N = N
        self.optimum = optimum
        self.Vs = 2*N
        self.effect_size_distribution = effect_size_distribution
        self.mutation_rate = mutation_rate
        
        self.output_file = output_file
        self.fixed_background = 0
        self.fixations = []
        self.new_fixations = []
    
    def handle_fixed_or_extinct_mutations(self):
        """
        Detect and handle fixed or extinct mutations in the population.
        """

        mutations_to_remove = []
        for mut in self.mutations:
            if mut.fixed():
                self.fixations.append(mut)
                self.new_fixations.append(mut)
                mutations_to_remove.append(mut)
                self.fixed_background += 2 * mut.a
            elif mut.extinct():
                mutations_to_remove.append(mut)
        
        for mut in mutations_to_remove:
            self.mutations.remove(mut)
        
    def update_mutation_frequencies(self,distance):
        """
        Update the frequencies of all mutations in the population based on their effect sizes and the distance to the optimum.
        """
        realized_changes = 0
        for mut in self.mutations:
            c = mut.next_generation(distance)
            realized_changes += c * 2 * mut.a
        return realized_changes
    
    def mean_phenotype(self):
        """
        Calculate the mean phenotype of the population.
        """
        mean = self.fixed_background
        for mut in self.mutations:
            mean += mut.contribution_to_phenotype()
        return mean
    
    def add_new_mutations(self):
        """
        Add new mutations to the population based on the mutation rate and effect size distribution.
        """
        n_new_mutations = np.random.poisson(2 * self.mutation_rate * self.N)
        for _ in range(n_new_mutations):
            a2 = self.effect_size_distribution.rvs()
            mut = Mutation(a2=a2, N=self.N, frequency=1/(2*self.N))
            self.mutations.append(mut)
        
    def next_generation(self,t):
        """
        Advance the population to the next generation.
        """
        
        distance = self.optimum - self.mean_phenotype()
        realized_change = self.update_mutation_frequencies(distance)
        self.handle_fixed_or_extinct_mutations()
        self.add_new_mutations()
        return self.generate_output()
        
    def generate_output(self):
        mutations = [(mut.a,mut.frequency) for mut in self.mutations]
        fixations = [mut.a for mut in self.new_fixations]
        self.new_fixations = []  # Clear new fixations after output
        return {
            'mutations': mutations,
            'fixed_background': self.fixed_background,
            'fixations': fixations,
            'mean_phenotype': self.mean_phenotype()}
        
    def switch_to_minor_alleles(self):
        """
        Force all mutations to be minor alleles (frequency <= 0.5)
        """
        for mut in self.mutations:
            mut.force_to_be_minor()
        

        
class Simulation:
    def __init__(self, N, distribution_type, optimum, shift, output_file, mutation_rate, burn_time,weight):
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
    
        self.population = Population(N, self.effect_size_distribution, optimum, mutation_rate,output_file=output_file)
        self.history = {}
        
        
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
        print(f"Expected phenotypic variance: {self.variance_per_mutational_input*2*self.N*self.mutation_rate}")
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
                    print(f"Generation {t}")
                sys.stdout.flush()
                self.output_copy_of_population(t)
            t += 1
        self.history[0]=self.population.generate_output()
        self.output_copy_of_population(t,name='burned_population.pkl')
        
    # shift the optimum of the population
    def shift_optimum(self):
        """
        Shift the optimum of the population by a specified amount.
        """
        self.population.optimum += self.shift
    
    def switch_to_minor_alleles(self):
        """
        Force all mutations in the population to be minor alleles (frequency <= 0.5)
        """
        self.population.switch_to_minor_alleles()
    
    # run the simulation for a specified number of generations
    def run(self, generations, t=1):
        """
        Run the simulation for a specified number of generations.
        """
        
        while t < generations:
            self.population.next_generation(t)
            self.history[t+1]=self.population.generate_output()
            if t % 10 == 0:
                if t % 100 == 0:
                    print(f"Generation {t}")
                sys.stdout.flush()
                self.output_copy_of_population(t)
            if len(self.history) > 10:
                self.output_history()
            t += 1
    
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
        print(f"Expected number of segregating mutations: {n_seg}")
        for _ in range(np.random.poisson(n_seg)):
            a2 = self.effect_size_distribution.rvs()
            mut = Mutation(a2=a2,N=self.N)
            mut.frequency = get_random_frequency(N=self.population.N, a2=a2, p=np.random.random())  # Get a random frequency for the mutation
            self.population.mutations.append(mut)

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
    
    
        
# to do:
# add method to calculate the expected phenotypic variance
# add method to calculate the realized phenotypic variance
