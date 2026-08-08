# %%
import sys
sys.path.append('/insomnia001/home/wm2377/cdwm/large_effect_alleles/scripts/individual_classes.py')
import individual_classes as classes
import pickle
import numpy as np
from scipy import stats
from scipy.integrate import quad
import os
import multiprocessing
import concurrent.futures as ccfutures
from scipy.special import comb
from scipy.integrate import quad
import time
import gzip

# function to run a single simulation
def run_simulation_function(idk):
    i,args = idk
    sigma2 = args.sigma2
    large_2NU = args.large_2NU
    N = args.N
    shift = args.shift
    burn_time = args.burn_time*N
    run_time = args.run_time*N
    output_file = f"/insomnia001/home/wm2377/cdwm/large_effect_alleles/data/individual_simulations/sigma2_{sigma2}/LargeN2U_{large_2NU}/shift_{shift}/iteration_{i}_with_mutation_counts.pkl"

    assert '.pkl' in output_file, 'Output file must be a .pkl file'

    # determine the effect size distribution and mutation rate
    small_sdist = stats.expon(scale=1)
    small_var_per_2NU = quad(lambda a2: classes.calculate_expected_variance_given_a2(N=N,a2=a2)*small_sdist.pdf(a2), 0, np.inf)[0]
    small_2NU = sigma2/ small_var_per_2NU
    N2U = small_2NU + large_2NU
    weight = large_2NU / N2U
    sys.stdout.flush()

    # create the simulation object
    sim = classes.Simulation(N=N, 
                            distribution_type='expon', 
                            optimum=0, 
                            shift = shift, 
                            output_file = output_file, 
                            mutation_rate=N2U/(2*N), 
                            burn_time = burn_time,
                            weight=weight,
                            stats_to_record=['fixed_background', 'fixations', 'phenotypic_moments','mutation_counts'])

    sim.calculate_expected_metrics()

    if os.path.exists(sim.population_output_file):
        print(f'Population {i} exists, loading it...')
        sys.stdout.flush()
        try:
            with open(sim.population_output_file, 'rb') as f:
                population, h, efd, dt, t = pickle.load(f)
            if t > 0:
                print(f'Population {i} already initialized, continuing simulation...',t)
                sys.stdout.flush()
                sim.history = h
                sim.population = population
                sim.run(generations=run_time+10, t = t)
                print(f'Simulation finished, saving population {i}...')
                return output_file
        except:
            print(f'loading population {i} file failed')
            os.remove(sim.population_output_file)
            if os.path.exists(sim.output_file):
                os.remove(sim.output_file)
            print(f'Population file {i} not found or corrupted, starting from scratch...')
        

    # check if the initialized and burned in population already exists
    if os.path.exists(sim.population_output_file.replace('.pkl','burned_population.pkl')):
        print(f'Burned population {i} already exists, loading it...')
        sys.stdout.flush()
        with open(sim.population_output_file.replace('.pkl','burned_population.pkl'), 'rb') as f:
            sim.population = pickle.load(f)[0]
    else:
        if os.path.exists(sim.population_output_file):
            print(f'Population {i} already exists but did not finish burn in, loading it...')
            sys.stdout.flush()
            with open(sim.population_output_file, 'rb') as f:
                sim.population, h, efd, dt, t = pickle.load(f)
            remaining_burn_time = np.abs(t)
            sim.burn_time = remaining_burn_time
            sim.burn()
            print(f'Finishing burn in for initialized population{i}...')
            sys.stdout.flush()
        else:
            print(f'Initializing population {i}...')
            sys.stdout.flush()
            sim.initialize_population()
            print(f'Burning in population {i}...')
            sys.stdout.flush()
            sim.burn()

    print(f'Population {i} burned in, starting simulation...')
    sys.stdout.flush()
    sim.shift_optimum()
    sim.run(generations=run_time+10, t = 1)
    print(f'Simulation finished, saving population {i}...')
    sys.stdout.flush()
    
    return output_file

def run_bssimulation_function(idk):
    i,args = idk
    sigma2 = args.sigma2
    large_2NU = args.large_2NU
    N = args.N
    shift = args.shift
    burn_time = args.burn_time*N
    run_time = args.run_time*N
    output_file = f"/insomnia001/home/wm2377/cdwm/large_effect_alleles/data/individual_simulations/sigma2_{sigma2}/LargeN2U_{large_2NU}/shift_{shift}/iteration_{i}.pkl"
    return output_file

def main():
    run_parallel_simulations()

# run multiple simulations in parallel
def run_parallel_simulations():
    tmp_output = snakemake.output.output_file.replace('.txt','_temp.pkl')
    if os.path.exists(tmp_output):
        with open(tmp_output,'rb') as infile:
            completed_files = pickle.load(infile)
    else:
        completed_files = {}
    completed_files_list = np.sort(list(completed_files.keys()))
    i_done = len(completed_files_list)
    print(f'Finished {i_done} simulations')

    args = snakemake.params
    ID = [(i,args) for i in range(snakemake.params.replicates) if i not in completed_files_list]
    print('Simulations to run or finish ',[i for (i,args) in ID])
    # We can use a with statement to ensure threads are cleaned up promptly
    with ccfutures.ProcessPoolExecutor(max_tasks_per_child=1) as executor:

        for (i,args),data  in zip(ID, executor.map(run_simulation_function, ID)):
            completed_files[i] = data
            with open(tmp_output,'wb') as outfile:
                pickle.dump(completed_files, outfile)
            print(f'Simulation {i} finished, output file: {data}')
            sys.stdout.flush()
    
    with open(snakemake.output.output_file,'wb') as outfile:
        pickle.dump(completed_files, outfile)
            
if __name__ == '__main__':
    main()
    
