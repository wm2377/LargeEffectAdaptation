import os
import sys

sys.path.append('/Users/will_milligan/PycharmProjects/Laura/n_dim_code')
sys.path.append('/burg/palab/users/wm2377/packages')
sys.path.append('/burg/palab/users/wm2377/rejection_sampling_freqs')
import numpy as np
import argparse
import scipy.stats as stats
import pickle
from simulations_10042023_classes import Simulation
from copy import deepcopy as dc
from scipy.optimize import root
from math import ceil

import multiprocessing
import concurrent.futures as ccfutures
from scipy.special import comb
from scipy.integrate import quad
import time
import gzip

# The goal of this file is to run simplified all allele simulations in parallel
def main():
    # First we get the args from snakemake. 
    # If the args are given in total variance and p, instead of sigma2 and 2NU, we make the necessary transformation
    args = snakemake.params
    sdist = args.sdist
    
    try:
        total_variance = args.total_variance
        p = args.p
        args.sigma2 = total_variance*(1-p)
        args.N2U = p*total_variance/4
    except:
        pass
    # If the shift size is too large, we ignore it.
    # Generate a null output, so snakemake thinks it did something
    if args.shift > np.sqrt(args.N*2):
        generate_null_output(args)
        return
    else:
        # Try and see if we have completed any of these simulations already
        trials_completed = get_trials_completed(args)
        trials_completed, p = simulations_with_given_parameters(args,trials_completed)
        output_results(trials_completed,args)

# Makes a null output for when we don't actually want to simulate these parameter combinations
def generate_null_output(args):
    trials_completed = {args.N2U:{args.shift:{args.sigma2:[np.nan]*args.trials}}}
    output_results(trials_completed,args)
    
# Check if the output already exists and see how many replicates we have already ran
def get_trials_completed(args):

    # If output exists, load the results
    if os.path.exists(snakemake.output.results_file_path):
        with gzip.open(snakemake.output.results_file_path, 'rb') as fout:
            trials_completed = pickle.load(fout)
    # if output does not exist, start a new output dict
    else:
        trials_completed = {}
        if args.N2U not in trials_completed.keys():
            trials_completed[args.N2U] = {}

            if args.shift not in trials_completed[args.N2U].keys():
                trials_completed[args.N2U][args.shift] = {}

                if args.sigma2 not in trials_completed[args.N2U][args.shift].keys():
                    trials_completed[args.N2U][args.shift][args.sigma2] = []

    return trials_completed

# Runs jobs_per_executor number of simulations, this is the function passed to the parallelization command
def run_simulation(idk):
    i,args = idk
    results = []
    for i in range(args.jobs_per_executor):
        # Initialize and run the simulation
        # Here, we only record the fixations that occured
        # This can be changed to get other metrics
        simulation_t = Simulation(N=args.N, sdist=args.sdist, N2U=args.N2U, sigma2=args.sigma2, shift=args.shift,output_full=False)
        simulation_t.run_simulation()
        results.append((simulation_t.stats['fixations']))
    return results

# Calculate the current estimate forr the probability of any large effect reaching fixation
def calc_p(trials_completed_sigma2):

    if len(trials_completed_sigma2) == 0:
        return 1,1,0
    num_fixations = sum([len(i) > 0 for i in trials_completed_sigma2])
    p = num_fixations/len(trials_completed_sigma2)
    p_ste = np.sqrt(p*(1-p)/len(trials_completed_sigma2))
    return p,p_ste,num_fixations

# Does the parallelization
def simulations_with_given_parameters(args,trials_completed):
    # Get the current estimate for the probability of any large effect reaching fixation and how many replicates have already been run
    try:
        p,p_ste,num_fixations = calc_p(trials_completed_sigma2=trials_completed[args.N2U][args.shift][args.sigma2])
        t = len(trials_completed[args.N2U][args.shift][args.sigma2])
    except:
        trials_completed = {args.N2U: {args.shift: {args.sigma2: trials_completed[2*args.N2U][args.shift][args.sigma2]}}}
        p,p_ste,num_fixations = calc_p(trials_completed_sigma2=trials_completed[args.N2U][args.shift][args.sigma2])
        t = len(trials_completed[args.N2U][args.shift][args.sigma2])

    # Runs the simulations, splitting the tasks over max_executor "nodes"
    # Import to outout results every time, so we have them in case this crashes/the job gets cancelled
    t_run = 0
    print(t)
    sys.stdout.flush()
    start = time.time()
    ID = [(i,args) for i in range(ceil((args.trials-t)/args.jobs_per_executor))]
    # We can use a with statement to ensure threads are cleaned up promptly
    with ccfutures.ProcessPoolExecutor(max_workers=args.max_executor,max_tasks_per_child=1) as executor:

        for (i,args),data  in zip(ID, executor.map(run_simulation, ID)):
            trials_completed[args.N2U][args.shift][args.sigma2].extend(data)
            p,p_ste,num_fixations = calc_p(trials_completed_sigma2=trials_completed[args.N2U][args.shift][args.sigma2])
            output_results(trials_completed,args)
            t_run += 1
            print(t_run,num_fixations)
            sys.stdout.flush()
                
    
    end = time.time()
    print('Time taken: ',end-start,t_run)
    output_results(trials_completed,args)
    p,p_ste,num_fixations = calc_p(trials_completed_sigma2=trials_completed[args.N2U][args.shift][args.sigma2])
    
    return trials_completed, p

# Stores the results
def output_results(simulation_results,args):
    results_file_path = snakemake.output.results_file_path
    with gzip.open(results_file_path, 'wb') as fout:
        pickle.dump(simulation_results, fout)
        pickle.dump(args,fout)

if __name__ == '__main__':
    main()
