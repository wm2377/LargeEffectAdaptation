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
import time
import gzip

def process_simple_data(data,args):
    # some variation in how results were stored across previous versions of simulations. This handles that.
    try:
        fixations = [i[0] for i in data]
    except:
        fixations = data

    # the goal is to return 1) the number of fixations in each replicate, so count number of fixations, 2) the number of segregating or new fixations in each replicate, and 3) the total adaptation from fixations in each replicate
    # if there are any nan values in fixations,  return zeros
    if np.nan in fixations:
        return 0, 0, 0
    total_fixations = np.array([len(i) for i in fixations])
    total_adaptation = np.array([sum([2*mut.a*(1-mut.trajectory[0]) for mut in i]) for i in fixations])
    segregating_fixations = np.array([sum([1 for mut in i if mut.trajectory[0] < 1.0 and mut.trajectory[0]>0.0]) for i in fixations])
    segregating_adaptation = np.array([sum([2*mut.a*(1-mut.trajectory[0]) for mut in i if mut.trajectory[0] < 1.0 and mut.trajectory[0]>0.0]) for i in fixations])
    new_fixations = total_fixations-segregating_fixations
    new_adaptation = total_adaptation - segregating_adaptation

    first_fixations = []
    for i in fixations:
        if len(i)>0:
            first_fixations.append([i[0]])
    
    return {'fixations':{'total':total_fixations,'segregating':segregating_fixations,'new':new_fixations},
            'adaptation':{'total':total_adaptation,'segregating':segregating_adaptation,'new':new_adaptation}.
            'first_fixations':first_fixations}
    
def process_data_for_dsq(data,args):
    # To find when the quasi-static state is reached, we need to find where 2*D*V=U
    # D is mean_trajectory, U is the third moment, V is the second moment
    
    dsq_t = []
    dsq_d = []
    for replicate in data:
        mean_trajectory = np.array(replicate['d_trajectory'])
        second_moment_trajectory = np.array(replicate['second_moment'])
        third_moment_trajectory = np.array(replicate['third_moment'])
    
        # higher order moments only calculated at certain time points to save space
        time_points = np.arange(0,mean_trajectory.shape[1],mean_trajectory.shape[1])
        time_points_of_higher_moments = []
        for t in time_points:
            if t < 100 or (t < 500 and t % 5 == 0) or (t < 1000 and t % 10 == 0) or t % 50 == 0:
                time_points_of_higher_moments.append(t)
                
        D_sparse = mean_trajectory[time_points_of_higher_moments]
        distance_variance = second_moment_trajectory[time_points_of_higher_moments]*D_sparse*2
        distance_variance_windowed = np.convolve(distance_variance,np.ones(10)/10,mode='same')
        skew_windowed = np.convolve(third_moment_trajectory[time_points_of_higher_moments],np.ones(10)/10,mode='same')
        
        # find the first time point where distance_variance_windowed is within 5% of skew_windowed
        # If no such time point exists, return 20000
        dsq_t_index = np.where(np.abs((distance_variance_windowed-skew_windowed)/distance_variance_windowed) < 0.05)
        if len(dsq_t_index[0]) > 0:
            dsq_t.append(time_points_of_higher_moments[dsq_t_index[0][0]])
            dsq_d.append(D_sparse[dsq_t_index[0][0]])
        else:
            dsq_t.append(20000)
            dsq_d.append(0)
    return {'dsq_t':np.array(dsq_t),'dsq_d':np.array(dsq_d)}    
    
def process_full_data(data,args):
    # some variation in how results were stored across previous versions of simulations. This handles that.
    fixations = []
    for replicate in data:
        fixations.append(replicate['fixations'])
    simple_results = process_simple_data(fixations,args)
    
    dsq_results = process_data_for_dsq(data,args)
    
    results = {'fixations':simple_results['fixations'],
               'adaptation':simple_results['adaptation'],
               'dsq_t':dsq_results['dsq_t'],
               'dsq_d':dsq_results['dsq_d']}
    return results  

def main():
    # get args
    args = snakemake.params
    sdist = args.sdist
    
    input_file_path = snakemake.input.results_file_path
    output_file_path = snakemake.output.results_file_path
    
    with gzip.open(input_file_path,'rb') as fin:
        results = pickle.load(fin)
    
    for data in results[args.N2U][args.shift][args.sigma2]:
        if args.output_type == 'full':
            analyzed_results = process_full_data(data,args)
        else:
            analyzed_results = process_simple_data(data,args)
    
    complete_analyzed_results = {args.N2U:{args.shift:{args.sigma2:analyzed_results}}}
    with gzip.open(output_file_path,'wb') as fout:
        pickle.dump(complete_analyzed_results,fout)
        
if __name__ == "__main__":
    main()
