import os
import sys
import gzip
sys.path.append('/Users/will_milligan/PycharmProjects/Laura/n_dim_code')
sys.path.append('/burg/palab/users/wm2377/packages')
sys.path.append('/burg/palab/users/wm2377/rejection_sampling_freqs')
import numpy as np
import argparse
import pickle
from simulations_10042023_classes_copy import Simulation,Mutation
from scipy.optimize import root
from scipy.integrate import quad
from scipy.special import comb
from collections import namedtuple as nt
from simulations_10042023_get_contour_levels_precalc import CustomSdist

def main():

    concatenate_shift()

def process_results(results,args):
    
    # trials_completed[args.N2U][args.shift][args.sigma2].extend(data)
    shift = args.shift
    sigma2 = args.sigma2
    N2U = args.N2U
    
    # some variation in how results were stored across previous versions of simulations. This handles that.
    try:
        real_results = results[args.N2U][shift][sigma2]
    except:
        real_results = results[args.N2U*2][shift][sigma2]
    try:
        fixations = [i[0] for i in real_results]
    except:
        fixations = real_results

    # if there are any nan values in fixations, return zeros
    if np.nan in fixations:
        return 0, 0, 0

    # count number of fixations. If error, return zeros
    try:
        k = sum([1 for i in fixations if len(i) > 0])
    except:
        return 0, 0, 0
    
    # Calculate the number of fixations and average adaptation from fixations in each replicate
    n = len(fixations)
    adaptation = np.mean([sum([2*mut.a*(1-mut.trajectory[0]) for mut in i]) for i in fixations])

    # Try to calculate integral D values. If error, return k,n,adaptation only. Error due to older simulation results that did not store integral D values.
    try:
        integral_D = [i[1] for i in real_results]
        sum_integral_D = np.sum(integral_D)
        n_integral_D = len(integral_D)
        sum_squared_integral_D = np.sum([i**2 for i in integral_D])

        return k,n,adaptation,sum_integral_D,sum_squared_integral_D, n_integral_D
    except:
        return k,n,adaptation

def get_args(args,filename):
    # check if args is right
    try:
        shift = args.shift
        N2U = args.N2U
        sigma2 = args.sigma2
        sdist = args.sdist
    except:
        # if not, load from filename
        N2U = eval(filename.split('N2U_')[1].split('/')[0])
        shift = eval(filename.split('shift_')[1].split('/')[0])
        sigma2 = eval(filename.split('sigma2_')[1].split('/')[0])
        sdist = filename.split('sdist_')[1].split('/')[0]
        #create args object
        args = nt('args', ['N2U', 'shift', 'sigma2', 'sdist'])
        args.N2U = N2U
        args.shift = shift
        args.sigma2 = sigma2
        args.sdist = sdist
    return args
        
# Store all the results together
def concatenate_shift():
    input_files = snakemake.input
    output_results = {}
    
    for filename in input_files:

        with gzip.open(filename,'rb') as fin:
            results = pickle.load(fin)
            args = pickle.load(fin)
        args = get_args(args=args,filename=filename)

        try:
            k,n,adaptation = process_results(results,args)
            integral_D = False
        except:
            k,n,adaptation,sum_integral_D,sum_squared_integral_D, n_integral_D = process_results(results,args)
            integral_D = True
        
        if args.N2U not in output_results:
            output_results[args.N2U] = {}
            
        if args.sigma2 not in output_results[args.N2U]:
            output_results[args.N2U][args.sigma2] = {}
        
        if integral_D:
            output_results[args.N2U][args.sigma2][args.shift] = (k,n,adaptation,sum_integral_D,sum_squared_integral_D, n_integral_D)
        else:
            output_results[args.N2U][args.sigma2][args.shift] = (k,n,adaptation)
        
        if np.isclose(args.N2U*4+args.sigma2,600,rtol=0.01):
            VA = args.N2U*4+args.sigma2
            print('VA:',VA,'N2U:',args.N2U,'shift:',args.shift,'sigma2:',args.sigma2,'k:',k,'n:',n,'adaptation:',adaptation)
    
    for N2U in output_results:
        for sigma2 in output_results[N2U]:
            for shift in output_results[N2U][sigma2]:
                if np.isclose(N2U*4+sigma2,600,rtol=0.01):
                    print('VA:',N2U*4+sigma2,'N2U:',N2U,'shift:',shift,'sigma2:',sigma2,'k:',output_results[N2U][sigma2][shift][0],'n:',output_results[N2U][sigma2][shift][1],'adaptation:',output_results[N2U][sigma2][shift][2])
                    
    with open(snakemake.output[0],'wb+') as fout:
        pickle.dump(output_results,fout)

if __name__ == '__main__':
    main()
