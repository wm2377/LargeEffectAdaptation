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

def main():
    
    # get args
    args = snakemake.params
    data = {}
    for input_path in snakemake.input.results_file_path:
        with gzip.open(input_path,'rb') as fin:
            data_part = pickle.load(fin)
        for N2U_key in data_part.keys():
            if N2U_key not in data.keys():
                data[N2U_key] = data_part[N2U_key]
            for shift_key in data_part[N2U_key].keys():
                if shift_key not in data[N2U_key].keys():
                    data[N2U_key][shift_key] = data_part[N2U_key][shift_key]
                for sigma2_key in data_part[N2U_key][shift_key].keys():
                    if sigma2_key not in data[N2U_key][shift_key].keys():
                        data[N2U_key][shift_key][sigma2_key] = data_part[N2U_key][shift_key][sigma2_key]
                    else:
                        raise ValueError("Overlapping keys in input data parts")
    
    with gzip.open(snakemake.output.processed_data_file_path,'wb') as fout:
        pickle.dump(data,fout)
        
if __name__ == "__main__":
    main()