import sys
sys.path.append('/insomnia001/home/wm2377/cdwm/large_effect_alleles/scripts/individual_classes.py')
import individual_classes as classes
import pickle
import numpy as np
from scipy import stats
from scipy.integrate import quad
import os
from matplotlib import pyplot as plt
import scipy
from statsmodels.nonparametric.smoothers_lowess import lowess
from collections import defaultdict as ddict
#import mpatches
import matplotlib as mpl
import matplotlib.patches as mpatches
#import lines2D
import matplotlib.lines as mlines
# simulation_results = "/insomnia001/home/wm2377/cdwm/large_effect_alleles/data/all_allele_simulations/sigma2_40/LargeN2U_0.001/shift_80/iteration_20.pkl"

# Calculate the skew of the genetic variance contributed by alleles at each time point
skew_results = {}
sigma2 = snakemake.params.sigma2
shift = snakemake.params.shift
N2U = snakemake.params.large_2NU

print(sigma2, shift, N2U)
sys.stdout.flush()
file_dict_i = {}
j = 0
for i in range(100):
    try:
        simulation_results = f"/insomnia001/home/wm2377/cdwm/large_effect_alleles/data/all_allele_simulations/sigma2_{sigma2}/LargeN2U_{N2U}/shift_{shift}/iteration_{i}.pkl"
        skew_dict = {}
        with open(simulation_results, 'rb') as f:
            while True:
                try:
                    g = pickle.load(f)
                except EOFError:
                    break
                # Process the loaded object 'g' here
                for t in g.keys():
                    skew = 0
                    for (a,x) in g[t]['mutations']:
                        if np.abs(a) < 10:
                            skew += a**3 * x * (1 - x) * (1 - 2 * x)
                    skew_dict[t] = skew
        file_dict_i[i] = skew_dict
        j += 1
        print(j)
        sys.stdout.flush()
        
    except FileNotFoundError:
        print(f"File not found: {simulation_results}")
        sys.stdout.flush()
        continue
m = {t:0 for t in range(20000)}
s = {t:0 for t in range(20000)}
n = {t:0 for t in range(20000)}
for i in file_dict_i:
    skew_dict = file_dict_i[i]
    for time in np.sort(list(skew_dict.keys())):
        m[time] += skew_dict[time]
        n[time] += 1
        s[time] += skew_dict[time]**2
mean = [m[t]/n[t] if n[t] > 0 else np.nan for t in range(20000)]
ste = [2*np.sqrt((s[t]/n[t] - (m[t]/n[t])**2)/n[t]) if n[t] > 0 else np.nan for t in range(20000)]
skew_results[(sigma2,shift,N2U)] = (mean,ste)

with open(snakemake.output.processed_results, 'wb') as f:
    pickle.dump(skew_results, f)
    
