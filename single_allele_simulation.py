import sys
sys.path.append('n_dim_code/')

import pickle
import numpy as np

import os
import warnings

import matplotlib as mpl
from matplotlib import pyplot as plt

from collections import defaultdict as ddict
from collections import OrderedDict as odict
from collections import namedtuple as nt
from scipy.optimize import minimize_scalar
from scipy.optimize import root
from math import erf

from scipy import stats
from scipy.integrate import quad
from scipy.optimize import broyden1 as minimize
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from scipy.special import dawsn
from scipy.special import erfi
from scipy.optimize import root

from scipy import stats

from copy import deepcopy
import seaborn as sns


def variance_star(S,x):
    return 2*S*x*(1-x)

def folded_sojourn_time(S,x,N=5000):
    if x < 0:
        raise ValueError
    elif x > 1/2:
        raise ValueError
    else:
        value = 2 * np.exp(-variance_star(S=S,x=x)/2) / (x * (1 - x))
        if x <= 1/(2*N):
            return 2 * N * x * value
        else:
            return value
        
def get_recursion(S,x0,D0,sigma2,sign,N=5000):

    x = x0
    D = D0
    a = sign*np.sqrt(S)
    t = 0
    Vs = 2*N
    
    x_trajectory = [x0]
    while x < 1 and x > 0:

        a = np.sqrt(S)*sign
        dx = a/Vs*x*(1-x)*(D-a*(1/2-x)*(1-D**2/Vs))

        x = np.random.binomial(2*N,min(1,max(0,x+dx)))/(2*N)
        D += - D/Vs*sigma2 - 2*a*dx
    x_trajectory.append(x)
            
    return x_trajectory

def cumulant(N, S, y):
    if y >= 0.5:
        return 1
    elif y <= 0:
        return 0
    top = quad(lambda x: folded_sojourn_time(S=S, x=x), 0, y, points=[1 / (2 * N)])[0]
    bottom = quad(lambda x: folded_sojourn_time(S=S, x=x), 0, 1 / 2, points=[1 / (2 * N)])[0]
    return top / bottom

def get_frequency(N, S):
    y0 = np.random.random()
    x = root(lambda y: cumulant(N=N, S=S, y=y) - y0, 1 / (2 * N)).x[0]
    if x < 0:
        y0, x = get_frequency(N=N, S=S)
    return y0, x

def main():
    N = snakemake.params.N
    trials = snakemake.params.trials
    shift = snakemake.params.shift
    sigma2 = snakemake.params.sigma2
    S = snakemake.params.S

    results = {shift: {sigma2: {S: {sign: {} for sign in [1]}}}}
    sign = 1
    for i in range(trials):
        if i % 100 == 0:
            print(i)
        if i not in results[shift][sigma2][S][sign].keys():
            results[shift][sigma2][S][sign][i] = get_recursion(S=S, x0=get_frequency(N=N, S = S)[1],D0=shift,sigma2=sigma2,sign=sign)
            
    with open(snakemake.output.results_file,'wb') as fout:
        pickle.dump(results,fout)

if __name__ == '__main__':
    main()
    
