import pickle
import numpy as np

from scipy import stats
from scipy.integrate import quad
from scipy.optimize import root
import time
from math import ceil
from math import floor

def variance_star(S, x):
    return 2 * S * x * (1 - x)


def folded_sojourn_time(S, x, N):
    if x < 0:
        raise ValueError
    elif x > 1 / 2:
        raise ValueError
    else:
        value = 2 * np.exp(-variance_star(S=S, x=x) / 2) / (x * (1 - x))
        if x <= 1 / (2 * N):
            return 2 * N * x * value
        else:
            return value

def S_cdf(z,prob_segregating, N, sdist):
    # sojourn-weighted cumulative over effect sizes in [sdist.ppf(0), z], normalized
    # by prob_segregating. Lower bound matches the prob_segregating integral
    # (was a hardcoded 100, which only happened to fit sdist = expon(loc=100)).
    return quad(lambda S: sdist.pdf(S)*quad(lambda x: folded_sojourn_time(S=S,x=x,N=N),0,1/2,points=[1/(2*N)])[0],sdist.ppf(0),z)[0]/prob_segregating

def get_S(prob_segregating,N,sdist,rng=None):
    # inverse-transform sample S from the segregating-site distribution: find the
    # effect size whose sojourn-weighted CDF equals a uniform draw.
    if rng is None:
        rng = np.random.default_rng()
    y = rng.random()
    return root(lambda z: S_cdf(z,prob_segregating,N,sdist) - y, sdist.ppf(y)).x[0]


def cumulant(S, y,N):
    if y >= 0.5:
        return 1
    elif y <= 0:
        return 0
    top = quad(lambda x: folded_sojourn_time(S=S, x=x, N=N), 0, y, points=[1 / (2 * N)])[0]
    bottom = quad(lambda x: folded_sojourn_time(S=S, x=x, N=N), 0, 1 / 2, points=[1 / (2 * N)])[0]
    return top / bottom

def get_frequency(S,N,rng=None):
    # inverse-transform sample a frequency from the steady-state sojourn density
    # for effect size S; retry (without recursion) if root finding undershoots 0.
    if rng is None:
        rng = np.random.default_rng()
    x = -1
    while x < 0:
        y0 = rng.random()
        x = root(lambda y: cumulant(S=S, y=y, N=N) - y0, 1 / (2 * N)).x[0]
    return y0, x


def generate_alleles(n,prob_segregating,N,sdist,rng=None):
    # print(f'generating {n} segregating alleles')
    # pass rng (a np.random.Generator) for reproducible standing variation
    if rng is None:
        rng = np.random.default_rng()
    mutations = []
    for nn in range(n):
        S = get_S(prob_segregating=prob_segregating,N=N,sdist=sdist,rng=rng)
        y0, x = get_frequency(S=S,N=N,rng=rng)
        mutations.append((x, S, y0))
    return mutations

def generate_lookuptable(sdist,sdist_name):
    N = 5000
    prob_segregating = quad(lambda S: quad(lambda x: folded_sojourn_time(S=S,x=x,N=N),0,1/2,points=[1/(2*N)])[0]*sdist.pdf(S),sdist.ppf(0),sdist.ppf(0.9999))[0]
    lookuptable = {}
    for y in np.linspace(0,1,1001):
        lookuptable[y] = root(lambda z: S_cdf(z,prob_segregating,N,sdist) - y, sdist.ppf(y)).x[0]
    with open(f'lookuptable_{sdist_name}.pickle','wb') as fout:
        pickle.dump(lookuptable,fout)

def get_S_from_lookuptable(sdist,lookuptable):
    y = np.random.random()
    y_floor = floor(y*1000)/1000
    y_ceil = ceil(y*1000)/1000
    y_diff = (y-y_floor)/0.001

    S_floor = lookuptable[y_floor]
    S_ceil = lookuptable[y_ceil]
    S = S_floor + (S_ceil-S_floor)*y_diff
    return S

