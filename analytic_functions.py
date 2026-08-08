# Probability that a large-effect allele fixes, for new and segregating alleles.

import numpy as np
from scipy.integrate import quad
from scipy.optimize import root

# Aligned fraction of the mutational input. The per-S functions below are all
# conditional on an allele being aligned; opposing alleles never fix. 2NU is the
# total influx over both sign classes, so the total_* functions carry this factor.
ALIGNED_FRACTION = 0.5

# Phenotypic variance from one locus of effect a at frequency x, where 2*N*a^2=S
def variance_star(S, x):
    return 2 * S * x * (1 - x)

# Density of alleles at frequency x given effect size a^2, where 2*N*a^2=S
def folded_sojourn_time(S, x, N):
    if x < 0 or x > 1 / 2:
        raise ValueError("folded_sojourn_time defined only for 0 <= x <= 1/2")
    w = 2 * np.exp(-variance_star(S=S, x=x) / 2)
    if x <= 1 / (2 * N):
        # 2N*x * w/(x(1-x)) simplifies to 2N*w/(1-x); stays finite as x -> 0
        return 2 * N * w / (1 - x)
    else:
        return w / (x * (1 - x))

# Expected per-generation change in x, with the mean phenotype `shift` from the
# optimum. sign=-1 flips the direction of selection, for opposing alleles.
def expected_delta_x(S,x,shift,N, sign=1):
    s = expected_S(S=S,x=x,shift=shift,N=N, sign=sign)
    return s*x*(1-x)

# Expected per-generation selection coefficient at frequency x
def expected_S(S,x,shift,N, sign=1):
    a = np.sqrt(S)*sign
    return a*(shift-a*(1/2-x))/(2*N)


# Probability of establishment
def establishment_prob(S,shift,x,N, x_est=1,sign=1):
    s = expected_S(S=S,x=x,shift=shift,N=N, sign=sign)
    if s < 0:
        return 0
    else:
        return (1 - np.exp(-4*N*s*x))/(1 - np.exp(-4*N*s*x_est))

# Deterministic double recursion; an allele reaching frequency 0.5 is taken to fix.
def get_recursion_deterministic(S,x0,D0,sigma2,N,fix_freq=0.5):
    
    x = x0
    D = D0
    a = np.sqrt(S)
    t = 0
    Vs = 2*N
    
    while x<fix_freq:

        dx = a/Vs*x*(1-x)*(D-a*(1/2-x)*(1-D**2/Vs))
        if dx < 0: 
            return 0
        
        x += dx
        D += - D/Vs*sigma2 - 2*a*dx
        t += 1
        
        if D < 0.5:
            return 0

    if x < fix_freq: 
        return 0
    else: 
        return 1
    
# Bisect for the minimum frequency from which a segregating allele reaches fixation.
def min_freq_segregating(S,x,N,shift,sigma2,minimum_freq=0,maximum_freq=1,tolerance=(1/2)**40):
    while maximum_freq - minimum_freq > tolerance:
        mid_freq = (minimum_freq + maximum_freq) / 2
        if get_recursion_deterministic(S,x0=mid_freq,D0=shift,sigma2=sigma2,N=N,fix_freq=0.5):
            maximum_freq = mid_freq
        else:
            minimum_freq = mid_freq
    return minimum_freq

# Bisect for the minimum shift at which a new allele reaches fixation.
def min_shift_new(S,N,sigma2,minimum_shift=0,maximum_shift=100,tolerance=(1/2)**40,account_for_time_establishment=True):
    
    while maximum_shift - minimum_shift > tolerance:
        mid_shift = (minimum_shift + maximum_shift) / 2
        if account_for_time_establishment:
            t_est,x_est = mean_t_til_establishment(p=1/(2*N),shift=mid_shift,a=np.sqrt(S),N=N)
            shift_use = mid_shift*np.exp(-t_est*sigma2/(2*N))
            fixed = get_recursion_deterministic(S,x0=x_est,D0=shift_use,sigma2=sigma2,N=N,fix_freq=0.5)
        else:
            fixed = get_recursion_deterministic(S,x0=1/(2*N),D0=mid_shift,sigma2=sigma2,N=N,fix_freq=0.5)
        if fixed:
            maximum_shift = mid_shift
        else:
            minimum_shift = mid_shift
    return minimum_shift

# Sojourn time times establishment probability, integrated from min_freq to 1/2.
def expected_number_of_fixed_segregating_alleles(S,N,shift,sigma2):
    min_freq = min_freq_segregating(S=S,x=1/(2*N),N=N,shift=shift,sigma2=sigma2)
    if min_freq == 1/2:
        return 0
    else:
        if min_freq < 1/(2*N):
            return quad(lambda x: folded_sojourn_time(S=S,x=x,N=N)*establishment_prob(x=x,S=S,shift=shift,N=N),min_freq,0.5,points=[1/(2*N)])[0]
        else:
            return quad(lambda x: folded_sojourn_time(S=S,x=x,N=N)*establishment_prob(x=x,S=S,shift=shift,N=N),min_freq,0.5)[0]

# As expected_number_of_fixed_segregating_alleles, reweighted by each allele's
# contribution 2*a*(1-x_0) to the mean phenotype.
def expected_contribution_of_fixed_segregating_alleles(S,N,shift,sigma2):
    a = np.sqrt(S)
    min_freq = min_freq_segregating(S=S,x=1/(2*N),N=N,shift=shift,sigma2=sigma2)
    if min_freq == 1/2:
        return 0
    else:
        integrand = lambda x: 2*a*(1-x)*folded_sojourn_time(S=S,x=x,N=N)*establishment_prob(x=x,S=S,shift=shift,N=N)
        if min_freq < 1/(2*N):
            return quad(integrand,min_freq,0.5,points=[1/(2*N)])[0]
        else:
            return quad(integrand,min_freq,0.5)[0]

# Establishment probability integrated over the time the distance, decaying as
# shift*exp(-t*sigma2/(2N)), exceeds min_shift.
def expected_number_of_fixed_new_alleles(S,N,shift,sigma2):
    if shift <= np.sqrt(S)/2:
        return 0
    min_shift = min_shift_new(S=S,N=N,sigma2=sigma2,minimum_shift=0,maximum_shift=shift+1,account_for_time_establishment=True)
    if min_shift > shift:
        return 0
    else:
        t_max = max(0,np.log(shift/min_shift)/sigma2*(2*N))
        t_unstable = max(0,np.log(shift/(np.sqrt(S)/2))/sigma2*(2*N))
        return quad(lambda t: establishment_prob(x=1/(2*N),S=S,shift=shift*np.exp(-t*sigma2/(2*N)),N=N),0,t_max,points=[t_unstable])[0]

# A new allele starts at x_0 = 1/(2N), so its contribution is the constant
# 2*a*(1-x_0) times the expected number that fix.
def expected_contribution_of_fixed_new_alleles(S,N,shift,sigma2):
    a = np.sqrt(S)
    x0 = 1/(2*N)
    return 2*a*(1-x0)*expected_number_of_fixed_new_alleles(S=S,N=N,shift=shift,sigma2=sigma2)

# Integration limits in S, optionally clipped to an effect-size window (S_lo, S_hi];
# the windows are how Figure S11 splits the expectations by effect size. None when
# the clip is empty, which the callers report as 0.
def _S_integration_limits(sdist,S_lo=None,S_hi=None,q_lo=0.0001,q_hi=0.9999):
    lo, hi = float(sdist.ppf(q_lo)), float(sdist.ppf(q_hi))
    if S_lo is not None:
        lo = max(lo, float(S_lo))
    if S_hi is not None and np.isfinite(S_hi):
        hi = min(hi, float(S_hi))
    return None if lo >= hi else (lo, hi)

# The per-S counts and contributions below, averaged over sdist and expressed per
# unit of the total mutational input 2NU. S_lo/S_hi restrict to an effect-size window.
def total_number_of_fixed_segregating_alleles(sdist,N,shift,sigma2,S_lo=None,S_hi=None):
    limits = _S_integration_limits(sdist,S_lo,S_hi)
    if limits is None:
        return 0.0
    return ALIGNED_FRACTION*quad(lambda S: expected_number_of_fixed_segregating_alleles(S=S,N=N,shift=shift,sigma2=sigma2)*sdist.pdf(S),*limits)[0]

def total_number_of_fixed_new_alleles(sdist,N,shift,sigma2,S_lo=None,S_hi=None):
    limits = _S_integration_limits(sdist,S_lo,S_hi)
    if limits is None:
        return 0.0
    return ALIGNED_FRACTION*quad(lambda S: expected_number_of_fixed_new_alleles(S=S,N=N,shift=shift,sigma2=sigma2)*sdist.pdf(S),*limits)[0]

def total_contribution_of_fixed_segregating_alleles(sdist,N,shift,sigma2,S_lo=None,S_hi=None):
    limits = _S_integration_limits(sdist,S_lo,S_hi)
    if limits is None:
        return 0.0
    return ALIGNED_FRACTION*quad(lambda S: expected_contribution_of_fixed_segregating_alleles(S=S,N=N,shift=shift,sigma2=sigma2)*sdist.pdf(S),*limits)[0]

def total_contribution_of_fixed_new_alleles(sdist,N,shift,sigma2,S_lo=None,S_hi=None):
    limits = _S_integration_limits(sdist,S_lo,S_hi)
    if limits is None:
        return 0.0
    return ALIGNED_FRACTION*quad(lambda S: expected_contribution_of_fixed_new_alleles(S=S,N=N,shift=shift,sigma2=sigma2)*sdist.pdf(S),*limits)[0]

############### FUNCTIONS FOR THE NUMBER OF ALLELES THAT ESTABLISH ######################################
# An allele establishes if it escapes stochastic loss while still beneficial. These
# are the expected_number_of_fixed_* functions with the fixation threshold dropped.
def expected_number_of_established_segregating_alleles(S,N,shift,sigma2):
    integrand = lambda x: folded_sojourn_time(S=S,x=x,N=N)*establishment_prob(x=x,S=S,shift=shift,N=N)
    return quad(integrand,0,0.5,points=[1/(2*N)])[0]

# Integrated over the window in which a new allele is beneficial, i.e. while the
# decaying distance exceeds sqrt(S)/2.
def expected_number_of_established_new_alleles(S,N,shift,sigma2):
    if shift <= np.sqrt(S)/2:
        return 0
    t_unstable = max(0,np.log(shift/(np.sqrt(S)/2))/sigma2*(2*N))
    return quad(lambda t: establishment_prob(x=1/(2*N),S=S,shift=shift*np.exp(-t*sigma2/(2*N)),N=N),0,t_unstable)[0]

# Averaged over sdist, per unit of the total mutational input.
def total_number_of_established_segregating_alleles(sdist,N,shift,sigma2):
    return ALIGNED_FRACTION*quad(lambda S: expected_number_of_established_segregating_alleles(S=S,N=N,shift=shift,sigma2=sigma2)*sdist.pdf(S),sdist.ppf(0.0001),sdist.ppf(0.9999))[0]

def total_number_of_established_new_alleles(sdist,N,shift,sigma2):
    return ALIGNED_FRACTION*quad(lambda S: expected_number_of_established_new_alleles(S=S,N=N,shift=shift,sigma2=sigma2)*sdist.pdf(S),sdist.ppf(0.0001),sdist.ppf(0.9999))[0]

############### DISTRIBUTION OF EFFECT SIZES AMONG FIXED ALLELES, f(a) ###############################
# With g = sdist and X(a) the per-effect-size expected number that fix, the density
# of effect sizes among fixed alleles is f(a) = g(a) X(a) / X, X = \int g(a) X(a) da.
# In terms of S = 2*N*a^2, and independent of the mutational input, which cancels.

def _trapz(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    return float(np.sum((y[1:] + y[:-1]) / 2.0 * np.diff(x)))

# X(S) = segregating + new, unscaled by the mutational input. A min_shift_interp
# reads the new-allele threshold from a lookup instead of re-bisecting at every S.
def expected_number_of_fixed_alleles(S,N,shift,sigma2,min_shift_interp=None):
    n_seg = expected_number_of_fixed_segregating_alleles(S=S,N=N,shift=shift,sigma2=sigma2)
    if min_shift_interp is None:
        n_new = expected_number_of_fixed_new_alleles(S=S,N=N,shift=shift,sigma2=sigma2)
    else:
        # lazy import avoids a circular import (build_min_shift_lookup imports this module)
        from build_min_shift_lookup import expected_number_of_fixed_new_alleles_lookup
        n_new = expected_number_of_fixed_new_alleles_lookup(S=S,N=N,shift=shift,sigma2=sigma2,min_shift_interp=min_shift_interp)
    return n_seg + n_new

# Returns (S_grid, f, X): f normalized to integrate to 1, X the per-2NU total number
# of fixations. f is all zeros and X == 0 if no effect size fixes at this shift.
def fixed_effect_size_distribution(sdist,N,shift,sigma2,S_grid=None,n_grid=200,q_lo=0.0001,q_hi=0.9999,min_shift_interp=None):
    if S_grid is None:
        S_grid = np.linspace(float(sdist.ppf(q_lo)),float(sdist.ppf(q_hi)),n_grid)
    S_grid = np.asarray(S_grid,dtype=float)
    X_of_S = np.array([expected_number_of_fixed_alleles(S=S,N=N,shift=shift,sigma2=sigma2,min_shift_interp=min_shift_interp) for S in S_grid])
    weight = np.asarray(sdist.pdf(S_grid),dtype=float)*X_of_S   # g(S) * X(S), unnormalized
    X = _trapz(weight,S_grid)                                   # total fixations (per 2NU)
    f = weight/X if X > 0 else np.zeros_like(weight)
    return S_grid, f, X

# Running trapezoidal integral of f, normalized to end at 1.
def _cdf_from_density(S_grid, f):
    cdf = np.concatenate([[0.0], np.cumsum((f[1:]+f[:-1])/2.0*np.diff(S_grid))])
    if cdf[-1] > 0:
        cdf = cdf/cdf[-1]
    return cdf

# Quantiles read off the CDF, keeping only strictly increasing points so np.interp
# can invert it; the mean is \int S f(S) dS.
def _boxplot_from_density(S_grid, f, quantiles=(0.025,0.25,0.5,0.75,0.975)):
    cdf = _cdf_from_density(S_grid, f)
    keep = np.concatenate([[True], np.diff(cdf) > 0])
    S_keep, cdf_keep = S_grid[keep], cdf[keep]
    keys = ("whislo","q1","med","q3","whishi")
    stats_out = {k: float(np.interp(q,cdf_keep,S_keep)) for k,q in zip(keys,quantiles)}
    stats_out["mean"] = _trapz(S_grid*f, S_grid)
    return stats_out

# A matplotlib Axes.bxp item for f(S), plus its 'mean'. None when no effect size fixes.
def fixed_effect_size_boxplot_stats(sdist,N,shift,sigma2,n_grid=200,quantiles=(0.025,0.25,0.5,0.75,0.975),min_shift_interp=None):
    S_grid, f, X = fixed_effect_size_distribution(sdist=sdist,N=N,shift=shift,sigma2=sigma2,n_grid=n_grid,min_shift_interp=min_shift_interp)
    if X <= 0:
        return None
    return _boxplot_from_density(S_grid, f, quantiles)

# Grid, density, CDF, total count X and boxplot stats from a single sweep over the S
# grid, so the expensive per-S X(S) is evaluated once. None when no effect size fixes.
def fixed_effect_size_summary(sdist,N,shift,sigma2,n_grid=200,quantiles=(0.025,0.25,0.5,0.75,0.975),min_shift_interp=None):
    S_grid, f, X = fixed_effect_size_distribution(sdist=sdist,N=N,shift=shift,sigma2=sigma2,n_grid=n_grid,min_shift_interp=min_shift_interp)
    if X <= 0:
        return None
    return {
        'S_grid': S_grid,
        'pdf': f,
        'cdf': _cdf_from_density(S_grid, f),
        'X': X,
        'boxplot': _boxplot_from_density(S_grid, f, quantiles),
    }

############### TIME FOR A NEW ALLELE TO ESTABLISH ######################################
def b(z,shift,a,N):
    return np.float128(z*(1-z)/(2*N))

def aa(z,shift,a,N):
    Vs = 2*N
    return np.float128(a/Vs*z*(1-z)*(shift-a*(1/2-z)*(1-shift**2/Vs)))

def psi(y,shift,a,N):
    integral = quad(lambda z: aa(z=z,shift=shift,a=a,N=N)/b(z=z,shift=shift,a=a,N=N),0,y)[0]
    return np.exp(-2*integral)

def p1(p,shift,a,N):
    top = quad(lambda y: psi(y=y,shift=shift,a=a,N=N),0,p)[0]
    bottom = quad(lambda y: psi(y=y,shift=shift,a=a,N=N),0,1)[0]
    return top/bottom

def p0(p,shift,a,N):
    return 1-p1(p=p,shift=shift,a=a,N=N)

def t_star(x,p,shift,a,N):
    g = b(z=x,shift=shift,a=a,N=N)*psi(y=x,shift=shift,a=a,N=N)
    f = 2*p1(p=x,shift=shift,a=a,N=N)

    if x <= p:
        h = p0(p=p,shift=shift,a=a,N=N)/p1(p=p,shift=shift,a=a,N=N)
        i = quad(lambda y: psi(y=y,shift=shift,a=a,N=N),0,x,points=[1/(2*N)])[0]
    else:
        h = 1
        ### Should this go to x_est instead?
        i = quad(lambda y: psi(y=y,shift=shift,a=a,N=N),x,1,points=[1/(2*N)])[0]
    return f*h*i/g

def find_xc_for_minimum_establishment_prob(S,shift,minimum_establishment_prob,N,x_initial = 0.001,sign=1):

    if x_initial > 0.4:
        return 0.4
        
    sol = root(lambda x: establishment_prob(S=S,shift=shift,x=x,sign=sign,N=N)-minimum_establishment_prob,x_initial)
    if not sol.success:
        return find_xc_for_minimum_establishment_prob(S=S,shift=shift,minimum_establishment_prob=minimum_establishment_prob,N=N,x_initial = x_initial*2,sign=sign)
    else:
        return sol.x[0]
    
def mean_t_til_establishment(p,shift,a,N):
    # p is initial frequency of the new allele
    sign = 1 if a > 0 else -1
    establishment_x = find_xc_for_minimum_establishment_prob(S=a**2,shift=shift,minimum_establishment_prob=0.95,N=N,x_initial=1/(2*N),sign=sign)
    if establishment_x >= 0.4:
        return np.inf, 1/(2*N)
    if p > establishment_x:
        return 0
    g = quad(lambda x: t_star(x=x,p=p,shift=shift,a=a,N=N),0.0,establishment_x,points=[p])
    return g[0],establishment_x

#####################################################################################################################

### Equations from supplement

def normalization_constant(S,N):
    return 1/(quad(lambda x: folded_sojourn_time(S=S,x=x,N=N),0,1/2,points=[1/(2*N)])[0])

from scipy.special import expi
    
def analytic_normalization_constant(S,N):
    return 1/(quad(lambda x: folded_sojourn_time(S=S,x=x,N=N),0,1/2,points=[1/(2*N)])[0])

def probability_of_establishment_integral(S,N,shift,c):
    return c*quad(lambda x: folded_sojourn_time(S=S,x=x,N=N)*establishment_prob(x=x,S=S,shift=shift,N=N),0,1/2,points=[1/(2*N)])[0]

def probability_of_fixation_integral(S,N,shift,sigma2,c):
    x_c = np.exp(-np.sqrt(S)*(shift-np.sqrt(S)/2)/sigma2)
    if np.isnan(x_c):
        return 0
    elif x_c > 1/2:
        return 0
    else:
        return c*quad(lambda x: folded_sojourn_time(S=S,x=x,N=N)*establishment_prob(x=x,S=S,shift=shift,N=N),x_c,1/2,points=[1/(2*N)])[0]
    
def Ei(x):
    return expi(x)

def h(y):
    return -Ei(-y)

def equation_S26(c,x_est,x_msdb,x_c):
    return c*(h(x_c/x_msdb)-h(x_c/x_msdb+x_c/x_est))
    
# When x_c << x_est and x_c << x_msdb, we can use the following approximation
def equation_S27(x_est,x_msdb,x_c,c):
    return c*(np.log(1+x_msdb/x_est)- x_c/x_est)

def equation_S28(x_est,x_msdb,x_c,zeta,c):
    return c/(zeta+x_est/x_msdb)*np.exp(-x_c/x_est*(zeta+x_est/x_msdb))

def equation_S36(sigma2, N, a, shift):
    V_S = 2 * N
    L = shift
    T = V_S/sigma2*np.log((shift/(np.log(2*N)*sigma2/a + a)))
    T = np.asarray(T, dtype=float)
    pref = (V_S / sigma2) * np.exp(a**2 / V_S)
    upper = -(2 * a * L / V_S) * np.exp(-sigma2 * T / V_S)
    lower = -(2 * a * L / V_S)
    value = T + pref * (expi(upper) - expi(lower))
    T_max = V_S/sigma2*np.log(shift)
    return value/T_max
    
def evaluate_equations(shift,a,N,sigma2):
    x_msdb = 1/(a**2)
    x_est = 1/((2*a)*(shift-a/2))
    x_c = np.exp(-a*(shift-a)/sigma2)
    
    normalization_real = normalization_constant(S=a**2,N=N)
    # x_msdb = 1/a^2 ignores the boundary at x < 1/(2N), leaving the approximate
    # x_msdb used below off by a factor of 2 from the exact normalization.
    normalization = 2*normalization_real
    S26 = equation_S26(x_est=x_est,x_msdb=x_msdb,x_c=x_c,c=normalization)
    S27 = equation_S27(x_est=x_est,x_msdb=x_msdb,x_c=x_c,c=normalization)#probability_of_fixation_integral(S=a**2,N=N,shift=shift,c=normalization,sigma2=sigma2)
    S28 = equation_S28(x_est=x_est,x_msdb=x_msdb,x_c=x_c,zeta=1/3,c=normalization)#probability_of_fixation_integral(S=a**2,N=N,shift=shift,c=normalization,sigma2=sigma2)
    S36 = equation_S36(shift=shift,a=a,N=N,sigma2=sigma2)
    return S26, S27, S28, S36

