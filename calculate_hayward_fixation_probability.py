from common_functions import fixation_probability_steady_state
from analytic_functions import folded_sojourn_time
import numpy as np
from scipy.integrate import quad
import pickle

def non_linear_change_in_x(x, a, shift, Va):
    """
    Computes the non-linear change in allele frequency x given selection coefficient a,
    shift, and variance Va.
    """
    exponent_term = shift*a/Va
    F_d_times_a = (np.exp(exponent_term) - 1)/(1+x*(np.exp(exponent_term) - 1))
    delta_x = F_d_times_a * x * (1 - x)
    x_new = x + delta_x
    x_new = np.clip(x_new, 0, 1)  # Ensure x_new is within [0, 1]
    return x_new

def hayward_fixation_probability(x, a, shift, Va):
    """
    Computes the fixation probability using the Hayward method.
    """
    x_new = non_linear_change_in_x(x, a, shift, Va)
    return fixation_probability_steady_state(x=x_new, a=a)


def integrate_hayward_fixation_probability(a, shift, Va, N):
    """
    Integrates the Hayward fixation probability over the initial allele frequency distribution.
    """
    
    result = quad(lambda x: hayward_fixation_probability(x, a, shift, Va) * folded_sojourn_time(S = a**2, x = x, N = N), 0, 1/2, points = [1/(2*N)])[0]
    normalization_constant = quad(lambda x: folded_sojourn_time(S = a**2, x = x, N = N), 0, 1/2, points = [1/(2*N)])[0]
    result /= normalization_constant  # Normalize the result
    return result

def main(snakemake):
    # Load parameters from snakemake
    a2 = snakemake.params["S"]
    Va = snakemake.params["Va"]
    shift = snakemake.params["shift"]
    N = snakemake.params["N"]
    sign = snakemake.params["sign"]
    output_file = snakemake.output[0]

    if not isinstance(sign, int):
        if sign == "pos":
            sign = 1
        elif sign == "neg":
            sign = -1
        else:
            raise ValueError(f"Invalid sign value: {sign}. Must be 'pos' or 'neg'.")
    

    # Compute the Hayward fixation probability
    hayward_fixation_prob = integrate_hayward_fixation_probability(a=np.sqrt(a2)*sign, shift=shift, Va=Va, N=N)
    results = {
        "parameters": {
            "a2": a2,
            "Va": Va,
            "shift": shift,
            "N": N,
            "sign": sign
        },
        "fixation_probability": {
            "fixation": hayward_fixation_prob
        }
    }
    # Save the Hayward results to the output file
    with open(output_file, "wb") as fh:
        pickle.dump(results, fh)

# Under Snakemake's `script:` directive the `snakemake` object is injected into
# globals; run automatically in that case (but stay importable for testing).
if "snakemake" in globals():
    main(snakemake)  # noqa: F821