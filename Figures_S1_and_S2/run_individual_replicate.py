"""Run ONE full individual-based replicate: 10N burn-in, shift the optimum, 4N recorded.
"""

import argparse
import hashlib
import os
import pickle
import sys
import time

import numpy as np
from scipy import stats
from scipy.integrate import quad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from all_allele_model import (EffectSizeDistribution, expected_variance_given_a2,
                              expected_segregating_per_input)
from individual_cpp import IndividualSimulation

LARGE_EFFECT_MIN_A = 10.0     # |a| at or above which a fixation counts as large-effect


def derive_inputs(N, sigma2, large_2NU, distribution_type="expon"):
    """Split the mutational input and size the initial population (same as the all-allele
    pipeline, except the rate is expressed PER GAMETE, which is what create_gamete draws)."""
    small_sdist = stats.expon(scale=1)
    var_per_input = quad(
        lambda a2: expected_variance_given_a2(N=N, a2=a2) * small_sdist.pdf(a2), 0, np.inf)[0]
    small_2NU = sigma2 / var_per_input
    total_2NU = small_2NU + large_2NU
    weight = large_2NU / total_2NU
    mutation_rate = total_2NU / (2 * N)
    dist = EffectSizeDistribution(weight=weight, distribution_type=distribution_type)
    n_seg = expected_segregating_per_input(N=N, effect_size_distribution=dist) * mutation_rate * 2 * N
    return dict(small_2NU=small_2NU, total_2NU=total_2NU, weight=weight,
                mutation_rate=mutation_rate, expected_segregating=n_seg)


def replicate_seed(sigma2, large_2NU, shift, replicate):
    key = f"individual|{sigma2}|{large_2NU}|{shift}|{replicate}".encode()
    return int(np.frombuffer(hashlib.sha256(key).digest()[:8], dtype=np.uint64)[0] % (2 ** 63))


def run(sigma2, large_2NU, shift, replicate, outdir, N=5000, burn_time_N=10,
        run_time_N=4, distribution_type="expon"):
    os.makedirs(outdir, exist_ok=True)
    derived = derive_inputs(N, sigma2, large_2NU, distribution_type)
    seed = replicate_seed(sigma2, large_2NU, shift, replicate)
    burn, recorded = burn_time_N * N, run_time_N * N

    print(f"individual replicate {replicate}: sigma2={sigma2} 2NU_large={large_2NU} "
          f"shift={shift} N={N}", flush=True)
    print(f"  total_2NU={derived['total_2NU']:.4g} weight={derived['weight']:.6g} "
          f"mu/gamete={derived['mutation_rate']:.6g} "
          f"E[n_seg]={derived['expected_segregating']:.0f} seed={seed}", flush=True)

    t0 = time.time()
    sim = IndividualSimulation(N=N, mutation_rate=derived["mutation_rate"], shift=shift,
                               weight=derived["weight"],
                               distribution_type=distribution_type, seed=seed)
    n_initial = sim.initialize_population(derived["expected_segregating"])
    print(f"  initialised {n_initial} mutations, {sim.n_segregating} segregating "
          f"({time.time() - t0:.1f}s)", flush=True)

    sim.burn(burn)
    t_burn = time.time() - t0
    print(f"  burn-in {burn} generations done ({t_burn / 60:.1f} min, "
          f"{t_burn / burn * 1000:.1f} ms/gen), {sim.n_segregating} segregating", flush=True)

    sim.shift_optimum()
    t1 = time.time()
    out = sim.run_recorded(recorded)
    t_run = time.time() - t1
    print(f"  recorded {recorded} generations ({t_run / 60:.1f} min, "
          f"{t_run / recorded * 1000:.1f} ms/gen)", flush=True)

    effects, times = sim.fixations
    after_shift = times > 0
    n_large = int(np.sum((effects >= LARGE_EFFECT_MIN_A) & after_shift))
    print(f"  fixations: {len(effects)} total, {int(after_shift.sum())} after the shift, "
          f"{n_large} large-effect aligned", flush=True)

    payload = dict(sigma2=sigma2, large_2NU=large_2NU, shift=shift, N=N, replicate=replicate,
                   seed=seed, engine="cpp-individual", n_initial=n_initial,
                   burn=burn, recorded=recorded, n_large_fixations=n_large,
                   fixation_effects=effects, fixation_times=times,
                   mean=out["mean"], variance=out["variance"], skew=out["skew"],
                   n_segregating=out["n_segregating"],
                   fixed_background=out["fixed_background"],
                   seconds=time.time() - t0, **derived)
    path = os.path.join(outdir, f"individual_iteration_{replicate}.pkl")
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)
    print(f"  wrote {path} ({(time.time() - t0) / 60:.1f} min total)", flush=True)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sigma2", type=float, required=True)
    ap.add_argument("--large-2NU", dest="large_2NU", type=float, required=True)
    ap.add_argument("--shift", type=float, required=True)
    ap.add_argument("--replicate", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--N", type=int, default=5000)
    ap.add_argument("--burn-time-N", dest="burn_time_N", type=int, default=10)
    ap.add_argument("--run-time-N", dest="run_time_N", type=int, default=4)
    run(**vars(ap.parse_args()))
