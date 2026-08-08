"""Run every replicate of ONE parameter combination, then pool them. One job per cell.

"""

import argparse
import concurrent.futures as futures
import hashlib
import os
import pickle
import sys
import time

import numpy as np
from scipy import stats
from scipy.integrate import quad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from all_allele_cpp import CppAllAlleleSimulation, moments_to_results, BIN_LIMITS, library
from all_allele_model import (EffectSizeDistribution, expected_variance_given_a2,
                              expected_segregating_per_input)

METRICS = ("mean", "variance", "skew")
AGGREGATE_NAME = "all_processed_results_with_mutation_counts.pkl"
SKEW_NAME = "skew_results.pkl"


# --------------------------------------------------------------------------------------
# derived inputs (computed once per cell, not once per replicate)
# --------------------------------------------------------------------------------------

def derive_inputs(N, sigma2, large_2NU, distribution_type="expon"):
    """Split the mutational input, and size the initial population.

    The small-effect input is whatever delivers background variance `sigma2` at steady
    state; `large_2NU` is added on top. 
    """
    small_sdist = stats.expon(scale=1)
    var_per_input = quad(
        lambda a2: expected_variance_given_a2(N=N, a2=a2) * small_sdist.pdf(a2), 0, np.inf)[0]
    small_2NU = sigma2 / var_per_input
    total_2NU = small_2NU + large_2NU
    weight = large_2NU / total_2NU

    dist = EffectSizeDistribution(weight=weight, distribution_type=distribution_type)
    n_seg = (expected_segregating_per_input(N=N, effect_size_distribution=dist)
             * (total_2NU / (2 * N)) * 2 * N)
    return dict(small_2NU=small_2NU, total_2NU=total_2NU, weight=weight,
                mutation_rate=total_2NU / (2 * N), expected_segregating=n_seg)


def replicate_seed(sigma2, large_2NU, shift, replicate):
    """Deterministic per-replicate seed, so any single replicate can be reproduced alone."""
    key = f"{sigma2}|{large_2NU}|{shift}|{replicate}".encode()
    return int(np.frombuffer(hashlib.sha256(key).digest()[:8], dtype=np.uint64)[0] % (2 ** 63))


# --------------------------------------------------------------------------------------
# one replicate
# --------------------------------------------------------------------------------------

def run_one(replicate, sigma2, large_2NU, shift, outdir, derived, N=5000,
            burn_time_N=10, run_time_N=4, distribution_type="expon", overwrite=False):
    target = os.path.join(outdir, f"processed_iteration_{replicate}.pkl")
    if os.path.exists(target) and not overwrite:
        return replicate, 0.0, "cached"

    seed = replicate_seed(sigma2, large_2NU, shift, replicate)
    t0 = time.time()
    sim = CppAllAlleleSimulation(
        N=N, mutation_rate=derived["mutation_rate"], shift=shift, weight=derived["weight"],
        burn_time=burn_time_N * N, distribution_type=distribution_type, seed=seed,
        bin_limits=BIN_LIMITS)
    n_initial = sim.initialize_population(derived["expected_segregating"])
    sim.burn()                       # GIL released here -- this is where the time goes
    sim.shift_optimum()
    sim.force_minor_alleles()
    out = sim.run_recorded(run_time_N * N)
    fixations = sim.fixations.tolist()
    seconds = time.time() - t0

    results = moments_to_results(out["moments"], bin_limits=BIN_LIMITS)
    with open(target, "wb") as fh:
        pickle.dump((results, fixations), fh)
    with open(os.path.join(outdir, f"iteration_{replicate}_meta.pkl"), "wb") as fh:
        pickle.dump(dict(sigma2=sigma2, large_2NU=large_2NU, shift=shift, N=N,
                         replicate=replicate, seed=seed, n_initial=n_initial,
                         engine="cpp", burn_time=burn_time_N * N,
                         generations=run_time_N * N, seconds=seconds, **derived), fh)
    return replicate, seconds, "ran"


# --------------------------------------------------------------------------------------
# pooling
# --------------------------------------------------------------------------------------

def pool(outdir, replicates, sigma2, shift, large_2NU):
    """Pool the cell's replicates into the two files the figure reads."""
    pooled, fixations, skew_by_bin, used = {}, {}, {}, 0

    for replicate in range(replicates):
        path = os.path.join(outdir, f"processed_iteration_{replicate}.pkl")
        if not os.path.exists(path):
            continue
        with open(path, "rb") as fh:
            results, fix = pickle.load(fh)
        if not results:
            continue
        used += 1
        fixations[path] = fix
        generations = sorted(results)
        for bin_value in results[generations[0]]:
            skew_by_bin.setdefault(bin_value, []).append(
                np.array([results[g][bin_value]["skew"] for g in generations]))
        for generation, bins in results.items():
            slot = pooled.setdefault(generation, {})
            for bin_value, metrics in bins.items():
                acc = slot.setdefault(
                    bin_value, {m: {"sum": 0.0, "sum_sq": 0.0, "count": 0} for m in METRICS})
                for metric in METRICS:
                    value = metrics[metric]
                    acc[metric]["sum"] += value
                    acc[metric]["sum_sq"] += value ** 2
                    acc[metric]["count"] += 1

    with open(os.path.join(outdir, AGGREGATE_NAME), "wb") as fh:
        pickle.dump(pooled, fh)          # two consecutive pickles, as the figure expects
        pickle.dump(fixations, fh)

    skew = {"bins": {}}
    for bin_value, arrays in skew_by_bin.items():
        n_gen = min(a.size for a in arrays)
        stacked = np.vstack([a[:n_gen] for a in arrays])
        n = stacked.shape[0]
        ste = 2 * stacked.std(axis=0, ddof=0) / np.sqrt(n) if n > 1 else np.zeros(n_gen)
        skew["bins"][bin_value] = (stacked.mean(axis=0).tolist(), ste.tolist())
    if skew["bins"]:
        # the small-effect bin keeps the original top-level key, for drop-in compatibility
        skew[(sigma2, shift, large_2NU)] = skew["bins"][min(skew["bins"])]
    with open(os.path.join(outdir, SKEW_NAME), "wb") as fh:
        pickle.dump(skew, fh)

    print(f"  pooled {used}/{replicates} replicates, {len(pooled)} generations, "
          f"skew bins {sorted(skew['bins'])}", flush=True)
    return used


# --------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------

def run_combination(sigma2, large_2NU, shift, outdir, replicates=100, threads=2, N=5000,
                    burn_time_N=10, run_time_N=4, distribution_type="expon",
                    overwrite=False):
    os.makedirs(outdir, exist_ok=True)
    library()                                     # build/load the core before threading
    t0 = time.time()
    derived = derive_inputs(N, sigma2, large_2NU, distribution_type)
    print(f"sigma2={sigma2} 2NU_large={large_2NU} shift={shift}: "
          f"total_2NU={derived['total_2NU']:.6g} weight={derived['weight']:.6g} "
          f"expected_segregating={derived['expected_segregating']:.1f} "
          f"({replicates} replicates on {threads} thread(s))", flush=True)

    ran, cached = 0, 0
    with futures.ThreadPoolExecutor(max_workers=threads) as pool_exec:
        jobs = [pool_exec.submit(run_one, r, sigma2, large_2NU, shift, outdir, derived,
                                 N=N, burn_time_N=burn_time_N, run_time_N=run_time_N,
                                 distribution_type=distribution_type, overwrite=overwrite)
                for r in range(replicates)]
        for done, future in enumerate(futures.as_completed(jobs), start=1):
            _replicate, seconds, status = future.result()
            ran += status == "ran"
            cached += status == "cached"
            if done % 10 == 0 or done == replicates:
                elapsed = time.time() - t0
                print(f"  {done}/{replicates} replicates ({elapsed / 60:.1f} min elapsed)",
                      flush=True)

    print(f"  {ran} run, {cached} already present, {(time.time() - t0) / 60:.1f} min",
          flush=True)
    pool(outdir, replicates, sigma2, shift, large_2NU)
    return outdir


if "snakemake" in globals():
    p = snakemake.params                                          # noqa: F821
    run_combination(sigma2=float(p.sigma2), large_2NU=float(p.large_2NU),
                    shift=float(p.shift),
                    outdir=os.path.dirname(os.path.abspath(snakemake.output[0])),  # noqa: F821
                    replicates=int(p.replicates), threads=int(snakemake.threads),  # noqa: F821
                    N=int(p.N), burn_time_N=int(p.burn_time_N),
                    run_time_N=int(p.run_time_N))
elif __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sigma2", type=float, required=True)
    ap.add_argument("--large-2NU", dest="large_2NU", type=float, required=True)
    ap.add_argument("--shift", type=float, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--replicates", type=int, default=100)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--N", type=int, default=5000)
    ap.add_argument("--burn-time-N", dest="burn_time_N", type=int, default=10)
    ap.add_argument("--run-time-N", dest="run_time_N", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    run_combination(**vars(ap.parse_args()))
