"""Run every replicate of ONE individual-based parameter combination, threaded.

"""

import argparse
import concurrent.futures as futures
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from individual_cpp import library
from run_individual_replicate import LARGE_EFFECT_MIN_A, derive_inputs, run

SUMMARY_NAME = "individual_summary.pkl"


def _one(args):
    replicate, kw = args
    target = os.path.join(kw["outdir"], f"individual_iteration_{replicate}.pkl")
    if os.path.exists(target):
        return replicate, "cached"
    run(replicate=replicate, **kw)
    return replicate, "ran"


def summarise(outdir, replicates, sigma2, large_2NU, shift):
    """Pool the cell's replicates into what Figure S1 needs: large-effect fixation counts."""
    counts, used = [], 0
    for replicate in range(replicates):
        path = os.path.join(outdir, f"individual_iteration_{replicate}.pkl")
        if not os.path.exists(path):
            continue
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        counts.append(payload["n_large_fixations"])
        used += 1

    counts = np.array(counts, dtype=float)
    summary = dict(sigma2=sigma2, large_2NU=large_2NU, shift=shift,
                   n_replicates=used, counts=counts,
                   mean=float(counts.mean()) if used else np.nan,
                   ste=float(counts.std(ddof=1) / np.sqrt(used)) if used > 1 else 0.0,
                   large_effect_min_a=LARGE_EFFECT_MIN_A)
    with open(os.path.join(outdir, SUMMARY_NAME), "wb") as fh:
        pickle.dump(summary, fh)
    print(f"  pooled {used}/{replicates}: mean large-effect fixations "
          f"{summary['mean']:.3f} +/- {summary['ste']:.3f}", flush=True)
    return summary


def run_cell(sigma2, large_2NU, shift, outdir, replicates=100, threads=2, N=5000,
             burn_time_N=10, run_time_N=4):
    """Run every replicate of one cell on `threads` threads, then pool them."""
    os.makedirs(outdir, exist_ok=True)
    library()                                  # build/load the core before threading
    kw = dict(sigma2=sigma2, large_2NU=large_2NU, shift=shift, outdir=outdir, N=N,
              burn_time_N=burn_time_N, run_time_N=run_time_N)

    t0 = time.time()
    print(f"cell sigma2={sigma2} 2NU_large={large_2NU} shift={shift}: "
          f"{replicates} replicates on {threads} thread(s)", flush=True)
    done = 0
    with futures.ThreadPoolExecutor(max_workers=threads) as pool:
        for _replicate, _status in pool.map(_one, [(r, kw) for r in range(replicates)]):
            done += 1
            if done % 10 == 0 or done == replicates:
                elapsed = (time.time() - t0) / 60
                print(f"  {done}/{replicates} replicates, {elapsed:.1f} min elapsed, "
                      f"~{elapsed / done * (replicates - done):.1f} min remaining", flush=True)

    summary = summarise(outdir, replicates, sigma2, large_2NU, shift)
    print(f"cell done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return summary


def main():
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
    run_cell(**vars(ap.parse_args()))


if "snakemake" in globals():
    _p = snakemake.params                                                   # noqa: F821
    run_cell(sigma2=float(_p.sigma2), large_2NU=float(_p.large_2NU),
             shift=float(_p.shift),
             outdir=os.path.dirname(os.path.abspath(snakemake.output[0])),  # noqa: F821
             replicates=int(_p.replicates), threads=int(snakemake.threads), # noqa: F821
             N=int(_p.N), burn_time_N=int(_p.burn_time_N),
             run_time_N=int(_p.run_time_N))


if __name__ == "__main__":
    main()
