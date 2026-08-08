"""
Snakemake script: run a BATCH of phase-space grid cells serially in one job.

One cluster job per grid cell would be thousands of mostly-seconds-long jobs (4181 for
Figure 6 alone), so phase_space_grids.grid_batches() packs the cells into batches and this
script runs one batch's cells one after another. Each cell is an ordinary run_simulation-
style run -- same engine, same pickle format, same checkpoint/resume behaviour -- and still
spreads its replicates across the job's threads; only the outer loop is serial.

Required snakemake.params
    cells        : list of dicts, in run order, one per cell, each with
                       path   : absolute path of that cell's output pickle
                       sdist  : effect-size distribution
                       N2U    : population-scaled mutation influx
                       sigma2 : background genetic variance
                       shift  : size of the optimum shift
    N            : Wright-Fisher population size
    n_replicates : replicates per cell (the same count for every cell in the grid)

Optional snakemake.params
    stop_time / record_moments / seed / checkpoint_every -- as in simulation_run.py

Output
    snakemake.output : the cells' pickles, in the same order as `cells`. Each is written by
    simulation_run.run_replicates(), which checkpoints it as it goes and, on a rerun, reuses
    the replicates already in it. A cell finished before a timeout therefore costs only the
    re-pickling of its existing replicates when the batch is rerun, and a cell interrupted
    mid-way resumes from its last checkpoint.
"""

import os
import sys
import time

# make this script's directory importable when run as a Snakemake script (mirrors
# simulation_run.py, which does the same for simulation_classes)
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass

from simulation_run import run_replicates


def _optional(params, name, default):
    """Read an optional snakemake param, falling back to `default` if absent."""
    try:
        return params[name]
    except (KeyError, IndexError, TypeError, AttributeError):
        return default


def main(snakemake):
    p = snakemake.params
    cells = p["cells"]
    n_replicates = p["n_replicates"]
    threads = getattr(snakemake, "threads", 1)

    t_batch = time.time()
    for i, cell in enumerate(cells, 1):
        t_cell = time.time()
        # run_replicates writes and resumes cell['path'] itself, so an interrupted batch
        # keeps every finished cell and every finished replicate of the cell it died in.
        os.makedirs(os.path.dirname(cell["path"]), exist_ok=True)
        run_replicates(
            N=p["N"],
            sdist=cell["sdist"],
            N2U=cell["N2U"],
            sigma2=cell["sigma2"],
            shift=cell["shift"],
            n_replicates=n_replicates,
            stop_time=_optional(p, "stop_time", 1e4),
            record_moments=_optional(p, "record_moments", False),
            seed=_optional(p, "seed", None),
            threads=threads,
            output_path=cell["path"],
            checkpoint_every=_optional(p, "checkpoint_every", 50),
        )
        # progress goes to the job's log: batches are long, and the per-cell timings are
        # what the batch cost model in phase_space_grids.py is calibrated against
        print(f"[{i}/{len(cells)}] 2NU={cell['N2U']:g} sigma2={cell['sigma2']:g} "
              f"shift={cell['shift']:g}  {time.time() - t_cell:.1f} s  "
              f"({time.time() - t_batch:.1f} s into the batch)", flush=True)


# Under Snakemake's `script:` directive the `snakemake` object is injected into globals;
# run automatically in that case (but stay importable for testing).
if "snakemake" in globals():
    main(snakemake)  # noqa: F821
