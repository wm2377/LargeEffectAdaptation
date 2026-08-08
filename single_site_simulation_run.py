"""
Snakemake script: run `n_replicates` of the single-site adaptation simulation
(single_site_simulation_classes) with one set of parameters and pickle the
per-replicate output.

Required snakemake.params
    mode   : 'segregating' (allele present at the shift) or 'new' (allele arises
             at time t0 after the shift)
    N      : Wright-Fisher population size (V_S = 2N)
    S      : squared effect size of the large-effect allele (a = sign*sqrt(S))
    sign   : +1 or -1, sign of the allele's effect
    sigma2 : background (infinitesimal) genetic variance
    shift  : size of the optimum shift, i.e. the initial distance Lambda
             (the param may instead be named 'Lambda')
    n_replicates : number of independent replicates to run (alias: `n`)

Optional snakemake.params
    record_trajectories (default False) -- record every replicate's full (x, D, time)
                              trajectory. Large sweeps make big pickles, so keep it off
                              unless the trajectories are needed.
    x0       (default None) -- segregating mode only: fixed initial frequency; None
                              draws x0 from the MSDB steady state.
    t0       (default None) -- new mode only: fixed arise time; None draws
                              t0 ~ Uniform[0, T], T the Lande time to relax to 1.
    seed     (default None) -- int for reproducible runs, None for random
    checkpoint_every (default 50) -- re-pickle after this many finished replicates

Parallelism
    Replicates run across `snakemake.threads` worker processes (1 = serial) and are
    gathered in replicate order, so results do not depend on the worker count.

Output
    snakemake.output[0] : pickle of the results dict produced by run_replicates().

    Written incrementally: the full dict is re-pickled atomically every
    `checkpoint_every` replicates, so an interrupted job keeps its finished replicates
    and a rerun completes only the rest. Per-replicate RNG streams are spawned for the
    full target, so replicate i uses the same seed and record flag either way.

    For Snakemake to keep the partial file, flag this rule's output with update() and
    run with keep-incomplete and rerun-incomplete.
"""

import os
import sys
import pickle
import tempfile
from concurrent.futures import ProcessPoolExecutor

# keep workers single-threaded so the process pool does not oversubscribe cores
# (must precede the numpy import; respects user overrides)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

# make sibling modules importable under Snakemake
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass

from single_site_simulation_classes import SegregatingSimulation, NewMutationSimulation


def _run_one_replicate(args):
    """Run a single replicate and return its summary.

    Defined at module level so it can be dispatched to worker processes. `args`
    packs all simulation parameters, the record flag, and this replicate's child seed.
    """
    (mode, N, S, sign, sigma2, Lambda, x0, t0, record, child_seed) = args

    if mode == "segregating":
        sim = SegregatingSimulation(N=N, S=S, sign=sign, sigma2=sigma2,
                                    Lambda=Lambda, seed=child_seed)
        result = sim.run(x0=x0, record_trajectory=record)
    elif mode == "new":
        sim = NewMutationSimulation(N=N, S=S, sign=sign, sigma2=sigma2,
                                    Lambda=Lambda, seed=child_seed)
        result = sim.run(t0=t0, record_trajectory=record)
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'segregating' or 'new'")

    rep = {
        'fixed': bool(result.fixed),
        'n_generations': int(result.n_generations),
        'initial_frequency': float(result.initial_frequency),
        'arise_time': result.arise_time,
    }
    if record:
        rep['x_trajectory'] = result.x_trajectory
        rep['D_trajectory'] = result.D_trajectory
        rep['time_trajectory'] = result.time_trajectory
    return rep


def _atomic_pickle_dump(obj, path):
    """Pickle `obj` to `path` atomically: write a temp file, fsync, then os.replace.

    A crash mid-write can therefore never leave a truncated/corrupt pickle at `path`
    -- `path` always holds either the previous checkpoint or the new one, nothing in
    between. The temp file gets a unique name in the target's own directory (same
    filesystem, so the replace is atomic).
    """
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=os.path.basename(path) + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, 'wb') as fout:
            pickle.dump(obj, fout, protocol=pickle.HIGHEST_PROTOCOL)
            fout.flush()
            os.fsync(fout.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _assemble_results(parameters, replicates):
    """Build the results dict (same shape as the final pickle) from a replicate list."""
    fixed = np.array([rep['fixed'] for rep in replicates], dtype=bool)
    return {
        'parameters': parameters,
            'fixed': fixed,
        'fixation_probability': float(fixed.mean()) if fixed.size else float('nan'),
        'replicates': replicates,
    }


# Parameters that must match for a partial output to be resumable. n_replicates and
# threads are excluded; record_trajectories is included, since changing it changes
# what a replicate stores.
_RESUME_MATCH_KEYS = ('mode', 'N', 'S', 'sign', 'sigma2', 'Lambda',
                      'x0', 't0', 'record_trajectories', 'seed')


def _resumable_replicates(path, parameters):
    """Replicates reusable from an existing output at `path`, or [] to start fresh.

    Reuse requires the stored run-defining parameters to match the current ones; a
    missing, unreadable/corrupt, or mismatched file yields [] so the run starts over
    and overwrites it. At most `n_replicates` replicates are reused.
    """
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, 'rb') as fin:
            prev = pickle.load(fin)
        prev_params = prev['parameters']
        replicates = prev['replicates']
    except Exception:
        return []  # corrupt / unreadable / old format -> recompute from scratch
    if any(prev_params.get(k) != parameters[k] for k in _RESUME_MATCH_KEYS):
        return []
    return list(replicates[:int(parameters['n_replicates'])])


def run_replicates(mode, N, S, sign, sigma2, Lambda, n_replicates,
                   record_trajectories=False, x0=None, t0=None, seed=None, threads=1,
                   output_path=None, checkpoint_every=50):
    """Run `n_replicates` independent single-site simulations and collect their output.

    Each replicate gets its own RNG stream spawned from `seed`, so the set is
    reproducible when `seed` is an int. With `threads > 1` replicates run in parallel
    and are gathered in order, so the output does not depend on the worker count.

    If `output_path` is given, results are checkpointed there every
    `checkpoint_every` replicates, and any replicates already saved at that path by an
    interrupted run with identical parameters are reused.

    Returns a dict with:
        'parameters'           : the parameters used (incl. seed, threads)
        'fixed'                : bool array, one per replicate (True = fixed)
        'fixation_probability' : fraction of replicates in which the allele fixed
        'replicates'           : list of per-replicate dicts, each with
                                     'fixed'             : bool
                                     'n_generations'     : generations segregated
                                     'initial_frequency' : x0 (or 1/(2N) for new)
                                     'arise_time'         : t0 for new, None for seg
                                 and, when record_trajectories is True,
                                     'x_trajectory' / 'D_trajectory' / 'time_trajectory'
    """
    n_replicates = int(n_replicates)
    record_trajectories = bool(record_trajectories)
    threads = max(1, int(threads))
    checkpoint_every = max(1, int(checkpoint_every))

    parameters = {
        'mode': mode, 'N': N, 'S': S, 'sign': sign, 'sigma2': sigma2,
        'Lambda': Lambda, 'x0': x0, 't0': t0,
        'record_trajectories': record_trajectories, 'seed': seed,
        'n_replicates': n_replicates, 'threads': threads,
    }

    # reuse replicates checkpointed by an interrupted run with identical parameters;
    # child seeds are spawned for the full target, so replicate i is unchanged.
    replicates = _resumable_replicates(output_path, parameters)
    n_done = len(replicates)

    child_seeds = np.random.SeedSequence(seed).spawn(n_replicates)
    pending = [
        (mode, N, S, sign, sigma2, Lambda, x0, t0, record_trajectories, child_seeds[i])
        for i in range(n_done, n_replicates)
    ]

    def _checkpoint():
        if output_path:
            _atomic_pickle_dump(_assemble_results(parameters, replicates), output_path)

    if pending:
        # ex.map yields in submission order, so `replicates` stays a contiguous
        # prefix -- what a later resume continues from
        try:
            if threads > 1:
                with ProcessPoolExecutor(max_workers=threads) as ex:
                    for i, rep in enumerate(ex.map(_run_one_replicate, pending), 1):
                        replicates.append(rep)
                        if i % checkpoint_every == 0:
                            _checkpoint()
            else:
                for i, a in enumerate(pending, 1):
                    replicates.append(_run_one_replicate(a))
                    if i % checkpoint_every == 0:
                        _checkpoint()
        finally:
            _checkpoint()
    else:
        # already complete; still (re)write so output_path exists
        _checkpoint()

    return _assemble_results(parameters, replicates)


def _require(params, name):
    """Read a required snakemake param, with a clear error if it is missing."""
    try:
        return params[name]
    except (KeyError, IndexError, TypeError, AttributeError):
        raise KeyError(f"required snakemake param '{name}' is missing")


def _optional(params, name, default):
    """Read an optional snakemake param, falling back to `default` if absent."""
    try:
        return params[name]
    except (KeyError, IndexError, TypeError, AttributeError):
        return default


def main(snakemake):
    p = snakemake.params

    # number of replicates: accept either 'n_replicates' or 'n'
    n_replicates = _optional(p, 'n_replicates', _optional(p, 'n', None))
    if n_replicates is None:
        raise KeyError("required snakemake param 'n_replicates' (or 'n') is missing")

    # optimum shift: accept 'shift' (project convention) or 'Lambda'
    Lambda = _optional(p, 'Lambda', None)
    if Lambda is None:
        Lambda = _require(p, 'shift')

    # run_replicates checkpoints to (and resumes from) output[0] itself.
    run_replicates(
        mode=_require(p, 'mode'),
        N=_require(p, 'N'),
        S=_require(p, 'S'),
        sign=_require(p, 'sign'),
        sigma2=_require(p, 'sigma2'),
        Lambda=Lambda,
        n_replicates=n_replicates,
        record_trajectories=_optional(p, 'record_trajectories', False),
        x0=_optional(p, 'x0', None),
        t0=_optional(p, 't0', None),
        seed=_optional(p, 'seed', None),
        threads=getattr(snakemake, 'threads', 1),
        output_path=snakemake.output[0],
        checkpoint_every=_optional(p, 'checkpoint_every', 50),
    )


# Under Snakemake's `script:` directive the `snakemake` object is injected into
# globals; run automatically in that case (but stay importable for testing).
if "snakemake" in globals():
    main(snakemake)  # noqa: F821
