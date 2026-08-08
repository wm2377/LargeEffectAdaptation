"""
Snakemake script: run `n_replicates` of the polygenic-adaptation Simulation with
one set of parameters and pickle the per-replicate output.

Required snakemake.params
    N            : Wright-Fisher population size
    sdist        : scipy.stats distribution of scaled selection coefficients S
    N2U          : 2NU, total mutation influx for large-effect alleles (both signs)
    sigma2       : background (infinitesimal) genetic variance
    shift        : size of the optimum shift (initial distance d)
    n_replicates : number of independent replicates to run (alias: `n`)

Optional snakemake.params
    tracking_time  (default 50)
    stop_time      (default 1e4)
    record_moments (default False)   -- also store the trait second/third moments
    full_output    (default False)   -- also store the mean trajectory, each fixing
                                        allele's frequency trajectory, and allele-state
                                        snapshots (large; use with few replicates)
    seed           (default None)    -- int for reproducible runs, None for random
    checkpoint_every (default 50)    -- re-pickle after this many finished replicates

Parallelism
    Replicates run across `snakemake.threads` worker processes (1 = serial) and are
    gathered in replicate order, so results do not depend on the worker count.

Engine
    Each replicate runs in the C++ core (cpp/sim_core.cpp via simulation_cpp.py), which
    is ~200x faster than simulation_classes.Simulation and statistically equivalent, not
    bit-identical. Set SIM_ENGINE=python for the reference implementation, or
    SIM_ENGINE=cpp to make an unusable C++ core an error rather than a silent fallback.
    validate_sim_core.py compares the two.

Output
    snakemake.output[0] : pickle of the results dict produced by run_replicates().

    Written incrementally: the full dict is re-pickled atomically every
    `checkpoint_every` replicates, so an interrupted job (e.g. a SLURM timeout) keeps its
    finished replicates and a rerun completes only the rest. Per-replicate RNG streams are
    spawned for the full target, so replicate i uses the same seed either way and a
    resumed run matches an uninterrupted one exactly.

    For Snakemake to keep the partial file, flag this rule's output with update() and run
    with keep-incomplete and rerun-incomplete.
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

from simulation_classes import Simulation

# 'auto' uses the C++ core when it can be built and its effect-size distribution is
# supported, else falls back to Python. Override with SIM_ENGINE.
ENGINE = os.environ.get('SIM_ENGINE', 'auto').lower()
if ENGINE not in ('auto', 'cpp', 'python'):
    raise ValueError(f"SIM_ENGINE must be auto/cpp/python, got {ENGINE!r}")


def _resolve_engine(sdist):
    """'cpp' or 'python' for this run, honouring ENGINE and what the C++ core supports."""
    if ENGINE == 'python':
        return 'python'
    try:
        import simulation_cpp
        simulation_cpp._sdist_components(sdist)      # raises if the sdist has no C++ form
        if not simulation_cpp.available():
            raise RuntimeError('C++ core unavailable')
    except Exception as exc:
        if ENGINE == 'cpp':
            raise RuntimeError(f"SIM_ENGINE=cpp but the C++ core is unusable: {exc}") from exc
        return 'python'
    return 'cpp'


def _run_one_replicate(args):
    """Run a single replicate and return its summary.

    Defined at module level so it can be dispatched to worker processes. `args`
    packs all simulation parameters plus this replicate's child seed and the
    engine to run it with.
    """
    (N, sdist, N2U, sigma2, shift, tracking_time, stop_time, record_moments,
     full_output, child_seed, engine) = args

    if engine == 'cpp':
        import simulation_cpp
        return simulation_cpp.run_replicate(
            N=N, sdist=sdist, N2U=N2U, sigma2=sigma2, shift=shift,
            tracking_time=tracking_time, stop_time=stop_time,
            record_moments=record_moments, full_output=full_output, seed=child_seed)

    sim = Simulation(N=N, sdist=sdist, N2U=N2U, sigma2=sigma2, shift=shift,
                     tracking_time=tracking_time, stop_time=stop_time,
                     record_moments=record_moments, full_output=full_output,
                     seed=child_seed)
    sim.run_simulation()

    fixations = sim.stats['fixations']
    rep = {
        'n_fixations': len(fixations),
        'fixed_effect_sizes': np.array([m.a for m in fixations]),
        'fixed_arrival_times': np.array([m.t_initial for m in fixations]),
        'fixed_initial_frequencies': np.array([m.x0 for m in fixations]),
        # first generation the 10-gen window means satisfy the convergence
        # criterion, and D_w there; NaN if never
        'convergence_time': sim.convergence_time,
        'convergence_D_window': sim.convergence_D_window,
    }
    if record_moments:
        rep['second_moment'] = np.asarray(sim.stats['second_moment'])
        rep['third_moment'] = np.asarray(sim.stats['third_moment'])
    if full_output:
        # mean trajectory (shift - d), per-allele frequency trajectories, and the
        # allele-state snapshots
        d_traj = np.asarray(sim.stats['d_trajectory'], dtype=float)
        rep['d_trajectory'] = d_traj
        rep['mean_trajectory'] = sim.shift - d_traj
        rep['recorded_trajectories'] = sim.stats['recorded_trajectories']
        rep['snapshots'] = sim.stats['snapshots']
    return rep


def _atomic_pickle_dump(obj, path):
    """Pickle `obj` to `path` atomically: write a temp file, fsync, then os.replace.

    A crash mid-write can therefore never leave a truncated/corrupt pickle at `path`
    -- `path` always holds either the previous checkpoint or the new one, nothing in
    between. The temp file gets a unique name in the target's own directory (same
    filesystem, so the replace is atomic), so even if two writers ever targeted the
    same `path` they could not clobber each other's temp file -- os.replace would
    simply be last-writer-wins, never a torn file. (Within a job there is only one
    writer: the main process checkpoints serially while the workers only return data.)
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
    return {
        'parameters': parameters,
        'n_fixations': np.array([rep['n_fixations'] for rep in replicates]),
        'replicates': replicates,
    }


def _sdist_signature(sdist):
    """A value-based fingerprint of an effect-size distribution.

    Used to decide whether an existing partial output was produced with the same
    sdist. scipy frozen distributions and our MixtureDistribution have neither a
    stable repr nor __eq__, so identity/repr can't be compared across runs; this
    reduces them to their defining values (mixtures recurse into their components).
    """
    if sdist is None:
        return None
    components = getattr(sdist, 'components', None)
    weights = getattr(sdist, 'weights', None)
    if components is not None and weights is not None:
        return ('mixture',
                tuple(_sdist_signature(c) for c in components),
                tuple(np.asarray(weights, dtype=float).tolist()))
    dist = getattr(sdist, 'dist', None)
    name = getattr(dist, 'name', None)
    if name is not None:
        args = tuple(np.asarray(getattr(sdist, 'args', ()), dtype=object).tolist())
        kwds = tuple(sorted(getattr(sdist, 'kwds', {}).items()))
        return ('scipy', name, args, kwds)
    return ('repr', repr(sdist))


# 'total' means 2NU is the influx over both sign classes. Pickles predating this
# convention carry no such stamp, so listing it in _RESUME_MATCH_KEYS below stops them
# being resumed into a run under the current one.
MUTATIONAL_INPUT_CONVENTION = 'total'

# Parameters that must match for a partial output to be resumable. n_replicates and
# threads are excluded (neither changes a replicate's result) and sdist is checked
# separately; 'engine' is included so a pickle never mixes replicates from both.
_RESUME_MATCH_KEYS = ('N', 'N2U', 'sigma2', 'shift', 'tracking_time', 'stop_time',
                      'record_moments', 'full_output', 'seed',
                      'mutational_input_convention', 'engine')


def _resumable_replicates(path, parameters):
    """Replicates reusable from an existing output at `path`, or [] to start fresh.

    Reuse requires the stored run-defining parameters (and sdist fingerprint) to match
    the current ones; a missing, unreadable/corrupt, or mismatched file yields [] so the
    run starts over and overwrites it. At most `n_replicates` replicates are reused.
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
    if _sdist_signature(prev_params.get('sdist')) != _sdist_signature(parameters['sdist']):
        return []
    return list(replicates[:int(parameters['n_replicates'])])


def run_replicates(N, sdist, N2U, sigma2, shift, n_replicates,
                   tracking_time=50, stop_time=1e4, record_moments=False,
                   full_output=False, seed=None, threads=1,
                   output_path=None, checkpoint_every=50):
    """Run `n_replicates` independent simulations and collect their outputs.

    Each replicate gets its own RNG stream spawned from `seed`, so replicates are
    statistically independent yet the whole set is reproducible when `seed` is an
    int (and non-deterministic when `seed` is None).

    With `threads > 1`, replicates run in parallel across that many worker
    processes. Results are gathered in replicate order, so the output is identical
    regardless of the number of workers.

    If `output_path` is given, the results dict is checkpointed there (atomically)
    every `checkpoint_every` finished replicates and once at the end; on entry, any
    replicates already saved at that path by a previous interrupted run with identical
    parameters are reused, and only the remaining replicates are run.

    Replicates run through the C++ core (simulation_cpp) when it is usable, and
    through simulation_classes.Simulation otherwise; set SIM_ENGINE=python to force
    the reference implementation. Which one ran is recorded in parameters['engine'],
    and a partial output produced by one engine is never resumed by the other.

    Returns a dict with:
        'parameters'  : the parameters used (incl. sdist, seed, threads, engine)
        'n_fixations' : array of fixation counts, one per replicate (the usual
                        quantity of interest)
        'replicates'  : list of per-replicate dicts, each with
                            'n_fixations'              : int
                            'fixed_effect_sizes'       : array of signed effect sizes a
                            'fixed_arrival_times'      : array of birth generations
                            'fixed_initial_frequencies': array of initial frequencies x_0
                            'convergence_time'         : first generation the 10-gen
                                window means satisfy the convergence criterion (NaN
                                if never reached)
                            'convergence_D_window'     : the windowed distance D_window
                                at convergence_time (NaN if never reached)
                        and, when record_moments is True,
                            'second_moment' / 'third_moment' : moment time series
                        and, when full_output is True,
                            'd_trajectory'         : distance d each generation
                            'mean_trajectory'      : phenotypic mean each generation
                                                     (= shift - d_trajectory)
                            'recorded_trajectories': list, one dict per allele that
                                fixes or ever reaches frequency 0.01 (alleles lost
                                below it are discarded), with id / a / t_initial /
                                x0 / fate ('fixed'/'extinct'/'segregating') /
                                fixation_time and the full (generation, frequency)
                                'trajectory' array
                            'snapshots'            : dict keyed 'initial'/'gen20'/
                                'quasi_static'/'gen300'/'final', each holding the
                                generation 'time', the distance 'D' from the optimum,
                                and the full list of 'segregating' and 'fixed'
                                alleles at that time
    """
    n_replicates = int(n_replicates)
    threads = max(1, int(threads))
    checkpoint_every = max(1, int(checkpoint_every))

    # each checkpoint re-pickles the whole dict, so a fixed interval is O(n^2) I/O;
    # treat `checkpoint_every` as a floor and cap the number of writes at ~100
    checkpoint_every = max(checkpoint_every, n_replicates // 100)

    engine = _resolve_engine(sdist)

    parameters = {
        'N': N, 'sdist': sdist, 'N2U': N2U, 'sigma2': sigma2, 'shift': shift,
        'tracking_time': tracking_time, 'stop_time': stop_time,
        'record_moments': record_moments, 'full_output': full_output, 'seed': seed,
        'n_replicates': n_replicates, 'threads': threads,
        'mutational_input_convention': MUTATIONAL_INPUT_CONVENTION,
        'engine': engine,
    }

    # reuse replicates checkpointed by an interrupted run with identical parameters.
    # Child seeds are spawned for the full target, so replicate i has the same RNG
    # stream either way.
    replicates = _resumable_replicates(output_path, parameters)
    n_done = len(replicates)

    child_seeds = np.random.SeedSequence(seed).spawn(n_replicates)
    pending = [(N, sdist, N2U, sigma2, shift, tracking_time, stop_time, record_moments,
                full_output, cs, engine)
               for cs in child_seeds[n_done:]]

    def _checkpoint():
        if output_path:
            _atomic_pickle_dump(_assemble_results(parameters, replicates), output_path)

    if pending:
        # ex.map yields in submission order, so `replicates` stays a contiguous
        # prefix -- what a later resume continues from. The `finally` checkpoint
        # persists finished work even on an in-process error.
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

    n_replicates = _optional(p, 'n_replicates', _optional(p, 'n', None))
    if n_replicates is None:
        raise KeyError("required snakemake param 'n_replicates' (or 'n') is missing")

    # run_replicates checkpoints to (and resumes from) output[0] itself
    run_replicates(
        N=_require(p, 'N'),
        sdist=_require(p, 'sdist'),
        N2U=_require(p, 'N2U'),
        sigma2=_require(p, 'sigma2'),
        shift=_require(p, 'shift'),
        n_replicates=n_replicates,
        tracking_time=_optional(p, 'tracking_time', 50),
        stop_time=_optional(p, 'stop_time', 1e4),
        record_moments=_optional(p, 'record_moments', False),
        full_output=_optional(p, 'full_output', False),
        seed=_optional(p, 'seed', None),
        threads=getattr(snakemake, 'threads', 1),
        output_path=snakemake.output[0],
        checkpoint_every=_optional(p, 'checkpoint_every', 50),
    )


# Under Snakemake's `script:` directive the `snakemake` object is injected into
# globals; run automatically in that case (but stay importable for testing).
if "snakemake" in globals():
    main(snakemake)  # noqa: F821
