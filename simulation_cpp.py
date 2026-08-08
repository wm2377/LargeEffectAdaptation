"""ctypes bridge to the C++ simulation core (cpp/sim_core.cpp).

`run_replicate()` is a drop-in replacement for building a
simulation_classes.Simulation, running it, and packing its stats into the
per-replicate dict, including full_output trajectories and snapshots.

Profiling a 2NU = 10 replicate put ~85% of the time in the standing-variation
sampler and most of the rest in per-generation Python overhead; both are gone in the
C++ port.

The library is built on demand and rebuilt whenever a source file is newer than the
.so, so no separate build step is needed. Builds are guarded by a lock file so a
job's worker processes cannot race each other.

The RNG algorithms differ from NumPy's, so a C++ replicate is a draw from the same
distributions rather than bit-identical to the Python one. Replicate i's stream is
still determined by the base seed, so runs stay reproducible and resumable;
validate_sim_core.py checks the two engines agree statistically.

Frozen scipy.stats.expon / uniform distributions and MixtureDistribution over them
are translated to the C++ side; anything else raises UnsupportedSdist, which
simulation_run.py treats as a signal to fall back to the Python engine.
"""

import ctypes
import os
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CPP_DIR = os.path.join(_HERE, "cpp")
_LIB_PATH = os.path.join(_HERE, "libsimcore.so")
_SOURCES = ("sim_core.cpp", "rng.hpp", "sdist.hpp")

# snapshot slots, in the order the C++ core stores them
_SNAPSHOT_KEYS = ("initial", "gen20", "quasi_static", "gen300", "final")
_FATES = ("fixed", "extinct", "segregating")


class UnsupportedSdist(Exception):
    """Raised when an effect-size distribution has no C++ equivalent."""


class BuildError(Exception):
    """Raised when the shared library could not be built."""


# --------------------------------------------------------------------------- #
# Building / loading the shared library
# --------------------------------------------------------------------------- #
def _needs_build():
    if not os.path.exists(_LIB_PATH):
        return True
    lib_mtime = os.path.getmtime(_LIB_PATH)
    return any(os.path.getmtime(os.path.join(_CPP_DIR, s)) > lib_mtime for s in _SOURCES)


def _build(timeout=300):
    """Compile the library, serialized across processes by a lock directory.

    A run_simulation job starts 16 workers at once and each imports this module;
    without the lock they would compile to the same output path concurrently. The
    loser of the race waits for the winner's .so to appear instead of building.
    """
    lock = _LIB_PATH + ".lock"
    try:
        os.mkdir(lock)                      # atomic: only one process wins
    except FileExistsError:
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(0.25)
            if not os.path.exists(lock) and not _needs_build():
                return
        raise BuildError(f"timed out waiting for another process to build {_LIB_PATH}")
    try:
        cmd = ["make", "-s", "-C", _CPP_DIR]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
        except FileNotFoundError:
            # no make on this node: invoke the compiler directly with the same flags
            cxx = os.environ.get("CXX", "g++")
            subprocess.run(
                [cxx, "-O2", "-std=c++17", "-fPIC", "-ffp-contract=off", "-shared",
                 "-o", _LIB_PATH, os.path.join(_CPP_DIR, "sim_core.cpp")],
                check=True, capture_output=True, timeout=timeout)
        except subprocess.CalledProcessError as exc:
            raise BuildError(f"building {_LIB_PATH} failed:\n"
                             f"{exc.stderr.decode('utf-8', 'replace')}") from exc
    finally:
        try:
            os.rmdir(lock)
        except OSError:
            pass


_lib = None


def _load():
    """Return the loaded library, building it first if needed."""
    global _lib
    if _lib is not None:
        return _lib
    if _needs_build():
        _build()
    lib = ctypes.CDLL(_LIB_PATH)

    c_d, c_i, c_p, c_u64 = ctypes.c_double, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint64
    dp, ip = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)

    lib.sim_last_error.restype = ctypes.c_char_p
    lib.sim_ctx_create.restype = c_p
    lib.sim_ctx_create.argtypes = [c_i, c_d, c_d, c_d, c_i, c_d, c_i, c_i,
                                   c_i, ip, dp, dp, dp, c_i, c_i]
    lib.sim_ctx_free.argtypes = [c_p]
    lib.sim_expected_n_segregating.restype = c_d
    lib.sim_expected_n_segregating.argtypes = [c_p]
    lib.sim_sample_standing.argtypes = [c_p, c_u64, c_i, dp, dp]
    lib.sim_test_binomial.argtypes = [c_u64, ctypes.c_int64, c_d, c_i, dp]
    lib.sim_test_poisson.argtypes = [c_u64, c_d, c_i, dp]
    lib.sim_run.restype = c_p
    lib.sim_run.argtypes = [c_p, c_u64]
    lib.sim_result_free.argtypes = [c_p]

    lib.sim_n_fixations.restype = c_i
    lib.sim_n_fixations.argtypes = [c_p]
    lib.sim_get_fixations.argtypes = [c_p, dp, dp, dp]
    lib.sim_convergence_time.restype = c_d
    lib.sim_convergence_time.argtypes = [c_p]
    lib.sim_convergence_D.restype = c_d
    lib.sim_convergence_D.argtypes = [c_p]
    lib.sim_n_d_trajectory.restype = c_i
    lib.sim_n_d_trajectory.argtypes = [c_p]
    lib.sim_get_d_trajectory.argtypes = [c_p, dp]
    lib.sim_n_moments.restype = c_i
    lib.sim_n_moments.argtypes = [c_p]
    lib.sim_get_moments.argtypes = [c_p, dp, dp]
    lib.sim_n_trajectories.restype = c_i
    lib.sim_n_trajectories.argtypes = [c_p]
    lib.sim_trajectory_meta.argtypes = [c_p, c_i, dp]
    lib.sim_trajectory_len.restype = c_i
    lib.sim_trajectory_len.argtypes = [c_p, c_i]
    lib.sim_get_trajectory.argtypes = [c_p, c_i, dp, dp]
    lib.sim_snapshot_present.restype = c_i
    lib.sim_snapshot_present.argtypes = [c_p, c_i]
    lib.sim_snapshot_scalars.argtypes = [c_p, c_i, dp]
    lib.sim_snapshot_n_segregating.restype = c_i
    lib.sim_snapshot_n_segregating.argtypes = [c_p, c_i]
    lib.sim_get_snapshot_segregating.argtypes = [c_p, c_i, dp, dp, dp, dp, dp]
    lib.sim_snapshot_n_fixed.restype = c_i
    lib.sim_snapshot_n_fixed.argtypes = [c_p, c_i]
    lib.sim_get_snapshot_fixed.argtypes = [c_p, c_i, dp, dp, dp]

    _lib = lib
    return _lib


def available():
    """True if the C++ engine can be loaded (builds it if necessary)."""
    try:
        _load()
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Translating an effect-size distribution
# --------------------------------------------------------------------------- #
def _component(frozen):
    """(kind, loc, scale) for a frozen scipy expon/uniform, else UnsupportedSdist."""
    dist = getattr(frozen, "dist", None)
    name = getattr(dist, "name", None)
    if name not in ("expon", "uniform"):
        raise UnsupportedSdist(f"effect-size distribution {name!r} has no C++ equivalent")
    args = list(getattr(frozen, "args", ()))
    kwds = dict(getattr(frozen, "kwds", {}))
    loc = float(kwds.pop("loc", args[0] if len(args) > 0 else 0.0))
    scale = float(kwds.pop("scale", args[1] if len(args) > 1 else 1.0))
    if kwds:
        raise UnsupportedSdist(f"unsupported extra shape parameters {sorted(kwds)}")
    return (0 if name == "expon" else 1), loc, scale


def _sdist_components(sdist):
    """[(kind, loc, scale, weight), ...] for a frozen distribution or a mixture."""
    components = getattr(sdist, "components", None)
    weights = getattr(sdist, "weights", None)
    if components is not None and weights is not None:
        out = []
        for comp, w in zip(components, np.asarray(weights, dtype=float)):
            kind, loc, scale = _component(comp)
            out.append((kind, loc, scale, float(w)))
        return out
    kind, loc, scale = _component(sdist)
    return [(kind, loc, scale, 1.0)]


# --------------------------------------------------------------------------- #
# Context (parameters + steady-state tables), cached per process
# --------------------------------------------------------------------------- #
class _Context:
    def __init__(self, lib, handle):
        self._lib = lib
        self.handle = handle

    def __del__(self):
        try:
            self._lib.sim_ctx_free(self.handle)
        except Exception:
            pass


_ctx_cache = {}


def _get_context(N, sdist, N2U, sigma2, shift, tracking_time, stop_time,
                 record_moments, full_output):
    """A C++ context for these parameters, reusing one already built if possible.

    The steady-state tables are the expensive part of setup and depend only on
    (sdist, N), but they live inside the context, so caching on the full
    parameter tuple keeps every replicate of a run_simulation job sharing one build.
    """
    comps = _sdist_components(sdist)
    key = (int(N), tuple(comps), float(N2U), float(sigma2), float(shift),
           int(tracking_time), float(stop_time), bool(record_moments), bool(full_output))
    ctx = _ctx_cache.get(key)
    if ctx is not None:
        return ctx

    lib = _load()
    n = len(comps)
    kinds = (ctypes.c_int * n)(*[c[0] for c in comps])
    locs = (ctypes.c_double * n)(*[c[1] for c in comps])
    scales = (ctypes.c_double * n)(*[c[2] for c in comps])
    weights = (ctypes.c_double * n)(*[c[3] for c in comps])
    handle = lib.sim_ctx_create(
        int(N), float(N2U), float(sigma2), float(shift), int(tracking_time),
        float(stop_time), 1 if record_moments else 0, 1 if full_output else 0,
        n, kinds, locs, scales, weights, 0, 0)
    if not handle:
        raise RuntimeError(f"sim_ctx_create failed: {lib.sim_last_error().decode()}")
    ctx = _Context(lib, handle)
    _ctx_cache[key] = ctx
    return ctx


# --------------------------------------------------------------------------- #
# Seeds
# --------------------------------------------------------------------------- #
def _seed_to_u64(seed):
    """A 64-bit seed for the C++ RNG from whatever the caller passes.

    simulation_run hands out np.random.SeedSequence children, one per replicate;
    those are folded down to 64 bits here so replicate i keeps its own
    reproducible stream regardless of how many workers run.
    """
    if isinstance(seed, np.random.SeedSequence):
        # fold the child's two 64-bit words together; the odd multiplier spreads
        # the second word across all 64 bits so neighbouring children cannot
        # collide by differing in one low bit
        state = seed.generate_state(2, dtype=np.uint64)
        mixed = int(state[0]) ^ ((int(state[1]) * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF)
        return mixed & 0xFFFFFFFFFFFFFFFF
    if seed is None:
        return int.from_bytes(os.urandom(8), sys.byteorder)
    return int(np.random.SeedSequence(seed).generate_state(1, dtype=np.uint64)[0])


# --------------------------------------------------------------------------- #
# Running a replicate
# --------------------------------------------------------------------------- #
def _buf(n):
    return np.empty(int(n), dtype=np.float64)


def _ptr(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def run_replicate(N, sdist, N2U, sigma2, shift, tracking_time=50, stop_time=1e4,
                  record_moments=False, full_output=False, seed=None):
    """Run one replicate in C++ and return the per-replicate result dict.

    Same keys, dtypes and shapes as the Python engine (see
    simulation_run.run_replicates' docstring for the full description).
    """
    ctx = _get_context(N, sdist, N2U, sigma2, shift, tracking_time, stop_time,
                       record_moments, full_output)
    lib = _load()
    res = lib.sim_run(ctx.handle, ctypes.c_uint64(_seed_to_u64(seed)))
    if not res:
        raise RuntimeError(f"sim_run failed: {lib.sim_last_error().decode()}")
    try:
        n_fix = lib.sim_n_fixations(res)
        a, t, x0 = _buf(n_fix), _buf(n_fix), _buf(n_fix)
        if n_fix:
            lib.sim_get_fixations(res, _ptr(a), _ptr(t), _ptr(x0))
        rep = {
            'n_fixations': int(n_fix),
            'fixed_effect_sizes': a,
            'fixed_arrival_times': t,
            'fixed_initial_frequencies': x0,
            'convergence_time': float(lib.sim_convergence_time(res)),
            'convergence_D_window': float(lib.sim_convergence_D(res)),
        }
        if record_moments:
            n_m = lib.sim_n_moments(res)
            m2, m3 = _buf(n_m), _buf(n_m)
            if n_m:
                lib.sim_get_moments(res, _ptr(m2), _ptr(m3))
            rep['second_moment'] = m2
            rep['third_moment'] = m3
        if full_output:
            n_d = lib.sim_n_d_trajectory(res)
            d_traj = _buf(n_d)
            if n_d:
                lib.sim_get_d_trajectory(res, _ptr(d_traj))
            rep['d_trajectory'] = d_traj
            rep['mean_trajectory'] = shift - d_traj
            rep['recorded_trajectories'] = _read_trajectories(lib, res)
            rep['snapshots'] = _read_snapshots(lib, res)
        return rep
    finally:
        lib.sim_result_free(res)


def _read_trajectories(lib, res):
    out = []
    meta = _buf(6)
    for i in range(lib.sim_n_trajectories(res)):
        lib.sim_trajectory_meta(res, i, _ptr(meta))
        n = lib.sim_trajectory_len(res, i)
        t, x = _buf(n), _buf(n)
        if n:
            lib.sim_get_trajectory(res, i, _ptr(t), _ptr(x))
        out.append({
            'id': int(meta[0]),
            'a': float(meta[1]),
            't_initial': float(meta[2]),
            'x0': float(meta[3]),
            'fate': _FATES[int(meta[4])],
            'fixation_time': float(meta[5]),
            # (generation, frequency) pairs, as np.array(list of tuples) in Python
            'trajectory': np.column_stack([t, x]) if n else np.empty((0, 2)),
        })
    return out


def _read_snapshots(lib, res):
    snaps = {}
    scal = _buf(2)
    for k, key in enumerate(_SNAPSHOT_KEYS):
        if not lib.sim_snapshot_present(res, k):
            continue
        lib.sim_snapshot_scalars(res, k, _ptr(scal))
        n_seg = lib.sim_snapshot_n_segregating(res, k)
        sid, sa, sx, st, sx0 = (_buf(n_seg) for _ in range(5))
        if n_seg:
            lib.sim_get_snapshot_segregating(res, k, _ptr(sid), _ptr(sa), _ptr(sx),
                                             _ptr(st), _ptr(sx0))
        n_fix = lib.sim_snapshot_n_fixed(res, k)
        fa, ft, fx0 = (_buf(n_fix) for _ in range(3))
        if n_fix:
            lib.sim_get_snapshot_fixed(res, k, _ptr(fa), _ptr(ft), _ptr(fx0))
        snaps[key] = {
            'time': int(scal[0]),
            'D': float(scal[1]),
            'segregating': {
                'id': sid.astype(np.int64),
                'a': sa, 'x': sx, 't_initial': st, 'x0': sx0,
            },
            'fixed': {'a': fa, 't_initial': ft, 'x0': fx0},
        }
    return snaps


# --------------------------------------------------------------------------- #
# Helpers used by validate_sim_core.py
# --------------------------------------------------------------------------- #
def expected_n_segregating(N, sdist, N2U):
    """Expected number of standing-variation sites (Simulation.total_n())."""
    ctx = _get_context(N, sdist, N2U, 0.0, 0.0, 50, 1e4, False, False)
    return float(_load().sim_expected_n_segregating(ctx.handle))


def sample_binomial(n_trials, p, count, seed=0):
    """`count` draws from the C++ binomial sampler (validation hook)."""
    out = _buf(count)
    _load().sim_test_binomial(ctypes.c_uint64(_seed_to_u64(seed)),
                              ctypes.c_int64(int(n_trials)), float(p), int(count), _ptr(out))
    return out


def sample_poisson(lam, count, seed=0):
    """`count` draws from the C++ Poisson sampler (validation hook)."""
    out = _buf(count)
    _load().sim_test_poisson(ctypes.c_uint64(_seed_to_u64(seed)), float(lam),
                             int(count), _ptr(out))
    return out


def sample_standing_variation(N, sdist, n, seed=0):
    """Draw `n` standing-variation alleles, returning (x, S) arrays."""
    ctx = _get_context(N, sdist, 1.0, 0.0, 0.0, 50, 1e4, False, False)
    x, s = _buf(n), _buf(n)
    _load().sim_sample_standing(ctx.handle, ctypes.c_uint64(_seed_to_u64(seed)),
                                int(n), _ptr(x), _ptr(s))
    return x, s
