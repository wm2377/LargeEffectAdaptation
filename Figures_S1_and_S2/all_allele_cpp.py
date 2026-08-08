"""ctypes wrapper around cpp/all_allele_core.cpp -- the fast path for the all-allele model.

Same model and same outputs as all_allele_model.py, but the generation loop runs in C++.
The library is built on demand with a direct g++ call; set CXX to override the compiler.

"""

import ctypes
import os
import subprocess
import sys
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "cpp", "all_allele_core.cpp")
LIBRARY = os.path.join(HERE, "libaacore.so")

# S = a^2 bins; an allele joins the FIRST bin whose limit it falls under. Must match
# process_all_allele_replicate.BIN_LIMITS.
BIN_LIMITS = (100.0, 20000.0)
# |a| below which an allele counts as small-effect for the skew trajectory, matching
# calculate_skew_all_alleles.py.
SKEW_MAX_A = 10.0

_build_lock = threading.Lock()
_lib = None


class BuildError(RuntimeError):
    """Raised when the shared library could not be built."""


def _build():
    cxx = os.environ.get("CXX", "g++")
    cmd = [cxx, "-O2", "-std=c++17", "-fPIC", "-ffp-contract=off", "-shared",
           "-o", LIBRARY, SOURCE]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise BuildError(f"{' '.join(cmd)}\n{proc.stderr}")


def library():
    """The loaded shared library, building it if it is missing or out of date."""
    global _lib
    if _lib is not None:
        return _lib
    with _build_lock:
        if _lib is None:
            stale = (not os.path.exists(LIBRARY)
                     or os.path.getmtime(LIBRARY) < os.path.getmtime(SOURCE))
            if stale:
                _build()
            lib = ctypes.CDLL(LIBRARY)
            _declare(lib)
            _lib = lib
    return _lib


def _declare(lib):
    c_d, c_i, c_p = ctypes.c_double, ctypes.c_int, ctypes.c_void_p
    c_ll, c_ull = ctypes.c_longlong, ctypes.c_ulonglong
    dp, ip = ctypes.POINTER(c_d), ctypes.POINTER(c_i)

    lib.aa_last_error.restype = ctypes.c_char_p
    lib.aa_create.restype = c_p
    lib.aa_create.argtypes = [c_i, c_d, c_d, c_d, c_i, c_d, c_d, c_i, dp, c_d, c_ull]
    lib.aa_destroy.argtypes = [c_p]
    lib.aa_initialize.argtypes = [c_p, c_i]
    lib.aa_poisson.restype = c_i
    lib.aa_poisson.argtypes = [c_p, c_d]
    lib.aa_burn.argtypes = [c_p, c_ll]
    lib.aa_shift_optimum.argtypes = [c_p, c_d]
    lib.aa_force_minor_alleles.argtypes = [c_p]
    lib.aa_run_recorded.restype = c_ll
    lib.aa_run_recorded.argtypes = [c_p, c_ll, dp, dp, ip, dp]
    lib.aa_n_fixations.restype = c_i
    lib.aa_n_fixations.argtypes = [c_p]
    lib.aa_get_fixations.argtypes = [c_p, dp]
    lib.aa_n_segregating.restype = c_i
    lib.aa_n_segregating.argtypes = [c_p]
    lib.aa_get_state.argtypes = [c_p, dp, dp]
    lib.aa_fixed_background.restype = c_d
    lib.aa_fixed_background.argtypes = [c_p]
    lib.aa_mean_phenotype.restype = c_d
    lib.aa_mean_phenotype.argtypes = [c_p]
    lib.aa_sojourn_time.restype = c_d
    lib.aa_sojourn_time.argtypes = [c_i, c_d, c_d]


def _ptr(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


class CppAllAlleleSimulation:
    """The C++ core, with the same phases as all_allele_model.AllAlleleSimulation."""

    def __init__(self, N, mutation_rate, shift, weight, burn_time, optimum=0.0,
                 distribution_type="expon", seed=0, bin_limits=BIN_LIMITS,
                 skew_max_a=SKEW_MAX_A):
        self.lib = library()
        self.N = int(N)
        self.shift = float(shift)
        self.burn_time = int(burn_time)
        self.bin_limits = np.asarray(bin_limits, dtype=np.float64)

        if distribution_type == "expon":
            large_kind, large_loc, large_scale = 0, 100.0, 400.0
        elif distribution_type == "uniform":
            large_kind, large_loc, large_scale = 1, 100.0, 900.0
        else:
            raise ValueError(f"Unsupported distribution type: {distribution_type!r}")

        self.handle = self.lib.aa_create(
            self.N, float(mutation_rate), float(optimum), float(weight),
            large_kind, large_loc, large_scale,
            self.bin_limits.size, _ptr(self.bin_limits), float(skew_max_a),
            ctypes.c_ulonglong(int(seed) & (2 ** 64 - 1)))
        if not self.handle:
            raise RuntimeError(self.lib.aa_last_error().decode())

    def __del__(self):
        handle = getattr(self, "handle", None)
        if handle:
            self.lib.aa_destroy(handle)
            self.handle = None

    # -- setup ---------------------------------------------------------------------
    def initialize_population(self, expected_segregating):
        """Seed Poisson(expected_segregating) alleles from the sojourn-time density."""
        n = self.lib.aa_poisson(self.handle, float(expected_segregating))
        self.lib.aa_initialize(self.handle, int(n))
        return n

    # -- phases --------------------------------------------------------------------
    def burn(self, generations=None):
        self.lib.aa_burn(self.handle, int(self.burn_time if generations is None else generations))

    def shift_optimum(self, shift=None):
        self.lib.aa_shift_optimum(self.handle, float(self.shift if shift is None else shift))

    def force_minor_alleles(self):
        self.lib.aa_force_minor_alleles(self.handle)

    def run_recorded(self, generations):
        """Advance `generations`, returning per-generation summaries as arrays.

        Keys: 'moments' (generations, n_bins, 3) laid out [mean, variance, skew];
        'skew_small'; 'n_segregating'; 'mean_phenotype'.
        """
        generations = int(generations)
        n_bins = self.bin_limits.size
        moments = np.zeros((generations, n_bins, 3), dtype=np.float64)
        skew_small = np.zeros(generations, dtype=np.float64)
        n_seg = np.zeros(generations, dtype=np.int32)
        mean_phenotype = np.zeros(generations, dtype=np.float64)
        self.lib.aa_run_recorded(
            self.handle, generations, _ptr(moments), _ptr(skew_small),
            n_seg.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), _ptr(mean_phenotype))
        return dict(moments=moments, skew_small=skew_small,
                    n_segregating=n_seg, mean_phenotype=mean_phenotype)

    # -- state ---------------------------------------------------------------------
    @property
    def fixations(self):
        n = self.lib.aa_n_fixations(self.handle)
        out = np.zeros(max(n, 1), dtype=np.float64)
        if n:
            self.lib.aa_get_fixations(self.handle, _ptr(out))
        return out[:n]

    def state(self):
        """(a, x) for every segregating allele."""
        n = self.lib.aa_n_segregating(self.handle)
        a = np.zeros(max(n, 1), dtype=np.float64)
        x = np.zeros(max(n, 1), dtype=np.float64)
        if n:
            self.lib.aa_get_state(self.handle, _ptr(a), _ptr(x))
        return a[:n], x[:n]

    @property
    def fixed_background(self):
        return self.lib.aa_fixed_background(self.handle)

    def mean_phenotype(self):
        return self.lib.aa_mean_phenotype(self.handle)


def moments_to_results(moments, bin_limits=BIN_LIMITS, first_generation=1):
    """Convert the C++ moment array into the {generation: {bin: {metric: value}}} dict
    that process_all_allele_replicate.py produces, so downstream code is unchanged."""
    names = ("mean", "variance", "skew")
    out = {}
    for t in range(moments.shape[0]):
        out[first_generation + t] = {
            float(b) if not float(b).is_integer() else int(b): {
                name: float(moments[t, i, j]) for j, name in enumerate(names)}
            for i, b in enumerate(bin_limits)
        }
    return out


if __name__ == "__main__":
    library()
    print(f"built {LIBRARY}")
