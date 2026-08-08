"""ctypes wrapper around cpp/individual_core.cpp -- the individual-based simulation.

A different MODEL from the all-allele engine in all_allele_cpp, not a different
implementation of it: N explicit diploid genotypes, fitness-proportional reproduction, and
free recombination, so drift and selection emerge from sampling individuals rather than
from a diffusion approximation. Figure S1 compares the two.
"""

import ctypes
import os
import subprocess
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "cpp", "individual_core.cpp")
LIBRARY = os.path.join(HERE, "libindcore.so")

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
    """The loaded shared library, building it if missing or older than the source."""
    global _lib
    if _lib is not None:
        return _lib
    with _build_lock:
        if _lib is None:
            if (not os.path.exists(LIBRARY)
                    or os.path.getmtime(LIBRARY) < os.path.getmtime(SOURCE)):
                _build()
            lib = ctypes.CDLL(LIBRARY)
            _declare(lib)
            _lib = lib
    return _lib


def _declare(lib):
    c_d, c_i, c_p = ctypes.c_double, ctypes.c_int, ctypes.c_void_p
    c_ll, c_ull = ctypes.c_longlong, ctypes.c_ulonglong
    dp, ip, llp = ctypes.POINTER(c_d), ctypes.POINTER(c_i), ctypes.POINTER(c_ll)

    lib.ind_last_error.restype = ctypes.c_char_p
    lib.ind_create.restype = c_p
    lib.ind_create.argtypes = [c_i, c_d, c_d, c_d, c_i, c_d, c_d, c_ull]
    lib.ind_destroy.argtypes = [c_p]
    lib.ind_poisson.restype = c_i
    lib.ind_poisson.argtypes = [c_p, c_d]
    lib.ind_initialize.argtypes = [c_p, c_i]
    lib.ind_burn.argtypes = [c_p, c_ll]
    lib.ind_shift_optimum.argtypes = [c_p, c_d]
    lib.ind_run_recorded.restype = c_ll
    lib.ind_run_recorded.argtypes = [c_p, c_ll, dp, dp, dp, ip, dp]
    lib.ind_n_fixations.restype = c_i
    lib.ind_n_fixations.argtypes = [c_p]
    lib.ind_get_fixations.argtypes = [c_p, dp, llp]
    lib.ind_n_segregating.restype = c_i
    lib.ind_n_segregating.argtypes = [c_p]
    lib.ind_fixed_background.restype = c_d
    lib.ind_fixed_background.argtypes = [c_p]
    lib.ind_moments.argtypes = [c_p, dp, dp, dp]


def _ptr(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


class IndividualSimulation:
    """Burn in at a stationary optimum, shift it, then record the phenotypic response."""

    def __init__(self, N, mutation_rate, shift, weight, optimum=0.0,
                 distribution_type="expon", seed=0):
        self.lib = library()
        self.N = int(N)
        self.shift = float(shift)

        if distribution_type == "expon":
            kind, loc, scale = 0, 100.0, 400.0
        elif distribution_type == "uniform":
            kind, loc, scale = 1, 100.0, 900.0
        else:
            raise ValueError(f"Unsupported distribution type: {distribution_type!r}")

        self.handle = self.lib.ind_create(
            self.N, float(mutation_rate), float(optimum), float(weight),
            kind, loc, scale, ctypes.c_ulonglong(int(seed) & (2 ** 64 - 1)))
        if not self.handle:
            raise RuntimeError(self.lib.ind_last_error().decode())

    def __del__(self):
        handle = getattr(self, "handle", None)
        if handle:
            self.lib.ind_destroy(handle)
            self.handle = None

    def initialize_population(self, expected_segregating):
        """Seed Poisson(expected_segregating) mutations from the sojourn-time density."""
        n = self.lib.ind_poisson(self.handle, float(expected_segregating))
        self.lib.ind_initialize(self.handle, int(n))
        return n

    def burn(self, generations):
        self.lib.ind_burn(self.handle, int(generations))

    def shift_optimum(self, shift=None):
        self.lib.ind_shift_optimum(self.handle, float(self.shift if shift is None else shift))

    def run_recorded(self, generations):
        """Advance `generations`, returning per-generation phenotype summaries."""
        g = int(generations)
        mean = np.zeros(g); var = np.zeros(g); skew = np.zeros(g)
        n_seg = np.zeros(g, dtype=np.int32); bg = np.zeros(g)
        self.lib.ind_run_recorded(
            self.handle, g, _ptr(mean), _ptr(var), _ptr(skew),
            n_seg.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), _ptr(bg))
        return dict(mean=mean, variance=var, skew=skew,
                    n_segregating=n_seg, fixed_background=bg)

    @property
    def fixations(self):
        """(signed effect sizes, generations) of every mutation that fixed."""
        n = self.lib.ind_n_fixations(self.handle)
        effects = np.zeros(max(n, 1))
        times = np.zeros(max(n, 1), dtype=np.int64)
        if n:
            self.lib.ind_get_fixations(
                self.handle, _ptr(effects),
                times.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)))
        return effects[:n], times[:n]

    def n_large_fixations(self, min_a=10.0, after=0):
        """Figure S1's statistic: aligned large-effect fixations after the shift."""
        effects, times = self.fixations
        return int(np.sum((effects >= min_a) & (times > after)))

    @property
    def n_segregating(self):
        return self.lib.ind_n_segregating(self.handle)

    @property
    def fixed_background(self):
        return self.lib.ind_fixed_background(self.handle)

    def moments(self):
        m = ctypes.c_double(); v = ctypes.c_double(); s = ctypes.c_double()
        self.lib.ind_moments(self.handle, ctypes.byref(m), ctypes.byref(v), ctypes.byref(s))
        return m.value, v.value, s.value


if __name__ == "__main__":
    library()
    print(f"built {LIBRARY}")
