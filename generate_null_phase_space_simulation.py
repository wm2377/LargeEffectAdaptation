"""
Snakemake script: write a NULL placeholder for a phase-space simulation cell that is out
of the model's scope and therefore is not simulated.

Panel D of Figure 6 grids the optimum shift in units of sqrt(total variance); where the
resulting ABSOLUTE shift exceeds the scope cutoff (phase_space_grids.SHIFT_SCOPE_MAX = 100,
i.e. Lambda > sqrt(V_S)) the (expensive) multiple-allele simulation is skipped and this
rule emits a null instead: zero replicates, no fixations. aggregate_phase_space_data.py
reads it like any other cell and, seeing n_replicates == 0, leaves that grid position blank
(white / no fixation).

Every placeholder for the whole grid is written by ONE job (rule
phase_space_null_simulations): each is a zero-replicate stub that takes microseconds, so a
job per file was hundreds of times more scheduling overhead than work.

Required snakemake.params : cells -- a list of dicts with path/N2U/shift/sigma2 (and N).
                            A single cell may also be given as N2U/shift/sigma2 params
                            with the file as output[0].
Output                    : one null pickle (reduced-simulation format) per cell
"""

import os
import pickle

import numpy as np


def make_null(out_path, N2U, shift, sigma2, N):
    null = {
        "parameters": {"N": N, "N2U": float(N2U), "shift": float(shift), "sigma2": float(sigma2)},
        "n_replicates": 0,
        "n_fixations": np.array([], dtype=int),
        "fixed_effect_sizes": np.array([], dtype=float),
        "fixed_initial_frequencies": np.array([], dtype=float),
        "out_of_scope": True,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump(null, fh)


def make_nulls(cells, N=None):
    """Write one placeholder per cell dict (keys path/N2U/shift/sigma2, optional N)."""
    for c in cells:
        make_null(c["path"], N2U=c["N2U"], shift=c["shift"], sigma2=c["sigma2"],
                  N=c.get("N", N))
    return len(cells)


if "snakemake" in globals():
    p = snakemake.params        # noqa: F821
    if hasattr(p, "cells"):
        n = make_nulls(p.cells, N=getattr(p, "N", None))
        print("wrote %d null placeholder(s)" % n)
    else:
        make_null(snakemake.output[0],  # noqa: F821
                  N2U=p.N2U, shift=p.shift, sigma2=p.sigma2, N=getattr(p, "N", None))
