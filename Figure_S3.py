"""
Standalone script: Figure S3, the fixation probability of a large-effect allele as a
function of its squared effect size S = a^2, relative to the neutral fixation
probability 1/(2N).

The three curves are computed in closed form from
common_functions.fixation_probability_steady_state, so the script needs no pickles.

Three single-copy (x_0 = 1/(2N)) scenarios are shown, each divided by the neutral
fixation probability:
    MSDB     (black) : steady-state fixation probability with no optimum shift.
    Aligned  (blue)  : an optimum shift of +`shift` first raises the allele's
                       frequency (frequency_after_shift), then it fixes from there.
    Opposing (green) : the same but with a -`shift` shift.

Run either as a Snakemake script (saves to snakemake.output[0]) or directly:
    python Figure_S3.py [output.png]
Direct runs default to results/plots/Figure_S3.png next to this script.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common_functions as cf


# illustrative parameters: population size, optimum-shift size, background variance
N = 5000
SHIFT = 40
VA = 400

FONTSIZE = 11


def fixation_prob_steady_state(a, N):
    """Steady-state (MSDB) fixation probability of a single-copy allele of effect a."""
    return cf.fixation_probability_steady_state(x=1 / (2 * N), a=a)


def frequency_after_shift(a, N, x, shift, Va):
    """Frequency of an effect-a allele right after an optimum shift, from x.

    The shift changes the frequency by shift*a*x*(1-x)/Va, clipped to [0, 1].
    """
    new_x = x + shift * a * x * (1 - x) / Va
    return min(1.0, max(0.0, new_x))


def fixation_prob_shift(a, N, shift, Va):
    """Fixation probability of a single-copy allele after an optimum shift of `shift`.

    A positive `shift` is aligned with the allele's effect; a negative one opposes it.
    """
    x_after = frequency_after_shift(a, N, 1 / (2 * N), shift, Va)
    return cf.fixation_probability_steady_state(x=x_after, a=a)


def plot_fixation_prob(fig_width, fig_height, N, shift, Va):
    """Draw the three fixation-probability curves vs S = a^2 (relative to neutral).

    The axes box is forced square so make_figure_set_width yields a square panel.
    """
    # tick labels take the rcParam, not a fontsize=
    plt.rcParams["font.size"] = FONTSIZE

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_box_aspect(1)

    S_values = np.logspace(-2, 2, 100)
    a_values = np.sqrt(S_values)

    fp_steady = [fixation_prob_steady_state(a, N) for a in a_values]
    fp_aligned = [fixation_prob_shift(a, N, shift, Va) for a in a_values]
    fp_opposing = [fixation_prob_shift(a, N, -shift, Va) for a in a_values]
    # neutral reference: an effectively zero-effect allele
    fp_neutral = fixation_prob_steady_state(a=0.0001, N=N)

    ax.plot(S_values, [fp / fp_neutral for fp in fp_steady], label="MSDB", color="black")
    ax.plot(S_values, [fp / fp_neutral for fp in fp_aligned], label="Aligned", color="blue")
    ax.plot(S_values, [fp / fp_neutral for fp in fp_opposing], label="Opposing", color="green")

    ax.set_xscale("log")
    ax.set_ylabel("Fixation probability (relative to neutral)", fontsize=FONTSIZE)
    ax.set_xlabel(r"Effect size squared ($S_e=a^2$)", fontsize=FONTSIZE)
    ax.set_xlim(0.01, 100)
    ax.set_ylim(0, 1.05)
    ax.legend(edgecolor="black", loc="upper right", fontsize=FONTSIZE, handlelength=1)

    plt.tight_layout()
    return fig


def main(out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig = plot_fixation_prob(fig_width=4, fig_height=4, N=N, shift=SHIFT, Va=VA)
    cf.make_figure_set_width(fig, out_path, target_width_inches=4, dpi=300)


if "snakemake" in globals():
    out = snakemake.output  # noqa: F821
    main(getattr(out, "figure_s3", None) or out[0])
elif __name__ == "__main__":
    default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "plots", "Figure_S3.png")
    main(sys.argv[1] if len(sys.argv) > 1 else default)
