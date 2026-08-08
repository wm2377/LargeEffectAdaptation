"""
Standalone script: Figure 2, the dynamics of adaptation after an optimum shift.

Panels
    A: trajectory of the mean distance from the optimum, split into a rapid phase
       and an equilibrium phase (with an inset showing the phenotype distribution
       sweeping toward the new optimum during the rapid phase).
    C: cartoon allele-frequency trajectories of small / intermediate-effect
       alleles (some fix, some are lost).
    D: cartoon trajectories of large-effect alleles (none fix).

(The mosaic labels the three stacked axes A/C/D to match their panel titles; the
"B." title belongs to the inset inside panel A, so there is no B mosaic cell.)

Run either as a Snakemake script (saves to snakemake.output[0]) or directly:
    python Figure_2.py [output.png]
Direct runs default to results/plots/Figure_2.png next to this script.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import root
from scipy.integrate import quad

from common_functions import fixation_probability_steady_state as fixation_prob_laura, make_figure_set_width
from analytic_functions import folded_sojourn_time

N = 5000


# --------------------------------------------------------------------------- #
# Quantile of the folded sojourn-time distribution of allele frequency.
# --------------------------------------------------------------------------- #
def ppf_x_helper(S, y, N=5000):
    if y < 0:
        return y
    elif y > 1 / 2:
        return 2 * y
    else:
        top = quad(lambda x: folded_sojourn_time(S=S, x=x, N=N), 0, y, points=[1 / (2 * N)])[0]
        bottom = quad(lambda x: folded_sojourn_time(S=S, x=x, N=N), 0, 1 / 2, points=[1 / (2 * N)])[0]
        return top / bottom


def ppf_x(S, p, initial_value=0.1):
    return root(lambda y: ppf_x_helper(S=S, y=y) - p, initial_value).x[0]


# --------------------------------------------------------------------------- #
# Figure 2 panels
# --------------------------------------------------------------------------- #
def color_sign(sign):
    if sign == 1:
        color = (0, 0.6, 0.2)
    else:
        color = (0, 0.2, 0.6)
    return color


### plot cartoon allelic trajectories
def figure_2_B_and_C(ax, sigma2, shift, S, sign, n_fix, n_extinct):

    color = color_sign(sign)

    lw = 0.7

    ax.plot([0, 5e4], [0.5, 0.5], ls='--', color='darkgray', lw=lw)

    avg_x = ppf_x(S=S, p=0.97)
    a = np.sqrt(S) * sign
    x_traj = [avg_x]
    D = shift
    t_rapid = int(np.log(shift) * 2 * N / sigma2)
    while len(x_traj) < t_rapid:
        x = x_traj[-1]
        dx = a / (2 * N) * x * (1 - x) * (D - a * (1 / 2 - x))

        x_traj.append(x + dx)
        D += -sigma2 * D / (2 * N) - 2 * a * dx

    x_rapid = x_traj[:t_rapid]
    ax.plot(x_traj[:t_rapid], color=color, lw=1)

    x_star = x_rapid[-1]
    t_values = np.logspace(np.log10(t_rapid), np.log10(5e4), 100) - t_rapid

    p_fix = fixation_prob_laura(a=a, x=x_star)
    n_fix = int(10 * p_fix)
    n_extinct = 10 - n_fix
    if n_fix == 0:
        flux = 0.9
        t_die = 5e3
    else:
        flux = 0.7
        t_die = 6e3
    for i in range(n_fix):
        realized_flux = i / n_fix * flux * 2 + 1 - flux
        x_equilibrium = x_star / (x_star + np.exp(-(t_values) / (t_die * realized_flux)) * (1 - x_star))
        ax.plot(t_values + t_rapid, x_equilibrium, color=color, lw=lw)
    for i in range(n_extinct):
        realized_flux = i / n_extinct * flux * 2 + 1 - flux
        x_equilibrium = x_star * np.exp(-(t_values) / (t_die * realized_flux))
        ax.plot(t_values + t_rapid, x_equilibrium, color=color, lw=lw)

    ax.set_xscale('log')
    ax.set_ylabel('Frequency', size=11)
    ax.set_xlabel('Generations after shift', size=11, labelpad=0.1)
    ax.set_ylim([0, 1])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlim([1, 5e4])

    if sign == 1:
        ax.fill_between([0, t_rapid], [0, 0], [1, 1], color='k', alpha=0.1, edgecolor=[1, 1, 1, 0])

    if n_fix > 0:
        ax.set_title(r'$\bf{C.}$ Trajectories of small- and intermediate-effect alleles', size=11, loc='left')
    else:
        ax.set_title(r'$\bf{D.}$ Trajectories of large-effect alleles', size=11, loc='left')


### plot the trajectory of the mean distance from the optimum
def figure_2_A(ax, sigma2, shift):
    times = np.logspace(0, np.log10(5e4), 1000)
    distance = shift * np.exp(-(times - 1) * sigma2 / (2 * N))
    ax.plot(times, distance, color='k')
    ax.set_xscale('log')
    ax.set_xlabel('Generations after shift', size=11, labelpad=0.1)
    ax.set_ylabel('Distance', size=11)
    ax.set_xlim([1, 5e4])
    ax.set_ylim([0, shift])
    ax.set_ylim
    ax.set_title(r'$\bf{A.}$ Trajectory of mean distance from optimum', size=11, loc='left')
    t_rapid = int(np.log(shift) * 2 * N / sigma2)
    ax.fill_between([0, t_rapid], [0, 0], [shift, shift], color='k', alpha=0.1, edgecolor=[1, 1, 1, 0])
    ax.text(x=t_rapid, y=shift * 0.875, s='Rapid phase  ', horizontalalignment='right', color='k', size=11)
    ax.text(x=t_rapid, y=shift * 0.875, s='  Equilibrium phase', horizontalalignment='left', color=[0, 0, 0, 0.5], size=11)
    ax.set_yticks([0, shift / 2, shift])
    ax.set_yticklabels(['0', r'$\Lambda / 2$', r'$\Lambda$'])


### functions for plotting the inset panel
def plot_normal_distribution(VA, mean, color, ax):
    x = np.linspace(-100, 100, 1000)
    y = stats.norm.pdf(x + mean, loc=mean, scale=np.sqrt(VA))
    ax.plot(x + mean, y, color=color)


def make_highly_polygenic_phenotypic_evolution(ax, VA=400, shift=80, fontsize=10):
    cmap = matplotlib.colormaps['Greys']
    std = np.sqrt(VA)
    for mean in np.linspace(0, shift, 5):
        plot_normal_distribution(VA=VA, mean=mean, color=cmap(mean / shift * 0.8 + 0.2), ax=ax)
        if mean == 0 or mean == shift:
            max_y = stats.norm.pdf(mean, loc=mean, scale=std)
            ax.plot([mean, mean], [0, max_y], color=cmap(mean / shift * 0.8 + 0.2), ls='--')
    ax.set_xlabel('Phenotype', size=fontsize, labelpad=-10)
    ax.set_ylabel('Density', size=fontsize)
    ax.set_title(r'$\bf{B.}$ Phenotypic evolution in rapid phase', size=fontsize, loc='left', pad=0)
    ax.set_ylim([0, 0.03])
    ax.set_xlim([-std * 2.5, shift + std * 2.5])
    ax.set_xticks([0, shift])
    ax.set_xticklabels(['0', r'$\Lambda$'])
    ax.tick_params(axis='x', labelsize=8)
    arrow_height = 0.025
    ax.annotate('',
                xy=(shift, arrow_height), xycoords='data',
                xytext=(10, arrow_height), textcoords='data',
                arrowprops=dict(facecolor='black', shrink=0.0, headlength=5),
                horizontalalignment='center', verticalalignment='bottom')
    for i in range(shift - 2):
        ax.annotate('',
                xy=(i + 1, arrow_height), xycoords='data',
                xytext=(i, arrow_height), textcoords='data',
                arrowprops=dict(facecolor=cmap(i / shift * 0.8 + 0.2), width=7.5, shrink=0.02, headlength=0.01, headwidth=1, edgecolor=cmap(i / shift * 0.8 + 0.2)),
                horizontalalignment='center', verticalalignment='bottom')

    ax.text(x=shift / 2, y=arrow_height, s=r'time', horizontalalignment='center', verticalalignment='center', size=8, color='white')
    ax.set_yticks([])


def make_figure_2_1(out_path):
    # mosaic labels match the panel titles: A (mean distance), C (small/
    # intermediate-effect trajectories), D (large-effect trajectories). "B." is the
    # inset inside panel A, so there is no B mosaic cell.
    fig, axes = plt.subplot_mosaic("AAA;CCC;DDD", dpi=500, figsize=(6.5, 6.5))

    sigma2 = 400
    shift = np.sqrt(sigma2) * 2
    figure_2_B_and_C(ax=axes['C'], sigma2=sigma2, shift=shift, S=5, sign=1, n_fix=5, n_extinct=5)
    figure_2_B_and_C(ax=axes['C'], sigma2=sigma2, shift=shift, S=5, sign=-1, n_fix=3, n_extinct=5)
    figure_2_B_and_C(ax=axes['D'], sigma2=sigma2, shift=shift, S=35, sign=1, n_fix=0, n_extinct=5)
    figure_2_B_and_C(ax=axes['D'], sigma2=sigma2, shift=shift, S=35, sign=-1, n_fix=0, n_extinct=5)
    figure_2_A(ax=axes['A'], sigma2=sigma2, shift=shift)
    plt.subplots_adjust(wspace=0., hspace=0.55)

    axins = axes['A'].inset_axes(
        [0.48, 0.22, 0.47, 0.47], xticklabels=[], yticklabels=[])
    make_highly_polygenic_phenotypic_evolution(ax=axins, VA=400, shift=80)

    text_height = 0.875
    gt_sep = axes['C'].text(x=500, y=text_height, s='>', horizontalalignment='center', size=11, color='k')
    fig.canvas.draw()       # realize a renderer so the text extent below is valid
    r = fig.canvas.get_renderer()
    bb = gt_sep.get_window_extent(renderer=r)
    inv = axes['C'].transData.inverted()
    ## get the x and y in data coordinates
    x_min, y_min = inv.transform(bb.corners()[0])
    x_max, y_max = inv.transform(bb.corners()[2])

    axes['C'].text(x=x_min * 0.95, y=text_height, s='#aligned', horizontalalignment='right', size=11, color=color_sign(1))
    axes['C'].text(x=x_max / 0.95, y=text_height, s='#opposing', horizontalalignment='left', size=11, color=color_sign(-1))
    axes['C'].text(x=500, y=y_min - 0.05, s='fixations', horizontalalignment='center', verticalalignment='top', size=11, color='k')

    axes['D'].text(x=500, y=text_height, s='no fixations', horizontalalignment='center', size=11, color='k')
    plt.tight_layout(h_pad=0.5)
    make_figure_set_width(fig, out_path, target_width_inches=6.5, dpi=400)
    plt.close(fig)


def main(out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    make_figure_2_1(out_path=out_path)


if "snakemake" in globals():
    out = snakemake.output  # noqa: F821
    main(getattr(out, "figure_2", None) or out[0])
elif __name__ == "__main__":
    default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "plots", "Figure_2.png")
    main(sys.argv[1] if len(sys.argv) > 1 else default)
