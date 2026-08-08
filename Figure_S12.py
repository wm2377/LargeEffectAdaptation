'''
Creates Figure S12, which shows the probability of fixation for segregating mutations of different effect sizes ranging from a^2=1e-3 to a^2=1e3
Each panel shows the probability of fixation for a different combination of background variance (sigma^2) and shift size (S). 
The simulated fixation probabilities are shown as points with 2SEs, and the analytic predictions are shown as lines. 
The simulated fixation probabilities are computed from single-site simulations, where the initial frequency of the allele is drawn from the stationary distribution. 
There are two analytic predictions:
1) The double-recursion method, which is more accurate for large effect alleles where the probability of fixation is determined by the probability that the allele establishes (stochastic) and then can reach frequency 1/2 (deterministic).
2) The Hayward method, which assumes an instaneous and deterministic jump in frequency due to the shift followed by a stochastic phase where the allele may drift to fixation or loss. This method is more accurate for small effect alleles.
'''



import os
import sys
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

import common_functions as cf
import analytic_functions as af

PANEL_LETTER = {
    (80, 40): "A",
    (80, 80): "B",
    (50, 80): "C"}


FONTSIZE = 11

def _load_single_site_simulations(input_files: list):
    '''
    Given a set of input_files where the simulation results are stored
    returns {(shift, sigma2): (a^2, p_fix, p_fix_2se)} for each single-site simulation pickle in `pickles`.
    '''
    results = {}
    if input_files:
        for path in input_files:
            with open(path, "rb") as fh:
                r = pickle.load(fh)
            key = (r["parameters"]["Lambda"], r["parameters"]["sigma2"])
            simulated_results = r['fixed']
            mean_fixation_probability = sum(simulated_results) / len(simulated_results)
            se_fixation_probability = np.std(simulated_results) / np.sqrt(len(simulated_results))
            if key not in results:
                results[key] = [],[],[]
            results[key][0].append(r["parameters"]["S"])
            results[key][1].append(mean_fixation_probability)
            results[key][2].append(2*se_fixation_probability)  # 2 standard errors
        return results
    return {}

def _load_hayward(input_files):
    '''
    Given a set of input_files where the analytic results are stored
    returns {(shift, sigma2): (a^2, p_fix)} for each Hayward analytic pickle in `pickles`.
    '''
    
    results = {}
    if input_files:
        for path in input_files:
            with open(path, "rb") as fh:
                r = pickle.load(fh)
            key = (r["parameters"]["shift"], r["parameters"]["Va"])
            if key not in results:
                results[key] = [],[]
            results[key][0].append(r["parameters"]["a2"])
            results[key][1].append(r["fixation_probability"]["fixation"])
        return results
    return {}


def _load_double_recursion(input_files):
    '''
    Given a set of input_files where the analytic results are stored
    returns {(shift, sigma2): (a^2, p_fix)} for each double-recursion analytic pickle in `pickles`.
    '''
    
    
    results = {}
    if input_files:
        for path in input_files:
            with open(path, "rb") as fh:
                r = pickle.load(fh)
            key = (r["parameters"]["shift"], r["parameters"]["sigma2"])
            if key not in results:
                results[key] = [],[]
            results[key][0].append(r["parameters"]["S"])
            results[key][1].append(r["fixation_probability"]["segregating"])
        return results
    return {} 


def _draw_effect_size_ranges(ax):
    
    large_effect_range = (1e2, 1e3)
    high_midrange = (30, 1e2)
    low_midrange = (10, 30)
    
    # shade the effect size ranges in increasingly darker shades of gray
    ax.axvspan(large_effect_range[0], large_effect_range[1], color="0.4", alpha=1, zorder=0)
    ax.axvspan(high_midrange[0], high_midrange[1], color="0.6", alpha=1, zorder=0)
    ax.axvspan(low_midrange[0], low_midrange[1], color="0.8", alpha=1, zorder=0)

def plot_panel(ax, shift, sigma2, sim_results, hayward_results, double_recursion_results,ax_index):
    '''
    Given a matplotlib axis `ax`, a shift and sigma2 value, and the results from the simulations and analytic predictions,
    plots the probability of fixation for segregating mutations of different effect sizes.
    '''
    sim_a2, sim_pfix, sim_pfix_2se = sim_results[(shift, sigma2)]
    hayward_a2, hayward_pfix = hayward_results[(shift, sigma2)]
    double_recursion_a2, double_recursion_pfix = double_recursion_results[(shift, sigma2)]

    color = cf.color_by_both(shift=shift, sigma2=sigma2)
    ax.errorbar(sim_a2, sim_pfix, yerr=sim_pfix_2se, marker=".", ls="", markersize=5, color=color)
    _draw_effect_size_ranges(ax)

    # sorted by a^2 so the line is drawn in order
    sorted_indices = np.argsort(hayward_a2)
    hayward_a2 = np.array(hayward_a2)[sorted_indices]
    hayward_pfix = np.array(hayward_pfix)[sorted_indices]
    ax.plot(hayward_a2, hayward_pfix, label='Hayward', color=color, ls='--')

    sorted_indices = np.argsort(double_recursion_a2)
    double_recursion_a2 = np.array(double_recursion_a2)[sorted_indices]
    double_recursion_pfix = np.array(double_recursion_pfix)[sorted_indices]
    ax.plot(double_recursion_a2, double_recursion_pfix, label='Double Recursion', color=color, ls='-')

    ax.set_xscale('log')
    ax.set_xlabel('Effect size squared\n'+r'($S=a^2$)')
    if ax_index == 0:
        ax.set_ylabel('Fixation probability')
    else:
        ax.set_ylabel('')
    ax.set_title(rf"$\bf{{{PANEL_LETTER[(shift, sigma2)]}.}}$"+ rf" $\Lambda$={shift}, $\sigma^2={sigma2}$")
    ax.set_ylim(0, 1)
    ax.set_xlim(1e-3, 1e3)
    ax.set_xticks([1e-2, 1e0, 1e2])
    ax.set_xticklabels([r"$10^{-2}$", r"$10^{0}$", r"$10^{2}$"])
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels([r"$0$", r"$\frac{1}{2}$", r"$1$"])
    
    

def plot_figure_S12(fig, axes, sim_results, hayward_results, double_recursion_results, output_file):
    '''
    Given the results from the simulations and analytic predictions, plots Figure S12.
    '''
    
    
    for ax_index, (ax, (shift, sigma2)) in enumerate(zip(axes, PANEL_LETTER.keys())):
        plot_panel(ax, shift, sigma2, sim_results, hayward_results, double_recursion_results, ax_index)
    
    double_recursion = Line2D([], [], ls="-", color="k", label="Eq. S16")
    hayward = Line2D([], [], ls="--", color="k", label="Hayward and Sella (2022)")
    mean_handle = axes[1].errorbar([np.nan], [np.nan], yerr=[np.nan], marker=".",
                                      ls="", markersize=5, color="k",
                                      label=r"Mean $\pm$ 2SEs")
    blank = Line2D([], [], ls="", marker="", label="")
    
    large_effect_size_range = mpatches.Patch(color='0.4', label='Large')
    high_midrange_effect_size_range = mpatches.Patch(color='0.6', label='High midrange')
    low_midrange_effect_size_range = mpatches.Patch(color='0.8', label='Low midrange')
    
    
    # column-major order, as matplotlib fills the legend grid top-to-bottom
    handles = [double_recursion, low_midrange_effect_size_range, 
               hayward, high_midrange_effect_size_range, 
               mean_handle, large_effect_size_range]
    
    axes[1].legend(handles=handles, ncols=3, loc="center", handlelength=1.8,
                      fontsize=FONTSIZE, bbox_to_anchor=(0.5, -0.55),
                      edgecolor="k", framealpha=1)
    
    plt.subplots_adjust(wspace=0.2)
    return fig
    
def main():
    
    SIMULATION_PICKLES = snakemake.input.simulation_pickles
    HAYWARD_PICKLES = snakemake.input.hayward_pickles
    DOUBLE_RECURSION_PICKLES = snakemake.input.double_recursion_pickles
    OUTPUT_FILE = snakemake.output[0]
    
    sim_results = _load_single_site_simulations(SIMULATION_PICKLES)
    hayward_results = _load_hayward(HAYWARD_PICKLES)
    double_recursion_results = _load_double_recursion(DOUBLE_RECURSION_PICKLES)


    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    fig = plot_figure_S12(fig, axes, sim_results, hayward_results, double_recursion_results, OUTPUT_FILE)
    cf.make_figure_set_width(fig, OUTPUT_FILE, target_width_inches=6.5, dpi=300)
    
    
if __name__ == "__main__":
    main()