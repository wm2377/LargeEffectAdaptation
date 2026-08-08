"""
Standalone script: Figure 1, a five-panel schematic of the model.

Unlike Figure_4 / Figure_S7, this figure has no data inputs -- every panel is
drawn from hard-coded, illustrative parameters -- so the script can be run on its
own without any simulation/analytic pickles.

Panels
    A: genotype -> phenotype mapping (additive allelic + environmental effects)
    B: the (Gaussian) fitness function
    C: the two effect-size-squared distributions (exponential vs uniform)
    D: the Wright-Fisher population life cycle
    E: the evolutionary scenario (a shift in the optimum)

Run either as a Snakemake script (saves to snakemake.output[0]) or directly:
    python Figure_1.py [output.png]
Direct runs default to results/plots/Figure_1.png next to this script.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import stats

from common_functions import CurvedText


# --------------------------------------------------------------------------- #
# Panel C: distribution of effect sizes
# --------------------------------------------------------------------------- #
def plot_effect_size_distribution(sdist, color, ax):
    z_values = np.linspace(0, 1, 200)[:-1]
    x_values = sdist.ppf(z_values)
    y_values = sdist.pdf(x_values)
    ax.plot(x_values, y_values, color=color)
    ax.plot([0, 100, 100], [0, 0, y_values[0]], color=color, ls='-')
    ax.plot([x_values[-1], x_values[-1], 10000], [y_values[-1], 0, 0], color=color, ls='-')


def make_figure_1_C(ax):
    sdist = stats.expon(scale=400, loc=100)
    plot_effect_size_distribution(sdist=sdist, color='r', ax=ax)
    sdist = stats.uniform(100, 900)
    plot_effect_size_distribution(sdist=sdist, color='purple', ax=ax)
    ax.set_xlabel('Effect size squared', size=11)
    ax.set_ylabel('Density', size=11)
    ax.set_title(r'$\bf{C.}$ Distribution of effect sizes', size=11)
    ax.set_xlim([0, 2000])
    ax.set_xticks([100, 1000, 2000])
    ax.set_ylim([0, 0.0026])
    ax.set_yticks([])
    # Get the bounding box of the axes
    bbox = ax.get_window_extent()


# --------------------------------------------------------------------------- #
# Panel B: fitness function
# --------------------------------------------------------------------------- #
def plot_fitness_function(N, ax):
    fitness_dist = stats.norm(loc=0, scale=(2 * N) ** (0.5))
    z_values = np.linspace(0, 1, 2000)[1:-1]
    x_values = fitness_dist.ppf(z_values)
    y_values = fitness_dist.pdf(x_values)
    ax.plot(x_values, y_values, color='k')

    fitness_width = np.sqrt(2 * N) * 1.4
    ax.plot([0, fitness_width], [fitness_dist.pdf(fitness_width)] * 2, color='k', ls='-', lw=0.5, alpha=0.5)
    ax.plot([0, 0], [0, fitness_dist.pdf(0)], color='k', ls='-', lw=0.5, alpha=0.5)
    ax.text(x=fitness_width / 2, y=fitness_dist.pdf(fitness_width) / 1.05, s=r'$\sqrt{V_S}$', horizontalalignment='center', verticalalignment='top', size=11, color='k', alpha=1)
    ax.set_xlabel('Phenotype', size=11)
    ax.set_ylabel('Fitness', size=11)
    ax.set_title(r'$\bf{B.}$ Fitness function', size=11)
    ax.set_xticks([0])
    ax.set_yticks([])
    ax.set_xticklabels(['Optimum'])
    ## set x tick label font size to be 11
    ax.tick_params(axis='x', labelsize=11)
    ax.set_xlim([min(x_values), max(x_values)])
    ax.set_ylim([0, max(y_values) * 1.1])
    # ax.set_xlim([0,1])


def make_figure_1_B(ax):
    plot_fitness_function(N=5000, ax=ax)


# --------------------------------------------------------------------------- #
# Panel A: genotype -> phenotype mapping
# --------------------------------------------------------------------------- #
def plot_genotype_to_phenotype_mapping(ax):
    mu = 0
    aligned_effects = [10, 20, 15, 25, 12, 28]
    aligned_effects = np.array(aligned_effects) * 2.25
    opposing_effects = [-20, -15, -17, -25]
    opposing_effects = np.array(opposing_effects) * 2.25
    current_position = 0
    for effect in aligned_effects:
        current_position += effect
        ax.annotate('',
            xy=(current_position, 0), xycoords='data',
            xytext=(current_position - effect, 0), textcoords='data',
            arrowprops=dict(facecolor='blue', shrink=0.02, edgecolor='blue', headlength=5, headwidth=5, width=2),
            horizontalalignment='center', verticalalignment='bottom')
    for effect in opposing_effects:
        # plot an arrow
        current_position += effect
        ax.annotate('',
            xy=(current_position, 1), xycoords='data',
            xytext=(current_position - effect, 1), textcoords='data',
            arrowprops=dict(facecolor='green', shrink=0.02, edgecolor='green', headlength=5, headwidth=5, width=2),
            horizontalalignment='center', verticalalignment='bottom')
        # ax.arrow(current_position,1,effect+head_length+5,0,head_width=head_width, head_length=head_length, fc='green', ec='green',lw=5)

    # ax.arrow(current_position,2,50,0,head_width=head_width, head_length=head_length, fc='k', ec='k')
    environmental_effect = 240
    current_position += environmental_effect
    ax.annotate('',
            xy=(current_position, 2), xycoords='data',
            xytext=(current_position - environmental_effect, 2), textcoords='data',
            arrowprops=dict(facecolor='black', shrink=0.02, headlength=5, headwidth=5, width=2),
            horizontalalignment='center', verticalalignment='bottom')
    ax.set_yticks([])
    ax.set_ylim([-0.25, 2.7])
    ax.set_xticks([mu, current_position])
    ax.set_xticklabels([r'$0$', r'$z$'])
    ax.set_xlim([-10, current_position + 10])
    ax.set_xlabel('Phenotype', size=11, labelpad=0.1)
    ax.text(x=current_position - environmental_effect / 2, y=2.175, s='environmental effect', horizontalalignment='center', size=11)
    ax.text(x=(current_position - environmental_effect) - sum(opposing_effects) / 2, y=1.175, s='negative genetic effects', horizontalalignment='center', size=11, color='green')
    ax.text(x=sum(aligned_effects) / 2, y=0.175, s='positive genetic effects', horizontalalignment='center', size=11, color='blue')
    ax.plot([current_position, current_position], [-10, 20], color='k', ls='--', lw=0.5)


def make_figure_1_A(ax):
    plot_genotype_to_phenotype_mapping(ax=ax)
    ax.set_title(r'$\bf{A.}$ Genotype to phenotype mapping', size=11)


# --------------------------------------------------------------------------- #
# Panel E: evolutionary scenario (optimum shift)
# --------------------------------------------------------------------------- #
def plot_normal_distribution(N, mean, color, ax, ls='-', fill=False, label_x=0, label='', label_name='', label_name_x=0, fontsize=11, lw=1, annotation_lw=0.5):

    normdist = stats.norm(loc=mean, scale=np.sqrt(2 * N))
    z = np.linspace(0, 1, 20000)
    x = normdist.ppf(z)
    y = normdist.pdf(x) / normdist.pdf(mean)
    ax.plot(x, y, color=color, ls=ls, lw=lw)
    text_width = normdist.std()
    if not fill:
        ax.plot([mean, mean], [0, 1], color=color, ls=ls, lw=annotation_lw, alpha=1)
    else:
        ax.fill_between(x, y, 0, color=color, alpha=0.1)
    if label:
        if label == r'$\sqrt{V_S}$':
            text_width -= 1
            y_label = normdist.pdf(text_width) / normdist.pdf(mean)
            y_label_adjust = 0.02
            verticalalignment = 'bottom'
        else:
            text_width += 1
            y_label = normdist.pdf(text_width) / normdist.pdf(mean)
            y_label_adjust = -0.02
            verticalalignment = 'top'
        ax.plot([0, text_width], [y_label] * 2, color=color, ls=ls, zorder=100, lw=annotation_lw)
        # ax.text(x=text_width/2*0.9,y=y_label+y_label_adjust,s=label,horizontalalignment='center',size=11,color=color,verticalalignment=verticalalignment)
        ax.text(x=text_width / 2, y=y_label + y_label_adjust, s=label, horizontalalignment='center', size=fontsize, color=color, verticalalignment=verticalalignment)

    if label_name:
        if label_name_x == 0:
            label_name_x = -130
        x_label = np.linspace(label_name_x, 250, 1000)
        y_label = normdist.pdf(x_label) / normdist.pdf(mean)
        if label_name == 'Initial fitness function':
            x_label = x_label - 5
        elif label_name == 'New fitness function':
            x_label = x_label + 5
        else:
            y_label = y_label + 0.01
            x_label = x_label - 3
        g = CurvedText(x=x_label, y=y_label, text=label_name, axes=ax, fontsize=fontsize, color=color)


def make_figure_1_E(ax, fontsize=11, lw=1, annotation_lw=0.5):
    shift = 75
    plot_normal_distribution(N=5000, mean=0, color='k', ax=ax, label_x=100, label=r'$\sqrt{V_S}$', label_name='Initial fitness function', label_name_x=-145, fontsize=fontsize, lw=lw, annotation_lw=annotation_lw)
    plot_normal_distribution(N=5000, mean=shift, color='k', ls='--', ax=ax, label_name='New fitness function', label_name_x=110, fontsize=fontsize, lw=lw, annotation_lw=annotation_lw)
    plot_normal_distribution(N=800 / 2, mean=0, color=[0.773, 0.353, 0.067], ls='-', ax=ax, fill=True, label_x=40, label=r'$\sqrt{V_A}$', label_name='Initial trait distribution', label_name_x=-150, fontsize=fontsize, lw=lw, annotation_lw=annotation_lw)
    ax.set_xlabel('Phenotype', size=11)
    width = 200
    ax.set_xlim([shift / 2 - width, shift / 2 + width])
    ax.set_xticks([0, shift])
    ax.set_xticklabels(['Initial\noptimum', 'New\noptimum'])
    ## set x tick label font size to be 11
    ax.tick_params(axis='x', labelsize=11)
    ax.annotate('',
                xy=(shift, 1.05), xycoords='data',
                xytext=(0, 1.05), textcoords='data',
                arrowprops=dict(facecolor='k', shrink=0.02, edgecolor='k', headlength=5),
                horizontalalignment='center', verticalalignment='bottom')
    ax.set_ylim([0, 1.2])
    ax.set_yticks([])
    ax.text(x=shift / 2, y=1.1, s='Shift size ' + r'$\Lambda$', horizontalalignment='center', size=fontsize)
    ax.set_title(r'$\bf{E.}$ Evolutionary Scenario', size=11)


# --------------------------------------------------------------------------- #
# Panel D: population life cycle
# --------------------------------------------------------------------------- #
def make_figure_1_D(ax):
    ax.set_title(r'$\bf{D.}$ Population dynamics', size=11)
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    rect_height_total = 0.5
    rect_height = rect_height_total / 5
    distance_between_rects = (0.92 - rect_height_total) / 4
    rectangles = {}
    color = [0.773, 0.353, 0.067]
    edge_color_save = [jj * 0.6 for jj in color]

    for i in range(5):
        min_y = 0.05 + (rect_height + distance_between_rects) * i
        max_y = min_y + rect_height
        min_x = 0.2
        max_x = min_x + 0.775

        if i == 4:
            face_color = [1, 1, 1, 0]
            edge_color = 'k'
        else:
            face_color = color
            edge_color = edge_color_save
        rect = FancyBboxPatch((min_x, min_y), max_x - min_x, max_y - min_y, boxstyle="round,pad=0.02", ec=edge_color, fc=face_color, mutation_aspect=1, lw=1.5)

        if i == 4:
            ax.text(x=(min_x + max_x) / 2, y=(min_y + max_y) / 2, s=r'Population of size $N$', horizontalalignment='center', verticalalignment='center', size=11, color='k')
        elif i == 3:
            ax.text(x=(min_x + max_x) / 2, y=(min_y + max_y) / 2, s=r'Selection', horizontalalignment='center', verticalalignment='center', size=11, color='white')
        elif i == 2:
            ax.text(x=(min_x + max_x) / 2, y=(min_y + max_y) / 2, s=r'Mutation', horizontalalignment='center', verticalalignment='center', size=11, color='white')
        elif i == 1:
            ax.text(x=(min_x + max_x) / 2, y=(min_y + max_y) / 2, s='Free recombination &\nmendelian segregation', horizontalalignment='center', verticalalignment='center', size=11, color='white')
        elif i == 0:
            ax.text(x=(min_x + max_x) / 2, y=(min_y + max_y) / 2, s='Random union of\ngametes', horizontalalignment='center', verticalalignment='center', size=11, color='white')
        ax.add_patch(rect)
        rectangles[i] = (min_x, max_x, min_y, max_y)

    for i, j in [(4, 3), (3, 2), (2, 1), (1, 0)]:
        min_x, max_x, min_y, max_y = rectangles[i]
        min_x2, max_x2, min_y2, max_y2 = rectangles[j]
        mid_x = (min_x + max_x) / 2
        headlength = 0.02
        ax.annotate('',
                xy=(mid_x, max_y2 + headlength), xycoords='data',
                xytext=(mid_x, min_y - headlength), textcoords='data',
                arrowprops=dict(facecolor=edge_color_save, edgecolor='None', shrink=0.02, headlength=5, headwidth=7, width=2),
                horizontalalignment='center', verticalalignment='center')

    rect_top = rectangles[4]
    rect_bottom = rectangles[0]
    min_x, max_x, min_y, max_y = rect_top
    min_x2, max_x2, min_y2, max_y2 = rect_bottom
    mid_y = (min_y + max_y) / 2
    mid_y2 = (min_y2 + max_y2) / 2
    ax.annotate('',
        xy=(0.175, mid_y), xycoords='data',
        xytext=(0.1, mid_y), textcoords='data',
        arrowprops=dict(facecolor=edge_color_save, edgecolor='None', shrink=0.02, headlength=5, headwidth=7, width=2),
        horizontalalignment='center', verticalalignment='center')
    ax.annotate('',
        xy=(0.1, mid_y + 0.0205), xycoords='data',
        xytext=(0.1, mid_y2 - 0.0205), textcoords='data',
        arrowprops=dict(facecolor=edge_color_save, edgecolor='None', shrink=0.02, headlength=0.0001, headwidth=0.0001, width=2),
        horizontalalignment='center', verticalalignment='center')
    ax.annotate('',
        xy=(0.175, mid_y2), xycoords='data',
        xytext=(0.1, mid_y2), textcoords='data',
        arrowprops=dict(facecolor=edge_color_save, edgecolor='None', shrink=0.02, headlength=0.0001, headwidth=0.0001, width=2),
        horizontalalignment='center', verticalalignment='center')
    ax.text(x=0.05, y=0.5, s=r'Next generation', horizontalalignment='center', verticalalignment='center', size=11, rotation=90)


# --------------------------------------------------------------------------- #
# Assemble the full figure
# --------------------------------------------------------------------------- #
def make_figure_1(out_path, fig_width=8, fig_height=8):

    fig = plt.figure(dpi=300, figsize=(8, 6))
    y_scaling = fig_height / fig_width

    b_width = 0.25
    b_height = b_width * fig_width / fig_height

    a_min_x = 0.025
    a_max_x = 0.35
    a_max_y = 0.96
    a_min_y = 0.75
    a_y = a_min_y
    a_x = a_min_x
    a_width = a_max_x - a_min_x
    a_height = a_max_y - a_min_y

    pad_between_axes = 0.06
    b_x = a_max_x + pad_between_axes
    b_width = (1 - 0.025 - b_x - pad_between_axes) / 2
    b_height = b_width * y_scaling
    b_y = 0.96 - b_height

    c_x = 1 - 0.025 - b_width
    c_width = b_width
    c_height = b_height
    c_y = b_y

    a = fig.add_axes(rect=[a_x, a_y, a_width, a_height])
    b = fig.add_axes(rect=[b_x, b_y, b_width, b_height])
    c = fig.add_axes(rect=[c_x, c_y, c_width, c_height])

    d_top = 0.5
    d_height = 0.5
    d_x = 0
    d_width = 0.3
    d_y = d_top - d_height

    e_x = d_x + d_width + pad_between_axes
    e_width = 1 - e_x - 0.025
    e_top = d_top
    e_bottom = 0.115
    e_height = e_top - e_bottom
    e_y = e_bottom
    # a=fig.add_axes(rect=[0+pad,0.75+pad,0.4-pad,0.25-pad])

    # b=fig.add_axes(rect=[0.4+pad,0.7+pad,0.3-pad,0.3-pad])
    # c=fig.add_axes(rect=[0.7+pad,0.7+pad,0.3-pad,0.3-pad])
    d = fig.add_axes(rect=[d_x, d_y, d_width, d_height])
    e = fig.add_axes(rect=[e_x, e_y, e_width, e_height])

    make_figure_1_A(a)
    make_figure_1_B(b)
    make_figure_1_C(c)
    make_figure_1_E(e)
    make_figure_1_D(d)

    fig.text(x=(a_min_x + a_max_x) / 2, y=0.65, s=r'$z=\Sigma_{i=1}^{L}(a_i+a^{\prime}_i)+\epsilon$,  $\epsilon$~$N(0,V_E)$', horizontalalignment='center', size=11)
    fig.text(x=(a_min_x + a_max_x) / 2, y=0.61, s=r'$a_i$ & $a^{\prime}_i$ - effect size of alleles at site $i$', horizontalalignment='center', size=11)
    # fig.text(x=(a_min_x+a_max_x)/2,y=0.57,s=r'$\mu=$ phenotype of fixed background',horizontalalignment='center',size=11)
    fig.text(x=b_x + b_width / 2, y=0.5875, s=r'$1/\sqrt{V_S}$ - strength of selection', horizontalalignment='center', size=11)
    fig.text(x=c_x + c_width / 2, y=0.605, s=r'$a^2=100+x$  $x$~$Exp(\frac{1}{400})$', horizontalalignment='center', size=11, clip_on=False, color='r')
    fig.text(x=c_x + c_width / 2, y=0.57, s=r'$a^2$~$U(100,1000)$', horizontalalignment='center', size=11, clip_on=False, color='purple')
    plt.tight_layout(h_pad=0, w_pad=0)

    facecolor = [0.92, 0.95, 1]
    edgecolor = 'navy'
    #
    row_1_y = d_top + 0.05
    row_1_height = 1 - row_1_y
    row_2_y = 0
    row_2_height = row_1_y - 0.01
    fig.patches.extend([FancyBboxPatch((0, row_1_y), a_width + a_x + 0.02, row_1_height, boxstyle='Round, pad=0,rounding_size=0.01', mutation_aspect=1, facecolor=facecolor, edgecolor=edgecolor, clip_on=False, lw=1.5, transform=fig.transFigure, figure=fig, zorder=-1)])
    fig.patches.extend([FancyBboxPatch((a_width + a_x + 0.03, row_1_y), (1 - (a_width + a_x + 0.03) - 0.01) / 2, row_1_height, boxstyle='Round, pad=0,rounding_size=0.01', mutation_aspect=1, facecolor=facecolor, edgecolor=edgecolor, clip_on=False, zorder=-1, lw=1.5, transform=fig.transFigure, figure=fig)])
    fig.patches.extend([FancyBboxPatch((a_width + a_x + 0.03 + (1 - (a_width + a_x + 0.03) - 0.01) / 2 + 0.01, row_1_y), (1 - (a_width + a_x + 0.03) - 0.01) / 2, row_1_height, boxstyle='Round, pad=0,rounding_size=0.01', mutation_aspect=1, facecolor=facecolor, edgecolor=edgecolor, clip_on=False, zorder=-1, lw=1.5, transform=fig.transFigure, figure=fig)])
    # fig.patches.extend([FancyBboxPatch((0,0),1,1, boxstyle='Round, pad=0,rounding_size=0.01', mutation_aspect=1,facecolor=[1,0,0,0.1],edgecolor=edgecolor,clip_on=False,zorder=-1,lw=1.5,transform=fig.transFigure, figure=fig)])
    fig.patches.extend([FancyBboxPatch((0, row_2_y), ((d_x + d_width) + e_x) / 2 - 0.005, row_2_height, boxstyle='Round, pad=0,rounding_size=0.01', mutation_aspect=1, facecolor=facecolor, edgecolor=edgecolor, clip_on=False, zorder=-1, lw=1.5, transform=fig.transFigure, figure=fig)])
    fig.patches.extend([FancyBboxPatch((((d_x + d_width) + e_x) / 2 + 0.005, row_2_y), 1 - (((d_x + d_width) + e_x) / 2 + 0.005), row_2_height, boxstyle='Round, pad=0,rounding_size=0.01', mutation_aspect=1, facecolor=facecolor, edgecolor=edgecolor, clip_on=False, zorder=-1, lw=1.5, transform=fig.transFigure, figure=fig)])
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def main(out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    make_figure_1(out_path=out_path)


if "snakemake" in globals():
    out = snakemake.output  # noqa: F821
    main(getattr(out, "figure_1", None) or out[0])
elif __name__ == "__main__":
    default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "plots", "Figure_1.png")
    main(sys.argv[1] if len(sys.argv) > 1 else default)
