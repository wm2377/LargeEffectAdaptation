"""
Standalone script: Figure S15, the GWAS power to detect a single large-effect fixation as a
function of the divergence time between two populations.

Panel A: the sample size needed for 90% power to detect (via a one-degree-of-freedom F
    test) the difference in mean phenotype caused by one large-effect fixed allele, vs
    divergence time T (in 2N generations).
Panel B: the corresponding variance explained by that fixation at the sample size n = 200.

Both panels are drawn for three distributions of squared effect sizes (the DFE):
    * Exponential mixture (cornflowerblue): small + large exponential components (CustomSDist)
    * Log-uniform         (firebrick):      scipy.stats.loguniform(0.01, 1000)
    * Simons et al. (2022) (purple):        an empirical SSD read from Simons_2022_SSD_dfe.mat
and for two background additive variances V_A in {100, 300} (solid / dashed).

Run either as a Snakemake script (reads snakemake.input.ssd, saves to
snakemake.output.figure_s15) or directly:
    python Figure_S15.py [output.png] [Simons_2022_SSD_dfe.mat]
Direct runs default to results/plots/Figure_S15.png next to this script, reading
Simons_2022_SSD_dfe.mat from the same directory. The DFE integrals are somewhat slow
(a minute or two).
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.io as sio
from scipy.integrate import quad
from scipy.optimize import root
from scipy.interpolate import CubicSpline

import common_functions as cf
from common_functions import CurvedText
from common_functions import fixation_probability_steady_state as probability_fixation
from analytic_functions import folded_sojourn_time as sojourn_time


# fixed model parameters
ALPHA = 0.05           # test significance level
MIN_POWER = 0.9        # target power for Panel A's sample-size curve
H2 = 0.5               # heritability
N = 5000               # population size
A2 = 100               # squared effect size of the focal large-effect fixation
N_SAMPLE = 200         # sample size used for Panel B's variance-explained curve
VA_VALUES = (100, 300)         # background additive variances (solid / dashed)
LINESTYLES = ("-", "--")
COLORS = ("cornflowerblue", "firebrick", "purple")   # one per DFE (order matches build_sdists)


# --------------------------------------------------------------------------- #
# Variance contributions of the DFE (per unit of mutational input 2NU)
# --------------------------------------------------------------------------- #
def get_variance_from_new_background_fixations_given_a2(a2, N):
    """Variance contributed by a *new* background allele of squared effect a2 that fixes."""
    a = np.sqrt(a2)
    return a2 / 2 * probability_fixation(a=a, x=1 / (2 * N))


def get_variance_from_background_fixations_seg_given_a2(a2, N):
    """Variance from a *segregating* background allele of effect a2 that goes on to fix."""
    a = np.sqrt(a2)
    return quad(lambda x: a2 / 2 * probability_fixation(a=a, x=x) * sojourn_time(S=a2, x=x, N=N),
                0, 1 / 2, points=[1 / (2 * N)])[0]


def get_variance_from_segregating_alleles_given_a(a2, N):
    """Standing (polymorphic) variance from segregating alleles of effect a2."""
    return quad(lambda x: 2 * a2 * (x / 2) * (1 - x / 2) * sojourn_time(S=a2, x=x, N=N),
                0, 1 / 2, points=[1 / (2 * N)])[0]


def get_v_divergence_per_mutational_input(N, sdist):
    """(new, seg) divergence variance per unit mutational input, integrated over the DFE."""
    new = 2 * quad(lambda a2: get_variance_from_new_background_fixations_given_a2(a2=a2, N=N) * sdist.pdf(a2),
                   0, sdist.ppf(0.9999), points=[sdist.ppf(0.01), sdist.ppf(0.99)])[0]
    seg = 2 * quad(lambda a2: get_variance_from_background_fixations_seg_given_a2(a2=a2, N=N) * sdist.pdf(a2),
                   0, sdist.ppf(0.9999), points=[sdist.ppf(0.01), sdist.ppf(0.99)])[0]
    return new, seg


def get_v_polymorphism_per_mutational_input(N, sdist):
    """Standing polymorphism variance per unit mutational input, integrated over the DFE."""
    return 2 * quad(lambda a2: get_variance_from_segregating_alleles_given_a(a2=a2, N=N) * sdist.pdf(a2),
                    0, sdist.ppf(0.9999), points=[sdist.ppf(0.01), sdist.ppf(0.99)])[0]


def get_variance_contribution_given_a2(a2, N):
    return quad(lambda x: 2 * x * (1 - x) * a2 * sojourn_time(S=a2, x=x, N=N),
                0, 1 / 2, points=[1 / (2 * N)])[0]


def get_variance_contribution_per_mutational_input(N, sdist):
    return quad(lambda a2: get_variance_contribution_given_a2(a2=a2, N=N) * sdist.pdf(a2),
                0, sdist.ppf(0.9999), points=[sdist.ppf(0.01), sdist.ppf(0.99)])[0]


def get_mutational_input(Va, sdist, N):
    """Solve for the mutational input 2NU that yields the target background variance Va."""
    seg = get_variance_contribution_per_mutational_input(N, sdist)
    return Va / (seg * 2 * N)


def slope_intercept(Va, sdist, N):
    """Return (slope, intercept) of background variance V_A(T) = slope*T + intercept.

    The slope is the divergence variance accumulated per generation (new fixations); the
    intercept is the standing polymorphism variance. Both are scaled by the mutational
    input 2NU implied by the target background variance Va.
    """
    U = get_mutational_input(Va, sdist, N)
    new, seg = get_v_divergence_per_mutational_input(N, sdist)
    new = new * 2 * N * U
    v_poly = get_v_polymorphism_per_mutational_input(N, sdist) * 2 * N * U
    return new, v_poly


def calculate_V_total(T, slope, intercept, h2):
    """Total phenotypic variance at divergence time T (background + the focal fixation)."""
    Va = T * slope + intercept + 100 / 2
    return Va / h2


def calculate_he(T, slope, intercept, h2, a2):
    """Heritability contributed by the focal fixation of squared effect a2 at time T."""
    return (a2 / 2) / calculate_V_total(T, slope, intercept, h2)


# --------------------------------------------------------------------------- #
# Power / sample size for detecting the focal fixation (non-central F test)
# --------------------------------------------------------------------------- #
def get_power(T, slope, intercept, h2, a2, n, alpha):
    """(power, he) to detect the fixation at sample size n via a 1-df F test."""
    he = calculate_he(T, slope, intercept, h2, a2)
    noncentrality_parameter = n * he
    f_central = stats.f(dfn=1, dfd=n)
    f_noncentral = stats.ncf(dfn=1, dfd=n, nc=noncentrality_parameter)
    lambda_alpha = f_central.ppf(1 - alpha)
    beta = f_noncentral.cdf(lambda_alpha)
    return 1 - beta, he


def get_sample_size_needed_for_minimum_power(T, slope, intercept, h2, a2, min_power, alpha, default_n=100):
    """Smallest sample size n reaching min_power; retries from a smaller guess if root fails."""
    sol = root(lambda n: get_power(T, slope, intercept, h2, a2, n, alpha)[0] - min_power, default_n)
    if sol.success or default_n != 100:
        return sol.x[0]
    return get_sample_size_needed_for_minimum_power(T, slope, intercept, h2, a2, min_power, alpha, default_n=10)


def calculate_min_sample_sizes_for_T(sdist, N, Va, times, h2, a2, min_power, alpha):
    slope, intercept = slope_intercept(Va=Va, sdist=sdist, N=N)
    return [get_sample_size_needed_for_minimum_power(T=t, slope=slope, intercept=intercept,
                                                     h2=h2, a2=a2, min_power=min_power, alpha=alpha)
            for t in times]


def calculate_power_given_n_for_T(sdist, N, Va, times, h2, a2, n, alpha):
    slope, intercept = slope_intercept(Va=Va, sdist=sdist, N=N)
    return [get_power(t, slope, intercept, h2, a2, n, alpha) for t in times]


# --------------------------------------------------------------------------- #
# Distributions of squared effect sizes (DFEs)
# --------------------------------------------------------------------------- #
class CustomSDist:
    """Exponential mixture DFE: a small-effect and a large-effect exponential component."""

    def __init__(self):
        self.small_effect_distribution = stats.expon(loc=0, scale=1)
        self.large_effect_distribution = stats.expon(loc=100, scale=400)
        self.p = 0.5

    def pdf(self, a2):
        return self.small_effect_distribution.pdf(a2) * (1 - self.p) + \
            self.large_effect_distribution.pdf(a2) * self.p

    def cdf(self, a2):
        return self.small_effect_distribution.cdf(a2) * (1 - self.p) + \
            self.large_effect_distribution.cdf(a2) * self.p

    def ppf(self, q):
        if q < self.p:
            default = self.small_effect_distribution.ppf(q * 2)
        else:
            default = self.large_effect_distribution.ppf((q - 0.5) * 2)
        sol = root(lambda a2: self.cdf(a2) - q, default)
        if sol.success:
            return sol.x[0]
        raise ValueError(f"CustomSDist.ppf failed for q={q}")


class YuvalSdist:
    """Empirical DFE interpolated (cubic spline) from the Simons et al. SSD data."""

    def __init__(self, s_values, pdf_values, interpolation):
        self.x = s_values
        self.y = pdf_values
        self.interpolation = interpolation
        self.cdf_total = 0
        self.max_x = 100
        self.min_x = min(s_values)

    def __initialize__(self):
        # normalisation over the supported range, evaluated once
        self.cdf_total = quad(lambda aa: self.pdf(aa), self.min_x, self.max_x)[0]

    def pdf(self, a2):
        return max(0, self.interpolation(a2))

    def cdf(self, a2):
        return quad(lambda aa: self.pdf(aa), 0, a2)[0] / self.cdf_total

    def ppf(self, q):
        return root(lambda a2: self.cdf(a2) - q, 1).x[0]


def build_sdists(ssd_path):
    """Load the empirical SSD and return the three DFEs (in COLORS order)."""
    data = sio.loadmat(ssd_path)
    thing1, _thing2 = data["figdata"][0][0]
    ssd_plots = thing1[0][0]
    s_values = 10 ** (np.reshape(ssd_plots[0], [129, 1])[:, 0]) * 2 * 20000
    pdf_values = np.reshape(ssd_plots[1], [129, 1])[:, 0]
    ff = CubicSpline(s_values, pdf_values)   # cubic-spline interpolation of the SSD

    expon_mixture = CustomSDist()
    loguniform = stats.loguniform(0.01, 1000)
    yy = YuvalSdist(s_values, pdf_values, ff)
    yy.__initialize__()
    return [expon_mixture, loguniform, yy]


def compute_results(sdists, N, times, h2, a2, n, min_power, alpha):
    """results[Va][sdist] = (min_sample_sizes[T], [(power, he)][T]) for each (Va, DFE)."""
    results_dict = {}
    for Va in VA_VALUES:
        results_dict[Va] = {}
        for sdist in sdists:
            results_dict[Va][sdist] = (
                calculate_min_sample_sizes_for_T(sdist, N, Va, times, h2, a2, min_power, alpha),
                calculate_power_given_n_for_T(sdist, N, Va, times, h2, a2, n, alpha),
            )
    return results_dict


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def create_figure(fig_width, fig_height, sdists, results_dict, times, N):
    """Draw Panels A (sample size) and B (variance explained).

    Returns (fig, fig_width, fig_height, width_inches, height_inches) for
    common_functions.make_figure_set_width to rescale to a target width.
    """
    fig, axes = plt.subplots(ncols=2, dpi=300, figsize=(fig_width, fig_height))
    fontsize = 11

    # Panel A: sample size for 90% power vs divergence time
    ax = axes[0]
    plt.sca(ax)
    for Va, ls in zip(VA_VALUES, LINESTYLES):
        for sdist, color in zip(sdists, COLORS):
            min_sample_size = results_dict[Va][sdist][0]
            plt.plot(times / (2 * N), min_sample_size, color=color, ls=ls)
            # label the V_A = 100/300 curves in place along the log-uniform (firebrick) line
            if color == "firebrick":
                t0 = int(1000 / 20)
                CurvedText(axes=ax, x=times[t0:] / (2 * N), y=np.array(min_sample_size[t0:]) + 8,
                           text=f"V ={Va}", color="k", size=12)
                CurvedText(axes=ax, x=times[t0:] / (2 * N) + 0.1 / 2, y=np.array(min_sample_size[t0:]) + 5,
                           text=" A", color="k", size=8)
    plt.xlim([0, 10])
    plt.ylim([0, 400])
    plt.xticks(np.linspace(0, 10, 3))
    plt.yticks([0, 100, 200, 300, 400])
    plt.xlabel("Divergence time ($2N$ generations)", size=fontsize, labelpad=0)
    plt.ylabel("Sample size", size=fontsize, labelpad=0)
    plt.title(r"$\bf{A.}$ Sample size for 90% power", size=fontsize)
    plt.plot([], [], color="cornflowerblue", label="Exponential")
    plt.plot([], [], color="firebrick", label="Log-uniform")
    plt.plot([], [], color="purple", label="Simons et al. (2025)")
    plt.legend(framealpha=1, edgecolor="k", fontsize=fontsize, handlelength=1,
               handletextpad=0.5, ncol=3, loc=(0.08, -0.33))

    # Panel B: variance explained at fixed sample size n
    ax = axes[1]
    plt.sca(ax)
    for Va, ls in zip(VA_VALUES, LINESTYLES):
        for sdist, color in zip(sdists, COLORS):
            power = np.array([i[1] for i in results_dict[Va][sdist][1]])   # he = variance explained
            plt.plot(times / N, power, color=color, ls=ls)
            if color == "firebrick":
                t0 = int(1000 / 20)
                CurvedText(axes=ax, x=times[t0:] / (2 * N), y=np.array(power[t0:]) + 8,
                           text=f"V ={Va}", color="k", size=10)
                CurvedText(axes=ax, x=times[t0:] / (2 * N) + 0.1, y=np.array(power[t0:]) + 5,
                           text=" A", color="k", size=6)
    plt.xlim([0, 10])
    plt.xticks(np.linspace(0, 10, 3))
    plt.ylim([0, 0.165])
    plt.yticks([0, 0.05, 0.1, 0.15])
    plt.title(r"$\bf{B.}$ Variance explained by large fixation", size=fontsize)
    plt.xlabel("Divergence time ($2N$ generations)", size=fontsize, labelpad=0)
    plt.ylabel("Variance explained", size=fontsize, labelpad=0)
    plt.subplots_adjust(wspace=0.3)

    bbox = ax.get_window_extent()
    width_inches = bbox.width / fig.dpi
    height_inches = bbox.height / fig.dpi
    return fig, fig_width, fig_height, width_inches, height_inches


def main(out_path, ssd_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    times = np.linspace(0, 20, 1000) * N
    sdists = build_sdists(ssd_path)
    results_dict = compute_results(sdists, N=N, times=times, h2=H2, a2=A2,
                                   n=N_SAMPLE, min_power=MIN_POWER, alpha=ALPHA)
    fig = create_figure(fig_width=6.5, fig_height=3, sdists=sdists,
                        results_dict=results_dict, times=times, N=N)[0]
    cf.make_figure_set_width(fig, out_path, target_width_inches=6.5, dpi=300)


if "snakemake" in globals():
    out = snakemake.output    # noqa: F821
    inp = snakemake.input     # noqa: F821
    main(getattr(out, "figure_s15", None) or out[0],
         getattr(inp, "ssd", None) or inp[0])
elif __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    default_out = os.path.join(here, "results", "plots", "Figure_S15.png")
    default_ssd = os.path.join(here, "Simons_2022_SSD_dfe.mat")
    main(sys.argv[1] if len(sys.argv) > 1 else default_out,
         sys.argv[2] if len(sys.argv) > 2 else default_ssd)
