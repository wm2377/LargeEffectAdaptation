// C++ core for the all-allele (allele-frequency) simulation.
//
// Same model as scripts/all_allele_model.py, Each generation, with the population a distance D from
// the optimum and V_S = 2N:
//
//     E[dx] = a / V_S * (D - a * (1/2 - x) * (1 - D^2 / V_S)) * x * (1 - x)
//     x'    = Binomial(2N, clamp(x + E[dx], 0, 1)) / 2N
//
// Alleles at x = 1 fold into a fixed background (shifting the mean phenotype by 2a);
// alleles at x = 0 are dropped; Poisson(2 * mu * N) new alleles enter at x = 1 / (2N).
// The distance is measured from the mean phenotype BEFORE the frequencies are updated.


#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace {

std::string g_last_error;

// ---------------------------------------------------------------------------------
// effect-size distribution: a two-component mixture of S = a^2
// ---------------------------------------------------------------------------------
struct EffectSizeDistribution {
    // component kinds: 0 = exponential(loc, scale), 1 = uniform(loc, loc + scale)
    int    small_kind  = 0;
    double small_loc   = 0.0;
    double small_scale = 1.0;
    int    large_kind  = 0;
    double large_loc   = 100.0;
    double large_scale = 400.0;
    double weight      = 0.0;   // probability a new allele is large-effect

    double draw(std::mt19937_64& rng) const {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        const bool is_large = u(rng) <= weight;
        const int    kind  = is_large ? large_kind  : small_kind;
        const double loc   = is_large ? large_loc   : small_loc;
        const double scale = is_large ? large_scale : small_scale;
        if (kind == 0) {
            std::exponential_distribution<double> e(1.0);
            return loc + scale * e(rng);
        }
        return loc + scale * u(rng);
    }
};

// ---------------------------------------------------------------------------------
// population
// ---------------------------------------------------------------------------------
struct Population {
    int    N  = 5000;
    double Vs = 10000.0;           // 2N
    double optimum = 0.0;
    double mutation_rate = 0.0;
    EffectSizeDistribution sdist;

    std::vector<double> a;         // signed effect size
    std::vector<double> a2;        // squared effect size
    std::vector<double> x;         // frequency

    double fixed_background = 0.0;
    std::vector<double> fixations;         // signed effect of every allele that fixed
    std::vector<double> new_fixations;     // since the last record

    std::mt19937_64 rng;

    double mean_phenotype() const {
        // pairwise summation would be tighter, but the allele count is small enough that
        // the naive sum matches the numpy reduction to well inside binomial noise
        double mean = fixed_background;
        for (std::size_t i = 0; i < a.size(); ++i) mean += 2.0 * a[i] * x[i];
        return mean;
    }

    void update_frequencies(double distance) {
        const double two_n = 2.0 * static_cast<double>(N);
        for (std::size_t i = 0; i < a.size(); ++i) {
            const double dx = a[i] / Vs
                            * (distance - a[i] * (0.5 - x[i]) * (1.0 - distance * distance / Vs))
                            * x[i] * (1.0 - x[i]);
            double p = x[i] + dx;
            // the Python engine clips here too; std::binomial_distribution requires p in
            // [0, 1] and the original Python would have raised
            p = std::min(1.0, std::max(0.0, p));
            std::binomial_distribution<long long> binom(static_cast<long long>(2 * N), p);
            x[i] = static_cast<double>(binom(rng)) / two_n;
        }
    }

    void handle_fixed_or_extinct() {
        std::size_t keep = 0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (x[i] >= 1.0) {
                fixations.push_back(a[i]);
                new_fixations.push_back(a[i]);
                fixed_background += 2.0 * a[i];
                continue;                       // drop
            }
            if (x[i] <= 0.0) continue;          // drop
            a[keep] = a[i]; a2[keep] = a2[i]; x[keep] = x[i];
            ++keep;
        }
        a.resize(keep); a2.resize(keep); x.resize(keep);
    }

    void add_new_mutations() {
        std::poisson_distribution<int> pois(2.0 * mutation_rate * static_cast<double>(N));
        const int n_new = pois(rng);
        if (n_new <= 0) return;
        std::uniform_real_distribution<double> u(0.0, 1.0);
        const double entry = 1.0 / (2.0 * static_cast<double>(N));
        for (int k = 0; k < n_new; ++k) {
            const double s = sdist.draw(rng);
            const double sign = (u(rng) < 0.5) ? -1.0 : 1.0;
            a2.push_back(s);
            a.push_back(std::sqrt(s) * sign);
            x.push_back(entry);
        }
    }

    void next_generation() {
        const double distance = optimum - mean_phenotype();
        update_frequencies(distance);
        handle_fixed_or_extinct();
        add_new_mutations();
    }

    void force_minor_alleles() {
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (x[i] > 0.5) { x[i] = 1.0 - x[i]; a[i] = -a[i]; }
        }
    }
};

// ---------------------------------------------------------------------------------
// steady-state sojourn-time density, for the initial frequencies
// ---------------------------------------------------------------------------------
inline double variance_star(double a2, double x) { return 2.0 * a2 * x * (1.0 - x); }

inline double sojourn_time(int N, double a2, double x) {
    const double v = 2.0 * std::exp(-variance_star(a2, x) / 2.0) / (1.0 - x);
    const double entry = 1.0 / (2.0 * static_cast<double>(N));
    return v * ((x < entry) ? 2.0 * static_cast<double>(N) : 1.0 / x);
}

// Inverse-CDF draw from the sojourn-time density, tabulated on the same style of grid the
// Python engine uses (dense below the entry frequency, log-spaced above it).
double draw_initial_frequency(int N, double a2, std::mt19937_64& rng, int n_grid = 2048) {
    const double entry = 1.0 / (2.0 * static_cast<double>(N));
    std::vector<double> grid;
    grid.reserve(16 + n_grid);
    for (int i = 0; i < 16; ++i) grid.push_back(entry * static_cast<double>(i) / 16.0);
    const double log_lo = std::log(entry), log_hi = std::log(0.5);
    for (int i = 0; i < n_grid; ++i) {
        grid.push_back(std::exp(log_lo + (log_hi - log_lo) * static_cast<double>(i)
                                         / static_cast<double>(n_grid - 1)));
    }
    std::vector<double> cdf(grid.size(), 0.0);
    for (std::size_t i = 1; i < grid.size(); ++i) {
        const double d0 = sojourn_time(N, a2, grid[i - 1]);
        const double d1 = sojourn_time(N, a2, grid[i]);
        cdf[i] = cdf[i - 1] + 0.5 * (d0 + d1) * (grid[i] - grid[i - 1]);
    }
    const double total = cdf.back();
    if (!(total > 0.0)) return entry;
    std::uniform_real_distribution<double> u(0.0, 1.0);
    const double p = u(rng) * total;
    const auto it = std::lower_bound(cdf.begin(), cdf.end(), p);
    const std::size_t i = static_cast<std::size_t>(it - cdf.begin());
    if (i == 0) return grid[0];
    const double c0 = cdf[i - 1], c1 = cdf[i];
    const double w = (c1 > c0) ? (p - c0) / (c1 - c0) : 0.0;
    return grid[i - 1] + w * (grid[i] - grid[i - 1]);
}

struct Context {
    Population pop;
    std::vector<double> bin_limits;
    double skew_max_a = 10.0;
};

}  // namespace

// ---------------------------------------------------------------------------------
// C interface
// ---------------------------------------------------------------------------------
extern "C" {

const char* aa_last_error(void) { return g_last_error.c_str(); }

// bin_limits: upper limits on S = a^2; an allele joins the FIRST bin it falls under.
void* aa_create(int N, double mutation_rate, double optimum, double weight,
                int large_kind, double large_loc, double large_scale,
                int n_bins, const double* bin_limits, double skew_max_a,
                unsigned long long seed) {
    try {
        Context* ctx = new Context();
        ctx->pop.N = N;
        ctx->pop.Vs = 2.0 * static_cast<double>(N);
        ctx->pop.optimum = optimum;
        ctx->pop.mutation_rate = mutation_rate;
        ctx->pop.sdist.weight = weight;
        ctx->pop.sdist.large_kind = large_kind;
        ctx->pop.sdist.large_loc = large_loc;
        ctx->pop.sdist.large_scale = large_scale;
        ctx->pop.rng.seed(seed);
        ctx->bin_limits.assign(bin_limits, bin_limits + n_bins);
        ctx->skew_max_a = skew_max_a;
        return ctx;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return nullptr;
    }
}

void aa_destroy(void* handle) { delete static_cast<Context*>(handle); }

// Seed the population from the steady-state sojourn-time density.
void aa_initialize(void* handle, int n_alleles) {
    Context* ctx = static_cast<Context*>(handle);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int i = 0; i < n_alleles; ++i) {
        const double s = ctx->pop.sdist.draw(ctx->pop.rng);
        const double sign = (u(ctx->pop.rng) < 0.5) ? -1.0 : 1.0;
        ctx->pop.a2.push_back(s);
        ctx->pop.a.push_back(std::sqrt(s) * sign);
        ctx->pop.x.push_back(draw_initial_frequency(ctx->pop.N, s, ctx->pop.rng));
    }
}

// Draw a Poisson count with the given mean, using the context RNG (so the caller can size
// the initial population exactly as the Python engine does).
int aa_poisson(void* handle, double mean) {
    Context* ctx = static_cast<Context*>(handle);
    std::poisson_distribution<int> pois(mean);
    return pois(ctx->pop.rng);
}

// Advance `generations` generations, recording nothing. This is the burn-in.
void aa_burn(void* handle, long long generations) {
    Context* ctx = static_cast<Context*>(handle);
    for (long long t = 0; t < generations; ++t) ctx->pop.next_generation();
}

void aa_shift_optimum(void* handle, double shift) {
    static_cast<Context*>(handle)->pop.optimum += shift;
}

void aa_force_minor_alleles(void* handle) {
    static_cast<Context*>(handle)->pop.force_minor_alleles();
}

// Advance `generations` generations, filling caller-provided buffers:
//   moments   : generations * n_bins * 3, laid out [generation][bin][mean, variance, skew]
//   skew_small: generations, the skew of alleles with |a| < skew_max_a
//   n_segregating / mean_phenotype: generations each (diagnostics; may be null)
// Returns the number of generations written.
long long aa_run_recorded(void* handle, long long generations, double* moments,
                          double* skew_small, int* n_segregating, double* mean_phenotype) {
    Context* ctx = static_cast<Context*>(handle);
    const int n_bins = static_cast<int>(ctx->bin_limits.size());
    for (long long t = 0; t < generations; ++t) {
        ctx->pop.next_generation();

        double* row = moments + t * n_bins * 3;
        for (int b = 0; b < n_bins * 3; ++b) row[b] = 0.0;
        double skew_acc = 0.0;

        for (std::size_t i = 0; i < ctx->pop.a.size(); ++i) {
            const double a = ctx->pop.a[i];
            const double x = ctx->pop.x[i];
            const double s = a * a;
            for (int b = 0; b < n_bins; ++b) {
                if (s < ctx->bin_limits[b]) {
                    row[b * 3 + 0] += a * 2.0 * x;
                    row[b * 3 + 1] += 2.0 * s * x * (1.0 - x);
                    row[b * 3 + 2] += a * a * a * x * (1.0 - x) * (1.0 - 2.0 * x);
                    break;
                }
            }
            if (std::fabs(a) < ctx->skew_max_a) {
                skew_acc += a * a * a * x * (1.0 - x) * (1.0 - 2.0 * x);
            }
        }
        if (skew_small)     skew_small[t] = skew_acc;
        if (n_segregating)  n_segregating[t] = static_cast<int>(ctx->pop.a.size());
        if (mean_phenotype) mean_phenotype[t] = ctx->pop.mean_phenotype();
        ctx->pop.new_fixations.clear();
    }
    return generations;
}

int    aa_n_fixations(void* handle) {
    return static_cast<int>(static_cast<Context*>(handle)->pop.fixations.size());
}

void   aa_get_fixations(void* handle, double* out) {
    Context* ctx = static_cast<Context*>(handle);
    std::copy(ctx->pop.fixations.begin(), ctx->pop.fixations.end(), out);
}

int    aa_n_segregating(void* handle) {
    return static_cast<int>(static_cast<Context*>(handle)->pop.a.size());
}

// Copy the current allele state out (a and x), for tests and for the optional full dump.
void   aa_get_state(void* handle, double* a_out, double* x_out) {
    Context* ctx = static_cast<Context*>(handle);
    std::copy(ctx->pop.a.begin(), ctx->pop.a.end(), a_out);
    std::copy(ctx->pop.x.begin(), ctx->pop.x.end(), x_out);
}

double aa_fixed_background(void* handle) {
    return static_cast<Context*>(handle)->pop.fixed_background;
}

double aa_mean_phenotype(void* handle) {
    return static_cast<Context*>(handle)->pop.mean_phenotype();
}

// exposed so the Python side can check the two implementations agree exactly
double aa_sojourn_time(int N, double a2, double x) { return sojourn_time(N, a2, x); }

}  // extern "C"
