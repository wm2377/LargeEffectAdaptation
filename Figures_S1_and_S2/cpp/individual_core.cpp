// C++ core for the INDIVIDUAL-BASED simulation.
//
// Same model as scripts/individual_classes.py (the engine behind Figure S1's individual
// series)
// So drift and selection emerge from the sampling of individuals 

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

std::string g_last_error;

// ---------------------------------------------------------------------------------
// effect-size distribution: a two-component mixture of S = a^2
// ---------------------------------------------------------------------------------
struct EffectSizeDistribution {
    int    small_kind = 0;      // 0 = exponential(loc, scale), 1 = uniform(loc, loc+scale)
    double small_loc = 0.0, small_scale = 1.0;
    int    large_kind = 0;
    double large_loc = 100.0, large_scale = 400.0;
    double weight = 0.0;        // probability a new mutation is large-effect

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
// sojourn-time density, for the initial frequencies (same form as the all-allele core)
// ---------------------------------------------------------------------------------
inline double sojourn_time(int N, double a2, double x) {
    const double v = 2.0 * std::exp(-(2.0 * a2 * x * (1.0 - x)) / 2.0) / (1.0 - x);
    const double entry = 1.0 / (2.0 * static_cast<double>(N));
    return v * ((x < entry) ? 2.0 * static_cast<double>(N) : 1.0 / x);
}

double draw_initial_frequency(int N, double a2, std::mt19937_64& rng, int n_grid = 1024) {
    const double entry = 1.0 / (2.0 * static_cast<double>(N));
    std::vector<double> grid;
    grid.reserve(16 + n_grid);
    for (int i = 0; i < 16; ++i) grid.push_back(entry * static_cast<double>(i) / 16.0);
    const double lo = std::log(entry), hi = std::log(0.5);
    for (int i = 0; i < n_grid; ++i)
        grid.push_back(std::exp(lo + (hi - lo) * static_cast<double>(i) / (n_grid - 1)));

    std::vector<double> cdf(grid.size(), 0.0);
    for (std::size_t i = 1; i < grid.size(); ++i)
        cdf[i] = cdf[i - 1] + 0.5 * (sojourn_time(N, a2, grid[i - 1]) + sojourn_time(N, a2, grid[i]))
                            * (grid[i] - grid[i - 1]);
    const double total = cdf.back();
    if (!(total > 0.0)) return entry;
    std::uniform_real_distribution<double> u(0.0, 1.0);
    const double p = u(rng) * total;
    const auto it = std::lower_bound(cdf.begin(), cdf.end(), p);
    const std::size_t i = static_cast<std::size_t>(it - cdf.begin());
    if (i == 0) return grid[0];
    const double w = (cdf[i] > cdf[i - 1]) ? (p - cdf[i - 1]) / (cdf[i] - cdf[i - 1]) : 0.0;
    return grid[i - 1] + w * (grid[i] - grid[i - 1]);
}

// ---------------------------------------------------------------------------------
// population
// ---------------------------------------------------------------------------------
using Genotype = std::vector<std::pair<int32_t, uint8_t>>;   // (mutation id, copies), sorted
using Gamete   = std::vector<int32_t>;                       // mutation ids, sorted

struct Population {
    int N = 1000;
    double Vs = 2000.0;
    double optimum = 0.0;
    double mutation_rate = 0.0;         // new mutations per GAMETE
    EffectSizeDistribution sdist;
    std::mt19937_64 rng;

    // Mutation ids are RECYCLED. `effect` is indexed by id and only grows to the high-water
    // mark of CONCURRENTLY live mutations (~10^4 here), not to the number of mutations ever
    // created (~10^3 per generation, so ~5x10^7 over a 10N burn-in). Without recycling,
    // every per-generation pass sized to effect.size() grew linearly with elapsed time and
    // made the whole simulation O(generations^2) -- which is exactly what it did.
    std::vector<double> effect;         // mutation id -> signed effect a
    std::vector<int32_t> counts;        // id -> copies in the population; reused each gen
    std::vector<int32_t> live_ids;      // ids currently segregating
    std::vector<int32_t> free_ids;      // ids of extinct/fixed mutations, ready for reuse
    std::vector<char> fixed_flag;       // id -> was it fixed this generation
    std::vector<int32_t> scratch_ids;   // reused buffer for the surviving-id list

    std::vector<Genotype> individuals;
    std::vector<double> phenotype, fitness;

    double fixed_background = 0.0;
    std::vector<double> fixation_effect;   // signed effect of each fixed mutation
    std::vector<int64_t> fixation_time;    // generation at which it fixed

    int32_t new_mutation(double a2) {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        const double sign = (u(rng) < 0.5) ? -1.0 : 1.0;
        const double a = std::sqrt(a2) * sign;

        int32_t id;
        if (!free_ids.empty()) {          // reuse a slot left by an extinct/fixed mutation
            id = free_ids.back();
            free_ids.pop_back();
            effect[id] = a;
        } else {
            effect.push_back(a);
            counts.push_back(0);
            fixed_flag.push_back(0);
            id = static_cast<int32_t>(effect.size() - 1);
        }
        counts[id] = 0;
        live_ids.push_back(id);
        return id;
    }

    void compute_phenotypes() {
        phenotype.assign(individuals.size(), fixed_background);
        for (std::size_t i = 0; i < individuals.size(); ++i) {
            double p = fixed_background;
            for (const auto& [id, copies] : individuals[i])
                p += effect[id] * static_cast<double>(copies);
            phenotype[i] = p;
        }
    }

    void compute_fitness() {
        compute_phenotypes();
        fitness.resize(phenotype.size());
        for (std::size_t i = 0; i < phenotype.size(); ++i) {
            const double d = phenotype[i] - optimum;
            fitness[i] = std::exp(-(d * d) / (2.0 * Vs));
        }
    }

    Gamete make_gamete(const Genotype& g) {
        Gamete out;
        out.reserve(g.size());
        std::uniform_real_distribution<double> u(0.0, 1.0);
        for (const auto& [id, copies] : g) {
            if (copies >= 2 || u(rng) < 0.5) out.push_back(id);   // free recombination
        }
        std::poisson_distribution<int> pois(mutation_rate);
        const int n_new = pois(rng);
        for (int k = 0; k < n_new; ++k) out.push_back(new_mutation(sdist.draw(rng)));
        std::sort(out.begin(), out.end());
        return out;
    }

    void reproduce() {
        compute_fitness();
        std::discrete_distribution<int> pick(fitness.begin(), fitness.end());

        std::vector<Genotype> offspring;
        offspring.reserve(individuals.size());
        Gamete a, b;
        for (int i = 0; i < N; ++i) {
            a = make_gamete(individuals[pick(rng)]);
            b = make_gamete(individuals[pick(rng)]);
            Genotype child;
            child.reserve(a.size() + b.size());
            // merge two sorted id lists, collapsing a shared id into a homozygote
            std::size_t ia = 0, ib = 0;
            while (ia < a.size() && ib < b.size()) {
                if (a[ia] == b[ib])      { child.emplace_back(a[ia], 2); ++ia; ++ib; }
                else if (a[ia] < b[ib])  { child.emplace_back(a[ia], 1); ++ia; }
                else                     { child.emplace_back(b[ib], 1); ++ib; }
            }
            for (; ia < a.size(); ++ia) child.emplace_back(a[ia], 1);
            for (; ib < b.size(); ++ib) child.emplace_back(b[ib], 1);
            offspring.push_back(std::move(child));
        }
        individuals.swap(offspring);
    }

    // Fold fixed mutations into the background, drop them from every genotype, and return
    // both fixed and extinct ids to the free list.
    //
    // Every pass here is over `live_ids` (~10^4) or over the genotypes, never over the full
    // id space -- so the per-generation cost is flat in elapsed time rather than growing
    // with the number of mutations ever created.
    void handle_fixations(int64_t t) {
        for (int32_t id : live_ids) counts[id] = 0;
        for (const auto& g : individuals)
            for (const auto& [id, copies] : g) counts[id] += copies;

        // fixed ids are kept in their own list: they are interleaved with the extinct ones
        // in free_ids, so there is no contiguous range to clear flags from afterwards
        std::vector<int32_t> fixed_now;
        scratch_ids.clear();
        for (int32_t id : live_ids) {
            if (counts[id] == 2 * N) {
                fixed_background += 2.0 * effect[id];
                fixation_effect.push_back(effect[id]);
                fixation_time.push_back(t);
                fixed_flag[id] = 1;
                fixed_now.push_back(id);
            } else if (counts[id] == 0) {
                free_ids.push_back(id);            // extinct: no genotype references it
            } else {
                scratch_ids.push_back(id);         // still segregating
            }
        }
        live_ids.swap(scratch_ids);

        if (!fixed_now.empty()) {
            for (auto& g : individuals) {
                g.erase(std::remove_if(g.begin(), g.end(),
                                       [&](const std::pair<int32_t, uint8_t>& e) {
                                           return fixed_flag[e.first];
                                       }), g.end());
            }
            // clear exactly the flags we set, then release those slots for reuse
            for (int32_t id : fixed_now) {
                fixed_flag[id] = 0;
                free_ids.push_back(id);
            }
        }
    }

    void next_generation(int64_t t) {
        reproduce();
        handle_fixations(t);
        compute_phenotypes();
    }

    // mean, variance and skewness of the phenotype distribution
    void moments(double* mean_out, double* var_out, double* skew_out) const {
        const double n = static_cast<double>(phenotype.size());
        double mean = 0.0;
        for (double p : phenotype) mean += p;
        mean /= n;
        double m2 = 0.0, m3 = 0.0;
        for (double p : phenotype) {
            const double d = p - mean;
            m2 += d * d;
            m3 += d * d * d;
        }
        m2 /= n; m3 /= n;
        *mean_out = mean;
        *var_out = m2;
        // Fisher-Pearson skewness, matching scipy.stats.skew's default (bias=True)
        *skew_out = (m2 > 0.0) ? m3 / std::pow(m2, 1.5) : 0.0;
    }

    // O(1): live_ids is maintained by handle_fixations, so this no longer walks every
    // genotype (nor allocates an array sized to the whole id space) once per generation.
    int n_segregating() const { return static_cast<int>(live_ids.size()); }
};

struct Context { Population pop; };

}  // namespace

// ---------------------------------------------------------------------------------
// C interface
// ---------------------------------------------------------------------------------
extern "C" {

const char* ind_last_error(void) { return g_last_error.c_str(); }

void* ind_create(int N, double mutation_rate, double optimum, double weight,
                 int large_kind, double large_loc, double large_scale,
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
        ctx->pop.individuals.assign(N, Genotype{});
        return ctx;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return nullptr;
    }
}

void ind_destroy(void* handle) { delete static_cast<Context*>(handle); }

int ind_poisson(void* handle, double mean) {
    Context* ctx = static_cast<Context*>(handle);
    std::poisson_distribution<int> pois(mean);
    return pois(ctx->pop.rng);
}

// Seed `n_mutations` segregating mutations, each at a frequency drawn from the
// steady-state sojourn density and then scattered over individuals as Binomial(2, freq).
void ind_initialize(void* handle, int n_mutations) {
    Context* ctx = static_cast<Context*>(handle);
    Population& pop = ctx->pop;
    for (int k = 0; k < n_mutations; ++k) {
        const double a2 = pop.sdist.draw(pop.rng);
        const int32_t id = pop.new_mutation(a2);
        const double freq = draw_initial_frequency(pop.N, a2, pop.rng);
        std::binomial_distribution<int> binom(2, freq);
        for (int i = 0; i < pop.N; ++i) {
            const int copies = binom(pop.rng);
            if (copies > 0)
                pop.individuals[i].emplace_back(id, static_cast<uint8_t>(copies));
        }
    }
    for (auto& g : pop.individuals) std::sort(g.begin(), g.end());
    pop.compute_phenotypes();
}

void ind_burn(void* handle, long long generations) {
    Context* ctx = static_cast<Context*>(handle);
    for (long long t = -generations; t <= 0; ++t) ctx->pop.next_generation(t);
}

void ind_shift_optimum(void* handle, double shift) {
    static_cast<Context*>(handle)->pop.optimum += shift;
}

// Advance `generations`, filling caller-provided per-generation buffers (any may be null).
long long ind_run_recorded(void* handle, long long generations, double* mean,
                           double* variance, double* skew, int* n_segregating,
                           double* fixed_background) {
    Context* ctx = static_cast<Context*>(handle);
    for (long long t = 1; t <= generations; ++t) {
        ctx->pop.next_generation(t);
        double m, v, s;
        ctx->pop.moments(&m, &v, &s);
        const std::size_t i = static_cast<std::size_t>(t - 1);
        if (mean)             mean[i] = m;
        if (variance)         variance[i] = v;
        if (skew)             skew[i] = s;
        if (n_segregating)    n_segregating[i] = ctx->pop.n_segregating();
        if (fixed_background) fixed_background[i] = ctx->pop.fixed_background;
    }
    return generations;
}

int ind_n_fixations(void* handle) {
    return static_cast<int>(static_cast<Context*>(handle)->pop.fixation_effect.size());
}

// Copy out the signed effect size and fixation generation of every fixed mutation.
void ind_get_fixations(void* handle, double* effects, long long* times) {
    Context* ctx = static_cast<Context*>(handle);
    std::copy(ctx->pop.fixation_effect.begin(), ctx->pop.fixation_effect.end(), effects);
    std::copy(ctx->pop.fixation_time.begin(), ctx->pop.fixation_time.end(), times);
}

int    ind_n_segregating(void* handle) { return static_cast<Context*>(handle)->pop.n_segregating(); }
double ind_fixed_background(void* handle) { return static_cast<Context*>(handle)->pop.fixed_background; }

void ind_moments(void* handle, double* mean, double* variance, double* skew) {
    static_cast<Context*>(handle)->pop.moments(mean, variance, skew);
}

}  // extern "C"
