

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

extern "C" {
void*  ind_create(int N, double mutation_rate, double optimum, double weight,
                  int large_kind, double large_loc, double large_scale,
                  unsigned long long seed);
void   ind_destroy(void*);
int    ind_poisson(void*, double mean);
void   ind_initialize(void*, int n_mutations);
void   ind_burn(void*, long long generations);
int    ind_n_segregating(void*);
int    ind_n_fixations(void*);
}

static double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr,
                     "usage: %s <N> <mutation_rate> <weight> <n_seg> <generations> [shift]\n",
                     argv[0]);
        return 2;
    }
    const int    N      = std::atoi(argv[1]);
    const double mu     = std::atof(argv[2]);
    const double weight = std::atof(argv[3]);
    const double n_seg  = std::atof(argv[4]);
    const long long gens = std::atoll(argv[5]);
    const double shift  = (argc > 6) ? std::atof(argv[6]) : 80.0;

    std::printf("N=%d  mutation_rate=%.6g (per gamete)  weight=%.6g  E[n_seg]=%.0f\n",
                N, mu, weight, n_seg);

    auto t0 = std::chrono::steady_clock::now();
    void* sim = ind_create(N, mu, 0.0, weight, 0, 100.0, 400.0, 12345ULL);
    if (!sim) { std::fprintf(stderr, "ind_create failed\n"); return 1; }
    const int drawn = ind_poisson(sim, n_seg);
    ind_initialize(sim, drawn);
    std::printf("init: %d mutations seeded in %.2f s, %d segregating\n",
                drawn, seconds_since(t0), ind_n_segregating(sim));

    double last_per_gen = 0.0;
    long long done = 0;
    for (long long chunk : {2LL, 4LL, 8LL, 16LL, 32LL, 64LL, 128LL, 256LL, 512LL, 1024LL, 2048LL}) {
        if (done + chunk > gens) break;
        t0 = std::chrono::steady_clock::now();
        ind_burn(sim, chunk);
        const double dt = seconds_since(t0);
        done += chunk;
        last_per_gen = dt / static_cast<double>(chunk);
        std::printf("  +%2lld generations: %7.2f s -> %8.3f s/gen   (%d segregating, %d fixed)\n",
                    chunk, dt, last_per_gen, ind_n_segregating(sim), ind_n_fixations(sim));
    }

    std::printf("\nextrapolating at %.3f s/gen (LOWER BOUND -- the site count is still rising):\n",
                last_per_gen);
    const struct { const char* label; long long g; } phases[] = {
        {"10N burn-in", 10LL * N}, {"4N recorded", 4LL * N}, {"14N total", 14LL * N}};
    for (const auto& p : phases) {
        const double hours = last_per_gen * static_cast<double>(p.g) / 3600.0;
        std::printf("  %-12s %7lld gens -> %10.1f CPU-hours (%.1f days)\n",
                    p.label, p.g, hours, hours / 24.0);
    }
    ind_destroy(sim);
    return 0;
}
