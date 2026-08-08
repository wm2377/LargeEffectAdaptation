// Random number generation for the polygenic-adaptation simulation core.
//
// xoshiro256** (Blackman & Vigna) seeded through splitmix64, plus the two
// discrete samplers the recursion needs every generation:
//
//   * binomial(n, p) -- one Wright-Fisher drift draw per segregating allele.
//     Inversion (BINV) for small n*p, transformed rejection (BTRS, Hoermann
//     1993) otherwise. This is the same pair of algorithms NumPy/TensorFlow use.
//   * poisson(lambda) -- the per-generation count of new mutations. Knuth's
//     product-of-uniforms for small lambda, PTRS (Hoermann 1993) otherwise.
//
// These do NOT reproduce NumPy's exact streams (different algorithms and
// seeding), so a C++ replicate is not bit-identical to the Python one; it is
// drawn from the same distributions. validate_sim_core.py checks that
// statistically.

#ifndef SIM_RNG_HPP
#define SIM_RNG_HPP

#include <cmath>
#include <cstdint>

namespace sim {

class Rng {
public:
    explicit Rng(std::uint64_t seed) {
        // splitmix64 expands the 64-bit seed into xoshiro's 256-bit state
        std::uint64_t z = seed;
        for (int i = 0; i < 4; ++i) {
            z += 0x9e3779b97f4a7c15ULL;
            std::uint64_t x = z;
            x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
            x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
            s_[i] = x ^ (x >> 31);
        }
        // discard a few outputs so low-entropy seeds (0, 1, 2, ...) decorrelate
        for (int i = 0; i < 16; ++i) (void)next_u64();
    }

    std::uint64_t next_u64() {
        const std::uint64_t result = rotl(s_[1] * 5, 7) * 9;
        const std::uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return result;
    }

    // uniform on the OPEN interval (0, 1): never returns 0 or 1, so callers may
    // take log() of a variate without guarding against -inf.
    double uniform() { return (static_cast<double>(next_u64() >> 11) + 0.5) * (1.0 / 9007199254740992.0); }

    // random sign, +1 or -1 with probability 1/2 each (one bit, no float compare)
    double sign() { return (next_u64() >> 63) ? 1.0 : -1.0; }

    std::int64_t binomial(std::int64_t n, double p) {
        if (n <= 0 || p <= 0.0) return 0;
        if (p >= 1.0) return n;
        // the samplers below assume p <= 1/2; reflect otherwise
        if (p > 0.5) return n - binomial(n, 1.0 - p);
        if (static_cast<double>(n) * p < 30.0) return binomial_inversion(n, p);
        return binomial_btrs(n, p);
    }

    std::int64_t poisson(double lam) {
        if (!(lam > 0.0)) return 0;
        if (lam < 10.0) return poisson_knuth(lam);
        return poisson_ptrs(lam);
    }

private:
    static std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

    // Sum of Bernoulli trials by inverse-cdf search, walking up the pmf from 0.
    // Only used when n*p < 30, where the expected number of steps is small and
    // (1-p)^n cannot underflow.
    std::int64_t binomial_inversion(std::int64_t n, double p) {
        const double q = 1.0 - p;
        const double qn = std::exp(static_cast<double>(n) * std::log1p(-p));
        const double np = static_cast<double>(n) * p;
        const double bound = std::fmin(static_cast<double>(n), np + 10.0 * std::sqrt(np * q + 1.0));
        while (true) {
            double u = uniform();
            double px = qn;
            std::int64_t x = 0;
            while (u > px) {
                u -= px;
                ++x;
                if (x > bound) break;             // tail excursion: restart the draw
                px *= (static_cast<double>(n - x + 1) * p) / (static_cast<double>(x) * q);
            }
            if (x <= bound) return x;
        }
    }

    // ln(k!) - Stirling's series, i.e. the correction term
    // ln(k!) - [k ln k - k + 0.5 ln(2 pi k)]. Table for small k, asymptotic
    // expansion beyond.
    static double stirling_tail(std::int64_t k) {
        static const double kTail[10] = {
            0.0810614667953272, 0.0413406959554093, 0.0276779256849983,
            0.0207906721037650, 0.0166446911898211, 0.0138761288230707,
            0.0118967099458917, 0.0104112652619720, 0.0092554621827127,
            0.0083305634333594};
        if (k < 10) return kTail[k];
        const double kk = static_cast<double>(k) + 1.0;
        const double inv = 1.0 / (kk * kk);
        return (1.0 / 12.0 - (1.0 / 360.0 - 1.0 / 1260.0 * inv) * inv) / kk;
    }

    // BTRS: transformed rejection with squeeze (Hoermann 1993), for n*p >= 30.
    std::int64_t binomial_btrs(std::int64_t n, double p) {
        const double nd = static_cast<double>(n);
        const double spq = std::sqrt(nd * p * (1.0 - p));
        const double b = 1.15 + 2.53 * spq;
        const double a = -0.0873 + 0.0248 * b + 0.01 * p;
        const double c = nd * p + 0.5;
        const double v_r = 0.92 - 4.2 / b;
        const double r = p / (1.0 - p);
        const double alpha = (2.83 + 5.1 / b) * spq;
        const double m = std::floor((nd + 1.0) * p);

        while (true) {
            const double u = uniform() - 0.5;
            double v = uniform();
            const double us = 0.5 - std::fabs(u);
            const double kd = std::floor((2.0 * a / us + b) * u + c);
            if (kd < 0.0 || kd > nd) continue;
            // squeeze: accept the bulk without evaluating the pmf
            if (us >= 0.07 && v <= v_r) return static_cast<std::int64_t>(kd);
            v = std::log(v * alpha / (a / (us * us) + b));
            const double ub =
                (m + 0.5) * std::log((m + 1.0) / (r * (nd - m + 1.0))) +
                (nd + 1.0) * std::log((nd - m + 1.0) / (nd - kd + 1.0)) +
                (kd + 0.5) * std::log(r * (nd - kd + 1.0) / (kd + 1.0)) +
                stirling_tail(static_cast<std::int64_t>(m)) +
                stirling_tail(static_cast<std::int64_t>(nd - m)) -
                stirling_tail(static_cast<std::int64_t>(kd)) -
                stirling_tail(static_cast<std::int64_t>(nd - kd));
            if (v <= ub) return static_cast<std::int64_t>(kd);
        }
    }

    // Knuth: multiply uniforms until the product drops below exp(-lambda).
    std::int64_t poisson_knuth(double lam) {
        const double enlam = std::exp(-lam);
        std::int64_t x = 0;
        double prod = 1.0;
        while (true) {
            prod *= uniform();
            if (prod <= enlam) return x;
            ++x;
        }
    }

    // PTRS: transformed rejection with squeeze (Hoermann 1993), for lambda >= 10.
    std::int64_t poisson_ptrs(double lam) {
        const double slam = std::sqrt(lam);
        const double loglam = std::log(lam);
        const double b = 0.931 + 2.53 * slam;
        const double a = -0.059 + 0.02483 * b;
        const double inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
        const double v_r = 0.9277 - 3.6224 / (b - 2.0);
        while (true) {
            const double u = uniform() - 0.5;
            double v = uniform();
            const double us = 0.5 - std::fabs(u);
            const double kd = std::floor((2.0 * a / us + b) * u + lam + 0.43);
            if (us >= 0.07 && v <= v_r) return static_cast<std::int64_t>(kd);
            if (kd < 0.0 || (us < 0.013 && v > us)) continue;
            if (std::log(v * inv_alpha / (a / (us * us) + b)) <=
                -lam + kd * loglam - std::lgamma(kd + 1.0)) {
                return static_cast<std::int64_t>(kd);
            }
        }
    }

    std::uint64_t s_[4];
};

}  // namespace sim

#endif  // SIM_RNG_HPP
