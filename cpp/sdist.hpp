// Effect-size distributions and the mutation-selection-drift steady state.
//
// Two pieces:
//
//   Sdist   -- the distribution of scaled selection coefficients S (pdf / cdf /
//              ppf / rvs). Mirrors the distributions the pipeline actually uses:
//              scipy.stats.expon, scipy.stats.uniform, and
//              mixture_distribution.MixtureDistribution over those.
//
//   Steady  -- the standing variation present at the moment of the shift. The
//              Python sampler (generate_segregating_mutations.generate_alleles)
//              inverse-transform samples S from the sojourn-weighted effect-size
//              distribution and then x from the folded sojourn density, doing a
//              root-find over nested quadratures per allele -- which is what
//              dominates the runtime of the whole simulation. Here the two
//              inverse cdfs are tabulated once per parameter set and shared by
//              every replicate.
//
// The folded sojourn time (analytic_functions.folded_sojourn_time) is
//
//     T(S, x) = 2 exp(-S x (1-x)) / (x (1-x))            for x  > 1/(2N)
//     T(S, x) = 2N x * 2 exp(-S x (1-x)) / (x (1-x))     for x <= 1/(2N)
//              = 4N exp(-S x (1-x)) / (1-x)
//
// on x in [0, 1/2]. The x <= 1/(2N) branch is the boundary correction that keeps
// the density integrable as x -> 0; note it removes the 1/x singularity there,
// leaving a bounded integrand on [0, 1/(2N)] and a log-singular one above it.
// Both pieces are integrated in the variable that makes them smooth: x itself
// below 1/(2N), ln(x) above it.

#ifndef SIM_SDIST_HPP
#define SIM_SDIST_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace sim {

// ── effect-size distribution ────────────────────────────────────────────────
// A finite mixture of exponential and uniform components, which covers every
// sdist in the Snakefile: "expon" is a single expon component and "mixexpunif"
// is an even mixture of an expon and a uniform.

enum class CompKind { Expon = 0, Uniform = 1 };

struct Component {
    CompKind kind;
    double loc;
    double scale;

    double pdf(double s) const {
        if (kind == CompKind::Expon) {
            const double z = (s - loc) / scale;
            return z < 0.0 ? 0.0 : std::exp(-z) / scale;
        }
        return (s < loc || s > loc + scale) ? 0.0 : 1.0 / scale;
    }

    double cdf(double s) const {
        if (kind == CompKind::Expon) {
            const double z = (s - loc) / scale;
            return z <= 0.0 ? 0.0 : -std::expm1(-z);
        }
        if (s <= loc) return 0.0;
        if (s >= loc + scale) return 1.0;
        return (s - loc) / scale;
    }

    double ppf(double q) const {
        if (kind == CompKind::Expon) return loc - scale * std::log1p(-q);
        return loc + scale * q;
    }
};

class Sdist {
public:
    Sdist(std::vector<Component> components, std::vector<double> weights)
        : comps_(std::move(components)), w_(std::move(weights)) {
        if (comps_.empty() || comps_.size() != w_.size()) {
            throw std::invalid_argument("sdist: components and weights must match and be non-empty");
        }
        double total = 0.0;
        for (double w : w_) {
            if (w < 0.0) throw std::invalid_argument("sdist: negative weight");
            total += w;
        }
        if (!(total > 0.0)) throw std::invalid_argument("sdist: weights must sum to > 0");
        for (double& w : w_) w /= total;
    }

    double pdf(double s) const {
        double v = 0.0;
        for (std::size_t i = 0; i < comps_.size(); ++i) v += w_[i] * comps_[i].pdf(s);
        return v;
    }

    double cdf(double s) const {
        double v = 0.0;
        for (std::size_t i = 0; i < comps_.size(); ++i) v += w_[i] * comps_[i].cdf(s);
        return v;
    }

    // Quantile. Analytic for a single component; bisection on the (monotone)
    // mixture cdf otherwise, matching MixtureDistribution.ppf's numeric inversion.
    double ppf(double q) const {
        if (q <= 0.0) return support_lo();
        if (comps_.size() == 1) return comps_[0].ppf(q);
        if (q >= 1.0) return support_hi_finite(q);
        double lo = support_lo();
        double hi = lo + 1.0;
        while (cdf(hi) < q) {
            hi = lo + (hi - lo) * 2.0;
            if (hi > 1e12) break;
        }
        for (int i = 0; i < 200; ++i) {
            const double mid = 0.5 * (lo + hi);
            if (cdf(mid) < q) lo = mid; else hi = mid;
        }
        return 0.5 * (lo + hi);
    }

    // One variate, drawn exactly (component choice, then the component's inverse cdf).
    template <class RngT>
    double rvs(RngT& rng) const {
        std::size_t k = 0;
        if (comps_.size() > 1) {
            const double u = rng.uniform();
            double acc = 0.0;
            for (; k + 1 < comps_.size(); ++k) {
                acc += w_[k];
                if (u < acc) break;
            }
        }
        return comps_[k].ppf(rng.uniform());
    }

    double support_lo() const {
        double lo = comps_[0].loc;
        for (const Component& c : comps_) lo = std::fmin(lo, c.loc);
        return lo;
    }

private:
    double support_hi_finite(double q) const {
        // q >= 1: fall back to a large-but-finite quantile so callers never see inf
        double hi = support_lo() + 1.0;
        while (cdf(hi) < 1.0 - 1e-15 && hi < 1e12) hi *= 2.0;
        (void)q;
        return hi;
    }

    std::vector<Component> comps_;
    std::vector<double> w_;
};

// ── steady-state (standing variation) sampler ───────────────────────────────

class Steady {
public:
    // n_s_nodes  : effect-size grid, laid out in quantile space so the node
    //              spacing already follows the effect-size density
    // n_x_nodes  : frequency grid per effect-size node (split between the two
    //              branches of the folded sojourn time)
    Steady(const Sdist& sdist, int N, int n_s_nodes = 513, int n_x_nodes = 1025)
        : N_(N), n_s_(n_s_nodes), q_hi_(0.9999) {
        if (n_s_ < 3) throw std::invalid_argument("steady: too few S nodes");
        build_x_grid(n_x_nodes);
        build_tables(sdist);
    }

    // Expected number of segregating sites per unit 2NU, i.e.
    // simulation_classes.Simulation.total_n() divided by N2U. The quantile
    // bounds match that function's sdist.ppf(1e-7) / sdist.ppf(0.99999).
    double sojourn_integral() const { return sojourn_integral_; }

    // Draw one standing-variation allele: its scaled effect S and its folded
    // frequency x in (0, 1/2].
    template <class RngT>
    void sample(RngT& rng, double* s_out, double* x_out) const {
        // 1. S from the sojourn-weighted effect-size distribution, by inverting
        //    the tabulated cdf (the Python get_S root-find).
        const double u = rng.uniform();
        const double pos = invert_s_cdf(u);              // continuous index into the S grid
        const int j = std::min(static_cast<int>(pos), n_s_ - 2);
        const double frac = pos - static_cast<double>(j);
        *s_out = s_nodes_[j] + frac * (s_nodes_[j + 1] - s_nodes_[j]);

        // 2. x from the folded sojourn density at that S (the Python
        //    get_frequency root-find), interpolating between the two bracketing
        //    effect-size nodes so the frequency tracks S continuously.
        const double v = rng.uniform();
        const double x_lo = invert_x_cdf(j, v);
        const double x_hi = invert_x_cdf(j + 1, v);
        double x = x_lo + frac * (x_hi - x_lo);
        if (x <= 0.0) x = 1.0 / (2.0 * static_cast<double>(N_));
        if (x > 0.5) x = 0.5;
        *x_out = x;
    }

private:
    // folded sojourn time, both branches
    double sojourn(double S, double x) const {
        const double bnd = 1.0 / (2.0 * static_cast<double>(N_));
        if (x <= bnd) {
            return 4.0 * static_cast<double>(N_) * std::exp(-S * x * (1.0 - x)) / (1.0 - x);
        }
        return 2.0 * std::exp(-S * x * (1.0 - x)) / (x * (1.0 - x));
    }

    // Frequency grid: linear on [0, 1/(2N)] where the corrected integrand is
    // bounded, geometric on [1/(2N), 1/2] where it behaves like 2/x. Both halves
    // carry an odd number of nodes so the cumulative integral can use Simpson.
    void build_x_grid(int n_x_nodes) {
        const double bnd = 1.0 / (2.0 * static_cast<double>(N_));
        int n_lo = n_x_nodes / 4;
        if (n_lo % 2 == 0) ++n_lo;                    // odd -> even number of panels
        int n_hi = n_x_nodes - n_lo;
        if (n_hi % 2 == 0) ++n_hi;
        x_grid_.clear();
        x_grid_.reserve(static_cast<std::size_t>(n_lo + n_hi));
        for (int i = 0; i < n_lo; ++i) {
            x_grid_.push_back(bnd * static_cast<double>(i) / static_cast<double>(n_lo - 1));
        }
        const double log_lo = std::log(bnd), log_hi = std::log(0.5);
        for (int i = 1; i < n_hi; ++i) {              // skip i=0: that is bnd again
            x_grid_.push_back(std::exp(log_lo + (log_hi - log_lo) * static_cast<double>(i) /
                                                    static_cast<double>(n_hi - 1)));
        }
        n_x_ = static_cast<int>(x_grid_.size());
        i_bnd_ = n_lo - 1;                            // index of x = 1/(2N)
    }

    // Cumulative integral of the sojourn density over x_grid_, for one S. The
    // panel integral uses the trapezoid rule in the variable each branch is
    // smooth in (x below the boundary, ln x above it), which is exact to
    // O(h^2) with h ~ 1e-2 in ln x -- well inside the Monte Carlo noise of the
    // replicates that consume it.
    void cumulative_x(double S, double* out) const {
        out[0] = 0.0;
        for (int i = 1; i <= i_bnd_; ++i) {
            const double x0 = x_grid_[i - 1], x1 = x_grid_[i];
            out[i] = out[i - 1] + 0.5 * (sojourn(S, x0) + sojourn(S, x1)) * (x1 - x0);
        }
        for (int i = i_bnd_ + 1; i < n_x_; ++i) {
            const double x0 = x_grid_[i - 1], x1 = x_grid_[i];
            // in ln x: integrand is sojourn(S,x) * x, which is bounded and smooth
            const double f0 = sojourn(S, x0) * x0;
            const double f1 = sojourn(S, x1) * x1;
            out[i] = out[i - 1] + 0.5 * (f0 + f1) * (std::log(x1) - std::log(x0));
        }
    }

    void build_tables(const Sdist& sdist) {
        // Effect-size nodes in quantile space. Sampling S with density
        // proportional to pdf(S) * T(S) is, after the substitution S = ppf(q),
        // sampling q with density proportional to T(S(q)) -- so the effect-size
        // pdf never has to be evaluated. The upper bound q_hi_ = 0.9999 matches
        // the prob_segregating integral in Simulation.initiate_mutations, which
        // is what normalizes the Python sampler's cdf.
        s_nodes_.resize(n_s_);
        for (int j = 0; j < n_s_; ++j) {
            const double q = q_hi_ * static_cast<double>(j) / static_cast<double>(n_s_ - 1);
            s_nodes_[j] = sdist.ppf(q);
        }

        // Per-node frequency cdfs, and the total sojourn time T(S) that weights
        // the effect-size cdf (it is the last entry of each cumulative integral).
        x_cdf_.assign(static_cast<std::size_t>(n_s_) * static_cast<std::size_t>(n_x_), 0.0);
        std::vector<double> T(n_s_);
        for (int j = 0; j < n_s_; ++j) {
            double* row = &x_cdf_[static_cast<std::size_t>(j) * static_cast<std::size_t>(n_x_)];
            cumulative_x(s_nodes_[j], row);
            T[j] = row[n_x_ - 1];
            const double inv = T[j] > 0.0 ? 1.0 / T[j] : 0.0;
            for (int i = 0; i < n_x_; ++i) row[i] *= inv;    // normalize to a cdf
        }

        // Effect-size cdf: cumulative trapezoid of T over the quantile grid.
        s_cdf_.assign(n_s_, 0.0);
        const double dq = q_hi_ / static_cast<double>(n_s_ - 1);
        for (int j = 1; j < n_s_; ++j) s_cdf_[j] = s_cdf_[j - 1] + 0.5 * (T[j - 1] + T[j]) * dq;
        const double total = s_cdf_[n_s_ - 1];
        if (total > 0.0) {
            for (int j = 0; j < n_s_; ++j) s_cdf_[j] /= total;
        }

        // Expected number of segregating sites per unit 2NU: the same integral
        // over Simulation.total_n()'s wider quantile range [1e-7, 0.99999].
        sojourn_integral_ = tail_corrected_integral(sdist, total);
    }

    // total_n()'s integral, computed on its own quantile bounds. The shared
    // [0, 0.9999] part is `total` (already integrated above); the piece from
    // 0.9999 to 0.99999 is added on a small dedicated grid, and the piece below
    // 1e-7 is subtracted.
    double tail_corrected_integral(const Sdist& sdist, double total) const {
        const int n_tail = 129;
        double upper = 0.0;
        {
            const double q0 = q_hi_, q1 = 0.99999;
            const double dq = (q1 - q0) / static_cast<double>(n_tail - 1);
            std::vector<double> tmp(n_x_);
            double prev = 0.0;
            for (int j = 0; j < n_tail; ++j) {
                cumulative_x(sdist.ppf(q0 + dq * static_cast<double>(j)), tmp.data());
                const double cur = tmp[n_x_ - 1];
                if (j > 0) upper += 0.5 * (prev + cur) * dq;
                prev = cur;
            }
        }
        double lower = 0.0;
        {
            const double q0 = 0.0, q1 = 1e-7;
            const double dq = (q1 - q0) / static_cast<double>(n_tail - 1);
            std::vector<double> tmp(n_x_);
            double prev = 0.0;
            for (int j = 0; j < n_tail; ++j) {
                cumulative_x(sdist.ppf(q0 + dq * static_cast<double>(j)), tmp.data());
                const double cur = tmp[n_x_ - 1];
                if (j > 0) lower += 0.5 * (prev + cur) * dq;
                prev = cur;
            }
        }
        return total + upper - lower;
    }

    // Continuous index into the S grid at which the tabulated cdf equals u.
    double invert_s_cdf(double u) const {
        const int j = static_cast<int>(std::lower_bound(s_cdf_.begin(), s_cdf_.end(), u) -
                                       s_cdf_.begin());
        if (j <= 0) return 0.0;
        if (j >= n_s_) return static_cast<double>(n_s_ - 1);
        const double c0 = s_cdf_[j - 1], c1 = s_cdf_[j];
        const double frac = (c1 > c0) ? (u - c0) / (c1 - c0) : 0.0;
        return static_cast<double>(j - 1) + frac;
    }

    // Frequency whose tabulated cdf at S node `j` equals v.
    double invert_x_cdf(int j, double v) const {
        const double* row = &x_cdf_[static_cast<std::size_t>(j) * static_cast<std::size_t>(n_x_)];
        const int i = static_cast<int>(std::lower_bound(row, row + n_x_, v) - row);
        if (i <= 0) return x_grid_[0];
        if (i >= n_x_) return x_grid_[n_x_ - 1];
        const double c0 = row[i - 1], c1 = row[i];
        const double frac = (c1 > c0) ? (v - c0) / (c1 - c0) : 0.0;
        // interpolate in the variable the branch is smooth in
        if (i <= i_bnd_) return x_grid_[i - 1] + frac * (x_grid_[i] - x_grid_[i - 1]);
        return std::exp(std::log(x_grid_[i - 1]) +
                        frac * (std::log(x_grid_[i]) - std::log(x_grid_[i - 1])));
    }

    int N_;
    int n_s_;
    int n_x_ = 0;
    int i_bnd_ = 0;                    // grid index of x = 1/(2N)
    double q_hi_;
    double sojourn_integral_ = 0.0;
    std::vector<double> s_nodes_;      // S at each quantile node
    std::vector<double> s_cdf_;        // sojourn-weighted cdf over those nodes
    std::vector<double> x_grid_;       // shared frequency grid
    std::vector<double> x_cdf_;        // n_s_ x n_x_ row-major frequency cdfs
};

}  // namespace sim

#endif  // SIM_SDIST_HPP
