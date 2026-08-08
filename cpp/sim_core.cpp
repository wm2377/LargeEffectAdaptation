// C++ core of the polygenic-adaptation simulation, a port of
// simulation_classes.Simulation with the same model, outputs and recording
// rules. The Python class stays the reference implementation; this is what the
// production sweeps run through (see simulation_cpp.py, which loads this via
// ctypes, and validate_sim_core.py, which checks the two agree).
//
// What the port changes, and why it is faster
// -------------------------------------------
//   * The standing variation at the shift is drawn from tabulated inverse cdfs
//     built once per parameter set (cpp/sdist.hpp), instead of a scipy root-find
//     over nested quadratures per allele. That alone was ~85% of the runtime of
//     a 2NU = 10 replicate.
//   * The generation loop is a plain loop over contiguous arrays: no temporary
//     allocations, no scipy `rvs` call per generation, no Python-level bookkeeping.
//
// What it does NOT change: the dynamics. Deterministic selection, the binomial
// drift draw, the Poisson mutation influx, the pruning rule, the convergence
// criterion, the recording schedule and the full_output snapshots all follow
// simulation_classes.py line for line. Because the RNG algorithms differ from
// NumPy's, a replicate is statistically equivalent rather than bit-identical.
//
// The exported interface is plain C (see the extern "C" block at the bottom) so
// the Python side needs only ctypes -- no pybind11, no build-time dependency on
// the Python or NumPy headers.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <new>
#include <string>
#include <vector>

#include "rng.hpp"
#include "sdist.hpp"

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// full_output records the frequency trajectory of any allele that fixes or that
// ever reaches this frequency (Simulation.TRAJECTORY_RECORD_THRESHOLD).
constexpr double kTrajectoryThreshold = 0.01;

// snapshot slots, in the order simulation_classes takes them
enum SnapKey { kSnapInitial = 0, kSnapGen20 = 1, kSnapQuasiStatic = 2, kSnapGen300 = 3, kSnapFinal = 4 };
constexpr int kNumSnapshots = 5;

// fate codes for recorded trajectories (decoded back to strings in Python)
enum Fate { kFateFixed = 0, kFateExtinct = 1, kFateSegregating = 2 };

std::string g_last_error;

struct Params {
    int N = 5000;
    double N2U = 0.0;
    double sigma2 = 0.0;
    double shift = 0.0;
    int tracking_time = 50;
    double stop_time = 1e4;
    bool record_moments = false;
    bool full_output = false;
};

// A fixed allele: the Mutation record in stats['fixations'].
struct Fixation {
    double a;
    double t_initial;
    double x0;
};

struct TrajectoryRecord {
    std::int64_t id;
    double a;
    double t_initial;
    double x0;
    int fate;
    double fixation_time;
    std::vector<double> t;
    std::vector<double> x;
};

struct Snapshot {
    bool present = false;
    double time = 0.0;
    double D = 0.0;
    std::vector<double> seg_id, seg_a, seg_x, seg_t_initial, seg_x0;
    std::vector<double> fix_a, fix_t_initial, fix_x0;
};

struct Result {
    std::vector<Fixation> fixations;
    double convergence_time = kNaN;
    double convergence_D_window = kNaN;
    std::vector<double> d_trajectory;
    std::vector<double> second_moment;
    std::vector<double> third_moment;
    std::vector<TrajectoryRecord> trajectories;
    Snapshot snapshots[kNumSnapshots];
};

// Everything shared by every replicate of one parameter set: the parameters
// themselves and the (expensive to build, read-only) steady-state tables.
struct Context {
    Params p;
    sim::Sdist sdist;
    sim::Steady steady;

    Context(const Params& params, sim::Sdist s, sim::Steady st)
        : p(params), sdist(std::move(s)), steady(std::move(st)) {}
};

// ── one replicate ───────────────────────────────────────────────────────────

class Replicate {
public:
    Replicate(const Context& ctx, std::uint64_t seed)
        : ctx_(ctx), p_(ctx.p), rng_(seed), twoN_(2.0 * static_cast<double>(ctx.p.N)) {}

    void run(Result* out) {
        out_ = out;
        initiate_mutations();
        recursion();
    }

private:
    // ── state ───────────────────────────────────────────────────────────────
    // one entry per segregating allele, kept in parallel vectors
    std::vector<double> x_, a_, t_initial_, x_initial_;
    std::vector<std::int64_t> ids_;
    // full_output only: live frequency trajectory per allele (interleaved t, x)
    // and whether it has passed the recording threshold. Held parallel to the
    // state vectors, so pruning compacts them together and no id lookup is needed.
    std::vector<std::vector<double>> traj_t_, traj_x_;
    std::vector<char> reached_;
    std::int64_t next_id_ = 0;

    const Context& ctx_;
    const Params& p_;
    sim::Rng rng_;
    const double twoN_;
    Result* out_ = nullptr;

    void push_allele(double x, double a, double t_initial, double x_initial) {
        x_.push_back(x);
        a_.push_back(a);
        t_initial_.push_back(t_initial);
        x_initial_.push_back(x_initial);
        ids_.push_back(next_id_++);
        if (p_.full_output) {
            traj_t_.emplace_back();
            traj_x_.emplace_back();
            reached_.push_back(0);
        }
    }

    // Seed the standing variation present the moment the optimum shifts. The
    // count is Poisson about the expectation for the TOTAL mutational input
    // (2NU covers both sign classes); the aligned/opposing split is the random
    // sign drawn per allele, exactly as in Simulation.initiate_mutations.
    void initiate_mutations() {
        const double n_expected = ctx_.steady.sojourn_integral() * p_.N2U;
        const std::int64_t n = rng_.poisson(n_expected);
        x_.reserve(static_cast<std::size_t>(n));
        a_.reserve(static_cast<std::size_t>(n));
        t_initial_.reserve(static_cast<std::size_t>(n));
        x_initial_.reserve(static_cast<std::size_t>(n));
        ids_.reserve(static_cast<std::size_t>(n));
        for (std::int64_t i = 0; i < n; ++i) {
            double S = 0.0, x = 0.0;
            ctx_.steady.sample(rng_, &S, &x);
            push_allele(x, std::sqrt(S) * rng_.sign(), 0.0, x);
        }
    }

    // A Poisson(2NU) number of new alleles enter as a single copy each; returns
    // their total contribution to the mean phenotype, sum(2 a / 2N).
    double add_new_mutations(double t) {
        const std::int64_t k = rng_.poisson(p_.N2U);
        const double x_new = 1.0 / twoN_;
        double change_in_d = 0.0;
        for (std::int64_t i = 0; i < k; ++i) {
            const double a = std::sqrt(ctx_.sdist.rvs(rng_)) * rng_.sign();
            change_in_d += 2.0 * a * x_new;
            push_allele(x_new, a, t, x_new);
        }
        return change_in_d;
    }

    // Advance every allele one generation: deterministic selection, then a
    // Wright-Fisher binomial draw, then record fixations and prune.
    double update_mutation_frequencies(double d, double t) {
        const std::size_t n = x_.size();
        const std::int64_t twoN_int = static_cast<std::int64_t>(std::llround(twoN_));
        double change_in_d = 0.0;

        for (std::size_t i = 0; i < n; ++i) {
            const double xi = x_[i], ai = a_[i];
            const double det_dx = ai / twoN_ * (d - ai * (0.5 - xi)) * xi * (1.0 - xi);
            double exp_x = xi + det_dx;
            if (exp_x < 0.0) exp_x = 0.0;
            if (exp_x > 1.0) exp_x = 1.0;
            const double new_x = static_cast<double>(rng_.binomial(twoN_int, exp_x)) / twoN_;
            change_in_d += 2.0 * ai * (new_x - xi);
            x_[i] = new_x;
        }

        // fixations, in index order (as in the Python loop over np.nonzero)
        for (std::size_t i = 0; i < n; ++i) {
            if (x_[i] >= 1.0) out_->fixations.push_back(Fixation{a_[i], t_initial_[i], x_initial_[i]});
        }

        if (p_.full_output) {
            for (std::size_t i = 0; i < n; ++i) {
                traj_t_[i].push_back(t);
                traj_x_[i].push_back(x_[i]);
                if (x_[i] >= kTrajectoryThreshold) reached_[i] = 1;
            }
            // fixed alleles are always recorded ...
            for (std::size_t i = 0; i < n; ++i) {
                if (x_[i] >= 1.0) record_trajectory(i, kFateFixed, t);
            }
            // ... lost ones only if they ever reached the threshold
            for (std::size_t i = 0; i < n; ++i) {
                if (x_[i] <= 0.0 && reached_[i]) record_trajectory(i, kFateExtinct, kNaN);
            }
        }

        prune();
        return change_in_d;
    }

    // Keep only still-segregating alleles (drop fixed x >= 1 and lost x <= 0),
    // preserving order.
    void prune() {
        std::size_t w = 0;
        for (std::size_t i = 0; i < x_.size(); ++i) {
            if (x_[i] > 0.0 && x_[i] < 1.0) {
                if (w != i) {
                    x_[w] = x_[i];
                    a_[w] = a_[i];
                    t_initial_[w] = t_initial_[i];
                    x_initial_[w] = x_initial_[i];
                    ids_[w] = ids_[i];
                    if (p_.full_output) {
                        traj_t_[w] = std::move(traj_t_[i]);
                        traj_x_[w] = std::move(traj_x_[i]);
                        reached_[w] = reached_[i];
                    }
                }
                ++w;
            }
        }
        x_.resize(w);
        a_.resize(w);
        t_initial_.resize(w);
        x_initial_.resize(w);
        ids_.resize(w);
        if (p_.full_output) {
            traj_t_.resize(w);
            traj_x_.resize(w);
            reached_.resize(w);
        }
    }

    // Move allele i's completed trajectory into the results, tagged with its fate.
    void record_trajectory(std::size_t i, int fate, double fixation_time) {
        TrajectoryRecord rec;
        rec.id = ids_[i];
        rec.a = a_[i];
        rec.t_initial = t_initial_[i];
        rec.x0 = x_initial_[i];
        rec.fate = fate;
        rec.fixation_time = fixation_time;
        rec.t = std::move(traj_t_[i]);
        rec.x = std::move(traj_x_[i]);
        traj_t_[i].clear();
        traj_x_[i].clear();
        reached_[i] = 0;
        out_->trajectories.push_back(std::move(rec));
    }

    void take_snapshot(int key, double t, double d) {
        Snapshot& s = out_->snapshots[key];
        s.present = true;
        s.time = t;
        s.D = d;
        const std::size_t n = x_.size();
        s.seg_id.resize(n);
        s.seg_a.resize(n);
        s.seg_x.resize(n);
        s.seg_t_initial.resize(n);
        s.seg_x0.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            s.seg_id[i] = static_cast<double>(ids_[i]);
            s.seg_a[i] = a_[i];
            s.seg_x[i] = x_[i];
            s.seg_t_initial[i] = t_initial_[i];
            s.seg_x0[i] = x_initial_[i];
        }
        const std::size_t nf = out_->fixations.size();
        s.fix_a.resize(nf);
        s.fix_t_initial.resize(nf);
        s.fix_x0.resize(nf);
        for (std::size_t i = 0; i < nf; ++i) {
            s.fix_a[i] = out_->fixations[i].a;
            s.fix_t_initial[i] = out_->fixations[i].t_initial;
            s.fix_x0[i] = out_->fixations[i].x0;
        }
    }

    // total phenotypic variance: large-effect alleles plus the background sigma2
    double second_moment() const {
        double v = 0.0;
        for (std::size_t i = 0; i < x_.size(); ++i) v += 2.0 * a_[i] * a_[i] * x_[i] * (1.0 - x_[i]);
        return v + p_.sigma2;
    }

    // third moment; the symmetric background contributes nothing
    double third_moment() const {
        double v = 0.0;
        for (std::size_t i = 0; i < x_.size(); ++i) {
            v += 2.0 * a_[i] * a_[i] * a_[i] * x_[i] * (1.0 - x_[i]) * (1.0 - 2.0 * x_[i]);
        }
        return v;
    }

    void recursion() {
        double d = p_.shift;
        out_->d_trajectory.clear();
        out_->d_trajectory.push_back(d);

        if (p_.full_output) take_snapshot(kSnapInitial, 0.0, d);

        // 10-generation sliding windows of D, V and u3, maintained only until the
        // quasi-static criterion first fires
        constexpr int kWin = 10;
        std::deque<double> D_buf, V_buf, u3_buf;

        std::int64_t t = 0;
        while (static_cast<double>(t) < p_.stop_time || std::fabs(d) > 1.0) {
            const double td = static_cast<double>(t);

            double change_in_d = update_mutation_frequencies(d, td);
            change_in_d += add_new_mutations(td);

            d += -change_in_d - p_.sigma2 * d / twoN_;
            out_->d_trajectory.push_back(d);

            if (std::isnan(out_->convergence_time)) {
                if (static_cast<int>(D_buf.size()) == kWin) {
                    D_buf.pop_front();
                    V_buf.pop_front();
                    u3_buf.pop_front();
                }
                D_buf.push_back(d);
                V_buf.push_back(second_moment());
                u3_buf.push_back(third_moment());
                if (static_cast<int>(D_buf.size()) == kWin) {
                    double D_w = 0.0, V_w = 0.0, u3_w = 0.0;
                    for (int i = 0; i < kWin; ++i) {
                        D_w += D_buf[i];
                        V_w += V_buf[i];
                        u3_w += u3_buf[i];
                    }
                    D_w /= kWin;
                    V_w /= kWin;
                    u3_w /= kWin;
                    if (D_w != 0.0 && V_w != 0.0 &&
                        std::fabs((D_w - u3_w / (2.0 * V_w)) / D_w) < 0.05) {
                        out_->convergence_time = td;
                        out_->convergence_D_window = D_w;
                        if (p_.full_output) take_snapshot(kSnapQuasiStatic, td, d);
                    }
                }
            }

            if (p_.full_output) {
                if (t == 20) take_snapshot(kSnapGen20, td, d);
                else if (t == 300) take_snapshot(kSnapGen300, td, d);
            }

            // moments on a schedule that samples densely early, sparsely later
            if (p_.record_moments &&
                (t < 100 || (t < 500 && t % 5 == 0) || (t < 1000 && t % 10 == 0) || t % 50 == 0)) {
                out_->second_moment.push_back(second_moment());
                out_->third_moment.push_back(third_moment());
            }

            ++t;
        }

        if (p_.full_output) {
            take_snapshot(kSnapFinal, static_cast<double>(t), d);
            for (std::size_t i = 0; i < ids_.size(); ++i) {
                if (reached_[i]) record_trajectory(i, kFateSegregating, kNaN);
            }
        }
    }
};

}  // namespace

// ── C interface ─────────────────────────────────────────────────────────────
// Handles are opaque pointers; every getter takes the handle plus caller-owned
// output buffers whose sizes come from the matching count getter. Errors are
// reported by returning null and leaving a message in sim_last_error().

extern "C" {

const char* sim_last_error(void) { return g_last_error.c_str(); }

// components: n_components entries of (kind, loc, scale, weight), kind 0 = expon,
// 1 = uniform. n_s_nodes / n_x_nodes size the steady-state tables (0 = default).
void* sim_ctx_create(int N, double N2U, double sigma2, double shift, int tracking_time,
                     double stop_time, int record_moments, int full_output,
                     int n_components, const int* kinds, const double* locs,
                     const double* scales, const double* weights,
                     int n_s_nodes, int n_x_nodes) {
    try {
        Params p;
        p.N = N;
        p.N2U = N2U;
        p.sigma2 = sigma2;
        p.shift = shift;
        p.tracking_time = tracking_time;
        p.stop_time = stop_time;
        p.record_moments = record_moments != 0;
        p.full_output = full_output != 0;

        std::vector<sim::Component> comps;
        std::vector<double> w;
        comps.reserve(static_cast<std::size_t>(n_components));
        w.reserve(static_cast<std::size_t>(n_components));
        for (int i = 0; i < n_components; ++i) {
            sim::Component c;
            c.kind = (kinds[i] == 0) ? sim::CompKind::Expon : sim::CompKind::Uniform;
            c.loc = locs[i];
            c.scale = scales[i];
            comps.push_back(c);
            w.push_back(weights[i]);
        }
        sim::Sdist sdist(std::move(comps), std::move(w));
        sim::Steady steady(sdist, N,
                           n_s_nodes > 0 ? n_s_nodes : 513,
                           n_x_nodes > 0 ? n_x_nodes : 1025);
        return new Context(p, std::move(sdist), std::move(steady));
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return nullptr;
    } catch (...) {
        g_last_error = "unknown error in sim_ctx_create";
        return nullptr;
    }
}

void sim_ctx_free(void* ctx) { delete static_cast<Context*>(ctx); }

// Expected number of standing-variation sites, i.e. Simulation.total_n().
double sim_expected_n_segregating(void* ctx) {
    const Context* c = static_cast<const Context*>(ctx);
    return c->steady.sojourn_integral() * c->p.N2U;
}

// Draw `n` standing-variation alleles without running a replicate (validation).
void sim_sample_standing(void* ctx, std::uint64_t seed, int n, double* x_out, double* s_out) {
    const Context* c = static_cast<const Context*>(ctx);
    sim::Rng rng(seed);
    for (int i = 0; i < n; ++i) c->steady.sample(rng, &s_out[i], &x_out[i]);
}

// Test hooks: draw `count` variates straight from the discrete samplers, so
// validate_sim_core.py can compare them with NumPy's without going through a
// whole replicate.
void sim_test_binomial(std::uint64_t seed, std::int64_t n, double p, int count, double* out) {
    sim::Rng rng(seed);
    for (int i = 0; i < count; ++i) out[i] = static_cast<double>(rng.binomial(n, p));
}

void sim_test_poisson(std::uint64_t seed, double lam, int count, double* out) {
    sim::Rng rng(seed);
    for (int i = 0; i < count; ++i) out[i] = static_cast<double>(rng.poisson(lam));
}

void* sim_run(void* ctx, std::uint64_t seed) {
    try {
        const Context* c = static_cast<const Context*>(ctx);
        Result* r = new Result();
        Replicate rep(*c, seed);
        rep.run(r);
        return r;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return nullptr;
    } catch (...) {
        g_last_error = "unknown error in sim_run";
        return nullptr;
    }
}

void sim_result_free(void* res) { delete static_cast<Result*>(res); }

int sim_n_fixations(void* res) {
    return static_cast<int>(static_cast<Result*>(res)->fixations.size());
}

void sim_get_fixations(void* res, double* a, double* t_initial, double* x0) {
    const Result* r = static_cast<const Result*>(res);
    for (std::size_t i = 0; i < r->fixations.size(); ++i) {
        a[i] = r->fixations[i].a;
        t_initial[i] = r->fixations[i].t_initial;
        x0[i] = r->fixations[i].x0;
    }
}

double sim_convergence_time(void* res) { return static_cast<Result*>(res)->convergence_time; }
double sim_convergence_D(void* res) { return static_cast<Result*>(res)->convergence_D_window; }

int sim_n_d_trajectory(void* res) {
    return static_cast<int>(static_cast<Result*>(res)->d_trajectory.size());
}

void sim_get_d_trajectory(void* res, double* out) {
    const Result* r = static_cast<const Result*>(res);
    std::memcpy(out, r->d_trajectory.data(), r->d_trajectory.size() * sizeof(double));
}

int sim_n_moments(void* res) {
    return static_cast<int>(static_cast<Result*>(res)->second_moment.size());
}

void sim_get_moments(void* res, double* m2, double* m3) {
    const Result* r = static_cast<const Result*>(res);
    std::memcpy(m2, r->second_moment.data(), r->second_moment.size() * sizeof(double));
    std::memcpy(m3, r->third_moment.data(), r->third_moment.size() * sizeof(double));
}

int sim_n_trajectories(void* res) {
    return static_cast<int>(static_cast<Result*>(res)->trajectories.size());
}

// out6: id, a, t_initial, x0, fate code (0 fixed / 1 extinct / 2 segregating),
// fixation_time
void sim_trajectory_meta(void* res, int i, double* out6) {
    const TrajectoryRecord& rec = static_cast<const Result*>(res)->trajectories[i];
    out6[0] = static_cast<double>(rec.id);
    out6[1] = rec.a;
    out6[2] = rec.t_initial;
    out6[3] = rec.x0;
    out6[4] = static_cast<double>(rec.fate);
    out6[5] = rec.fixation_time;
}

int sim_trajectory_len(void* res, int i) {
    return static_cast<int>(static_cast<const Result*>(res)->trajectories[i].t.size());
}

void sim_get_trajectory(void* res, int i, double* t, double* x) {
    const TrajectoryRecord& rec = static_cast<const Result*>(res)->trajectories[i];
    std::memcpy(t, rec.t.data(), rec.t.size() * sizeof(double));
    std::memcpy(x, rec.x.data(), rec.x.size() * sizeof(double));
}

int sim_snapshot_present(void* res, int key) {
    return static_cast<const Result*>(res)->snapshots[key].present ? 1 : 0;
}

// out2: generation, distance D
void sim_snapshot_scalars(void* res, int key, double* out2) {
    const Snapshot& s = static_cast<const Result*>(res)->snapshots[key];
    out2[0] = s.time;
    out2[1] = s.D;
}

int sim_snapshot_n_segregating(void* res, int key) {
    return static_cast<int>(static_cast<const Result*>(res)->snapshots[key].seg_id.size());
}

void sim_get_snapshot_segregating(void* res, int key, double* id, double* a, double* x,
                                  double* t_initial, double* x0) {
    const Snapshot& s = static_cast<const Result*>(res)->snapshots[key];
    const std::size_t n = s.seg_id.size();
    std::memcpy(id, s.seg_id.data(), n * sizeof(double));
    std::memcpy(a, s.seg_a.data(), n * sizeof(double));
    std::memcpy(x, s.seg_x.data(), n * sizeof(double));
    std::memcpy(t_initial, s.seg_t_initial.data(), n * sizeof(double));
    std::memcpy(x0, s.seg_x0.data(), n * sizeof(double));
}

int sim_snapshot_n_fixed(void* res, int key) {
    return static_cast<int>(static_cast<const Result*>(res)->snapshots[key].fix_a.size());
}

void sim_get_snapshot_fixed(void* res, int key, double* a, double* t_initial, double* x0) {
    const Snapshot& s = static_cast<const Result*>(res)->snapshots[key];
    const std::size_t n = s.fix_a.size();
    std::memcpy(a, s.fix_a.data(), n * sizeof(double));
    std::memcpy(t_initial, s.fix_t_initial.data(), n * sizeof(double));
    std::memcpy(x0, s.fix_x0.data(), n * sizeof(double));
}

}  // extern "C"
