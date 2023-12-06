#pragma once
#include <common/point/internal/pi_base.h>
#include <common/types.h>
#include <functional>
#include <type_traits>

namespace frisk {

/*
 * Base class for internal implementation of Gaussian elastic-net point solver.
 * This contains all the common interface and members across all versions of gaussian:
 *      - dense gaussian naive
 */
struct ElnetPointInternalGaussianBase
    : ElnetPointInternalBase
{
private:
    using base_t = ElnetPointInternalBase;

protected:
    using typename base_t::vec_t;
    using typename base_t::ivec_t;

public:
    using typename base_t::value_t;
    using typename base_t::index_t;
    using typename base_t::bool_t;

    template <class IAType
            , class XVType
            , class VPType
            , class CLType
            , class JUType>
    ElnetPointInternalGaussianBase(
            value_t thr,
            index_t maxit,
            index_t nx,
            index_t& nlp,
            IAType& ia,
            const XVType& xv,
            const VPType& vp,
            const CLType& cl,
            const JUType& ju,
            value_t rsq = 0.0)
        : base_t(thr, maxit, nx, nlp, ia, vp, cl, ju)
        , rsq_(rsq)
        , xv_(xv.data(), xv.size())
    {}

    GLMNETPP_STRONG_INLINE bool is_warm_ever() const { return iz_; }
    GLMNETPP_STRONG_INLINE void set_warm_ever() { iz_ = true; }

    GLMNETPP_STRONG_INLINE
    void update_dlx(index_t k, value_t beta_diff) {
        base_t::update_dlx(beta_diff, xv_(k));
    }

    GLMNETPP_STRONG_INLINE constexpr void update_intercept() const {}

    GLMNETPP_STRONG_INLINE auto rsq() const { return rsq_; }
    GLMNETPP_STRONG_INLINE auto rsq_prev() const { return rsq_prev_; }

    /* Static interface */

    GLMNETPP_STRONG_INLINE
    static void
    update_rsq(value_t& rsq, value_t beta_diff, value_t gk, value_t x_var) { 
        rsq += beta_diff * (2.0 * gk - beta_diff * x_var);
    }

protected:
    using base_t::update_dlx;
    using base_t::update_intercept;

    GLMNETPP_STRONG_INLINE auto& rsq() { return rsq_; }
    GLMNETPP_STRONG_INLINE void initialize() { rsq_prev_ = rsq_; }

    GLMNETPP_STRONG_INLINE
    void update_rsq(index_t k, value_t beta_diff, value_t gk) { 
        update_rsq(rsq_, beta_diff, gk, xv_(k));
    }

    GLMNETPP_STRONG_INLINE
    auto x_var(index_t i) const { return xv_[i]; }

private:
    // internal non-captures
    bool iz_ = false;           // true if a partial fit was done with a previous lambda (warm ever)
    value_t rsq_;               // R^2
    value_t rsq_prev_ = 0.0;    // previous R^2

    // captures
    Eigen::Map<const vec_t> xv_;        // variance of columns of x
};

/*
 * Base class for internal implementation of Gaussian univariate-response methods.
 */

struct ElnetPointInternalGaussianUniBase
    : ElnetPointInternalGaussianBase
{
private:
    using base_t = ElnetPointInternalGaussianBase;

protected:
    using typename base_t::vec_t;
    using typename base_t::value_t;
    using typename base_t::index_t;

    template <class IAType
            , class XVType
            , class VPType
            , class CLType
            , class JUType>
    ElnetPointInternalGaussianUniBase(
            value_t thr,
            index_t maxit,
            index_t nx,
            index_t& nlp,
            IAType& ia,
            const XVType& xv,
            const VPType& vp,
            const CLType& cl,
            const JUType& ju)
        : base_t(thr, maxit, nx, nlp, ia, xv, vp, cl, ju)
        , a_(xv.size())
    {
        a_.setZero(); 
    }

    GLMNETPP_STRONG_INLINE
    void update_beta(index_t k, value_t ab, value_t dem, value_t gk) {
        const auto& cl = this->endpts();
        base_t::update_beta(
                a_(k), gk, this->x_var(k), this->penalty()(k),
                cl(0,k), cl(1,k), ab, dem);
    }

public:
    GLMNETPP_STRONG_INLINE auto beta(index_t k) const { return a_(k); }

private:
    vec_t a_;                   // uncompressed beta
};

/*
 * Base class for internal implementation of Gaussian naive method.
 * This contains all the common interface and members for gaussian naive methods:
 *      - dense gaussian naive
 *      - sparse gaussian naive
 */
struct ElnetPointInternalGaussianNaiveBase
        : ElnetPointInternalGaussianUniBase
{
private:
    using base_t = ElnetPointInternalGaussianUniBase;

protected:
    using typename base_t::vec_t;
    using typename base_t::mat_t;

public:
    using typename base_t::value_t;
    using typename base_t::index_t;

    template <class IAType
            , class XVType
            , class VPType
            , class CLType
            , class JUType>
    ElnetPointInternalGaussianNaiveBase(
            value_t thr,
            index_t maxit,
            index_t nx,
            index_t& nlp,
            IAType& ia,
            const XVType& xv,
            const VPType& vp,
            const CLType& cl,
            const JUType& ju)
        : base_t(thr, maxit, nx, nlp, ia, xv, vp, cl, ju)
        , g_(ju.size())
        , ix_(ju.size(), false)
    {
        g_.setZero();
    }
    
    using base_t::update_intercept;

    GLMNETPP_STRONG_INLINE bool is_excluded(index_t j) const { return !ix_[j]; }

    template <class InitialFitIntType>
    GLMNETPP_STRONG_INLINE
    constexpr bool initial_fit(InitialFitIntType f) const {
        return initial_fit([&]() { return this->has_reached_max_passes(); }, f);
    }

    template <class PointPackType>
    GLMNETPP_STRONG_INLINE
    void initialize(const PointPackType& pack) {
        base_t::initialize();
        initialize_strong_set(pack);
    }

    GLMNETPP_STRONG_INLINE
    void update_rsq(index_t k, value_t beta_diff) { 
        base_t::update_rsq(k, beta_diff, gk_cache_);
    }

    template <class AbsGradFType>
    GLMNETPP_STRONG_INLINE
    bool check_kkt(value_t ab, AbsGradFType abs_grad_f) {
        auto skip_f = [&](auto k) { return !is_excluded(k) || !this->exclusion()[k]; };
        update_abs_grad(g_, abs_grad_f, skip_f);
        return check_kkt(g_, this->penalty(), ix_, ab, skip_f);
    }

    const auto& abs_grad() const { return g_; }

    /* Static interface */

    template <class RType, class XType>
    GLMNETPP_STRONG_INLINE
    static void
    update_resid(
            RType&& r,
            value_t beta_diff,
            const XType& x) 
    {
        r -= beta_diff * x;
    }

    template <class RType, class VType>
    GLMNETPP_STRONG_INLINE
    static value_t 
    update_intercept(
            value_t& intercept,
            RType&& r,
            value_t& dlx,
            bool intr,
            value_t r_sum,
            value_t var,
            const VType& v)
    {
        auto d = base_t::update_intercept(intercept, dlx, intr, r_sum, var);
        if (d) update_resid(r, d, v);
        return d;
    }

    template <class HasReachedMaxPassesType, class InitialFitIntType>
    GLMNETPP_STRONG_INLINE
    constexpr static bool initial_fit(
            HasReachedMaxPassesType has_reached_max_passes, 
            InitialFitIntType f) 
    { 
        // Keep doing initial fit until either doesn't converge or 
        // converged and kkt passed.
        while (1) {
            if (has_reached_max_passes()) { 
                throw util::maxit_reached_error();
            }
            bool converged = false, kkt_passed = false;
            std::tie(converged, kkt_passed) = f();
            if (!converged) break;
            if (kkt_passed) return true;
        }
        return false;
    }

    /*
     * Updates absolute gradient abs_grad by iterating through each element
     * and assigning compute_grad_f(k). Iteration skips over k whenever skip_f(k) is true.
     */
    template <class AbsGradType, class ComputeAbsGradFType, class SkipFType>
    GLMNETPP_STRONG_INLINE
    static void update_abs_grad(
            AbsGradType&& abs_grad,
            ComputeAbsGradFType compute_abs_grad_f,
            SkipFType skip_f) 
    {
        base_t::for_each_with_skip(
                util::counting_iterator<index_t>(0), 
                util::counting_iterator<index_t>(abs_grad.size()),
                [&](index_t j) { abs_grad(j) = compute_abs_grad_f(j); },
                skip_f);
    }

    /*
     * Checks KKT condition and computes strong map. See base_t::compute_strong_map;
     * Returns true if no update occured (KKT all passed).
     */
    template <class AbsGradType
            , class PenaltyType
            , class StrongMapType
            , class SkipFType>
    GLMNETPP_STRONG_INLINE
    static bool check_kkt(
            AbsGradType&& abs_grad,
            const PenaltyType& penalty,
            StrongMapType&& strong_map,
            value_t l1_regul,
            SkipFType skip_f) {
        return !base_t::compute_strong_map(abs_grad, penalty, strong_map, l1_regul, skip_f);
    }

    template <class AbsGradType
            , class PenaltyType
            , class StrongMapType
            , class FType
            , class SkipFType>
    GLMNETPP_STRONG_INLINE
    static bool check_kkt(
            AbsGradType&& abs_grad,
            const PenaltyType& penalty,
            StrongMapType&& strong_map,
            value_t l1_regul,
            FType f,
            SkipFType skip_f) {
        return !base_t::compute_strong_map(abs_grad, penalty, strong_map, l1_regul, f, skip_f);
    }

protected:
    GLMNETPP_STRONG_INLINE
    void update_beta(index_t k, value_t ab, value_t dem, value_t gk) {
        gk_cache_ = gk; 
        base_t::update_beta(k, ab, dem, gk_cache_);
    }

    template <class PointPackType>
    GLMNETPP_STRONG_INLINE
    void initialize_strong_set(const PointPackType& pack) 
    {
        base_t::compute_strong_map(
                g_, this->penalty(), ix_, 
                pack.elastic_prop(), pack.lmda(), pack.prev_lmda(),
                [&](auto k) { return !is_excluded(k) || !this->exclusion()[k]; }
                );
    }

    template <class AbsGradFType>
    GLMNETPP_STRONG_INLINE
    void construct(AbsGradFType abs_grad_f) {
        update_abs_grad(g_, abs_grad_f,
                [&](auto j) { return !this->exclusion()[j]; });
    }

private:
    value_t gk_cache_ = 0.0;    // caches gradient at k when updating beta
    vec_t g_;                   // cwise-absolute gradient
    std::vector<bool> ix_;      // strong set indicators
};

} // namespace frisk
