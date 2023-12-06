#pragma once
#include <common/point/internal/pi_decl.h>
#include <common/point/internal/pi_gaussian_base.h>

namespace frisk {

struct ElnetPointInternal
        : ElnetPointInternalGaussianNaiveBase
{
private:
    using base_t = ElnetPointInternalGaussianNaiveBase;

public:
    using typename base_t::value_t;
    using typename base_t::index_t;

    ElnetPointInternal(double thr,
                       int maxit,
                       int nx,
                       int& nlp,
                       Eigen::Map<Eigen::VectorXi>& ia,
                       Eigen::VectorXd& y,
                       const Eigen::MatrixXd& X,
                       const Eigen::VectorXd& xv,
                       const Eigen::VectorXd& vp,
                       const Eigen::MatrixXd& cl,
                       const std::vector<bool>& ju);

    template <class PointPackType>
    GLMNETPP_STRONG_INLINE
    void update_beta(index_t k, const PointPackType& pack) {
        base_t::update_beta(k, pack.ab, pack.dem, compute_grad(k));
    }

    GLMNETPP_STRONG_INLINE
    void update_resid(index_t k, value_t beta_diff) {
        base_t::update_resid(y_, beta_diff, X_.col(k));
    }

    template <class PointPackType>
    GLMNETPP_STRONG_INLINE
    bool check_kkt(const PointPackType& pack) {
        return base_t::check_kkt(pack.ab, [this](index_t k) { return compute_abs_grad(k); });
    }

private:
    GLMNETPP_STRONG_INLINE
    value_t compute_grad(index_t k) const {
        return base_t::compute_grad(y_, X_.col(k));
    }

    GLMNETPP_STRONG_INLINE
    value_t compute_abs_grad(index_t k) const {
        return std::abs(compute_grad(k));
    }

    using typename base_t::vec_t;
    using typename base_t::mat_t;

    Eigen::Map<const mat_t> X_; // data matrix
    Eigen::Map<vec_t> y_;       // scaled residual vector
                                // Note: this is slightly different from sparse version residual vector.
                                // Sparse one will not be scaled by sqrt(weights), but this one will.
};

} // namespace frisk
