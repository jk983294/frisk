#include <common/point/internal/pi_gaussian_naive.h>

namespace frisk {
ElnetPointInternal::ElnetPointInternal(double thr,
                   int maxit,
                   int nx,
                   int& nlp,
                   Eigen::Map<Eigen::VectorXi>& ia,
                   Eigen::VectorXd& y,
                   const Eigen::MatrixXd& X,
                   const Eigen::VectorXd& xv,
                   const Eigen::VectorXd& vp,
                   const Eigen::MatrixXd& cl,
                   const std::vector<bool>& ju)
    : base_t(thr, maxit, nx, nlp, ia, xv, vp, cl, ju)
    , X_(X.data(), X.rows(), X.cols())
    , y_(y.data(), y.size())
{
    base_t::construct([this](index_t k) { return compute_abs_grad(k); });
}
}