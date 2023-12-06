#include <common/point/point_gaussian_naive.h>

namespace frisk {
ElnetPoint::ElnetPoint(double thr,
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
    : ElnetPointInternal(thr, maxit, nx, nlp, ia, y, X, xv, vp, cl, ju)
{

}
}