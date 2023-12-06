#include <common/driver_gaussian.h>
#include <common/elnet.h>

namespace frisk {

void ElNet::elnet_exp(
    double parm, // alpha
    Eigen::MatrixXd& x,
    Eigen::VectorXd& y,
    Eigen::VectorXd& w,
    const Eigen::Map<Eigen::VectorXd>& vp, // vp=as.double(penalty.factor)
    Eigen::MatrixXd& cl, // cl=rbind(lower.limits, upper.limits)
    int ne, // as.integer(dfmax)
    int nx, // as.integer(pmax)
    int nlam, // as.integer(length(lambda))
    double flmin, // as.double(lambda.min.ratio)
    const Eigen::Map<Eigen::VectorXd>& ulam, // ulam=as.double(rev(sort(lambda)))
    double thr, // Convergence threshold for coordinate descent, default 1E-7
    bool isd, // Logical flag for x variable standardization, default true
    bool intr, // intercept
    int maxit, // Maximum number of passes over the data for all lambda values, default is 10^5.
    Eigen::Map<Eigen::VectorXd>& a0, // double(nlam)
    Eigen::Map<Eigen::MatrixXd>& ca, // matrix(0.0, nrow=nx, ncol=nlam),
    Eigen::Map<Eigen::VectorXi>& ia, // integer(nx),
    Eigen::Map<Eigen::VectorXi>& nin, // integer(nlam),
    Eigen::Map<Eigen::VectorXd>& rsq, // double(nlam),
    Eigen::Map<Eigen::VectorXd>& alm // double(nlam),
) {
//    std::cout << "w:\n" << w << "\njd:\n" << jd << "\nvp:\n"<< vp << "\ncl:\n"<< cl
//              << "\nulam:\n"<< ulam << "\na0:\n"<< a0 << "\nca:\n"<< ca << "\nia:\n"
//              << ia << "\nnin:\n"<< nin << "\nrsq:\n"<< rsq << "\nalm:\n"<< alm << "\n";
    using elnet_driver_t = ElnetDriver;
    elnet_driver_t driver;
    try {
        driver.fit(
            parm, x, y, w, vp, cl, ne, nx, nlam, flmin,
            ulam, thr, isd, intr, maxit,
            lmu, a0, ca, ia, nin, rsq, alm, nlp, jerr, InternalParams());
    }
    catch (const std::bad_alloc&) {
        jerr = util::bad_alloc_error().err_code();
    }
    catch (const std::exception&) {
        jerr = 10001;
    }
}

void ElNet::fit(Eigen::MatrixXd X, Eigen::VectorXd y, double alpha, int nlambda, double lambda_min_ratio,
         std::vector<double> lambda_path,
         bool standardize, bool fit_intercept, double tolerance,
         int maxit, int max_features,
         std::vector<double> lower_limits, std::vector<double> upper_limits) {
    long nobs = X.rows();
    int nvars = X.cols();
    Eigen::VectorXd weights(nobs, 1);
    weights.setOnes();
    std::vector<double> penalty_factor_(nvars, 1.);
    Eigen::Map<Eigen::VectorXd> vp(penalty_factor_.data(), penalty_factor_.size());
    if (max_features <= 0 || max_features > nvars + 1) max_features = nvars + 1;
    pmax = std::min(max_features * 2 + 20, nvars); // Limit the maximum number of variables ever to be nonzero
    if (lambda_path.empty()) {
        ulam_ = {1};
    } else {
        for (size_t i = 0; i < lambda_path.size(); ++i) {
            if (lambda_path[i] < 0) throw std::runtime_error("lambdas should be non-negative");
        }
        std::sort(lambda_path.begin(), lambda_path.end());
        std::reverse(lambda_path.begin(), lambda_path.end());
        nlambda = lambda_path.size();
        ulam_ = lambda_path;
        lambda_min_ratio = 1;
    }

    Eigen::MatrixXd cl(2, nvars);
    for (int k = 0; k < nvars; ++k) {
        if ((size_t)k < lower_limits.size()) cl(0, k) = lower_limits[k];
        else cl(0, k) = -INFINITY;

        if ((size_t)k < upper_limits.size()) cl(1, k) = upper_limits[k];
        else cl(1, k) = INFINITY;
    }

    if(nobs < nvars) lambda_min_ratio = 1e-2;

    m_nlambda = nlambda;
    a0_.resize(nlambda, 0.);
    ca_.resize(pmax * nlambda, 0);
    ia_.resize(pmax, 0);
    nin_.resize(nlambda, 0);
    rsq_.resize(nlambda, 0.);
    alm_.resize(nlambda, 0.);

    Eigen::Map<Eigen::VectorXd> ulam(ulam_.data(), ulam_.size());
    Eigen::Map<Eigen::VectorXd> a0(a0_.data(), a0_.size());
    Eigen::Map<Eigen::MatrixXd> ca(ca_.data(), pmax, nlambda);
    Eigen::Map<Eigen::VectorXi> ia(ia_.data(), ia_.size());
    Eigen::Map<Eigen::VectorXi> nin(nin_.data(), nin_.size());
    Eigen::Map<Eigen::VectorXd> rsq(rsq_.data(), rsq_.size());
    Eigen::Map<Eigen::VectorXd> alm(alm_.data(), alm_.size());

    elnet_exp(alpha, X, y, weights, vp, cl,
              max_features, pmax, nlambda, lambda_min_ratio, ulam, tolerance, standardize, fit_intercept, maxit,
                  a0, ca, ia, nin, rsq, alm);

//    std::cout << alm << std::endl;
//    std::cout << a0 << std::endl;
}

Eigen::VectorXd ElNet::predit(Eigen::MatrixXd newx, double s) {
    if (lmu < 1) return {};
    int m_idx = 6;
    int ninmax = 0;
    for (int i = 0; i < lmu; ++i) {
        if (nin_[i] > ninmax) ninmax = nin_[i];
    }
    Eigen::Map<Eigen::MatrixXd> ca_map(ca_.data(), pmax, m_nlambda);
    Eigen::MatrixXd ca = ca_map.block(0, 0, ninmax, lmu);
    Eigen::VectorXd coef_ca = ca.col(m_idx);
//    std::cout << "coef_ca:\n" << coef_ca << std::endl;

    double intercept = a0_[m_idx];
    Eigen::VectorXd coefs(ia_.size());
    coefs.setZero();
    for (int k = 0; k < ninmax; ++k) {
        coefs(ia_[k] - 1) = coef_ca(k);
    }
    return (newx * coefs).array() + intercept;
}

std::vector<double> ElNet::get_lambdas() {
    if (lmu < 1) return {};
    return std::vector<double>(alm_.begin(), alm_.begin() + lmu);
}
}