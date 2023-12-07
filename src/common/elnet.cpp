#include <common/driver_gaussian.h>
#include <common/elnet.h>

namespace frisk {

void ElNet::elnet_exp(
    double parm, // alpha
    Eigen::MatrixXd& x,
    Eigen::VectorXd& y,
    Eigen::MatrixXd& cl, // cl=rbind(lower.limits, upper.limits)
    int ne, // as.integer(dfmax)
    int nx, // as.integer(pmax)
    int nlam, // as.integer(length(lambda))
    double flmin, // as.double(lambda.min.ratio)
    double thr, // Convergence threshold for coordinate descent, default 1E-7
    bool isd, // Logical flag for x variable standardization, default true
    bool intr, // intercept
    int maxit // Maximum number of passes over the data for all lambda values, default is 10^5.
) {
    long nobs = x.rows();
    int nvars = x.cols();
    Eigen::VectorXd weights(nobs, 1);
    weights.setOnes();
    std::vector<double> penalty_factor_(nvars, 1.);
    Eigen::Map<Eigen::VectorXd> vp(penalty_factor_.data(), penalty_factor_.size());

    Eigen::Map<Eigen::VectorXd> ulam(ulam_.data(), ulam_.size()); // ulam=as.double(rev(sort(lambda)))
    Eigen::Map<Eigen::VectorXd> a0(a0_.data(), a0_.size()); // double(nlam)
    Eigen::Map<Eigen::MatrixXd> ca(ca_.data(), pmax, m_nlambda); // matrix(0.0, nrow=nx, ncol=nlam),
    Eigen::Map<Eigen::VectorXi> ia(ia_.data(), ia_.size()); // integer(nx),
    Eigen::Map<Eigen::VectorXi> nin(nin_.data(), nin_.size()); // integer(nlam),
    Eigen::Map<Eigen::VectorXd> rsq(rsq_.data(), rsq_.size()); // double(nlam),
    Eigen::Map<Eigen::VectorXd> alm(alm_.data(), alm_.size()); // double(nlam),
//    std::cout << "w:\n" << w << "\njd:\n" << jd << "\nvp:\n"<< vp << "\ncl:\n"<< cl
//              << "\nulam:\n"<< ulam << "\na0:\n"<< a0 << "\nca:\n"<< ca << "\nia:\n"
//              << ia << "\nnin:\n"<< nin << "\nrsq:\n"<< rsq << "\nalm:\n"<< alm << "\n";
    using elnet_driver_t = ElnetDriver;
    elnet_driver_t driver;
    try {
        driver.fit(
            parm, x, y, weights, vp, cl, ne, nx, nlam, flmin,
            ulam, thr, isd, intr, maxit,
            lmu, a0, ca, ia, nin, rsq, alm, nlp, jerr, InternalParams());
    }
    catch (const std::bad_alloc&) {
        jerr = util::bad_alloc_error().err_code();
    }
    catch (const std::exception&) {
        jerr = 10001;
    }

    if (jerr > 0) return;
    m_lambda_min = driver.m_lambda_min;
    m_lambda_se = driver.m_lambda_se;
}

void ElNet::fit(Eigen::MatrixXd X, Eigen::VectorXd y, double alpha, int nlambda, double lambda_min_ratio,
         std::vector<double> lambda_path,
         bool standardize, bool fit_intercept, double tolerance,
         int maxit, int max_features,
         std::vector<double> lower_limits, std::vector<double> upper_limits) {
    long nobs = X.rows();
    int nvars = X.cols();
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
    a0_ = std::vector<double>(nlambda, 0.);
    ca_ = std::vector<double>(pmax * nlambda, 0);
    ia_ = std::vector<int>(pmax, 0);
    nin_ = std::vector<int>(nlambda, 0);
    rsq_ = std::vector<double>(nlambda, 0.);
    alm_ = std::vector<double>(nlambda, 0.);

    elnet_exp(alpha, X, y, cl,
              max_features, pmax, nlambda, lambda_min_ratio, tolerance, standardize, fit_intercept, maxit);
}

Eigen::VectorXd ElNet::predict(const Eigen::Map<const Eigen::MatrixXd>& newx, double s) {
    if (lmu < 1) return {};
    int _idx = m_lambda_se;
    if (std::isfinite(s)) {
        for (_idx = 0; _idx < lmu; ++_idx) {
            if (alm_[_idx] < s) break;
        }
        if (_idx >= lmu - 1)
            _idx = lmu - 1;
        else if (_idx >= 0) {
            if (s - alm_[_idx] > alm_[_idx + 1] -s) {
                _idx++;
            }
        }
    }
    int ninmax = 0;
    for (int i = 0; i < lmu; ++i) {
        if (nin_[i] > ninmax) ninmax = nin_[i];
    }
    Eigen::Map<Eigen::MatrixXd> ca_map(ca_.data(), pmax, m_nlambda);
    Eigen::VectorXd coef_ca = ca_map.block(0, 0, ninmax, lmu).col(_idx);
//    std::cout << "coef_ca:\n" << coef_ca << std::endl;

    double intercept = a0_[_idx];
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