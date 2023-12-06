#pragma once

#include <Eigen/Dense>

namespace frisk {

struct ElNet {
    /**
     * @param x
     * @param y
     * @param alpha 0 <= alpha <= 1, 0 for ridge, 1 for lasso
     * @param n_lambda Maximum number of lambda values to compute
     * @param min_lambda_ratio the ratio of the smallest and largest values of lambda computed.
     * @param lambda_path In place of supplying n_lambda, provide an array of specific values
        to compute. The specified values must be in decreasing order. When
        None, the path of lambda values will be determined automatically. A
        maximum of `n_lambda` values will be computed.
     * @param standardize Standardize input features prior to fitting. The final coefficients
        will be on the scale of the original data regardless of the value
        of standardize.
     * @param fit_intercept Include an intercept term in the model.
     * @param tolerance Convergence tolerance.
     * @param max_iter Maximum passes over the data for all values of lambda.
     * @param max_features Optional maximum number of features with nonzero coefficients after regularization.
     * @param lower_limits Array of lower limits for each coefficient, must be non-positive.
     * @param upper_limits Array of upper limits for each coefficient, must be positive.
     */
    void fit(Eigen::MatrixXd x, Eigen::VectorXd y, double alpha = 1., int n_lambda = 100, double min_lambda_ratio = 1e-4,
             std::vector<double> lambda_path = {},
             bool standardize = true, bool fit_intercept= true, double tolerance = 1e-7,
             int max_iter = 100000, int max_features = -1,
             std::vector<double> lower_limits = {}, std::vector<double> upper_limits = {});

    Eigen::VectorXd predit(Eigen::MatrixXd newx, double s = NAN);
    std::vector<double> get_lambdas();

private:
    void elnet_exp(
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
    );

public:
    std::vector<double> ulam_;
    std::vector<double> a0_; // intercept
    std::vector<double> ca_; // coefs[i] = ca_[, i]
    std::vector<int> ia_; // index of variables selected
    std::vector<int> nin_; // number of variables selected
    std::vector<double> rsq_; // dev.ratio=1-dev/nulldev, The fraction of (null) deviance explained.
    std::vector<double> alm_; // value of lambdas
    int nlp = 0; // total passes over the data
    int jerr = 0; // Error flag
    int lmu = 0; // valid model number
    int pmax = 0, m_nlambda = 0;
};
}


