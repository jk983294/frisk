#pragma once

#include <Eigen/Dense>

namespace frisk {

struct ElNet {
    void elnet_exp(
        bool is_cov, // covariance=1, naive=2,
        double parm, // alpha
        Eigen::MatrixXd x,
        Eigen::VectorXd y,
        Eigen::VectorXd w,
        const Eigen::Map<Eigen::VectorXi> jd, // if(length(exclude)>0) match(exclude,seq(nvars),0)
        const Eigen::Map<Eigen::VectorXd> vp, // vp=as.double(penalty.factor)
        Eigen::MatrixXd cl, // cl=rbind(lower.limits, upper.limits)
        int ne, // as.integer(dfmax)
        int nx, // as.integer(pmax)
        int nlam, // as.integer(length(lambda))
        double flmin, // as.double(lambda.min.ratio)
        const Eigen::Map<Eigen::VectorXd> ulam, // ulam=as.double(rev(sort(lambda)))
        double thr, // Convergence threshold for coordinate descent, default 1E-7
        int isd, // Logical flag for x variable standardization, default true
        int intr, // intercept
        int maxit, // Maximum number of passes over the data for all lambda values, default is 10^5.
        int& lmu, // 0
        Eigen::Map<Eigen::VectorXd>& a0, // double(nlam)
        Eigen::Map<Eigen::MatrixXd>& ca, // matrix(0.0, nrow=nx, ncol=nlam),
        Eigen::Map<Eigen::VectorXi>& ia, // integer(nx),
        Eigen::Map<Eigen::VectorXi>& nin, // integer(nlam),
        Eigen::Map<Eigen::VectorXd>& rsq, // double(nlam),
        Eigen::Map<Eigen::VectorXd>& alm, // double(nlam),
        int& nlp, // 0
        int& jerr  // 0
    );
};
}


