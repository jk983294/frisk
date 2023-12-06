#pragma once
#include <Eigen/Core>
#include <type_traits>

namespace frisk {

struct Standardize1
{
    static void eval(
        Eigen::MatrixXd& x,
        Eigen::VectorXd& y,
        Eigen::VectorXd& w,
        bool isd, 
        bool intr, 
        const std::vector<bool>& ju,
        Eigen::VectorXd& xm,
        Eigen::VectorXd& xs,
        double& ym,
        double& ys,
        Eigen::VectorXd& xv)
    {
        auto nvars = x.cols();

        w /= w.sum();

        Eigen::VectorXd v = w.array().sqrt().matrix();

        // without intercept
        if (!intr) {

            ym = 0.0;
            y.array() *= v.array();

            // trevor
            ys = y.norm(); 
            y /= ys;
            for (int j = 0; j < nvars; ++j) {
                if (!ju[j]) continue; 
                xm(j) = 0.0; 
                auto x_j = x.col(j);
                x_j.array() *= v.array();
                xv(j) = x_j.squaredNorm();
                if (isd) { 
                    auto xbq = x_j.dot(v);
                    xbq *= xbq;
                    auto vc = xv(j) - xbq;
                    xs(j) = std::sqrt(vc); 
                    x_j /= xs(j); 
                    xv(j) = 1.0 + xbq/vc;
                }
                else { xs(j) = 1.0; }
            }

        } 

        // with intercept
        else {

            for (int j = 0; j < nvars; ++j) {
                if (!ju[j]) continue;
                auto x_j = x.col(j);
                xm(j) = x_j.dot(w); 
                x_j.array() = v.array() * (x_j.array() - xm(j));
                xv(j) = x_j.squaredNorm(); 
                if (isd) xs(j) = std::sqrt(xv(j));
            }
            if (!isd) { xs.setOnes(); }
            else {
                 for (int j = 0; j < nvars; ++j) {
                     if (!ju[j]) continue; 
                     auto x_j = x.col(j);
                     x_j /= xs(j);
                 }
                 xv.setOnes();
            }
            ym = y.dot(w);
            y.array() = v.array() * (y.array() - ym); 
            ys = y.norm(); 
            y /= ys;
        }
    }
};

}
