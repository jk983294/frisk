#pragma once
#include <common/chkvars.h>
#include <common/path/path_base.h>
#include <common/path/path_gaussian_naive.h>
#include <common/point/internal/pi_gaussian_naive.h>
#include <common/point/point_gaussian_naive.h>
#include <common/standardize.h>
#include <common/type_traits.h>
#include <common/types.h>
#include <Eigen/Core>
#include <iostream>
#include <vector>

namespace frisk {

struct FitPathGaussian
{
    static void eval(
            double alpha,
            const Eigen::MatrixXd& x,
            Eigen::VectorXd& y,
            Eigen::VectorXd& g,
            const Eigen::VectorXd& w,
            const std::vector<bool>& ju,
            const Eigen::VectorXd& vq,
            const Eigen::VectorXd& x_var,
            const Eigen::MatrixXd& cl,
            int ne,
            int nx,
            int nlam,
            double flmin,
            const Eigen::VectorXd& vlam,
            double thr,
            int maxit,
            int& lmu,
            Eigen::Map<Eigen::VectorXd>& a0,
            Eigen::Map<Eigen::MatrixXd>& ca,
            Eigen::Map<Eigen::VectorXi>& ia,
            Eigen::Map<Eigen::VectorXi>& nin,
            Eigen::Map<Eigen::VectorXd>& rsq,
            Eigen::Map<Eigen::VectorXd>& alm,
            int& nlp,
            int& jerr,
            InternalParams int_param
            )
    {
        // naive method
        {
            ElnetPath elnet_path;
            elnet_path.fit(alpha, ju, vq, cl, y, ne, nx, x, nlam, flmin, vlam, thr, maxit, x_var,
                    lmu, ca, ia, nin, rsq, alm, nlp, jerr, int_param);
        }
    }
};

struct ElnetDriverBase
{
    void normalize_penalty(Eigen::VectorXd& vq) const {
        if (vq.maxCoeff() <= 0) throw util::non_positive_penalty_error();
        vq.array() = vq.array().max(0.0); // truncate those < 0 to 0
        vq *= vq.size() / vq.sum();
    }

    void init_inclusion(std::vector<bool>& ju) const {
        // can't find true value in ju
        if (std::find_if(ju.begin(), ju.end(), [](auto x) { return x;}) == ju.end()) {
            throw util::all_excluded_error();
        }
    }
};

struct ElnetDriver;

struct ElnetDriver
    : ElnetDriverBase
{
public:
    void fit(
        double alpha_,
        Eigen::MatrixXd& x,
        Eigen::VectorXd& y,
        Eigen::VectorXd& w,
        const Eigen::Map<Eigen::VectorXd>& vp,
        Eigen::MatrixXd& cl,
        int ne,
        int nx,
        int nlam,
        double flmin,
        const Eigen::Map<Eigen::VectorXd>& ulam,
        double thr,
        bool isd,
        bool intr,
        int maxit,
        int& lmu,
        Eigen::Map<Eigen::VectorXd>& a0,
        Eigen::Map<Eigen::MatrixXd>& ca,
        Eigen::Map<Eigen::VectorXi>& ia,
        Eigen::Map<Eigen::VectorXi>& nin,
        Eigen::Map<Eigen::VectorXd>& rsq,
        Eigen::Map<Eigen::VectorXd>& alm,
        int& nlp,
        int& jerr,
        InternalParams int_param
    ) {
        using chkvars_t = Chkvars;
        using standardize_naive_t = Standardize1;

        try {
            Eigen::VectorXd vq = vp; // normalized penalty
            normalize_penalty(vq);
            // std::cout << "normalized penalty: " << vq << std::endl;

            auto nvars = x.cols();

            Eigen::VectorXd g;
            Eigen::VectorXd x_mean(nvars);
            x_mean.setZero(); // x mean
            Eigen::VectorXd x_sd(nvars);
            x_sd.setZero(); // x sd
            Eigen::VectorXd x_var(nvars);
            x_var.setZero(); // x variance
            Eigen::VectorXd vlam(nlam); vlam.setZero();
            std::vector<bool> ju(nvars, false);

            chkvars_t::eval(x, ju);
            init_inclusion(ju);

            double y_mean = 0; // y mean
            double y_sd = 0; // y sd

            // naive method
            standardize_naive_t::eval(x, y, w, isd, intr, ju, x_mean, x_sd, y_mean, y_sd, x_var);

            cl /= y_sd;
            if (isd) {
                for (int j = 0; j < nvars; ++j) {
                    cl.col(j) *= x_sd(j);
                }
            }

            if (flmin >= 1.0) vlam = ulam / y_sd;

            FitPathGaussian::eval(alpha_, x, y, g, w, ju, vq, x_var, cl, ne, nx,
                nlam, flmin, vlam, thr, maxit,
                lmu, a0, ca, ia, nin, rsq, alm, nlp, jerr, int_param
            );

            if (jerr > 0) return;

            calc_lambda_index(lmu, nin, ca, ia, rsq);

            for (int k = 0; k < lmu; ++k) {
                alm(k) *= y_sd;
                auto nk = nin(k);
                for (int l = 0; l < nk; ++l) {
                    ca(l,k) *= y_sd / x_sd(ia(l)-1);
                }
                a0(k)=0.0;
                if (intr) {
                    for (int i = 0; i < nk; ++i) {
                        a0(k) -= ca(i, k) * x_mean(ia(i)-1);
                    }
                    a0(k) += y_mean;
                }
            }
        }
        catch (const util::elnet_error& e) {
            jerr = e.err_code(0);
        }
    }

    void calc_lambda_index(int lmu, const Eigen::Map<Eigen::VectorXi>& nin,
                           const Eigen::Map<Eigen::MatrixXd>& ca,
                           const Eigen::Map<Eigen::VectorXi>& ia,
                           const Eigen::Map<Eigen::VectorXd>& rsq) {
        int ninmax = 0;
        for (int i = 0; i < lmu; ++i) {
            if (nin[i] > ninmax) ninmax = nin[i];
        }

        double max_rsq = -1;
        for (int k = 0; k < lmu; ++k) {
            if (rsq(k) > max_rsq) {
                m_lambda_min = k;
                max_rsq = rsq(k);
            }
        }

        m_lambda_se = m_lambda_min;
        double sd = calc_sd(rsq.data(), lmu);
        if (std::isfinite(sd)) {
            for (; m_lambda_se >= 0; m_lambda_se--) {
                if (rsq(m_lambda_se) + sd < max_rsq) break;
            }
        }

//        Eigen::MatrixXd ca_ = ca.block(0, 0, ninmax, lmu);
//        for (int k = 0; k < lmu; ++k) {
//            Eigen::VectorXd coef_ca = ca.col(k);
//            int df = 0;
//            for (int i = 0; i < ninmax; ++i) {
//                if (std::abs(coef_ca(i)) > 1e-9) df++;
//            }
//            m_dfs[k] = df;
//        }
//        printf("m_lambda_min=%d, m_lambda_se=%d\n", m_lambda_min, m_lambda_se);
    }

    double calc_sd(const double* y, size_t size) {
        double sum_y2 = 0, sum_y{0};
        size_t valid_num = 0;
        for (size_t i = 0; i < size; i++) {
            if (!std::isfinite(y[i])) continue;
            sum_y2 += y[i] * y[i];
            sum_y += y[i];
            valid_num++;
        }

        if (valid_num <= 2) return NAN;
        double mean = sum_y / valid_num;
        return sqrt((sum_y2 - mean * mean * valid_num) / (valid_num - 1.0));
    }

    double calc_sd(const std::vector<double>& y) {
        return calc_sd(y.data(), y.size());
    }

    int m_lambda_min{-1};
    int m_lambda_se{-1};
};
}



