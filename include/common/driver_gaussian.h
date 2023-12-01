#pragma once
#include <common/chkvars.h>
#include <common/standardize.h>
#include <common/type_traits.h>
#include <common/types.h>
#include <common/path/path_base.h>
#include <common/path/path_gaussian_cov.h>
#include <common/path/path_gaussian_naive.h>
#include <common/point/internal/pi_gaussian_cov.h>
#include <common/point/internal/pi_gaussian_naive.h>
#include <common/point/point_gaussian_cov.h>
#include <common/point/point_gaussian_naive.h>
#include <Eigen/Core>
#include <vector>

namespace frisk {

struct FitPathGaussian
{
    static void eval(
            bool is_cov,
            double parm,
            const Eigen::MatrixXd& x,
            Eigen::VectorXd& y,
            Eigen::VectorXd& g,
            const Eigen::VectorXd& w,
            const std::vector<bool>& ju,
            const Eigen::VectorXd& vq,
            const Eigen::VectorXd& xm,
            const Eigen::VectorXd& xs,
            const Eigen::VectorXd& xv,
            const Eigen::MatrixXd& cl,
            int ne,
            int nx,
            int nlam,
            double flmin,
            const Eigen::VectorXd& vlam,
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
            )
    {
        constexpr util::glm_type glm = util::glm_type::gaussian;
        using mode_t = util::mode_type<glm>;

        // cov method
        if (is_cov) {
            ElnetPathCov elnet_path;
            elnet_path.fit(
                    parm, ju, vq, cl, g, ne, nx, x, nlam, flmin, vlam, thr, maxit, xv,
                    lmu, ca, ia, nin, rsq, alm, nlp, jerr, int_param);
        }
        // naive method
        else {
            ElnetPath<glm, mode_t::naive> elnet_path;
            elnet_path.fit(
                    parm, ju, vq, cl, y, ne, nx, x, nlam, flmin, vlam, thr, maxit, xv,
                    lmu, ca, ia, nin, rsq, alm, nlp, jerr, int_param);
        }
    }
};

struct ElnetDriverBase
{
    template <class VType>
    void normalize_penalty(VType&& vq) const {
        if (vq.maxCoeff() <= 0) throw util::non_positive_penalty_error();
        vq.array() = vq.array().max(0.0);
        vq *= vq.size() / vq.sum();
    }

    template <class JDType, class JUType>
    void init_inclusion(const JDType& jd, JUType&& ju) const {
        if (jd(0) > 0) {
            for (int i = 1; i < jd(0) + 1; ++i) {
                ju[jd(i)-1] = false;
            }
        }
        // can't find true value in ju
        if (std::find_if(ju.begin(), ju.end(), [](auto x) { return x;}) == ju.end()) {
            throw util::all_excluded_error();
        }
    }
};

template <util::glm_type glm>
struct ElnetDriver;

template <>
struct ElnetDriver<util::glm_type::gaussian>
    : ElnetDriverBase
{
private:
    static constexpr util::glm_type glm = util::glm_type::gaussian;
    using mode_t = util::mode_type<glm>;

public:

    void fit(
        bool is_cov,
        double parm,
        Eigen::MatrixXd& x,
        Eigen::VectorXd& y,
        Eigen::VectorXd& w,
        const Eigen::Map<Eigen::VectorXi>& jd,
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
    ) const
    {
        using chkvars_t = Chkvars;
        using standardize_cov_t = Standardize;
        using standardize_naive_t = Standardize1;

        try {
            Eigen::VectorXd vq = vp;
            this->normalize_penalty(vq);

            auto ni = x.cols();

            Eigen::VectorXd g;            // only naive version uses it
            Eigen::VectorXd xm(ni); xm.setZero();
            Eigen::VectorXd xs(ni); xs.setZero();
            Eigen::VectorXd xv(ni); xv.setZero();
            Eigen::VectorXd vlam(nlam); vlam.setZero();
            std::vector<bool> ju(ni, false);

            chkvars_t::eval(x, ju);
            this->init_inclusion(jd, ju);

            double ym = 0;
            double ys = 0;

            // cov method
            if (is_cov) {
                g.setZero(ni);
                standardize_cov_t::eval(x, y, w, isd, intr, ju, g, xm, xs, ym, ys, xv);
            }

                // naive method
            else {
                standardize_naive_t::eval(x, y, w, isd, intr, ju, xm, xs, ym, ys, xv);
            }

            cl /= ys;
            if (isd) {
                for (int j = 0; j < ni; ++j) {
                    cl.col(j) *= xs(j);
                }
            }

            if (flmin >= 1.0) vlam = ulam / ys;

            FitPathGaussian::eval(
                is_cov, parm, x, y, g, w, ju, vq, xm, xs, xv, cl, ne, nx,
                nlam, flmin, vlam, thr, isd, intr, maxit,
                lmu, a0, ca, ia, nin, rsq, alm, nlp, jerr, int_param
            );

            if (jerr > 0) return;

            for (int k = 0; k < lmu; ++k) {
                alm(k) *= ys;
                auto nk = nin(k);
                for (int l = 0; l < nk; ++l) {
                    ca(l,k) *= ys / xs(ia(l)-1);
                }
                a0(k)=0.0;
                if (intr) {
                    for (int i = 0; i < nk; ++i) {
                        a0(k) -= ca(i, k) * xm(ia(i)-1);
                    }
                    a0(k) += ym;
                }
            }
        }
        catch (const util::elnet_error& e) {
            jerr = e.err_code(0);
        }
    }
};
}



