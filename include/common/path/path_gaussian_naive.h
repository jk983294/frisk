#pragma once
#include <common/path/path_decl.h>
#include <common/path/path_gaussian_base.h>
#include <common/point/point_gaussian_naive.h>

namespace frisk {

/* 
 * Gaussian naive method elastic net path-solver.
 */
struct ElnetPath
        : ElnetPathGaussianBase
{
private:
    using base_t = ElnetPathGaussianBase;
    using elnet_point_t = ElnetPoint;

    struct FitPack
    {
        using sub_pack_t = typename base_t::FitPackGsBase;
        using value_t = typename sub_pack_t::value_t;
        using int_t = typename sub_pack_t::int_t;

        int_t& err_code() const { return sub_pack.err_code(); }
        int_t path_size() const { return sub_pack.path_size(); }

        sub_pack_t sub_pack;
        Eigen::VectorXd& y;
    };

public:
    using ElnetPathGaussianBase::process_path_fit;

    void fit(
        double beta,
        const std::vector<bool>& ju,
        const Eigen::VectorXd& vp,
        const Eigen::MatrixXd& cl,
        Eigen::VectorXd& y,
        int ne,
        int nx,
        const Eigen::MatrixXd& x,
        int nlam,
        double flmin,
        const Eigen::VectorXd& ulam,
        double thr,
        int maxit,
        const Eigen::VectorXd& x_var,
        int& lmu,
        Eigen::Map<Eigen::MatrixXd>& ao,
        Eigen::Map<Eigen::VectorXi>& ia,
        Eigen::Map<Eigen::VectorXi>& kin,
        Eigen::Map<Eigen::VectorXd>& rsqo,
        Eigen::Map<Eigen::VectorXd>& almo,
        int& nlp,
        int& jerr,
        const InternalParams& int_param) const
    {
        FitPack pack
        {
            // build sub-pack
            {
                // build sub-pack
                {beta, ju, vp, cl, ne, nx, x, nlam, flmin,
                 ulam, thr, maxit, lmu, ao, ia, kin, almo, nlp, jerr, int_param},
                // add new members
             x_var, rsqo
            }, 
            // add new members
            y
        };
        fit(pack);
    }

    void fit(const FitPack& pack) const;

    ElnetPoint get_elnet_point(const FitPack& pack, const PathConfigPackBase&) const;

    PathConfigPackBase initialize_path(const FitPack& pack) const
    {
        return ElnetPathGaussianBase::initialize_path(pack.sub_pack);
    }

    PointConfigPackBase initialize_point(
        int m,
        double& lmda_curr,
            const FitPack& pack,
            const PathConfigPackBase& path_pack,
            const ElnetPoint& elnet_point) const
    {
        return base_t::initialize_point(m, lmda_curr, pack.sub_pack, path_pack, elnet_point.abs_grad());
    }

    template <class FitPackType
            , class PointConfigPackType
            , class PathConfigPackType
            , class ElnetPointType>
    state_t process_point_fit(
            const FitPackType& pack, 
            const PathConfigPackType& path_pack,
            const PointConfigPackType& point_pack,
            const ElnetPointType& elnet_point) const
    {
        return base_t::process_point_fit(pack.sub_pack, path_pack, point_pack, elnet_point);
    }
};

}
