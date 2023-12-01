#pragma once
#include <common/path/path_decl.h>
#include <common/path/path_gaussian_base.h>

namespace frisk {

/* 
 * Gaussian covariance method elastic net path-solver.
 */
template <class ElnetPointPolicy>
struct ElnetPath<
    util::glm_type::gaussian,
    util::mode_type<util::glm_type::gaussian>::cov,
    ElnetPointPolicy>
        : ElnetPathGaussianBase
        , ElnetPathCRTPBase<
            ElnetPath<
                util::glm_type::gaussian,
                util::mode_type<util::glm_type::gaussian>::cov,
                ElnetPointPolicy> >
{
private:
    using base_t = ElnetPathGaussianBase;
    using crtp_base_t = ElnetPathCRTPBase<
            ElnetPath<util::glm_type::gaussian,
                      util::mode_type<util::glm_type::gaussian>::cov,
                      ElnetPointPolicy> >;
    using elnet_point_t = ElnetPointPolicy;

    struct FitPackCov {
        using sub_pack_t = typename base_t::FitPackGsBase;
        using value_t = typename sub_pack_t::value_t;
        using int_t = typename sub_pack_t::int_t;

        int_t& err_code() const { return sub_pack.err_code(); }
        int_t path_size() const { return sub_pack.path_size(); }

        FitPackGsBase sub_pack;
        Eigen::VectorXd& g;
    };

public:
    using ElnetPathGaussianBase::process_path_fit;

    void fit(
        double beta,
        const std::vector<bool>& ju,
        const Eigen::VectorXd& vp,
        const Eigen::MatrixXd& cl,
        Eigen::VectorXd& g,
        int ne,
        int nx,
        const Eigen::MatrixXd& x,
        int nlam,
        double flmin,
        const Eigen::VectorXd& ulam,
        double thr,
        int maxit,
        const Eigen::VectorXd& xv,
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
        FitPackCov pack
        {
            // build sub-pack
            {
                // build sub-pack
                {beta, ju, vp, cl, ne, nx, x, nlam, flmin,
                 ulam, thr, maxit, lmu, ao, ia, kin, almo, nlp, jerr, int_param},
                // add new members
                xv, rsqo
            }, 
            // add new members
            g
        };
        crtp_base_t::fit(pack);
    }

    /*
     * Builds a point-solver using the arguments.
     *
     * @param   pack        object of FitPack of the current class.
     */
    template <class FitPackType, class PathConfigPackType>
    auto get_elnet_point(const FitPackType& pack, const PathConfigPackType&) const
    {
        auto& sp = pack.sub_pack;
        auto& ssp = sp.sub_pack;
        return elnet_point_t(
                ssp.thr, ssp.maxit, ssp.nx, ssp.nlp, ssp.ia, pack.g, ssp.x, 
                sp.xv, ssp.vp, ssp.cl, ssp.ju);
    }

    template <class FitPackType>
    auto initialize_path(const FitPackType& pack) const
    {
        return ElnetPathGaussianBase::initialize_path(pack.sub_pack);
    }

    template <class FitPackType
        , class PathConfigPackType
        , class ElnetPointType>
    PointConfigPackBase initialize_point(
            int m, 
            double& lmda_curr,
            const FitPackType& pack,
            const PathConfigPackType& path_pack,
            const ElnetPointType&) const
    {
        return base_t::initialize_point(m, lmda_curr, pack.sub_pack, path_pack, pack.g);
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
        return ElnetPathGaussianBase::process_point_fit(pack.sub_pack, path_pack, point_pack, elnet_point);
    }
};

}
