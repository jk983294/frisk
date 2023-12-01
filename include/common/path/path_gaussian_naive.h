#pragma once
#include <common/path/path_decl.h>
#include <common/path/path_gaussian_base.h>

namespace frisk {

/* 
 * Gaussian naive method elastic net path-solver.
 */
template <class ElnetPointPolicy>
struct ElnetPath<
    util::glm_type::gaussian,
    util::mode_type<util::glm_type::gaussian>::naive,
    ElnetPointPolicy>
        : ElnetPathGaussianBase
{
private:
    using base_t = ElnetPathGaussianBase;
    using elnet_point_t = ElnetPointPolicy;

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
        FitPack pack
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
            y
        };
        fit(pack);
    }


    void fit(const FitPack& pack) const
    {
        using value_t = double;
        using int_t = int;

        auto& jerr = pack.err_code();

        try {
            auto&& path_config_pack = initialize_path(pack);

            auto&& elnet_point = get_elnet_point(pack, path_config_pack);

            value_t lmda_curr = 0; // this makes the math work out in the point solver

            for (int_t m = 0; m < pack.path_size(); ++m) {

                PointConfigPackBase&& point_config_pack =
                    initialize_point(m, lmda_curr, pack, path_config_pack, elnet_point);

                try {
                    elnet_point.fit(point_config_pack);
                }
                catch (const util::maxit_reached_error& e) {
                    jerr = e.err_code(m);
                    return;
                }
                catch (const util::bnorm_maxit_reached_error& e) {
                    jerr = e.err_code(m);
                    return;
                }
                catch (const util::elnet_error& e) {
                    jerr = e.err_code(m);
                    break;
                }

                state_t state = process_point_fit(pack, path_config_pack, point_config_pack, elnet_point);

                if (state == state_t::continue_) continue;
                if (state == state_t::break_) break;
            }

            process_path_fit(pack, elnet_point);
        }
        catch (const util::elnet_error& e) {
            jerr = e.err_code(0);
        }
    }



    template <class FitPackType, class PathConfigPackType>
    auto get_elnet_point(const FitPackType& pack, const PathConfigPackType&) const 
    {
        auto& sp = pack.sub_pack;
        auto& ssp = sp.sub_pack;
        return elnet_point_t(
                ssp.thr, ssp.maxit, ssp.nx, ssp.nlp, ssp.ia, pack.y, ssp.x, 
                sp.xv, ssp.vp, ssp.cl, ssp.ju);
    }

    template <class FitPackType>
    auto initialize_path(const FitPackType& pack) const 
    {
        return base_t::initialize_path(pack.sub_pack);
    }

    template <class FitPackType
            , class PathConfigPackType
            , class ElnetPointType>
    PointConfigPackBase initialize_point(
        int m,
        double& lmda_curr,
            const FitPackType& pack, 
            const PathConfigPackType& path_pack,
            const ElnetPointType& elnet_point) const
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
