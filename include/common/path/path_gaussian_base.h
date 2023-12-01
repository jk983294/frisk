#pragma once
#include <common/path/path_base.h>
#include <common/types.h>
#include <Eigen/Core>
#include <algorithm>
#include <limits>
#include <type_traits>

namespace frisk {

/*
 * Common routines across all Gaussian path-solvers.
 */
struct ElnetPathGaussianBase
    : ElnetPathBase
{
private:
    using base_t = ElnetPathBase;

protected:
    using typename base_t::state_t;
    using base_t::process_point_fit;

    /* 
     * Common FitPack base class for all Gaussian path-solvers.
     */
    struct FitPackGsBase {
        using sub_pack_t = typename ElnetPathBase::FitPackBase;
        using value_t = double;
        using int_t = int;

        GLMNETPP_STRONG_INLINE int_t& err_code() const { return sub_pack.err_code(); }
        GLMNETPP_STRONG_INLINE int_t path_size() const { return sub_pack.path_size(); }

        FitPackBase sub_pack;
        const Eigen::VectorXd& xv;
        Eigen::Map<Eigen::VectorXd>& rsqo;
    };

    /*
     * Delegate to base class method with the base pack.
     */
    GLMNETPP_STRONG_INLINE
    PathConfigPackBase initialize_path(const FitPackGsBase& pack) const
    {
        return ElnetPathBase::initialize_path(pack.sub_pack);
    }

    GLMNETPP_STRONG_INLINE
    PointConfigPackBase initialize_point(
            int m, 
            double& lmda_curr,
            const FitPackGsBase& pack,
            const PathConfigPackBase& path_pack,
            const Eigen::VectorXd& g) const
    {
        return base_t::initialize_point(m, lmda_curr, pack.sub_pack, path_pack, g);
    }

    /*
     * Common routine for all Gaussian path-solvers after point-solver fit.
     * See fit() in base.hpp for usage.
     *
     * @param   pack        object of FitPack of current class.
     * @param   path_pack   object of PathConfigPack of current class.
     * @param   point_pack  object of PointConfigPack of current class.
     * @param   elnet_point point-solver object.
     */
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
        using int_t = typename std::decay_t<PointConfigPackType>::int_t;
        using value_t = typename std::decay_t<PointConfigPackType>::value_t;

        auto& sp = pack.sub_pack;

        auto& ao = sp.ao;
        auto& rsqo = pack.rsqo;
        auto m = point_pack.m;
        auto n_active = elnet_point.n_active();
        auto rsq = elnet_point.rsq();
        auto rsq0 = elnet_point.rsq_prev();

        base_t::store_beta_compressed(
                elnet_point.active_begin(), elnet_point.active_end(),
                ao.col(m), [&](int_t k) { return elnet_point.beta(k); } );
        rsqo(m) = rsq;

        int_t me = (ao.col(m).head(n_active).array() != 0).count();
        auto prop_dev_change = (rsq == 0) ? 
            std::numeric_limits<value_t>::infinity() : 
            (rsq - rsq0) / rsq;

        state_t state = base_t::process_point_fit(
                m, n_active, me, prop_dev_change, rsq, sp, path_pack, point_pack
                );
        if (state == state_t::continue_ || 
            state == state_t::break_) return state;

        return state_t::noop_;
    }

    /*
     * Common finishing routine for all Gaussian path-solvers after (path) fit.
     * See fit() in base.hpp for usage.
     */
    template <class FitPackType, class ElnetPointType>
    GLMNETPP_STRONG_INLINE
    constexpr void process_path_fit(const FitPackType&, const ElnetPointType&) const {}
};

}
