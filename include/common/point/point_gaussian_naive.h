#pragma once
#include <common/functional.h>
#include <common/macros.h>
#include <common/point/point_decl.h>
#include <common/point/point_gaussian_base.h>
#include <common/types.h>

namespace frisk {

template <class ElnetPointInternalPolicy>
struct ElnetPoint<
    util::glm_type::gaussian, 
    util::mode_type<util::glm_type::gaussian>::naive,
    ElnetPointInternalPolicy>
        : ElnetPointGaussianBase<
            ElnetPoint<
                util::glm_type::gaussian, 
                util::mode_type<util::glm_type::gaussian>::naive,
                ElnetPointInternalPolicy> >
{
private:
    using base_t = ElnetPointGaussianBase<
        ElnetPoint<util::glm_type::gaussian,
                   util::mode_type<util::glm_type::gaussian>::naive,
                   ElnetPointInternalPolicy> >;
    using typename base_t::update_t;
    using typename base_t::value_t;
    using typename base_t::index_t;
    using typename base_t::state_t;

public:
    using base_t::base_t;

    template <update_t upd, class PointPackType>
    GLMNETPP_STRONG_INLINE
    void update(index_t k, const PointPackType& pack)
    {
        value_t beta_diff = 0;

        auto state = base_t::template update<upd>(k, pack, beta_diff);
        if (state == state_t::continue_) return;
        
        this->update_resid(k, beta_diff);
    }
};

} // namespace frisk
