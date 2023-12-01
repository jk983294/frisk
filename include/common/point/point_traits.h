#pragma once
#include <common/point/point_decl.h>
#include <common/type_traits.h>
#include <common/types.h>

namespace frisk {
namespace details {

template <class T> struct traits;

template <util::glm_type g
        , util::mode_type<g> m
        , class ElnetPointInternalType>
struct traits<ElnetPoint<g, m, ElnetPointInternalType> >
{
    static constexpr util::glm_type glm = g;
    static constexpr util::mode_type<g> mode = m;
    using internal_t = ElnetPointInternalType;
};
} // namespace details
} // namespace frisk
