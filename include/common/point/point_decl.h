#pragma once
#include <common/point/internal/pi_decl.h>
#include <common/types.h>

namespace frisk {

template <class ElnetPointDerived>
struct ElnetPointCRTPBase;

template <class Derived>
struct ElnetPointGaussianBase;

template <util::glm_type glm
        , util::mode_type<glm> mode
        , class ElnetPointInternalPolicy=ElnetPointInternal<glm, mode> >
struct ElnetPoint;


}
