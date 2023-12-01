#pragma once
#include <common/point/point_decl.h>
#include <common/types.h>

namespace frisk {

template <class ElnetPathDerived>
struct ElnetPathCRTPBase;

struct ElnetPathBase;
struct ElnetPathGaussianBase;
struct ElnetPathBinomialBase;

template <util::glm_type glm
        , util::mode_type<glm> mode
        , class ElnetPointPolicy=ElnetPoint<glm, mode> >
struct ElnetPath;

}
