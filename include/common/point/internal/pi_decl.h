#pragma once
#include <common/types.h>

namespace frisk {

template <class ValueType
        , class IndexType
        , class BoolType>
struct ElnetPointInternalBase;

template <class ValueType
        , class IndexType
        , class BoolType>
struct ElnetPointInternalBinomialBase;

template <class ValueType
        , class IndexType
        , class BoolType>
struct ElnetPointInternalBinomialTwoClassBase;

template <class ValueType
        , class IndexType
        , class BoolType>
struct ElnetPointInternalGaussianBase;

template <class ValueType
        , class IndexType
        , class BoolType>
struct ElnetPointInternalGaussianCovBase;

template <class ValueType
        , class IndexType
        , class BoolType>
struct ElnetPointInternalGaussianNaiveBase;

template <util::glm_type g
        , util::mode_type<g> mode
        , class ValueType = double
        , class IndexType = int
        , class BoolType = bool>
struct ElnetPointInternal;

template <util::glm_type g
        , util::mode_type<g> mode
        , class ValueType = double
        , class IndexType = int
        , class BoolType = bool>
struct SpElnetPointInternal;

} // namespace frisk
