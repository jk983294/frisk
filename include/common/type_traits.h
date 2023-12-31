#pragma once
#include <type_traits>
#include <vector>
#include <Eigen/Core>

namespace frisk {
namespace util {
namespace details {

// Dummy class that defines a conversion from F<something> to this class.
template <template <class> class F>
struct conversion_tester
{
    template <class T>
    conversion_tester (const F<T>&);
};

// Checks if From can be converted to To<something>.
template <class From, template <class> class To>
struct is_instance_of
{
    static const bool value =
        std::is_convertible<From, conversion_tester<To>>::value;
};

} // namespace details

namespace details {

// Helper meta-programming tool to initialize ju correctly.
// If BoolType is bool, use std::vector<bool> as it is more efficient.
// Otherwise, use Eigen.
template <class BoolType>
struct bvec_type
{
    using type = Eigen::Matrix<BoolType, Eigen::Dynamic, 1>;
};

template <>
struct bvec_type<bool>
{
    using type = std::vector<bool>;
};

} // namespace details

template <class T>
using bvec_t = typename details::bvec_type<T>::type;

template <class BoolType>
struct init_bvec
{
    template <class BVecType>
    static constexpr auto eval(BVecType&& bvec) {
        using bvec_t = typename details::bvec_type<BoolType>::type;
        return Eigen::Map<const bvec_t>(bvec.data(), bvec.size());
    }
};

template <>
struct init_bvec<bool>
{
    template <class BVecType>
    static constexpr const BVecType& eval(const BVecType& bvec) {
        return bvec;
    }
};

} // namespace util
}
