#pragma once

#include <common/exceptions.h>
#include <common/type_traits.h>

namespace frisk {


struct InternalParams
{
    static constexpr double sml = 1e-5;
    static constexpr double eps = 1e-6;
    static constexpr double big = 9.9e35;
    static constexpr int mnlam = 5;
    static constexpr double rsqmax = 0.999;
    static constexpr double pmin = 1e-9;
    static constexpr double exmx = 250.0;
    static constexpr int itrace = 0;
    static constexpr double bnorm_thr = 1e-10;
    static constexpr int bnorm_mxit = 100;
};

namespace util {
enum class glm_type { gaussian };

// A struct that defines the enum class of mode types
// for a specific glm.
template <glm_type glm>
struct Mode;

// Specializations
template <>
struct Mode<glm_type::gaussian> {
    enum class type { naive, cov };
};

// Helper alias to get the mode enum class type for each glm.
template <glm_type glm>
using mode_type = typename Mode<glm>::type;

// Represents a state of a function as a way for the caller to do control flow.
enum class control_flow
{
    noop_,
    continue_,
    break_,
    return_
};

enum class update_type
{
    full,
    partial
};
}
}
