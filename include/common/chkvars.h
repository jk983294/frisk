#pragma once 
#include <algorithm>
#include <type_traits>

namespace frisk {

struct Chkvars
{
    template <class XType, class JUType>
    static void eval(const XType& X, JUType& ju)
    {
        using index_t = typename std::decay_t<XType>::Index;
        for (index_t j = 0; j < X.cols(); ++j) {
            ju[j] = false; 
            auto t = X.coeff(0,j);
            auto x_j_rest = X.col(j).tail(X.rows()-1);
            ju[j] = (x_j_rest.array() != t).any();
        }
    }
};

}

