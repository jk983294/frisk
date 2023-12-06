#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <type_traits>

namespace frisk {

struct Chkvars
{
    static void eval(const Eigen::MatrixXd& X, std::vector<bool>& ju)
    {
        for (long j = 0; j < X.cols(); ++j) {
            ju[j] = false;
            /**
             * if first row equals any row below, exclude that column
             */
            auto t = X.coeff(0,j);
            auto x_j_rest = X.col(j).tail(X.rows()-1);
            ju[j] = (x_j_rest.array() != t).any();
        }
    }
};

}

