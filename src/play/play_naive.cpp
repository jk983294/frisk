#include <iostream>
#include <common/elnet.h>
#include <Eigen/Dense>
#include <random>
#include <math_stats.h>

using namespace std;
using namespace frisk;


int main() {
    ElNet net;

    int nobs = 100;  // Number of observations
    int nvars = 10;  // Number of predictors included in model
    int real_vars = 5;  // Number of true predictors
    Eigen::MatrixXd X(nobs, nvars);
    Eigen::VectorXd y(nobs);

    random_device rd;
    // mt19937 generator(rd());
    mt19937 generator(42);
    normal_distribution<double> uid(0, 1);

    for (int i = 0; i < nobs; ++i) {
        double sum_ = 0;
        for (int j = 0; j < nvars; ++j) {
            double rv = uid(generator);
            X(i, j) = rv;
            if (j < real_vars) sum_ += rv;
        }
        y(i) = sum_;
    }

    Eigen::MatrixXd cl(2, nvars);
    for (int k = 0; k < nvars; ++k) {
        cl(0, k) = -INFINITY;
        cl(1, k) = INFINITY;
    }

    net.fit(X, y, 0.5);

    if (net.lmu < 1) {
        printf("an empty model has been returned; probably a convergence issue!");
    }
    int ninmax = *std::max_element(net.nin_.begin(), net.nin_.end());
    printf("%d\n", ninmax);

    auto fitted = net.predit(X);
    const Eigen::VectorXd& residual = y - fitted;
    Eigen::VectorXd ytot = y.array() - y.array().mean();
    double r_squared = 1 - residual.dot(residual) / ytot.dot(ytot);
    double pcor = ornate::corr(fitted.data(), y.data(), nobs);
    printf("r_squared=%f, pcor=%f\n", r_squared, pcor);
    return 0;
}
