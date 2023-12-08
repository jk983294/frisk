#include <iostream>
#include <common/elnet.h>
#include <Eigen/Dense>
#include <random>
#include <chrono>
#include <math_stats.h>

using namespace std;
using namespace frisk;
using namespace std::chrono;

int main() {
    ElNet net;

    int nobs = 100000;  // Number of observations
    int nvars = 20;  // Number of predictors included in model
    int real_vars = 6;  // Number of true predictors

    steady_clock::time_point start = steady_clock::now();
    Eigen::MatrixXd X1(nobs, nvars);
    Eigen::VectorXd y1(nobs);
    steady_clock::time_point end = steady_clock::now();
    cout << "took " << nanoseconds{end - start}.count() / 1000 / 1000 << " ms." << endl;

    start = steady_clock::now();
    Eigen::MatrixXd X = X1;
    Eigen::VectorXd y = y1;
    end = steady_clock::now();
    cout << "took " << nanoseconds{end - start}.count() / 1000 / 1000 << " ms." << endl;

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
        y(i) = sum_ + uid(generator);
    }

    Eigen::MatrixXd cl(2, nvars);
    for (int k = 0; k < nvars; ++k) {
        cl(0, k) = -INFINITY;
        cl(1, k) = INFINITY;
    }

    start = steady_clock::now();
    net.fit(X, y, 0.5);
    end = steady_clock::now();
    cout << "took " << nanoseconds{end - start}.count() / 1000 / 1000 << " ms." << endl;

    if (net.lmu < 1) {
        printf("an empty model has been returned; probably a convergence issue!");
    }

    Eigen::Map<const Eigen::MatrixXd> mx(X.data(), X.rows(), X.cols());
    auto fitted = net.predict(mx);
    const Eigen::VectorXd& residual = y - fitted;
    Eigen::VectorXd ytot = y.array() - y.array().mean();
    double r_squared = 1 - residual.dot(residual) / ytot.dot(ytot);
    double pcor = ornate::corr(fitted.data(), y.data(), nobs);
    printf("r_squared=%f, pcor=%f\n", r_squared, pcor);
    return 0;
}
