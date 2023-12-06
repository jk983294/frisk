#include <iostream>
#include <Eigen/Dense>
#include <random>

using namespace std;


int main() {
    int nobs = 10;  // Number of observations
    int nvars = 3;  // Number of predictors included in model
    int real_vars = 2;  // Number of true predictors
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

    cout << X << endl;
//    cout << y << endl;

    Eigen::VectorXd w(nobs, 1);
    w.setOnes();
    w /= w.sum();

    Eigen::VectorXd v = w.array().sqrt().matrix();
    cout << "t:\n" << w << endl;
    cout << "v:\n" << v << endl;

    return 0;
}
