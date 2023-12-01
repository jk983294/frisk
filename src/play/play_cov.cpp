#include <iostream>
#include <common/elnet.h>
#include <Eigen/Dense>
#include <random>

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
    mt19937 generator(rd());
    // mt19937 generator(43);
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

    double alpha = 0.5;
    Eigen::VectorXd weights(nobs, 1);
    weights.setOnes();
    std::vector<int> jd_ = {0};  // exclude
    Eigen::Map<Eigen::VectorXi> jd(jd_.data(), jd_.size());
    std::vector<double> penalty_factor_(nvars, 1.);
    Eigen::Map<Eigen::VectorXd> vp(penalty_factor_.data(), penalty_factor_.size());
    int dfmax = nvars + 1;
    int pmax= std::min(dfmax*2+20, nvars);
    int nlambda = 100;
    double lambda_min_ratio= 1e-4;
    if(nobs < nvars) lambda_min_ratio = 1e-2;
    std::vector<double> ulam_ = {1};
    Eigen::Map<Eigen::VectorXd> ulam(ulam_.data(), ulam_.size());
    double thresh=1e-7;
    int isd = 1;
    int intr = 1;
    int maxit = 100000;
    std::vector<double> a0_(nlambda, 0.);
    Eigen::Map<Eigen::VectorXd> a0(a0_.data(), a0_.size());
    std::vector<double> ca_(nobs * nlambda, 0);
    Eigen::Map<Eigen::MatrixXd> ca(ca_.data(), nobs, nlambda);
    std::vector<int> ia_(pmax, 0);
    Eigen::Map<Eigen::VectorXi> ia(ia_.data(), ia_.size());
    std::vector<int> nin_(nlambda, 0);
    Eigen::Map<Eigen::VectorXi> nin(nin_.data(), nin_.size());
    std::vector<double> rsq_(nlambda, 0.);
    Eigen::Map<Eigen::VectorXd> rsq(rsq_.data(), rsq_.size());
    std::vector<double> alm_(nlambda, 0.);
    Eigen::Map<Eigen::VectorXd> alm(alm_.data(), alm_.size());
    int nlp = 0, jerr = 0, lmu = 0;
    net.elnet_exp(true, alpha, X, y, weights, jd, vp, cl,
                  dfmax, pmax, nlambda, lambda_min_ratio, ulam, thresh, isd, intr, maxit, lmu,
                  a0, ca, ia, nin, rsq, alm, nlp, jerr);

    if (lmu < 1) {
        printf("an empty model has been returned; probably a convergence issue!");
    }
    int ninmax = *std::max_element(nin_.begin(), nin_.end());
    printf("%d\n", ninmax);
//    cout << X << endl;
//    cout << y << endl;

    return 0;
}
