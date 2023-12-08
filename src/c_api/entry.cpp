#include <common/elnet.h>

extern "C" {
__attribute__((visibility("default"))) void* get_model() {
    return new frisk::ElNet;
}
__attribute__((visibility("default"))) void free_model(void* model) {
    auto net = reinterpret_cast<frisk::ElNet*>(model);
    delete net;
}

__attribute__((visibility("default"))) int fit(void* model, const double* x, const double* y, long nobs,
                                                 int nvars, double alpha, int n_lambda,
                                                 const double* lambda_path, int lambda_path_len,
                                                 bool standardize, bool fit_intercept,
                                                 int max_iter, int max_features) {
    /*
    printf("m=%p, x=%p, y=%p, nobs=%ld, nvars=%d,%f,%d,%p,%d,%d,%d,%d,%d\n",
           model, x, y, nobs, nvars, alpha, n_lambda, lambda_path, lambda_path_len, standardize,
           fit_intercept, max_iter, max_features);
           */
    auto net = reinterpret_cast<frisk::ElNet*>(model);
    net->fit1(x, y, nobs, nvars, alpha, n_lambda, lambda_path, lambda_path_len, standardize, fit_intercept, max_iter, max_features);
    //printf("fit m_lambda_se=%d, m_lambda_min=%d\n", net->m_lambda_se, net->m_lambda_min);
    return net->jerr;
}

__attribute__((visibility("default"))) double* predict(void* model, const double* x, long nobs,
                                                    int nvars, double s) {
    auto net = reinterpret_cast<frisk::ElNet*>(model);
    Eigen::Map<const Eigen::MatrixXd> mx(x, nobs, nvars);
    Eigen::VectorXd res = net->predict(mx, s);
    double * ret = new double[nobs];
    std::copy(res.data(), res.data() + nobs, ret);
    // printf("predict m=%p, x=%p, nobs=%ld, nvars=%d\n", model, x, nobs, nvars);
    return ret;
}

__attribute__((visibility("default"))) double* get_coef(void* model, double s) {
    auto net = reinterpret_cast<frisk::ElNet*>(model);
    double intercept = NAN;
    Eigen::VectorXd res = net->get_coef(intercept, s);
    double * ret = new double[res.size() + 1];
    ret[0] = intercept;
    std::copy(res.data(), res.data() + res.size(), ret + 1);
    return ret;
}

__attribute__((visibility("default"))) void free_data(double* y) {
    delete []y;
}
}