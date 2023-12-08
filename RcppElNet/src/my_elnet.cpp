// [[Rcpp::depends(RcppEigen)]]

#include <RcppEigen.h>
#include <R.h>
#include <Rinternals.h>
#include <common/elnet.h>

using namespace Rcpp;

//' fit_model
//'
//' @import RcppEigen
//' @export
// [[Rcpp::export]]
List fit_model(Eigen::MatrixXd x, Eigen::VectorXd y, double alpha=1.,
             bool standardize = true, bool fit_intercept= true, int max_features = -1) {
    frisk::ElNet net;
    net.fit(std::move(x), std::move(y), alpha, 100, 1e-4, {}, standardize, fit_intercept,
            1e-7, 100000, max_features);
    return List::create(_("a0") = net.a0_, _("ca") = net.ca_, _("ia") = net.ia_,
        _("nin") = net.nin_, _("rsq") = net.rsq_, _("alm") = net.alm_, _("nlp") = net.nlp, _("jerr") = net.jerr,
        _("lmu") = net.lmu, _("pmax") = net.pmax, _("nlambda") = net.m_nlambda, _("lambda_min") = net.m_lambda_min,
        _("lambda_se") = net.m_lambda_se);
}

//' model_predict
//'
//' @param model model
//' @param newx newx
//' @param s default NAN.
//' @import RcppEigen
//' @export
// [[Rcpp::export]]
Eigen::VectorXd model_predict(List model, Eigen::Map<Eigen::MatrixXd> newx, double s = NA_REAL) {
    frisk::ElNet net;
    net.a0_ = as<std::vector<double>>(model["a0"]);
    net.ca_ = as<std::vector<double>>(model["ca"]);
    net.ia_ = as<std::vector<int>>(model["ia"]);
    net.nin_ = as<std::vector<int>>(model["nin"]);
    net.rsq_ = as<std::vector<double>>(model["rsq"]);
    net.alm_ = as<std::vector<double>>(model["alm"]);
    net.nlp = as<int>(model["nlp"]);
    net.jerr = as<int>(model["jerr"]);
    net.lmu = as<int>(model["lmu"]);
    net.pmax = as<int>(model["pmax"]);
    net.m_nlambda = as<int>(model["nlambda"]);
    net.m_lambda_min = as<int>(model["lambda_min"]);
    net.m_lambda_se = as<int>(model["lambda_se"]);
    Eigen::Map<const Eigen::MatrixXd> newx1(newx.data(), newx.rows(), newx.cols());
    return net.predict(newx1, s);
}