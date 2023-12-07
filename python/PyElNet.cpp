#include <common/elnet.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using pydouble = py::array_t<double>;
using NumericVector = pydouble;
using NumericMatrix = pydouble;

template <typename T>
inline void check_continues(const T &x) {
    auto flags = x.flags();
    bool has_flag = false;
    if ((flags & x.c_style) == x.c_style) {
        has_flag = true;
    }
    if ((flags & x.f_style) == x.f_style) {
        has_flag = true;
    }
    if (!has_flag) {
        throw std::runtime_error("input numpy array is discontinuous, please consider copy the numpy array from pandas with copy(), or replace pandas.set_index() with pandas.sort_values()");
    }
}

struct PyElNet : public frisk::ElNet {
    void py_fit(const Eigen::Ref<const Eigen::MatrixXd>& x, const Eigen::Ref<const Eigen::VectorXd>& y, double alpha = 1.,
                int n_lambda = 100,
                std::vector<double> lambda_path = {},
                bool standardize = true, bool fit_intercept= true,
                int max_iter = 100000, int max_features = -1) {
        fit(x, y, alpha, n_lambda, 1e-4,
            lambda_path, standardize, fit_intercept, 1e-7, max_iter, max_features, {}, {});
    }

    Eigen::VectorXd py_predict(const Eigen::Ref<const Eigen::MatrixXd>& newx, double s = NAN) {
        Eigen::Map<const Eigen::MatrixXd> newx1(newx.data(), newx.rows(), newx.cols());
        return predict(newx1, s);
    }
};


PYBIND11_MODULE(pyelnet, m) {
    m.doc() = "pyelnet";

    py::class_<PyElNet>(m, "PyElNet")
        .def(py::init<>())
        .def("fit", &PyElNet::py_fit, py::arg("x"), py::arg("y"), py::arg("alpha") = 1.0,
             py::arg("n_lambda") = 100, py::arg("lambda_path") = std::vector<double>(), py::arg("standardize") = true,
             py::arg("fit_intercept") = true, py::arg("max_iter") = 100000, py::arg("max_features") = -1)
        .def("predict", &PyElNet::py_predict, py::arg("newx"), py::arg("s") = NAN)
        ;
}