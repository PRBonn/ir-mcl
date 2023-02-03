#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "pf_library.h"
#include "stl_vector_eigen.h"
#include <eigen3/Eigen/Core>
#include <vector>
namespace py = pybind11;
using namespace py::literals;

PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector4d>);
namespace pf_library {
PYBIND11_MODULE(pf_library_pybind, m) {
  auto vector3dvector = pybind_eigen_vector_of_vector<Eigen::Vector4d>(
      m, "_Vector4dVector", "std::vector<Eigen::Vector4d>",
      py::py_array_to_vectors_double<Eigen::Vector4d>);

  py::class_<MotionModelAndResample> pf(m, "_MotionModelAndResampling",
                                        "Don't use this");
  pf.def(py::init<>())
      .def("_predict", &MotionModelAndResample::predict, "particles"_a,
           "control"_a)
      .def("_resample_uniform", &MotionModelAndResample::resample_uniform,
           "weights"_a);
}
} // namespace pf_library
