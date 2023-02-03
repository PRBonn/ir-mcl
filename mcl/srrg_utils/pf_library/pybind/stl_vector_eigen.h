#pragma once
#include <Eigen/Core>
#include <vector>

// pollute namespace with py
#include <pybind11/pybind11.h>
namespace py = pybind11;
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace pybind11 {

template <typename Vector, typename holder_type = std::unique_ptr<Vector>, typename... Args>
py::class_<Vector, holder_type> bind_vector_without_repr(py::module &m,
                                                         std::string const &name,
                                                         Args &&...args) {
    // hack function to disable __repr__ for the convenient function
    // bind_vector()
    using Class_ = py::class_<Vector, holder_type>;
    Class_ cl(m, name.c_str(), std::forward<Args>(args)...);
    cl.def(py::init<>());
    cl.def(
        "__bool__", [](const Vector &v) -> bool { return !v.empty(); },
        "Check whether the list is nonempty");
    cl.def("__len__", &Vector::size);
    return cl;
}

// - This function is used by Pybind for std::vector<SomeEigenType> constructor.
//   This optional constructor is added to avoid too many Python <-> C++ API
//   calls when the vector size is large using the default biding method.
//   Pybind matches np.float64 array to py::array_t<double> buffer.
// - Directly using templates for the py::array_t<double> and py::array_t<int>
//   and etc. doesn't work. The current solution is to explicitly implement
//   bindings for each py array types.
template <typename EigenVector>
std::vector<EigenVector> py_array_to_vectors_double(
    py::array_t<double, py::array::c_style | py::array::forcecast> array) {
    int64_t eigen_vector_size = EigenVector::SizeAtCompileTime;
    if (array.ndim() != 2 || array.shape(1) != eigen_vector_size) {
        throw py::cast_error();
    }
    std::vector<EigenVector> eigen_vectors(array.shape(0));
    auto array_unchecked = array.mutable_unchecked<2>();
    for (auto i = 0; i < array_unchecked.shape(0); ++i) {
        eigen_vectors[i] = Eigen::Map<EigenVector>(&array_unchecked(i, 0));
    }
    return eigen_vectors;
}

template <typename EigenVector>
std::vector<EigenVector> py_array_to_vectors_int(
    py::array_t<int, py::array::c_style | py::array::forcecast> array) {
    int64_t eigen_vector_size = EigenVector::SizeAtCompileTime;
    if (array.ndim() != 2 || array.shape(1) != eigen_vector_size) {
        throw py::cast_error();
    }
    std::vector<EigenVector> eigen_vectors(array.shape(0));
    auto array_unchecked = array.mutable_unchecked<2>();
    for (auto i = 0; i < array_unchecked.shape(0); ++i) {
        eigen_vectors[i] = Eigen::Map<EigenVector>(&array_unchecked(i, 0));
    }
    return eigen_vectors;
}

}  // namespace pybind11

template <typename EigenVector,
          typename Vector = std::vector<EigenVector>,
          typename holder_type = std::unique_ptr<Vector>,
          typename InitFunc>
py::class_<Vector, holder_type> pybind_eigen_vector_of_vector(py::module &m,
                                                              const std::string &bind_name,
                                                              const std::string &repr_name,
                                                              InitFunc init_func) {
    using Scalar = typename EigenVector::Scalar;
    auto vec = py::bind_vector_without_repr<std::vector<EigenVector>>(
        m, bind_name, py::buffer_protocol(), py::module_local());
    vec.def(py::init(init_func));
    vec.def_buffer([](std::vector<EigenVector> &v) -> py::buffer_info {
        size_t rows = EigenVector::RowsAtCompileTime;
        return py::buffer_info(v.data(), sizeof(Scalar), py::format_descriptor<Scalar>::format(), 2,
                               {v.size(), rows}, {sizeof(EigenVector), sizeof(Scalar)});
    });
    vec.def("__repr__", [repr_name](const std::vector<EigenVector> &v) {
        return repr_name + std::string(" with ") + std::to_string(v.size()) +
               std::string(" elements.\n") + std::string("Use numpy.asarray() to access data.");
    });
    vec.def("__copy__", [](std::vector<EigenVector> &v) { return std::vector<EigenVector>(v); });
    vec.def("__deepcopy__",
            [](std::vector<EigenVector> &v) { return std::vector<EigenVector>(v); });

    // py::detail must be after custom constructor
    using Class_ = py::class_<Vector, std::unique_ptr<Vector>>;
    py::detail::vector_if_copy_constructible<Vector, Class_>(vec);
    py::detail::vector_if_equal_operator<Vector, Class_>(vec);
    py::detail::vector_modifiers<Vector, Class_>(vec);
    py::detail::vector_accessor<Vector, Class_>(vec);

    return vec;
}

