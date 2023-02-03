add_library(pf_library pf_library.cpp)
target_link_libraries(pf_library Eigen3::Eigen)
pybind11_add_module(pf_library_pybind pf_library_pybind.cpp)
target_link_libraries(pf_library_pybind PRIVATE pf_library)
