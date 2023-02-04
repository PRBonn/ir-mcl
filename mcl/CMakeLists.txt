cmake_minimum_required(VERSION 3.16)
project(pf_library)

# Set build type
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED)

add_subdirectory(srrg_utils/pf_library/pybind)
