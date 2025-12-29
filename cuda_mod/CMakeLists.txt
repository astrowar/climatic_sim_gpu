cmake_minimum_required(VERSION 3.18)
project(cuda_fem_solver LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 11)

# Output directory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)


# Build shared library for Python (using pybind11 via FetchContent)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Baixar pybind11 automaticamente se não estiver disponível
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# CUDA sources

# Adiciona climatic_kernels.cu como fonte CUDA
set(SOURCES
    src/cuda_fem_solver.cu
    src/pybind_wrapper.cpp
    src/climatic_kernels.cu
)

add_library(cuda_fem_solver SHARED ${SOURCES})

target_include_directories(cuda_fem_solver PRIVATE ${Python3_INCLUDE_DIRS} ${pybind11_INCLUDE_DIRS} src)
target_link_libraries(cuda_fem_solver PRIVATE Python3::Python pybind11::module)

set_target_properties(cuda_fem_solver PROPERTIES PREFIX "" SUFFIX ".so")

# Enable CUDA
set_target_properties(cuda_fem_solver PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# RPATH for local import
set_target_properties(cuda_fem_solver PROPERTIES BUILD_RPATH "$ORIGIN")
