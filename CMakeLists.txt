cmake_minimum_required(VERSION 3.31)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES native)

project(cc7515_project)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)
include_directories(src)
include_directories(src/include)

set(SOURCES
        src/main_CUDA.cu
        src/main_CPU.cpp
        src/main.cpp
)

add_executable(cc7515_project
        ${SOURCES})
