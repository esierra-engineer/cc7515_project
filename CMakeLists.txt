cmake_minimum_required(VERSION 3.31)
project(cc7515_project)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)
include_directories(src)
include_directories(src/include)

add_executable(cc7515_project
        src/main.cpp)
