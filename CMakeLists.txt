cmake_minimum_required(VERSION 3.31)
project(cc7515_project)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)
include_directories(src)
include_directories(src/include)

find_package(PNGwriter 0.7.0)

add_executable(cc7515_project
        src/main.cpp src/pngwriter.c)

if(PNGwriter_FOUND)
    target_link_libraries(cc7515_project PRIVATE PNGwriter::PNGwriter)
endif(PNGwriter_FOUND)