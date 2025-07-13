cmake_minimum_required(VERSION 3.31)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES native)

project(CC7515_Project LANGUAGES CXX CUDA C)

set(CMAKE_CXX_STANDARD 20)

find_package(glfw3 3.3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glm CONFIG REQUIRED)

set(SOURCES src/Main.cpp src/shaderClass.cpp src/EBO.cpp src/glad.c
        src/VAO.cpp src/VBO.cpp src/updateTemp_CPU.cpp src/updateTemp_GPU.cu
        src/utils.cpp)
file(GLOB SHADERS "src/shaders/*.vert" "src/shaders/*.frag")

set(IMGUI_SRC
        ${CMAKE_SOURCE_DIR}/external/include/imgui/imgui.cpp
        ${CMAKE_SOURCE_DIR}/external/include/imgui/imgui_draw.cpp
        ${CMAKE_SOURCE_DIR}/external/include/imgui/imgui_tables.cpp
        ${CMAKE_SOURCE_DIR}/external/include/imgui/imgui_widgets.cpp
        ${CMAKE_SOURCE_DIR}/external/include/imgui/imgui_demo.cpp
        ${CMAKE_SOURCE_DIR}/external/include/imgui/backends/imgui_impl_glfw.cpp
        ${CMAKE_SOURCE_DIR}/external/include/imgui/backends/imgui_impl_opengl3.cpp
)

add_executable(CC7515_Project ${SOURCES} ${IMGUI_SRC})
target_include_directories(CC7515_Project PRIVATE
        include external/include src/shaders
        ${CMAKE_SOURCE_DIR}/external/include/imgui
        ${CMAKE_SOURCE_DIR}/external/include/imgui/backends)

target_link_libraries(CC7515_Project glfw OpenGL::GL glm::glm)

foreach(SHADER ${SHADERS})
    add_custom_command(TARGET CC7515_Project POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${SHADER} $<TARGET_FILE_DIR:CC7515_Project>)
endforeach()