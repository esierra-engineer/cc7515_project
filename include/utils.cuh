//
// Created by erick on 7/13/25.
//
#pragma once

#ifndef UTILS_H
#define UTILS_H
#include "shaderClass.h"
#include "Square.h"
#include "glm/glm.hpp"

// Definir macro para compatibilidad C++/CUDA
#ifdef __CUDACC__
#define CUDA_HD __host__ __device__
#else
#define CUDA_HD
#endif

using namespace glm;

void initDisk(Square* squares, int W, int H, float dx, float dy, float Height, float Width,
              float radius2, float Tin, float Tout, vec<2, int>* center_coordinates);

CUDA_HD inline int getIndex(const int i, const int j, const int width) {
    return i * width + j;
}

void drawSquares(Square* squares, Shader* shaderProgram, int N);

#endif //UTILS_H
