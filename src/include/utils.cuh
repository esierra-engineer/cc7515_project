#pragma once

// Definir macro para compatibilidad C++/CUDA
#ifdef __CUDACC__
#define CUDA_HD __host__ __device__
#else
#define CUDA_HD
#endif

// La función DEBE ser inline y definida completamente aquí
CUDA_HD inline int getIndex(const int i, const int j, const int width) {
    return i * width + j;
}
