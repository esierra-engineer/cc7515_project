#pragma once

// Definir macro para compatibilidad C++/CUDA
#ifdef __CUDACC__
#define CUDA_HD __host__ __device__
#else
#define CUDA_HD
#endif

/* Convert 2D index layout to unrolled 1D layout
 *
 * \param[in] i      Row index
 * \param[in] j      Column index
 * \param[in] width  The width of the area
 *
 * \returns An index in the unrolled 1D array.
 */
CUDA_HD inline int getIndex(const int i, const int j, const int width) {
    return i * width + j;
}
