//
// Created by erick on 6/21/25.
//
/*
 * Based on CSC materials from:
 *
 * https://github.com/csc-training/openacc/tree/master/exercises/heat
 *
 */
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "utils.cuh"
#include "main_CUDA.cuh"
// #include "pngwriter.h"

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

/* Convert 2D index layout to unrolled 1D layout
 *
 * \param[in] i      Row index
 * \param[in] j      Column index
 * \param[in] width  The width of the area
 *
 * \returns An index in the unrolled 1D array.
 */
// int __host__ __device__ getIndex(const int i, const int j, const int width)
// {
//     return i*width + j;
// }

__global__ void evolve_kernel(const float* Un, float* Unp1, const int nx, const int ny, const float dx2, const float dy2, const float aTimesDt)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i > 0 && i < nx - 1)
    {
        int j = threadIdx.y + blockIdx.y*blockDim.y;
        if (j > 0 && j < ny - 1)
        {
            const int index = getIndex(i, j, ny);
            float uij = Un[index];
            float uim1j = Un[getIndex(i-1, j, ny)];
            float uijm1 = Un[getIndex(i, j-1, ny)];
            float uip1j = Un[getIndex(i+1, j, ny)];
            float uijp1 = Un[getIndex(i, j+1, ny)];

            // Explicit scheme
            Unp1[index] = uij + aTimesDt * ( (uim1j - 2.0*uij + uip1j)/dx2 + (uijm1 - 2.0*uij + uijp1)/dy2 );
        }
    }
}

void mainCUDA(int nx, int ny, float a, float dt, int numSteps, int outputEvery, int numElements, float dx2, float dy2)
{
    // Allocate two sets of data for current and next timesteps
    float* h_Un   = (float*)calloc(numElements, sizeof(float));
    float* h_Unp1 = (float*)calloc(numElements, sizeof(float));

    // Initializing the data with a pattern of disk of radius of 1/6 of the width
    float radius2 = (nx/6.0) * (nx/6.0);
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            int index = getIndex(i, j, ny);
            // Distance of point i, j from the origin
            float ds2 = (i - nx/2) * (i - nx/2) + (j - ny/2)*(j - ny/2);
            if (ds2 < radius2)
            {
                h_Un[index] = 65.0;
            }
            else
            {
                h_Un[index] = 5.0;
            }
        }
    }

    // Fill in the data on the next step to ensure that the boundaries are identical.
    memcpy(h_Unp1, h_Un, numElements*sizeof(float));

    float* d_Un;
    float* d_Unp1;

    cudaMalloc((void**)&d_Un, numElements*sizeof(float));
    cudaMalloc((void**)&d_Unp1, numElements*sizeof(float));

    dim3 numBlocks(nx/BLOCK_SIZE_X + 1, ny/BLOCK_SIZE_Y + 1);
    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    // Timing
    clock_t start = clock();

    // Main loop
    for (int n = 0; n <= numSteps; n++)
    {
        cudaMemcpy(d_Un, h_Un, numElements*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Unp1, h_Unp1, numElements*sizeof(float), cudaMemcpyHostToDevice);

        evolve_kernel<<<numBlocks, threadsPerBlock>>>(d_Un, d_Unp1, nx, ny, dx2, dy2, a*dt);

        cudaMemcpy(h_Un, d_Un, numElements*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Unp1, d_Unp1, numElements*sizeof(float), cudaMemcpyDeviceToHost);

        // Write the output if needed
        if (n % outputEvery == 0)
        {
            cudaError_t errorCode = cudaGetLastError();
            if (errorCode != cudaSuccess)
            {
                printf("Cuda error %d: %s\n", errorCode, cudaGetErrorString(errorCode));
                exit(0);
            }
            char filename[64];
            sprintf(filename, "heat_%04d.png", n);
            //save_png(h_Un, nx, ny, filename, 'c');
        }
        // Swapping the pointers for the next timestep
        std::swap(h_Un, h_Unp1);
    }

    // Timing
    clock_t finish = clock();
    printf("[CUDA] It took %f seconds\n", (double)(finish - start) / CLOCKS_PER_SEC);

    // Release the memory
    free(h_Un);
    free(h_Unp1);

    cudaFree(d_Un);
    cudaFree(d_Unp1);
}