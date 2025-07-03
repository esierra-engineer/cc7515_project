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

#include "Point.h"
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

__global__ void evolve_kernel(simulationConf* conf, Point* points, const float aTimesDt)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i > 0 && i < conf->nx - 1)
    {
        int j = threadIdx.y + blockIdx.y*blockDim.y;
        if (j > 0 && j < conf->ny - 1)
        {
            const int index = getIndex(i, j, conf->ny);
            float uij = conf->Un[index];
            float uim1j = conf->Un[getIndex(i-1, j, conf->ny)];
            float uijm1 = conf->Un[getIndex(i, j-1, conf->ny)];
            float uip1j = conf->Un[getIndex(i+1, j, conf->ny)];
            float uijp1 = conf->Un[getIndex(i, j+1, conf->ny)];

            // Explicit scheme
            conf->Unp1[index] = uij + aTimesDt * ( (uim1j - 2.0*uij + uip1j)/conf->dx2 + (uijm1 - 2.0*uij + uijp1)/conf->dy2 );
            points[index].T = conf->Unp1[index];
        }
    }
}

void mainCUDA(simulationConf* conf, Point* points)
{
    float* Un = conf->Un;
    float* Unp1 = conf->Unp1;
    int nx = conf->nx, ny = conf->ny, numSteps = conf->numSteps,
    outputEvery = conf->outputEvery, numElements = conf->numElements;
    float a= conf->a, dt= conf->dt;
    const char* output_filename = conf->output_filename_GPU;

    float* d_Un;
    float* d_Unp1;
    Point* d_points;
    simulationConf* d_conf;

    cudaMalloc((void**)&d_Un, numElements*sizeof(float));
    cudaMalloc((void**)&d_Unp1, numElements*sizeof(float));
    cudaMalloc((void**)&d_points, numElements*sizeof(Point));
    cudaMalloc((void**)&d_conf, 1*sizeof(simulationConf));

    dim3 numBlocks(nx/BLOCK_SIZE_X + 1, ny/BLOCK_SIZE_Y + 1);
    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    // Timing
    clock_t start = clock();


    // Main loop
    for (int n = 0; n <= numSteps; n++)
    {
        cudaMemcpy(d_Un, Un, numElements*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Unp1, Unp1, numElements*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_points, points, numElements*sizeof(Point), cudaMemcpyHostToDevice);
        conf->Un = d_Un;
        conf->Unp1 = d_Unp1;
        cudaMemcpy(d_conf, conf, 1*sizeof(simulationConf), cudaMemcpyHostToDevice);

        evolve_kernel<<<numBlocks, threadsPerBlock>>>(d_conf, d_points, a*dt);

        cudaMemcpy(Un, d_Un, numElements*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(Unp1, d_Unp1, numElements*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(points, d_points, numElements*sizeof(Point), cudaMemcpyDeviceToHost);
        cudaMemcpy(conf, d_conf, 1*sizeof(simulationConf), cudaMemcpyDeviceToHost);
        conf->Un = Un;
        conf->Unp1 = Unp1;

        // Write the output if needed
        if (n % outputEvery == 0)
        {
            cudaError_t errorCode = cudaGetLastError();
            if (errorCode != cudaSuccess)
            {
                printf("Cuda error %d: %s\n", errorCode, cudaGetErrorString(errorCode));
                exit(0);
            }
            //char filename[64];
            //sprintf(filename, "heat_%04d.png", n);
            //save_png(h_Un, nx, ny, filename, 'c');
            printArray(points, numElements, output_filename, n);
        }
        // Swapping the pointers for the next timestep
        std::swap(Un, Unp1);
    }

    // Timing
    clock_t finish = clock();
    printf("[CUDA] It took %f seconds\n", (double)(finish - start) / CLOCKS_PER_SEC);;

    cudaFree(d_Un);
    cudaFree(d_Unp1);
    cudaFree(d_points);
    cudaFree(d_conf);
}