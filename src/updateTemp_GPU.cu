//
// Created by erick on 7/13/25.
//
#include "updateTemp.h"
#include "utils.cuh"

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

__global__ void kernel(Square* squares_t0, Square* squares_t1,
    int W, int H,
    float dx2, float dy2,
    float a, float dt){

    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i > 0 && i < W - 1)
    {
        int j = threadIdx.y + blockIdx.y*blockDim.y;
        if (j > 0 && j < H - 1)
        {
            const int index = getIndex(i, j, H);
            float uij = squares_t0[index].temp;
            float uim1j = squares_t0[getIndex(i-1, j, H)].temp;
            float uijm1 = squares_t0[getIndex(i, j-1, H)].temp;
            float uip1j = squares_t0[getIndex(i+1, j, H)].temp;
            float uijp1 = squares_t0[getIndex(i, j+1, H)].temp;

            // Explicit scheme
            squares_t1[index].temp = uij + a * dt * ( (uim1j - 2.0f*uij + uip1j)/dx2 + (uijm1 - 2.0f*uij + uijp1)/dy2 );
            squares_t0[index].temp = squares_t1[index].temp;
            float temp_factor = squares_t1[index].temp/100.0f;
            squares_t0[index].color = vec3(1.0f * temp_factor, 0.0f, 1.0f * (1 - temp_factor));
        }
    }
}

void updateTemp_GPU(Square* squares_t0, Square* squares_t1,
    int N, int W, int H,
    float dx2, float dy2,
    float a, float dt) {

    Square* squares_t0_device;
    Square* squares_t1_device;

    cudaMalloc((void**)&squares_t0_device, N*sizeof(Square));
    cudaMalloc((void**)&squares_t1_device, N*sizeof(Square));

    dim3 numBlocks(H/BLOCK_SIZE_X + 1, W/BLOCK_SIZE_Y + 1);
    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    cudaMemcpy(squares_t0_device, squares_t0, N*sizeof(Square), cudaMemcpyHostToDevice);
    cudaMemcpy(squares_t1_device, squares_t1, N*sizeof(Square), cudaMemcpyHostToDevice);

    kernel<<<numBlocks, threadsPerBlock>>>(squares_t0_device, squares_t1_device,
                                            W, H, dx2, dy2, a, dt);

    cudaMemcpy(squares_t0, squares_t0_device, N*sizeof(Square), cudaMemcpyDeviceToHost);
    cudaMemcpy(squares_t1, squares_t1_device, N*sizeof(Square), cudaMemcpyDeviceToHost);

    std::swap(squares_t0, squares_t1);
}