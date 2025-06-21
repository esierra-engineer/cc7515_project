/*
 * Based on CSC materials from:
 *
 * https://github.com/csc-training/openacc/tree/master/exercises/heat
 *
 */

#include "main_CUDA.cuh"
#include "main_CPU.h"
// #include "pngwriter.h"


int main(int argc, char** argv)
{
    const int nx = 200;   // Width of the area
    const int ny = 200;   // Height of the area

    const float a = 0.5;     // Diffusion constant

    const float dx = 0.01;   // Horizontal grid spacing
    const float dy = 0.01;   // Vertical grid spacing

    const float dx2 = dx*dx;
    const float dy2 = dy*dy;

    const float dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2)); // Largest stable time step
    const int numSteps = 5000;                             // Number of time steps
    const int outputEvery = 1000;                          // How frequently to write output image

    int numElements = nx*ny;

    mainCPU(nx, ny, a, dt, numSteps, outputEvery, numElements, dx2, dy2);
    mainCUDA(nx, ny, a, dt, numSteps, outputEvery, numElements, dx2, dy2);

    return 0;
}