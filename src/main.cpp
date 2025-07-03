/*
 * Based on CSC materials from:
 *
 * https://github.com/csc-training/openacc/tree/master/exercises/heat
 *
 */

#include <cstdlib>
#include <ctime>
#include <iosfwd>
#include <iostream>
#include <fstream>
#include <numeric>
#include <vector>

#include "main_CUDA.cuh"
#include "main_CPU.h"
#include "Point.h"
// #include "pngwriter.h"


int main(int argc, char** argv)
{
    simulationConf* conf = new simulationConf;

    conf->a = 0.5;     // Diffusion constant

    const float dx = 0.01;   // Horizontal grid spacing
    const float dy = 0.01;   // Vertical grid spacing

    conf->dx2 = dx*dx;
    conf->dy2 = dy*dy;

    conf->dt = conf->dx2 * conf->dy2 / (2.0 * conf->a * (conf->dx2 + conf->dy2)); // Largest stable time step
    conf->numSteps = 5000;                             // Number of time steps
    conf->outputEvery = 100;                          // How frequently to write output image

    conf->output_filename_CPU = "/media/storage/git/cc7515_project/output/outputCPU.csv";
    conf->output_filename_GPU = "/media/storage/git/cc7515_project/output/outputGPU.csv";
    conf->Tin = 100.0f;
    conf->Tout = 000.0f;

    std::ofstream cpuFile("/media/storage/git/cc7515_project/output/benchmarkCPU.csv");
    std::ofstream gpuFile("/media/storage/git/cc7515_project/output/benchmarkGPU.csv");

    if (!(gpuFile.is_open() && cpuFile.is_open())) std::cerr << "Error: Unable to open file for writing.\n";

    cpuFile << "engine,size,time\n";
    gpuFile << "engine,size,time\n";

    for (int sim = 300 ; sim <= 300 ; sim += 100) {
        conf->nx = sim;   // Width of the area
        conf->ny = sim;   // Height of the area

        conf->numElements = conf->nx*conf->ny;

        // Allocate two sets of data for current and next timesteps
        float* Un   = (float*)calloc(conf->numElements, sizeof(float));
        float* Unp1 = (float*)calloc(conf->numElements, sizeof(float));

        conf->Un = Un; conf->Unp1 = Unp1;

        Point* points = (Point*)malloc(conf->numElements * sizeof(Point));

        initDisk(conf, conf->nx/10.0, points);

        clock_t start = clock();
        mainCPU(conf, points);
        clock_t finish = clock();

        char cpu_obuf[64];
        sprintf(cpu_obuf, "CPU,%d,%f\n", sim, (double)(finish - start) / CLOCKS_PER_SEC);
        cpuFile << cpu_obuf;

        initDisk(conf, conf->nx/10.0, points);

        start = clock();
        mainCUDA(conf, points);
        finish = clock();

        char gpu_obuf[64];
        sprintf(gpu_obuf, "GPU,%d,%f\n", sim, (double)(finish - start) / CLOCKS_PER_SEC);
        gpuFile << gpu_obuf;

        // Release the memory
        free(Un);
        free(Unp1);
        free(points);
    }
    free(conf);
    return 0;
}