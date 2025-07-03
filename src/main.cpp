/*
 * Based on CSC materials from:
 *
 * https://github.com/csc-training/openacc/tree/master/exercises/heat
 *
 */

#include <cstdlib>

#include "main_CUDA.cuh"
#include "main_CPU.h"
#include "Point.h"
// #include "pngwriter.h"


int main(int argc, char** argv)
{
    simulationConf* conf = new simulationConf;
    conf->nx = 300;   // Width of the area
    conf->ny = 300;   // Height of the area

    conf->a = 0.5;     // Diffusion constant

    const float dx = 0.01;   // Horizontal grid spacing
    const float dy = 0.01;   // Vertical grid spacing

    conf->dx2 = dx*dx;
    conf->dy2 = dy*dy;

    conf->dt = conf->dx2 * conf->dy2 / (2.0 * conf->a * (conf->dx2 + conf->dy2)); // Largest stable time step
    conf->numSteps = 5000;                             // Number of time steps
    conf->outputEvery = 100;                          // How frequently to write output image

    conf->numElements = conf->nx*conf->ny;

    conf->output_filename_CPU = "/media/storage/git/cc7515_project/output/outputCPU.csv";
    conf->output_filename_GPU = "/media/storage/git/cc7515_project/output/outputGPU.csv";
    conf->Tin = 100.0f;
    conf->Tout = 000.0f;

    // Allocate two sets of data for current and next timesteps
    float* Un   = (float*)calloc(conf->numElements, sizeof(float));
    float* Unp1 = (float*)calloc(conf->numElements, sizeof(float));

    conf->Un = Un; conf->Unp1 = Unp1;

    Point* points = (Point*)malloc(conf->numElements * sizeof(Point));

    initDisk(conf, conf->nx/10.0, points);
    mainCPU(conf, points);

    initDisk(conf, conf->nx/10.0, points);
    mainCUDA(conf, points);

    // Release the memory
    free(Un);
    free(Unp1);
    free(points);

    return 0;
}