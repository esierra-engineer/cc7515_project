//
// Created by erick on 6/21/25.
//
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <ctime>
#include "utils.cuh"
#include "main_CPU.h"

#include <iostream>

#include "Point.h"
#include "simulationConf.h"

void mainCPU(simulationConf* conf, Point* points) {

    float* Un = conf->Un;
    float* Unp1 = conf->Unp1;
    int nx = conf->nx, ny = conf->ny, numSteps = conf->numSteps,
    outputEvery = conf->outputEvery, numElements = conf->numElements;
    float a= conf->a, dt= conf->dt, dx2 = conf->dx2, dy2 = conf->dy2;
    const char* output_filename = conf->output_filename_CPU;

    // Fill in the data on the next step to ensure that the boundaries are identical.
    memcpy(Unp1, Un, numElements*sizeof(float));

    // Timing
    clock_t start = clock();

    // Main loop
    for (int n = 0; n <= numSteps; n++)
    {
        // Going through the entire area
        for (int i = 1; i < nx-1; i++)
        {
            for (int j = 1; j < ny-1; j++)
            {
                const int index = getIndex(i, j, ny);
                float uij = Un[index];
                float uim1j = Un[getIndex(i-1, j, ny)];
                float uijm1 = Un[getIndex(i, j-1, ny)];
                float uip1j = Un[getIndex(i+1, j, ny)];
                float uijp1 = Un[getIndex(i, j+1, ny)];

                // Explicit scheme
                Unp1[index] = uij + a * dt * ( (uim1j - 2.0*uij + uip1j)/dx2 + (uijm1 - 2.0*uij + uijp1)/dy2 );
                points[index].T = Unp1[index];
            }
        }
        // Write the output if needed
        if (n % outputEvery == 0)
        {
            //char filename[64];
            //sprintf(filename, "heat_%04d.png", n);
            printArray(points, numElements, output_filename, n);
            //save_png(Un, nx, ny, filename, 'c');
        }
        // Swapping the pointers for the next timestep
        std::swap(Un, Unp1);
    }

    // Timing
    clock_t finish = clock();
    printf("[CPU] It took %f seconds\n", (double)(finish - start) / CLOCKS_PER_SEC);
}
