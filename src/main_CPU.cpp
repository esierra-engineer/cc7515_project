//
// Created by erick on 6/21/25.
//
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "utils.cuh"
#include "main_CPU.h"

void mainCPU(int nx, int ny, float a, float dt, int numSteps, int outputEvery, int numElements, float dx2, float dy2) {
    // Allocate two sets of data for current and next timesteps
    float* Un   = (float*)calloc(numElements, sizeof(float));
    float* Unp1 = (float*)calloc(numElements, sizeof(float));

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
                Un[index] = 65.0;
            }
            else
            {
                Un[index] = 5.0;
            }
        }
    }

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
            }
        }
        // Write the output if needed
        if (n % outputEvery == 0)
        {
            char filename[64];
            sprintf(filename, "heat_%04d.png", n);
            //save_png(Un, nx, ny, filename, 'c');
        }
        // Swapping the pointers for the next timestep
        std::swap(Un, Unp1);
    }

    // Timing
    clock_t finish = clock();
    printf("[CPU] It took %f seconds\n", (double)(finish - start) / CLOCKS_PER_SEC);

    // Release the memory
    free(Un);
    free(Unp1);
}