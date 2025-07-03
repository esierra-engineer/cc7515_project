//
// Created by erick on 7/3/25.
//

#include "Point.h"

#include <iostream>

#include <fstream>

#include "utils.cuh"
#include "Point.h"
#include "simulationConf.h"


int initDisk(simulationConf* conf, float R, Point* points){
    int nx = conf->nx, ny = conf->ny;
    float* Un = conf->Un, Tin = conf->Tin, Tout = conf->Tout;

    // Initializing the data with a pattern of disk of radius of 1/6 of the width
    float radius2 = R * R;
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            int index = getIndex(i, j, ny);
            points[index].x = i;
            points[index].y = j;
            // Distance of point i, j from the origin
            float ds2 = (i - nx/2) * (i - nx/2) + (j - ny/2)*(j - ny/2);
            if (ds2 < radius2)
            {
                Un[index] = Tin;
                points[index].T = Tin;
            }
            else
            {
                Un[index] = Tout;
                points[index].T = Tout;
            }
        }
    }
    return 1;
}

void printArray(Point* points, int N, const char* output_filename, int step) {
    std::ofstream outputFile(output_filename, std::ios::app);
    if (!output_filename) {
        if (!outputFile.is_open()) std::cerr << "Error: Unable to open file for writing.\n";
    }
    outputFile << (step > 0 ? "" : "step,x,y,T\n");

    for (int i=0; i < N; i++) {
        const Point p = points[i];
        if (output_filename) {
            char line[50];
            sprintf(line, "%d,%d,%d,%f\n", step, p.x, p.y, p.T);
            outputFile << line;
        } else {
            printf("Step %d: Point (%d, %d) T = %f\n", step, p.x, p.y, p.T);
        }
    }

    outputFile.close();
}
