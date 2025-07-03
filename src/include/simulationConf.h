//
// Created by erick on 7/3/25.
//

#ifndef SIMULATIONCONF_H
#define SIMULATIONCONF_H


class simulationConf {
public:
    float* Un;
    float* Unp1;
    int nx, ny, numSteps, outputEvery, numElements;
    float a, dt, dx2, dy2, Tin, Tout;
    char* output_filename_CPU;
    char* output_filename_GPU;
};

#endif //SIMULATIONCONF_H
