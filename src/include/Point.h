//
// Created by erick on 7/3/25.
//

#ifndef POINT_H
#define POINT_H
#include "simulationConf.h"

//
struct Point {
    int x;
    int y;
    float T;

};

int initDisk(simulationConf* conf, float R, Point* points);
void printArray(Point* points, int N, const char* output_filename, int step);
#endif //POINT_H
