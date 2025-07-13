//
// Created by erick on 7/13/25.
//

#ifndef UPDATETEMP_H
#define UPDATETEMP_H
#include "Square.h"
//
// Created by erick on 7/13/25.
//
void updateTemp_CPU(Square* squares_t0, Square* squares_t1,
    int N, int W, int H,
    float dx2, float dy2,
    float a, float dt);

void updateTemp_GPU(Square* squares_t0, Square* squares_t1,
    int N, int W, int H,
    float dx2, float dy2,
    float a, float dt);
#endif //UPDATETEMP_H
