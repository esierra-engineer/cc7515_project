#include <cstring>

#include "updateTemp.h"
#include "utils.cuh"
//
// Created by erick on 7/13/25.
//
void updateTemp_CPU(Square* squares_t0, Square* squares_t1,
    int N, int W, int H,
    float dx2, float dy2,
    float a, float dt) {

    // copy input to next step
    std::memcpy(squares_t1, squares_t0, N*sizeof(Square));
    // Going through the entire area
    for (int i = 1; i < W-1; i++)
    {
        for (int j = 1; j < H-1; j++)
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
    std::swap(squares_t0, squares_t1);
}