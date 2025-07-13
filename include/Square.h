//
// Created by erick on 7/13/25.
//

#ifndef SQUARE_H
#define SQUARE_H
#include "glm/glm.hpp"

using namespace glm;
class Square {
public:
    vec3 position = vec3(0.0f);
    vec3 color = vec3(1.0f);
    float temp = 0.0f;
    Square() = default;
};
#endif //SQUARE_H
