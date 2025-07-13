//
// Created by erick on 7/13/25.
//

#include "utils.cuh"
#include <glm/gtc/type_ptr.hpp>

using namespace glm;

void drawSquares(Square* squares, Shader* shaderProgram, int N) {
    using namespace glm;

    for (int i = 0; i < N; ++i) {
        Square sqi = squares[i];
        mat4 model = mat4(1.0f);
        model = translate(model, sqi.position);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram->ID, "model"), 1, GL_FALSE, value_ptr(model));
        glUniform3f(glGetUniformLocation(shaderProgram->ID, "color"), sqi.color.x, sqi.color.y, sqi.color.z);
        // Draw primitives, number of indices, datatype of indices, index of indices
        glDrawElements(GL_TRIANGLES, 9, GL_UNSIGNED_INT, nullptr);
    }
}

void initDisk(Square* squares, int W, int H, float dx, float dy, float Height, float Width,
    float radius2, float Tin, float Tout, vec<2, int>* center_coordinates) {
    for (int i = 0; i < W; ++i) {
        for (int j=0; j < H; ++j) {
            int index = getIndex(i, j, W);

            float x = dx * j - Height/2.0f;
            float y = dy * i - Width/2.0f;

            if (abs(x)<0.01f && abs(y)<0.01f) {
                center_coordinates->x = i;
                center_coordinates->y = j;
                printf("center found at index %d,%d \n", center_coordinates->x, center_coordinates->y);
            }

            // printf("index=%d, x=%f, y=%f\n", index, squares[index].position.x, squares[index].position.y);
            // printf("ds2=%f \n", ds2);
            squares[index].position = vec3(
                x,
                y,
                0.0f
                );
            float distance = glm::distance(squares[index].position, vec3(0.0f));
            // printf("distance=%f \n", distance);
            float ds2 = distance * distance;
            // printf("ds2=%f \n", ds2);
            if (ds2 < radius2) {
                squares[index].color = vec3(1.0f, 0.0f, 0.0f);
                squares[index].temp = Tin;
            } else {
                squares[index].color = vec3(0.0f, 0.0f, 0.0f);
                squares[index].temp = Tout;
            }
        }
    }
}