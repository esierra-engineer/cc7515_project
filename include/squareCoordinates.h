//
// Created by erick on 7/13/25.
//

#ifndef SQUARECOORDINATES_H
#define SQUARECOORDINATES_H
// Vertices coordinates
GLfloat vertices[] =
{ //               COORDINATES
    -0.5f, -0.5f, 0.0f, 1.0f, 1.0f, 1.0f,
     0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 1.0f,
     0.5f,  0.5f, 0.0f, 1.0f, 0.0f, 1.0f,
    -0.5f,	0.5f, 0.0f, 0.0f, 1.0f, 1.0f,
};

// Indices for vertices order
GLuint indices[] =
{
    0, 1, 2,
    0, 2, 3
};
#endif //SQUARECOORDINATES_H
