# Real-Time 2D Heat Transfer Simulation

A real-time 2D heat transfer simulation with OpenGL visualization, featuring both CPU and GPU (CUDA) implementations.

## Overview

This project implements a real-time 2D heat diffusion simulation that models heat transfer across a grid using the explicit finite difference method. The simulation visualizes a circular hot disk in the center of a cooler domain and displays the temperature evolution in real-time with color-coded visualization. Users can switch between CPU and GPU implementations and adjust parameters through an interactive GUI.

**Video Demo:** [Link](https://youtube.com/shorts/UIFWh9c5I8Y)

## Features

- **Real-time Visualization**: OpenGL-based rendering with color-coded temperature display
- **Interactive GUI**: ImGui interface for parameter adjustment and simulation control
- **Dual Implementation**: Both CPU and GPU (CUDA) versions with runtime switching
- **Dynamic Parameters**: Adjustable diffusion constant, temperatures, disk radius, and time step
- **Performance Monitoring**: Real-time display of center temperature and elapsed time
- **Keyboard Controls**: Toggle GUI, pause/resume simulation, and exit
- **Shader-based Rendering**: Custom vertex and fragment shaders for efficient visualization

## Project Structure

```
src/
├── Main.cpp                  # Main program with OpenGL setup and GUI
├── updateTemp_CPU.cpp        # CPU implementation of heat equation solver
├── updateTemp_GPU.cu         # CUDA/GPU implementation
├── shaderClass.cpp           # OpenGL shader management
├── VAO.cpp                   # Vertex Array Object wrapper
├── VBO.cpp                   # Vertex Buffer Object wrapper
├── EBO.cpp                   # Element Buffer Object wrapper
├── utils.cpp                 # Utility functions and square drawing
├── shaders/
│   ├── default.vert          # Vertex shader
│   └── default.frag          # Fragment shader
└── include/                  # Header files
    ├── Square.h              # Square structure definition
    ├── squareCoordinates.h   # Grid coordinate utilities
    ├── updateTemp.h          # Temperature update function declarations
    └── utils.cuh             # Utility functions and CUDA helpers
```

## Requirements

### Software Dependencies
- **OpenGL** 3.3 or higher
- **GLFW** 3.x (window management)
- **GLAD** (OpenGL loader)
- **GLM** (OpenGL mathematics)
- **ImGui** (immediate mode GUI)
- **CUDA Toolkit** (for GPU implementation)
- **C++ Compiler** supporting C++14 standard

### Hardware Requirements
- OpenGL 3.3 compatible graphics card
- NVIDIA GPU with CUDA support (for GPU mode)
- Sufficient memory for grid allocation

## Installation

1. **Install CUDA Toolkit:**
   Download and install from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

3. **Clone and build:**
   ```bash
   git clone https://github.com/esierra-engineer/cc7515_project.git
   cd cc7515_project
   mkdir build && cd build
   cmake ..
   make
   ```

## Usage

### Running the Simulation

```bash
./cc7515_project
```

### Controls

- **C**: Toggle configuration GUI
- **SPACE**: Pause/Resume simulation
- **ESC**: Exit application

### GUI Configuration

The configuration panel allows real-time adjustment of:

- **Simulation Engine**: Switch between CPU and GPU implementations
- **Time Step**: Adjust simulation speed (0.01 - 1.00)
- **Inner Temperature**: Hot disk temperature (0-100°C)
- **Outer Temperature**: Background temperature (0-100°C)
- **Transfer Coefficient**: Diffusion constant (0.01 - 1.0)
- **Disk Radius**: Size of the hot disk (0.0 - 0.5)
- **Reset**: Restart simulation with current parameters
- **Resume/Pause**: Control simulation execution

### Status Display

The main menu bar shows:
- Current center temperature
- Total grid elements
- Elapsed simulation time

## Algorithm Details

### Heat Equation
Solves the 2D heat diffusion equation:
```
∂T/∂t = α(∂²T/∂x² + ∂²T/∂y²)
```

### Finite Difference Implementation
Uses explicit finite difference discretization:
```cpp
T_new[i,j] = T[i,j] + α·Δt·((T[i-1,j] - 2T[i,j] + T[i+1,j])/Δx² + 
                              (T[i,j-1] - 2T[i,j] + T[i,j+1])/Δy²)
```

### Visualization
- **Color Mapping**: Temperature mapped to red-blue gradient
- **Real-time Updates**: Color updates every frame based on current temperature
- **Shader Rendering**: GPU-accelerated rendering using OpenGL shaders

## Implementation Details

### CPU Implementation (`updateTemp_CPU.cpp`)
- Sequential processing with nested loops
- Memory copying for double buffering
- Direct temperature and color updates

### GPU Implementation (`updateTemp_GPU.cu`)
- CUDA kernel with 16x16 thread blocks
- Parallel processing of grid points
- Device memory allocation and transfers
- Boundary condition handling in kernel

### OpenGL Rendering
- **VAO/VBO/EBO**: Efficient vertex data management
- **Instanced Rendering**: Each grid cell rendered as a colored square
- **Uniform Variables**: Dynamic scaling and transformation matrices
- **Shader Pipeline**: Vertex transformation and fragment coloring

## Performance Characteristics

### CPU Mode
- Single-threaded execution
- Suitable for smaller grids (≤ 128x128)
- Consistent performance across hardware

### GPU Mode
- Massively parallel execution
- Optimal for larger grids (≥ 64x64)
- Requires CUDA-compatible GPU
- Memory transfer overhead per frame

## Configuration

### Default Parameters
- **Grid Size**: 64x64 elements
- **Domain Size**: 1.0 x 1.0 units
- **Diffusion Constant**: 0.3
- **Hot Disk Radius**: 0.15
- **Inner Temperature**: 90°C
- **Outer Temperature**: 0°C
- **Window Size**: 900x900 pixels

### Stability Constraint
The explicit scheme requires:
```
Δt ≤ Δx²Δy² / (2α(Δx² + Δy²))
```

The simulation automatically calculates the maximum stable time step and applies the user's time step multiplier.