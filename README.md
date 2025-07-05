# CC7515 Project - 2D Heat Transfer Simulation

A high-performance 2D heat transfer simulation comparing CPU and GPU (CUDA) implementations using the finite difference method.

## Overview

This project implements a 2D heat diffusion simulation that models heat transfer across a grid using the explicit finite difference scheme. The simulation initializes a circular hot disk in the center of a cooler domain and tracks the temperature evolution over time. The project provides both CPU and GPU implementations to compare performance characteristics.

## Features

- **Dual Implementation**: Both CPU and GPU (CUDA) versions for performance comparison
- **Configurable Parameters**: Adjustable grid size, time steps, diffusion constant, and temperatures
- **Performance Benchmarking**: Built-in timing and CSV output for performance analysis
- **Data Export**: Temperature field data exported to CSV format for visualization
- **Explicit Finite Difference**: Uses explicit scheme for solving the 2D heat equation

## Project Structure

```
cc7515_project/
├── src/
│   ├── main.cpp              # Main program and benchmarking
│   ├── main_CPU.cpp          # CPU implementation
│   ├── main_CUDA.cu          # CUDA/GPU implementation
│   ├── Point.cpp             # Point structure and utility functions
│   └── include/
│       ├── Point.h           # Point structure definition
│       ├── main_CPU.h        # CPU implementation header
│       ├── main_CUDA.cuh     # CUDA implementation header
│       ├── simulationConf.h  # Simulation configuration class
│       └── utils.cuh         # Utility functions and macros
├── output/                   # Output directory for CSV files
├── CMakeLists.txt           # CMake configuration
└── README.md                # This file
```

## Requirements

### Software Dependencies
- **CMake** 3.31 or higher
- **CUDA Toolkit** (with nvcc compiler)
- **C++ Compiler** supporting C++14 standard
- **NVIDIA GPU** with compute capability support

### Hardware Requirements
- NVIDIA GPU with CUDA support
- Sufficient memory for grid allocation (varies with grid size)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/esierra-engineer/cc7515_project.git
   cd cc7515_project
   ```

2. **Create output directory:**
   ```bash
   mkdir -p output
   ```

3. **Configure CUDA path** (if needed):
   Edit `CMakeLists.txt` to set the correct CUDA compiler path:
   ```cmake
   set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
   ```

4. **Build the project:**
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

## Usage

### Running the Simulation

```bash
./cc7515_project
```

The program will automatically run both CPU and GPU implementations and generate benchmark results.

### Configuration Parameters

The simulation parameters can be modified in `src/main.cpp`:

- `a`: Diffusion constant (default: 0.5)
- `dx`, `dy`: Grid spacing (default: 0.01)
- `numSteps`: Number of time steps (default: 5000)
- `outputEvery`: Output frequency (default: 100)
- `Tin`: Initial temperature of hot disk (default: 100.0°C)
- `Tout`: Background temperature (default: 0.0°C)
- `nx`, `ny`: Grid dimensions (configurable in loop)

### Output Files

The program generates several output files in the `output/` directory:

- `benchmarkCPU.csv`: CPU performance metrics
- `benchmarkGPU.csv`: GPU performance metrics  
- `outputCPU.csv`: Temperature field data from CPU simulation
- `outputGPU.csv`: Temperature field data from GPU simulation

## Algorithm Details

### Heat Equation
The simulation solves the 2D heat diffusion equation:
```
∂T/∂t = α(∂²T/∂x² + ∂²T/∂y²)
```

### Finite Difference Scheme
Uses explicit finite difference discretization:
```
T[i,j]^(n+1) = T[i,j]^n + α·Δt·((T[i-1,j] - 2T[i,j] + T[i+1,j])/Δx² + (T[i,j-1] - 2T[i,j] + T[i,j+1])/Δy²)
```

### Initial Conditions
- Hot circular disk of radius R = nx/10 at center with temperature `Tin`
- Background temperature `Tout` everywhere else
- Boundary conditions maintain initial temperatures

## Performance Comparison

The project includes built-in benchmarking that compares:
- **CPU Implementation**: Sequential processing using standard loops
- **GPU Implementation**: Parallel processing using CUDA kernels
- **Block Size**: 16x16 thread blocks for optimal GPU utilization

## CUDA Implementation Details

- **Grid Configuration**: 2D thread blocks (16x16)
- **Memory Management**: Explicit device memory allocation and transfers
- **Kernel Launch**: One kernel per time step with synchronization
- **Error Handling**: CUDA error checking after kernel execution

## Limitations

- Uses explicit time stepping (stability constraint: Δt ≤ Δx²Δy²/[2α(Δx²+Δy²)])
- Fixed boundary conditions
- Single precision floating point
- Memory transfers occur every time step (not optimized for multiple steps)
