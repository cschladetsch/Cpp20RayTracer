cmake_minimum_required(VERSION 3.10)

project(RayTracerCUDA LANGUAGES CXX CUDA)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.6/bin/nvcc)
set(CMAKE_CUDA_STANDARD 14)

# Specify the architecture
set(CMAKE_CUDA_ARCHITECTURES 75)

# Define source files
set(SOURCES
    g.cu
    bmp_writer.cpp
)

# Specify the executable target
add_executable(raytracer ${SOURCES})

# Link necessary libraries
target_link_libraries(raytracer ${CUDA_LIBRARIES})

