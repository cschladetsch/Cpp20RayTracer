#pragma once

#include <cuda_runtime.h>
#include <iostream>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
inline void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line 
                  << " '" << func << "' " << cudaGetErrorString(result) << std::endl;
        cudaDeviceReset();
        exit(99);
    }
}

