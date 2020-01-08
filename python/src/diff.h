// stillleben differentiation module CUDA kernels
// Author: Arul Periyasamy <arul.periyasamy@ais.uni-bonn.de>

#ifndef STILLLEBEN_DIFF_H
#define STILLLEBEN_DIFF_H

#include <torch/all.h>

#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <cmath>
#include <tuple>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

namespace diff
{
    void generateSobelValidMaskCuda(torch::Tensor& instanceIndices,
        torch::Tensor& depthImage, torch::Tensor validMask);
    void dilateObjectMaskCuda(torch::Tensor& objectMask,
        torch::Tensor& sobelValidMask, torch::Tensor& coordinates,
        torch::Tensor& dilatedMask, torch::Tensor& dilatedCoordinates);
}

#endif
