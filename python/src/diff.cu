// stillleben differentiation module CUDA kernels
// Author: Arul Periyasamy <arul.periyasamy@ais.uni-bonn.de>

#include "diff.h"
#include <THC/THC.h>

extern THCState *state;

__device__ void clamp(int & idx, int maxIdx) {
    idx = idx > 0 ? idx : 0;
    idx = idx < (maxIdx - 1) ? idx : (maxIdx - 1);
}

const int filterWidth=3;

__global__ void generateSobelValidMaskKernel(const int16_t *instanceIndices,
    const float *depthImage, bool *validMask,
    const int numRows, const int numCols, const int shBlockElements)
{
    // copy block data to shared memory
    extern __shared__ int16_t shMemorySobel[];

    // divide the common shared memory between individual variables 
    int16_t *shInstanceIndices = shMemorySobel;
    float *shDepthImage = (float*)&shInstanceIndices[shBlockElements];

    const int2 thread2DIdx = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y);

    int halfWidth = filterWidth / 2;

    if(thread2DIdx.x < numCols && thread2DIdx.y < numRows)
    {
        // check if the thread corresponds boundary pixel of the block
        // if yes, load pixels in the neighboring window
        // else, load only the corresponding pixel.
        if(threadIdx.x == 0 || threadIdx.y == 0 || threadIdx.x == blockDim.x-1 || threadIdx.y == blockDim.y-1)
        {
            // boundary block pixels
            for(int x=-halfWidth; x<=halfWidth; ++x)
            {
                for(int y=-halfWidth; y<=halfWidth; ++y)
                {
                    int sh1DIdx = ((threadIdx.y + halfWidth + y) * (blockDim.x + (2 * halfWidth))) + (threadIdx.x + halfWidth + x) ;

                    int ty = thread2DIdx.y + y;
                    int tx = thread2DIdx.x + x;

                    clamp(ty, numRows);
                    clamp(tx, numCols); 

                    int thread1DIdx =  ty * numCols + tx;

                    shInstanceIndices[sh1DIdx] = instanceIndices[thread1DIdx];
                    shDepthImage[sh1DIdx] = depthImage[thread1DIdx];
                }
            }
        }
        else
        {
            // interior block pixels
            int thread1DIdx = thread2DIdx.y * numCols + thread2DIdx.x;
            int sh1DIdx = ((threadIdx.y+halfWidth) * (blockDim.x + (2* halfWidth))) + (threadIdx.x+halfWidth);
            shInstanceIndices[sh1DIdx] = instanceIndices[thread1DIdx];
            shDepthImage[sh1DIdx] = depthImage[thread1DIdx];
        }
        __syncthreads();

        int  thread1DIdx = thread2DIdx.y * numCols + thread2DIdx.x;
        int sh1DIdx = ((threadIdx.y+halfWidth) * (blockDim.x + (2* halfWidth))) + (threadIdx.x+halfWidth);

        bool isPixelValid = 1;

        // shInstanceIndices[sh1DIdx] == 0 are background pixels
        // background pixels are not interesting
        if(shInstanceIndices[sh1DIdx] != 0)
        {
            int16_t currentIndex = shInstanceIndices[sh1DIdx];
            float currentDepth= shDepthImage[sh1DIdx];
            for(int x=-halfWidth; x<=halfWidth; ++x)
            {
                for(int y=-halfWidth; y<=halfWidth; ++y)
                {
                    int shWindow1DIdx = ((threadIdx.y + halfWidth + y) * (blockDim.x + (2* halfWidth))) + (threadIdx.x + halfWidth + x);
                    if((shInstanceIndices[shWindow1DIdx] != currentIndex) &&
                        (shInstanceIndices[shWindow1DIdx] != 0) &&
                        (shDepthImage[shWindow1DIdx] < currentDepth))
                    {
                        isPixelValid = 0;
                    }
                }
            }
        }

        // write if isPixelValid is 0
        if(isPixelValid == 0)
        {
            validMask[ thread1DIdx ] = 0;
        }

    } //if(thread2DIdx.x < numCols && thread1DIdx.y < numRows)
}

__global__ void dilateObjectMaskKernel(const bool *objectMask, const bool *sobelValidMask, const float3 *coordinates,
    bool *dilatedMask, float3 *dilatedCoordinates, const int numRows, const int numCols, const int shBlockElements)
{
    // copy block data to shared memory
    extern __shared__ bool shMemoryDilate[];

    bool *shObjectMask = shMemoryDilate;
    bool *shSobelValidMask = (bool*)&shObjectMask[shBlockElements];
    float3 *shCoordinates = (float3*)&shSobelValidMask[shBlockElements];

    int halfWidth = filterWidth / 2;

    const int2 thread2DIdx = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y);

    // check if the thread corresponds boundary pixel of the block
    // if yes, load pixels in the neighboring window
    // else, load only the corresponding pixel.

    if(thread2DIdx.x < numCols && thread2DIdx.y < numRows)
    {
        if(threadIdx.x == 0 || threadIdx.y == 0 || threadIdx.x == blockDim.x-1 || threadIdx.y == blockDim.y-1)
        {
            // boundary block pixels
            for(int x=-halfWidth; x<=halfWidth; ++x)
            {
                for(int y=-halfWidth; y<=halfWidth; ++y)
                {
                    int sh1DIdx = ((threadIdx.y + halfWidth + y) * (blockDim.x + (2 * halfWidth))) + (threadIdx.x  + halfWidth + x);
                    
                    int ty = thread2DIdx.y + y;
                    int tx = thread2DIdx.x + x;
                    clamp(ty, numRows);
                    clamp(tx, numCols); 
                    int thread1DIdx =  ty * numCols + tx;

                    shObjectMask[sh1DIdx] = objectMask[thread1DIdx];
                    shSobelValidMask[sh1DIdx] = sobelValidMask[thread1DIdx];
                    shCoordinates[sh1DIdx] = coordinates[thread1DIdx];
                }
            }
        }
        else
        {   // interior block pixels
            int thread1DIdx = thread2DIdx.y * numCols + thread2DIdx.x;
            int sh1DIdx = ((threadIdx.y+halfWidth) * (blockDim.x + (2 * halfWidth))) + (threadIdx.x+halfWidth);
            shObjectMask[sh1DIdx] = objectMask[thread1DIdx];
            shSobelValidMask[sh1DIdx] = sobelValidMask[thread1DIdx];
            shCoordinates[sh1DIdx] = coordinates[thread1DIdx];
        }

        __syncthreads();

        int thread1DIdx = thread2DIdx.y * numCols + thread2DIdx.x;
        int sh1DIdx = ((threadIdx.y+halfWidth) * (blockDim.x + (2 * halfWidth))) + (threadIdx.x+halfWidth);

        bool outputMask = shObjectMask[sh1DIdx];
        float3 outputCoords = shCoordinates[sh1DIdx];

        if(outputMask == 0)
        {
            bool allValid = true;
            bool allBackground = true;
            for(int x=-halfWidth; x<=halfWidth; ++x)
            {
                for(int y=-halfWidth; y<=halfWidth; ++y)
                {
                    int shWindow1DIdx = ((threadIdx.y + halfWidth + y) * (blockDim.x + (2 * halfWidth))) + (threadIdx.x + halfWidth + x);
                    if(shObjectMask[shWindow1DIdx] != 0)
                    {
                        allBackground = false;
                        outputCoords = shCoordinates[shWindow1DIdx];
                    }

                    if(shSobelValidMask[shWindow1DIdx] == 0)
                    {
                        allValid = false;
                        break;
                    }
                }
            }

            // write dilated mask and coordinates for valid pixels
            if(!allBackground && allValid)
            {
                outputMask = 1;
            }
        } // if(shObjectMask[sh1DIdx] != 0)

        dilatedMask[thread1DIdx] = outputMask;
        dilatedCoordinates[thread1DIdx] = outputCoords;
    } // if(thread2DIdx.x < numCols && thread1DIdx.y < numRows)
}

namespace diff
{
    void generateSobelValidMaskCuda(torch::Tensor& instanceIndices,
        torch::Tensor& depthImage, torch::Tensor validMask)
    {
        // Type and shape checks are already done in bridge.cpp
        // skip sanity checks
        int16_t *instanceIndicesPtr = instanceIndices.data_ptr<int16_t>();
        float *depthImagePtr = depthImage.data_ptr<float>();
        bool *validMaskPtr = validMask.data_ptr<bool>();
        auto numRows = instanceIndices.size(0);
        auto numCols = instanceIndices.size(1);

        int threadsX = 32;
        int threadsY = 32;
        int sharedX = threadsX + 2 * (filterWidth / 2);
        int sharedY = threadsY + 2 * (filterWidth / 2);
        auto blocksX  = 1 + ((numCols - 1) / threadsX);
        auto blocksY  = 1 + ((numRows - 1) / threadsY);
        const dim3 blockSize(threadsX, threadsY);
        const dim3 gradSize(blocksX, blocksY);

        // const dim3 blockSize(1, 1);
        // const dim3 gradSize(1, 1);

        int shBlockElements = sharedX * sharedY;

        // compute total share memory per block needed
        int64_t shTotalMemory;
        shTotalMemory = sizeof(/* instanceIndices */ int16_t) * shBlockElements + sizeof (/* depthImage */ float) * shBlockElements;
        generateSobelValidMaskKernel<<<gradSize, blockSize, shTotalMemory>>> (instanceIndicesPtr, depthImagePtr, validMaskPtr, numRows, numCols, shBlockElements);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    } // void generateSobelValidMaskCuda

    void dilateObjectMaskCuda(torch::Tensor& objectMask,
        torch::Tensor& sobelValidMask, torch::Tensor& coordinates,
        torch::Tensor& dilatedMask, torch::Tensor& dilatedCoordinates)
    {
        // Type and shape checks are already done in bridge.cpp
        // skip sanity checks
        bool *objectMaskPtr = objectMask.data_ptr<bool>();
        bool *sobelValidMaskPtr = sobelValidMask.data_ptr<bool>();
        float *coordsPtr = coordinates.data_ptr<float>();
        bool *dilatedMaskPtr = dilatedMask.data_ptr<bool>();
        float *dilatedCoordsPtr = dilatedCoordinates.data_ptr<float>();

        float3 *coordinatesPtr = (float3 *)coordsPtr;
        float3 *dilatedCoordinatesPtr = (float3 *)dilatedCoordsPtr;

        int numRows = objectMask.size(0);
        int numCols = objectMask.size(1);

        int threadsX = 32;
        int threadsY = 32;
        int sharedX = threadsX + 2 * (filterWidth / 2);
        int sharedY = threadsY + 2 * (filterWidth / 2);
        int blocksX  = 1 + ((numCols - 1) / threadsX);
        int blocksY  = 1 + ((numRows - 1) / threadsY);
        const dim3 blockSize(threadsX, threadsY);
        const dim3 gradSize(blocksX, blocksY);

        // const dim3 blockSize(1, 1);
        // const dim3 gradSize(1, 1);

        int shBlockElements = sharedX * sharedY;

        // compute total share memory per block needed
        int64_t shTotalMemory;
        shTotalMemory = sizeof(/* objectMask */  bool) * shBlockElements + 
                        sizeof(/* sobelValidMask */ bool) * shBlockElements +
                        sizeof(/* coordsPtr */ float3) * shBlockElements;

        // shBlockElements can also be computed from blockDim.x, blockDim.y, and filterWidth insize each kernel thread
        // But, here we are doing it once and pass it as a param
        dilateObjectMaskKernel <<<gradSize, blockSize, shTotalMemory>>>
            (objectMaskPtr, sobelValidMaskPtr, coordinatesPtr, dilatedMaskPtr, dilatedCoordinatesPtr, numRows, numCols, shBlockElements);

        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    } // void dilateObjectMaskCuda
} //namespace diff
