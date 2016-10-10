#include <layers/SoftMaxLayer.hpp>

#include <cuda_runtime.h>

#include <stdexcept>
#include <iostream>

namespace
{

const unsigned int THREADS_PER_BLOCK = 1024;

// Utils
__device__
int64_t
countDim(const Dim *inDim, int64_t inAxis = 4)
{
    int64_t theCount = 0;
    if (inAxis >= 1)
        theCount = inDim->x;
    if (inAxis >= 2)
        theCount *= inDim->y;
    if (inAxis >= 3)
        theCount *= inDim->z;
    if (inAxis >= 4)
        theCount *= inDim->w;
    return theCount;
}

__device__
int64_t
indexToOffset(int64_t inLinearIndex, const Dim *inDim, const Dim *inStrides)
{
    int64_t theXCount = countDim(inDim, 1);
    int64_t theYCount = countDim(inDim, 2);
    int64_t theZCount = countDim(inDim, 3);
    int64_t w = inLinearIndex / theZCount;
    inLinearIndex -= theZCount * w;
    int64_t z = inLinearIndex / theYCount;
    inLinearIndex -= theYCount * z;
    int64_t y = inLinearIndex / theXCount;
    inLinearIndex -= theXCount * y;
    int64_t x = inLinearIndex;
    return inStrides->w * w + inStrides->z * z + inStrides->y * y + inStrides->x * x;
}

// Get max
template<typename TDataType>
__device__
TDataType
getMaxOfTwo(TDataType inArg1, TDataType inArg2)
{
    return inArg1 >= inArg2 ? inArg1 : inArg2;
}

template<typename TDataType>
__device__
inline
TDataType
ShuffleDown(TDataType inData, unsigned int inOffset)
{
    return __shfl_down(inData, inOffset);
}

template<>
__device__
inline
double
ShuffleDown<double>(double inData, unsigned int inOffset)
{
    int2 theDataForShuffle = *reinterpret_cast<int2*>(&inData);
    int2 theShuffled;
    theShuffled.x = __shfl_down(theDataForShuffle.x, inOffset);
    theShuffled.y = __shfl_down(theDataForShuffle.y, inOffset);
    return *reinterpret_cast<double*>(&theShuffled);
}

template<typename TDataType>
__device__
inline
TDataType
calExp(TDataType inOprand)
{
    return expf(inOprand);
}

template<>
__device__
inline
double
calExp<double>(double inOprand)
{
    return exp(inOprand);
}

//TODO[sjiagc]: Refactor to remove duplicated code

template<typename TDataType>
__global__
void
getMax(TDataType *outMaxes, const TDataType *inData, Dim inDim, Dim inStrides)
{
    extern __shared__ TDataType theMaxes_shared[];
    __shared__ int64_t theElementCount;
    unsigned int theThreadId = threadIdx.x;
    unsigned int theGridSize = THREADS_PER_BLOCK * gridDim.x * 2;

    if (theThreadId == 0)
        theElementCount = countDim(&inDim);
    __syncthreads();

    TDataType  theLocalMax = inData[0];

    int64_t theIndexOfThread = blockIdx.x * THREADS_PER_BLOCK * 2 + theThreadId;
    while (theIndexOfThread < theElementCount) {
        theLocalMax = getMaxOfTwo(theLocalMax, inData[indexToOffset(theIndexOfThread, &inDim, &inStrides)]);
        if (theIndexOfThread + THREADS_PER_BLOCK < theElementCount)
            theLocalMax = getMaxOfTwo(theLocalMax, inData[indexToOffset(theIndexOfThread + THREADS_PER_BLOCK, &inDim, &inStrides)]);
        theIndexOfThread += theGridSize;
    }

    theMaxes_shared[theThreadId] = theLocalMax;
    __syncthreads();

    if (THREADS_PER_BLOCK >= 1024 && theThreadId < 512) {
        theMaxes_shared[theThreadId] = theLocalMax = getMaxOfTwo(theLocalMax, theMaxes_shared[theThreadId + 512]);
    }
    __syncthreads();
    if (THREADS_PER_BLOCK >= 512 && theThreadId < 256) {
        theMaxes_shared[theThreadId] = theLocalMax = getMaxOfTwo(theLocalMax, theMaxes_shared[theThreadId + 256]);
    }
    __syncthreads();
    if (THREADS_PER_BLOCK >= 256 && theThreadId < 128) {
        theMaxes_shared[theThreadId] = theLocalMax = getMaxOfTwo(theLocalMax, theMaxes_shared[theThreadId + 128]);
    }
    __syncthreads();
    if (THREADS_PER_BLOCK >= 128 && theThreadId < 64) {
        theMaxes_shared[theThreadId] = theLocalMax = getMaxOfTwo(theLocalMax, theMaxes_shared[theThreadId + 64]);
    }
    __syncthreads();

    if (theThreadId < 32) {
        if (THREADS_PER_BLOCK >= 64)
            theLocalMax = getMaxOfTwo(theLocalMax, theMaxes_shared[theThreadId + 32]);
#pragma unroll
        for (unsigned int theOffset = warpSize / 2; theOffset > 0; theOffset /= 2) {
            TDataType theShuffled = ShuffleDown(theLocalMax, theOffset);
            theLocalMax = getMaxOfTwo(theLocalMax, theShuffled);
        }
    }
    if (theThreadId == 0)
        outMaxes[theThreadId] = theLocalMax;
}

template<typename TDataType>
__global__
void
expAll(TDataType *outOutput, Dim inDim, Dim inOutputStrides, const TDataType *inInput, Dim inInputStrides, TDataType *inMax)
{
    unsigned int theThreadId = threadIdx.x;
    int64_t theIndexOfThread = blockIdx.x * THREADS_PER_BLOCK + theThreadId;
    int64_t theOutputOffset = indexToOffset(theIndexOfThread, &inDim, &inOutputStrides);
    int64_t theInputOffset = indexToOffset(theIndexOfThread, &inDim, &inInputStrides);
    outOutput[theOutputOffset] = calExp<TDataType>(inInput[theInputOffset] - *inMax);
}

template<typename TDataType>
__global__
void
sumUp(TDataType *outSum, const TDataType *inData, Dim inDim, Dim inStrides)
{
    extern __shared__ TDataType theSums_shared[];
    __shared__ int64_t theElementCount;
    unsigned int theThreadId = threadIdx.x;
    unsigned int theGridSize = THREADS_PER_BLOCK * gridDim.x * 2;

    if (theThreadId == 0)
        theElementCount = countDim(&inDim);
    __syncthreads();

    TDataType  theLocalSum = 0;

    int64_t theIndexOfThread = blockIdx.x * THREADS_PER_BLOCK * 2 + theThreadId;
    while (theIndexOfThread < theElementCount) {
        theLocalSum += inData[indexToOffset(theIndexOfThread, &inDim, &inStrides)];
        if (theIndexOfThread + THREADS_PER_BLOCK < theElementCount)
            theLocalSum += inData[indexToOffset(theIndexOfThread + THREADS_PER_BLOCK, &inDim, &inStrides)];
        theIndexOfThread += theGridSize;
    }

    theSums_shared[theThreadId] = theLocalSum;
    __syncthreads();

    if (THREADS_PER_BLOCK >= 1024 && theThreadId < 512) {
        theSums_shared[theThreadId] = theLocalSum = theLocalSum + theSums_shared[theThreadId + 512];
    }
    __syncthreads();
    if (THREADS_PER_BLOCK >= 512 && theThreadId < 256) {
        theSums_shared[theThreadId] = theLocalSum = theLocalSum + theSums_shared[theThreadId + 256];
    }
    __syncthreads();
    if (THREADS_PER_BLOCK >= 256 && theThreadId < 128) {
        theSums_shared[theThreadId] = theLocalSum = theLocalSum + theSums_shared[theThreadId + 128];
    }
    __syncthreads();
    if (THREADS_PER_BLOCK >= 128 && theThreadId < 64) {
        theSums_shared[theThreadId] = theLocalSum = theLocalSum + theSums_shared[theThreadId + 64];
    }
    __syncthreads();

    if (theThreadId < 32) {
        if (THREADS_PER_BLOCK >= 64)
            theLocalSum += theSums_shared[theThreadId + 32];
#pragma unroll
        for (unsigned int theOffset = warpSize / 2; theOffset > 0; theOffset /= 2) {
            TDataType theShuffled = ShuffleDown(theLocalSum, theOffset);
            theLocalSum += theShuffled;
        }
    }
    if (theThreadId == 0)
        outSum[theThreadId] = theLocalSum;
}

template<typename TDataType>
__global__
void
calProb(TDataType *outOutput, Dim inDim, Dim inOutputStrides, const TDataType *inInput, Dim inInputStrides, TDataType *inSum)
{
    unsigned int theThreadId = threadIdx.x;
    int64_t theIndexOfThread = blockIdx.x * THREADS_PER_BLOCK + theThreadId;
    int64_t theOutputOffset = indexToOffset(theIndexOfThread, &inDim, &inOutputStrides);
    int64_t theInputOffset = indexToOffset(theIndexOfThread, &inDim, &inInputStrides);
    outOutput[theOutputOffset] = inInput[theInputOffset] / *inSum;
}

}

namespace layer
{

template<typename TDataType>
void
SoftMaxLayer<TDataType>::forwardGPU()
{
    const utils::Dimension &theInputDim = m_input->getDimension();
    const utils::Dimension &theInputStrides = m_input->getStride();
    const utils::Dimension &theOutputDim = m_data->getDimension();
    const utils::Dimension &theOutputStrides = m_data->getStride();
    int64_t theElementCount = theOutputDim.count();

    unsigned int theBlockCount = theElementCount / THREADS_PER_BLOCK > 0 ? theElementCount / THREADS_PER_BLOCK : 1;
    TDataType *theIntermediateBuffer_d = nullptr;
    cudaMalloc(&theIntermediateBuffer_d, sizeof(TDataType) * theBlockCount);
    // Get max
    {
        TDataType *theOutput_d = theIntermediateBuffer_d;
        const TDataType *theInput_d = m_input->getGPUData();
        unsigned int theResultCount = theBlockCount;
        Dim theDim = theInputDim.toDim();
        Dim theStrides = theInputStrides.toDim();
        cudaError_t theCudaStatus = cudaSuccess;
        while (theResultCount) {
            getMax<TDataType><<<theBlockCount, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(TDataType)>>>(theOutput_d, theInput_d, theDim, theStrides);
            theCudaStatus = cudaGetLastError();
            if (theCudaStatus != cudaSuccess)
                throw std::runtime_error("SoftMaxLayer::forwardGPU: executing kernel for max failed");
            if (theResultCount <= 1)
                break;
            theDim.w = theDim.z = theDim.y = 1;
            theDim.x = theResultCount;
            theStrides.w = theStrides.z = theStrides.y = theResultCount;
            theStrides.x = 1;
            theResultCount = theResultCount/ THREADS_PER_BLOCK > 0 ? theResultCount/ THREADS_PER_BLOCK : 1;
        }
//        TDataType theMax = 0;
//        theCudaStatus = cudaMemcpy(&theMax, theOutput_d, sizeof(TDataType), cudaMemcpyDeviceToHost);
//        if (theCudaStatus != cudaSuccess) {
//            throw std::runtime_error(std::string("SoftMaxLayer::forwardGPU: get max failed, ") + cudaGetErrorString(theCudaStatus));
//        }
//        std::cout << "Max: " << theMax << std::endl;
    }
    // Calculate exp
    {
        expAll<TDataType><<<(theElementCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>
            (m_data->getMutableGPUData(), theOutputDim.toDim(), theOutputStrides.toDim(), m_input->getGPUData(), theInputStrides.toDim(), theIntermediateBuffer_d);
    }
    // Sum
    {
        TDataType *theOutput_d = theIntermediateBuffer_d;
        const TDataType *theInput_d = m_data->getGPUData();
        unsigned int theResultCount = theBlockCount;
        Dim theDim = theOutputDim.toDim();
        Dim theStrides = theOutputStrides.toDim();
        cudaError_t theCudaStatus = cudaSuccess;
        while (theResultCount) {
            sumUp<TDataType><<<theBlockCount, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(TDataType)>>>(theOutput_d, theInput_d, theDim, theStrides);
            theCudaStatus = cudaGetLastError();
            if (theCudaStatus != cudaSuccess)
                throw std::runtime_error("SoftMaxLayer::forwardGPU: executing kernel for max failed");
            if (theResultCount <= 1)
                break;
            theDim.w = theDim.z = theDim.y = 1;
            theDim.x = theResultCount;
            theStrides.w = theStrides.z = theStrides.y = theResultCount;
            theStrides.x = 1;
            theResultCount = theResultCount/ THREADS_PER_BLOCK > 0 ? theResultCount/ THREADS_PER_BLOCK : 1;
        }
//        TDataType theSum = 0;
//        theCudaStatus = cudaMemcpy(&theSum, theOutput_d, sizeof(TDataType), cudaMemcpyDeviceToHost);
//        if (theCudaStatus != cudaSuccess) {
//            throw std::runtime_error(std::string("SoftMaxLayer::forwardGPU: get max failed, ") + cudaGetErrorString(theCudaStatus));
//        }
//        std::cout << "Sum: " << theSum << std::endl;
    }
    // Calculate probability
    {
        calProb<TDataType><<<(theElementCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>
            (m_data->getMutableGPUData(), theOutputDim.toDim(), theOutputStrides.toDim(), m_data->getGPUData(), theOutputStrides.toDim(), theIntermediateBuffer_d);
    }

    cudaFree(theIntermediateBuffer_d);
}

template void SoftMaxLayer<double>::forwardGPU();

}
