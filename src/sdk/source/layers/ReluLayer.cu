#include <layers/ReluLayer.hpp>

#include <stdexcept>

namespace
{

const unsigned int BLOCK_DIM = 1024;

template<typename TDataType>
__global__
void
ReluLayer_forwardKernel(TDataType *inDst, const TDataType *inSrc, int64_t inElementCount, TDataType inNegativeSlope)
{
    int theThreadId = blockIdx.x * gridDim.x + threadIdx.x;
    if (inElementCount <= theThreadId)
        return;
    if (inSrc[theThreadId] <= 0)
        inDst[theThreadId] = inSrc[theThreadId] * inNegativeSlope;
    else
        inDst[theThreadId] = inSrc[theThreadId];
}

}

namespace layer
{

template<typename TDataType>
void
ReluLayer<TDataType>::forwardGPU()
{
    struct Dim theDim = m_input->getDimension().toDim();
    int64_t theElementCount = theDim.w * theDim.z * theDim.y * theDim.x;
    ReluLayer_forwardKernel<TDataType><<<static_cast<unsigned int>((theElementCount + BLOCK_DIM - 1) / BLOCK_DIM), BLOCK_DIM>>>
        (m_data->getMutableGPUData(), m_input->getGPUData(), theElementCount, m_negativeSlope);
    if (cudaGetLastError() != cudaSuccess)
        throw std::runtime_error("ReluLayer::forwardGPU: kernel execution failed");
}

template void ReluLayer<double>::forwardGPU();

}
