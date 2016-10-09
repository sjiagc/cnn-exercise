#include "DataBlock.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

namespace utils
{

template<typename TDataType>
DataBlock<TDataType>::DataBlock(size_t inElementCount)
    : m_elementCount(inElementCount)
    , m_cpuData(nullptr, [](TDataType *p) -> void { cudaFreeHost(p); })
    , m_gpuData(nullptr, [](TDataType *p) -> void { cudaFree(p); })
    , m_cpuDirty(false)
    , m_gpuDirty(false)
{
    TDataType *theCPUPtr = nullptr;
    TDataType *theGPUPtr = nullptr;
    if (cudaMallocHost(&theCPUPtr, inElementCount * sizeof(TDataType)) != cudaSuccess)
        throw std::runtime_error("DataBlock::DataBlock: cannot allocate CPU memory");
    m_cpuData.reset(theCPUPtr);
    if (cudaMalloc(&theGPUPtr, inElementCount * sizeof(TDataType)) != cudaSuccess)
        throw std::runtime_error("DataBlock::DataBlock: cannot allocate GPU memory");
    m_gpuData.reset(theGPUPtr);
}

template<typename TDataType>
const TDataType*
DataBlock<TDataType>::getData()
{
    syncCPUData();
    return m_cpuData.get();
}

template<typename TDataType>
TDataType*
DataBlock<TDataType>::getMutableData()
{
    syncCPUData();
    m_cpuDirty = true;
    return m_cpuData.get();
}

template<typename TDataType>
const TDataType*
DataBlock<TDataType>::getGPUData()
{
    syncGPUData();
    return m_gpuData.get();
}

template<typename TDataType>
TDataType*
DataBlock<TDataType>::getMutableGPUData()
{
    syncGPUData();
    m_gpuDirty = true;
    return m_gpuData.get();
}

template<typename TDataType>
void
DataBlock<TDataType>::syncCPUData()
{
    if (m_cpuDirty && m_gpuDirty)
        throw std::runtime_error("DataBlock::syncData: data lost sync - both CPU and GPU data are dirty");
    if (m_gpuDirty) {
        if (cudaMemcpy(m_cpuData.get(),
                        m_gpuData.get(),
                        m_elementCount * sizeof(TDataType),
                        cudaMemcpyDeviceToHost) != cudaSuccess)
            throw std::runtime_error("DataBlock::syncGPUData: data sync failed");
        m_gpuDirty = false;
    }
}

template<typename TDataType>
void
DataBlock<TDataType>::syncGPUData()
{
    if (m_cpuDirty && m_gpuDirty)
        throw std::runtime_error("DataBlock::syncGPUData: data lost sync - both CPU and GPU data are dirty");
    if (m_cpuDirty) {
        if (cudaMemcpy(m_gpuData.get(),
                        m_cpuData.get(),
                        m_elementCount * sizeof(TDataType),
                        cudaMemcpyHostToDevice) != cudaSuccess)
            throw std::runtime_error("DataBlock::syncGPUData: data sync failed");
        m_cpuDirty = false;
    }
}

template class DataBlock<double>;

}
