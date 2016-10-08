#ifndef TOY_CNN_SDK_UTILS_DATA_BLOCK_HPP__
#define TOY_CNN_SDK_UTILS_DATA_BLOCK_HPP__

#include <memory>

namespace utils
{

template<typename TDataType>
class DataBlock
{
public:
    DataBlock(size_t inElementCount);

    const TDataType* getData();
    TDataType* getMutableData();
    const TDataType* getGPUData();
    TDataType* getMutableGPUData();

private:
    DataBlock(const DataBlock&) = delete;
    DataBlock& operator = (const DataBlock&) = delete;

    void syncCPUData();
    void syncGPUData();

private:
    size_t m_elementCount;
    std::unique_ptr<TDataType, void(*)(TDataType*)> m_cpuData;
    std::unique_ptr<TDataType, void(*)(TDataType*)> m_gpuData;
    bool m_cpuDirty;
    bool m_gpuDirty;
};

}

#endif // TOY_CNN_SDK_UTILS_DATA_BLOCK_HPP__
