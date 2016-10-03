#include <Layer.hpp>

#include "layers/ConvolutionLayer.hpp"
#include "layers/InnerProductLayer.hpp"
#include "layers/PoolingLayer.hpp"
#include "layers/ReluLayer.hpp"
#include "layers/SoftMaxLayer.hpp"

namespace layer
{

template<typename TDataType>
typename Layer<TDataType>::TUniqueHandle
Layer<TDataType>::create(const std::string &inLayerType, const TLayerConfig &inLayerConfig)
{
    if (inLayerType.compare(ConvolutionLayer<TDataType>::TYPE) == 0) {
        return TUniqueHandle(new ConvolutionLayer<TDataType>(inLayerConfig), Layer<TDataType>::release);
    } else if (inLayerType.compare(InnerProductLayer<TDataType>::TYPE) == 0) {
        return TUniqueHandle(new InnerProductLayer<TDataType>(inLayerConfig), Layer<TDataType>::release);
    } else if (inLayerType.compare(PoolingLayer<TDataType>::TYPE) == 0) {
        return TUniqueHandle(new PoolingLayer<TDataType>(inLayerConfig), Layer<TDataType>::release);
    } else if (inLayerType.compare(ReluLayer<TDataType>::TYPE) == 0) {
        return TUniqueHandle(new ReluLayer<TDataType>(inLayerConfig), Layer<TDataType>::release);
    } else if (inLayerType.compare(SoftMaxLayer<TDataType>::TYPE) == 0) {
        return TUniqueHandle(new SoftMaxLayer<TDataType>(inLayerConfig), Layer<TDataType>::release);
    }
    return TUniqueHandle(nullptr, nullptr);
}

template<typename TDataType>
void
Layer<TDataType>::release(Layer *inInstance)
{
    delete inInstance;
}

template class Layer<double>;

}
