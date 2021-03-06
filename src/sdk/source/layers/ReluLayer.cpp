#include <layers/ReluLayer.hpp>

#include <limits>

namespace layer
{

template<typename TDataType> const char* ReluLayer<TDataType>::TYPE = "Relu";
template<typename TDataType> const char* ReluLayer<TDataType>::CONFIG_NEGATIVE_SLOPE = "negative_slope";
template<typename TDataType>
ReluLayer<TDataType>::ReluLayer(const typename Layer<TDataType>::TLayerConfig &inConfig)
    : m_negativeSlope(0)
    , m_input(nullptr)
    , m_diffAhead(nullptr)
    , m_mode(ComputeModeEnum::CPU)
    , m_forwardMethod(&ReluLayer::forwardCPU)
{
    if (inConfig.count(CONFIG_NEGATIVE_SLOPE)) {
        const std::string &theNegativeSlope = inConfig.find(CONFIG_NEGATIVE_SLOPE)->second;
        m_negativeSlope = atof(theNegativeSlope.c_str());
    }
}

template<typename TDataType>
ReluLayer<TDataType>::~ReluLayer()
{

}

template<typename TDataType>
const char*
ReluLayer<TDataType>::getType()
{
    return TYPE;
}

template<typename TDataType>
void
ReluLayer<TDataType>::setMode(ComputeModeEnum inMode)
{
    switch(inMode) {
    case ComputeModeEnum::CPU:
        m_mode = ComputeModeEnum::CPU;
        m_forwardMethod = &ReluLayer::forwardCPU;
        break;
    case ComputeModeEnum::GPU:
        m_mode = ComputeModeEnum::GPU;
        m_forwardMethod = &ReluLayer::forwardGPU;
        break;
    };
}

template<typename TDataType>
ComputeModeEnum
ReluLayer<TDataType>::getMode()
{
    return m_mode;
}

template<typename TDataType>
void
ReluLayer<TDataType>::connect(Layer<TDataType> &inDescendentLayer)
{
    inDescendentLayer.setForwardInput(*getOutput());
    setBackwardDiff(*inDescendentLayer.getDiff());
}

template<typename TDataType>
void
ReluLayer<TDataType>::restore(const typename Layer<TDataType>::TDataRestoring &inStoredData)
{

}

template<typename TDataType>
void
ReluLayer<TDataType>::forward()
{
    (this->*m_forwardMethod)();
}

template<typename TDataType>
void
ReluLayer<TDataType>::backward()
{

}

template<typename TDataType>
const utils::Matrix<TDataType>*
ReluLayer<TDataType>::getOutput() const
{
    return m_data.get();
}

template<typename TDataType>
const utils::Matrix<TDataType>*
ReluLayer<TDataType>::getDiff() const
{
    return m_diff.get();
}

template<typename TDataType>
void
ReluLayer<TDataType>::setForwardInput(const utils::Matrix<TDataType> &inInput)
{
    const utils::Dimension &theInputDim = inInput.getDimension();
    if (theInputDim.axisCount() < 1)
        throw std::runtime_error("ReluLayer::setForwardInput: invalid input dimension");
    m_data.reset(new utils::Matrix<TDataType>(theInputDim));
    m_input = &inInput;
}

template<typename TDataType>
void
ReluLayer<TDataType>::setBackwardDiff(const utils::Matrix<TDataType> &inDiff)
{

}

template<typename TDataType>
void
ReluLayer<TDataType>::forwardCPU()
{
    const utils::Dimension &theSrcDim = m_input->getDimension();
    typename utils::Matrix<TDataType>::data_type *theDstData = m_data->getMutableData();
    const typename utils::Matrix<TDataType>::data_type *theSrcData = m_input->getData();
    for (int64_t w = 0, theWDim = theSrcDim.getW(); w < theWDim; ++w) {
        for (int64_t z = 0, theZDim = theSrcDim.getZ(); z < theZDim; ++z) {
            for (int64_t y = 0, theYDim = theSrcDim.getY(); y < theYDim; ++y) {
                for (int64_t x = 0, theXDim = theSrcDim.getX(); x < theXDim; ++x) {
                    int64_t theSrcOffset = m_input->offset(x, y, z, w);
                    TDataType theSrcValue = theSrcData[theSrcOffset];
                    if (theSrcValue <= 0)
                        theSrcValue *= m_negativeSlope;
                    int64_t theDstOffset = m_data->offset(x, y, z, w);
                    theDstData[theDstOffset] = theSrcValue;
                }
            }
        }
    }
}

template class ReluLayer<double>;
}
