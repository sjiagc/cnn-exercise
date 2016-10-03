#include <layers/SoftMaxLayer.hpp>

#include <limits>

namespace layer
{

template<typename TDataType> const char* SoftMaxLayer<TDataType>::TYPE = "SoftMax";

template<typename TDataType>
SoftMaxLayer<TDataType>::SoftMaxLayer(const Layer<TDataType>::TLayerConfig &inConfig)
    : m_input(nullptr)
    , m_diffAhead(nullptr)
{
}

template<typename TDataType>
SoftMaxLayer<TDataType>::~SoftMaxLayer()
{

}

template<typename TDataType>
const char*
SoftMaxLayer<TDataType>::getType()
{
    return TYPE;
}

template<typename TDataType>
void
SoftMaxLayer<TDataType>::connect(Layer<TDataType> &inDescendentLayer)
{
    inDescendentLayer.setForwardInput(*getOutput());
    setBackwardDiff(*inDescendentLayer.getDiff());
}

template<typename TDataType>
void
SoftMaxLayer<TDataType>::restore(const TDataRestoring &inStoredData)
{

}

template<typename TDataType>
void
SoftMaxLayer<TDataType>::forward()
{
    const utils::Dimension &theSrcDim = m_input->getDimension();
    utils::Matrix<TDataType>::data_type *theDstData = m_data->getData();
    const utils::Matrix<TDataType>::data_type *theSrcData = m_input->getData();

    // Get max value
    utils::Matrix<TDataType>::data_type theMax = theSrcData[m_input->offset()];
    for (int64_t w = 0, theWDim = theSrcDim.getW(); w < theWDim; ++w) {
        for (int64_t z = 0, theZDim = theSrcDim.getZ(); z < theZDim; ++z) {
            for (int64_t y = 0, theYDim = theSrcDim.getY(); y < theYDim; ++y) {
                for (int64_t x = 0, theXDim = theSrcDim.getX(); x < theXDim; ++x) {
                    int64_t theValue = theSrcData[m_input->offset(x, y, z, w)];
                    if (theMax < theValue)
                        theMax = theValue;
                }
            }
        }
    }
    // Substract max, exp and sum
    utils::Matrix<TDataType>::data_type theSum = 0;
    for (int64_t w = 0, theWDim = theSrcDim.getW(); w < theWDim; ++w) {
        for (int64_t z = 0, theZDim = theSrcDim.getZ(); z < theZDim; ++z) {
            for (int64_t y = 0, theYDim = theSrcDim.getY(); y < theYDim; ++y) {
                for (int64_t x = 0, theXDim = theSrcDim.getX(); x < theXDim; ++x) {
                    utils::Matrix<TDataType>::data_type theValue = exp(theSrcData[m_input->offset(x, y, z, w)] - theMax);
                    theDstData[m_data->offset(x, y, z, w)] = theValue;
                    theSum += theValue;
                }
            }
        }
    }
    // Divide
    for (int64_t w = 0, theWDim = theSrcDim.getW(); w < theWDim; ++w) {
        for (int64_t z = 0, theZDim = theSrcDim.getZ(); z < theZDim; ++z) {
            for (int64_t y = 0, theYDim = theSrcDim.getY(); y < theYDim; ++y) {
                for (int64_t x = 0, theXDim = theSrcDim.getX(); x < theXDim; ++x) {
                    theDstData[m_data->offset(x, y, z, w)] /= theSum;
                }
            }
        }
    }
}

template<typename TDataType>
void
SoftMaxLayer<TDataType>::backward()
{

}

template<typename TDataType>
const utils::Matrix<TDataType>*
SoftMaxLayer<TDataType>::getOutput() const
{
    return m_data.get();
}

template<typename TDataType>
const utils::Matrix<TDataType>*
SoftMaxLayer<TDataType>::getDiff() const
{
    return m_diff.get();
}

template<typename TDataType>
void
SoftMaxLayer<TDataType>::setForwardInput(const utils::Matrix<TDataType> &inInput)
{
    const utils::Dimension &theInputDim = inInput.getDimension();
    if (theInputDim.axisCount() < 1)
        throw std::runtime_error("SoftMaxLayer::setForwardInput: invalid input dimension");
    m_data.reset(new utils::Matrix<TDataType>(theInputDim));
    m_input = &inInput;
}

template<typename TDataType>
void
SoftMaxLayer<TDataType>::setBackwardDiff(const utils::Matrix<TDataType> &inDiff)
{

}

template class SoftMaxLayer<double>;
}
