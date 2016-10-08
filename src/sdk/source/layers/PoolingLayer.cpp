#include <layers/PoolingLayer.hpp>

#include <limits>

namespace layer
{

template<typename TDataType> const char* PoolingLayer<TDataType>::TYPE = "Pooling";
template<typename TDataType> const char* PoolingLayer<TDataType>::CONFIG_POOL = "max";
template<typename TDataType> const char* PoolingLayer<TDataType>::CONFIG_KERNEL_SIZE = "kernel_size";
template<typename TDataType> const char* PoolingLayer<TDataType>::CONFIG_KERNEL_SIZE_X = "kernel_size_x";
template<typename TDataType> const char* PoolingLayer<TDataType>::CONFIG_KERNEL_SIZE_Y = "kernel_size_y";
template<typename TDataType> const char* PoolingLayer<TDataType>::CONFIG_STRIDE = "stride";
template<typename TDataType> const char* PoolingLayer<TDataType>::CONFIG_STRIDE_X = "stride_x";
template<typename TDataType> const char* PoolingLayer<TDataType>::CONFIG_STRIDE_Y = "stride_y";
template<typename TDataType> const char* PoolingLayer<TDataType>::CONFIG_PADDING = "padding";
template<typename TDataType> const char* PoolingLayer<TDataType>::CONFIG_PADDING_X = "padding_x";
template<typename TDataType> const char* PoolingLayer<TDataType>::CONFIG_PADDING_Y = "padding_y";

template<typename TDataType>
PoolingLayer<TDataType>::PoolingLayer(const Layer<TDataType>::TLayerConfig &inConfig)
    : m_numOfOutput(1)
    , m_kernelSize(3, 3, 1, 0)
    , m_padding(0)
    , m_stride(1, 1, 0)
    , m_input(nullptr)
    , m_diffAhead(nullptr)
{
    if (inConfig.count(CONFIG_KERNEL_SIZE)) {
        const std::string &theKernelSize = inConfig.find(CONFIG_KERNEL_SIZE)->second;
        m_kernelSize.setX(atoll(theKernelSize.c_str()));
        m_kernelSize.setY(atoll(theKernelSize.c_str()));
    }
    if (inConfig.count(CONFIG_KERNEL_SIZE_X))
        m_kernelSize.setX(atoll(inConfig.find(CONFIG_KERNEL_SIZE_X)->second.c_str()));
    if (inConfig.count(CONFIG_KERNEL_SIZE_Y))
        m_kernelSize.setY(atoll(inConfig.find(CONFIG_KERNEL_SIZE_Y)->second.c_str()));
    if (inConfig.count(CONFIG_PADDING)) {
        const std::string &thePadding = inConfig.find(CONFIG_PADDING)->second;
        m_padding.setX(atoll(thePadding.c_str()));
        m_padding.setY(atoll(thePadding.c_str()));
    }
    if (inConfig.count(CONFIG_PADDING_X))
        m_padding.setX(atoll(inConfig.find(CONFIG_PADDING_X)->second.c_str()));
    if (inConfig.count(CONFIG_PADDING_Y))
        m_padding.setY(atoll(inConfig.find(CONFIG_PADDING_Y)->second.c_str()));
    if (inConfig.count(CONFIG_STRIDE)) {
        const std::string &theStride = inConfig.find(CONFIG_STRIDE)->second;
        m_stride.setX(atoll(theStride.c_str()));
        m_stride.setY(atoll(theStride.c_str()));
    }
    if (inConfig.count(CONFIG_STRIDE_X))
        m_stride.setX(atoll(inConfig.find(CONFIG_STRIDE_X)->second.c_str()));
    if (inConfig.count(CONFIG_STRIDE_Y))
        m_stride.setY(atoll(inConfig.find(CONFIG_STRIDE_Y)->second.c_str()));
}

template<typename TDataType>
PoolingLayer<TDataType>::~PoolingLayer()
{

}

template<typename TDataType>
const char*
PoolingLayer<TDataType>::getType()
{
    return TYPE;
}

template<typename TDataType>
void
PoolingLayer<TDataType>::setMode(ComputeModeEnum inMode)
{
    (void)inMode;
}

template<typename TDataType>
ComputeModeEnum
PoolingLayer<TDataType>::getMode()
{
    return ComputeModeEnum::CPU;
}

template<typename TDataType>
void
PoolingLayer<TDataType>::connect(Layer<TDataType> &inDescendentLayer)
{
    inDescendentLayer.setForwardInput(*getOutput());
    setBackwardDiff(*inDescendentLayer.getDiff());
}

template<typename TDataType>
void
PoolingLayer<TDataType>::restore(const TDataRestoring &inStoredData)
{

}

template<typename TDataType>
void
PoolingLayer<TDataType>::forward()
{
    const utils::Dimension &theDataDim = m_data->getDimension();
    int64_t theSrcStartOffsetX = - m_padding.getX();
    int64_t theSrcStartOffsetY = - m_padding.getY();
    utils::Matrix<TDataType>::data_type *theDstData = m_data->getMutableData();
    const utils::Matrix<TDataType>::data_type *theSrcData = m_input->getData();
    for (int64_t theInputIndex = 0, theInputCount = theDataDim.getW(); theInputIndex < theInputCount; ++theInputIndex) {
        for (int64_t theChannelIndex = 0, theChannelCount = theDataDim.getZ(); theChannelIndex < theChannelCount; ++theChannelIndex) {
            for (int64_t x = 0, theXDim = theDataDim.getX(); x < theXDim; ++x) {
                for (int64_t y = 0, theYDim = theDataDim.getY(); y < theYDim; ++y) {
                    utils::Matrix<TDataType>::data_type theDstValue = std::numeric_limits<TDataType>::lowest();
                    for (int64_t theXInFilter = 0, theXDimOfFilter = m_kernelSize.getX(); theXInFilter < theXDimOfFilter; ++theXInFilter) {
                        for (int64_t theYInFilter = 0, theYDimOfFilter = m_kernelSize.getY(); theYInFilter < theYDimOfFilter; ++theYInFilter) {
                            TDataType theValueFromSrc = std::numeric_limits<TDataType>::lowest();
                            try {
                                int64_t theSrcOffset = m_input->offset(theSrcStartOffsetX + m_stride.getX() * x + theXInFilter,
                                                                       theSrcStartOffsetY + m_stride.getY() * y + theYInFilter,
                                                                       theChannelIndex,
                                                                       theInputIndex);
                                theValueFromSrc = theSrcData[theSrcOffset];
                            } catch (std::runtime_error &e) {
                                (void)e;
                            }
                            if (theDstValue < theValueFromSrc)
                                theDstValue = theValueFromSrc;
                        }
                    }
                    int64_t theDstOffset = m_data->offset(x, y, theChannelIndex, theInputIndex);
                    theDstData[theDstOffset] = theDstValue;
                }
            }
        }
    }
}

template<typename TDataType>
void
PoolingLayer<TDataType>::backward()
{

}

template<typename TDataType>
const utils::Matrix<TDataType>*
PoolingLayer<TDataType>::getOutput() const
{
    return m_data.get();
}

template<typename TDataType>
const utils::Matrix<TDataType>*
PoolingLayer<TDataType>::getDiff() const
{
    return m_diff.get();
}

template<typename TDataType>
void
PoolingLayer<TDataType>::setForwardInput(const utils::Matrix<TDataType> &inInput)
{
    const utils::Dimension &theInputDim = inInput.getDimension();
    if (theInputDim.axisCount() < 2)
        throw std::runtime_error("PoolingLayer::setForwardInput: invalid input dimension");
    const int64_t theXForStride = theInputDim.getX() - m_kernelSize.getX() + m_padding.getX() * 2;
    const int64_t theYForStride = theInputDim.getY() - m_kernelSize.getY() + m_padding.getY() * 2;
    m_data.reset(new utils::Matrix<TDataType>(utils::Dimension((theXForStride + m_stride.getX() - 1) / m_stride.getX() + 1,
                                              (theYForStride + m_stride.getY() - 1) / m_stride.getY() + 1,
                                              theInputDim.getZ(),
                                              theInputDim.getW())));
    m_input = &inInput;
}

template<typename TDataType>
void
PoolingLayer<TDataType>::setBackwardDiff(const utils::Matrix<TDataType> &inDiff)
{

}

template class PoolingLayer<double>;
}
