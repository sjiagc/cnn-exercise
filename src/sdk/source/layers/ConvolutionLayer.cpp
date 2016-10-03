#include <layers/ConvolutionLayer.hpp>

#include <stdexcept>

namespace layer
{

template<typename TDataType> const char* ConvolutionLayer<TDataType>::TYPE = "Convolution";
template<typename TDataType> const char* ConvolutionLayer<TDataType>::CONFIG_NUM_OUTPUT = "num_output";
template<typename TDataType> const char* ConvolutionLayer<TDataType>::CONFIG_KERNEL_SIZE = "kernel_size";
template<typename TDataType> const char* ConvolutionLayer<TDataType>::CONFIG_KERNEL_SIZE_X = "kernel_size_x";
template<typename TDataType> const char* ConvolutionLayer<TDataType>::CONFIG_KERNEL_SIZE_Y = "kernel_size_y";
template<typename TDataType> const char* ConvolutionLayer<TDataType>::CONFIG_STRIDE = "stride";
template<typename TDataType> const char* ConvolutionLayer<TDataType>::CONFIG_STRIDE_X = "stride_x";
template<typename TDataType> const char* ConvolutionLayer<TDataType>::CONFIG_STRIDE_Y = "stride_y";
template<typename TDataType> const char* ConvolutionLayer<TDataType>::CONFIG_PADDING = "padding";
template<typename TDataType> const char* ConvolutionLayer<TDataType>::CONFIG_PADDING_X = "padding_x";
template<typename TDataType> const char* ConvolutionLayer<TDataType>::CONFIG_PADDING_Y = "padding_y";
template<typename TDataType> const char* ConvolutionLayer<TDataType>::CONFIG_BIAS_TERM = "bias_term";
template<typename TDataType> const char* ConvolutionLayer<TDataType>::RESTORE_WEIGHTS = "weights";
template<typename TDataType> const char* ConvolutionLayer<TDataType>::RESTORE_BIAS = "bias";

template<typename TDataType>
ConvolutionLayer<TDataType>::ConvolutionLayer(const Layer<TDataType>::TLayerConfig &inConfig)
    : m_numOfOutput(1)
    , m_kernelSize(3, 3, 1, 0)
    , m_padding(0)
    , m_stride(1, 1, 0)
    , m_hasBias(false)
    , m_input(nullptr)
    , m_diffAhead(nullptr)
{
    if (inConfig.count(CONFIG_NUM_OUTPUT))
        m_numOfOutput = atoll(inConfig.find(CONFIG_NUM_OUTPUT)->second.c_str());
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
    if (inConfig.count(CONFIG_BIAS_TERM))
        m_hasBias = inConfig.find(CONFIG_BIAS_TERM)->second == "true";
}

template<typename TDataType>
ConvolutionLayer<TDataType>::~ConvolutionLayer()
{

}

template<typename TDataType>
const char*
ConvolutionLayer<TDataType>::getType()
{
    return TYPE;
}

template<typename TDataType>
void
ConvolutionLayer<TDataType>::connect(Layer<TDataType> &inDescendentLayer)
{
    inDescendentLayer.setForwardInput(*getOutput());
    setBackwardDiff(*inDescendentLayer.getDiff());
}

template<typename TDataType>
void
ConvolutionLayer<TDataType>::restore(const TDataRestoring &inStoredData)
{
    if (inStoredData.count(RESTORE_WEIGHTS)) {
        const utils::Matrix<TDataType> *theWeights = inStoredData.find(RESTORE_WEIGHTS)->second;
        if (theWeights) {
            if (theWeights->getDimension() != m_weights->getDimension())
                throw std::runtime_error("ConvolutionLayer:restore: weights dimensions do not match");
            *m_weights = *theWeights;
        }
    }
    if (m_hasBias && inStoredData.count(RESTORE_BIAS)) {
        const utils::Matrix<TDataType> *theBias= inStoredData.find(RESTORE_BIAS)->second;
        if (theBias) {
            if (theBias->getDimension() != m_bias->getDimension())
                throw std::runtime_error("ConvolutionLayer:restore: bias dimensions do not match");
            *m_bias = *theBias;
        }
    }
}

template<typename TDataType>
void
ConvolutionLayer<TDataType>::forward()
{
    const utils::Dimension &theDataDim = m_data->getDimension();
    const utils::Dimension &theFilterDim = m_weights->getDimension();
    int64_t theSrcStartOffsetX = - m_padding.getX();
    int64_t theSrcStartOffsetY = - m_padding.getY();
    utils::Matrix<TDataType>::data_type *theDstData = m_data->getData();
    const utils::Matrix<TDataType>::data_type *theSrcData = m_input->getData();
    const utils::Matrix<TDataType>::data_type *theFilterData = m_weights->getData();
    for (int64_t theInputIndex = 0, theInputCount = theDataDim.getW(); theInputIndex < theInputCount; ++theInputIndex) {
        for (int64_t theOutputIndex = 0, theOutputCount = theDataDim.getZ(); theOutputIndex < theOutputCount; ++theOutputIndex) {
            for (int64_t x = 0, theXDim = theDataDim.getX(); x < theXDim; ++x) {
                for (int64_t y = 0, theYDim = theDataDim.getY(); y < theYDim; ++y) {
                    utils::Matrix<TDataType>::data_type theDstValue = 0;
                    for (int64_t theXInFilter = 0, theXDimOfFilter = theFilterDim.getX(); theXInFilter < theXDimOfFilter; ++theXInFilter) {
                        for (int64_t theYInFilter = 0, theYDimOfFilter = theFilterDim.getY(); theYInFilter < theYDimOfFilter; ++theYInFilter) {
                            TDataType theValue = 0;
                            for (int64_t theChannelIndex = 0, theChannelCount = theFilterDim.getZ(); theChannelIndex < theChannelCount; ++theChannelIndex) {
                                TDataType theValueFromSrc = 0;
                                try {
                                    int64_t theSrcOffset = m_input->offset(theSrcStartOffsetX + m_stride.getX() * x + theXInFilter,
                                                                           theSrcStartOffsetY + m_stride.getY() * y + theYInFilter,
                                                                           theChannelIndex,
                                                                           theInputIndex);
                                    theValueFromSrc = theSrcData[theSrcOffset];
                                } catch (std::runtime_error e) {
                                    ;
                                }
                                int64_t theFilterOffset = m_weights->offset(theXInFilter, theYInFilter, theChannelIndex, theOutputIndex);
                                theValue += theValueFromSrc * theFilterData[theFilterOffset];
                            }
                            theDstValue +=  theValue;
                        }
                    }
                    int64_t theDstOffset = m_data->offset(x, y, theOutputIndex, theInputIndex);
                    if (m_hasBias) {
                        theDstData[theDstOffset] = m_bias->getData()[m_bias->offset(theOutputIndex)];
                    } else {
                        theDstData[theDstOffset] = 0;
                    }
                    theDstData[theDstOffset] += theDstValue;
                }
            }
        }
    }
}

template<typename TDataType>
void
ConvolutionLayer<TDataType>::backward()
{

}

template<typename TDataType>
const utils::Matrix<TDataType>*
ConvolutionLayer<TDataType>::getOutput() const
{
    return m_data.get();
}

template<typename TDataType>
const utils::Matrix<TDataType>*
ConvolutionLayer<TDataType>::getDiff() const
{
    return m_diff.get();
}

template<typename TDataType>
void
ConvolutionLayer<TDataType>::setForwardInput(const utils::Matrix<TDataType> &inInput)
{
    const utils::Dimension &theInputDim = inInput.getDimension();
    if (theInputDim.axisCount() < 2)
        throw std::runtime_error("ConvolutionLayer::setForwardInput: invalid input dimension");
    int64_t theInputCount = theInputDim.getW();
    int64_t theInputChannel = theInputDim.getZ();
    theInputCount = theInputCount >= 1 ? theInputCount : 1;
    theInputChannel = theInputChannel >= 1 ? theInputChannel : 1;
    const int64_t theXForStride = theInputDim.getX() - m_kernelSize.getX() + m_padding.getY() * 2;
    const int64_t theYForStride = theInputDim.getY() - m_kernelSize.getY() + m_padding.getY() * 2;
    if (theXForStride % m_stride.getX())
        throw std::runtime_error("ConvolutionLayer::setForwardInput: stride cannot divide dimension X");
    if (theYForStride % m_stride.getY())
        throw std::runtime_error("ConvolutionLayer::setForwardInput: stride cannot divide dimension Y");
    m_data.reset(new utils::Matrix<TDataType>(utils::Dimension(theXForStride / m_stride.getX() + 1,
                                              theYForStride / m_stride.getY() + 1,
                                              m_numOfOutput,
                                              theInputCount)));
    m_weights.reset(new utils::Matrix<TDataType>(utils::Dimension(m_kernelSize.getX(),
                                                                  m_kernelSize.getY(),
                                                                  theInputChannel,
                                                                  m_numOfOutput)));
    m_bias.reset(new utils::Matrix<TDataType>(utils::Dimension(m_numOfOutput)));
    m_input = &inInput;
}

template<typename TDataType>
void
ConvolutionLayer<TDataType>::setBackwardDiff(const utils::Matrix<TDataType> &inDiff)
{

}

template class ConvolutionLayer<double>;

}
