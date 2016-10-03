#include <layers/InnerProductLayer.hpp>

#include <stdexcept>

namespace layer
{

template<typename TDataType> const char* InnerProductLayer<TDataType>::TYPE = "InnerProduct";
template<typename TDataType> const char* InnerProductLayer<TDataType>::CONFIG_NUM_OUTPUT = "num_output";
template<typename TDataType> const char* InnerProductLayer<TDataType>::CONFIG_BIAS_TERM = "bias_term";
template<typename TDataType> const char* InnerProductLayer<TDataType>::RESTORE_WEIGHTS = "weights";
template<typename TDataType> const char* InnerProductLayer<TDataType>::RESTORE_BIAS = "bias";


template<typename TDataType>
InnerProductLayer<TDataType>::InnerProductLayer(const Layer<TDataType>::TLayerConfig &inConfig)
    : m_numOfOutput(1)
    , m_hasBias(false)
    , m_input(nullptr)
    , m_diffAhead(nullptr)
{
    if (inConfig.count(CONFIG_NUM_OUTPUT))
        m_numOfOutput = atoll(inConfig.find(CONFIG_NUM_OUTPUT)->second.c_str());
    if (inConfig.count(CONFIG_BIAS_TERM))
        m_hasBias = inConfig.find(CONFIG_BIAS_TERM)->second == "true";
}

template<typename TDataType>
InnerProductLayer<TDataType>::~InnerProductLayer()
{

}

template<typename TDataType>
const char*
InnerProductLayer<TDataType>::getType()
{
    return TYPE;
}

template<typename TDataType>
void
InnerProductLayer<TDataType>::connect(Layer<TDataType> &inDescendentLayer)
{
    inDescendentLayer.setForwardInput(*getOutput());
    setBackwardDiff(*inDescendentLayer.getDiff());
}

template<typename TDataType>
void
InnerProductLayer<TDataType>::restore(const TDataRestoring &inStoredData)
{
    if (inStoredData.count(RESTORE_WEIGHTS)) {
        const utils::Matrix<TDataType> *theWeights = inStoredData.find(RESTORE_WEIGHTS)->second;
        if (theWeights) {
            if (theWeights->getDimension() != m_weights->getDimension())
                throw std::runtime_error("InnerProductLayer:restore: weights dimensions do not match");
            *m_weights = *theWeights;
        }
    }
    if (m_hasBias && inStoredData.count(RESTORE_BIAS)) {
        const utils::Matrix<TDataType> *theBias= inStoredData.find(RESTORE_BIAS)->second;
        if (theBias) {
            if (theBias->getDimension() != m_bias->getDimension())
                throw std::runtime_error("InnerProductLayer:restore: bias dimensions do not match");
            *m_bias = *theBias;
        }
    }
}

template<typename TDataType>
void
InnerProductLayer<TDataType>::forward()
{
    const utils::Dimension &theDataDim = m_data->getDimension();
    const utils::Dimension &theSrcDim = m_input->getDimension();
    utils::Matrix<TDataType>::data_type *theDstData = m_data->getData();
    const utils::Matrix<TDataType>::data_type *theSrcData = m_input->getData();
    const utils::Matrix<TDataType>::data_type *theFilterData = m_weights->getData();

    for (int64_t theInputIndex = 0, theInputCount = theDataDim.getW(); theInputIndex < theInputCount; ++theInputIndex) {
        for (int64_t theOutputIndex = 0, theOutputCount = theDataDim.getZ(); theOutputIndex < theOutputCount; ++theOutputIndex) {
            int64_t theFilterIndex = 0;
            utils::Matrix<TDataType>::data_type theInnerProduct = 0;
            for (int64_t z = 0, theZDim = theSrcDim.getZ(); z < theZDim; ++z) {
                for (int64_t y = 0, theYDim = theSrcDim.getY(); y < theYDim; ++y) {
                    for (int64_t x = 0, theXDim = theSrcDim.getX(); x < theXDim; ++x) {
                        TDataType theValueFromSrc = 0;
                        try {
                            int64_t theSrcOffset = m_input->offset(x, y, z, theInputIndex);
                            theValueFromSrc = theSrcData[theSrcOffset];
                        } catch (std::runtime_error e) {
                            throw std::runtime_error("InnerProductLayer::forward: invalid source indices");
                        }
                        TDataType theValueFromFilter = 0;
                        try {
                            int64_t theFilterOffset = m_weights->offset(theFilterIndex, theOutputIndex);
                            theValueFromFilter = theFilterData[theFilterOffset];
                        } catch (std::runtime_error e) {
                            throw std::runtime_error("InnerProductLayer::forward: invalid weight indices");
                        }
                        theInnerProduct += theValueFromSrc * theValueFromFilter;
                        ++theFilterIndex;
                    }
                }
            }

            int64_t theDstOffset = m_data->offset(0, 0, theOutputIndex, theInputIndex);
            if (m_hasBias) {
                theDstData[theDstOffset] = m_bias->getData()[m_bias->offset(theOutputIndex)];
            } else {
                theDstData[theDstOffset] = 0;
            }
            theDstData[theDstOffset] += theInnerProduct;
        }
    }
}

template<typename TDataType>
void
InnerProductLayer<TDataType>::backward()
{

}

template<typename TDataType>
const utils::Matrix<TDataType>*
InnerProductLayer<TDataType>::getOutput() const
{
    return m_data.get();
}

template<typename TDataType>
const utils::Matrix<TDataType>*
InnerProductLayer<TDataType>::getDiff() const
{
    return m_diff.get();
}

template<typename TDataType>
void
InnerProductLayer<TDataType>::setForwardInput(const utils::Matrix<TDataType> &inInput)
{
    const utils::Dimension &theInputDim = inInput.getDimension();
    if (theInputDim.axisCount() < 1)
        throw std::runtime_error("InnerProductLayer::setForwardInput: invalid input dimension");
    int64_t theInputCount = theInputDim.getW();
    theInputCount = theInputCount >= 1 ? theInputCount : 1;
    int64_t theInputFeatureCount = theInputDim.getZ() * theInputDim.getY() * theInputDim.getX();
    if (theInputFeatureCount <= 0)
        throw std::runtime_error("InnerProductLayer::setForwardInput: invalid input feature count");
    m_data.reset(new utils::Matrix<TDataType>(utils::Dimension(1, 1, m_numOfOutput, theInputCount)));
    m_weights.reset(new utils::Matrix<TDataType>(
                        utils::Dimension(theInputDim.getX() * theInputDim.getY() * theInputDim.getZ(), m_numOfOutput)));
    m_bias.reset(new utils::Matrix<TDataType>(utils::Dimension(m_numOfOutput)));
    m_input = &inInput;
}

template<typename TDataType>
void
InnerProductLayer<TDataType>::setBackwardDiff(const utils::Matrix<TDataType> &inDiff)
{

}

template class InnerProductLayer<double>;

}
