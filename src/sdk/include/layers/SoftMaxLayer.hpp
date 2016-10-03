#ifndef TOY_CNN_SDK_LAYERS_SOFT_MAX_LAYER_HPP__
#define TOY_CNN_SDK_LAYERS_SOFT_MAX_LAYER_HPP__

#include <Layer.hpp>
#include <utils/Matrix.hpp>

namespace layer
{

template<typename TDataType>
class SoftMaxLayer final: public Layer<TDataType>
{
public:
    static const char *TYPE;

    SoftMaxLayer(const Layer<TDataType>::TLayerConfig &inConfig);

    virtual const char* getType();
    virtual void connect(Layer<TDataType> &inDescendentLayer) override;
    virtual void restore(const TDataRestoring &inStoredData) override;
    virtual void forward() override;
    virtual void backward() override;
    virtual const utils::Matrix<TDataType>* getOutput() const override;
    virtual const utils::Matrix<TDataType>* getDiff() const override;

private:
    virtual ~SoftMaxLayer();
    virtual void setForwardInput(const utils::Matrix<TDataType> &inInput) override;
    virtual void setBackwardDiff(const utils::Matrix<TDataType> &inDiff) override;

private:
    TDataType m_negativeSlope;
    std::unique_ptr<utils::Matrix<TDataType>> m_data;
    std::unique_ptr<utils::Matrix<TDataType>> m_diff;
    const utils::Matrix<TDataType> *m_input;
    const utils::Matrix<TDataType> *m_diffAhead;
};

}

#endif // TOY_CNN_SDK_LAYERS_SOFT_MAX_LAYER_HPP__