#ifndef TOY_CNN_SDK_LAYERS_RELU_LAYER_HPP__
#define TOY_CNN_SDK_LAYERS_RELU_LAYER_HPP__

#include <Layer.hpp>
#include <utils/Matrix.hpp>

namespace layer
{

template<typename TDataType>
class ReluLayer final: public Layer<TDataType>
{
public:
    static const char *TYPE;
    static const char *CONFIG_NEGATIVE_SLOPE;

    ReluLayer(const typename Layer<TDataType>::TLayerConfig &inConfig);

    virtual const char* getType() override;
    virtual void setMode(ComputeModeEnum inMode) override;
    virtual ComputeModeEnum getMode() override;
    virtual void connect(Layer<TDataType> &inDescendentLayer) override;
    virtual void restore(const typename Layer<TDataType>::TDataRestoring &inStoredData) override;
    virtual void forward() override;
    virtual void backward() override;
    virtual const utils::Matrix<TDataType>* getOutput() const override;
    virtual const utils::Matrix<TDataType>* getDiff() const override;

private:
    virtual ~ReluLayer();
    virtual void setForwardInput(const utils::Matrix<TDataType> &inInput) override;
    virtual void setBackwardDiff(const utils::Matrix<TDataType> &inDiff) override;

    void forwardCPU();
    void forwardGPU();

private:
    TDataType m_negativeSlope;
    std::unique_ptr<utils::Matrix<TDataType>> m_data;
    std::unique_ptr<utils::Matrix<TDataType>> m_diff;
    const utils::Matrix<TDataType> *m_input;
    const utils::Matrix<TDataType> *m_diffAhead;

    ComputeModeEnum m_mode;
    void (ReluLayer::*m_forwardMethod)();
};

}

#endif // TOY_CNN_SDK_LAYERS_RELU_LAYER_HPP__
