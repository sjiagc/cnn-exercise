#ifndef TOY_CNN_SDK_LAYERS_INNER_PRODUCT_LAYER_HPP__
#define TOY_CNN_SDK_LAYERS_INNER_PRODUCT_LAYER_HPP__

#include "../Layer.hpp"

namespace layer
{

template<typename TDataType>
class InnerProductLayer final: public Layer<TDataType>
{
public:
    static const char *TYPE;
    static const char *CONFIG_NUM_OUTPUT;
    static const char *CONFIG_BIAS_TERM;
    static const char *RESTORE_WEIGHTS;
    static const char *RESTORE_BIAS;

    InnerProductLayer(const Layer<TDataType>::TLayerConfig &inConfig);

    virtual const char* getType() override;
    virtual void setMode(ComputeModeEnum inMode) override;
    virtual ComputeModeEnum getMode() override;
    virtual void connect(Layer<TDataType> &inDescendentLayer) override;
    virtual void restore(const TDataRestoring &inStoredData) override;
    virtual void forward() override;
    virtual void backward() override;
    virtual const utils::Matrix<TDataType>* getOutput() const override;
    virtual const utils::Matrix<TDataType>* getDiff() const override;

private:
    virtual ~InnerProductLayer();
    virtual void setForwardInput(const utils::Matrix<TDataType> &inInput) override;
    virtual void setBackwardDiff(const utils::Matrix<TDataType> &inDiff) override;

private:
    int64_t m_numOfOutput;
    bool m_hasBias;
    std::unique_ptr<utils::Matrix<TDataType>> m_weights;
    std::unique_ptr<utils::Matrix<TDataType>> m_data;
    std::unique_ptr<utils::Matrix<TDataType>> m_diff;
    std::unique_ptr<utils::Matrix<TDataType>> m_bias;
    const utils::Matrix<TDataType> *m_input;
    const utils::Matrix<TDataType> *m_diffAhead;
};

}

#endif // TOY_CNN_SDK_LAYERS_INNER_PRODUCT_LAYER_HPP__
