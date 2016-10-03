#ifndef TOY_CNN_SDK_LAYERS_POOLING_LAYER_HPP__
#define TOY_CNN_SDK_LAYERS_POOLING_LAYER_HPP__

#include <Layer.hpp>
#include <utils/Matrix.hpp>

namespace layer
{

template<typename TDataType>
class PoolingLayer final: public Layer<TDataType>
{
public:
    static const char *TYPE;
    static const char *CONFIG_POOL;
    static const char *CONFIG_KERNEL_SIZE;
    static const char *CONFIG_KERNEL_SIZE_X;
    static const char *CONFIG_KERNEL_SIZE_Y;
    static const char *CONFIG_STRIDE;
    static const char *CONFIG_STRIDE_X;
    static const char *CONFIG_STRIDE_Y;
    static const char *CONFIG_PADDING;
    static const char *CONFIG_PADDING_X;
    static const char *CONFIG_PADDING_Y;

    PoolingLayer(const Layer<TDataType>::TLayerConfig &inConfig);

    virtual const char* getType();
    virtual void connect(Layer<TDataType> &inDescendentLayer) override;
    virtual void restore(const TDataRestoring &inStoredData) override;
    virtual void forward() override;
    virtual void backward() override;
    virtual const utils::Matrix<TDataType>* getOutput() const override;
    virtual const utils::Matrix<TDataType>* getDiff() const override;

private:
    virtual ~PoolingLayer();
    virtual void setForwardInput(const utils::Matrix<TDataType> &inInput) override;
    virtual void setBackwardDiff(const utils::Matrix<TDataType> &inDiff) override;

private:
    int64_t m_numOfOutput;
    utils::Dimension m_kernelSize;
    utils::Dimension m_padding;
    utils::Dimension m_stride;
    std::unique_ptr<utils::Matrix<TDataType>> m_data;
    std::unique_ptr<utils::Matrix<TDataType>> m_diff;
    const utils::Matrix<TDataType> *m_input;
    const utils::Matrix<TDataType> *m_diffAhead;
};

}

#endif // TOY_CNN_SDK_LAYERS_POOLING_LAYER_HPP__
