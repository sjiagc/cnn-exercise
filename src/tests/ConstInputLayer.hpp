#ifndef TOY_CNN_TESTS_CONST_INPUT_LAYER_HPP__
#define TOY_CNN_TESTS_CONST_INPUT_LAYER_HPP__

#include <Layer.hpp>
#include <utils/Matrix.hpp>

namespace layer {

template<typename TDataType>
class ConstInputLayer: public Layer<TDataType>
{
public:
    ConstInputLayer(const utils::Matrix<TDataType> &inInput): m_input(inInput) {}

    virtual const char* getType() { return "ConstInput"; }
    virtual void setMode(ComputeModeEnum inMode) {}
    virtual ComputeModeEnum getMode() { return ComputeModeEnum::CPU; }
    virtual void connect(layer::Layer<TDataType> &inDescendentLayer) override
    {
        inDescendentLayer.setForwardInput(m_input);
    }
    virtual void restore(const typename Layer<TDataType>::TDataRestoring&) override {}
    virtual void forward() override {}
    virtual void backward() override {}
    virtual const utils::Matrix<TDataType>* getOutput() const override { return &m_input; }
    virtual const utils::Matrix<TDataType>* getDiff() const override { return nullptr; }

private:
    virtual void setForwardInput(const utils::Matrix<TDataType> &inInput) override {}
    virtual void setBackwardDiff(const utils::Matrix<TDataType> &inDiff) override {}

    const utils::Matrix<TDataType> &m_input;
};

}

#endif // TOY_CNN_TESTS_CONST_INPUT_LAYER_HPP__
