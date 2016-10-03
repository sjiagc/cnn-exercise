#include "ReluLayerTests.hpp"

#include <layers/ReluLayer.hpp>

#include "PrintUtils.hpp"

#include <iostream>

namespace
{

using namespace test;

using utils::Dimension;
using utils::Matrix;

template<typename TDataType>
void
fillMatrix(Matrix<TDataType> &inMatrix)
{
    const Dimension &theMatrixDim = inMatrix.getDimension();
    Matrix<TDataType>::data_type *theData = inMatrix.getData();
    for (int64_t w = 0; w < theMatrixDim.getW(); ++w) {
        for (int64_t z = 0; z < theMatrixDim.getZ(); ++z) {
            for (int64_t y = 0; y < theMatrixDim.getY(); ++y) {
                for (int64_t x = 0; x < theMatrixDim.getX(); ++x) {
                    theData[inMatrix.offset(x, y, z, w)] = static_cast<double>(w * 1000 + z * 100 + y * 10 + x) * (x % 2 ? -1 : 1);
                }
            }
        }
    }
}

template<typename TDataType>
class InputLayer: public layer::Layer<TDataType>
{
public:
    InputLayer(const Matrix<TDataType> &inInput): m_input(inInput) {}

    virtual const char* getType() { return "input"; }
    virtual void connect(layer::Layer<TDataType> &inDescendentLayer) override
    {
        inDescendentLayer.setForwardInput(m_input);
    }
    virtual void restore(const TDataRestoring &inStoredData) override {}
    virtual void forward() override {}
    virtual void backward() override {}
    virtual const Matrix<TDataType>* getOutput() const override { return &m_input; }
    virtual const Matrix<TDataType>* getDiff() const override { return nullptr; }

private:
    virtual void setForwardInput(const Matrix<TDataType> &inInput) override {}
    virtual void setBackwardDiff(const Matrix<TDataType> &inDiff) override {}

    const Matrix<TDataType> &m_input;
};

bool
reluDefault()
{
    const Dimension theInputDim(9, 9, 1);

    Matrix<double> theInput(theInputDim);

    fillMatrix(theInput);

    InputLayer<double> theInputLayer(theInput);
    layer::Layer<double>::TUniqueHandle theReluLayer(layer::Layer<double>::create(layer::ReluLayer<double>::TYPE, layer::Layer<double>::TLayerConfig()));
    theInputLayer.connect(*theReluLayer.get());

    theReluLayer->forward();
    const Matrix<double> *theOutput = theReluLayer->getOutput();
    const Dimension &theOutputDim = theOutput->getDimension();

    if (theOutputDim != Dimension(9, 9, 1, 1) ||
            theOutput->getData()[theOutput->offset()] != 0 ||
            theOutput->getData()[theOutput->offset(4, 3)] != 34 ||
            theOutput->getData()[theOutput->offset(7, 7)] != 0) {
        std::cerr << "Relu.reluDefault test failed:" << std::endl
                  << "input:" << std::endl
                  << theInput << std::endl
                  << "output:" << std::endl
                  << *theOutput << std::endl;
    }

    return true;
}

bool
reluHalf()
{
    const Dimension theInputDim(9, 9, 3, 2);

    Matrix<double> theInput(theInputDim);

    fillMatrix(theInput);

    InputLayer<double> theInputLayer(theInput);
    layer::Layer<double>::TLayerConfig theReluConfig;
    theReluConfig.insert(std::make_pair(std::string(layer::ReluLayer<double>::CONFIG_NEGATIVE_SLOPE), "0.5"));
    layer::Layer<double>::TUniqueHandle theReluLayer(layer::Layer<double>::create(layer::ReluLayer<double>::TYPE, theReluConfig));
    theInputLayer.connect(*theReluLayer.get());

    theReluLayer->forward();
    const Matrix<double> *theOutput = theReluLayer->getOutput();
    const Dimension &theOutputDim = theOutput->getDimension();

    if (theOutputDim != Dimension(9, 9, 3, 2) ||
            theOutput->getData()[theOutput->offset()] != 0 ||
            theOutput->getData()[theOutput->offset(3, 3, 1, 1)] != -1133.0 / 2 ||
            theOutput->getData()[theOutput->offset(4, 4, 2, 1)] != 1244) {
        std::cerr << "Relu.reluHalf test failed:" << std::endl
                  << "input:" << std::endl
                  << theInput << std::endl
                  << "output:" << std::endl
                  << *theOutput << std::endl;
    }

    return true;
}

}

namespace test
{

void
ReluLayerTests::run()
{
    reluDefault();
    reluHalf();
}

}
