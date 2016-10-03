#include "SoftMaxLayerTests.hpp"

#include <layers/SoftMaxLayer.hpp>

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
softMax1D()
{
    const Dimension theInputDim(10);

    Matrix<double> theInput(theInputDim);

    fillMatrix(theInput);

    InputLayer<double> theInputLayer(theInput);
    layer::Layer<double>::TUniqueHandle theSoftMaxLayer(layer::Layer<double>::create(layer::SoftMaxLayer<double>::TYPE, layer::Layer<double>::TLayerConfig()));
    theInputLayer.connect(*theSoftMaxLayer.get());

    theSoftMaxLayer->forward();
    const Matrix<double> *theOutput = theSoftMaxLayer->getOutput();
    const Dimension &theOutputDim = theOutput->getDimension();

    bool theIsFailed = false;
    if (theOutputDim != Dimension(10)) {
        theIsFailed = true;
    } else {
        utils::Matrix<double>::data_type theSum = 0;
        for (int64_t w = 0, theWDim = theOutputDim.getW(); w < theWDim; ++w) {
            for (int64_t z = 0, theZDim = theOutputDim.getZ(); z < theZDim; ++z) {
                for (int64_t y = 0, theYDim = theOutputDim.getY(); y < theYDim; ++y) {
                    for (int64_t x = 0, theXDim = theOutputDim.getX(); x < theXDim; ++x) {
                        theSum += theOutput->getData()[theOutput->offset(x, y, z, w)];
                    }
                }
            }
        }
        if (fabs(theSum - 1) >= 1e-6)
            theIsFailed = true;
    }

    if (theIsFailed) {
        std::cerr << "SoftMax.softMax1D test failed:" << std::endl
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
SoftMaxLayerTests::run()
{
    softMax1D();
}

}