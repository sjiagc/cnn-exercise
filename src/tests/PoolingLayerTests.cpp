#include "PoolingLayerTests.hpp"

#include <layers/PoolingLayer.hpp>

#include "ConstInputLayer.hpp"
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
    typename Matrix<TDataType>::data_type *theData = inMatrix.getMutableData();
    for (int64_t w = 0; w < theMatrixDim.getW(); ++w) {
        for (int64_t z = 0; z < theMatrixDim.getZ(); ++z) {
            for (int64_t y = 0; y < theMatrixDim.getY(); ++y) {
                for (int64_t x = 0; x < theMatrixDim.getX(); ++x) {
                    theData[inMatrix.offset(x, y, z, w)] = static_cast<double>(w * 1000 + z * 100 + y * 10 + x);
                }
            }
        }
    }
}

bool
pooling1()
{
    const Dimension theInputDim(9, 9, 1);

    Matrix<double> theInput(theInputDim);

    fillMatrix(theInput);

    layer::ConstInputLayer<double> theInputLayer(theInput);
    layer::Layer<double>::TLayerConfig thePoolingConfig;
    thePoolingConfig.insert(std::make_pair(std::string(layer::PoolingLayer<double>::CONFIG_KERNEL_SIZE), "2"));
    thePoolingConfig.insert(std::make_pair(std::string(layer::PoolingLayer<double>::CONFIG_STRIDE), "1"));
    thePoolingConfig.insert(std::make_pair(std::string(layer::PoolingLayer<double>::CONFIG_PADDING), "0"));
    layer::Layer<double>::TUniqueHandle thePoolingLayer(layer::Layer<double>::create(layer::PoolingLayer<double>::TYPE, thePoolingConfig));
    theInputLayer.connect(*thePoolingLayer.get());

    thePoolingLayer->forward();
    const Matrix<double> *theOutput = thePoolingLayer->getOutput();
    const Dimension &theOutputDim = theOutput->getDimension();

    if (theOutputDim != Dimension(8, 8, 1, 1) ||
            theOutput->getData()[theOutput->offset()] != 11 ||
            theOutput->getData()[theOutput->offset(3, 3)] != 44 ||
            theOutput->getData()[theOutput->offset(7, 7)] != 88) {
        std::cerr << "Pooling.pooling1 test failed:" << std::endl
                  << "input:" << std::endl
                  << theInput << std::endl
                  << "output:" << std::endl
                  << *theOutput << std::endl;
    }

    return true;
}

bool
pooling2()
{
    const Dimension theInputDim(9, 9, 3, 2);

    Matrix<double> theInput(theInputDim);

    fillMatrix(theInput);

    layer::ConstInputLayer<double> theInputLayer(theInput);
    layer::Layer<double>::TLayerConfig thePoolingConfig;
    thePoolingConfig.insert(std::make_pair(std::string(layer::PoolingLayer<double>::CONFIG_KERNEL_SIZE), "2"));
    thePoolingConfig.insert(std::make_pair(std::string(layer::PoolingLayer<double>::CONFIG_STRIDE), "2"));
    thePoolingConfig.insert(std::make_pair(std::string(layer::PoolingLayer<double>::CONFIG_PADDING), "0"));
    layer::Layer<double>::TUniqueHandle thePoolingLayer(layer::Layer<double>::create(layer::PoolingLayer<double>::TYPE, thePoolingConfig));
    theInputLayer.connect(*thePoolingLayer.get());

    thePoolingLayer->forward();
    const Matrix<double> *theOutput = thePoolingLayer->getOutput();
    const Dimension &theOutputDim = theOutput->getDimension();

    if (theOutputDim != Dimension(5, 5, 3, 2) ||
            theOutput->getData()[theOutput->offset()] != 11 ||
            theOutput->getData()[theOutput->offset(3, 3, 1, 1)] != 1177 ||
            theOutput->getData()[theOutput->offset(4, 4, 2, 1)] != 1288) {
        std::cerr << "Pooling.pooling2 test failed:" << std::endl
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
PoolingLayerTests::run()
{
    pooling1();
    pooling2();
}

}
