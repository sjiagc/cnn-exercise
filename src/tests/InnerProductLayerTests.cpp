#include "InnerProductLayerTests.hpp"

#include <utils/Matrix.hpp>
#include <Layer.hpp>
#include <layers/InnerProductLayer.hpp>

#include "ConstInputLayer.hpp"
#include "PrintUtils.hpp"

#include <stdexcept>
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
    Matrix<TDataType>::data_type *theData = inMatrix.getMutableData();
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
innerProduct1D()
{
    const Dimension theInputDim(9);

    Matrix<double> theInput(theInputDim);
    Matrix<double> theWeights(Dimension(9, 2));

    fillMatrix(theInput);

    {
        theWeights.getMutableData()[theWeights.offset(0)] = 1;
        theWeights.getMutableData()[theWeights.offset(1)] = 2;
        theWeights.getMutableData()[theWeights.offset(2)] = 1;
        theWeights.getMutableData()[theWeights.offset(3)] = 3;
        theWeights.getMutableData()[theWeights.offset(4)] = 1;
        theWeights.getMutableData()[theWeights.offset(5)] = 4;
        theWeights.getMutableData()[theWeights.offset(6)] = 1;
        theWeights.getMutableData()[theWeights.offset(7)] = 5;
        theWeights.getMutableData()[theWeights.offset(8)] = 1;

        theWeights.getMutableData()[theWeights.offset(0, 1)] = 2;
        theWeights.getMutableData()[theWeights.offset(1, 1)] = 3;
        theWeights.getMutableData()[theWeights.offset(2, 1)] = 2;
        theWeights.getMutableData()[theWeights.offset(3, 1)] = 4;
        theWeights.getMutableData()[theWeights.offset(4, 1)] = 2;
        theWeights.getMutableData()[theWeights.offset(5, 1)] = 5;
        theWeights.getMutableData()[theWeights.offset(6, 1)] = 2;
        theWeights.getMutableData()[theWeights.offset(7, 1)] = 6;
        theWeights.getMutableData()[theWeights.offset(8, 1)] = 2;
    }

    layer::ConstInputLayer<double> theInputLayer(theInput);
    layer::Layer<double>::TLayerConfig theIPConfig;
    theIPConfig.insert(std::make_pair(std::string(layer::InnerProductLayer<double>::CONFIG_NUM_OUTPUT), "2"));
    layer::Layer<double>::TUniqueHandle theIPLayer(layer::Layer<double>::create(layer::InnerProductLayer<double>::TYPE, theIPConfig));
    theInputLayer.connect(*theIPLayer.get());
    layer::Layer<double>::TDataRestoring theDataRestoring;
    theDataRestoring[layer::InnerProductLayer<double>::RESTORE_WEIGHTS] = &theWeights;
    theIPLayer->restore(theDataRestoring);

    theIPLayer->forward();
    const Matrix<double> *theOutput = theIPLayer->getOutput();
    const Dimension &theOutputDim = theOutput->getDimension();

    if (theOutputDim != Dimension(1, 1, 2, 1) ||
            theOutput->getData()[theOutput->offset(0, 0, 0, 0)] != 86 ||
            theOutput->getData()[theOutput->offset(0, 0, 1, 0)] != 122) {
        std::cerr << "InnerProduct.innerProduct1D test failed:" << std::endl
                  << "input:" << std::endl
                  << theInput << std::endl
                  << "filter:" << std::endl
                  << theWeights << std::endl
                  << "output:" << std::endl
                  << *theOutput << std::endl;
    }

    return true;
}

bool
innerProduct2D()
{
    const Dimension theInputDim(9, 6);

    Matrix<double> theInput(theInputDim);
    Matrix<double> theWeights(Dimension(54, 2));

    fillMatrix(theInput);
    fillMatrix(theWeights);

    layer::ConstInputLayer<double> theInputLayer(theInput);
    layer::Layer<double>::TLayerConfig theIPConfig;
    theIPConfig.insert(std::make_pair(std::string(layer::InnerProductLayer<double>::CONFIG_NUM_OUTPUT), "2"));
    layer::Layer<double>::TUniqueHandle theIPLayer(layer::Layer<double>::create(layer::InnerProductLayer<double>::TYPE, theIPConfig));
    theInputLayer.connect(*theIPLayer.get());
    layer::Layer<double>::TDataRestoring theDataRestoring;
    theDataRestoring[layer::InnerProductLayer<double>::RESTORE_WEIGHTS] = &theWeights;
    theIPLayer->restore(theDataRestoring);

    theIPLayer->forward();
    const Matrix<double> *theOutput = theIPLayer->getOutput();
    const Dimension &theOutputDim = theOutput->getDimension();

    if (theOutputDim != Dimension(1, 1, 2, 1) ||
            theOutput->getData()[theOutput->offset(0, 0, 0, 0)] != 56034 ||
            theOutput->getData()[theOutput->offset(0, 0, 1, 0)] != 71694) {
        std::cerr << "InnerProduct.innerProduct1D test failed:" << std::endl
                  << "input:" << std::endl
                  << theInput << std::endl
                  << "filter:" << std::endl
                  << theWeights << std::endl
                  << "output:" << std::endl
                  << *theOutput << std::endl;
    }

    return true;
}

bool
innerProduct3D()
{
    const Dimension theInputDim(9, 4, 3);

    Matrix<double> theInput(theInputDim);
    Matrix<double> theWeights(Dimension(108, 2));

    fillMatrix(theInput);
    fillMatrix(theWeights);

    layer::ConstInputLayer<double> theInputLayer(theInput);
    layer::Layer<double>::TLayerConfig theIPConfig;
    theIPConfig.insert(std::make_pair(std::string(layer::InnerProductLayer<double>::CONFIG_NUM_OUTPUT), "2"));
    layer::Layer<double>::TUniqueHandle theIPLayer(layer::Layer<double>::create(layer::InnerProductLayer<double>::TYPE, theIPConfig));
    theInputLayer.connect(*theIPLayer.get());
    layer::Layer<double>::TDataRestoring theDataRestoring;
    theDataRestoring[layer::InnerProductLayer<double>::RESTORE_WEIGHTS] = &theWeights;
    theIPLayer->restore(theDataRestoring);

    theIPLayer->forward();
    const Matrix<double> *theOutput = theIPLayer->getOutput();
    const Dimension &theOutputDim = theOutput->getDimension();

    if (theOutputDim != Dimension(1, 1, 2, 1) ||
            theOutput->getData()[theOutput->offset(0, 0, 0, 0)] != 959652 ||
            theOutput->getData()[theOutput->offset(0, 0, 1, 0)] != 1088172) {
        std::cerr << "InnerProduct.innerProduct1D test failed:" << std::endl
                  << "input:" << std::endl
                  << theInput << std::endl
                  << "filter:" << std::endl
                  << theWeights << std::endl
                  << "output:" << std::endl
                  << *theOutput << std::endl;
    }

    return true;
}

bool
innerProduct1DBias()
{
    const Dimension theInputDim(9);

    Matrix<double> theInput(theInputDim);
    Matrix<double> theWeights(Dimension(9, 2));
    Matrix<double> theBias(Dimension(2));

    fillMatrix(theInput);

    {
        theWeights.getMutableData()[theWeights.offset(0)] = 1;
        theWeights.getMutableData()[theWeights.offset(1)] = 2;
        theWeights.getMutableData()[theWeights.offset(2)] = 1;
        theWeights.getMutableData()[theWeights.offset(3)] = 3;
        theWeights.getMutableData()[theWeights.offset(4)] = 1;
        theWeights.getMutableData()[theWeights.offset(5)] = 4;
        theWeights.getMutableData()[theWeights.offset(6)] = 1;
        theWeights.getMutableData()[theWeights.offset(7)] = 5;
        theWeights.getMutableData()[theWeights.offset(8)] = 1;

        theWeights.getMutableData()[theWeights.offset(0, 1)] = 2;
        theWeights.getMutableData()[theWeights.offset(1, 1)] = 3;
        theWeights.getMutableData()[theWeights.offset(2, 1)] = 2;
        theWeights.getMutableData()[theWeights.offset(3, 1)] = 4;
        theWeights.getMutableData()[theWeights.offset(4, 1)] = 2;
        theWeights.getMutableData()[theWeights.offset(5, 1)] = 5;
        theWeights.getMutableData()[theWeights.offset(6, 1)] = 2;
        theWeights.getMutableData()[theWeights.offset(7, 1)] = 6;
        theWeights.getMutableData()[theWeights.offset(8, 1)] = 2;

        theBias.getMutableData()[theBias.offset(0)] = 1;
        theBias.getMutableData()[theBias.offset(1)] = 2.2;
    }

    layer::ConstInputLayer<double> theInputLayer(theInput);
    layer::Layer<double>::TLayerConfig theIPConfig;
    theIPConfig.insert(std::make_pair(std::string(layer::InnerProductLayer<double>::CONFIG_NUM_OUTPUT), "2"));
    theIPConfig.insert(std::make_pair(std::string(layer::InnerProductLayer<double>::CONFIG_BIAS_TERM), "true"));
    layer::Layer<double>::TUniqueHandle theIPLayer(layer::Layer<double>::create(layer::InnerProductLayer<double>::TYPE, theIPConfig));
    theInputLayer.connect(*theIPLayer.get());
    layer::Layer<double>::TDataRestoring theDataRestoring;
    theDataRestoring[layer::InnerProductLayer<double>::RESTORE_WEIGHTS] = &theWeights;
    theDataRestoring[layer::InnerProductLayer<double>::RESTORE_BIAS] = &theBias;
    theIPLayer->restore(theDataRestoring);

    theIPLayer->forward();
    const Matrix<double> *theOutput = theIPLayer->getOutput();
    const Dimension &theOutputDim = theOutput->getDimension();

    if (theOutputDim != Dimension(1, 1, 2, 1) ||
            theOutput->getData()[theOutput->offset(0, 0, 0, 0)] != 87 ||
            theOutput->getData()[theOutput->offset(0, 0, 1, 0)] != 124.2) {
        std::cerr << "InnerProduct.innerProduct1D test failed:" << std::endl
                  << "input:" << std::endl
                  << theInput << std::endl
                  << "filter:" << std::endl
                  << theWeights << std::endl
                  << "output:" << std::endl
                  << *theOutput << std::endl;
    }

    return true;
}

}

namespace test
{

void
InnerProductLayerTests::run()
{
    innerProduct1D();
    innerProduct2D();
    innerProduct3D();
    innerProduct1DBias();
}

}

