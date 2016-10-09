#include "ConvolutionLayerTests.hpp"

#include <utils/Matrix.hpp>
#include <Layer.hpp>
#include <layers/ConvolutionLayer.hpp>

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
conv1()
{
    const int64_t theKernelSize = 3;

    const Dimension theInputDim(9, 9, 1);
    const Dimension theFilterDim(theKernelSize, theKernelSize, 1);

    Matrix<double> theInput(theInputDim);
    Matrix<double> theFilter(theFilterDim);

    fillMatrix(theInput);

    {
        Matrix<double>::data_type *theFilterData = theFilter.getMutableData();
        theFilterData[theFilter.offset(0, 0)] = 0;
        theFilterData[theFilter.offset(0, 1)] = 1;
        theFilterData[theFilter.offset(0, 2)] = 0;
        theFilterData[theFilter.offset(1, 0)] = 1;
        theFilterData[theFilter.offset(1, 1)] = 0;
        theFilterData[theFilter.offset(1, 2)] = 1;
        theFilterData[theFilter.offset(2, 0)] = 0;
        theFilterData[theFilter.offset(2, 1)] = 1;
        theFilterData[theFilter.offset(2, 2)] = 0;
    }

    layer::ConstInputLayer<double> theInputLayer(theInput);
    layer::Layer<double>::TLayerConfig theConvConfig;
    theConvConfig.insert(std::make_pair(std::string(layer::ConvolutionLayer<double>::CONFIG_NUM_OUTPUT), "1"));
    theConvConfig.insert(std::make_pair(std::string(layer::ConvolutionLayer<double>::CONFIG_KERNEL_SIZE), "3"));
    theConvConfig.insert(std::make_pair(std::string(layer::ConvolutionLayer<double>::CONFIG_STRIDE), "1"));
    theConvConfig.insert(std::make_pair(std::string(layer::ConvolutionLayer<double>::CONFIG_PADDING), "0"));
    layer::Layer<double>::TUniqueHandle theConvLayer(layer::Layer<double>::create(layer::ConvolutionLayer<double>::TYPE, theConvConfig));
    theInputLayer.connect(*theConvLayer.get());
    layer::Layer<double>::TDataRestoring theDataRestoring;
    theDataRestoring[layer::ConvolutionLayer<double>::RESTORE_WEIGHTS] = &theFilter;
    theConvLayer->restore(theDataRestoring);

    theConvLayer->forward();
    const Matrix<double> *theOutput = theConvLayer->getOutput();
    const Dimension &theOutputDim = theOutput->getDimension();

    if (theOutputDim != Dimension(7, 7, 1, 1) ||
            theOutput->getData()[theOutput->offset()] != 44 ||
            theOutput->getData()[theOutput->offset(3, 3)] != 176 ||
            theOutput->getData()[theOutput->offset(6, 6)] != 308) {
        std::cerr << "Convolution.conv1 test failed:" << std::endl
                  << "input:" << std::endl
                  << theInput << std::endl
                  << "filter:" << std::endl
                  << theFilter << std::endl
                  << "output:" << std::endl
                  << *theOutput << std::endl;
    }

    return true;
}

bool
conv2()
{
    const int64_t theKernelSize = 3;

    const Dimension theInputDim(9, 9, 1, 2);
    const Dimension theFilterDim(theKernelSize, theKernelSize, 1);

    Matrix<double> theInput(theInputDim);
    Matrix<double> theFilter(theFilterDim);

    fillMatrix(theInput);

    {
        Matrix<double>::data_type *theFilterData = theFilter.getMutableData();
        theFilterData[theFilter.offset(0, 0)] = 0;
        theFilterData[theFilter.offset(0, 1)] = 1;
        theFilterData[theFilter.offset(0, 2)] = 0;
        theFilterData[theFilter.offset(1, 0)] = 1;
        theFilterData[theFilter.offset(1, 1)] = 0;
        theFilterData[theFilter.offset(1, 2)] = 1;
        theFilterData[theFilter.offset(2, 0)] = 0;
        theFilterData[theFilter.offset(2, 1)] = 1;
        theFilterData[theFilter.offset(2, 2)] = 0;
    }

    layer::ConstInputLayer<double> theInputLayer(theInput);
    layer::Layer<double>::TLayerConfig theConvConfig;
    theConvConfig.insert(std::make_pair(std::string(layer::ConvolutionLayer<double>::CONFIG_NUM_OUTPUT), "1"));
    theConvConfig.insert(std::make_pair(std::string(layer::ConvolutionLayer<double>::CONFIG_KERNEL_SIZE), "3"));
    theConvConfig.insert(std::make_pair(std::string(layer::ConvolutionLayer<double>::CONFIG_STRIDE), "1"));
    theConvConfig.insert(std::make_pair(std::string(layer::ConvolutionLayer<double>::CONFIG_PADDING), "0"));
    layer::Layer<double>::TUniqueHandle theConvLayer(layer::Layer<double>::create(layer::ConvolutionLayer<double>::TYPE, theConvConfig));
    theInputLayer.connect(*theConvLayer.get());
    layer::Layer<double>::TDataRestoring theDataRestoring;
    theDataRestoring[layer::ConvolutionLayer<double>::RESTORE_WEIGHTS] = &theFilter;
    theConvLayer->restore(theDataRestoring);

    theConvLayer->forward();
    const Matrix<double> *theOutput = theConvLayer->getOutput();
    const Dimension &theOutputDim = theOutput->getDimension();

    if (theOutputDim != Dimension(7, 7, 1, 2) ||
            theOutput->getData()[theOutput->offset()] != 44 ||
            theOutput->getData()[theOutput->offset(3, 3)] != 176 ||
            theOutput->getData()[theOutput->offset(0, 0, 0, 1)] != 4044 ||
            theOutput->getData()[theOutput->offset(6, 6, 0, 1)] != 4308) {
        std::cerr << "Convolution.conv2 test failed:" << std::endl
                  << "input:" << std::endl
                  << theInput << std::endl
                  << "filter:" << std::endl
                  << theFilter << std::endl
                  << "output:" << std::endl
                  << *theOutput << std::endl;
    }

    return true;
}

bool
conv3()
{
    const int64_t theKernelSize = 3;

    const Dimension theInputDim(9, 9, 1, 2);
    const Dimension theFilterDim(theKernelSize, theKernelSize, 1, 3);

    Matrix<double> theInput(theInputDim);
    Matrix<double> theFilter(theFilterDim);

    fillMatrix(theInput);

    {
        Matrix<double>::data_type *theFilterData = theFilter.getMutableData();
        theFilterData[theFilter.offset(0, 0)] = 0;
        theFilterData[theFilter.offset(0, 1)] = 1;
        theFilterData[theFilter.offset(0, 2)] = 0;
        theFilterData[theFilter.offset(1, 0)] = 1;
        theFilterData[theFilter.offset(1, 1)] = 0;
        theFilterData[theFilter.offset(1, 2)] = 1;
        theFilterData[theFilter.offset(2, 0)] = 0;
        theFilterData[theFilter.offset(2, 1)] = 1;
        theFilterData[theFilter.offset(2, 2)] = 0;

        theFilterData[theFilter.offset(0, 0, 0, 1)] = 1;
        theFilterData[theFilter.offset(0, 1, 0, 1)] = 1;
        theFilterData[theFilter.offset(0, 2, 0, 1)] = 1;
        theFilterData[theFilter.offset(1, 0, 0, 1)] = 1;
        theFilterData[theFilter.offset(1, 1, 0, 1)] = -8;
        theFilterData[theFilter.offset(1, 2, 0, 1)] = 1;
        theFilterData[theFilter.offset(2, 0, 0, 1)] = 1;
        theFilterData[theFilter.offset(2, 1, 0, 1)] = 1;
        theFilterData[theFilter.offset(2, 2, 0, 1)] = 1;

        theFilterData[theFilter.offset(0, 0, 0, 2)] = -1;
        theFilterData[theFilter.offset(0, 1, 0, 2)] = 0;
        theFilterData[theFilter.offset(0, 2, 0, 2)] = 1;
        theFilterData[theFilter.offset(1, 0, 0, 2)] = -2;
        theFilterData[theFilter.offset(1, 1, 0, 2)] = 0;
        theFilterData[theFilter.offset(1, 2, 0, 2)] = 2;
        theFilterData[theFilter.offset(2, 0, 0, 2)] = -1;
        theFilterData[theFilter.offset(2, 1, 0, 2)] = 0;
        theFilterData[theFilter.offset(2, 2, 0, 2)] = 1;
    }

    layer::ConstInputLayer<double> theInputLayer(theInput);
    layer::Layer<double>::TLayerConfig theConvConfig;
    theConvConfig.insert(std::make_pair(std::string(layer::ConvolutionLayer<double>::CONFIG_NUM_OUTPUT), "3"));
    theConvConfig.insert(std::make_pair(std::string(layer::ConvolutionLayer<double>::CONFIG_KERNEL_SIZE), "3"));
    theConvConfig.insert(std::make_pair(std::string(layer::ConvolutionLayer<double>::CONFIG_STRIDE), "1"));
    theConvConfig.insert(std::make_pair(std::string(layer::ConvolutionLayer<double>::CONFIG_PADDING), "0"));
    layer::Layer<double>::TUniqueHandle theConvLayer(layer::Layer<double>::create(layer::ConvolutionLayer<double>::TYPE, theConvConfig));
    theInputLayer.connect(*theConvLayer.get());
    layer::Layer<double>::TDataRestoring theDataRestoring;
    theDataRestoring[layer::ConvolutionLayer<double>::RESTORE_WEIGHTS] = &theFilter;
    theConvLayer->restore(theDataRestoring);

    theConvLayer->forward();
    const Matrix<double> *theOutput = theConvLayer->getOutput();
    const Dimension &theOutputDim = theOutput->getDimension();

    if (theOutputDim != Dimension(7, 7, 3, 2) ||
            theOutput->getData()[theOutput->offset()] != 44 ||
            theOutput->getData()[theOutput->offset(3, 3)] != 176 ||
            theOutput->getData()[theOutput->offset(0, 0, 0, 1)] != 4044 ||
            theOutput->getData()[theOutput->offset(3, 6, 1, 1)] != 0 ||
            theOutput->getData()[theOutput->offset(3, 3, 1, 0)] != 0 ||
            theOutput->getData()[theOutput->offset(0, 6, 1, 1)] != 0 ||
            theOutput->getData()[theOutput->offset(0, 0, 2, 0)] != 80 ||
            theOutput->getData()[theOutput->offset(3, 3, 2, 0)] != 80 ||
            theOutput->getData()[theOutput->offset(6, 4, 2, 1)] != 80) {
        std::cerr << "Convolution.conv3 test failed:" << std::endl
                  << "input:" << std::endl
                  << theInput << std::endl
                  << "filter:" << std::endl
                  << theFilter << std::endl
                  << "output:" << std::endl
                  << *theOutput << std::endl;
    }

    return true;
}

bool
convChannel()
{
    const int64_t theKernelSize = 3;

    const Dimension theInputDim(9, 9, 3, 1);
    const Dimension theFilterDim(theKernelSize, theKernelSize, 3, 1);

    Matrix<double> theInput(theInputDim);
    Matrix<double> theFilter(theFilterDim);

    fillMatrix(theInput);

    {
        Matrix<double>::data_type *theFilterData = theFilter.getMutableData();
        theFilterData[theFilter.offset(0, 0, 0)] = 0;
        theFilterData[theFilter.offset(0, 1, 0)] = 1;
        theFilterData[theFilter.offset(0, 2, 0)] = 0;
        theFilterData[theFilter.offset(1, 0, 0)] = 1;
        theFilterData[theFilter.offset(1, 1, 0)] = 0;
        theFilterData[theFilter.offset(1, 2, 0)] = 1;
        theFilterData[theFilter.offset(2, 0, 0)] = 0;
        theFilterData[theFilter.offset(2, 1, 0)] = 1;
        theFilterData[theFilter.offset(2, 2, 0)] = 0;

        theFilterData[theFilter.offset(0, 0, 1)] = 0;
        theFilterData[theFilter.offset(0, 1, 1)] = 1;
        theFilterData[theFilter.offset(0, 2, 1)] = 0;
        theFilterData[theFilter.offset(1, 0, 1)] = 1;
        theFilterData[theFilter.offset(1, 1, 1)] = 0;
        theFilterData[theFilter.offset(1, 2, 1)] = 1;
        theFilterData[theFilter.offset(2, 0, 1)] = 0;
        theFilterData[theFilter.offset(2, 1, 1)] = 1;
        theFilterData[theFilter.offset(2, 2, 1)] = 0;

        theFilterData[theFilter.offset(0, 0, 2)] = -1;
        theFilterData[theFilter.offset(0, 1, 2)] = 0;
        theFilterData[theFilter.offset(0, 2, 2)] = 1;
        theFilterData[theFilter.offset(1, 0, 2)] = -2;
        theFilterData[theFilter.offset(1, 1, 2)] = 0;
        theFilterData[theFilter.offset(1, 2, 2)] = 2;
        theFilterData[theFilter.offset(2, 0, 2)] = -1;
        theFilterData[theFilter.offset(2, 1, 2)] = 0;
        theFilterData[theFilter.offset(2, 2, 2)] = 1;
    }

    layer::ConstInputLayer<double> theInputLayer(theInput);
    layer::Layer<double>::TLayerConfig theConvConfig;
    theConvConfig.insert(std::make_pair(std::string(layer::ConvolutionLayer<double>::CONFIG_NUM_OUTPUT), "1"));
    theConvConfig.insert(std::make_pair(std::string(layer::ConvolutionLayer<double>::CONFIG_KERNEL_SIZE), "3"));
    theConvConfig.insert(std::make_pair(std::string(layer::ConvolutionLayer<double>::CONFIG_STRIDE), "1"));
    theConvConfig.insert(std::make_pair(std::string(layer::ConvolutionLayer<double>::CONFIG_PADDING), "0"));
    layer::Layer<double>::TUniqueHandle theConvLayer(layer::Layer<double>::create(layer::ConvolutionLayer<double>::TYPE, theConvConfig));
    theInputLayer.connect(*theConvLayer.get());
    layer::Layer<double>::TDataRestoring theDataRestoring;
    theDataRestoring[layer::ConvolutionLayer<double>::RESTORE_WEIGHTS] = &theFilter;
    theConvLayer->restore(theDataRestoring);

    theConvLayer->forward();
    const Matrix<double> *theOutput = theConvLayer->getOutput();
    const Dimension &theOutputDim = theOutput->getDimension();

    if (theOutputDim != Dimension(7, 7, 1, 1) ||
            theOutput->getData()[theOutput->offset()] != 568 ||
            theOutput->getData()[theOutput->offset(3, 3)] != 832 ||
            theOutput->getData()[theOutput->offset(6, 6)] != 1096) {
        std::cerr << "Convolution.convChannel test failed:" << std::endl
                  << "input:" << std::endl
                  << theInput << std::endl
                  << "filter:" << std::endl
                  << theFilter << std::endl
                  << "output:" << std::endl
                  << *theOutput << std::endl;
    }

    return true;
}

bool
convPadding()
{
    const int64_t theKernelSize = 3;

    const Dimension theInputDim(9, 9, 1);
    const Dimension theFilterDim(theKernelSize, theKernelSize, 1);

    Matrix<double> theInput(theInputDim);
    Matrix<double> theFilter(theFilterDim);

    fillMatrix(theInput);

    {
        Matrix<double>::data_type *theFilterData = theFilter.getMutableData();
        theFilterData[theFilter.offset(0, 0)] = 0;
        theFilterData[theFilter.offset(0, 1)] = 1;
        theFilterData[theFilter.offset(0, 2)] = 0;
        theFilterData[theFilter.offset(1, 0)] = 1;
        theFilterData[theFilter.offset(1, 1)] = 0;
        theFilterData[theFilter.offset(1, 2)] = 1;
        theFilterData[theFilter.offset(2, 0)] = 0;
        theFilterData[theFilter.offset(2, 1)] = 1;
        theFilterData[theFilter.offset(2, 2)] = 0;
    }

    layer::ConstInputLayer<double> theInputLayer(theInput);
    layer::Layer<double>::TLayerConfig theConvConfig;
    theConvConfig.insert(std::make_pair(std::string(layer::ConvolutionLayer<double>::CONFIG_NUM_OUTPUT), "1"));
    theConvConfig.insert(std::make_pair(std::string(layer::ConvolutionLayer<double>::CONFIG_KERNEL_SIZE), "3"));
    theConvConfig.insert(std::make_pair(std::string(layer::ConvolutionLayer<double>::CONFIG_STRIDE), "1"));
    theConvConfig.insert(std::make_pair(std::string(layer::ConvolutionLayer<double>::CONFIG_PADDING), "1"));
    layer::Layer<double>::TUniqueHandle theConvLayer(layer::Layer<double>::create(layer::ConvolutionLayer<double>::TYPE, theConvConfig));
    theInputLayer.connect(*theConvLayer.get());
    layer::Layer<double>::TDataRestoring theDataRestoring;
    theDataRestoring[layer::ConvolutionLayer<double>::RESTORE_WEIGHTS] = &theFilter;
    theConvLayer->restore(theDataRestoring);

    theConvLayer->forward();
    const Matrix<double> *theOutput = theConvLayer->getOutput();
    const Dimension &theOutputDim = theOutput->getDimension();

    if (theOutputDim != Dimension(9, 9, 1, 1) ||
            theOutput->getData()[theOutput->offset()] != 11 ||
            theOutput->getData()[theOutput->offset(3, 3)] != 132 ||
            theOutput->getData()[theOutput->offset(8, 8)] != 165) {
        std::cerr << "Convolution.convPadding test failed:" << std::endl
                  << "input:" << std::endl
                  << theInput << std::endl
                  << "filter:" << std::endl
                  << theFilter << std::endl
                  << "output:" << std::endl
                  << *theOutput << std::endl;
    }

    return true;
}

bool
convStride()
{
    const int64_t theKernelSize = 3;

    const Dimension theInputDim(9, 9, 1);
    const Dimension theFilterDim(theKernelSize, theKernelSize, 1);

    Matrix<double> theInput(theInputDim);
    Matrix<double> theFilter(theFilterDim);

    fillMatrix(theInput);

    {
        Matrix<double>::data_type *theFilterData = theFilter.getMutableData();
        theFilterData[theFilter.offset(0, 0)] = 0;
        theFilterData[theFilter.offset(0, 1)] = 1;
        theFilterData[theFilter.offset(0, 2)] = 0;
        theFilterData[theFilter.offset(1, 0)] = 1;
        theFilterData[theFilter.offset(1, 1)] = 0;
        theFilterData[theFilter.offset(1, 2)] = 1;
        theFilterData[theFilter.offset(2, 0)] = 0;
        theFilterData[theFilter.offset(2, 1)] = 1;
        theFilterData[theFilter.offset(2, 2)] = 0;
    }

    layer::ConstInputLayer<double> theInputLayer(theInput);
    layer::Layer<double>::TLayerConfig theConvConfig;
    theConvConfig.insert(std::make_pair(std::string(layer::ConvolutionLayer<double>::CONFIG_NUM_OUTPUT), "1"));
    theConvConfig.insert(std::make_pair(std::string(layer::ConvolutionLayer<double>::CONFIG_KERNEL_SIZE), "3"));
    theConvConfig.insert(std::make_pair(std::string(layer::ConvolutionLayer<double>::CONFIG_STRIDE), "2"));
    theConvConfig.insert(std::make_pair(std::string(layer::ConvolutionLayer<double>::CONFIG_PADDING), "0"));
    layer::Layer<double>::TUniqueHandle theConvLayer(layer::Layer<double>::create(layer::ConvolutionLayer<double>::TYPE, theConvConfig));
    theInputLayer.connect(*theConvLayer.get());
    layer::Layer<double>::TDataRestoring theDataRestoring;
    theDataRestoring[layer::ConvolutionLayer<double>::RESTORE_WEIGHTS] = &theFilter;
    theConvLayer->restore(theDataRestoring);

    theConvLayer->forward();
    const Matrix<double> *theOutput = theConvLayer->getOutput();
    const Dimension &theOutputDim = theOutput->getDimension();

    if (theOutputDim != Dimension(4, 4, 1, 1) ||
            theOutput->getData()[theOutput->offset()] != 44 ||
            theOutput->getData()[theOutput->offset(3, 3)] != 308 ||
            theOutput->getData()[theOutput->offset(1, 2)] != 212) {
        std::cerr << "Convolution.convStride test failed:" << std::endl
                  << "input:" << std::endl
                  << theInput << std::endl
                  << "filter:" << std::endl
                  << theFilter << std::endl
                  << "output:" << std::endl
                  << *theOutput << std::endl;
    }

    return true;
}

bool
convBias1()
{
    const int64_t theKernelSize = 3;

    const Dimension theInputDim(9, 9, 1);
    const Dimension theFilterDim(theKernelSize, theKernelSize, 1);
    const Dimension theBiasDim(1);

    Matrix<double> theInput(theInputDim);
    Matrix<double> theFilter(theFilterDim);
    Matrix<double> theBias(theBiasDim);

    fillMatrix(theInput);

    {
        Matrix<double>::data_type *theFilterData = theFilter.getMutableData();
        theFilterData[theFilter.offset(0, 0)] = 0;
        theFilterData[theFilter.offset(0, 1)] = 1;
        theFilterData[theFilter.offset(0, 2)] = 0;
        theFilterData[theFilter.offset(1, 0)] = 1;
        theFilterData[theFilter.offset(1, 1)] = 0;
        theFilterData[theFilter.offset(1, 2)] = 1;
        theFilterData[theFilter.offset(2, 0)] = 0;
        theFilterData[theFilter.offset(2, 1)] = 1;
        theFilterData[theFilter.offset(2, 2)] = 0;

        Matrix<double>::data_type *theBiasData = theBias.getMutableData();
        theBiasData[theBias.offset()] = 2.5;
    }

    layer::ConstInputLayer<double> theInputLayer(theInput);
    layer::Layer<double>::TLayerConfig theConvConfig;
    theConvConfig.insert(std::make_pair(std::string(layer::ConvolutionLayer<double>::CONFIG_NUM_OUTPUT), "1"));
    theConvConfig.insert(std::make_pair(std::string(layer::ConvolutionLayer<double>::CONFIG_KERNEL_SIZE), "3"));
    theConvConfig.insert(std::make_pair(std::string(layer::ConvolutionLayer<double>::CONFIG_STRIDE), "1"));
    theConvConfig.insert(std::make_pair(std::string(layer::ConvolutionLayer<double>::CONFIG_PADDING), "0"));
    theConvConfig.insert(std::make_pair(std::string(layer::ConvolutionLayer<double>::CONFIG_BIAS_TERM), "true"));
    layer::Layer<double>::TUniqueHandle theConvLayer(layer::Layer<double>::create(layer::ConvolutionLayer<double>::TYPE, theConvConfig));
    theInputLayer.connect(*theConvLayer.get());
    layer::Layer<double>::TDataRestoring theDataRestoring;
    theDataRestoring[layer::ConvolutionLayer<double>::RESTORE_WEIGHTS] = &theFilter;
    theDataRestoring[layer::ConvolutionLayer<double>::RESTORE_BIAS] = &theBias;
    theConvLayer->restore(theDataRestoring);

    theConvLayer->forward();
    const Matrix<double> *theOutput = theConvLayer->getOutput();
    const Dimension &theOutputDim = theOutput->getDimension();

    if (theOutputDim != Dimension(7, 7, 1, 1) ||
            theOutput->getData()[theOutput->offset()] != 46.5 ||
            theOutput->getData()[theOutput->offset(3, 3)] != 178.5 ||
            theOutput->getData()[theOutput->offset(6, 6)] != 310.5) {
        std::cerr << "Convolution.convBias1 test failed:" << std::endl
                  << "input:" << std::endl
                  << theInput << std::endl
                  << "filter:" << std::endl
                  << theFilter << std::endl
                  << "bias:" << std::endl
                  << theBias << std::endl
                  << "output:" << std::endl
                  << *theOutput << std::endl;
    }

    return true;
}

bool
convBias2()
{
    const int64_t theKernelSize = 3;

    const Dimension theInputDim(9, 9, 1);
    const Dimension theFilterDim(theKernelSize, theKernelSize, 1, 3);
    const Dimension theBiasDim(3);

    Matrix<double> theInput(theInputDim);
    Matrix<double> theFilter(theFilterDim);
    Matrix<double> theBias(theBiasDim);

    fillMatrix(theInput);

    {
        Matrix<double>::data_type *theFilterData = theFilter.getMutableData();
        theFilterData[theFilter.offset(0, 0)] = 0;
        theFilterData[theFilter.offset(0, 1)] = 1;
        theFilterData[theFilter.offset(0, 2)] = 0;
        theFilterData[theFilter.offset(1, 0)] = 1;
        theFilterData[theFilter.offset(1, 1)] = 0;
        theFilterData[theFilter.offset(1, 2)] = 1;
        theFilterData[theFilter.offset(2, 0)] = 0;
        theFilterData[theFilter.offset(2, 1)] = 1;
        theFilterData[theFilter.offset(2, 2)] = 0;

        theFilterData[theFilter.offset(0, 0, 0, 1)] = 1;
        theFilterData[theFilter.offset(0, 1, 0, 1)] = 1;
        theFilterData[theFilter.offset(0, 2, 0, 1)] = 1;
        theFilterData[theFilter.offset(1, 0, 0, 1)] = 1;
        theFilterData[theFilter.offset(1, 1, 0, 1)] = -8;
        theFilterData[theFilter.offset(1, 2, 0, 1)] = 1;
        theFilterData[theFilter.offset(2, 0, 0, 1)] = 1;
        theFilterData[theFilter.offset(2, 1, 0, 1)] = 1;
        theFilterData[theFilter.offset(2, 2, 0, 1)] = 1;

        theFilterData[theFilter.offset(0, 0, 0, 2)] = -1;
        theFilterData[theFilter.offset(0, 1, 0, 2)] = 0;
        theFilterData[theFilter.offset(0, 2, 0, 2)] = 1;
        theFilterData[theFilter.offset(1, 0, 0, 2)] = -2;
        theFilterData[theFilter.offset(1, 1, 0, 2)] = 0;
        theFilterData[theFilter.offset(1, 2, 0, 2)] = 2;
        theFilterData[theFilter.offset(2, 0, 0, 2)] = -1;
        theFilterData[theFilter.offset(2, 1, 0, 2)] = 0;
        theFilterData[theFilter.offset(2, 2, 0, 2)] = 1;

        Matrix<double>::data_type *theBiasData = theBias.getMutableData();
        theBiasData[theBias.offset(0)] = 2.5;
        theBiasData[theBias.offset(1)] = 4.6;
        theBiasData[theBias.offset(2)] = 1.2;
    }

    layer::ConstInputLayer<double> theInputLayer(theInput);
    layer::Layer<double>::TLayerConfig theConvConfig;
    theConvConfig.insert(std::make_pair(std::string(layer::ConvolutionLayer<double>::CONFIG_NUM_OUTPUT), "3"));
    theConvConfig.insert(std::make_pair(std::string(layer::ConvolutionLayer<double>::CONFIG_KERNEL_SIZE), "3"));
    theConvConfig.insert(std::make_pair(std::string(layer::ConvolutionLayer<double>::CONFIG_STRIDE), "1"));
    theConvConfig.insert(std::make_pair(std::string(layer::ConvolutionLayer<double>::CONFIG_PADDING), "0"));
    theConvConfig.insert(std::make_pair(std::string(layer::ConvolutionLayer<double>::CONFIG_BIAS_TERM), "true"));
    layer::Layer<double>::TUniqueHandle theConvLayer(layer::Layer<double>::create(layer::ConvolutionLayer<double>::TYPE, theConvConfig));
    theInputLayer.connect(*theConvLayer.get());
    layer::Layer<double>::TDataRestoring theDataRestoring;
    theDataRestoring[layer::ConvolutionLayer<double>::RESTORE_WEIGHTS] = &theFilter;
    theDataRestoring[layer::ConvolutionLayer<double>::RESTORE_BIAS] = &theBias;
    theConvLayer->restore(theDataRestoring);

    theConvLayer->forward();
    const Matrix<double> *theOutput = theConvLayer->getOutput();
    const Dimension &theOutputDim = theOutput->getDimension();

    if (theOutputDim != Dimension(7, 7, 3, 1) ||
            theOutput->getData()[theOutput->offset()] != 46.5 ||
            theOutput->getData()[theOutput->offset(3, 3)] != 178.5 ||
            theOutput->getData()[theOutput->offset(3, 3, 1, 0)] != 4.6 ||
            theOutput->getData()[theOutput->offset(0, 0, 2, 0)] != 81.2 ||
            theOutput->getData()[theOutput->offset(3, 3, 2, 0)] != 81.2) {
        std::cerr << "Convolution.convBias2 test failed:" << std::endl
                  << "input:" << std::endl
                  << theInput << std::endl
                  << "filter:" << std::endl
                  << theFilter << std::endl
                  << "bias:" << std::endl
                  << theBias << std::endl
                  << "output:" << std::endl
                  << *theOutput << std::endl;
    }

    return true;
}

}

namespace test
{

void
ConvolutionLayerTests::run()
{
    conv1();
    conv2();
    conv3();

    convChannel();
    convPadding();
    convStride();
    convBias1();
    convBias2();
}

}
