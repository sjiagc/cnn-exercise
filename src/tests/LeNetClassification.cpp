#include "LeNetClassification.hpp"

#include <utils/Matrix.hpp>
#include <Layer.hpp>
#include <layers/ConvolutionLayer.hpp>
#include <layers/InnerProductLayer.hpp>
#include <layers/PoolingLayer.hpp>
#include <layers/ReluLayer.hpp>
#include <layers/SoftMaxLayer.hpp>

#include "MemoryInputLayer.hpp"
#include "PrintUtils.hpp"

#include <stdexcept>
#include <fstream>
#include <memory>
#include <sstream>
#include <iostream>

namespace
{

#define MODEL_DATA_PREFIX "/data/develop/dl/cnn-exercise/data/lenet/model_data/"
#define TEST_DATA_PREFIX "/data/develop/dl/cnn-exercise/data/lenet/test_data/"
using namespace test;

using utils::Dimension;
using utils::Matrix;

std::unique_ptr<Matrix<double>>
ReadStoredData(const char *inFilePath)
{
    std::ifstream theInput(inFilePath);
    if (!theInput)
        return nullptr;
    char theBuf[1024];
    theInput.getline(theBuf, 1024);
    std::istringstream iss(theBuf);
    int64_t theWDim;
    int64_t theZDim;
    int64_t theYDim;
    int64_t theXDim;
    iss >> theWDim >> theZDim >> theYDim >> theXDim;
    std::unique_ptr<Matrix<double>> theMatrix(new Matrix<double>(Dimension(theXDim, theYDim, theZDim, theWDim)));
    Matrix<double>::data_type *theData = theMatrix->getMutableData();
    int64_t theDataCount = 0;
    while (theInput) {
        theInput.getline(theBuf, 1024);
        std::istringstream iss(theBuf);
        int64_t x, y, z, w;
        iss >> x >> y >> z >> w;
        Matrix<double>::data_type theValue;
        iss >> theValue;
        theData[theMatrix->offset(x, y, z, w)] = theValue;
        ++theDataCount;
    }
    std::cout << "Data read: " << theDataCount << std::endl;
    return theMatrix;
}

bool
classification()
{
    // input
    layer::MemoryInputLayer<double> theInput(Dimension(28, 28));
    // conv1
    layer::Layer<double>::TLayerConfig theConv1Config;
    theConv1Config[std::string(layer::ConvolutionLayer<double>::CONFIG_NUM_OUTPUT)] = "20";
    theConv1Config[std::string(layer::ConvolutionLayer<double>::CONFIG_KERNEL_SIZE)] = "5";
    theConv1Config[std::string(layer::ConvolutionLayer<double>::CONFIG_STRIDE)] = "1";
    theConv1Config[std::string(layer::ConvolutionLayer<double>::CONFIG_PADDING)] = "0";
    theConv1Config[std::string(layer::ConvolutionLayer<double>::CONFIG_BIAS_TERM)] = "true";
    layer::Layer<double>::TUniqueHandle theConv1(layer::Layer<double>::create(layer::ConvolutionLayer<double>::TYPE, theConv1Config));
    // pool1
    layer::Layer<double>::TLayerConfig thePool1Config;
    thePool1Config[std::string(layer::PoolingLayer<double>::CONFIG_KERNEL_SIZE)] = "2";
    thePool1Config[std::string(layer::PoolingLayer<double>::CONFIG_STRIDE)] = "2";
    thePool1Config[std::string(layer::PoolingLayer<double>::CONFIG_PADDING)] = "0";
    layer::Layer<double>::TUniqueHandle thePool1(layer::Layer<double>::create(layer::PoolingLayer<double>::TYPE, thePool1Config));
    // conv2
    layer::Layer<double>::TLayerConfig theConv2Config;
    theConv2Config[std::string(layer::ConvolutionLayer<double>::CONFIG_NUM_OUTPUT)] = "50";
    theConv2Config[std::string(layer::ConvolutionLayer<double>::CONFIG_KERNEL_SIZE)] = "5";
    theConv2Config[std::string(layer::ConvolutionLayer<double>::CONFIG_STRIDE)] = "1";
    theConv2Config[std::string(layer::ConvolutionLayer<double>::CONFIG_PADDING)] = "0";
    theConv2Config[std::string(layer::ConvolutionLayer<double>::CONFIG_BIAS_TERM)] = "true";
    layer::Layer<double>::TUniqueHandle theConv2(layer::Layer<double>::create(layer::ConvolutionLayer<double>::TYPE, theConv2Config));
    // pool2
    layer::Layer<double>::TLayerConfig thePool2Config;
    thePool2Config[std::string(layer::PoolingLayer<double>::CONFIG_KERNEL_SIZE)] = "2";
    thePool2Config[std::string(layer::PoolingLayer<double>::CONFIG_STRIDE)] = "2";
    thePool2Config[std::string(layer::PoolingLayer<double>::CONFIG_PADDING)] = "0";
    layer::Layer<double>::TUniqueHandle thePool2(layer::Layer<double>::create(layer::PoolingLayer<double>::TYPE, thePool2Config));
    // ip1
    layer::Layer<double>::TLayerConfig theIP1Config;
    theIP1Config.insert(std::make_pair(std::string(layer::InnerProductLayer<double>::CONFIG_NUM_OUTPUT), "500"));
    theIP1Config.insert(std::make_pair(std::string(layer::InnerProductLayer<double>::CONFIG_BIAS_TERM), "true"));
    layer::Layer<double>::TUniqueHandle theIP1(layer::Layer<double>::create(layer::InnerProductLayer<double>::TYPE, theIP1Config));
    // relu1
    layer::Layer<double>::TUniqueHandle theRelu1(layer::Layer<double>::create(layer::ReluLayer<double>::TYPE, layer::Layer<double>::TLayerConfig()));
    // ip2
    layer::Layer<double>::TLayerConfig theIP2Config;
    theIP2Config.insert(std::make_pair(std::string(layer::InnerProductLayer<double>::CONFIG_NUM_OUTPUT), "10"));
    theIP2Config.insert(std::make_pair(std::string(layer::InnerProductLayer<double>::CONFIG_BIAS_TERM), "true"));
    layer::Layer<double>::TUniqueHandle theIP2(layer::Layer<double>::create(layer::InnerProductLayer<double>::TYPE, theIP2Config));
    // prob
    layer::Layer<double>::TUniqueHandle theProb(layer::Layer<double>::create(layer::SoftMaxLayer<double>::TYPE, layer::Layer<double>::TLayerConfig()));

    // Link
    theInput.connect(*theConv1.get());
    theConv1->connect(*thePool1.get());
    thePool1->connect(*theConv2.get());
    theConv2->connect(*thePool2.get());
    thePool2->connect(*theIP1.get());
    theIP1->connect(*theRelu1.get());
    theRelu1->connect(*theIP2.get());
    theIP2->connect(*theProb.get());

    // Restore
    std::unique_ptr<Matrix<double>> theConv1Weights =
            ReadStoredData(MODEL_DATA_PREFIX"conv1_0.txt");
    std::unique_ptr<Matrix<double>> theConv1Bias =
            ReadStoredData(MODEL_DATA_PREFIX"conv1_1.txt");
    std::unique_ptr<Matrix<double>> theConv2Weights =
            ReadStoredData(MODEL_DATA_PREFIX"conv2_0.txt");
    std::unique_ptr<Matrix<double>> theConv2Bias=
            ReadStoredData(MODEL_DATA_PREFIX"conv2_1.txt");
    std::unique_ptr<Matrix<double>> theIP1Weights=
            ReadStoredData(MODEL_DATA_PREFIX"ip1_0.txt");
    std::unique_ptr<Matrix<double>> theIP1Bias=
            ReadStoredData(MODEL_DATA_PREFIX"ip1_1.txt");
    std::unique_ptr<Matrix<double>> theIP2Weights=
            ReadStoredData(MODEL_DATA_PREFIX"ip2_0.txt");
    std::unique_ptr<Matrix<double>> theIP2Bias=
            ReadStoredData(MODEL_DATA_PREFIX"ip2_1.txt");
    // conv1
    layer::Layer<double>::TDataRestoring theConv1Restoring;
    theConv1Restoring[layer::ConvolutionLayer<double>::RESTORE_WEIGHTS] = theConv1Weights.get();
    theConv1Bias->reshape(Dimension(20));
    theConv1Restoring[layer::ConvolutionLayer<double>::RESTORE_BIAS] = theConv1Bias.get();
    theConv1->restore(theConv1Restoring);
    // conv2
    layer::Layer<double>::TDataRestoring theConv2Restoring;
    theConv2Restoring[layer::ConvolutionLayer<double>::RESTORE_WEIGHTS] = theConv2Weights.get();
    theConv2Bias->reshape(Dimension(50));
    theConv2Restoring[layer::ConvolutionLayer<double>::RESTORE_BIAS] = theConv2Bias.get();
    theConv2->restore(theConv2Restoring);
    // ip1
    layer::Layer<double>::TDataRestoring theIP1Restoring;
    const Dimension &theIP1WeightsDim = theIP1Weights->getDimension();
    theIP1Weights->reshape(Dimension(theIP1WeightsDim.getZ(), theIP1WeightsDim.getW()));
    theIP1Restoring[layer::InnerProductLayer<double>::RESTORE_WEIGHTS] = theIP1Weights.get();
    theIP1Bias->reshape(Dimension(500));
    theIP1Restoring[layer::InnerProductLayer<double>::RESTORE_BIAS] = theIP1Bias.get();
    theIP1->restore(theIP1Restoring);
    // ip2
    layer::Layer<double>::TDataRestoring theIP2Restoring;
    const Dimension &theIP2WeightsDim = theIP2Weights->getDimension();
    theIP2Weights->reshape(Dimension(theIP2WeightsDim.getZ(), theIP2WeightsDim.getW()));
    theIP2Restoring[layer::InnerProductLayer<double>::RESTORE_WEIGHTS] = theIP2Weights.get();
    theIP2Bias->reshape(Dimension(10));
    theIP2Restoring[layer::InnerProductLayer<double>::RESTORE_BIAS] = theIP2Bias.get();
    theIP2->restore(theIP2Restoring);

    static const char *theInputs[] = {
        TEST_DATA_PREFIX"input_0.txt",
        TEST_DATA_PREFIX"input_1.txt",
        TEST_DATA_PREFIX"input_2.txt",
        TEST_DATA_PREFIX"input_3.txt",
        TEST_DATA_PREFIX"input_4.txt",
        TEST_DATA_PREFIX"input_5.txt",
        TEST_DATA_PREFIX"input_6.txt",
        TEST_DATA_PREFIX"input_7.txt",
        TEST_DATA_PREFIX"input_8.txt",
        TEST_DATA_PREFIX"input_9.txt",
    };
    for (size_t u = 0, theCount = sizeof(theInputs) / sizeof(theInputs[0]); u < theCount; ++u) {
        // Read input
        std::unique_ptr<Matrix<double>> theInputMatrix = ReadStoredData(theInputs[u]);
        if (!theInputMatrix)
            continue;
        std::cout << "Classifying " << theInputs[u] << " ..." << std::endl;
        theInput.setInput(*theInputMatrix.get());
        theInput.forward();
        theConv1->forward();
        thePool1->forward();
        theConv2->forward();
        thePool2->forward();
        theIP1->forward();
        theRelu1->forward();
        theIP2->forward();
        theProb->forward();

        const Matrix<double> *theOutput = theProb->getOutput();
        std::cout << *theOutput << std::endl;
    }
    return true;
}

}

namespace test
{

void
LeNetClassificationTests::run()
{
    classification();
}

}
