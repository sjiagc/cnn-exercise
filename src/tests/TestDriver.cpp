#include "ConvolutionLayerTests.hpp"
#include "DimensionTests.hpp"
#include "InnerProductLayerTests.hpp"
#include "MatrixTests.hpp"
#include "PoolingLayerTests.hpp"
#include "ReluLayerTests.hpp"
#include "SoftMaxLayerTests.hpp"

#include <exception>
#include <iostream>

#define RUN_TEST(tester) \
    do { \
        tester theTester; \
        theTester.run(); \
    } while(0)

int
main()
{
    try {
        RUN_TEST(test::DimensionTests);
        RUN_TEST(test::MatrixTests);
        RUN_TEST(test::ConvolutionLayerTests);
        RUN_TEST(test::InnerProductLayerTests);
        RUN_TEST(test::PoolingLayerTests);
        RUN_TEST(test::ReluLayerTests);
        RUN_TEST(test::SoftMaxLayerTests);
    } catch (std::exception e) {
        std::cerr << e.what() << std::endl;
    }
}
