project(toy-cnn-test C CXX)

set(SOURCES
    ConstInputLayer.hpp
    ConvolutionLayerTests.cpp
    ConvolutionLayerTests.hpp
    InnerProductLayerTests.cpp
    InnerProductLayerTests.hpp
    DimensionTests.cpp
    DimensionTests.hpp
    LeNetClassification.cpp
    LeNetClassification.hpp
    MatrixTests.cpp
    MatrixTests.hpp
    MemoryInputLayer.hpp
    PoolingLayerTests.cpp
    PoolingLayerTests.hpp
    PrintUtils.cpp
    PrintUtils.hpp
    ReluLayerTests.cpp
    ReluLayerTests.hpp
    SoftMaxLayerTests.cpp
    SoftMaxLayerTests.hpp
    TestDriver.cpp)

include_directories(../sdk/include)

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} toy-cnn ${CUDA_CUDART_LIBRARY})
