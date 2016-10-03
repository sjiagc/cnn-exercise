project(toy-cnn C CXX)

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
find_package(CUDA QUIET REQUIRED)
message("CUDA VERSION: " ${CUDA_VERSION_STRING})

include_directories(include)

set(HEADERS
    include/Layer.hpp
    include/layers/ConvolutionLayer.hpp
    include/layers/InnerProductLayer.hpp
    include/layers/PoolingLayer.hpp
    include/layers/ReluLayer.hpp
    include/layers/SoftMaxLayer.hpp
    include/utils/Matrix.hpp
    include/utils/Types.hpp)

set(SOURCES
    source/Layer.cpp
    source/layers/ConvolutionLayer.cpp
    source/layers/InnerProductLayer.cpp
    source/layers/PoolingLayer.cpp
    source/layers/ReluLayer.cpp
    source/layers/SoftMaxLayer.cpp
    source/utils/Matrix.cpp)

add_library(${PROJECT_NAME} STATIC ${HEADERS} ${SOURCES})
