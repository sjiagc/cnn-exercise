project(toy-cnn C CXX)

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
find_package(CUDA QUIET REQUIRED)
message("CUDA VERSION: " ${CUDA_VERSION_STRING})

include_directories(
    include
    ${CUDA_INCLUDE_DIRS})

set(HEADERS
    include/Layer.hpp
    include/ToyCnnDefs.hpp
    include/Types.h
    include/layers/ConvolutionLayer.hpp
    include/layers/InnerProductLayer.hpp
    include/layers/PoolingLayer.hpp
    include/layers/ReluLayer.hpp
    include/layers/SoftMaxLayer.hpp
    include/utils/Matrix.hpp
    include/utils/Dimension.hpp)

set(SOURCES
    source/Layer.cpp
    source/layers/ConvolutionLayer.cpp
    source/layers/InnerProductLayer.cpp
    source/layers/PoolingLayer.cpp
    source/layers/ReluLayer.cpp
    source/layers/SoftMaxLayer.cpp
    source/utils/DataBlock.cpp
    source/utils/DataBlock.hpp
    source/utils/Matrix.cpp)

set(CUDA_SOURCES
    source/layers/ReluLayer.cu)


set(GENCODE -arch=compute_20)

set(CUDA_SEPARABLE_COMPILATION ON)
cuda_compile(DEVICE_OBJS ${CUDA_SOURCES} OPTIONS ${GENCODE})
cuda_compute_separable_compilation_object_file_name(LINK_OBJS ${PROJECT_NAME} ${DEVICE_OBJS})
cuda_link_separable_compilation_objects(${LINK_OBJS} ${PROJECT_NAME} ${GENCODE} ${DEVICE_OBJS})


add_library(${PROJECT_NAME} STATIC ${DEVICE_OBJS} ${LINK_OBJS} ${HEADERS} ${SOURCES})
