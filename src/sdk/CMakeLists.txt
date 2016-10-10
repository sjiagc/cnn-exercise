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
    source/layers/ReluLayer.cu
    source/layers/SoftMaxLayer.cu)


set(GENCODE -arch=compute_35)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS} -G)

FIND_CUDA_HELPER_LIBS(cudadevrt)

set(CUDA_SEPARABLE_COMPILATION ON)
cuda_compile(DEVICE_OBJS ${CUDA_SOURCES} OPTIONS ${GENCODE})
cuda_compute_separable_compilation_object_file_name(LINK_OBJS ${PROJECT_NAME} "${DEVICE_OBJS}")
cuda_link_separable_compilation_objects(${LINK_OBJS} ${PROJECT_NAME} ${GENCODE} "${DEVICE_OBJS}")


add_library(${PROJECT_NAME} STATIC ${SOURCES} ${HEADERS} ${DEVICE_OBJS} ${LINK_OBJS})

target_link_libraries(${PROJECT_NAME} ${CUDA_CUDART_LIBRARY} ${CUDA_cudadevrt_LIBRARY})

set_target_properties(${PROJECT_NAME}
    PROPERTIES
    LINKER_LANGUAGE ${CUDA_C_OR_CXX})
