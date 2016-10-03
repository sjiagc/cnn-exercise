cmake_minimum_required(VERSION 3.4.0 FATAL_ERROR)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

#-------------------------------------------------------------------------------
# Root project
#-------------------------------------------------------------------------------
project(cnn-experiment C CXX)

add_subdirectory(src)
