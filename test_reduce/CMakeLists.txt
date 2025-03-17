cmake_minimum_required(VERSION 3.18)
project(MyCUDApp LANGUAGES CXX CUDA)  # 明确启用CUDA支持

add_library(test_reduce test_reduce.cu)

find_package(CUDAToolkit REQUIRED)
target_include_directories(test_reduce PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(test_reduce PRIVATE cudart)