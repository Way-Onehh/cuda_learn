cmake_minimum_required(VERSION 3.18)
project(MyCUDApp LANGUAGES CXX CUDA)  # 明确启用CUDA支持

add_library(My_kernel My_kernel.cu)

find_package(CUDAToolkit REQUIRED)
target_include_directories(My_kernel PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(My_kernel PRIVATE cudart)