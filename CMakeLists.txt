cmake_minimum_required(VERSION 3.18)
project(MyCUDApp LANGUAGES CXX CUDA)  # 明确启用CUDA支持
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)          # 设置C++标准
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
# 指定CUDA架构（根据GPU计算能力调整，例如sm_75对应Turing架构）
set(CMAKE_CUDA_ARCHITECTURES "75")

# 添加可执行文件，包含CUDA和C++源文件
add_executable(main
    main.cpp      # C++主程序
    My_kernel.cu
)

# 链接CUDA运行时库（现代CMake通常自动处理，显式链接更安全）
target_link_libraries(main PRIVATE cudart)

find_package(CUDAToolkit REQUIRED)
target_include_directories(main PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}  # 自动填充CUDA头文件路径
)