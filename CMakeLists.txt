cmake_minimum_required(VERSION 3.18)

#head
set(subPro My_kernel test_reduce)
set(includePath include ${subPro})

#lib
include_directories(${includePath})
foreach(i ${subPro})
    add_subdirectory(${i})
endforeach()

# 添加可执行文件，包含CUDA和C++源文件
add_executable(main
    main.cpp      
)

target_link_libraries(main ${subPro})