#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include<chrono>

#define CUDA_CHECK(function)\
do{\
    cudaError_t err=function;\
    if(err!=cudaSuccess){\
        printf("error in file : %s ,line : %d\n",__FILE__,__LINE__);\
        printf("    error code: %d ",err);\
        printf("name: %s\n",cudaGetErrorName(err));\
        exit(EXIT_FAILURE);\
    }\
} while (0)

void inline check_device()
{
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    printf("Find deviceCount : %d\n",deviceCount);
    for (int dev = 0; dev < deviceCount; dev++) {
        // 设置当前设备
        CUDA_CHECK(cudaSetDevice(dev));
        
        // 获取设备属性
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        
        printf("    Device %d: %s\n", dev, prop.name);
        printf("        Global Memory:%.3f GB\n", prop.totalGlobalMem/1e9);
        printf("        Multiprocessors:%d\n", prop.multiProcessorCount);
        printf("        regsPerMultiprocessor: %d\n",prop.regsPerMultiprocessor);        
        printf("        maxThreadsPerMultiProcessor:%d \n", prop.maxThreadsPerMultiProcessor);
        printf("        sharedMemPerMultiprocessor: %lld\n",prop.sharedMemPerMultiprocessor);
    }
}

template <typename timetype, typename calltype,typename...  Argtypes>
timetype testtime(calltype call,Argtypes... args)
{   
    auto start = std::chrono::high_resolution_clock::now();
    call(std::forward<Argtypes>(args)...);
    auto end = std::chrono::high_resolution_clock::now();   
    return std::chrono::duration_cast<timetype>(end - start); 
}