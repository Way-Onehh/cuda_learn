#include <cuda_runtime.h>
#include <stdlib.h>
#include <cuda_tool.hpp>
#include <cmath>
#define ____ByteSize(type,number)  number*sizeof(type)

__global__ void mallocIncArray_kernel(int * p,int s,int step)
{
    int offset=blockIdx.x*blockDim.x+threadIdx.x;
    *(p+offset)=s+offset*step;
}


#define ByteSize(number)  ____ByteSize(int,number)
int * mallocIncArray(int s, int e ,int step)
{
    //分配 堆 显存
    int number=(e-s)/step;
    int * h_p, * d_p=nullptr;
    h_p=(int *) malloc(ByteSize(number));
    CUDA_CHECK(cudaMalloc(&d_p,ByteSize(number)));
    if(!h_p||!d_p) return nullptr;
    //调用 核函数
    //mallocIncArray_kernel<<<(number + 383) / 384,384>>>(d_p,s,step);
    mallocIncArray_kernel<<<1,number>>>(d_p,s,step);
    CUDA_CHECK(cudaDeviceSynchronize() );
    //复制到堆 释放显存
    cudaMemcpy(h_p,d_p,ByteSize(number),cudaMemcpyDeviceToHost);
    CUDA_CHECK(cudaFree(d_p));
    return h_p;
}
#undef ByteSize

#define BLOCK_SIZE 256 
void __global__ sumKernel(int * input,int *output,int N)
{
    __shared__ float sharedData[BLOCK_SIZE];
 
    // 计算线程索引 
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
 
    // 将数据加载到共享内存 
    sharedData[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();
 
    // 在共享内存中执行部分求和 
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads();
    }
 
    // 将结果写入输出数组 
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

/*
    @param ptr      数组地址
    @param number   数组大小 
*/
int sum(int * h_input,int N)
{
    int *d_input, *d_output;
    int h_output = 0.0f;
 
    // 分配设备内存 
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) * sizeof(int));
 
    // 将数据从主机复制到设备 
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
 
    // 调用核函数 
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sumKernel<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, N);
 
    // 将部分结果从设备复制回主机 
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
 
    // 释放设备内存 
    cudaFree(d_input);
    cudaFree(d_output);
 
    return h_output;
}

