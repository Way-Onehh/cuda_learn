#include <stdio.h>
#include <cuda_runtime.h>


#define CUDA_CHECK(function)\
do{\
    cudaError_t err=function;\
    if(err!=0){\
        printf("error in file : %s ,line : %d\n",__FILE__,__LINE__);\
        printf("    error code: %d ",err);\
        printf("name: %s\n",cudaGetErrorName(err));\
        abort();\
    }\
} while (0)

__device__ void __Show_GPU_call_GPU()
{
    printf("        in \"__device__\" on GPU\n");
}

__global__ void __Show_CPU_call_GPU()
{
    printf("    in GPU\n");
    printf("    call \"__device__\"  __Show_GPU_call_GPU\n");
    __Show_GPU_call_GPU();
}

/*
key  __global__     __device__      __host__  (defualt)
    cpu call GPU    GPU call GPU  cpu call cpu
    host -> global -> device -> device
*/
__host__  void  Show_CPU_call()
{
    printf("in CPU\n");
    printf("call \"__global__\" __Show_CPU_call_GPU<<<1,2>>> \n");
    __Show_CPU_call_GPU<<<1,2>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
}


__global__ void __linear()
{   
    int total=gridDim.x*blockDim.x;
    int index=blockDim.x*blockIdx.x+threadIdx.x;
    printf("thread id %d of %d !\n",index,total);
}
/*
    网格 -----------------> x
    块  |     |     |  
    线程| | | | | | |
*/
void linear()
{
    __linear<<<2,6>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void __Show_ERROR(int * p)
{   
    *p=1;
    printf(" %d ",*p);
    printf(" %p ",p);
}
/*
    显存与堆、栈不共享 
    type=0  stack
    type=1  heap
    type=2  GPU memory
    由于出现cuda错误时，不会在分配内存所以隔离代码块使用
*/
void show_Memory_ERROR(int type)
{
    switch (type)
    {
        case 0://栈 
        {
            printf("____in stack____\n");
            int n=0;
            __Show_ERROR<<<1,1>>>(&n);
            CUDA_CHECK(cudaDeviceSynchronize());
            break;
        }
        case 1://堆
        {
            printf("____in heap____\n");
            int *p=new int;
            __Show_ERROR<<<1,1>>>(p);
            CUDA_CHECK(cudaDeviceSynchronize());
            delete p;
            break;
        }
        case 2://显存
        {
            printf("____in GPU memory____\n");
            int *gp=nullptr;
            CUDA_CHECK(cudaMalloc(&gp,sizeof(int)));
            __Show_ERROR<<<1,1>>>(gp);
            printf("    addres is %p \n",gp);
            if(gp!=nullptr) 
            printf("%p ",gp);
            else
            printf("    gp is nullptr\n");
            CUDA_CHECK(cudaDeviceSynchronize());
            cudaFree(gp);
            break;
        }
        default:{
            abort();
            break;
        }
    }
}


__global__ void __Malloc_in_GPU(int *p,size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        *(p+i)=i;
        printf("%d ",*(p+i));
    }
}
/*
    分配显存
*/
void Malloc_in_GPU()
{
    int *p=nullptr;
    size_t n=10;
    CUDA_CHECK(cudaMalloc(&p,sizeof(int)*n));
    printf("    addres is %p \n",p);
    __Malloc_in_GPU<<<1,1>>>(p,n);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(p);
}
