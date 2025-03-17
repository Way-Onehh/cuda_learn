
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <cuda_tool.hpp>

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

/*
    cudaMalloc           cudafree  在GPU上分配，手动控制迁移时机  !!!要手动检查分配是否成功,各个GPU条件不同
        cudaMemcpy        会同步一次device然后迁移可以指定方向
        cudaMemset
    cudaMallocManaged    cudafree  统一内存技术 cuda自动按需迁移数据 
    下面演示了对数组的并行操作
*/

__global__ void __GPU_ADD_ONE(long long int *p)
{
    int index=blockDim.x*blockIdx.x+threadIdx.x;
    ++*(p+index);
    printf("%lld ",*(p+index));
}

void GPU_ADD_ONE(size_t times,size_t type)
{
    for(int i=0;i<times;i++)
    {
        if(type==0)
        {
            long long int a[1024]={1,2,3,4,5,6};
            long long int * Gp=nullptr;
            CUDA_CHECK(cudaMalloc(&Gp,sizeof(a)));
            //???这里有bug 大于 1024 在这张显卡上分配不了??? 不是大小的原因换成longlongint也一样
            //10系显卡特有bug 数组大小超过1024*type会导致显存数组分配失败
            //解决方法减少数组大小就行吗 
            if(Gp==nullptr)exit(EXIT_FAILURE);
            CUDA_CHECK(cudaMemcpy(Gp,a,sizeof(a),cudaMemcpyHostToDevice));
            printf("in GPU ");
            __GPU_ADD_ONE<<<1,sizeof(a)/sizeof(long long int)>>>(Gp);
            CUDA_CHECK(cudaMemcpy(a,Gp,sizeof(a),cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(Gp));
            printf("in CPU  ");
            for (auto &&i : a)
            {   
                printf("%lld ",i);
            }
       }
       if(type==1)
       {
           long long int * Gp=nullptr;
           CUDA_CHECK(cudaMallocManaged(&Gp,sizeof(long long [1024])));//???这里有bug 大于 1024 在这张显卡上分配不了??? 不是大小的原因换成longlongint也一样
           if(Gp==nullptr)exit(EXIT_FAILURE);
           * Gp=100;
           //CUDA_CHECK(cudaMemcpy(Gp,a,sizeof(a),cudaMemcpyHostToDevice));
           printf("in GPU ");
           __GPU_ADD_ONE<<<1,sizeof(long long [1024])/sizeof(long long int)>>>(Gp);
           CUDA_CHECK(cudaDeviceSynchronize());
           //CUDA_CHECK(cudaMemcpy(a,Gp,sizeof(a),cudaMemcpyDeviceToHost));
           
           printf("in CPU  ");
           for (size_t i=0;i<1024;i++)
           {   
               printf("%lld ",*(Gp+i));
           }
           CUDA_CHECK(cudaFree(Gp));
      }
    }
}


/*
    损失在10%
*/
void MemoryTest()
{
    auto start=std::chrono::steady_clock::now();
    GPU_ADD_ONE(1000,0);
    auto end=std::chrono::steady_clock::now();
    
    auto start1=std::chrono::steady_clock::now();
    GPU_ADD_ONE(1000,1);
    auto end1=std::chrono::steady_clock::now();

    std::cout<<std::chrono::duration_cast<std::chrono::seconds>(end-start).count()<<"\n";
    std::cout<<std::chrono::duration_cast<std::chrono::seconds>(end1-start1).count()<<"\n";
}