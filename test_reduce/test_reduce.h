#include<chrono>

/*
    @param char s 第一个字节大小。
    @param char e 最后一个字节的大小。
    @param char step 步长。
    @return char * res 成功返回堆地址，失败返回空指针。
    @note 记得释放堆地址
*/
int * mallocIncArray(int s ,int e ,int step);

/*
    @param size_t number=100000000  在不考虑共享内存的情况下超过这个数量级GPU才会是GPU大于cpu
*/
void inline test_costtime_cpu_gpu1(size_t number=100000000)
{
    //
    auto t1=std::chrono::steady_clock::now();
    int *p= mallocIncArray(0,number,1);
    free(p);
    auto t2=std::chrono::steady_clock::now();
    auto d= std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1);
    printf("GPU cost %lld\n",d.count());
   
    //
    auto tt1=std::chrono::steady_clock::now();
    p=(int *) malloc(number*sizeof(int));
    for (size_t i = 0; i < number; i++)
    {
        *(p+i)=i;
    }
    free(p);
    auto tt2=std::chrono::steady_clock::now();
    auto td= std::chrono::duration_cast<std::chrono::milliseconds>(tt2-tt1);
    printf("CPU cost %lld\n",td.count());    
}
int sum(int * ptr,int number);