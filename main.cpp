#include<My_kernel.h>
#include<test_reduce.h>
#include<stdio.h>


int main(int argc, char const *argv[])
{   

    auto testfuns=[]()
    {
        int *p=mallocIncArray(0,1024,1);
        sum(p,1024);
        free(p);
    };
    printf("cost %lld",testtime<std::chrono::microseconds>(testfuns).count());
    return 0;
}
