#ifndef __MCS__RUNTIME__
#define __MCS__RUNTIME__

#include <omp.h>
#include "CudaInfo.cuh"
#include "Structure.cuh"


int MaxThreads = omp_get_max_threads();

struct rDots {                                                                               //4维数组,a,b,c,N
    Vec3 *Dots;
    rDots() : Dots(NULL) {}
    ~rDots() {}
};

void InitRDots(rDots *Tar, SuperCell *superCell) {
    return;
}
void DestroyDots(rDots *Tar) {                                                               //这里销毁了自身！
    free(Tar->Dots);
    free(Tar);
    return;
}

#endif