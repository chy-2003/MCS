#ifndef __MCS__RUNTIME__
#define __MCS__RUNTIME__

#include <omp.h>
#include "CudaInfo.cuh"
#include "Structure.cuh"

struct rUnit {                                                                               //运行时原胞，仅保留点上矢量信息，Dots为数组，对应UnitCell中数组
    Vec3 *Dots;
    rUnit() : Dots(NULL) {}
    ~rUnit() {}
};
void DestroyRUnit(rUnit *self) {
    checkCuda(cudaFree(self->Dots));
    checkCuda(cudaFree(self));
    return;
}

struct rMesh {                                                                               //运行时网格
    rUnit *Unit;                                                                             //Unit是三维数组, 三个维度下标都为 0~N-1
    rMesh() : Unit(NULL) {}
    ~rMesh() {}
};
void DestroyRMesh(rMesh *self, SuperCell *superCell) {
    int N = superCell->a * superCell->b * superCell->c;
    for (int i = 0; i < N; ++i)
        DestroyRUnit(self->Unit + i);
    checkCuda(cudaFree(self->Unit));
    checkCuda(cudaFree(self));
    return;
}

__global__ void BuildUnit(rMesh *self, UnitCell *unitCell, int Size) {                       //建立网格内单个元胞
    int N = blockIdx.x * blockDim.x + threadIdx.x;
    if (N >= Size) return;
    rUnit *target = self->Unit + N;
    for (int i = 0; i < unitCell->N; ++i)
        (target->Dots)[i] = *((unitCell->Dots)[i].a);
    return;
}

rMesh* BuildRMesh(rMesh *self, SuperCell *superCell) {                                       //建立网格
    checkCuda(cudaMallocManaged(&self, sizeof(rMesh)));                                      //定义网格
    //unfinished
    int N = superCell->a * superCell->b * superCell->c;
    checkCuda(cudaMallocManaged(&(self->Unit), sizeof(rUnit) * N));                          //定义网格内数组（Unit），数组内的单个元素为一个rUnit
    int MaxThreads = omp_get_max_threads();
    #pragma omp parallel for num_threads(MaxThreads)
    for (int i = 0; i < N; ++i) 
        checkCuda(cudaMallocManaged(&(self->Unit->Dots), sizeof(Vec3) * superCell->unitCell->N));//为rUnit分配内存，即申明rUnit的Dots数组

    size_t threadsPerBlock = 256;
    size_t numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    BuildUnit<<<numberOfBlocks, threadsPerBlock>>>(self, superCell->unitCell, N);
    cudaDeviceSynchronize();

    return self;
}

#endif