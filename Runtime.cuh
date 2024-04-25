#ifndef __MCS__RUNTIME__
#define __MCS__RUNTIME__

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
    rUnit *Unit;
    rMesh() : Unit(NULL) {}
    ~rMesh() {}
}
void DestroyRMesh(rMesh *self, SuperCell *superCell) {
    int N = superCell->a * superCell->b * superCell->c;
    for (int i = 0; i < N; ++i)
        DestroyRUnit(self->Unit + i);
    checkCuda(cudaFree(self));
    return;
}
rMesh* BuildRMesh(rMesh *self, SuperCell *superCell) {
    checkCuda(cudaMallocManaged(&self), sizeof(rMesh));
    //unfinished
    int N = superCell->a * superCell->b * superCell->c;
    dim3 threadPerBlock(32, 32);
    return self;
}

#endif