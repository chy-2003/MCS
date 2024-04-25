#ifndef __MCS__RUNTIME__
#define __MCS__RUNTIME__

#include "CudaInfo.cuh"
#include "Structure.cuh"

struct rUnit {
    Vec3 *Dot;
    rUnit() : Dot(NULL) {}
    ~rUnit() {}
};
void DestroyRUnit(rUnit *self) {
    checkCuda(cudaFree(self->Dot));
    checkCuda(cudaFree(self));
    return;
}

struct rMesh {
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
    return self;
}

#endif