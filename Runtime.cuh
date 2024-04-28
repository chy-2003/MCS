#ifndef __MCS__RUNTIME__
#define __MCS__RUNTIME__

#include <omp.h>
#include "CudaInfo.cuh"
#include "Structure.cuh"


int MaxThreads = omp_get_max_threads();

struct rMesh {                                                                               //4维数组,a,b,c,N
    Vec3 *Dots;
    Vec3 Field;                                                                             //外场
    double T;                                                                               //温度
    double Energy;
    rMesh() : Dots(NULL) {}
    ~rMesh() {}
};

rMesh* InitRMesh(SuperCell *superCell, Vec3 Field, double T) {
    rMesh* self = NULL;
    self = (rMesh*)malloc(sizeof(rMesh));
    self->Field = Field; self->T = T;
    self->Energy = 0;
    int N = superCell->a * superCell->b * superCell->c;
    int n = (superCell->unitCell).N;
    self->Dots = (Vec3*)calloc(N * n, sizeof(Vec3));
    #pragma omp parallel for num_threads(MaxThreads)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < n; ++j)
            self->Dots[i * n + j] = (superCell->unitCell).Dots[j].a;
    }
    return self;
}
void DestroyRMesh(rMesh *Tar) { free(Tar->Dots); free(Tar); return; }

SuperCell* CopyStructureToGPU(SuperCell *superCell) {                                        //共享内存，但注意减少CPU访问以提高运算速度
    SuperCell *tar;
    checkCuda(cudaMallocManaged(&tar, sizeof(SuperCell)));
    tar->a = superCell->a; tar->b = superCell->b; tar->c = superCell->c; 
    tar->unitCell = superCell->unitCell; tar->unitCell.Dots = NULL;
    checkCuda(cudaMallocManaged(&(tar->unitCell.Dots), sizeof(superCell->unitCell.Dots)));
    checkCuda(cudaMemcpy(tar->unitCell.Dots, superCell->unitCell.Dots, sizeof(superCell->unitCell.Dots), cudaMemcpyHostToDevice));
    return tar;
}
void DestroyStructureOnGPU(SuperCell *tar) { checkCuda(cudaFree(tar->unitCell.Dots)); checkCuda(cudaFree(tar)); return; }
rMesh* CopyRMeshToGPU(rMesh *RMesh) {
    rMesh* tar = NULL;
    checkCuda(cudaMallocManaged(&tar, sizeof(rMesh)));
    tar->Dots = NULL; 
    tar->Energy = RMesh->Energy; tar->Field = RMesh->Field; tar->T = RMesh->T;
    checkCuda(cudaMallocManaged(&(tar->Dots), sizeof(RMesh->Dots)));
    checkCuda(cudaMemcpy(tar->Dots, RMesh->Dots, sizeof(RMesh->Dots), cudaMemcpyHostToDevice));
    return tar;
}
void DestroyRMeshOnGPU(rMesh *tar) { checkCuda(cudaFree(tar->Dots)); checkCuda(cudaFree(tar)); return; }

#endif