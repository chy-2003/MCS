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
    int N, NDots;
    rMesh() : Dots(NULL), Field(), T(0), Energy(0), N(0), NDots(0) {}
    ~rMesh() {}
};

rMesh* InitRMesh(SuperCell *superCell, Vec3 Field, double T) {
    rMesh* self = NULL;
    self = (rMesh*)malloc(sizeof(rMesh));
    self->Field = Field; 
    self->T = T;
    self->Energy = 0;
    self->N = superCell->a * superCell->b * superCell->c;
    int n = superCell->unitCell.N;
    self->NDots = n * self->N;
    self->Dots = NULL;
    self->Dots = (Vec3*)malloc(self->NDots * sizeof(Vec3));
    #pragma omp parallel for num_threads(MaxThreads)
    for (int i = 0; i < self->N; ++i) {
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
    checkCuda(cudaMallocManaged(&(tar->unitCell.Dots), sizeof(Dot) * superCell->unitCell.N));
    checkCuda(cudaMemcpy(tar->unitCell.Dots, superCell->unitCell.Dots, sizeof(Dot) * superCell->unitCell.N, cudaMemcpyHostToDevice));
    return tar;
}
void DestroyStructureOnGPU(SuperCell *tar) { checkCuda(cudaFree(tar->unitCell.Dots)); checkCuda(cudaFree(tar)); return; }
rMesh* CopyRMeshToGPU(rMesh *RMesh) {
    rMesh* tar = NULL;
    checkCuda(cudaMallocManaged(&tar, sizeof(rMesh)));
    tar->Dots = NULL; 
    tar->Energy = RMesh->Energy; tar->Field = RMesh->Field; tar->T = RMesh->T;
    tar->N = RMesh->N; tar->NDots = RMesh->NDots;
    checkCuda(cudaMallocManaged(&(tar->Dots), sizeof(Vec3) * (RMesh->NDots)));
    checkCuda(cudaMemcpy(tar->Dots, RMesh->Dots, sizeof(Vec3) * (RMesh->NDots), cudaMemcpyHostToDevice));
    return tar;
}
void DestroyRMeshOnGPU(rMesh *tar) { checkCuda(cudaFree(tar->Dots)); checkCuda(cudaFree(tar)); return; }

struct rBond {
    int S, T, s, t;
    Vec9 A;
    rBond() : S(0), T(0), s(0), t(0), A() {}
    ~rBond() {}
};
struct rBonds {
    rBond* bonds;
    int *Index;
    int NBonds;
    int NIndex;
    int IdC;
    rBonds() : bonds(NULL), Index(NULL), NBonds(0), NIndex(0), IdC(0) {}
    ~rBonds() {}
};
rBonds* ExtractBonds(SuperCell *superCell) {
    rBonds *self = (rBonds*)malloc(sizeof(rBonds));
    int NCells = superCell->a * superCell->b * superCell->c;
    self->NBonds = NCells * superCell->unitCell.BondsCount;
    self->bonds = (rBond*)malloc(sizeof(rBond) * self->NBonds);
    self->IdC = superCell->unitCell.BondsCount * 2 + 1;
    self->NIndex = NCells * self->IdC;
    self->Index = (int*)malloc(sizeof(int) * (self->NIndex));
    
    #pragma omp parallel for num_threads(MaxThreads) 
    for (int i = 0; i < NCells; ++i) self->Index[i * self->IdC] = 0;

    Bond* bond = superCell->unitCell.bonds;
    int bondId = 0;
    while (bond != NULL) {
        #pragma omp parallel for num_threads(MaxThreads)
        for (int i = 0; i < NCells; ++i) {
            int sx = i / (superCell->b * superCell->c);
            int sy = (i - sx * (superCell->b * superCell->c)) / superCell->c;
            int sz = (i - (sx * superCell->b + sy) * superCell->c);
            int tx = sx + bond->Gx; if (tx < 0) tx += superCell->a; if (tx >= superCell->a) tx -= superCell->a;
            int ty = sy + bond->Gy; if (ty < 0) ty += superCell->b; if (ty >= superCell->b) ty -= superCell->b;
            int tz = sz + bond->Gz; if (tz < 0) tz += superCell->c; if (tz >= superCell->c) tz -= superCell->c;
            int id = i * superCell->unitCell.BondsCount + bondId;
            self->bonds[id].A = bond->A;
            self->bonds[id].S = (sx * superCell->b + sy) * superCell->c + sz;
            self->bonds[id].s = bond->s;
            self->bonds[id].T = (tx * superCell->b + ty) * superCell->c + tz;
            self->bonds[id].t = bond->t;
            int j = (tx * superCell->b + ty) * superCell->c + tz;
            self->Index[i * self->IdC] += 1;
            //if (self->Index[i * self->IdC] >= self->IdC) printf("!!!\n");
            self->Index[i * self->IdC + self->Index[i * self->IdC]] = id;
            if (i != j) {
                self->Index[j * self->IdC] += 1;
                self->Index[j * self->IdC + self->Index[j * self->IdC]] = id;
            }
        }
        bond = bond->Next;
        ++bondId;
    }
    return self;
}
void DestroyRBonds(rBonds* self) {
    if (self->bonds != NULL) free(self->bonds);
    if (self->Index != NULL) free(self->Index);
    free(self);
    return;
}
rBonds* CopyRBondsToGPU(rBonds* self) {
    rBonds* tar = NULL;
    checkCuda(cudaMallocManaged(&tar, sizeof(rBonds)));
    tar->IdC = self->IdC;
    tar->NBonds = self->NBonds;
    tar->NIndex = self->NIndex;
    checkCuda(cudaMallocManaged(&(tar->bonds), sizeof(rBond) * (self->NBonds)));
    checkCuda(cudaMallocManaged(&(tar->Index), sizeof(  int) * (self->NIndex)));
    checkCuda(cudaMemcpy(tar->bonds, self->bonds, sizeof(rBond) * (self->NBonds), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(tar->Index, self->Index, sizeof(  int) * (self->NIndex), cudaMemcpyHostToDevice));
    return tar;
}
void DestroyRBondsOnGPU(rBonds *self) {
    if (self->bonds != NULL) checkCuda(cudaFree(self->bonds));
    if (self->Index != NULL) checkCuda(cudaFree(self->Index));
    checkCuda(cudaFree(self));
    return;
}

#endif