/*
 *
 * Information
 * 
 * const
 *     MaxThreads                     最大并行线程数
 *
 *
 * define
 *     CUBlockSize 256                cuda每个block线程数量。根据显卡型号调整BlockSize以优化性能，需要为96~1024间的32倍数。一般为128或256
 *     CUBlockX 16                    blockDim.x for 3D
 *     CUBlockY 16                    blockDim.y for 3D
 *     CUBlockz 1                     blockDim.z for 3D
 * 
 *     ModelIsing 1                   only z
 *     ModelXY 2                      only xy
 *     ModelHeisenberg 3              both xyz
 * 
 * 
 * struct
 *     rMesh                          运行时网格信息
 *         double T;                      温度
 *         double Energy;                 总能量
 *         Vec3 Mag;                      总磁矩
 *         int N;                         网格总数
 *         int NDots;                     原子总数
 *         Vec3 *Dots;                    数组，大小为NDots，保存运行时Dot信息
 *     rBond                          解压缩后单条bond信息
 *         int S;                         起始网格编号
 *         int s;                         起始dot原胞内编号
 *         int T;                         终止网格编号
 *         int t;                         终止dot原胞内编号
 *         Vec9 A;                        交换作用，以Kelvin为单位
 *     rBonds                         解压缩后bond信息
 *         rBond *bonds;                  数组，共网格数*原胞bond数个元素
 *         int *Index;                    每个dot相连bond的下标
 *         int NBonds;                    bonds数量
 *         int NIndex;                    index最大数量
 *         int IdC;                       每个dot最大使用index数量
 * 
 * 
 * function
 *     double ReductionSum(double *Tar, int N);                            【对外接口】使用显存规约求和。注意，虽然对于4M个int，kernel
 *                                                                                    仅花费0.2ms，但内存到显存的拷贝花销巨大。在1G个
 *                                                                                    int求和量级之前，一般不会比CPU更快。所以对于内存
 *                                                                                    显存都不会很大的个人PC而言，不需要用到这个函数。
 *     void GetEnergyMCPU(rMesh *tar, SuperCell *superCell);               【对外接口】获得体系总能量，使用多线程CPU。 ModelM
 *     void GetEnergyMCPU_NoOMP(rMesh *tar, SuperCell *superCell);         【对外接口】获得体系总能量，使用单线程CPU。若外层已进行并发
 *                                                                                    则使用这个 ModelM
 *     void GetEnergyECPU_NoOMP(rMesh *tar, SuperCell *superCell);         【对外接口】获得体系总能量，使用单线程CPU。若外层已进行并发
 *                                                                                    则使用这个 ModelE
 *     void GetEnergyGPU(rMesh *tar, SuperCell *str, rBonds *RBonds);      【对外接口】获得体系总能量，使用GPU。与求和相同，对于内存和
 *                                                                                    显存都不是很大的个人PC而言，不需要使用这个函数。
 *     double GetDeltaEM_CPU(rMesh *Mesh, SuperCell* superCell, int X, int Y, int Z, int n, int id1, Vec3 S);
 *                                                                         【对外接口】将位于X,Y,Z,n的dot值改为S的能量差值。 ModelM
 *     double GetDeltaEE_CPU(rMesh *Mesh, SuperCell* superCell, int X, int Y, int Z, int n, int id1, Vec3 S);
 *                                                                         【对外接口】将位于X,Y,Z,n的dot值改为S的能量差值。 ModelE
 *     void UpdateHCPU_NoOMP(rMesh *tar, SuperCell *superCell, Vec3 HDelta);
 *                                                                         【对外接口】更改外场 ModelM
 *     void UpdateECPU_NoOMP(rMesh *tar, SuperCell *superCell, Vec3 HDelta);
 *                                                                         【对外接口】更改外场 ModelE
 *     void GetVec3(double Norm, int Model, double u, double v);           【对外接口】根据参数获得向量
 * 
 * 【以下函数仅和CPU、内存有关】
 *     rMesh* InitRMesh(SuperCell *superCell, Vec3 Field, double T);                  初始化体系，包括计算总磁矩和总能量
 *     rMesh* CopyRMesh(rMesh *RMesh);                                                深度复制一个网格
 *     void DestroyRMesh(rMesh *Tar);                                                 释放一个rMesh
 *     rBonds* ExtractBonds(SuperCell *superCell);                                    获得所有bond信息
 *     void DestroyRBonds(rBonds* self);                                              释放bond信息
 * 
 * 【以下函数与GPU、显存有关】
 *     SuperCell* CopyStructureToGPU(SuperCell *superCell);                           将一个SuperCell结构信息拷贝至显存
 *     void DestroyStructureOnGPU(SuperCell *tar);                                    释放位于显存的SuperCell
 *     rMesh* CopyRMeshToGPU(rMesh *RMesh);                                           将网格拷贝至显存
 *     void DestroyRMeshOnGPU(rMesh *tar);                                            释放位于显存的网格
 *     rBonds* CopyRBondsToGPU(rBonds* self);                                         将bond信息拷贝至显存
 *     void DestroyRBondsOnGPU(rBonds *self);                                         释放位于显存的bond信息
 *     template <unsigned int blockSize>                                              规约求和末端释放
 *         __device__ void warpReduce(volatile double *sdata, unsigned int tid)
 *     template <unsigned int blockSize>                                              规约求和本体
 *         __global__ void Sum_ReductionMain(double *IData, double *OData, unsigned int n)
 *     __global__ void GetDotE(double *e, rMesh *mesh, SuperCell *superCell, int n);  获得dot能量（不包含bond）
 *     __global__ void GetBondE(double *e, rMesh *mesh, rBonds *bonds, UnitCell *unitCell, int n, int Shift);
 *                                                                                    获得bond能量
 */

#ifndef __MCS__RUNTIME__
#define __MCS__RUNTIME__

#include <omp.h>
#include <random>
//#include "CudaInfo.cuh"
#include "Structure.h"
/*
#define CUBlockSize 256
#define CUBlockX 16
#define CUBlockY 16
#define CUBlockZ 1 
*/
#define ModelIsing 1
#define ModelXY 2
#define ModelHeisenberg 3

inline double CoefficientM(int x, int y, int z) { return 1; }
inline double CoefficientEC42AFE(int x, int y, int z) { return ((x + y) & 1) ? -1 : 1; }
inline double CoefficientEP22AFE(int x, int y, int z) { return (x & 1) ? -1 : 1; }
inline double CoefficientEP21FE(int x, int y, int z) { return 1; }

int MaxThreads = omp_get_max_threads();

Vec3 GetVec3(double Norm, int Model, double u, double v = 0) {
    double us, uc, vs, vc;
    switch (Model) {
        case ModelIsing :
            return Vec3(0, 0, (u < 0.5) ? Norm : -Norm);
        case ModelXY :
            return Vec3(std::sin(u) * Norm, std::cos(u) * Norm, 0);
        case ModelHeisenberg :
            u *= 2.0 * Pi; v = std::acos(2.0 * v - 1);
            us = std::sin(u); uc = std::cos(u); vs = std::sin(v); vc = std::cos(v);
            return Vec3(us * vs * Norm, uc * vs * Norm, vc * Norm);
        default :
            return Vec3(0, 0, 0);
    }
}

struct rMesh {                                                                               //4维数组,a,b,c,N
    Vec3 *Dots;
    Vec3 Field;                                                                             //外场
    double T;                                                                               //温度
    double Energy;                                                                          //总能量
    Vec3 Mag;                                                                               //平均Mag
    int N, NDots;
    rMesh() : Dots(NULL), Field(), T(0), Energy(0), N(0), NDots(0), Mag() {}
    ~rMesh() {}
};

void GetEnergyMCPU(rMesh *tar, SuperCell *superCell) {
    double Energy = 0;
    #pragma omp parallel for num_threads(MaxThreads) reduction(+: Energy)
    for (int x = 0; x < superCell->a; ++x)
        for (int y = 0; y < superCell->b; ++y) 
            for (int z = 0; z < superCell->c; ++z) {
                int id1 = ((x * superCell->b + y) * superCell->c + z) * superCell->unitCell.N;
                for (int n = 0; n < superCell->unitCell.N; ++n) {
                    Energy += Cal393(tar->Dots[id1 + n], superCell->unitCell.Dots[n].A, tar->Dots[id1 + n]) +
                              InMul(tar->Field, tar->Dots[id1 + n]);
                }
                Bond* bond = superCell->unitCell.bonds;
                int X, Y, Z, id2;
                while (bond != NULL) {
                    X = x + bond->Gx; if (X < 0) X += superCell->a; if (X >= superCell->a) X -= superCell->a;
                    Y = y + bond->Gy; if (Y < 0) Y += superCell->b; if (Y >= superCell->b) Y -= superCell->b;
                    Z = z + bond->Gz; if (Z < 0) Z += superCell->c; if (Z >= superCell->c) Z -= superCell->c;
                    id2 = ((X * superCell->b + Y) * superCell->c + z) * superCell->unitCell.N;
                    Energy += Cal393(tar->Dots[id1 + bond->s], bond->A, tar->Dots[id2 + bond->t]);
                    bond = bond->Next;
                }
            }
    tar->Energy = Energy;
    return;
}

void GetEnergyMCPU_NoOMP(rMesh *tar, SuperCell *superCell) {
    double Energy = 0;
    for (int x = 0; x < superCell->a; ++x)
        for (int y = 0; y < superCell->b; ++y) 
            for (int z = 0; z < superCell->c; ++z) {
                int id1 = ((x * superCell->b + y) * superCell->c + z) * superCell->unitCell.N;
                for (int n = 0; n < superCell->unitCell.N; ++n) {
                    Energy += Cal393(tar->Dots[id1 + n], superCell->unitCell.Dots[n].A, tar->Dots[id1 + n]) -
                              InMul(tar->Field, tar->Dots[id1 + n]);
                }
                Bond* bond = superCell->unitCell.bonds;
                int X, Y, Z, id2;
                while (bond != NULL) {
                    X = x + bond->Gx; if (X < 0) X += superCell->a; if (X >= superCell->a) X -= superCell->a;
                    Y = y + bond->Gy; if (Y < 0) Y += superCell->b; if (Y >= superCell->b) Y -= superCell->b;
                    Z = z + bond->Gz; if (Z < 0) Z += superCell->c; if (Z >= superCell->c) Z -= superCell->c;
                    id2 = ((X * superCell->b + Y) * superCell->c + z) * superCell->unitCell.N;
                    Energy += Cal393(tar->Dots[id1 + bond->s], bond->A, tar->Dots[id2 + bond->t]);
                    bond = bond->Next;
                }
            }
    tar->Energy = Energy;
    return;
}

inline int Sign(double x) { if (x > 0) return 1; else if (x < 0) return -1; return 0; }

void GetEnergyECPU_NoOMP(rMesh *tar, SuperCell *superCell) {
    double Energy = 0;
    for (int x = 0; x < superCell->a; ++x)
        for (int y = 0; y < superCell->b; ++y)
            for (int z = 0; z < superCell->c; ++z) {
                int id1 = ((x * superCell->b + y) * superCell->c + z) * superCell->unitCell.N;
                for (int n = 0; n < superCell->unitCell.N; ++n) {
                    double norm = superCell->unitCell.Dots[n].Norm;
                    norm = norm * norm;
                    Energy += superCell->unitCell.Dots[n].A.xx * norm / 2 + 
                            superCell->unitCell.Dots[n].A.xy * norm *norm / 4 + 
                            superCell->unitCell.Dots[n].A.zx * norm * norm * norm / 6 +
                            superCell->unitCell.Dots[n].A.yx * norm * norm * norm * norm / 8;
                    if (x == 0 || x == superCell->a - 1)
                        Energy += InMul(tar->Field, tar->Dots[id1 + n]);
                }
                Bond* bond = superCell->unitCell.bonds;
                int X, Y, Z, id2;
                while (bond != NULL) {
                    X = x + bond->Gx; if (X < 0) X += superCell->a; if (X >= superCell->a) X -= superCell->a;
                    Y = y + bond->Gy; if (Y < 0) Y += superCell->b; if (Y >= superCell->b) Y -= superCell->b;
                    Z = z + bond->Gz; if (Z < 0) Z += superCell->c; if (Z >= superCell->c) Z -= superCell->c;
                    id2 = ((X * superCell->b + Y) * superCell->c + z) * superCell->unitCell.N;
                    Energy += Cal393(tar->Dots[id1 + bond->s], bond->A, tar->Dots[id2 + bond->t]);
                    bond = bond->Next;
                }
            }
    for (int x = 0; x < superCell->a; ++x) {
        int y = 0;
        while (y + 1 < superCell->b && 
            Sign(tar->Dots[x * superCell->b + y].z) == Sign(tar->Dots[x * superCell->b + y + 1].z)) ++y;
        int Flg = 1, Cnt; 
        if (y >= superCell->b - 1) {
            Cnt = superCell->b;
            if (Cnt == 3) Energy += superCell->unitCell.Dots[0].A.yy;
            if (Cnt == 4) Energy += superCell->unitCell.Dots[0].A.yz;
            if (Cnt == 5) Energy += superCell->unitCell.Dots[0].A.zx;
            if (Cnt == 6) Energy += superCell->unitCell.Dots[0].A.zy;
            if (Cnt >= 7) Energy += superCell->unitCell.Dots[0].A.zz;
        } else {
            while (Flg) {
                Cnt = 1; ++y;
                if (y >= superCell->b) { y = 0; Flg = 0; }
                while (Sign(tar->Dots[x * superCell->b + y].z) == 
                    Sign(tar->Dots[x * superCell->b + ((y + 1 >= superCell->b) ? 0 : y + 1)].z)) {
                    ++y; ++Cnt;
                    if (y >= superCell->b) { y = 0; Flg = 0; }
                }
                if (Cnt == 3) Energy += superCell->unitCell.Dots[0].A.yy;
                if (Cnt == 4) Energy += superCell->unitCell.Dots[0].A.yz;
                if (Cnt == 5) Energy += superCell->unitCell.Dots[0].A.zx;
                if (Cnt == 6) Energy += superCell->unitCell.Dots[0].A.zy;
                if (Cnt >= 7) Energy += superCell->unitCell.Dots[0].A.zz;
            }
        }
    }
    tar->Energy = Energy;
    return;
}

void UpdateHCPU_NoOMP(rMesh *tar, SuperCell *superCell, Vec3 HDelta) {
    int N = superCell->a * superCell->b * superCell->c * superCell->unitCell.N;
    for (int i = 0; i < N; ++i)
        tar->Energy += -InMul(HDelta, tar->Dots[i]);
    tar->Field = Add(tar->Field, HDelta);
    return;
}

void UpdateECPU_NoOMP(rMesh *tar, SuperCell *superCell, Vec3 HDelta) {
    for (int y = 0; y < superCell->b; ++y) {
        tar->Energy += InMul(HDelta, tar->Dots[y]);
        tar->Energy += InMul(HDelta, tar->Dots[(superCell->a - 1) * superCell->b + y]);
    }
    tar->Field = Add(tar->Field, HDelta);
    return;
}

rMesh* InitRMesh(SuperCell *superCell, Vec3 Field, double T, int Model, void (*GetEnergy)(rMesh*, SuperCell*), double Coefficient(int, int, int)) {
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
    std::random_device RandomDevice;
    std::mt19937 Mt19937(RandomDevice());
    std::uniform_real_distribution<> URD(0, 1);
    for (int i = 0; i < self->N; ++i) {
        for (int j = 0; j < n; ++j) {
            self->Dots[i * n + j] = GetVec3(superCell->unitCell.Dots[j].Norm, Model, URD(Mt19937), URD(Mt19937));
            //self->Dots[i * n + j] = Vec3(0, 0, ((i / superCell->b + i % superCell->b) & 1) ? 
            //        -superCell->unitCell.Dots[j].Norm : superCell->unitCell.Dots[j].Norm);
        }
    }

    self->Mag = Vec3(0, 0, 0);
    for (int i = 0; i < superCell->a; ++i)
        for (int j = 0; j < superCell->b; ++j)
            for (int k = 0; k < superCell->c; ++k)
                for (int l = 0; l < superCell->unitCell.N; ++l) {
                    self->Mag = Add(self->Mag, 
                            Mul(self->Dots[((i * superCell->b + j) * superCell->c + k) * superCell->unitCell.N + l], Coefficient(i, j, k)));
                }

    GetEnergy(self, superCell);
    //printf("Energy = %12.8lf, T = %.2lf\n", self->Energy, self->T);
    return self;
}
void DestroyRMesh(rMesh *Tar) { free(Tar->Dots); free(Tar); return; }
/*
SuperCell* CopyStructureToGPU(SuperCell *superCell) {
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
    tar->Energy = RMesh->Energy; tar->Field = RMesh->Field; tar->T = RMesh->T; tar->Mag = RMesh->Mag;
    tar->N = RMesh->N; tar->NDots = RMesh->NDots;
    checkCuda(cudaMallocManaged(&(tar->Dots), sizeof(Vec3) * (RMesh->NDots)));
    checkCuda(cudaMemcpy(tar->Dots, RMesh->Dots, sizeof(Vec3) * (RMesh->NDots), cudaMemcpyHostToDevice));
    return tar;
}
*/
rMesh* CopyRMesh(rMesh *RMesh) {
    rMesh* tar = (rMesh*)malloc(sizeof(rMesh));
    tar->Energy = RMesh->Energy; tar->Field = RMesh->Field; tar->T = RMesh->T; tar->Mag = RMesh->Mag;
    tar->N = RMesh->N; tar->NDots = RMesh->NDots;
    tar->Dots = (Vec3*)malloc(sizeof(Vec3) * (RMesh->NDots));
    memcpy(tar->Dots, RMesh->Dots, sizeof(Vec3) * (RMesh->NDots));
    return tar;
}
//void DestroyRMeshOnGPU(rMesh *tar) { checkCuda(cudaFree(tar->Dots)); checkCuda(cudaFree(tar)); return; }

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
    self->NBonds = NCells * superCell->unitCell.NBonds;
    self->bonds = (rBond*)malloc(sizeof(rBond) * self->NBonds);
    self->IdC = superCell->unitCell.NBonds * 2 + 1;
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
            int id = i * superCell->unitCell.NBonds + bondId;
            self->bonds[id].A = bond->A;
            self->bonds[id].S = (sx * superCell->b + sy) * superCell->c + sz;
            self->bonds[id].s = bond->s;
            self->bonds[id].T = (tx * superCell->b + ty) * superCell->c + tz;
            self->bonds[id].t = bond->t;
            int j = (tx * superCell->b + ty) * superCell->c + tz;
            self->Index[i * self->IdC] += 1;
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
/*
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
*/
/*
template <unsigned int blockSize>
__device__ void warpReduce(volatile double *sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid +  8];
    if (blockSize >=  8) sdata[tid] += sdata[tid +  4];
    if (blockSize >=  4) sdata[tid] += sdata[tid +  2];
    if (blockSize >=  2) sdata[tid] += sdata[tid +  1];
    return;
}
template <unsigned int blockSize>
__global__ void Sum_ReductionMain(double *IData, double *OData, unsigned int n) {
    __shared__ double sdata[CUBlockSize];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    sdata[tid] = 0;

    while (i < n) { sdata[tid] += IData[i] + IData[i + blockSize]; i += gridSize; }
    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) sdata[tid] += sdata[tid +  64]; __syncthreads(); }
    if (tid < 32) warpReduce<CUBlockSize>(sdata, tid);
    if (tid == 0) OData[blockIdx.x] = sdata[0];
    return;
}

double ReductionSum(double *Tar, int N) {
    size_t threadsPerBlock = CUBlockSize;
    size_t numberOfBlocks;
    double *Swp = NULL;
    double *a = NULL;
    double *b = NULL;
    checkCuda(cudaMallocManaged(&a, sizeof(double) * (N + CUBlockSize)));
    checkCuda(cudaMemcpy(a, Tar, sizeof(double) * N, cudaMemcpyDeviceToDevice));
    checkCuda(cudaMallocManaged(&b, sizeof(double) * (N + CUBlockSize)));
    while (N >= (CUBlockSize << 4)) {
        numberOfBlocks = (N + (CUBlockSize << 1) - 1) / (CUBlockSize << 1);
        checkCuda(cudaMemset(b + numberOfBlocks, 0, CUBlockSize * sizeof(double)));
        Sum_ReductionMain<CUBlockSize><<<numberOfBlocks, threadsPerBlock, sizeof(double) * CUBlockSize>>>(a, b, N);
        cudaDeviceSynchronize();
        checkCuda(cudaGetLastError());
        N = numberOfBlocks;
        Swp = a; a = b; b = Swp;
    }
    double Ans = 0;
    for (int i = 0; i < N; ++i) Ans += a[i];
    checkCuda(cudaFree(a));
    checkCuda(cudaFree(b));
    return Ans;
}



__global__ void GetDotE(double *e, rMesh *mesh, SuperCell *superCell, int n) {
    int N = blockIdx.x * blockDim.x + threadIdx.x;
    if (N >= n) return;
    int nDot = blockIdx.y;
    int id = N * superCell->unitCell.N + nDot;
    e[id] = Cal393((mesh->Dots)[id], (superCell->unitCell).Dots[nDot].A, (mesh->Dots)[id]) -
            InMul(mesh->Field, (mesh->Dots)[id]);
    return;
}
__global__ void GetBondE(double *e, rMesh *mesh, rBonds *bonds, UnitCell *unitCell, int n, int Shift) {
    int N = blockIdx.x * blockDim.x + threadIdx.x;
    if (N >= n) return;
    e[Shift + N] = Cal393(mesh->Dots[bonds->bonds[N].S * unitCell->N + bonds->bonds[N].s], 
                    bonds->bonds[N].A,
                    mesh->Dots[bonds->bonds[N].T * unitCell->N + bonds->bonds[N].t]);
    return;
}

void GetEnergyGPU(rMesh *tar, SuperCell *str, rBonds *RBonds) {
    SuperCell *gStructure = CopyStructureToGPU(str);
    rMesh *gMesh = CopyRMeshToGPU(tar);
    rBonds *gBonds = CopyRBondsToGPU(RBonds);

    double *e = NULL;
    int N = str->a * str->b * str->c;
    int NDots = N * str->unitCell.N;
    int NBonds = N * str->unitCell.NBonds;
    checkCuda(cudaMallocManaged(&e, sizeof(double) * (NDots + NBonds)));

    size_t threadPerBlock = CUBlockSize;
    dim3 numberOfBlocks1((N + CUBlockSize - 1) / CUBlockSize, str->unitCell.N, 1);
    GetDotE<<<numberOfBlocks1, threadPerBlock>>>(e, gMesh, gStructure, N);
    cudaDeviceSynchronize();

    size_t numberOfBlocks2 = (gBonds->NBonds + CUBlockSize - 1) / CUBlockSize;
    GetBondE<<<numberOfBlocks2, threadPerBlock>>>(e, gMesh, gBonds, &(gStructure->unitCell), gBonds->NBonds, NDots);
    cudaDeviceSynchronize();

    tar->Energy = ReductionSum(e, NDots + NBonds);
    checkCuda(cudaFree(e));  

    DestroyRMeshOnGPU(gMesh);
    DestroyStructureOnGPU(gStructure);
    DestroyRBondsOnGPU(gBonds);
    return;
}
*/
double GetDeltaEM_CPU(rMesh *Mesh, SuperCell* superCell, int X, int Y, int Z, int n, Vec3 S) {
    int id1 = ((X * superCell->b + Y) * superCell->c + Z) * superCell->unitCell.N + n;
    int id2;
    int x, y, z;
    Bond* bond = superCell->unitCell.bonds;
    double Ans = Cal393(              S, superCell->unitCell.Dots[n].A,               S) - 
                 Cal393(Mesh->Dots[id1], superCell->unitCell.Dots[n].A, Mesh->Dots[id1]) - 
                 InMul(Mesh->Field,               S) + 
                 InMul(Mesh->Field, Mesh->Dots[id1]);
    while (bond != NULL) {
        if (n == bond->s) {
            x = X + bond->Gx; if (x < 0) x += superCell->a; if (x >= superCell->a) x -= superCell->a;
            y = Y + bond->Gy; if (y < 0) y += superCell->b; if (y >= superCell->b) y -= superCell->b;
            z = Z + bond->Gz; if (z < 0) z += superCell->c; if (z >= superCell->c) z -= superCell->c;
            id2 = ((x * superCell->b + y) * superCell->c + z) * superCell->unitCell.N + bond->t;
            Ans += Cal393(S, bond->A, Mesh->Dots[id2]) - Cal393(Mesh->Dots[id1], bond->A, Mesh->Dots[id2]);
        }
        if (n == bond->t) {
            x = X - bond->Gx; if (x < 0) x += superCell->a; if (x >= superCell->a) x -= superCell->a;
            y = Y - bond->Gy; if (y < 0) y += superCell->b; if (y >= superCell->b) y -= superCell->b;
            z = Z - bond->Gz; if (z < 0) z += superCell->c; if (z >= superCell->c) z -= superCell->c;
            id2 = ((x * superCell->b + y) * superCell->c + z) * superCell->unitCell.N + bond->s;
            Ans += Cal393(Mesh->Dots[id2], bond->A, S) - Cal393(Mesh->Dots[id2], bond->A, Mesh->Dots[id1]);
        }
        bond = bond->Next;
    }
    return Ans;
}

double GetDeltaEE_CPU(rMesh *Mesh, SuperCell *superCell, int X, int Y, int Z, int n, Vec3 S) {
    int id1 = ((X * superCell->b + Y) * superCell->c + Z) * superCell->unitCell.N + n;
    int id2;
    int x, y, z;
    Bond* bond = superCell->unitCell.bonds;
    double Ans = 0;
    if (X == 0 || X == superCell->a - 1) Ans = InMul(Mesh->Field, S) - InMul(Mesh->Field, Mesh->Dots[id1]);
    while (bond != NULL) {
        if (n == bond->s) {
            x = X + bond->Gx; if (x < 0) x += superCell->a; if (x >= superCell->a) x -= superCell->a;
            y = Y + bond->Gy; if (y < 0) y += superCell->b; if (y >= superCell->b) y -= superCell->b;
            z = Z + bond->Gz; if (z < 0) z += superCell->c; if (z >= superCell->c) z -= superCell->c;
            id2 = ((x * superCell->b + y) * superCell->c + z) * superCell->unitCell.N + bond->t;
            Ans += Cal393(S, bond->A, Mesh->Dots[id2]) - Cal393(Mesh->Dots[id1], bond->A, Mesh->Dots[id2]);
        }
        if (n == bond->t) {
            x = X - bond->Gx; if (x < 0) x += superCell->a; if (x >= superCell->a) x -= superCell->a;
            y = Y - bond->Gy; if (y < 0) y += superCell->b; if (y >= superCell->b) y -= superCell->b;
            z = Z - bond->Gz; if (z < 0) z += superCell->c; if (z >= superCell->c) z -= superCell->c;
            id2 = ((x * superCell->b + y) * superCell->c + z) * superCell->unitCell.N + bond->s;
            Ans += Cal393(Mesh->Dots[id2], bond->A, S) - Cal393(Mesh->Dots[id2], bond->A, Mesh->Dots[id1]);
        }
        bond = bond->Next;
    }
    int cnt1 = 0;
    y = Y;
    while (y > 0 && Sign(Mesh->Dots[id1].z) == Sign(Mesh->Dots[id1 - Y + y - 1].z)) { --y; ++cnt1; }
    y = Y;
    while (y + 1 < superCell->b && Sign(Mesh->Dots[id1].z) == Sign(Mesh->Dots[id1 - Y + y + 1].z)) { ++y; ++cnt1; }
    ++cnt1;
    if (cnt1 == 3) Ans -= superCell->unitCell.Dots[0].A.yy;
    if (cnt1 == 4) Ans -= superCell->unitCell.Dots[0].A.yz;
    if (cnt1 == 5) Ans -= superCell->unitCell.Dots[0].A.zx;
    if (cnt1 == 6) Ans -= superCell->unitCell.Dots[0].A.zy;
    if (cnt1 == 7) Ans -= superCell->unitCell.Dots[0].A.zz;

    y = Y; cnt1 = 0;
    while (y > 0 && -Sign(Mesh->Dots[id1].z) == Sign(Mesh->Dots[id1 - Y + y - 1].z)) { --y; ++cnt1; }
    y = Y;
    while (y + 1 < superCell->b && -Sign(Mesh->Dots[id1].z) == Sign(Mesh->Dots[id1 - Y + y + 1].z)) { ++y; ++cnt1; }
    ++cnt1;
    if (cnt1 == 3) Ans += superCell->unitCell.Dots[0].A.yy;
    if (cnt1 == 4) Ans += superCell->unitCell.Dots[0].A.yz;
    if (cnt1 == 5) Ans += superCell->unitCell.Dots[0].A.zx;
    if (cnt1 == 6) Ans += superCell->unitCell.Dots[0].A.zy;
    if (cnt1 == 7) Ans += superCell->unitCell.Dots[0].A.zz;

    return Ans;
}

#endif