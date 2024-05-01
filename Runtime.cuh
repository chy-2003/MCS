#ifndef __MCS__RUNTIME__
#define __MCS__RUNTIME__

#include <omp.h>
#include "CudaInfo.cuh"
#include "Structure.cuh"

#define CUBlockSize 256                                                                      //根据显卡型号调整BlockSize以优化性能，需要为96~1024间的32倍数。一般为128或256
#define CUBlockX 16
#define CUBlockY 16
#define CUBlockZ 1 



int MaxThreads = omp_get_max_threads();

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

void GetEnergyCPU(rMesh *tar, SuperCell *superCell) {
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

rMesh* InitRMesh(SuperCell *superCell, Vec3 Field, double T) {
    rMesh* self = NULL;
    self = (rMesh*)malloc(sizeof(rMesh));
    self->Field = Field; 
    if (T < 1e-9) T = 1e-9;
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
    self->Mag = Vec3(0, 0, 0);
    #pragma omp parallel for num_threads(MaxThreads)
    for (int j = 0; j < n; ++j)
        self->Mag = Add(self->Mag, Mul((superCell->unitCell).Dots[j].a, self->N));
    GetEnergyCPU(self, superCell);
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
    tar->Energy = RMesh->Energy; tar->Field = RMesh->Field; tar->T = RMesh->T; tar->Mag = RMesh->Mag;
    tar->N = RMesh->N; tar->NDots = RMesh->NDots;
    checkCuda(cudaMallocManaged(&(tar->Dots), sizeof(Vec3) * (RMesh->NDots)));
    checkCuda(cudaMemcpy(tar->Dots, RMesh->Dots, sizeof(Vec3) * (RMesh->NDots), cudaMemcpyHostToDevice));
    return tar;
}
rMesh* CopyRMesh(rMesh *RMesh) {
    rMesh* tar = (rMesh*)malloc(sizeof(rMesh));
    tar->Energy = RMesh->Energy; tar->Field = RMesh->Field; tar->T = RMesh->T; tar->Mag = RMesh->Mag;
    tar->N = RMesh->N; tar->NDots = RMesh->NDots;
    tar->Dots = (Vec3*)malloc(sizeof(Vec3) * (RMesh->NDots));
    memcpy(tar->Dots, RMesh->Dots, sizeof(Vec3) * (RMesh->NDots));
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
    e[id] = Cal393((mesh->Dots)[id], (superCell->unitCell).Dots[nDot].A, (mesh->Dots)[id]) +
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

#ifdef __MCS_DEBUG__
    std::chrono::steady_clock::time_point before = std::chrono::steady_clock::now();
#endif

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

#ifdef __MCS_DEBUG__
    std::chrono::steady_clock::time_point  after = std::chrono::steady_clock::now();
    fprintf(stderr, "[DEBUG][from MonteCarlo_GetEnergyGPU] Kernel Time Cost(s) = %.8lf\n", std::chrono::duration<double>(after - before));
#endif    

    DestroyRMeshOnGPU(gMesh);
    DestroyStructureOnGPU(gStructure);
    DestroyRBondsOnGPU(gBonds);
    return;
}

#endif