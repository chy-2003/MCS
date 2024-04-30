#ifndef __MCS_MONTECARLO__
#define __MCS_MONTECARLO__

#include <chrono>
#include "Structure.cuh"
#include "Runtime.cuh"
#define CUBlockSize 256                                                                      //根据显卡型号调整BlockSize以优化性能，需要为96~1024间的32倍数。一般为128或256
#define CUBlockX 16
#define CUBlockY 16
#define CUBlockZ 1 

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
    e[id] = Cal933((superCell->unitCell).Dots[nDot].A, (mesh->Dots)[id], (mesh->Dots)[id]) +
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

    std::chrono::steady_clock::time_point before = std::chrono::steady_clock::now();


    double *e = NULL;
    int N = str->a * str->b * str->c;
    int NDots = N * str->unitCell.N;
    int NBonds = N * str->unitCell.BondsCount;
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


    std::chrono::steady_clock::time_point  after = std::chrono::steady_clock::now();
    fprintf(stderr, "[DEBUG][from MonteCarlo_GetEnergyGPU] Kernel Time Cost(s) = %.8lf\n", std::chrono::duration<double>(after - before));
    
    DestroyRMeshOnGPU(gMesh);
    DestroyStructureOnGPU(gStructure);
    DestroyRBondsOnGPU(gBonds);
    return;
}

void GetEnergyCPU(rMesh *tar, SuperCell *superCell, rBonds *RBonds) {
    tar->Energy = 0;
    //#pragma omp parallel for num_threads(MaxThreads)
    for (int x = 0; x < superCell->a; ++x)
        for (int y = 0; y < superCell->b; ++y) 
            for (int z = 0; z < superCell->c; ++z) {
                int id1 = ((x * superCell->b + y) * superCell->c + z) * superCell->unitCell.N;
                for (int n = 0; n < superCell->unitCell.N; ++n) {
                    tar->Energy += Cal933(superCell->unitCell.Dots[n].A, tar->Dots[id1 + n], tar->Dots[id1 + n]);
                    tar->Energy += InMul(tar->Field, tar->Dots[id1 + n]);
                }
                Bond* bond = superCell->unitCell.bonds;
                int X, Y, Z, id2;
                while (bond != NULL) {
                    X = x + bond->Gx; if (X < 0) X += superCell->a; if (X >= superCell->a) X -= superCell->a;
                    Y = y + bond->Gy; if (Y < 0) Y += superCell->b; if (Y >= superCell->b) Y -= superCell->b;
                    Z = z + bond->Gz; if (Z < 0) Z += superCell->c; if (Z >= superCell->c) Z -= superCell->c;
                    id2 = ((X * superCell->b + Y) * superCell->c + z) * superCell->unitCell.N;
                    tar->Energy += Cal393(tar->Dots[id1 + bond->s], bond->A, tar->Dots[id2 + bond->t]);
                    bond = bond->Next;
                }
            }
    return;
}

void DoMonteCarlo(SuperCell *superCell, double *ans, double T, int NSkip, int NLoop, int NCall) {
    rMesh **Mesh = NULL;                                                                     //长度为 NLoop 的 rMesh* 数组
    Mesh = (rMesh**)malloc(sizeof(rMesh*) * NLoop);
    for (int i = 0; i < NLoop; ++i) Mesh[i] = NULL;
    Mesh[0] = InitRMesh(superCell, Vec3(), T);
    fprintf(stderr, "[DEBUG][from MonteCarlo_DoMonteCarlo] InitMesh End.\n"); fflush(stdout);
    rBonds *RBonds = ExtractBonds(superCell);
    fprintf(stderr, "[DEBUG][from MonteCarlo_DoMonteCarlo] Unzip bonds End.\n"); fflush(stdout);

    std::chrono::steady_clock::time_point before = std::chrono::steady_clock::now();
    GetEnergyGPU(Mesh[0], superCell, RBonds);
    std::chrono::steady_clock::time_point  after = std::chrono::steady_clock::now();
    fprintf(stderr, "[DEBUG][from MonteCarlo_DoMonteCarlo] GetEGPU = %.8lf, time(s) = %.8lf\n", Mesh[0]->Energy, 
            std::chrono::duration<double>(after - before));

    before = std::chrono::steady_clock::now();
    GetEnergyCPU(Mesh[0], superCell, RBonds);
    after = std::chrono::steady_clock::now();
    fprintf(stderr, "[DEBUG][from MonteCarlo_DoMonteCarlo] GetECPU = %.8lf, time(s) = %.8lf\n", Mesh[0]->Energy, 
            std::chrono::duration<double>(after - before));

    #pragma omp parallel for num_threads(MaxThreads)
    for (int i = 0; i < NLoop; ++i)  
        if (Mesh[i] != NULL) { DestroyRMesh(Mesh[i]); Mesh[i] = NULL; }
    free(Mesh);
    DestroyRBonds(RBonds);
    return;
}

#endif