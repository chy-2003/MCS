#ifndef __MCS_MONTECARLO__
#define __MCS_MONTECARLO__

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
    printf("Start Sum. N = %d, %d\n", N, sizeof(Tar)); fflush(stdout);
    size_t threadsPerBlock = CUBlockSize;
    size_t numberOfBlocks;
    double *Swp = NULL;
    double *Tmp = NULL;
    printf(".\n"); fflush(stdout);
    checkCuda(cudaMallocManaged(&Tmp, sizeof(double) * (N + (CUBlockSize << 1))));
    printf(".\n"); fflush(stdout);
    checkCuda(cudaMemset(Tmp, 0, sizeof(Tmp)));
    printf(".\n"); fflush(stdout);
    while (1) {
        printf("N = %d\n", N); fflush(stdout);
        numberOfBlocks = (N + (CUBlockSize << 1) - 1) / (CUBlockSize << 1);
        Sum_ReductionMain<CUBlockSize><<<numberOfBlocks, threadsPerBlock, sizeof(double) * CUBlockSize>>>(Tar, Tmp, N);
        cudaDeviceSynchronize();
        checkCuda(cudaGetLastError());
        N = numberOfBlocks;
        Swp = Tar; Tar = Tmp; Tmp = Swp;
        if (N >= (CUBlockSize << 4)) 
            checkCuda(cudaMemset(Tmp + N, 0, CUBlockSize * sizeof(double)));
        else
            break;
    }
    double Ans = 0;
    for (int i = 0; i < N; ++i) Ans += Tar[i];
    checkCuda(cudaFree(Tmp));
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
__global__ void GetBandE(double *e, rMesh *mesh, SuperCell *superCell, int n, int Shift) {
    int N = blockIdx.x * blockDim.x + threadIdx.x;
    if (N >= n) return;
    int X = N / superCell->b / superCell->c;
    int Y = (N - X * superCell->b * superCell->c) / superCell->c;
    int Z = (N - X * superCell->b * superCell->c - Y * superCell->c);
    int x, y, z;
    int nDot = blockIdx.y;
    Bond* bond = superCell->unitCell.Dots[nDot].bonds;
    int id = N * superCell->unitCell.BondsCount;
    while (bond != NULL) {
        x = X + bond->Gx; if (x < 0) x += superCell->a; if (x >= superCell->a) x -= superCell->a;
        y = Y + bond->Gy; if (y < 0) y += superCell->b; if (y >= superCell->b) y -= superCell->b;
        z = Z + bond->Gz; if (z < 0) z += superCell->c; if (z >= superCell->c) z -= superCell->c;
        e[id + bond->Index + Shift] = Cal393(mesh->Dots[N], bond->A, mesh->Dots[x * superCell->b * superCell->c + y * superCell->c + z]);
        bond = bond->Next;
    }
    return;
}

void GetEnergy(rMesh *tar, SuperCell *str) {
    SuperCell *gStructure = CopyStructureToGPU(str);
    rMesh *gMesh = CopyRMeshToGPU(tar);
    double *e = NULL;
    int N = str->a * str->b * str->c;
    int NDots = N * str->unitCell.N;
    int NBonds = N * str->unitCell.BondsCount;
    checkCuda(cudaMallocManaged(&e, sizeof(double) * (NDots + NBonds + (CUBlockSize << 1))));
    printf("%d, %d, %d, %d\n", N, NDots, NBonds, sizeof(e));

    size_t threadPerBlock = CUBlockSize;
    dim3 numberOfBlocks((N + CUBlockSize - 1) / CUBlockSize, str->unitCell.N, 1);
    GetDotE<<<numberOfBlocks, threadPerBlock>>>(e, gMesh, gStructure, N);
    GetBandE<<<numberOfBlocks, threadPerBlock>>>(e, gMesh, gStructure, N, NDots);
    cudaDeviceSynchronize();
    tar->Energy = ReductionSum(e, NDots + NBonds);

    DestroyRMeshOnGPU(gMesh);
    DestroyStructureOnGPU(gStructure);
    checkCuda(cudaFree(e));
    return;
}

void DoMonteCarlo(SuperCell *superCell, double *ans, double T, int NSkip, int NLoop, int NCall) {
    rMesh **Mesh = NULL;                                                                     //长度为 NLoop 的 rMesh* 数组
    Mesh = (rMesh**)malloc(sizeof(rMesh*) * NLoop);
    for (int i = 0; i < NLoop; ++i) Mesh[i] = NULL;
    Mesh[0] = InitRMesh(superCell, Vec3(), T);
    printf("InitMesh End.\n"); fflush(stdout);
    GetEnergy(Mesh[0], superCell);
    printf("GetE = %.8lf\n", Mesh[0]->Energy);

    #pragma omp parallel for num_threads(MaxThreads)
    for (int i = 0; i < NLoop; ++i)  
        if (Mesh[i] != NULL) { DestroyRMesh(Mesh[i]); Mesh[i] = NULL; }
    free(Mesh);
    return;
}

#endif