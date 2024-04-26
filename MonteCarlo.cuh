#ifndef __MCS_MONTECARLO__
#define __MCS_MONTECARLO__

#include "Structure.cuh"
#include "Runtime.cuh"
#define CUBlockSize 256

__global__ void GetEnergy(double *ans, rMesh *mesh, SuperCell *superCell, double *ReductionTemp) {
    //unfinished
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;
    int X, Y, Z;
    int n = threadIdx.z;
    int id1 = x * (superCell->b * superCell->c) + y * (superCell->c) + z;
    int Id = x * (superCell->b * superCell->c * superCell->unitCell->N) + 
              y * (superCell->c * superCell->unitCell->N) + 
              z * (superCell->unitCell->N) + 
              n;
    int id2;
    ReductionTemp[Id] = Cal933(superCell->unitCell->Dots[n].A, (mesh->Unit)[id1].Dots + n, (mesh->Unit)[id1].Dots + n);
    Bond *bond = superCell->unitCell->Dots->bonds;
    while (bond != NULL) {
        X = x + bond->Gx; if (X < 0) X += superCell->a; if (X >= superCell->a) X -= superCell->a;
        Y = y + bond->Gy; if (Y < 0) Y += superCell->b; if (Y >= superCell->b) Y -= superCell->b;
        Z = z + bond->Gz; if (Z < 0) Z += superCell->c; if (Z >= superCell->c) Z -= superCell->c;
        id2 = X * (superCell->b * superCell->c) + Y * (superCell->c) + Z;
        ReductionTemp[Id] += Cal933(bond->A, (mesh->Unit)[id1].Dots + n, (mesh->Unit)[id2].Dots + (bond->t));
        bond = bond->Next;
    }
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

double ReductionSum(double *Tar, int N, double *Tmp) {
    size_t threadsPerBlock = CUBlockSize;
    size_t numberOfBlocks;
    double *Swp = NULL;
    while (1) {
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
    return Ans;
}

void MonteCarlo_Range(double *ans, SuperCell *superCell, double L, double R, int Points, int NSkip, int NCal) {
    rMesh* RMesh;
    checkCuda(cudaMallocManaged(&RMesh, sizeof(rMesh) * Points));
    #pragma omp parallel for num_threads(Points)
    for (int i = 0; i < Points; ++i) BuildRMesh(RMesh + i, superCell);

    double *Energy = NULL;
    double *ReductionTemp = NULL;
    double *RedSwap = NULL;
    int N = superCell->a * superCell->b * superCell->c * superCell->unitCell->N;
    checkCuda(cudaMallocManaged(&Energy, sizeof(double) * Points));
    checkCuda(cudaMallocManaged(&ReductionTemp, sizeof(double) * N));
    checkCuda(cudaMallocManaged(&RedSwap, sizeof(double) * ((N + (CUBlockSize << 1) - 1) / (CUBlockSize << 1))));
    dim3 threadsPerBlock(16, 16, superCell->unitCell->N);
    dim3 numberOfBlocks((superCell->a + 15) / 16, (superCell->b + 15) / 16, superCell->c);
    GetEnergy<<<numberOfBlocks, threadsPerBlock>>>(Energy, RMesh, superCell, ReductionTemp);
    cudaDeviceSynchronize();
    Energy[0] = ReductionSum(ReductionTemp, N, RedSwap);
    #pragma omp parallel for num_threads(Points - 1)
    for (int i = 1; i < Points; ++i) Energy[i] = ReductionTemp[0];

    checkCuda(cudaFree(ReductionTemp));
    checkCuda(cudaFree(RedSwap));
    checkCuda(cudaFree(Energy));

    #pragma omp parallel for num_threads(Points)
    for (int i = 0; i < Points; ++i) DestroyRMesh(RMesh + i, superCell);
    checkCuda(cudaFree(RMesh));
}

#endif