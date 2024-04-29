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

void GetEnergy(rMesh *tar, SuperCell *str) {
    SuperCell *gStructure = CopyStructureToGPU(str);
    rMesh *gMesh = CopyRMeshToGPU(tar);
    double *e = NULL;
    int N = str->a * str->b * str->c;
    int NDots = N * str->unitCell.N;
    int NBonds = N * str->unitCell.BondsCount;
    checkCuda(cudaMallocManaged(&e, sizeof(double) * (NDots + NBonds)));
    printf("%d, %d, %d\n", N, NDots, NBonds); fflush(stdout);

    size_t threadPerBlock = CUBlockSize;
    dim3 numberOfBlocks((N + CUBlockSize - 1) / CUBlockSize, str->unitCell.N, 1);
    GetDotE <<<numberOfBlocks, threadPerBlock>>>(e, gMesh, gStructure, N);
    cudaDeviceSynchronize();
    //GetBandE<<<numberOfBlocks, threadPerBlock>>>(e, gMesh, gStructure, N, NDots);
    //cudaDeviceSynchronize();
    tar->Energy = ReductionSum(e, NDots + NBonds);
    checkCuda(cudaFree(e));

    DestroyRMeshOnGPU(gMesh);
    DestroyStructureOnGPU(gStructure);
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