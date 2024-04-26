#ifndef __MCS_MONTECARLO__
#define __MCS_MONTECARLO__

#include "Structure.cuh"
#include "Runtime.cuh"

__global__ void GetEnergy(double *ans, rMesh *mesh, SuperCell *superCell, double *ReductionTemp) {
    //unfinished
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;
    int n = threadIdx.z;
    int id = x * (superCell->b * superCell->c) + y * (superCell->c) + z;
    ReductionTemp[id] = Cal933(superCell->unitCell->Dots[n].A, (mesh->Unit)[id].Dots[n], (mesh->Unit)[id].Dots[n]);
    return;
}

void MonteCarlo_Range(double *ans, SuperCell *superCell, double L, double R, int Points, int NSkip, int NCal) {
    rMesh* RMesh;
    checkCuda(cudaMallocManaged(&RMesh, sizeof(rMesh) * Points));
    #pragma omp parallel for num_threads(Points)
    for (int i = 0; i < Points; ++i) BuildRMesh(RMesh + i, superCell);

    double *Energy = NULL;
    double *ReductionTemp = NULL;
    int N = superCell->a * superCell->b * superCell->c * superCell->unitCell->N;
    checkCuda(cudaMallocManaged(&Energy, sizeof(double) * Points));
    checkCuda(cudaMallocManaged(&ReductionTemp, sizeof(double) * N));
    
    dim3 threadsPerBlock(16, 16, superCell->unitCell->N);
    dim3 numberOfBlocks((superCell->a + 15) / 16, (superCell->b + 15) / 16, superCell->c);
    GetEnergy<<<numberOfBlocks, threadsPerBlock>>>(Energy, RMesh, superCell, ReductionTemp);
    cudaDeviceSynchronize();
    #pragma omp parallel for num_threads(Points - 1)
    for (int i = 1; i < Points; ++i) Energy[i] = Energy[0];

    checkCuda(cudaFree(ReductionTemp));
    checkCuda(cudaFree(Energy));

    #pragma omp parallel for num_threads(Points)
    for (int i = 0; i < Points; ++i) DestroyRMesh(RMesh + i, superCell);
    checkCuda(cudaFree(RMesh));
}

#endif