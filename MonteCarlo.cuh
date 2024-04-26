#ifndef __MCS_MONTECARLO__
#define __MCS_MONTECARLO__

#include "Structure.cuh"
#include "Runtime.cuh"

void MonteCarlo_Range(double *ans, SuperCell *superCell, double L, double R, int Points) {
    rMesh* RMesh;
    checkCuda(cudaMallocManaged(&RMesh, sizeof(rMesh) * Points));
    #pragma omp parallel for num_threads(Points)
    for (int i = 0; i < Points; ++i) BuildRMesh(RMesh + i, superCell);

    #pragma omp parallel for num_threads(Points)
    for (int i = 0; i < Points; ++i) DestroyRMesh(RMesh + i, superCell);
    checkCuda(cudaFree(RMesh));
}

#endif