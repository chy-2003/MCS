#ifndef __MCS_MONTECARLO__
#define __MCS_MONTECARLO__

#include <cmath>
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

#ifdef __MCS_DEBUG__
    std::chrono::steady_clock::time_point before = std::chrono::steady_clock::now();
#endif

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

#ifdef __MCS_DEBUG__
    std::chrono::steady_clock::time_point  after = std::chrono::steady_clock::now();
    fprintf(stderr, "[DEBUG][from MonteCarlo_GetEnergyGPU] Kernel Time Cost(s) = %.8lf\n", std::chrono::duration<double>(after - before));
#endif    

    DestroyRMeshOnGPU(gMesh);
    DestroyStructureOnGPU(gStructure);
    DestroyRBondsOnGPU(gBonds);
    return;
}

void GetEnergyCPU(rMesh *tar, SuperCell *superCell) {
    double Energy = 0;
    #pragma omp parallel for num_threads(MaxThreads) reduction(+: Energy)
    for (int x = 0; x < superCell->a; ++x)
        for (int y = 0; y < superCell->b; ++y) 
            for (int z = 0; z < superCell->c; ++z) {
                int id1 = ((x * superCell->b + y) * superCell->c + z) * superCell->unitCell.N;
                for (int n = 0; n < superCell->unitCell.N; ++n) {
                    Energy += Cal933(superCell->unitCell.Dots[n].A, tar->Dots[id1 + n], tar->Dots[id1 + n]) +
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

double GetDeltaE_CPU(rMesh *Mesh, SuperCell* superCell, int X, int Y, int Z, int n, int id1, Vec3 mag) {
    double Ans = 0;
    int id2;
    int x, y, z;
    Bond* bond = superCell->unitCell.bonds;
    while (bond) {
        if (n == bond->s) {
            x = X + bond->Gx; if (x < 0) x += superCell->a; if (x >= superCell->a) x -= superCell->a;
            y = Y + bond->Gy; if (y < 0) y += superCell->b; if (y >= superCell->b) y -= superCell->b;
            z = Z + bond->Gz; if (z < 0) z += superCell->c; if (z >= superCell->c) z -= superCell->c;
            id2 = ((x * superCell->b + y) * superCell->c + z) * superCell->unitCell.N + bond->t;
            Ans += Cal393(mag, bond->A, Mesh->Dots[id2]) - Cal393(Mesh->Dots[id1], bond->A, Mesh->Dots[id2]);
        }
        if (n == bond->t) {
            x = X - bond->Gx; if (x < 0) x += superCell->a; if (x >= superCell->a) x -= superCell->a;
            y = Y - bond->Gy; if (y < 0) y += superCell->b; if (y >= superCell->b) y -= superCell->b;
            z = Z - bond->Gz; if (z < 0) z += superCell->c; if (z >= superCell->c) z -= superCell->c;
            id2 = ((x * superCell->b + y) * superCell->c + z) * superCell->unitCell.N + bond->s;
            Ans += Cal393(Mesh->Dots[id2], bond->A, mag) - Cal393(Mesh->Dots[id2], bond->A, Mesh->Dots[id1]);
        }
        bond = bond->Next;
    }
    return Ans;
}

Vec3 DoMonteCarlo_Sin_Mag_Metropolis(SuperCell *superCell, double T, int NSkip, int NCall) {
    if (T < 1e-9) T = 1e-9;
    rMesh *Mesh = InitRMesh(superCell, Vec3(), T);
    GetEnergyCPU(Mesh, superCell);

    Vec3 mag(0, 0, 0);
    std::random_device RandomDevice;
    std::mt19937 Mt19937(RandomDevice());
    std::uniform_int_distribution<> UIDA(0, superCell->a - 1);
    std::uniform_int_distribution<> UIDB(0, superCell->b - 1);
    std::uniform_int_distribution<> UIDC(0, superCell->c - 1);
    std::uniform_int_distribution<> UIDN(0, superCell->unitCell.N - 1);
    std::uniform_real_distribution<> URD(0.0, 1.0);
    double u, v;
    int x, y, z, n;
    double us, uc, vs, vc, dE, RandV, RandC;
    int Agree, id;
    Vec3 MagS(0, 0, 0);

    for (int i = 0; i < NSkip + NCall; ++i) {
        x = UIDA(Mt19937); y = UIDB(Mt19937); z = UIDC(Mt19937); n = UIDN(Mt19937);
        id = ((x * superCell->b + y) * superCell->c + z) * superCell->unitCell.N + n;
        u = URD(Mt19937); v = URD(Mt19937); 
        u *= 2.0 * Pi; v = std::acos(2.0 * v - 1);
        us = std::sin(u); uc = std::cos(u); vs = std::sin(v); vc = std::cos(v);
        mag.x = us * vs * (superCell->unitCell.Dots[n].Norm);
        mag.y = uc * vs * (superCell->unitCell.Dots[n].Norm);
        mag.z = vc * (superCell->unitCell.Dots[n].Norm);
        //if (std::fabs(superCell->unitCell.Dots[n].Norm - std::sqrt(InMul(mag, mag))) > 1e-8) { printf("!!!\n"); fflush(stdout); }
        dE = GetDeltaE_CPU(Mesh, superCell, x, y, z, n, id, mag);
        Agree = 0;
        if (dE <= 0) Agree = 1;
        else {
            RandC = std::exp(-dE / T);
            RandV = URD(Mt19937);
            if (RandV < RandC) Agree = 1;
        }
        if (Agree == 1) {
            Mesh->Energy += dE;
            Mesh->Mag = Add(Mesh->Mag, Div(Rev(Mesh->Dots[id]), Mesh->NDots));
            Mesh->Mag = Add(Mesh->Mag, Div(mag, Mesh->NDots));
            Mesh->Dots[id] = mag;
        }
        if (i >= NSkip) MagS = Add(MagS, Mesh->Mag);
    }

    DestroyRMesh(Mesh); Mesh = NULL;
    return Div(MagS, NCall);
}

#endif