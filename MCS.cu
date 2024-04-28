#include <cstdio>
#include <cassert>
#include <chrono>
#include <random>
#include <omp.h>
#include "CudaInfo.cuh"
#include "Structure.cuh"
#include "Runtime.cuh"
#include "MonteCarlo.cuh"


//compile args : nvcc MCS.cu -o MCS -Xcompiler -openmp -Xptxas -O3
//  use -arch=sm_86 for RTX 4060ti
//  nvcc MCS.cu -o MCS -Xcompiler -openmp -Xptxas -O3 -arch=sm_86
//  use -arch=sm_80 for RTX 2050
//  nvcc MCS.cu -o MCS -Xcompiler -openmp -Xptxas -O3 -arch=sm_80
// suppressed warning -diag-suppress 20011 -diag-suppress 20014
//  
// RTX 4060Ti:
//     nvcc MCS.cu -o MCS -Xcompiler -openmp -Xptxas -O3 -arch=sm_86 -diag-suppress 20011 -diag-suppress 20014
// RTX 2050
//     nvcc MCS.cu -o MCS -Xcompiler -openmp -Xptxas -O3 -arch=sm_80 -diag-suppress 20011 -diag-suppress 20014


void CheckInput(SuperCell *self) {
    fprintf(stdout, "SuperCell Scale : %d %d %d\n", self->a, self->b, self->c);
    fprintf(stdout, "UnitCell a(%6.2f, %6.2lf, %6.2lf)\n", (self->unitCell).a.x, (self->unitCell).a.y, (self->unitCell).a.z);
    fprintf(stdout, "UnitCell b(%6.2f, %6.2lf, %6.2lf)\n", (self->unitCell).b.x, (self->unitCell).b.y, (self->unitCell).b.z);
    fprintf(stdout, "UnitCell c(%6.2f, %6.2lf, %6.2lf)\n", (self->unitCell).c.x, (self->unitCell).c.y, (self->unitCell).c.z);
    fprintf(stdout, "UnitCell %d dots.\n", (self->unitCell).N);
    int N = (self->unitCell).N;
    for (int i = 0; i < N; ++i) {
        fprintf(stdout, "Dot %d :\n", i);
        Dot* dot = (self->unitCell).Dots + i;
        fprintf(stdout, "    Position (%6.2lf, %6.2lf, %6.2lf)\n", (dot->Pos).x, (dot->Pos).y, (dot->Pos).z);
        fprintf(stdout, "    Value    (%6.2lf, %6.2lf, %6.2lf)\n", (dot->a).x, (dot->a).y, (dot->a).z);
        fprintf(stdout, "    Ani      (%6.2lf, %6.2lf, %6.2lf,\n", (dot->A).xx, (dot->A).xy, (dot->A).xz);
        fprintf(stdout, "              %6.2lf, %6.2lf, %6.2lf,\n", (dot->A).yx, (dot->A).yy, (dot->A).yz);
        fprintf(stdout, "              %6.2lf, %6.2lf, %6.2lf)\n", (dot->A).zx, (dot->A).zy, (dot->A).zz);
        Bond *bonds = (self->unitCell).Dots[i].bonds;
        while (bonds != NULL) {
            fprintf(stdout, "        To %d, Overlat %d, %d, %d\n", bonds->t, bonds->Gx, bonds->Gy, bonds->Gz);
            fprintf(stdout, "        Corr (%6.2lf, %6.2lf, %6.2lf,\n", (bonds->A).xx, (bonds->A).xy, (bonds->A).xz);
            fprintf(stdout, "              %6.2lf, %6.2lf, %6.2lf,\n", (bonds->A).yx, (bonds->A).yy, (bonds->A).yz);
            fprintf(stdout, "              %6.2lf, %6.2lf, %6.2lf)\n", (bonds->A).zx, (bonds->A).zy, (bonds->A).zz);
            bonds = bonds->Next;
        }
    }
    fprintf(stdout, "Check Input End.\n");
    fflush(stdout);
    return;
}
/*
void CheckMesh(rMesh *self, SuperCell *superCell, int x, int y, int z) {
    fprintf(stdout, "CheckMesh %d, (%d, %d, %d)\n", (int)self, x, y, z);
    int n = superCell->unitCell->N;
    int id = z * (superCell->a * superCell->b) + y * (superCell->a) + x;
    fprintf(stdout, "    Dots : %d, id = %d\n", n, id);
    for (int i = 0; i < n; ++i) {
        Vec3 temp = ((self->Unit + id)->Dots)[i];
        fprintf(stdout, "    Dot %d, Val : (%.2lf, %.2lf, %.2lf)\n", i, temp.x, temp.y, temp.z);
    }
    fprintf(stdout, "CheckMesh End.\n");
    fflush(stdout);
    return;
}



__global__ void Gen(double *tar) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    *(tar + x) = 1.0;
    return;
}
*/


int main() {

/*
    fprintf(stdout, "Check OpenMP.\n");
    omp_set_num_threads(8);
    #pragma omp parallel 
    {
        fprintf(stdout, "thread = %d/%d\n", omp_get_thread_num(), omp_get_max_threads());
    }
    fprintf(stdout, "Check OpenMP End.\n\n\n\n\n\n\n");
    fflush(stdout);
*/



    SuperCell *superCell = NULL;
    superCell = (SuperCell*) malloc(sizeof(SuperCell));
    FILE *structureInput = fopen("Input_Structure", "r");
    superCell = InitStructure(superCell, structureInput);
    fclose(structureInput);
    if (superCell == NULL) {
        fprintf(stderr, "[ERROR] Failed loading structure. Exit.\n");
        return 0;
    }

    CheckInput(superCell);
/*
    rMesh *Mesh = NULL;
    Mesh = BuildRMesh_PSelf(Mesh, superCell);
    CheckMesh(Mesh, superCell, 3, 3, 0);
    Mesh = DestroyRMesh_PSelf(Mesh, superCell);

    //reduct sum
    int N = 1 << 22;
    double *Val = NULL, *Tmp = NULL;
    checkCuda(cudaMallocManaged(&Val, sizeof(double) * (N + (CUBlockSize << 1))));
    checkCuda(cudaMallocManaged(&Tmp, sizeof(double) * ((N + (CUBlockSize << 1) - 1) / (CUBlockSize << 1))));
    size_t threadsPerBlock = 256;
    size_t numberOfThreads = N / threadsPerBlock;
    Gen<<<numberOfThreads, threadsPerBlock>>>(Val);
    cudaDeviceSynchronize();
    printf("Gen Finished.\n"); fflush(stdout);

    std::chrono::steady_clock::time_point TimeOld = std::chrono::steady_clock::now();
    double Ans = ReductionSum(Val, N, Tmp);
    std::chrono::steady_clock::time_point TimeNew = std::chrono::steady_clock::now();
    printf("GPU Time Cost(s) : %.6lf, Sum = %.2lf\n", 
        std::chrono::duration<double>(TimeNew - TimeOld).count(),
        Ans);
    fflush(stdout);
    checkCuda(cudaFree(Val));
    checkCuda(cudaFree(Tmp));
*/
/*
    double L = 30, R = 40;
    int Points = 11;
    double *ans = NULL;
    checkCuda(cudaMallocManaged(&ans, sizeof(double) * Points));
    MonteCarlo_Range(ans, superCell, L, R, Points, 2000, 8000);
    for (int i = 0; i < Points; ++i) 
        fprintf(stdout, "%.2lf, %.2lf\n", (R - L) / (Points - 1) * i + L, ans[i]);
    checkCuda(cudaFree(ans));
*/
    DestroySuperCell(superCell);  //Destroy 里销毁了superCell！
    fprintf(stderr, "[INFO] Program successfully ended.\n");
    return 0;
}