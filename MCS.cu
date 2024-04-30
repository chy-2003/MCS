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


int main() {
    FILE *structureInput = fopen("Input_Structure", "r");
    SuperCell *superCell = InitStructure(structureInput);
    fclose(structureInput);
    if (superCell == NULL) {
        fprintf(stderr, "[ERROR] Failed loading structure. Exit.\n");
        assert(superCell == NULL);
    }
    
    double L = 0.0, R = 100.0;
    int Interval = 100;
    for (int i = 0; i <= Interval; ++i) {
        double T = (R - L) / Interval * i + L;
        Vec3 Mag = DoMonteCarlo_Sin_Mag_Metropolis(superCell, T, 80000, 640000);
        fprintf(stdout, "%7.2lf (%20.8lf, %20.8lf, %20.8lf), norm = %20.8lf\n", T, Mag.x, Mag.y, Mag.z, std::sqrt(InMul(Mag, Mag)));
    }

    DestroySuperCell(superCell); superCell = NULL;
    fprintf(stderr, "[INFO][from MCS_main] Program successfully ended.\n");
    return 0;
}