#include <cstdio>
#include <cassert>
#include <chrono>
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



double *SumE2, *SumE, *SMag, *SMag2;
MCInfo mcInfo;

void calFunc(int Id, double Energy, Vec3 mag, int I) {
    #pragma omp critical (GetAns)
    {
        SMag[Id / mcInfo.NTimes] += std::sqrt(InMul(mag, mag));
        SMag2[Id / mcInfo.NTimes] += InMul(mag, mag);
        SumE2[Id / mcInfo.NTimes] += Energy * Energy;
        SumE[Id / mcInfo.NTimes] += Energy;
    }
    return;
}

int main() {
    FILE *structureInput = fopen("Input_Structure", "r");
    SuperCell *superCell = InitStructure(structureInput);
    fclose(structureInput);

    FILE *MCInput = fopen("Input_MC", "r");
    mcInfo = InitMCInfo(MCInput);
    fclose(MCInput);

    int N = superCell->a * superCell->b * superCell->c * superCell->unitCell.N;
    SMag  = (double*)calloc(mcInfo.TSteps, sizeof(double));
    SMag2 = (double*)calloc(mcInfo.TSteps, sizeof(double));
    SumE  = (double*)calloc(mcInfo.TSteps, sizeof(double));
    SumE2 = (double*)calloc(mcInfo.TSteps, sizeof(double));

    MonteCarloMetropolisCPU(superCell, mcInfo, calFunc);
    for (int i = 0; i < mcInfo.TSteps; ++i) { 
        SumE2[i] /= mcInfo.NCall * mcInfo.NTimes; SumE[i] /= mcInfo.NCall * mcInfo.NTimes; 
        SMag2[i] /= mcInfo.NCall * mcInfo.NTimes; SMag[i] /= mcInfo.NCall * mcInfo.NTimes; 
    }
    for (int i = 0; i < mcInfo.TSteps; ++i)
        printf("T = %6.2lf, M = %12.8lf, Cv = %20.8lf, Chi = %20.8lf\n", 
                mcInfo.TStart + i * mcInfo.TDelta, 
                SMag[i] / N, 
                (SumE2[i] - SumE[i] * SumE[i]) / (mcInfo.TStart + i * mcInfo.TDelta) / N, 
                (SMag2[i] - SMag[i] * SMag[i]) / (mcInfo.TStart + i * mcInfo.TDelta) / N);

    DestroySuperCell(superCell); superCell = NULL;
    free(SMag); free(SMag2); free(SumE); free(SumE2);
    fprintf(stderr, "[INFO][from MCS_main] Program successfully ended.\n");
    return 0;
}