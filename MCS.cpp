#include <cstdio>
#include <cassert>
#include <chrono>
#include <omp.h>
//#include "CudaInfo.cuh"
#include "Structure.h"
#include "Runtime.h"
#include "MonteCarlo.h"


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

//1meV = 11.604609K

//g++ MCS.cpp -o MCS -fopenmp -O3


double *SumE2, *SumE, *SMag2, *SMagA;
Vec3 *SMag;
int NH, NS;
MCInfo mcInfo;
SuperCell *superCell = NULL;

void calFunc(int Id, rMesh *mesh, int I) {
    int N = 0;
    if (mcInfo.HSteps > 0) N = (I / mcInfo.HSteps) % NH; 
    #pragma omp critical (GetAns)
    {
        SMag [Id / mcInfo.NTimes * NH + N]  = Add(SMag[Id / mcInfo.NTimes * NH + N], mesh->Mag);
        SMag2[Id / mcInfo.NTimes * NH + N] += InMul(mesh->Mag, mesh->Mag);
        SMagA[Id / mcInfo.NTimes * NH + N] += std::sqrt(InMul(mesh->Mag, mesh->Mag));
        SumE2[Id / mcInfo.NTimes * NH + N] += mesh->Energy * mesh->Energy;
        SumE [Id / mcInfo.NTimes * NH + N] += mesh->Energy;
    }
    return;
}

int main() {
    FILE *structureInput = fopen("Input_Structure", "r");
    superCell = InitStructure(structureInput);
    fclose(structureInput);

    FILE *MCInput = fopen("Input_MC", "r");
    mcInfo = InitMCInfo(MCInput);
    fclose(MCInput);
    if (mcInfo.HSteps <= 0) NH = 1;
    else NH = 4 * mcInfo.HTimes;
    if (mcInfo.HSteps > 0) {
        if (mcInfo.NCall % (NH * mcInfo.HSteps) != 0) 
            mcInfo.NCall += (NH * mcInfo.HSteps) - mcInfo.NCall % (NH * mcInfo.HSteps);
        if (mcInfo.NSkip % (NH * mcInfo.HSteps) != 0) 
            mcInfo.NSkip += (NH * mcInfo.HSteps) - mcInfo.NSkip % (NH * mcInfo.HSteps);
    }
    NS = mcInfo.NCall * mcInfo.NTimes / NH;
    NS = 1LL * NS * (mcInfo.PTInterval - mcInfo.PTSkip) / mcInfo.PTInterval;
    fprintf(stderr, "[INFO][from MCS_main] NSkip = %d, NCall = %d\n", mcInfo.NSkip, mcInfo.NCall);

    int N = superCell->a * superCell->b * superCell->c * superCell->unitCell.N;
    SMag  = (  Vec3*)calloc(mcInfo.TSteps * NH, sizeof(  Vec3));
    SMag2 = (double*)calloc(mcInfo.TSteps * NH, sizeof(double));
    SMagA = (double*)calloc(mcInfo.TSteps * NH, sizeof(double));
    SumE  = (double*)calloc(mcInfo.TSteps * NH, sizeof(double));
    SumE2 = (double*)calloc(mcInfo.TSteps * NH, sizeof(double));

    MonteCarloMetropolisCPU(superCell, mcInfo, calFunc);
    for (int i = 0; i < mcInfo.TSteps * NH; ++i) { 
        SumE [i] /= NS; 
        SumE2[i] /= NS; 
        SMag [i]  = Div(SMag[i], NS); 
        SMag2[i] /= NS;
        SMagA[i] /= NS;
    }
    if (mcInfo.HSteps <= 0) {
        if (superCell->Type == ModelM) {
            printf("T, M, Cv, Chi, aM, \n");
            for (int i = 0; i < mcInfo.TSteps; ++i) {
                printf("%6.2lf, %12.8lf, %20.8lf, %20.8lf, %20.8lf\n", 
                    mcInfo.TStart + i * mcInfo.TDelta, 
                    SMagA[i] / N, 
                    (SumE2[i] - SumE[i] * SumE[i]) / (mcInfo.TStart + i * mcInfo.TDelta) / N, 
                    (SMag2[i] - SMagA[i] * SMagA[i]) / (mcInfo.TStart + i * mcInfo.TDelta) / N,
                    SMag[i].z / N);
            }
        } else {
            switch (superCell->Type) {
            case ModelEC42AFE:
                printf("T, rP, P_C42AFE, \n");
                break;
            case ModelEP22AFE:
                printf("T, rP, P_P22AFE, \n");
                break;
            case ModelEP21FE:
                printf("T, rP, P_P21FE, \n");
                break;
            
            default:
                break;
            }
            for (int i = 0; i < mcInfo.TSteps; ++i) {
                printf("%6.2lf, %12.8lf, %12.8lf, \n", 
                    mcInfo.TStart + i * mcInfo.TDelta, 
                    SMag[i].z / N,
                    SMagA[i] / N);
            }
        }
    } else {
        for (int i = 0; i < mcInfo.TSteps; ++i) {
            printf("T = %6.2lf\n", mcInfo.TStart + i * mcInfo.TDelta);
            if (superCell->Type == ModelM) printf("Hz, Bz, M, \n");
            else printf("Ez, Pz, P, \n");
            Vec3 H = mcInfo.HStart;
            for (int j = 0; j < NH; ++j) {
                printf("%20.8lf, %20.8lf, %20.8lf\n", H.z, 
                        SMag[i * NH + j].z / N, 
                        SMagA[i * NH + j] / N);
                int t = (j / mcInfo.HTimes);
                if ((t & 3) == 0 || (t & 3) == 3) H = Add(H, mcInfo.HDelta);
                else H = Add(H, Rev(mcInfo.HDelta));
            }
        }
    }

    DestroySuperCell(superCell); superCell = NULL;
    free(SMag); free(SMag2); free(SumE); free(SumE2); free(SMagA);
    fprintf(stderr, "[INFO][from MCS_main] Program successfully ended.\n");
    return 0;
}