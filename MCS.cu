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

double SMag[10010], SumE2[10010], SumE[10010], SMag2[10010];
int Cnt = 0;
int NSkip = 800000;
int NCall = 6400000;
int Steps = 21;
double TStart = 30.0;
double TDelta = 1.0;

void calFunc(int Id, double Energy, Vec3 mag) {
    SMag[Id] += std::sqrt(InMul(mag, mag));
    SMag2[Id] += InMul(mag, mag);
    SumE2[Id] += Energy * Energy;
    SumE[Id] += Energy;
    ++Cnt;
    if (Cnt % 320000 == 0) {
        fprintf(stderr, "[INFO][from MCS_calFunc] progress : %6.2lf%%\n", 100.0 * Cnt / (Steps * NCall)); fflush(stdout);
    }
    return;
}

/*
void CheckInput(SuperCell *superCell) {
    printf("%d %d %d\n", superCell->a, superCell->b, superCell->c);
    printf("Dots %d\n", superCell->unitCell.N);
    for (int i = 0; i < superCell->unitCell.N; ++i) {
        printf("    %d, spin = %.2lf\n", i, superCell->unitCell.Dots[i].Norm);
        printf("        (%5.2lf, %5.2lf, %5.2lf, \n", superCell->unitCell.Dots[i].A.xx, superCell->unitCell.Dots[i].A.xy, superCell->unitCell.Dots[i].A.xz);
        printf("         %5.2lf, %5.2lf, %5.2lf, \n", superCell->unitCell.Dots[i].A.yx, superCell->unitCell.Dots[i].A.yy, superCell->unitCell.Dots[i].A.yz);
        printf("         %5.2lf, %5.2lf, %5.2lf) \n", superCell->unitCell.Dots[i].A.zx, superCell->unitCell.Dots[i].A.zy, superCell->unitCell.Dots[i].A.zz);
    }
    printf("Bonds %d\n", superCell->unitCell.NBonds);
    Bond *bonds = superCell->unitCell.bonds;
    while (bonds != NULL) {
        printf("    %d %d, %d %d %d\n", bonds->s, bonds->t, bonds->Gx, bonds->Gy, bonds->Gz);
        printf("        (%5.2lf, %5.2lf, %5.2lf, \n", bonds->A.xx, bonds->A.xy, bonds->A.xz);
        printf("         %5.2lf, %5.2lf, %5.2lf, \n", bonds->A.yx, bonds->A.yy, bonds->A.yz);
        printf("         %5.2lf, %5.2lf, %5.2lf) \n", bonds->A.zx, bonds->A.zy, bonds->A.zz);
        bonds = bonds->Next;
    }
    return;
}
*/

int main() {
    FILE *structureInput = fopen("Input_Structure", "r");
    SuperCell *superCell = InitStructure(structureInput);
    fclose(structureInput);
    //CheckInput(superCell);

    int N = superCell->a * superCell->b * superCell->c * superCell->unitCell.N;
    memset(SMag, 0, sizeof(SMag));
    memset(SMag2, 0, sizeof(SMag2));
    memset(SumE, 0, sizeof(SumE));
    memset(SumE2, 0, sizeof(SumE2));

    MonteCarloMetropolisCPU(superCell, TStart, TDelta, Steps, NSkip, NCall, ModelHeisenberg, calFunc);
    for (int i = 0; i < Steps; ++i) { SumE2[i] /= NCall; SumE[i] /= NCall; SMag[i] /= NCall; SMag2[i] /= NCall; }
    for (int i = 0; i < Steps; ++i)
        printf("T = %6.2lf, M = %12.8lf, Cv = %20.8lf, Chi = %20.8lf\n", 
                TStart + i * TDelta, 
                SMag[i] / N, 
                (SumE2[i] - SumE[i] * SumE[i]) / (TStart + i * TDelta) / N, 
                (SMag2[i] - SMag[i] * SMag[i]) / (TStart + i * TDelta) / N);

    DestroySuperCell(superCell); superCell = NULL;
    fprintf(stderr, "[INFO][from MCS_main] Program successfully ended.\n");
    return 0;
}