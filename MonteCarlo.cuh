#ifndef __MCS_MONTECARLO__
#define __MCS_MONTECARLO__

#include <cmath>
#include <chrono>
#include <random>
#include "Structure.cuh"
#include "Runtime.cuh"

#define ModelIsing 1
#define ModelXY 2
#define ModelHeisenberg 3

void MonteCarloMetropolisCPU(SuperCell *superCell, 
                double TStart, double TDelta, int Steps, 
                int NSkip, int NCall, int Model,
                void (*returnFunc)(int, double, Vec3)) {


    fprintf(stderr, "[Info][from MonteCarlo_MonteCarloMetropolisCPU] starting MC...\n");
    rMesh **Mesh = (rMesh**)malloc(sizeof(rMesh*) * Steps);
    #pragma omp parallel for num_threads(MaxThreads)
    for (int i = 0; i < Steps; ++i)
        Mesh[i] = NULL;
    Mesh[0] = InitRMesh(superCell, Vec3(), TStart);
    #pragma omp parallel for num_threads(MaxThreads)
    for (int i = 1; i < Steps; ++i) {
        Mesh[i] = CopyRMesh(Mesh[0]);
        Mesh[i]->T = TStart + i * TDelta;
    }

    #pragma omp parallel for num_threads(MaxThreads)
    for (int step = 0; step < Steps; ++step) {
        
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
        Vec3 S(0, 0, 0);
        double T = TStart + TDelta * step;
        for (int i = 0; i < NSkip + NCall; ) {
            x = UIDA(Mt19937); y = UIDB(Mt19937); z = UIDC(Mt19937); n = UIDN(Mt19937);
            id = ((x * superCell->b + y) * superCell->c + z) * superCell->unitCell.N + n;
            if (Model == ModelIsing) {
                S = Mesh[step]->Dots[id];
                S.z = -S.z;
            }
            if (Model == ModelXY) {
                u = URD(Mt19937);
                S.x = std::sin(u) * superCell->unitCell.Dots[n].Norm;
                S.y = std::cos(u) * superCell->unitCell.Dots[n].Norm;
                S.z = 0;
            }
            if (Model == ModelHeisenberg) {
                u = URD(Mt19937); v = URD(Mt19937); 
                u *= 2.0 * Pi; v = std::acos(2.0 * v - 1);
                us = std::sin(u); uc = std::cos(u); vs = std::sin(v); vc = std::cos(v);
                S.x = us * vs * (superCell->unitCell.Dots[n].Norm);
                S.y = uc * vs * (superCell->unitCell.Dots[n].Norm);
                S.z = vc * (superCell->unitCell.Dots[n].Norm);
            }
            dE = GetDeltaE_CPU(Mesh[step], superCell, x, y, z, n, S);
            Agree = 0;
            if (dE <= 0) Agree = 1;
            else {
                RandC = std::exp(-dE / (Mesh[step]->T));
                RandV = URD(Mt19937);
                if (RandV < RandC) Agree = 1;
            }
            if (Agree == 1) {
                Mesh[step]->Energy += dE;
                Mesh[step]->Mag = Add(Mesh[step]->Mag, Rev(Mesh[step]->Dots[id]));
                Mesh[step]->Mag = Add(Mesh[step]->Mag, S);
                Mesh[step]->Dots[id] = S;
                if (i >= NSkip) returnFunc(step, Mesh[step]->Energy, Mesh[step]->Mag);
                ++i;
            } 
        }
        fprintf(stderr, "[INFO][from MonteCarlo_MonteCarloMetropolisCPU] step %d finished.\n", step);
    }

    #pragma omp parallel for num_threads(MaxThreads)
    for (int i = 0; i < Steps; ++i) {
        if (Mesh[i] != NULL) DestroyRMesh(Mesh[i]); 
        Mesh[i] = NULL;
    }
    free(Mesh);
    fprintf(stderr, "[Info][from MonteCarlo_MonteCarloMetropolisCPU] MC Completed.\n");
    return;
}

#endif