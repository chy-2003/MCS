/*
 * Information
 *
 * const
 *     ProgressCount   INFO显示进程的速率，每ProgressCount步回报一次
 * 
 * 
 * struct
 *     int NSkip;        单次模拟中舍弃的步数
 *     int NCall;        单次模拟中计入统计的步数
 *     int NTimes;       同一个温度下重复模拟的次数
 *     double TStart;    模拟起始温度
 *     double TDelta;    温度步长
 *     int TSteps;       温度采样点个数
 *     int HSteps;       几个蒙卡步改变外场，小于等于0为不改变外场
 *     Vec3 HStart;      初始外场
 *     Vec3 HDelta;      外场增量
 *     int HTimes;       外场增量步数。外场会按 HStart -> HStart+HDelta*HTimes -> HStart-HDelta*HTimes -> HStart 循环
 *     int Model;        自由度 1表示Ising 2表示XY 3表示Heisenberg
 * 
 * function
 *     MCInfo InitMCInfo(FILE *file);                             从文件中读取蒙卡信息
 *     void MonteCarloMetropolisCPU(                              CPU蒙卡主程序
 *             SuperCell *superCell,                                  传入结构信息
 *             MCInfo mcInfo,                                         传入模拟MC信息
 *             void (*returnFunc)(int, double, Vec3, int));           返回数值以供统计，四个量分别为 mesh编号，（mesh 编号
 *                                                                    为 当前温度步数 * 重复统计次数 + 当前重复统计编号）
 *                                                                    总能量，总极化/磁化，蒙卡步。returnFunc函数需要加入
 *                                                                    omp critical限制，且名称不为ProgressCnt
 * 
 * 
 */

#ifndef __MCS_MONTECARLO__
#define __MCS_MONTECARLO__

#include <cmath>
#include <chrono>
#include "Structure.cuh"
#include "Runtime.cuh"

#define ProgressCount 100000

struct MCInfo {
    int NSkip;
    int NCall;
    int NTimes;
    double TStart, TDelta;
    int TSteps;
    int HSteps;
    Vec3 HStart, HDelta;
    int HTimes;
    int Model;
    MCInfo() : NSkip(0), NCall(0), NTimes(0), 
            TStart(0), TDelta(0), TSteps(0), 
            HSteps(0), HStart(), HDelta(), HTimes(0), 
            Model(0) {}
    ~MCInfo() {}
};

MCInfo InitMCInfo(FILE *file) {
    MCInfo Ans;
    fscanf(file, "%d%d%d%lf%lf%d%d%lf%lf%lf%lf%lf%lf%d%d", 
            &Ans.NSkip, &Ans.NCall, &Ans.NTimes,
            &Ans.TStart, &Ans.TDelta, &Ans.TSteps,
            &Ans.HSteps, 
            &Ans.HStart.x, &Ans.HStart.y, &Ans.HStart.z, 
            &Ans.HDelta.x, &Ans.HDelta.y, &Ans.HDelta.z, 
            &Ans.HTimes,
            &Ans.Model);
    return Ans;
}

void MonteCarloMetropolisCPU(SuperCell *superCell, MCInfo mcInfo, void (*returnFunc)(int, rMesh*, int)) {

    fprintf(stderr, "[INFO][from MonteCarlo_MonteCarloMetropolisCPU] starting MC...\n");
    int TotalMesh = mcInfo.TSteps * mcInfo.NTimes;
    rMesh **Mesh = (rMesh**)malloc(sizeof(rMesh*) * TotalMesh);
    #pragma omp parallel for num_threads(MaxThreads)
    for (int i = 0; i < TotalMesh; ++i) {
        switch (superCell->Type) {
        case ModelM :
            Mesh[i] = InitRMesh(superCell, mcInfo.HStart, mcInfo.TStart + mcInfo.TDelta * (i / mcInfo.NTimes), 
                mcInfo.Model, GetEnergyMCPU_NoOMP, CoefficientM);
            break;
        case ModelEC42AFE :
            Mesh[i] = InitRMesh(superCell, mcInfo.HStart, mcInfo.TStart + mcInfo.TDelta * (i / mcInfo.NTimes), 
                mcInfo.Model, GetEnergyMCPU_NoOMP, CoefficientEC42AFE);
            break;
        case ModelEP22AFE :
            Mesh[i] = InitRMesh(superCell, mcInfo.HStart, mcInfo.TStart + mcInfo.TDelta * (i / mcInfo.NTimes), 
                mcInfo.Model, GetEnergyMCPU_NoOMP, CoefficientEP22AFE);
            break;
        case ModelEP21FE :
            Mesh[i] = InitRMesh(superCell, mcInfo.HStart, mcInfo.TStart + mcInfo.TDelta * (i / mcInfo.NTimes), 
                mcInfo.Model, GetEnergyMCPU_NoOMP, CoefficientEP21FE);
            break;
        default:
            break;
        }
    }
    fprintf(stderr, "[INFO][from MonteCarlo_MonteCarloMetropolisCPU] Mesh build ok.\n");

    /*
    for (int i = 0; i < superCell->a; ++i) {
        for (int j = 0; j < superCell->b; ++j) 
            printf("%5.2lf ", Mesh[0]->Dots[i * superCell->b + j].z);
        printf("\n");
    }
    int CntP[8][8];
    memset(CntP, 0, sizeof(CntP));
    */
    double Cnt = 0;
    double TotalCnt = 1.0 * TotalMesh * (mcInfo.NSkip + mcInfo.NCall);

    #pragma omp parallel for num_threads(MaxThreads)
    for (int step = 0; step < TotalMesh; ++step) {
        std::random_device RandomDevice;
        std::mt19937 Mt19937(RandomDevice());
        std::uniform_int_distribution<> UIDA(0, superCell->a - 1);
        std::uniform_int_distribution<> UIDB(0, superCell->b - 1);
        std::uniform_int_distribution<> UIDC(0, superCell->c - 1);
        std::uniform_int_distribution<> UIDN(0, superCell->unitCell.N - 1);
        std::uniform_real_distribution<> URD(0.0, 1.0);
        double u, v;
        int x, y, z, n;

        double dE, RandV, RandC;
        int Agree, id;
        Vec3 S(0, 0, 0);
        for (int i = 0; i < mcInfo.NSkip + mcInfo.NCall; ) {
            x = UIDA(Mt19937); y = UIDB(Mt19937); z = UIDC(Mt19937); n = UIDN(Mt19937);
            //++CntP[x][y];
            u = URD(Mt19937); if (mcInfo.Model == ModelHeisenberg) v = URD(Mt19937);
            id = ((x * superCell->b + y) * superCell->c + z) * superCell->unitCell.N + n;
            S = GetVec3(superCell->unitCell.Dots[n].Norm, mcInfo.Model, u, v);
            if (superCell->Type == ModelM)
                dE = GetDeltaEM_CPU(Mesh[step], superCell, x, y, z, n, S);
            else 
                dE = GetDeltaEE_CPU(Mesh[step], superCell, x, y, z, n, S);
            Agree = 0;
            if (dE <= 0) Agree = 1;
            else if (Mesh[step]->T > 1e-9) {
                RandC = std::exp(-dE / (Mesh[step]->T));
                RandV = URD(Mt19937);
                if (RandV < RandC) Agree = 1;
            }
            //printf("dE = %.8lf, T = %.2lf, agree = %d, SP = %.8lf\n", dE, Mesh[step]->T, Agree, Mesh[step]->Mag.z);
            if (Agree == 1) {
                Mesh[step]->Energy += dE;
                switch (superCell->Type) {
                    case ModelM : 
                        Mesh[step]->Mag = Add(Mesh[step]->Mag, Mul(Add(S, Rev(Mesh[step]->Dots[id])), CoefficientM(x, y, z)));
                        break;
                    case ModelEC42AFE : 
                        Mesh[step]->Mag = Add(Mesh[step]->Mag, Mul(Add(S, Rev(Mesh[step]->Dots[id])), CoefficientEC42AFE(x, y, z)));
                        break;
                    case ModelEP22AFE : 
                        Mesh[step]->Mag = Add(Mesh[step]->Mag, Mul(Add(S, Rev(Mesh[step]->Dots[id])), CoefficientEP22AFE(x, y, z)));
                        break;
                    case ModelEP21FE : 
                        Mesh[step]->Mag = Add(Mesh[step]->Mag, Mul(Add(S, Rev(Mesh[step]->Dots[id])), CoefficientEP21FE(x, y, z)));
                        break;
                    default :
                        break;
                }
                Mesh[step]->Dots[id] = S;
                if (i >= mcInfo.NSkip) returnFunc(step, Mesh[step], i);
                ++i;
                if (mcInfo.HSteps > 0 && (i % mcInfo.HSteps == 0)) {
                    int t = (i / mcInfo.HSteps - 1) / mcInfo.HTimes;
                    if ((t & 3) == 0 || (t & 3) == 3) {
                        if (superCell->Type == ModelM)
                            UpdateHCPU_NoOMP(Mesh[step], superCell, mcInfo.HDelta);
                        else 
                            UpdateECPU_NoOMP(Mesh[step], superCell, mcInfo.HDelta);
                    } else {
                        if (superCell->Type == ModelM)
                            UpdateHCPU_NoOMP(Mesh[step], superCell, Rev(mcInfo.HDelta));
                        else
                            UpdateECPU_NoOMP(Mesh[step], superCell, Rev(mcInfo.HDelta));
                    }
                }
                if (i % ProgressCount == 0) {
                    #pragma omp critical (ProgressCnt)
                    {
                        Cnt += ProgressCount;
                        fprintf(stderr, "[INFO][from MonteCarlo_MonteCarloMetropolisCPU] MC Progress %6.2lf%%.\n", 100.0 * Cnt / TotalCnt);
                    }
                }
            }
        }                    
        #pragma omp critical (ProgressCnt)
        {
            Cnt += (mcInfo.NSkip + mcInfo.NCall) % ProgressCount;
            fprintf(stderr, "[INFO][from MonteCarlo_MonteCarloMetropolisCPU] MC Progress %6.2lf%%.\n", 100.0 * Cnt / TotalCnt);
        }
    }
/*
    for (int i = 0; i < superCell->a; ++i) {
        for (int j = 0; j < superCell->b; ++j) 
            printf("%5.2lf ", Mesh[0]->Dots[i * superCell->b + j].z);
        printf("\n");
    }
    for (int i = 0; i < superCell->a; ++i) {
        for (int j = 0; j < superCell->b; ++j) 
            printf("%10d ", CntP[i][j]);
        printf("\n");
    }
*/
    #pragma omp parallel for num_threads(MaxThreads)
    for (int i = 0; i < TotalMesh; ++i) {
        if (Mesh[i] != NULL) DestroyRMesh(Mesh[i]); 
        Mesh[i] = NULL;
    }
    free(Mesh);
    fprintf(stderr, "[INFO][from MonteCarlo_MonteCarloMetropolisCPU] MC Completed.\n");
    return;
}

#endif