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
#include <algorithm>
#include "Structure.h"
#include "Runtime.h"

#define ProgressCount 100000

struct MCInfo {
	int NSkip;
	int NCall;
	int NTimes;
	double TStart, TDelta;
	int TSteps;
	int PTCnt, PTInterval;
	double PTDT;
	int HSteps;
	Vec3 HStart, HDelta;
	int HTimes;
	int Model;
	int Alert;
	MCInfo() : NSkip(0), NCall(0), NTimes(0), 
			TStart(0), TDelta(0), TSteps(0), 
			PTCnt(0), PTDT(0), PTInterval(0), 
			HSteps(0), HStart(), HDelta(), HTimes(0), 
			Model(0), Alert(0) {}
	~MCInfo() {}
};

MCInfo InitMCInfo(FILE *file) {
	MCInfo Ans;
	fscanf(file, "%d%d%d", &Ans.NSkip, &Ans.NCall, &Ans.NTimes);
	fscanf(file, "%lf%lf%d", &Ans.TStart, &Ans.TDelta, &Ans.TSteps);
	fscanf(file, "%d%lf%d", &Ans.PTCnt, &Ans.PTDT, &Ans.PTInterval);
	fscanf(file, "%d", &Ans.HSteps);
	fscanf(file, "%lf%lf%lf", &Ans.HStart.x, &Ans.HStart.y, &Ans.HStart.z);
	fscanf(file, "%lf%lf%lf", &Ans.HDelta.x, &Ans.HDelta.y, &Ans.HDelta.z);
	fscanf(file, "%d", &Ans.HTimes);
	fscanf(file, "%d", &Ans.Model);
	fscanf(file, "%d", &Ans.Alert);
	return Ans;
}

void MonteCarloMetropolisCPU(SuperCell *superCell, MCInfo mcInfo, void (*returnFunc)(int, rMesh*, int)) {
	fprintf(stderr, "[INFO][from MonteCarlo_MonteCarloMetropolisCPU] starting MC...\n");
	int TotalMesh = mcInfo.TSteps * mcInfo.NTimes;
	rMesh **Mesh = (rMesh**)malloc(sizeof(rMesh*) * TotalMesh * mcInfo.PTCnt);
	#pragma omp parallel for num_threads(MaxThreads)
	for (int i = 0; i < TotalMesh; ++i) {
		for (int j = 0; j < mcInfo.PTCnt; ++j) {
			switch (superCell->Type) {
			case ModelM :
				Mesh[i * mcInfo.PTCnt + j] = InitRMesh(superCell, mcInfo.HStart, 
					mcInfo.TStart + mcInfo.TDelta * (i / mcInfo.NTimes) + j * mcInfo.PTDT, 
					mcInfo.Model, GetEnergyMCPU_NoOMP, CoefficientM);
				break;
			case ModelEC42AFE :
				Mesh[i * mcInfo.PTCnt + j] = InitRMesh(superCell, mcInfo.HStart, 
					mcInfo.TStart + mcInfo.TDelta * (i / mcInfo.NTimes) + j * mcInfo.PTDT, 
					mcInfo.Model, GetEnergyMCPU_NoOMP, CoefficientEC42AFE);
				break;
			case ModelEP22AFE :
				Mesh[i * mcInfo.PTCnt + j] = InitRMesh(superCell, mcInfo.HStart, 
					mcInfo.TStart + mcInfo.TDelta * (i / mcInfo.NTimes) + j * mcInfo.PTDT, 
					mcInfo.Model, GetEnergyMCPU_NoOMP, CoefficientEP22AFE);
				break;
			case ModelEP21FE :
				Mesh[i * mcInfo.PTCnt + j] = InitRMesh(superCell, mcInfo.HStart, 
					mcInfo.TStart + mcInfo.TDelta * (i / mcInfo.NTimes) + j * mcInfo.PTDT, 
					mcInfo.Model, GetEnergyMCPU_NoOMP, CoefficientEP21FE);
				break;
			default:
				break;
			}
		}
	}
	fprintf(stderr, "[INFO][from MonteCarlo_MonteCarloMetropolisCPU] Mesh build ok.\n");

	double ProgressCnt = 0;
	double TotalCnt = 1.0 * TotalMesh * (mcInfo.NSkip + mcInfo.NCall);

	for (int j = 0; j < superCell->b; ++j) { 
		for (int i = 0; i < superCell->a; ++i)
			printf("%c", (Mesh[0]->Dots[i * superCell->b + j].z > 0) ? 'O' : '.');
		printf("\n");
	}
	printf("%12.6lf, Energy = %12.6lf\n", Mesh[0]->Mag.z, Mesh[0]->Energy);

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
		int Agree, id, Step;
		Vec3 S(0, 0, 0);
		for (int i = 0; i < mcInfo.NSkip + mcInfo.NCall; ++i) {
			for (int j = 0; j < mcInfo.PTCnt; ++j) {
				Step = step * mcInfo.PTCnt + j;
				Agree = 0;
				for (int alert = 0; alert < mcInfo.Alert && Agree == 0; ++alert) {
					x = UIDA(Mt19937); y = UIDB(Mt19937); z = UIDC(Mt19937); n = UIDN(Mt19937);
					id = ((x * superCell->b + y) * superCell->c + z) * superCell->unitCell.N + n;
					if (mcInfo.Model != ModelIsing) u = URD(Mt19937); 
					if (mcInfo.Model == ModelHeisenberg) v = URD(Mt19937);
					if (mcInfo.Model == ModelIsing) S = Rev(Mesh[Step]->Dots[id]);
					else S = GetVec3(superCell->unitCell.Dots[n].Norm, mcInfo.Model, u, v);
					if (superCell->Type == ModelM) dE = GetDeltaEM_CPU(Mesh[Step], superCell, x, y, z, n, S);
					else dE = GetDeltaEE_CPU(Mesh[Step], superCell, x, y, z, n, S);
					if (dE <= 0) Agree = 1;
					else if (Mesh[Step]->T > 1e-9) {
						RandC = std::exp(-dE / (Mesh[Step]->T)); RandV = URD(Mt19937);
						if (RandV < RandC) Agree = 1;
					}
				}
				if (Agree) {
					Mesh[Step]->Energy += dE;
					switch (superCell->Type) {
						case ModelM : 
							Mesh[Step]->Mag = Add(Mesh[Step]->Mag, Mul(Add(S, Rev(Mesh[Step]->Dots[id])), CoefficientM(x, y, z)));
							break;
						case ModelEC42AFE : 
							Mesh[Step]->Mag = Add(Mesh[Step]->Mag, Mul(Add(S, Rev(Mesh[Step]->Dots[id])), CoefficientEC42AFE(x, y, z)));
							break;
						case ModelEP22AFE : 
							Mesh[Step]->Mag = Add(Mesh[Step]->Mag, Mul(Add(S, Rev(Mesh[Step]->Dots[id])), CoefficientEP22AFE(x, y, z)));
							break;
						case ModelEP21FE : 
							Mesh[Step]->Mag = Add(Mesh[Step]->Mag, Mul(Add(S, Rev(Mesh[Step]->Dots[id])), CoefficientEP21FE(x, y, z)));
							break;
						default :
							break;
					}
					Mesh[Step]->Dots[id] = S;
				}
				if (mcInfo.HSteps > 0 && ((i + 1) % mcInfo.HSteps == 0)) {
					int t = (i / mcInfo.HSteps - 1) / mcInfo.HTimes;
					if ((t & 3) == 0 || (t & 3) == 3) {
						if (superCell->Type == ModelM) UpdateHCPU_NoOMP(Mesh[Step], superCell, mcInfo.HDelta);
						else UpdateECPU_NoOMP(Mesh[Step], superCell, mcInfo.HDelta);
					} else {
						if (superCell->Type == ModelM) UpdateHCPU_NoOMP(Mesh[Step], superCell, Rev(mcInfo.HDelta));
						else UpdateECPU_NoOMP(Mesh[Step], superCell, Rev(mcInfo.HDelta));
					}
				}
			}
			if ((i + 1) % mcInfo.PTInterval == 0) {
				for (int j = mcInfo.PTCnt - 1; j > 0; --j) {
					Step = step * mcInfo.PTCnt + j;
					RandC = std::min(1.0, std::exp((Mesh[Step]->Energy - Mesh[Step - 1]->Energy) / 
							(1.0 / Mesh[Step]->T - 1.0 / Mesh[Step - 1]->T)));
					RandV = URD(Mt19937);
					if (RandV < RandC) {
						std::swap(Mesh[Step], Mesh[Step - 1]);
						std::swap(Mesh[Step]->T, Mesh[Step - 1]->T);
					}
				}
			}
			if (i >= mcInfo.NSkip) returnFunc(step, Mesh[step * mcInfo.PTCnt], i);
			if ((i + 1) % ProgressCount == 0) {
				#pragma omp critical (GetMCProgress)
				{
					ProgressCnt += ProgressCount;
					fprintf(stderr, "[INFO][from MonteCarlo_MonteCarloMetropolisCPU]MC Progress %6.2lf%%\n", 100.0 * ProgressCnt / TotalCnt);
				}
			}
		}
	}

	for (int j = 0; j < superCell->b; ++j) { 
		for (int i = 0; i < superCell->a; ++i)
			printf("%c", (Mesh[0]->Dots[i * superCell->b + j].z > 0) ? 'O' : '.');
		printf("\n");
	}
	printf("%12.6lf, Energy = %12.6lf\n", Mesh[0]->Mag.z, Mesh[0]->Energy);

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