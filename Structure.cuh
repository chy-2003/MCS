#ifndef __MCS_STRUCTURE__
#define __MCS_STRUCTURE__

#include <cstdio>
#include "CudaInfo.cuh"

struct Vec3 {                                                                                //3元向量
    double x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}
    ~Vec3() {}
};

__global__ void Add(Vec3 *a, Vec3 *b, Vec3 *ans) { ans->x = a->x + b->x; ans->y = a->y + b->y; ans->z = a->z + b->z; return; }
__global__ void Rev(Vec3 *a, Vec3 *ans) { ans->x = -a->x; ans->y = -a->y; ans->z = -a->z; return; }
__global__ void Dec(Vec3 *a, Vec3 *b, Vec3 *ans) { ans->x = a->x - b->x; ans->y = a->y - b->y; ans->z = a->z - b->z; return; }
__global__ void InMul(Vec3 *a, Vec3 *b, Vec3 *ans) { ans->x = a->x * b->x; ans->y = a->y * b->y; ans->z = a->z * b->z; return; }

struct Vec9 {                                                                                //3*3矩阵
    double xx, xy, xz, yx, yy, yz, zx, zy, zz;
    Vec9() : xx(0), xy(0), xz(0), yx(0), yy(0), yz(0), zx(0), zy(0), zz(0) {}
    Vec9(double _xx, double _yy, double _zz) : xx(_xx), xy(0), xz(0), yx(0), yy(_yy), yz(0), zx(0), zy(0), zz(_zz) {}
    Vec9(double _xx, double _xy, double _xz, 
         double _yx, double _yy, double _yz,
         double _zx, double _zy, double _zz) :
         xx(_xx), xy(_xy), xz(_xz), 
         yx(_yx), yy(_yy), yz(_yz),
         zx(_zx), zy(_zy), zz(_zz) {}
    ~Vec9() {}
};

__global__ void Add(Vec9 *a, Vec9 *b, Vec9 *ans) { 
    ans->xx = a->xx + b->xx; ans->xy = a->xy + b->xy; ans->xz = a->xz + b->xz; 
    ans->yx = a->yx + b->yx; ans->yy = a->yy + b->yy; ans->yz = a->yz + b->yz; 
    ans->zx = a->zx + b->zx; ans->zy = a->zy + b->zy; ans->zz = a->zz + b->zz;
    return;
}
__global__ void Rev(Vec9 *a, Vec9 *ans) { 
    ans->xx = -a->xx; ans->xy = -a->xy; ans->xz = -a->xz;
    ans->yx = -a->yx; ans->yy = -a->yy; ans->yz = -a->yz;
    ans->zx = -a->zx; ans->zy = -a->zy; ans->zz = -a->zz;
    return;
}
__global__ void Del(Vec9 *a, Vec9 *b, Vec9 *ans) { 
    ans->xx = a->xx - b->xx; ans->xy = a->xy - b->xy; ans->xz = a->xz - b->xz; 
    ans->yx = a->yx - b->yx; ans->yy = a->yy - b->yy; ans->yz = a->yz - b->yz; 
    ans->zx = a->zx - b->zx; ans->zy = a->zy - b->zy; ans->zz = a->zz - b->zz;
    return; 
}
__global__ void MaMul(Vec9 *a, Vec9 *b, Vec9 *ans) {                                         //3*3矩阵乘法
    ans->xx = a->xx * b->xx + a->xy * b->yx + a->xz * b->zx; ans->xy = a->xx * b->xy + a->xy * b->yy + a->xz * b->zy; ans->xz = a->xx * b->xz + a->xy * b->yz + a->xz * b->zz; 
    ans->yx = a->yx * b->xx + a->yy * b->yx + a->yz * b->zx; ans->yy = a->yx * b->xy + a->yy * b->yy + a->yz * b->zy; ans->yz = a->yx * b->xz + a->yy * b->yz + a->yz * b->zz; 
    ans->zx = a->zx * b->xx + a->zy * b->yx + a->zz * b->zx, ans->zy = a->zx * b->xy + a->zy * b->yy + a->zz * b->zy; ans->zz = a->zx * b->xz + a->zy * b->yz + a->zz * b->zz;
    return;
}
__global__ void CoMul(Vec3 *a, Vec3 *b, Vec9 *ans) {                                         //笛卡尔积
    ans->xx = a->x * b->x; ans->xy = a->x * b->y; ans->xz = a->x * b->z;
    ans->yx = a->y * b->x; ans->yy = a->y * b->y; ans->yz = a->y * b->z;
    ans->zx = a->z * b->x; ans->zy = a->z * b->y; ans->zz = a->z * b->z;
    return;
}

struct Bond {                                                                                 //二点之间的作用，前向星式存储
    int Gx, Gy, Gz, t;                                                                        //跨原胞位移， 指向原子原胞内编号
    Bond* Next;                                                                               //前向星
    Vec9* A;                                                                                  //s*A*t 关系，s为bond链首
    Bond() : Gx(0), Gy(0), Gz(0), t(0), A() {}
    ~Bond() {}                                                                                //【重要】 务必保证 DestroyBond 在析构函数之前被调用
};
void DestroyBond(Bond *self) {
    if (self->Next != NULL) DestroyBond(self->Next);                                          //这里保留了悬垂指针，因为马上会被销毁。所以请保证传入的是一个指向对象的正确指针
    checkCuda(cudaFree(self->A));
    checkCuda(cudaFree(self));
    return;
}

struct Dot {                                                                                  //原胞内有意义的点，在UnitCell中以数组存在
    Bond *bonds;                                                                              //前向星起始位置
    Vec3 *Pos;                                                                                //原胞内分数坐标
    Vec3 *a;                                                                                  //S或P等三个方向
    Vec9 *A;                                                                                  //a*A*a 各向异性
    Dot() : Pos(NULL), a(NULL), A(NULL), bonds(NULL) {}
    ~Dot() {}                                                                                 //【重要】  务必保证 DestroyDot 在析构函数前被调用
};
void DestroyDot(Dot *self) {
    if (self->bonds != NULL) DestroyBond(self->bonds);
    if (self->Pos != NULL) checkCuda(cudaFree(self->Pos));
    if (self->a != NULL) checkCuda(cudaFree(self->a));
    if (self->A != NULL) checkCuda(cudaFree(self->A));                                        //这里保留了悬垂指针，所以调用 DestroyDot 之后不应再使用该 Dot 如 _AppendBond
    //checkCuda(cudaFree(self));                                                              //保留 Dot 本身，在 DestroyUnitCell 时作为数组销毁
    return;
}
void _AppendBond(Dot *target, Bond *val) {                                                    //加入bond，前向星操作
    val->Next = target->bonds;
    target->bonds = val;
    return;
}

struct UnitCell {                                                                             //原胞，其中Dots为数组而非单个指针
    Vec3 a, b, c;                                                                             //原胞基失
    int N;                                                                                    //磁性原子/电偶极子数量
    Dot* Dots;                                                                                //磁性原子/极化
    UnitCell() : N(0), a(), b(), c(), Dots(NULL) {}
    ~UnitCell() {}                                                                            //【重要】  务必保证 DestroyUnitCell 在析构函数前被调用
};
void InitUnitCell(UnitCell *self, int N, Vec3 a, Vec3 b, Vec3 c) {
    self->N = N;
    self->a = a; self->b = b; self->c = c;
    checkCuda(cudaMallocManaged(&(self->Dots), N * sizeof(Dot)));
    for (int i = 0; i < N; ++i) {
        (self->Dots)[i].bonds = NULL;
        (self->Dots)[i].a = NULL;
        (self->Dots)[i].A = NULL;
    }
    return;
}
void SetDotPos(UnitCell *self, int s, Vec3 a) {                                              //设定点位置（分数坐标），传入参数为原胞指针、点编号、点位置
    Vec3 *Temp = NULL;
    checkCuda(cudaMallocManaged(&Temp, sizeof(Vec3)));
    *Temp = a;
    (self->Dots)[s].Pos = Temp;
    return;
}
void SetDotVal(UnitCell *self, int s, Vec3 a) {                                              //设定点上向量
    Vec3 *Temp = NULL;
    checkCuda(cudaMallocManaged(&Temp, sizeof(Vec3)));
    *Temp = a;
    (self->Dots)[s].a = Temp;
    return;
}
void SetDotAni(UnitCell *self, int s, Vec9 A) {                                              //设定点各向异性
    Vec9 *Temp = NULL;
    checkCuda(cudaMallocManaged(&Temp, sizeof(Vec9)));
    *Temp = A;
    (self->Dots)[s].A = Temp;
    return;
}
void AppendBond(UnitCell *self, int s, int t, int Gx, int Gy, int Gz, Vec9 A) {              //添加bond接口，传入原胞指针、起始点编号、终止点编号、跨晶格偏移和作用关系
    Bond *Temp = NULL;
    checkCuda(cudaMallocManaged(&Temp, sizeof(Bond)));
    Temp->Gx = Gx; Temp->Gy = Gy; Temp->Gz = Gz;
    Temp->t = t;
    checkCuda(cudaMallocManaged(&(Temp->A), sizeof(Vec9)));
    *(Temp->A) = A;
    _AppendBond((self->Dots) + s, Temp);
    return;
}
void DestroyUnitCell(UnitCell *self) {
    int N = self->N;
    for (int i = 0; i < N; ++i)
        DestroyDot((self->Dots) + i);
    checkCuda(cudaFree(self->Dots));
    checkCuda(cudaFree(self));
    return;
}

struct SuperCell {
    int a, b, c;
    UnitCell* unitCell;
    SuperCell() : a(1), b(1), c(1), unitCell(NULL) {}
    ~SuperCell() {}                                                                           //【重要】  务必保证 DestroySuperCell 在析构函数前被调用
};
SuperCell* InitSuperCell(SuperCell *self, int a, int b, int c) {
    checkCuda(cudaMallocManaged(&self, sizeof(SuperCell)));
    self->a = a; self->b = b; self->c = c;
    self->unitCell = NULL;
    checkCuda(cudaMallocManaged(&(self->unitCell), sizeof(UnitCell)));
    self->unitCell->a = Vec3();
    self->unitCell->b = Vec3();
    self->unitCell->c = Vec3();
    self->unitCell->N = 0;
    self->unitCell->Dots = NULL;
    return self;
};
void DestroySuperCell(SuperCell *self) {
    DestroyUnitCell(self->unitCell);
    checkCuda(cudaFree(self));
    return;
}

SuperCell* InitStructure(SuperCell *self, FILE *file) {                                      //从文件读取结构信息以及相互关联信息，不包括蒙卡部分
    fprintf(stderr, "[INFO] Start importing structure data.\n");
    int a, b, c;
    if (fscanf(file, "%d%d%d", &a, &b, &c) != 3) {
        fprintf(stderr, "[ERROR] Unable to get supercell scale.\n");
        return NULL;
    }
    Vec3 A, B, C;
    int N;
    if (fscanf(file, "%lf%lf%lf", &(A.x), &(A.y), &(A.z)) != 3) {
        fprintf(stderr, "[ERROR] Unable to get unitcell arg a.\n");
        return NULL;
    }
    if (fscanf(file, "%lf%lf%lf", &(B.x), &(B.y), &(B.z)) != 3) {
        fprintf(stderr, "[ERROR] Unable to get unicell arg b.\n");
        return NULL;
    }
    if (fscanf(file, "%lf%lf%lf", &(C.x), &(C.y), &(C.z)) != 3) {
        fprintf(stderr, "[ERROR] Unable to get unicell arg c.\n");
        return NULL;
    }
    if (fscanf(file, "%d", &N) != 1) {
        fprintf(stderr, "[ERROR] Unable to get number of elements in unitcell.\n");
        return NULL;
    }
    self = InitSuperCell(self, a, b, c);
    InitUnitCell(self->unitCell, N, A, B, C);
    Vec9 D;
    for (int i = 0; i < N; ++i) {
        if (fscanf(file, "%lf%lf%lf", &(A.x), &(A.y), &(A.z)) != 3) {
            fprintf(stderr, "[ERROR] Unable to get %dth position.\n", i);
            DestroySuperCell(self);
            return NULL;
        }
        SetDotPos(self->unitCell, i, A);
        if (fscanf(file, "%lf%lf%lf", &(A.x), &(A.y), &(A.z)) != 3) {
            fprintf(stderr, "[ERROR] Unable to get %dth val.\n", i);
            DestroySuperCell(self);
            return NULL;
        }
        SetDotVal(self->unitCell, i, A);
        if (fscanf(file, "%lf%lf%lf%lf%lf%lf%lf%lf%lf", &(D.xx), &(D.xy), &(D.xz), &(D.yx), &(D.yy), &(D.yz), &(D.zx), &(D.zy), &(D.zz)) != 9) {
            fprintf(stderr, "[ERROR] Unable to get %dth args.\n", i);
            DestroySuperCell(self);
            return NULL;
        }
        SetDotAni(self->unitCell, i, D);
    }
    if (fscanf(file, "%d", &N) != 1) {
        fprintf(stderr, "[ERROR] Unable to get number of bonds.\n");
        DestroySuperCell(self);
        return NULL;
    }
    int x, y;
    for (int i = 0; i < N; ++i) {
        if (fscanf(file, "%d%d", &x, &y) != 2) {
            fprintf(stderr, "[ERROR] Missing Bond info s/t.\n");
            DestroySuperCell(self);
            return NULL;
        }
        if (x < 0 || x >= self->unitCell->N || y < 0 || y >= self->unitCell->N) {
            fprintf(stderr, "[ERROR] Invalid index of bond %d.\n", i);
            DestroySuperCell(self);
            return NULL;
        }
        if (fscanf(file, "%d%d%d", &a, &b, &c) != 3) {
            fprintf(stderr, "[ERROR] Missing overlat. info of bond %d.\n", i);
            DestroySuperCell(self);
            return NULL;
        }
        if (fscanf(file, "%lf%lf%lf%lf%lf%lf%lf%lf%lf", &(D.xx), &(D.xy), &(D.xz), &(D.yx), &(D.yy), &(D.yz), &(D.zx), &(D.zy), &(D.zz)) != 9) {
            fprintf(stderr, "[ERROR] Unable to get %dth bond info.\n", i);
            DestroySuperCell(self);
            return NULL;
        }
        AppendBond(self->unitCell, x, y, a, b, c, D);
        AppendBond(self->unitCell, y, x, a, b, c, D);
    }
    fprintf(stderr, "[INFO] Structure data successfully imported.\n");
    return self;
}

#endif