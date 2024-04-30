#ifndef __MCS_STRUCTURE__
#define __MCS_STRUCTURE__

#include <cstdio>
#include "CudaInfo.cuh"

/*
 * 这一部分提供了如下内容：
 *     1：向量 Vec3 和 Vec9 以及响应计算
 *     2：【重要】CPU内的数据存储
 * 
 */

struct Vec3 {                                                                                //3元向量
    double x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}
    ~Vec3() {}
};

__host__ __device__ Vec3 Add(const Vec3 &a, const Vec3 &b) { return Vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
__host__ __device__ Vec3 Rev(const Vec3 &a) { return Vec3(-a.x, -a.y, -a.z); }
__host__ __device__ Vec3 Dec(const Vec3 &a, const Vec3 &b) { return Vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
__host__ __device__ Vec3 CoMul(const Vec3 &a, const Vec3 &b) { return Vec3(a.x * b.x, a.y * b.y, a.z * b.z); }
__host__ __device__ double InMul(const Vec3 &a, const Vec3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

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

__host__ __device__ Vec9 Add(const Vec9 &a, const Vec9 &b) { 
    return Vec9(a.xx + b.xx, a.xy + b.xy, a.xz + b.xz,  
                a.yx + b.yx, a.yy + b.yy, a.yz + b.yz, 
                a.zx + b.zx, a.zy + b.zy, a.zz + b.zz);
}
__host__ __device__ Vec9 Rev(const Vec9 &a) { 
    return Vec9(-a.xx, -a.xy, -a.xz, -a.yx, -a.yy, -a.yz, -a.zx, -a.zy, -a.zz);
}
__host__ __device__ Vec9 Del(const Vec9 &a, const Vec9 &b) { 
    return Vec9(a.xx - b.xx, a.xy - b.xy, a.xz - b.xz, 
                a.yx - b.yx, a.yy - b.yy, a.yz - b.yz, 
                a.zx - b.zx, a.zy - b.zy, a.zz - b.zz); 
}
__host__ __device__ Vec9 MaMul(const Vec9 &a, const Vec9 &b) {                                         //3*3矩阵乘法
    return Vec9(a.xx * b.xx + a.xy * b.yx + a.xz * b.zx, a.xx * b.xy + a.xy * b.yy + a.xz * b.zy, a.xx * b.xz + a.xy * b.yz + a.xz * b.zz, 
                a.yx * b.xx + a.yy * b.yx + a.yz * b.zx, a.yx * b.xy + a.yy * b.yy + a.yz * b.zy, a.yx * b.xz + a.yy * b.yz + a.yz * b.zz, 
                a.zx * b.xx + a.zy * b.yx + a.zz * b.zx, a.zx * b.xy + a.zy * b.yy + a.zz * b.zy, a.zx * b.xz + a.zy * b.yz + a.zz * b.zz);
}
__host__ __device__ Vec9 DeMul(const Vec3 &a, const Vec3 &b) {                                         //笛卡尔积
    return Vec9(a.x * b.x, a.x * b.y, a.x * b.z, a.y * b.x, a.y * b.y, a.y * b.z, a.z * b.x, a.z * b.y, a.z * b.z);
}
__host__ __device__ double Cal393(const Vec3 &s, const Vec9 &A, const Vec3 &t) {                                       //s^T \cdot A \cdot t
    return (s.x * A.xx + s.y * A.yx + s.z * A.zx) * t.x + 
           (s.x * A.xy + s.y * A.yy + s.z * A.zy) * t.y +
           (s.x * A.xz + s.y * A.yz + s.z * A.zz) * t.z;
}
__host__ __device__ double Cal933(const Vec9 &A, const Vec3 &s, const Vec3 &t) {                                        //A(st)
    return A.xx * s.x * t.x + A.xy * s.x * t.y + A.xz * s.x * t.z + 
           A.yx * s.y * t.x + A.yy * s.y * t.y + A.yz * s.y * t.z + 
           A.zx * s.z * t.x + A.zy * s.z * t.y + A.zz * s.z * t.z;
}

struct Bond {                                                                                 //二点之间的作用，前向星式存储
    int Gx, Gy, Gz, s, t;                                                                        //跨原胞位移， 指向原子原胞内编号
    Bond* Next;                                                                               //前向星
    Vec9 A;                                                                                  //A(st) 关系，s为bond链首
    Bond() : Gx(0), Gy(0), Gz(0), s(0), t(0), A() {}
    ~Bond() {}                                                                                //【重要】 务必保证 DestroyBond 在析构函数之前被调用
};
void DestroyBond(Bond *self) {
    if (self->Next != NULL) DestroyBond(self->Next);                                          //这里保留了悬垂指针，因为马上会被销毁。所以请保证传入的是一个指向对象的正确指针
    free(self);
    return;
}

struct Dot {                                                                                  //原胞内有意义的点，在UnitCell中以数组存在
    Vec3 Pos;                                                                                //原胞内分数坐标
    Vec3 a;                                                                                  //S或P等三个方向
    Vec9 A;                                                                                  //A(aa) 各向异性
    Dot() : Pos(), a(), A() {}
    ~Dot() {}                                                                                 //【重要】  务必保证 DestroyDot 在析构函数前被调用
};

struct UnitCell {                                                                             //原胞
    int N;                                                                                    //磁性原子/电偶极子数量
    Vec3 a, b, c;                                                                             //原胞基失
    Dot* Dots;                                                                                //磁性原子/极化 【array】
    int BondsCount;                                                                           //bond数量
    Bond *bonds;                                                                              //前向星起始位置
    UnitCell() : N(0), a(), b(), c(), Dots(NULL), BondsCount(0), bonds(NULL) {}
    ~UnitCell() {}                                                                            //【重要】  务必保证 DestroyUnitCell 在析构函数前被调用
};
void InitUnitCell(UnitCell *self, int N, Vec3 a, Vec3 b, Vec3 c) {
    self->N = N;
    self->a = a; self->b = b; self->c = c;
    self->BondsCount = 0;
    self->Dots = (Dot*) calloc(N, sizeof(Dot));
    return;
}
void SetDotPos(UnitCell *self, int s, Vec3 a) {                                              //设定点位置（分数坐标），传入参数为原胞指针、点编号、点位置
    (self->Dots)[s].Pos = a; return;
}
void SetDotVal(UnitCell *self, int s, Vec3 a) {                                              //设定点上向量
    (self->Dots)[s].a = a; return;
}
void SetDotAni(UnitCell *self, int s, Vec9 A) {                                              //设定点各向异性
    (self->Dots)[s].A = A; return;
}
void _AppendBond(UnitCell *target, Bond *val) {                                                    //加入bond，前向星操作
    val->Next = target->bonds;
    target->bonds = val;
    return;
}
void AppendBond(UnitCell *self, int s, int t, int Gx, int Gy, int Gz, Vec9 A) {              //添加bond接口，传入原胞指针、起始点编号、终止点编号、跨晶格偏移和作用关系
    Bond *Temp = NULL;
    Temp = (Bond*) malloc(sizeof(Bond));
    Temp->A = A; 
    Temp->Gx = Gx; Temp->Gy = Gy; Temp->Gz = Gz; 
    Temp->s = s; Temp->t = t; 
    Temp->Next = NULL;
    _AppendBond(self, Temp);
    return;
}
void DestroyUnitCell(UnitCell *self) {
    if (self->bonds != NULL) DestroyBond(self->bonds);
    free(self->Dots);
    return;
}

struct SuperCell {
    int a, b, c;
    UnitCell unitCell;
    SuperCell() : a(1), b(1), c(1), unitCell() {}
    ~SuperCell() {}                                                                           //【重要】  务必保证 DestroySuperCell 在析构函数前被调用
};
SuperCell* InitSuperCell(int a, int b, int c) {
    SuperCell *self = NULL;
    self = (SuperCell*)malloc(sizeof(SuperCell));
    self->a = a; self->b = b; self->c = c;
    memset(&(self->unitCell), 0, sizeof(UnitCell));
    return self;
};
void DestroySuperCell(SuperCell *self) {                                                     //【重要】这里free了自己！
    DestroyUnitCell(&(self->unitCell)); 
    free(self); 
    return; 
}

SuperCell* InitStructure(FILE *file) {                                      //从文件读取结构信息以及相互关联信息，不包括蒙卡部分
    fprintf(stderr, "[INFO][from Structure_InitStructure] Start importing structure data.\n");
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
    SuperCell* self = InitSuperCell(a, b, c);
    InitUnitCell(&(self->unitCell), N, A, B, C);
    Vec9 D;
    for (int i = 0; i < N; ++i) {
        if (fscanf(file, "%lf%lf%lf", &(A.x), &(A.y), &(A.z)) != 3) {
            fprintf(stderr, "[ERROR] Unable to get %dth position.\n", i);
            DestroySuperCell(self);
            return NULL;
        }
        SetDotPos(&(self->unitCell), i, A);
        if (fscanf(file, "%lf%lf%lf", &(A.x), &(A.y), &(A.z)) != 3) {
            fprintf(stderr, "[ERROR] Unable to get %dth val.\n", i);
            DestroySuperCell(self);
            return NULL;
        }
        SetDotVal(&(self->unitCell), i, A);
        if (fscanf(file, "%lf%lf%lf%lf%lf%lf%lf%lf%lf", &(D.xx), &(D.xy), &(D.xz), &(D.yx), &(D.yy), &(D.yz), &(D.zx), &(D.zy), &(D.zz)) != 9) {
            fprintf(stderr, "[ERROR] Unable to get %dth args.\n", i);
            DestroySuperCell(self);
            return NULL;
        }
        SetDotAni(&(self->unitCell), i, D);
    }
    if (fscanf(file, "%d", &N) != 1) {
        fprintf(stderr, "[ERROR] Unable to get number of bonds.\n");
        DestroySuperCell(self);
        return NULL;
    }
    int x, y;
    self->unitCell.BondsCount = N;
    for (int i = 0; i < N; ++i) {
        if (fscanf(file, "%d%d", &x, &y) != 2) {
            fprintf(stderr, "[ERROR] Missing Bond info s/t.\n");
            DestroySuperCell(self);
            return NULL;
        }
        if (x < 0 || x >= (self->unitCell).N || y < 0 || y >= (self->unitCell).N) {
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
        AppendBond(&(self->unitCell), x, y, a, b, c, D);
    }
    fprintf(stderr, "[INFO][from Structure_InitStructure] Structure data successfully imported.\n");
    return self;
}

#endif