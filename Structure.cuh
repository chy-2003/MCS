#ifndef __MCS_STRUCTURE__
#define __MCS_STRUCTURE__

#include "CudaInfo.cuh"
#define TypeUse long double

struct Vec3 {
    TypeUse x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(TypeUse _x, TypeUse _y, TypeUse _z) : x(_x), y(_y), z(_z) {}
    ~Vec3() {}
};

__global__ void Add(Vec3 *a, Vec3 *b, Vec3 *ans) { ans->x = a->x + b->x; ans->y = a->y + b->y; ans->z = a->z + b->z; return; }
__global__ void Rev(Vec3 *a, Vec3 *ans) { ans->x = -a->x; ans->y = -a->y; ans->z = -a->z; return; }
__global__ void Dec(Vec3 *a, Vec3 *b, Vec3 *ans) { ans->x = a->x - b->x; ans->y = a->y - b->y; ans->z = a->z - b->z; return; }
__global__ void InMul(Vec3 *a, Vec3 *b, Vec3 *ans) { ans->x = a->x * b->x; ans->y = a->y * b->y; ans->z = a->z * b->z; return; }

struct Vec9 {
    TypeUse xx, xy, xz, yx, yy, yz, zx, zy, zz;
    Vec9() : xx(0), xy(0), xz(0), yx(0), yy(0), yz(0), zx(0), zy(0), zz(0) {}
    Vec9(TypeUse _xx, TypeUse _yy, TypeUse _zz) : xx(_xx), xy(0), xz(0), yx(0), yy(_yy), yz(0), zx(0), zy(0), zz(_zz) {}
    Vec9(TypeUse _xx, TypeUse _xy, TypeUse _xz, 
         TypeUse _yx, TypeUse _yy, TypeUse _yz,
         TypeUse _zx, TypeUse _zy, TypeUse _zz) :
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
__global__ void MaMul(Vec9 *a, Vec9 *b, Vec9 *ans) {
    ans->xx = a->xx * b->xx + a->xy * b->yx + a->xz * b->zx; ans->xy = a->xx * b->xy + a->xy * b->yy + a->xz * b->zy; ans->xz = a->xx * b->xz + a->xy * b->yz + a->xz * b->zz; 
    ans->yx = a->yx * b->xx + a->yy * b->yx + a->yz * b->zx; ans->yy = a->yx * b->xy + a->yy * b->yy + a->yz * b->zy; ans->yz = a->yx * b->xz + a->yy * b->yz + a->yz * b->zz; 
    ans->zx = a->zx * b->xx + a->zy * b->yx + a->zz * b->zx, ans->zy = a->zx * b->xy + a->zy * b->yy + a->zz * b->zy; ans->zz = a->zx * b->xz + a->zy * b->yz + a->zz * b->zz;
    return;
}
__global__ void CoMul(Vec3 *a, Vec3 *b, Vec9 *ans) {
    ans->xx = a->x * b->x; ans->xy = a->x * b->y; ans->xz = a->x * b->z;
    ans->yx = a->y * b->x; ans->yy = a->y * b->y; ans->yz = a->y * b->z;
    ans->zx = a->z * b->x; ans->zy = a->z * b->y; ans->zz = a->z * b->z;
    return;
}

struct MagBond {
    int Gx, Gy, Gz, t;   //跨原胞位移， 指向原子原胞内编号
    MagBond* Next;       //前向星
    Vec9* J;             //关系
    MagBond() : Gx(0), Gy(0), Gz(0), Ax(0), Ay(0), J() {}
    ~MagBond() {}           //【重要】 务必保证 DestroyBond() 在析构函数之前被调用
};
__global__ DestroyBond(MagBond *self) {
    if (self->Next != NULL) DestroyBond(self->Next);   //这里保留了悬垂指针，因为马上会被销毁。所以请保证传入的是一个指向对象的正确指针
    checkCuda(cudaFree(J));
    checkCuda(cudaFree(self));
    return;
}

struct MagAtom {
    TypeUse x, y, z;  //原胞内分数坐标
    Vec3 s;           //S三个方向
    Vec9 sni;         //晶格关系/各项异性
    MagBond *bonds;
    Atom() : x(0), y(0), z(0), s(), ani(), bonds(NULL) {}
    ~Atom() {}
};

struct UnitCell {
    int N;                //磁性原子数量
    Vec3 a, b, c;         //原胞基失
    MagAtom* magAtoms;          //磁性原子
    UnitCell() : a(), b(), c(), magAtoms(NULL) {}
    ~UnitCell() {}
};

struct SuperCell {
    int a, b, c;
    UnitCell* unitCell;
    UnitCell() : unitCell(NULL) {}
    ~UnitCell() {}
};

#endif