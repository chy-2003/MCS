#include <cstdio>
#include <cassert>
#include <chrono>

struct Vec3 {
    long double x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(long double _x, long double _y, long double _z) : x(_x), y(_y), z(_z) {}
    ~Vec3() {}
};

struct Vec9 {
    long double xx, xy, xz, yx, yy, yz, zx, zy, zz;
    Vec9() : xx(0), xy(0), xz(0), yx(0), yy(0), yz(0), zx(0), zy(0), zz(0) {}
    Vec9(long double _xx, long double _xy, long double _xz, 
         long double _yx, long double _yy, long double _yz,
         long double _zx, long double _zy, long double _zz) :
         xx(_xx), xy(_xy), xz(_xz), 
         yx(_yx), yy(_yy), yz(_yz),
         zx(_zx), zy(_zy), zz(_zz) {}
    ~Vec9() {}
};

struct Atom {
    long double x, y, z, Sx, Sy, Sz;  //原胞内分数坐标，S三个方向
    Atom() : x(0), y(0), z(0), Sx(0), Sy(0), Sz(0) {}
    Atom(long double  _x, long double  _y, long double  _z,
         long double _Sx, long double _Sy, long double _Sz) :
          x( _x),  y( _y),  z( _z),
         Sx(_Sx), Sy(_Sy), Sz(_Sz) {}
    ~Atom() {}
};

struct Bond {
    int Gx, Gy, Gz, Ax, Ay;   //跨原胞位移， 始末原胞原子编号
    Bond() : Gx(0), Gy(0), Gz(0), Ax(0), Ay(0) {}
    Bond(int _Gx, int _Gy, int _Gz, int _Ax, int _Ay) : 
        Gx(_Gx), Gy(_Gy), Gz(_Gz), Ax(_Ax), Ay(_Ay) {}
    ~Bond() {}
};

struct UnitCell {
    Vec3 a, b, c;         //原胞基失
    Atom* Atoms;          //磁性原子
    UnitCell() : a(), b(), c(), Atoms(NULL) {}
    ~UnitCell() {}
};

struct SuperCell {
};

int main() {
    return 0;
}