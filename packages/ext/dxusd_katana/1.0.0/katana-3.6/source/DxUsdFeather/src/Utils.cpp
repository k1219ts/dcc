#include <stdint.h>
#include <iostream>
#include <vector>

#include <OpenEXR/ImathVec.h>
#include <OpenEXR/ImathMatrix.h>

#include "Utils.h"

namespace DxUsdFeather
{

void PrintMat(Imath::M44f m, std::string prefix)
{
    std::cout << prefix << std::endl;
    std::cout << m << std::endl;
}


Imath::V3f MulVecToMat(Imath::V3f v, Imath::M44f m)
{
    Imath::V4f v4(v.x, v.y, v.z, 0.0f);
    v4 *= m;
    return Imath::V3f(v4.x, v4.y, v4.z);
}

bool AxisToMatrix(
    Imath::M44f &m,
    Imath::V3f t, Imath::V3f RAx, Imath::V3f RAy, Imath::V3f RAz, float s
)
{
    float xl = RAx.length();
    float yl = RAy.length();
    float zl = RAz.length();

    if(EqualTo(xl, 0.0f) || EqualTo(yl, 0.0f) || EqualTo(zl, 0.0f))
        return false;

    RAx /= xl;
    RAy /= yl;
    RAz /= zl;

    m = Imath::M44f(
        s*RAx.x, s*RAx.y, s*RAx.z, 0.0f,
        s*RAy.x, s*RAy.y, s*RAy.z, 0.0f,
        s*RAz.x, s*RAz.y, s*RAz.z, 0.0f,
        t.x,     t.y,     t.z,     1.0f
    );
    return true;
}


} //DxUsdFeather
