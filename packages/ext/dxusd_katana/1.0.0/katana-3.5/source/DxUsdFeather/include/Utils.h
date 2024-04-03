#ifndef FnGeolibOp_DxUsdFeather_Utils_H
#define FnGeolibOp_DxUsdFeather_Utils_H

#include <stdint.h>
#include <iostream>
#include <iomanip>
#include <vector>

#include <OpenEXR/ImathVec.h>
#include <OpenEXR/ImathMatrix.h>

#define SHAP  "################################################################"
#define DASH  "----------------------------------------------------------------"


namespace DxUsdFeather
{

template<typename T>
void PrintVec(T vec, std::string prefix="")
{
    if(std::is_same<T, Imath::V2f>::value)
    {
        std::cout << prefix;
        std::cout << vec[0] << ", " << vec[0] << std::endl;
    }
    else if(std::is_same<T, Imath::V3f>::value)
    {
        std::cout << prefix;
        std::cout << vec[0] << ", " << vec[1] << ", " << vec[2] << std::endl;
    }
}

template<typename T>
void PrintVecs(std::vector<T> vecs, std::string prefix="")
{
    for(auto vec : vecs)
        PrintVec(vec, prefix);
}

void PrintMat(Imath::M44f m, std::string prefix="");

Imath::V3f MulVecToMat(Imath::V3f v, Imath::M44f m);

bool AxisToMatrix(
    Imath::M44f &m,
    Imath::V3f t, Imath::V3f RAx, Imath::V3f RAy, Imath::V3f RAz, float s=1.0f
);

template<typename T>
bool EqualTo(T s, T d, int c=3)
{
    const T e = c*Imath::limits<T>::epsilon();
    return Imath::equal(s, d, e);
}

} //DxUsdFeather
#endif
