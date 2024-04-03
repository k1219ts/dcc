#ifndef FnGeolibOp_DxUsdFeather_NurbsCurve_H
#define FnGeolibOp_DxUsdFeather_NurbsCurve_H

#include <stdint.h>
#include <iostream>
#include <vector>

#include <OpenEXR/ImathVec.h>
#include <OpenEXR/ImathMatrix.h>

#include "Utils.h"

namespace DxUsdFeather
{

class NurbsCurve
{
public:
    std::vector<Imath::V3f> CV;
    std::vector<int> knot;
    int degree;

    NurbsCurve() { };
    ~NurbsCurve() { }

    NurbsCurve(const std::vector<Imath::V3f>& cvs, const int& d=3)
    {
        CV = cvs;
        degree = d;

        int knotCount = CV.size() - degree;
        if(!knotCount)
            return;

        // initialize
        for(int i = 0; i <= knotCount; i++)
            knot.push_back(i);

        //  insert 0, -1 index
        for(int i = 0; i < degree; ++i)
        {
            knot.insert(knot.begin(), 0);
            knot.push_back(knot.back());
        }
    }

    Imath::V3f GetPosition(float u, bool tangent=false);
    void Print(bool debug = true);

private:
    int GetSpan(float u);
    std::vector<std::vector<float>> Derivative(float u, int span, int order);

};

} //DxUsdFeather
#endif
