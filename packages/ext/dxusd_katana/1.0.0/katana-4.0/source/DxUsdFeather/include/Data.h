#ifndef FnGeolibOp_DxUsdFeather_Data_H
#define FnGeolibOp_DxUsdFeather_Data_H

#include <stdint.h>
#include <iostream>
#include <vector>

#include <OpenEXR/ImathVec.h>
#include <OpenEXR/ImathMatrix.h>

#include "Utils.h"

namespace DxUsdFeather
{

class Data
{
public:
    // time sample
    int64_t numSamples;
    std::vector<float> samples;

    // time-sampled point attributes
    std::vector<std::vector<float>> P;

    // none time-sampled point attributes
    std::vector<float> W;
    std::vector<float> CD;
    std::vector<float> ST;
    std::vector<float> FUV;

    // none time-sampled face attributes
    std::vector<int32_t> NV; // numver of vertices
    std::vector<int32_t> FID;
    std::vector<int32_t> FID2;
    std::vector<int32_t> BID;
    std::vector<float> BST;

    Data(){ }
    ~Data(){ }

    template<typename T>
    void SetTimeSamples(T attr)
    {
        numSamples = attr.getNumberOfTimeSamples();
        for(int64_t t=0; t<numSamples; t++)
        {
            samples.push_back(attr.getSampleTime(t));
            P.push_back(std::vector<float>());
        }
    }

    template<typename T>
    void AddImathVec(T src, std::vector<float> &dst)
    {
        int size = 2;
        if(std::is_same<T, Imath::V3f>::value)
            size = 3;

        for(int i=0; i<size; i++)
        {
            dst.push_back(src[i]);
        }
    }

    void SetPointData(
        int64_t t,
        const Imath::V3f &p,  const float &w,           const Imath::V3f &Cd,
        const Imath::V2f &st, const Imath::V3f &fuv
    )
    {
        AddImathVec(p, P[t]);
        if(!t)
        {
            W.push_back(w);
            AddImathVec(Cd, CD);
            AddImathVec(st, ST);
            AddImathVec(fuv, FUV);
        }
    }

    void SetFaceData(
        int64_t t,
        const int32_t &nv, const Imath::V2f &bST,
        const int32_t &fid2, const int32_t &bid
    )
    {
        if(!t)
        {
            NV.push_back(nv);
            AddImathVec(bST, BST);
            FID2.push_back(fid2);
            BID.push_back(bid);
        }
    }

    void SetPrimitiveData(
        int64_t t,
        const int32_t &fid
    )
    {
        if(!t)
        {
            FID.push_back(fid);
        }
    }

    FnAttribute::FloatAttribute Get_P()
    {
        std::vector<const float*> Ps;
        Ps.reserve(numSamples);
        for(int t=0; t<numSamples; t++)
            Ps[t] = &(P[t][0]);

        FnAttribute::FloatAttribute res(
            samples.data(), numSamples, Ps.data(), P[0].size(), 3
        );
        return res;
    }

    FnAttribute::IntAttribute Get_NV()
    {
        return FnAttribute::IntAttribute(NV.data(), NV.size(), 1);
    }

    FnAttribute::FloatAttribute Get_W()
    {
        return FnAttribute::FloatAttribute(W.data(), W.size(), 1);
    }

    FnAttribute::FloatAttribute Get_CD()
    {
        return FnAttribute::FloatAttribute(CD.data(), CD.size(), 3);
    }

    FnAttribute::FloatAttribute Get_ST()
    {
        return FnAttribute::FloatAttribute(ST.data(), ST.size(), 2);
    }

    FnAttribute::FloatAttribute Get_FUV()
    {
        return FnAttribute::FloatAttribute(FUV.data(), FUV.size(), 3);
    }

    FnAttribute::IntAttribute Get_FID()
    {
        return FnAttribute::IntAttribute(FID.data(), FID.size(), 1);
    }

    FnAttribute::IntAttribute Get_FID2()
    {
        return FnAttribute::IntAttribute(FID2.data(), FID2.size(), 1);
    }

    FnAttribute::IntAttribute Get_BID()
    {
        return FnAttribute::IntAttribute(BID.data(), BID.size(), 1);
    }

    FnAttribute::FloatAttribute Get_BST()
    {
        return FnAttribute::FloatAttribute(BST.data(), BST.size(), 2);
    }
};

} //DxUsdFeather
#endif
