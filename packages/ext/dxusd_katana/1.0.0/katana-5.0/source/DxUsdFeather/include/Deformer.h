#ifndef FnGeolibOp_DxUsdFeather_Deformer_H
#define FnGeolibOp_DxUsdFeather_Deformer_H

#include <stdint.h>
#include <iostream>
#include <vector>

#include <OpenEXR/ImathVec.h>
#include <OpenEXR/ImathMatrix.h>
#include <OpenEXR/ImathQuat.h>

#include "NurbsCurve.h"
#include "Data.h"

namespace DxUsdFeather
{

struct Barb
{
    int32_t num;
    int32_t startCV;
    int32_t bid;
    Imath::V2f RE;
    //initial cvs on uVec
    std::vector<Imath::V3f> uCV;
    //vector param(richis) to param(edge) in tangentSpace
    Imath::V3f uVec;
    Imath::V3f uTangent;

    std::vector<Imath::V3f> CV;
    std::vector<Imath::V2f> ST;
    std::vector<Imath::V3f> FUV;
    std::vector<float> U;
    std::vector<float> W;
    std::vector<float> offset;
    std::vector<Imath::V3f> Cd;
    std::vector<Imath::Quatf> Q;
};

struct Lamination
{
    int64_t fid = 0;     // fid
    int64_t lid = 0;     // lamination id
    Imath::M44f M;
    Imath::M44f IM;
    float scale = 1.0;
    NurbsCurve crvs[3];
    std::vector<Imath::V3f> Ps[3];
    Imath::V2f bodyST;
};

class Deformer
{
public:
    Deformer() { }
    ~Deformer() { }

    std::string fthGrpPath;
    std::string lamPath;
    std::string fthPath;

    float frontParm;
    int32_t numv; // number of lam vtcs

    std::vector<Barb> barbs[3];
    Lamination slam;
    std::vector<Lamination> lams;

    std::vector<int32_t> skip;

    bool SetFeatherElements(const std::vector<std::string> elms);

    void Deform(
        int64_t t, Data &data,
        bool debug=false, int32_t debuglam=-1, int32_t debugbarb=-1
    );

    void Debug(bool debug);
    bool Debug_Lam(Lamination lam, bool debug, int32_t idx);
    bool Debug_Barb(Barb barb, bool debug, int32_t idx);

private:
    const std::string LMS = "Laminations";
    const std::string FTH = "Feather";

    float GetFrontParm();
    void  ResolveSourceFeather(bool debug, int32_t debugbarb);
    bool  BarbSpace(
        Imath::M44f &m, Lamination &lam, int32_t bid,
        float rp, Imath::V3f r, Imath::V3f F, bool confirm=false
    );
    bool  LamSpace(Lamination &lam, bool setScale=true);
    bool  GetK(Imath::V3f &k, Imath::V3f e, Imath::V3f U, Imath::M44f TS);
    float Reparameter(float u, float k);
    void  Deform_Rachis(
        Lamination lam, int64_t t, Data &data,
        bool debug = false
    );
    void  Deform_Barbs(
        Lamination lam, int32_t s, int64_t t, Data &data,
        bool debug, int32_t debugbarb
    );
};

} //DxUsdFeather
#endif
