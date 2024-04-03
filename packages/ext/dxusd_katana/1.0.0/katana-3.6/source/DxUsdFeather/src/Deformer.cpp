#include <FnAttribute/FnAttribute.h>

#include <OpenEXR/ImathVec.h>
#include <OpenEXR/ImathMatrix.h>
#include <OpenEXR/ImathMatrixAlgo.h>
#include <algorithm>
#include <cmath>

#include <time.h>

#include "Deformer.h"
#include "Utils.h"
#include "Data.h"

namespace DxUsdFeather
{

bool Deformer::SetFeatherElements(const std::vector<std::string> elms)
{
    for(int j = 0; j < elms.size(); j++)
    {
        if(elms[j] == LMS)
            lamPath = fthGrpPath + "/" + elms[j];
        else if(elms[j] == FTH)
            fthPath = fthGrpPath + "/" + elms[j];
    }
    if(lamPath == "" || fthPath == "")
        return false;

    return true;
}

void Deformer::Debug(bool debug)
{
    if(!debug)
        return;

    std::cout << std::cout.precision(10);
    std::cout << "\n\n\n" << std::endl;
    std::cout << SHAP << std::endl;
    std::cout << fthGrpPath << std::endl;
    std::cout << SHAP << std::endl;
    std::cout << "# Original Lamination Vertices : frontParm[";
    std::cout << frontParm << "]" << std::endl;
    std::cout << "# Num Barbs : " << barbs[1].size() * 2 + 1 << std::endl;
    PrintVecs(slam.Ps[0], "# olam[0] : ");
    std::cout << DASH << std::endl;
    PrintVecs(slam.Ps[1], "# olam[1] : ");
    std::cout << DASH << std::endl;
    PrintVecs(slam.Ps[2], "# olam[2] : ");
}

bool Deformer::Debug_Lam(Lamination lam, bool debug, int32_t idx)
{
    if(!debug)
        return false;

    if(idx >= 0 && lam.fid != idx)
        return true;

    std::cout << SHAP << std::endl;
    std::cout << "# Deformed Lamination (" << lam.fid << ") - Scale : ";
    std::cout << lam.scale << std::endl;
    std::cout << SHAP << std::endl;
    PrintMat(lam.IM, "# Inverse Matrix ");
    std::cout << DASH << std::endl;
    PrintVecs(lam.Ps[0], "# dlam[0] : ");
    std::cout << DASH << std::endl;
    PrintVecs(lam.Ps[1], "# dlam[1] : ");
    std::cout << DASH << std::endl;
    PrintVecs(lam.Ps[2], "# dlam[2] : ");
    std::cout << "bodyST : " << lam.bodyST[0] << ", " << lam.bodyST[1] << std::endl;

    return false;
}

bool Deformer::Debug_Barb(Barb barb, bool debug, int32_t idx)
{
    if(!debug)
        return false;

    if(idx >=0 && barb.bid != idx)
        return false;

    std::cout << SHAP << std::endl;
    std::cout << "# Barb [" << barb.bid << "] : startCV [" << barb.startCV;
    std::cout << "] " << barb.RE[0] << ", " << barb.RE[1] << std::endl;
    std::cout << "# U : ";
    for(auto u : barb.U)
        std::cout << u << ", ";
    std::cout << std::endl;
    std::cout << DASH << std::endl;
    PrintVecs(barb.CV, "# cv : ");
    std::cout << DASH << std::endl;
    PrintVecs(barb.uCV, "# uCV : ");
    std::cout << DASH << std::endl;
    PrintVec(barb.uVec, "# uVec : ");

    return true;
}

float Deformer::GetFrontParm()
{
    Imath::V3f v1 = slam.Ps[2][0] - slam.Ps[1][0];
    Imath::V3f v2 = slam.Ps[0][1] - slam.Ps[0][0];
    Imath::V3f v3 = slam.Ps[0][0] - slam.Ps[1][0];

    if(v1.x == 0.0f && v2.x != 0.0f)
        return v3.x / v1.x;
    return (v2.y*v3.x/v2.x - v3.y) / (v2.y*v1.x/v2.x - v1.y);
}

bool Deformer::LamSpace(Lamination &lam, bool setScale)
{
    if(setScale)
    {
        Imath::V3f OL = slam.Ps[1][0] - slam.Ps[0][0];
        Imath::V3f OR = slam.Ps[2][0] - slam.Ps[0][0];
        Imath::V3f DL =  lam.Ps[1][0] -  lam.Ps[0][0];
        Imath::V3f DR =  lam.Ps[2][0] -  lam.Ps[0][0];
        lam.scale = (DL.length()/OL.length() + DR.length()/OR.length()) / 2;
        if(lam.scale < 0.001)
            return false;
    }

    Imath::V3f T   = lam.Ps[0][0];
    Imath::V3f DRL = (lam.Ps[1][0] - lam.Ps[2][0]).normalized();
    Imath::V3f Y   = (lam.Ps[0][1] - lam.Ps[0][0]).normalized();
    Imath::V3f Z   = DRL.cross(Y);
    Imath::V3f X   = Y.cross(Z);

    Imath::M44f m;
    if(!AxisToMatrix(m, T, X, Y, Z, lam.scale))
        return false;

    lam.M  = m;
    lam.IM = m.inverse();
    return true;
}

bool Deformer::BarbSpace(
    Imath::M44f &m, Lamination &lam, int32_t bid,
    float rp, Imath::V3f r, Imath::V3f F, bool confirm
)
{
    if(!confirm && std::find(skip.begin(), skip.end(), bid) != skip.end())
        return false;

    float erp = rp / lam.crvs[0].knot.back() * lam.crvs[1].knot.back();

    // r, f in lam matrix
    Imath::V3f r_IM = r*lam.IM;
    Imath::V3f F_IM = MulVecToMat(F, lam.IM);
    Imath::V3f ep1 = lam.crvs[1].GetPosition(erp)*lam.IM - r_IM;
    Imath::V3f ep2 = lam.crvs[2].GetPosition(erp)*lam.IM - r_IM;

    float ep1l = ep1.length();
    float ep2l = ep2.length();

    if(confirm && (EqualTo(ep1l, 0.0f) || EqualTo(ep2l, 0.0f)))
    {
        skip.push_back(bid);
        return false;
    }
    // normalize
    ep1 /= ep1l;
    ep2 /= ep2l;

    // T is twist axis
    Imath::V3f T;
    Imath::V3f U;

    if(EqualTo(ep1.z, 0.0f, 1))
        T = Imath::V3f(-ep1.x, ep1.y, ep1.z);
    else if(EqualTo(ep2.z, 0.0f, 1))
        T = Imath::V3f(ep2);
    else
        T = ep2/fabs(sinf(ep2.z)) - ep1/fabs(sinf(ep1.z));

    U = F_IM.cross(T);
    T = U.cross(F_IM);

    bool res = AxisToMatrix(m, r_IM, F_IM, T, U);
    m *= lam.M;

    if(confirm && !res)
    {
        skip.push_back(bid);
        return false;
    }
    return true;
}

bool Deformer::GetK(Imath::V3f &k, Imath::V3f e, Imath::V3f U, Imath::M44f TS)
{
    Imath::M44f ITS = TS.inverse();
    Imath::V3f e_TS = e*ITS;
    Imath::V3f nu   = Imath::V3f(U);

    float el_TS = e_TS.length();
    float nul   = nu.length();

    if(EqualTo(el_TS, 0.0f) || EqualTo(nul, 0.0f))
        return false;

    float th = nu.dot(e_TS)/(el_TS*nul);
    th = (th > 1.0f)? 1.0f : th;
    th = 2*acosf(th) / M_PI;
    th = (th > 1.0f)? 1.0f : th;

    float min = 0.05f;
    float max = 0.8f;

    k = (th*(max - min) + min)*U;
    k *= TS;


    return true;
}

float Deformer::Reparameter(float u, float k)
{
    u = (u < 0.0f)? 0.0f : u;
    u = (u > 1.0f)? 1.0f : u;

    if(EqualTo(k, 0.5f))
        return u;

    float tmp = 1.0f - 2*k;
    return (sqrtf(k*k + u*tmp) - k) / tmp;
}

void Deformer::ResolveSourceFeather(bool debug, int32_t debugbarb)
{
    // source lam space
    LamSpace(slam, false);

    // set curves
    for(int i=0; i<3; i++)
    {
        // transform all lam's vertices in slam space
        for(int v=0; v<slam.Ps[i].size(); v++)
            slam.Ps[i][v] *= slam.IM;

        slam.crvs[i] = NurbsCurve(slam.Ps[i]);
    }

    for(int s=1; s<3; s++)
    {
        for(auto &barb : barbs[s])
        {
            // transform all cvs in slam space
            for(int v=0; v<barb.CV.size(); v++)
                barb.CV[v]  *= slam.IM;

            Imath::V3f r = slam.crvs[0].GetPosition(barb.RE[0]);
            Imath::V3f F = slam.crvs[0].GetPosition(barb.RE[0], true);
            Imath::V3f e = slam.crvs[s].GetPosition(barb.RE[1]);
            Imath::V3f rToE = e - r;

            // get barb space
            Imath::M44f TS;
            if(!BarbSpace(TS, slam, barb.bid, barb.RE[0], r, F, true))
                continue;

            // set barb's uCV
            for(int v=0; v<barb.CV.size(); v++)
            {
                Imath::V3f uCV = barb.U[v]*rToE + r;
                Imath::V3f uToCV = barb.CV[v] - uCV;
                barb.uCV.push_back(uCV);

                float angle = 0.0f;
                float gap   = uToCV.length();
                barb.uTangent = rToE.normalized();
                uToCV /= gap;
                Imath::V3f axis = barb.uTangent.cross(uToCV);

                if(EqualTo(axis.length(), 0.0f))
                {
                    Imath::V3f zAxis = MulVecToMat(Imath::V3f(0, 0, 1), TS);
                    axis = barb.uTangent.cross(zAxis);
                }

                axis.normalize();
                angle = acosf(uToCV.dot(barb.uTangent));
                axis *= sinf(0.5f*angle);
                angle = cosf(0.5f*angle);
                Imath::Quatf quat(Imath::Quatf(angle, axis));
                barb.Q.push_back(quat);
                barb.offset.push_back(gap);
            }

            // set barb's uVec in barb space
            barb.uVec = e * TS.inverse();

            Debug_Barb(barb, debug, debugbarb);
        }
    }
}

void Deformer::Deform(
    int64_t t, Data &data,
    bool debug, int32_t debuglam, int32_t debugbarb
)
{
    frontParm = GetFrontParm();
    Debug(debug);
    ResolveSourceFeather(debug, debugbarb);

    clock_t tDeformStart = clock();

    for(auto &lam : lams)
    {
        if(!LamSpace(lam))
        {
            std::cout << "# Warning : Skipped lamination - ";
            std::cout << lam.lid << std::endl;
            continue;
        }
        if(Debug_Lam(lam, debug, debuglam))
            continue;

        for(int i=0; i<3; i++)
            lam.crvs[i] = NurbsCurve(lam.Ps[i]);

        Deform_Rachis(lam,   t, data, debug);
        Deform_Barbs(lam, 1, t, data, debug, debugbarb);
        Deform_Barbs(lam, 2, t, data, debug, debugbarb);
    }

    std::cout << fthGrpPath << " feathering time(" << t << ") : ";
    std::cout << (double)(clock() - tDeformStart)/CLOCKS_PER_SEC;
    std::cout << "s" << std::endl;

    data.SetPrimitiveData(t, slam.fid);
}

void Deformer::Deform_Rachis(
    Lamination lam, int64_t t, Data &data,
    bool debug
)
{
    std::vector<size_t> idcs;
    for(size_t i=0; i<lam.Ps[0].size(); i++)
        idcs.push_back(i);

    idcs.insert(idcs.begin(), 0);
    idcs.push_back(idcs.back());

    for(size_t i : idcs)
    {
        data.SetPointData(t,
            lam.Ps[0][i],       barbs[0][0].W[i]*lam.scale,  barbs[0][0].Cd[i],
            barbs[0][0].ST[i],  barbs[0][0].FUV[i]
        );
    }
    data.SetFaceData(t,
        barbs[0][0].num,     lam.bodyST,
        lam.fid,             barbs[0][0].bid
    );
}

void Deformer::Deform_Barbs(
    Lamination lam, int32_t s, int64_t t, Data &data,
    bool debug, int32_t debugbarb
)
{
    for(auto &barb : barbs[s])
    {
        // only
        if(debug && debugbarb >= 0 && debugbarb != barb.bid)
            continue;
        // ^ Deforming Time(0) : 5.17s


        Imath::V3f r = lam.crvs[0].GetPosition(barb.RE[0]);
        Imath::V3f F = lam.crvs[0].GetPosition(barb.RE[0], true);
        Imath::V3f e = lam.crvs[s].GetPosition(barb.RE[1]);
        // ^ Deforming Time(0) : 11.36s


        NurbsCurve rke;
        Imath::M44f TS;
        Imath::V3f k;
        float rat;
        bool  res = true;

        try
        {
            // get TS
            if(!BarbSpace(TS, lam, barb.bid, barb.RE[0], r, F))
                throw(0);
            // ^ Deforming Time(0) : 16.59s


            // r - k - e curve
            if(!GetK(k, e, barb.uVec, TS))
                throw(0);
            // ^ Deforming Time(0) : 17.36s


            std::vector<Imath::V3f> rkeps = {r, k, e};
            rke = NurbsCurve(rkeps, 2);

            // ^ Deforming Time(0) : 20.92s

            // ratio of k between r and e
            float rkl = (r - k).length();
            float ekl = (e - k).length();
            if(EqualTo(rkl + ekl, 0.0f))
                throw(0);

            rat = rkl / (rkl + ekl);
            // ^ Deforming Time(0) : 21.24s

        }
        catch(int expn)
        {
            std::cout << "# Warning : Skipped barb - " << barb.bid << std::endl;
            res = false;
        }

        // deform barbs
        for(int c=1; c<barb.num-1; c++)
        {
            Imath::V3f cv;
            if(res)
            {
                float p = Reparameter(barb.U[c], rat);
                // ^ Deforming Time(0) : 21.92s

                cv = rke.GetPosition(p);
                // ^ Deforming Time(0) : 32.42s


                if(c != 0 && !EqualTo(barb.offset[c], 0.0f))
                {
                    Imath::V3f disp = rke.GetPosition(p, true);
                    // ^ Deforming Time(0) : 47.03s

                    disp *= barb.offset[c] / disp.length();
                    disp = MulVecToMat(disp, lam.IM);
                    // ^ Deforming Time(0) : 49.96s

                    disp = disp * barb.Q[c] * lam.scale;
                    disp = MulVecToMat(disp, lam.M);
                    cv += disp;
                    // ^ Deforming Time(0) : 51.76s
                }
            }

            // -----------------------------------------------------------------
            // set data
            std::vector<size_t> cidcs;
            if(c == 1) cidcs.push_back(0);
            cidcs.push_back(c);
            if(c == barb.num-2) cidcs.push_back(c+1);

            for(size_t cidx : cidcs)
            {
                data.SetPointData(t,
                    cv,             barb.W[cidx]*lam.scale, barb.Cd[cidx],
                    barb.ST[cidx],  barb.FUV[cidx]
                );
            }
        }

        data.SetFaceData(t,
            barb.num,       lam.bodyST,
            lam.fid,        barb.bid
        );

        // ---------------------------------------------------------------------
        // debugging
        if(debug && (debugbarb < 0 || debugbarb == barb.bid))
        {
            std::cout << SHAP << std::endl;
            std::cout << "# Deformed Barb[" << barb.bid << "]" << std::endl;
            Debug_Barb(barb, debug, debugbarb);
            std::cout << SHAP << std::endl;
            std::cout << "# RP : " << barb.RE[0] << std::endl;
            PrintVec(r, "# R : ");
            PrintVec(F, "# F : ");
            PrintVec(e, "# E : ");
            std::cout << "# Barb Space" << std::endl;
            PrintVec(barb.uVec, "# uVec : ");
            PrintMat(TS, "# TS : ");
            PrintVec(k, "# K : ");
            std::cout << "# RAT : " << rat << std::endl;
        }
    }
}























} //DxUsdFeather
