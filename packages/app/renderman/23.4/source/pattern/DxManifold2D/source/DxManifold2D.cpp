/*
# ------------------------------------------------------------------------------
# create 2019.11.28 $0
# ------------------------------------------------------------------------------
*/

#include "RixPredefinedStrings.hpp"
#include "RixPattern.h"
#include "RixShadingUtils.h"
class DxManifold2D : public RixPattern
{
public:

    DxManifold2D();
    virtual ~DxManifold2D();

    virtual int Init(RixContext &, RtUString const pluginpath) override;
    virtual RixSCParamInfo const *GetParamTable() override;
    virtual void Synchronize(
        RixContext&,
        RixSCSyncMsg,
        RixParameterList const*) override
    {
    }

    virtual void Finalize(RixContext &) override;

    virtual int ComputeOutputParams(RixShadingContext const *,
                                    RtInt *noutputs,
                                    OutputSpec **outputs,
                                    RtPointer instanceData,
                                    RixSCParamInfo const *) override;

    virtual bool Bake2dOutput(
        RixBakeContext const*,
        Bake2dSpec&,
        RtPointer) override
    {
        return false;
    }

    virtual bool Bake3dOutput(
        RixBakeContext const*,
        Bake3dSpec&,
        RtPointer) override
    {
        return false;
    }

private:
    // Defaults
    RtFloat const m_angle;
    RtFloat const m_scaleS;
    RtFloat const m_scaleT;
    RtFloat const m_offsetS;
    RtFloat const m_offsetT;
    RtInt const m_invertS;
    RtInt const m_invertT;
    RtFloat m_inputS;
    RtFloat m_inputT;
};

DxManifold2D::DxManifold2D() :
    m_angle(0.0f),
    m_scaleS(1.0f),
    m_scaleT(1.0f),
    m_offsetS(0.0f),
    m_offsetT(0.0f),
    m_invertS(0),
    m_invertT(1),
    m_inputS(0.0f),
    m_inputT(0.0f)
{
}

DxManifold2D::~DxManifold2D()
{
}

int
DxManifold2D::Init(RixContext &ctx, RtUString const pluginpath)
{
    PIXAR_ARGUSED(ctx);
    PIXAR_ARGUSED(pluginpath);

    return 0;
}

enum paramId
{
    // Outputs
    k_resultS = 0,
    k_resultT,
    k_resultBegin,
      k_resultQ,       // "result.Q"
      k_resultQradius, // "result.Qradius"
    k_resultEnd,

    // Inputs
    k_angle,
    k_scaleS,
    k_scaleT,
    k_offsetS,
    k_offsetT,
    k_invertS,
    k_invertT,
    k_inputS,
    k_inputT
};

RixSCParamInfo const *
DxManifold2D::GetParamTable()
{
    static RixSCParamInfo s_ptable[] =
    {
        // outputs
        RixSCParamInfo(RtUString("resultS"), k_RixSCFloat, k_RixSCOutput),
        RixSCParamInfo(RtUString("resultT"), k_RixSCFloat, k_RixSCOutput),
        RixSCParamInfo(RtUString("PxrManifold"), RtUString("result"), k_RixSCStructBegin,
                                                k_RixSCOutput),
        RixSCParamInfo(RtUString("Q"), k_RixSCPoint, k_RixSCOutput),
        RixSCParamInfo(RtUString("Qradius"), k_RixSCFloat, k_RixSCOutput),
        RixSCParamInfo(RtUString("PxrManifold"), RtUString("result"), k_RixSCStructEnd,
                                                k_RixSCOutput),

        // inputs
        RixSCParamInfo(RtUString("angle"), k_RixSCFloat),
        RixSCParamInfo(RtUString("scaleS"), k_RixSCFloat),
        RixSCParamInfo(RtUString("scaleT"), k_RixSCFloat),
        RixSCParamInfo(RtUString("offsetS"), k_RixSCFloat),
        RixSCParamInfo(RtUString("offsetT"), k_RixSCFloat),
        RixSCParamInfo(RtUString("invertS"), k_RixSCInteger),
        RixSCParamInfo(RtUString("invertT"), k_RixSCInteger),
        RixSCParamInfo(RtUString("inputS"), k_RixSCFloat),
        RixSCParamInfo(RtUString("inputT"), k_RixSCFloat),
        RixSCParamInfo() // end of table
    };
    return &s_ptable[0];
 }

void
DxManifold2D::Finalize(RixContext &ctx)
{
    PIXAR_ARGUSED(ctx);
}


int
DxManifold2D::ComputeOutputParams(RixShadingContext const *sctx,
                                RtInt *noutputs, OutputSpec **outputs,
                                RtPointer instanceData,
                                RixSCParamInfo const *ignored)
{
    PIXAR_ARGUSED(instanceData);
    PIXAR_ARGUSED(ignored);

    bool varying = true;
    bool uniform = false;

    RtFloat const *angle;
    sctx->EvalParam(k_angle, -1, &angle, &m_angle, varying);

    RtFloat const *scaleS;
    sctx->EvalParam(k_scaleS, -1, &scaleS, &m_scaleS, varying);

    RtFloat const *scaleT;
    sctx->EvalParam(k_scaleT, -1, &scaleT, &m_scaleT, varying);

    RtFloat const *offsetS;
    sctx->EvalParam(k_offsetS, -1, &offsetS, &m_offsetS, varying);

    RtFloat const *offsetT;
    sctx->EvalParam(k_offsetT, -1, &offsetT, &m_offsetT, varying);

    RtInt const *invertSp;
    sctx->EvalParam(k_invertS, -1, &invertSp, &m_invertS, uniform);
    RtInt const invertS(*invertSp);

    RtInt const *invertTp;
    sctx->EvalParam(k_invertT, -1, &invertTp, &m_invertT, uniform);
    RtInt const invertT(*invertTp);

    RtFloat const *inputS;
    sctx->EvalParam(k_inputS, -1, &inputS, &m_inputS, varying);

    RtFloat const *inputT;
    sctx->EvalParam(k_inputT, -1, &inputT, &m_inputT, varying);

    // Allocate and bind our outputs
    RixSCType type;
    RixSCConnectionInfo cinfo;
    RixSCParamInfo const* paramTable = GetParamTable();

    RixShadingContext::Allocator pool(sctx);
    OutputSpec *o = pool.AllocForPattern<OutputSpec>(4);
    *outputs = o;
    *noutputs = 4;

    RtFloat *resultS = NULL;
    o[0].paramId = k_resultS;
    o[0].detail  = k_RixSCInvalidDetail;
    type = paramTable[k_resultS].type;
    sctx->GetParamInfo(k_resultS, &type, &cinfo);
    if (cinfo == k_RixSCNetworkValue)
    {
        resultS = pool.AllocForPattern<RtFloat>(sctx->numPts);
        o[0].detail  = k_RixSCVarying;
    }
    o[0].value = (RtPointer) resultS;

    RtFloat *resultT = NULL;
    o[1].paramId = k_resultT;
    o[1].detail  = k_RixSCInvalidDetail;
    type = paramTable[k_resultT].type;
    sctx->GetParamInfo(k_resultT, &type, &cinfo);
    if (cinfo == k_RixSCNetworkValue)
    {
        resultT = pool.AllocForPattern<RtFloat>(sctx->numPts);
        o[1].detail  = k_RixSCVarying;
    }
    o[1].value = (RtPointer) resultT;

    // We always allocate memory for the manifold
    RtPoint3 *Q = pool.AllocForPattern<RtPoint3>(sctx->numPts);
    RtFloat *Qradius = pool.AllocForPattern<RtFloat>(sctx->numPts);
    o[2].paramId = k_resultQ;
    o[2].detail = k_RixSCVarying;
    o[2].value = (RtPointer) Q;
    o[3].paramId = k_resultQradius;
    o[3].detail = k_RixSCVarying;
    o[3].value = (RtPointer) Qradius;

    // Default ST
    RtFloat2 const *pv2;
    RtFloat const *pv2Width;
    RtFloat2 fill (0.f, 0.f);
    sctx->GetPrimVar(Rix::k_st, fill, &pv2, &pv2Width);
    for(int i = 0; i < sctx->numPts; ++i)
    {
        Q[i] = RtPoint3(pv2[i][0], pv2[i][1], 0.f);
        Qradius[i] = pv2Width[i];
    }

    // inputS
    type = paramTable[k_inputS].type;
    sctx->GetParamInfo(k_inputS, &type, &cinfo);
    if (cinfo == k_RixSCNetworkValue)
    {
        for (int i=0; i<sctx->numPts; ++i)
        {
            Q[i].x = inputS[i];
        }
    }

    // inputT
    type = paramTable[k_inputT].type;
    sctx->GetParamInfo(k_inputT, &type, &cinfo);
    if (cinfo == k_RixSCNetworkValue)
    {
        for (int i=0; i<sctx->numPts; ++i)
        {
            Q[i].y = inputT[i];
        }
    }

    // Now we need to perform transformations
    for(int i = 0; i < sctx->numPts; ++i)
    {
        // Invert S
        float invS;
        if (invertS)
            invS = floorf(Q[i].x) + 1.0f - RixFractional(Q[i].x);
        else
            invS = Q[i].x;

        // Invert T
        float invT;
        if (invertT)
            invT = floorf(Q[i].y) + 1.0f - RixFractional(Q[i].y);
        else
            invT = Q[i].y;

        // Offset & scale
        Q[i].x = scaleS[i] * invS + offsetS[i];
        Q[i].y = scaleT[i] * invT + offsetT[i];
        Qradius[i] = RixMin(scaleS[i] * Qradius[i], scaleT[i] * Qradius[i]);

        // Rotate
        if (angle[i] != 0.f)
        {
            float rx, ry;
            float cs, sn;
            RixSinCos(RixDegreesToRadians(angle[i]), &sn, &cs);
            rx = Q[i].x * cs - Q[i].y * sn;
            ry = Q[i].x * sn + Q[i].y * cs;
            Q[i].x = rx;
            Q[i].y = ry;
        }

        // Copy to other outputs if connected
        if (resultS) resultS[i] = Q[i].x;
        if (resultT) resultT[i] = Q[i].y;
    }

    return 0;
}

RIX_PATTERNCREATE
{
    PIXAR_ARGUSED(hint);

    return new DxManifold2D();
}

RIX_PATTERNDESTROY
{
    delete ((DxManifold2D*)pattern);
}
