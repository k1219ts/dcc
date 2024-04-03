/*
# ------------------------------------------------------------------------------
# Pixar
# 1200 Park Ave
# Emeryville CA 94608
#
# ------------------------------------------------------------------------------
*/

#include "RixPredefinedStrings.hpp"
#include "RixPattern.h"
#include "RixShadingUtils.h"
#include <cstring>

class DxManifold3D : public RixPattern
{
public:

    DxManifold3D();
    virtual ~DxManifold3D();

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
    enum UsePosition {
        k_UseP=0,
        k_UsePo,
        k_UsePrefObject,
        k_UsePrefWorld
    };

    // Defaults
    RtInt const m_use;
    RtFloat const m_scale;
};

DxManifold3D::DxManifold3D() :
    m_use(k_UseP),
    m_scale(1.0f)
{
}

DxManifold3D::~DxManifold3D()
{
}

int
DxManifold3D::Init(RixContext &ctx, RtUString const pluginpath)
{
    PIXAR_ARGUSED(ctx);
    PIXAR_ARGUSED(pluginpath);

    return 0;
}

enum paramId
{
    // Outputs
    k_resultX = 0,
    k_resultY,
    k_resultZ,
    k_resultBegin,
      k_resultQ,       // "result.Q"
      k_resultQradius, // "result.Qradius"
    k_resultEnd,

    // Inputs
    k_scale,
    k_use,
    k_pref,
    k_coordsys,
};

RixSCParamInfo const *
DxManifold3D::GetParamTable()
{
    static RixSCParamInfo s_ptable[] =
    {
        // outputs
        RixSCParamInfo(RtUString("resultX"), k_RixSCFloat, k_RixSCOutput),
        RixSCParamInfo(RtUString("resultY"), k_RixSCFloat, k_RixSCOutput),
        RixSCParamInfo(RtUString("resultZ"), k_RixSCFloat, k_RixSCOutput),
        RixSCParamInfo(RtUString("PxrManifold"), RtUString("result"), k_RixSCStructBegin,
                                                k_RixSCOutput),
            RixSCParamInfo(RtUString("Q"), k_RixSCPoint, k_RixSCOutput),
            RixSCParamInfo(RtUString("Qradius"), k_RixSCFloat, k_RixSCOutput),
        RixSCParamInfo(RtUString("PxrManifold"), RtUString("result"), k_RixSCStructEnd,
                                                k_RixSCOutput),

        // inputs
        RixSCParamInfo(RtUString("scale"), k_RixSCFloat),
        RixSCParamInfo(RtUString("use"), k_RixSCInteger),
        RixSCParamInfo(RtUString("pref"), k_RixSCString),
        RixSCParamInfo(RtUString("coordsys"), k_RixSCString),
        RixSCParamInfo() // end of table
    };
    return &s_ptable[0];
 }

void
DxManifold3D::Finalize(RixContext &ctx)
{
    PIXAR_ARGUSED(ctx);
}


int
DxManifold3D::ComputeOutputParams(RixShadingContext const *sctx,
                                RtInt *noutputs, OutputSpec **outputs,
                                RtPointer instanceData,
                                RixSCParamInfo const *ignored)
{
    PIXAR_ARGUSED(instanceData);
    PIXAR_ARGUSED(ignored);

    RtUString empty = Rix::k_empty;
    RtUString defaultCoordsys = Rix::k_object;
    RtUString const *pref = NULL;
    sctx->EvalParam(k_pref, -1, &pref, &empty);

    RtInt const *use;
    sctx->EvalParam(k_use, -1, &use, &m_use, false);

    bool hasPref = (*pref && !pref->Empty());

    RtUString const* coordsysPtr;
    sctx->EvalParam(k_coordsys, -1, &coordsysPtr, &defaultCoordsys);
    RtUString const coordsys = coordsysPtr->Empty() ? Rix::k_object : *coordsysPtr;

    RtFloat const *scale;
    sctx->EvalParam(k_scale, -1, &scale, &m_scale, true /*varying*/);

    // Pull in any fractalize state
    RtFloat3 const *fractalizerState;
    const static RtUString US_PXRFRACTALIZERSTATE("PxrFractalizerState");
    sctx->GetPrimVar(US_PXRFRACTALIZERSTATE,
                      RtFloat3(0.f,1.f,0.f), // octave, frequency, offset
                      &fractalizerState);

    // Allocate and bind our outputs
    RixSCType type;
    RixSCConnectionInfo cinfo;
    RixSCParamInfo const* paramTable = GetParamTable();

    RixShadingContext::Allocator pool(sctx);
    OutputSpec *o = pool.AllocForPattern<OutputSpec>(5);
    *outputs = o;
    *noutputs = 5;

    RtFloat *resultX = NULL;
    o[0].paramId = k_resultX;
    o[0].detail  = k_RixSCInvalidDetail;
    type = paramTable[k_resultX].type;
    sctx->GetParamInfo(k_resultX, &type, &cinfo);
    if (cinfo == k_RixSCNetworkValue)
    {
        resultX = pool.AllocForPattern<RtFloat>(sctx->numPts);
        o[0].detail  = k_RixSCVarying;
    }
    o[0].value = (RtPointer) resultX;

    RtFloat *resultY = NULL;
    o[1].paramId = k_resultY;
    o[1].detail  = k_RixSCInvalidDetail;
    type = paramTable[k_resultY].type;
    sctx->GetParamInfo(k_resultY, &type, &cinfo);
    if (cinfo == k_RixSCNetworkValue)
    {
        resultY = pool.AllocForPattern<RtFloat>(sctx->numPts);
        o[1].detail  = k_RixSCVarying;
    }
    o[1].value = (RtPointer) resultY;

    RtFloat *resultZ = NULL;
    o[2].paramId = k_resultZ;
    o[2].detail  = k_RixSCInvalidDetail;
    type = paramTable[k_resultZ].type;
    sctx->GetParamInfo(k_resultZ, &type, &cinfo);
    if (cinfo == k_RixSCNetworkValue)
    {
        resultZ = pool.AllocForPattern<RtFloat>(sctx->numPts);
        o[2].detail  = k_RixSCVarying;
    }
    o[2].value = (RtPointer) resultZ;

    // We always allocate memory for the manifold
    RtPoint3 *Q = pool.AllocForPattern<RtPoint3>(sctx->numPts);
    RtFloat *Qradius = pool.AllocForPattern<RtFloat>(sctx->numPts);
    o[3].paramId = k_resultQ;
    o[3].detail = k_RixSCVarying;
    o[3].value = (RtPointer) Q;
    o[4].paramId = k_resultQradius;
    o[4].detail = k_RixSCVarying;
    o[4].value = (RtPointer) Qradius;

    // First get the primvar.
    RtFloat3 const *pv3;
    RtFloat const *pv3Width;
    RtFloat3 fill (0.f, 0.f, 0.f);


    // lookup pref primvar.
    RixSCDetail pvarDetail = k_RixSCInvalidDetail;
    if (hasPref)
    {
        pvarDetail = sctx->GetPrimVar(*pref, fill, &pv3, &pv3Width);
    }

    if (pvarDetail == k_RixSCInvalidDetail)
    {
        // the pref parameter is empty or the lookup has failed.
        if (*use == k_UsePo)
        {
            sctx->GetBuiltinVar(RixShadingContext::k_Po, &pv3);
        }
        else
        {
            // whatever variable was selected, use P.
            sctx->GetBuiltinVar(RixShadingContext::k_P, &pv3);
        }
        sctx->GetBuiltinVar(RixShadingContext::k_PRadius, &pv3Width);
    }

    // always copy positions before applying transformations, otherwise you
    // modify the original values.
    //
    memcpy(Q,       pv3,      sizeof(RtPoint3) * sctx->numPts);
    memcpy(Qradius, pv3Width, sizeof(RtFloat)  * sctx->numPts);

    // Note: regular object-space __Pref has already been automatically
    //       transformed to current space.
    //       For world-space __WPref, we need to first undo the automatic
    //       transformation to current space, and then apply the world->current
    //       transform.
    //
    if (hasPref && *use == k_UsePrefWorld)
    {
        sctx->Transform(RixShadingContext::k_AsPoints,
                        Rix::k_current, Rix::k_object, Q, Qradius);
        sctx->Transform(RixShadingContext::k_AsPoints,
                        Rix::k_world, Rix::k_current, Q, Qradius);
    }

    // Convert to the named coordinate system
    sctx->Transform(RixShadingContext::k_AsPoints,
                    Rix::k_current, coordsys, Q, Qradius);

    for(int i = 0; i < sctx->numPts; ++i)
    {
        // XXX: We haven't implemented octave or offset yet
        //  p' = p*freq + offset * PxrMix(-.5, .5, noise(octave))
        float freq = scale[i] * fractalizerState[0][1];

        // Scale our results
        Q[i].x *= freq;
        Q[i].y *= freq;
        Q[i].z *= freq;
        Qradius[i] *= freq;

        // Copy to other outputs if connected
        if (resultX) resultX[i] = Q[i].x;
        if (resultY) resultY[i] = Q[i].y;
        if (resultZ) resultZ[i] = Q[i].z;
    }

    return 0;
}

RIX_PATTERNCREATE
{
    PIXAR_ARGUSED(hint);

    return new DxManifold3D();
}

RIX_PATTERNDESTROY
{
    delete ((DxManifold3D*)pattern);
}

