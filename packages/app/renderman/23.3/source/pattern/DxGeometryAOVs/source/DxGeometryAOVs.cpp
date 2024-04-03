/*
Dexter $Date: 2018/08/29 $  $Revision: #1 $
# ------------------------------------------------------------------------------
#
#   Dexter
#
#       sanghun.kim
#
# ------------------------------------------------------------------------------
*/
#include "RixPredefinedStrings.hpp"
#include "RixPattern.h"
#include "RixShadingUtils.h"
#include "RixColorUtils.h"
#include "RixIntegrator.h"
#include <cstring>
#include <stdint.h>

#define MAX_NUM_AOVS 2

const static RtUString US_NN("Nnx");
const static RtUString US_MV("MV");
static RtUString const s_aovNames[MAX_NUM_AOVS] = {
    US_NN,US_MV
};

class DxGeometryAOVs : public RixPattern
{
public:
    DxGeometryAOVs();
    virtual ~DxGeometryAOVs();

    virtual int Init(RixContext &, RtUString pluginpath) override;
    virtual RixSCParamInfo const *GetParamTable() override;
    virtual void Synchronize(RixContext&, RixSCSyncMsg, RixParameterList const*) override
    {
    }
    virtual void CreateInstanceData(RixContext&, RtUString const, RixParameterList const*, InstanceData*) override;
    virtual void SynchronizeInstanceData(RixContext&, RtUString const, RixParameterList const*, uint32_t editHints, InstanceData*) override;
    virtual void Finalize(RixContext &) override;
    virtual int ComputeOutputParams(RixShadingContext const *, RtInt *n, OutputSpec **outputs, RtPointer instanceData, RixSCParamInfo const *) override;
    virtual bool Bake2dOutput(RixBakeContext const*, Bake2dSpec&, RtPointer) override
    {
        return false;
    }
    virtual bool Bake3dOutput(RixBakeContext const*, Bake3dSpec&, RtPointer) override
    {
        return false;
    }

private:
    void initializeInstanceData(RixContext& ctx, RtUString const handle, RixParameterList const* plist, InstanceData* idata);
    // Defaults
    RtInt const m_inputAOV;
    RtInt const m_enable;

    RixMessages *m_msg;
};

DxGeometryAOVs::DxGeometryAOVs() :
    m_inputAOV(19),
    m_enable(1),
    m_msg(NULL)
{
}

DxGeometryAOVs::~DxGeometryAOVs()
{
}

int
DxGeometryAOVs::Init(RixContext &ctx, RtUString const pluginpath)
{
    PIXAR_ARGUSED(pluginpath);

    m_msg = (RixMessages*)ctx.GetRixInterface(k_RixMessages);
    if (!m_msg)
    {
        return 1;
    }

    return 0;
}

enum paramId
{
    k_resultAOV=0,  // int output
    k_inputAOV,     // int input
    k_enable
};

enum geoAovs
{
    k_Nn=0,
    k_MV,
    k_numAovs
};

RixSCParamInfo const *
DxGeometryAOVs::GetParamTable()
{
    static RixSCParamInfo s_ptable[] =
    {
        // outputs
        RixSCParamInfo(RtUString("resultAOV"), k_RixSCInteger, k_RixSCOutput),

        // inputs
        RixSCParamInfo(RtUString("inputAOV"), k_RixSCInteger),
        RixSCParamInfo(RtUString("enable"),   k_RixSCInteger),

        RixSCParamInfo(),
    };
    return &s_ptable[0];
}

void
DxGeometryAOVs::Finalize(RixContext &ctx)
{
    PIXAR_ARGUSED(ctx);
}

void
DxGeometryAOVs::CreateInstanceData(
    RixContext& ctx, RtUString const handle, RixParameterList const* plist, InstanceData* idata
)
{
    PIXAR_ARGUSED(ctx);
    PIXAR_ARGUSED(handle);
    PIXAR_ARGUSED(plist);

    idata->synchronizeHints = RixShadingPlugin::SynchronizeHints::k_All;
}

void
DxGeometryAOVs::initializeInstanceData(RixContext& ctx, RtUString const handle, RixParameterList const* plist, InstanceData* idata)
{
    PIXAR_ARGUSED(handle);
    PIXAR_ARGUSED(plist);

    if (idata->data && idata->freefunc)
    {
        (idata->freefunc)(idata->data);
    }
    idata->data = nullptr;
    idata->paramtable = nullptr;
    idata->datalen = 0;
    idata->freefunc = nullptr;

    RixRenderState *rstate;
    rstate = (RixRenderState *) ctx.GetRixInterface(k_RixRenderState);
    RixRenderState::FrameInfo finfo;
    rstate->GetFrameInfo(&finfo);
    RixIntegratorEnvironment *integEnv;
    integEnv = (RixIntegratorEnvironment*) finfo.integratorEnv;

    if (!integEnv) return;

    RtInt numDisplays = integEnv->numDisplays;
    RixDisplayChannel const* displayChannels = integEnv->displays;

    size_t datalen = sizeof(RtInt) * MAX_NUM_AOVS;
    RtInt *idlist  = (RtInt*)malloc(datalen);
    for (unsigned i=0; i < MAX_NUM_AOVS; i++)
        idlist[i] = -1;

    RtInt idx(-1);
    for (unsigned dc=0; dc < numDisplays; ++dc)
    {
        for (int n=0; n < k_numAovs; n++)
        {
            if (strcmp(displayChannels[dc].channel.CStr(), s_aovNames[n].CStr()) == 0)
            {
                idlist[n] = displayChannels[dc].id;
                ++idx;
                break;
            }
        }
    }

    if (idx < 0)
    {
        free(idlist);
        return;
    }

    idata->data = idlist;
    idata->datalen = datalen;
    idata->freefunc = free;

    return;
}

void
DxGeometryAOVs::SynchronizeInstanceData(
    RixContext& rixCtx, RtUString const handle, RixParameterList const* instanceParams, uint32_t editHints, InstanceData* instanceData
)
{
    PIXAR_ARGUSED(editHints);
    assert(instanceData);

    initializeInstanceData(rixCtx, handle, instanceParams, instanceData);

    return;
}

int
DxGeometryAOVs::ComputeOutputParams(
    RixShadingContext const *sctx, RtInt *noutputs, OutputSpec **outputs, RtPointer instanceData, RixSCParamInfo const *ignored
)
{
    PIXAR_ARGUSED(ignored);

    if (!sctx->scTraits.primaryHit || !sctx->scTraits.eyePath || sctx->scTraits.shadingMode != k_RixSCScatterQuery)
        return 1;

    RtInt const *idlist = (RtInt const *) instanceData;

    if (!idlist) return 1;

    // evaluate input parameters
    {
        RtInt const *inputAOVPtr;
        sctx->EvalParam(k_inputAOV, -1, &inputAOVPtr, &m_inputAOV, false);
    }

    RtInt const *enablePtr;
    sctx->EvalParam(k_enable, -1, &enablePtr, &m_enable, false);
    RtInt const enable(*enablePtr);
    if (!enable) return 1;

    RixShadingContext::Allocator pool(sctx);

    RixSCParamInfo const* paramTable = GetParamTable();
    int numOutputs = -1;
    while (paramTable[++numOutputs].access == k_RixSCOutput) {}

    OutputSpec* out = pool.AllocForPattern<OutputSpec>(numOutputs);
    *outputs = out;
    *noutputs= numOutputs;
    RtInt* resultAOV = pool.AllocForPattern<RtInt>(1);
    out[0].paramId = k_resultAOV;
    out[0].detail  = k_RixSCUniform;
    out[0].value   = (RtPointer) resultAOV;

    resultAOV[0] = 19;

    RixDisplayServices *dispSvc = sctx->GetDisplayServices();

    RixRenderState *rstate;
    rstate = (RixRenderState *) sctx->GetRixInterface(k_RixRenderState);

    RtFloat3 *aovValue = NULL;


    for (RtInt d=0; d<k_numAovs; d++)
    {
        if (idlist[d] > -1)
        {
            if (!aovValue)
                aovValue = pool.AllocForPattern<RtFloat3>(sctx->numPts);

            switch (d) {
                case k_Nn:
                {
                    // geometry check
                    RtInt hairNormal = 0;
                    RtFloat const *hairNormal_pvr;
                    if (sctx->GetPrimVar(RtUString("useHairNormal"), 0.0f, &hairNormal_pvr) != k_RixSCInvalidDetail)
                    {
                        hairNormal = round(hairNormal_pvr[0]);
                    }

                    RtNormal3 const *tmpN;
                    if (hairNormal == 1) {
                        sctx->GetBuiltinVar(RixShadingContext::k_Tn, &tmpN);
                        memcpy(aovValue, tmpN, sctx->numPts*sizeof(RtFloat3));
                    } else {
                        sctx->GetBuiltinVar(RixShadingContext::k_Nn, &tmpN);
                        memcpy(aovValue, tmpN, sctx->numPts*sizeof(RtFloat3));
                        sctx->Transform(RixShadingContext::k_AsNormals, Rix::k_current, Rix::k_camera, const_cast<RtFloat3*>(aovValue));
                    }

                    for (int i=0; i<sctx->numPts; i++)
                    {
                        dispSvc->Write(idlist[d], sctx->integratorCtxIndex[i], (RtColorRGB)aovValue[i]);
                    }
                    break;
                }
                case k_MV:
                {
                    RtVector3 const *dPdtime;
                    sctx->GetBuiltinVar(RixShadingContext::k_dPdtime, &dPdtime);
                    RtFloat3 const *tmpP;
                    sctx->GetBuiltinVar(RixShadingContext::k_P, &tmpP);

                    RtPoint3 *rasterP = pool.AllocForPattern<RtPoint3>(sctx->numPts);
                    memcpy(rasterP, tmpP, sctx->numPts*sizeof(RtFloat3));
                    sctx->Transform(RixShadingContext::k_AsPoints, Rix::k_current, Rix::k_raster, rasterP);

                    RtVector3 *rasterPdPdtime = pool.AllocForPattern<RtVector3>(sctx->numPts);
                    for (int i=0; i < sctx->numPts; i++) {
                        rasterPdPdtime[i].x = tmpP[i].x + dPdtime[i].x;
                        rasterPdPdtime[i].y = tmpP[i].y + dPdtime[i].y;
                        rasterPdPdtime[i].z = tmpP[i].z + dPdtime[i].z;
                    }
                    sctx->Transform(RixShadingContext::k_AsVectors, Rix::k_current, Rix::k_raster, rasterPdPdtime);

                    for (int i=0; i < sctx->numPts; i++) {
                        aovValue[i].x = rasterPdPdtime[i].x - rasterP[i].x;
                        aovValue[i].y = (rasterPdPdtime[i].y - rasterP[i].y) * -1;
                        aovValue[i].z = rasterPdPdtime[i].z - rasterP[i].z;

                        dispSvc->Write(idlist[d], sctx->integratorCtxIndex[i], (RtColorRGB)aovValue[i]);
                    }
                    break;
                }
            }
        }
    }
    return 0;
}

RIX_PATTERNCREATE
{
    PIXAR_ARGUSED(hint);

    return new DxGeometryAOVs();
}

RIX_PATTERNDESTROY
{
    delete ((DxGeometryAOVs*)pattern);
}
