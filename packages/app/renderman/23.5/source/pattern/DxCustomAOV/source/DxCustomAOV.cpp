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
#include <string.h>

class DxCustomAOV : public RixPattern
{
    public:

        DxCustomAOV();
        virtual ~DxCustomAOV();

        virtual int Init(RixContext &, RtUString const pluginpath) override;
        virtual RixSCParamInfo const *GetParamTable() override;
        virtual void Synchronize(
            RixContext&, RixSCSyncMsg, RixParameterList const*
        ) override
        {
        }

        virtual void CreateInstanceData(
            RixContext& ctx, RtUString const handle, RixParameterList const* parms, InstanceData *instance
        ) override;

        virtual void SynchronizeInstanceData(
            RixContext&, RtUString const, RixParameterList const*, uint32_t editHints, InstanceData*
        ) override;

        virtual void Finalize(RixContext &) override;

        virtual int ComputeOutputParams(
            RixShadingContext const *,
            RtInt *n, OutputSpec **outputs,
            RtPointer instanceData, RixSCParamInfo const* instanceTable
        ) override;

        virtual bool Bake2dOutput(
            RixBakeContext const*, Bake2dSpec&, RtPointer
        ) override
        {
            return false;
        }

        virtual bool Bake3dOutput(
            RixBakeContext const*, Bake3dSpec&, RtPointer
        ) override
        {
            return false;
        }

    private:
        static void initializeInstanceData(RixContext& ctx, RtUString const handle, RixParameterList const* parms, InstanceData* instance);

        RtInt       m_inputAOV;
        RtInt       m_enable;
        RtColorRGB  m_aovColor;
        RtString    m_aovName;

        RixMessages *m_msg;
};

DxCustomAOV::DxCustomAOV() :
    m_inputAOV(19),
    m_enable(1),
    m_aovColor(RtColorRGB(1.f,1.f,1.f)),
    m_aovName(NULL),
    m_msg(NULL)
{
}

DxCustomAOV::~DxCustomAOV()
{
}

int
DxCustomAOV::Init(RixContext &ctx, RtUString const pluginpath)
{
    PIXAR_ARGUSED(pluginpath);

    m_msg = (RixMessages*)ctx.GetRixInterface(k_RixMessages);
    if( !m_msg ) {
        return 1;
    }
    return 0;
}

enum paramID
{
    k_resultAOV=0,  // int output
    k_inputAOV,     // int input
    k_enable,
    k_aovColor,
    k_aovName,
    k_numParams
};

const static RtUString US_INPUTAOV("inputAOV");
const static RtUString US_ENABLE("enable");
const static RtUString US_AOVCOLOR("aovColor");
const static RtUString US_AOVNAME("aovName");

RixSCParamInfo const *
DxCustomAOV::GetParamTable()
{
    static RixSCParamInfo s_ptable[] =
    {
        // outputs
        RixSCParamInfo(RtUString("resultAOV"), k_RixSCInteger, k_RixSCOutput),

        // inputs
        RixSCParamInfo(US_INPUTAOV, k_RixSCInteger),
        RixSCParamInfo(US_ENABLE, k_RixSCInteger),
        RixSCParamInfo(US_AOVCOLOR, k_RixSCColor),
        RixSCParamInfo(US_AOVNAME, k_RixSCString),

        RixSCParamInfo()
    };
    return s_ptable;
}

void
DxCustomAOV::Finalize(RixContext &ctx)
{
    PIXAR_ARGUSED(ctx);
}

void
DxCustomAOV::CreateInstanceData(
    RixContext& ctx, RtUString const handle, RixParameterList const* plist, InstanceData* idata
)
{
    PIXAR_ARGUSED(ctx);
    PIXAR_ARGUSED(handle);
    PIXAR_ARGUSED(plist);

    idata->synchronizeHints = RixShadingPlugin::SynchronizeHints::k_All;
}

void
DxCustomAOV::initializeInstanceData(
    RixContext& ctx, RtUString const handle, RixParameterList const* parms, InstanceData* instance
)
{
    if (instance->data && instance->freefunc)
    {
        (instance->freefunc)(instance->data);
    }
    instance->data = nullptr;
    instance->paramtable = nullptr;
    instance->datalen = 0;
    instance->freefunc = nullptr;

    RixRenderState *state = reinterpret_cast<RixRenderState *>(ctx.GetRixInterface(k_RixRenderState));
    RixMessages *msg = reinterpret_cast<RixMessages *>(ctx.GetRixInterface(k_RixMessages));

    RixRenderState::FrameInfo frame;
    state->GetFrameInfo(&frame);
    RixIntegratorEnvironment const *env = reinterpret_cast<RixIntegratorEnvironment const *>(frame.integratorEnv);

    RtInt paramId;
    RixSCType typ;
    RixSCConnectionInfo cnx;

    RtUString aovName;
    if (parms->GetParamId(US_AOVNAME, &paramId))
    {
        return;
    }
    parms->EvalParam(paramId, 0, &aovName);

    if (env)
    {
        for (int index=0; index < env->numDisplays; ++index)
        {
            if (env->displays[index].channel != aovName) continue;
            instance->datalen = sizeof(RixChannelId);
            instance->data = malloc(instance->datalen);
            instance->freefunc = free;
            *reinterpret_cast<RixChannelId*>(instance->data) = env->displays[index].id;
        }
    }
    if (!instance->data)
    {
        return;
    }
    return;
}

void
DxCustomAOV::SynchronizeInstanceData(
    RixContext& rixCtx, RtUString const handle, RixParameterList const* instanceParams, uint32_t editHints, InstanceData* instanceData
)
{
    PIXAR_ARGUSED(editHints);
    assert(instanceData);

    initializeInstanceData(rixCtx, handle, instanceParams, instanceData);
    return;
}

int
DxCustomAOV::ComputeOutputParams(
    RixShadingContext const *sctx, RtInt *noutputs, OutputSpec **outputs,
    RtPointer instanceData, RixSCParamInfo const *instanceTable
)
{
    // Only execute on camera hits
    if (!sctx->scTraits.primaryHit || !sctx->scTraits.eyePath || sctx->scTraits.shadingMode != k_RixSCScatterQuery)
        return 1;

    RixSCType type;
    RixSCConnectionInfo cInfo;

    RtColorRGB const *aovColor;
    sctx->EvalParam(k_aovColor, -1, &aovColor, &m_aovColor, true);

    // evaluate input parameters
    {
        RtInt const *inputAOVPtr;
        sctx->EvalParam(k_inputAOV, -1, &inputAOVPtr, &m_inputAOV, false);
    }

    RtInt const *enablePtr;
    sctx->EvalParam(k_enable, -1, &enablePtr, &m_enable, false);
    RtInt const enable(*enablePtr);
    if (!enable) return 1;

    if (!instanceData) return 1;

    // Allocate and bind our outputs
    RixShadingContext::Allocator pool(sctx);
    OutputSpec *spec = pool.AllocForPattern<OutputSpec>(1);

    RtInt* resultAOV = pool.AllocForPattern<RtInt>(1);
    spec[0].paramId = k_resultAOV;
    spec[0].detail  = k_RixSCUniform;
    spec[0].value   = (RtPointer) resultAOV;
    *noutputs = 1;
    *outputs  = spec;

    RixDisplayServices *dspySvc = sctx->GetDisplayServices();
    for (int index=0; index < sctx->numPts; ++index )
    {
        dspySvc->Write(*reinterpret_cast<RixChannelId const *>(instanceData), sctx->integratorCtxIndex[index], aovColor[index]);
    }

    return 0;
}

RIX_PATTERNCREATE
{
    PIXAR_ARGUSED(hint);

    return new DxCustomAOV();
}

RIX_PATTERNDESTROY
{
    delete ((DxCustomAOV*)pattern);
}
