/*
Dexter $Date: 2018/12/01 $  $Revision: #1 $
# ------------------------------------------------------------------------------
#
#   Dexter
#
#   AOV : MotionVector -> MV, object_id -> oID, group_id -> gID
#
# ------------------------------------------------------------------------------
*/
#include "RixPredefinedStrings.hpp"
#include "RixPattern.h"
#include "RixShadingUtils.h"
#include "RixIntegrator.h"

#include <cstdio>
#include <cstring>

struct displayData {
    RtInt       id;
    RtUString   name;
};

class DxObjectAOV : public RixPattern
{
    public:

        DxObjectAOV();
        virtual ~DxObjectAOV();

        virtual int Init(RixContext &, RtUString pluginpath);
        virtual RixSCParamInfo const *GetParamTable();
        virtual void Synchronize(
            RixContext&, RixSCSyncMsg, RixParameterList const*
        )
        {
        }

        virtual int CreateInstanceData(
            RixContext &ctx, RtUString const handle, RixParameterList const *plist, InstanceData *idata
        );

        virtual void Finalize(RixContext &);

        virtual int ComputeOutputParams(
            RixShadingContext const *,
            RtInt *n, RixPattern::OutputSpec **outputs,
            RtPointer instanceData, RixSCParamInfo const *
        );

        virtual bool Bake2dOutput(
            RixBakeContext const*, Bake2dSpec&, RtPointer
        )
        {
            return false;
        }

        virtual bool Bake3dOutput(
            RixBakeContext const*, Bake3dSpec&, RtPointer
        )
        {
            return false;
        }

    private:
        RtInt m_inputAOV;
        RtInt m_enable;
        RtInt m_dispSize;
        RixMessages *m_msg;
        bool starts_with(const char* str, const char* prefix);
};

bool
DxObjectAOV::starts_with(const char* str, const char* prefix)
{
    while(*prefix)
    {
        if(*prefix++ != *str++)
            return false;
    }
    return true;
}

DxObjectAOV::DxObjectAOV() :
    m_inputAOV(19),
    m_enable(1),
    m_dispSize(1),
    m_msg(NULL)
{
}

DxObjectAOV::~DxObjectAOV()
{
}

int
DxObjectAOV::Init(RixContext &ctx, RtUString const pluginpath)
{
    m_msg = (RixMessages*)ctx.GetRixInterface(k_RixMessages);
    if( !m_msg ) {
        return 1;
    }
    return 0;
}

enum paramId
{
    k_resultAOV=0,  // int output
    k_inputAOV,     // int input
    k_enable,
    k_numParams
};

RixSCParamInfo const *
DxObjectAOV::GetParamTable()
{
    static RixSCParamInfo s_ptable[] =
    {
        // outputs
        RixSCParamInfo(RtUString("resultAOV"), k_RixSCInteger, k_RixSCOutput),

        // inputs
        RixSCParamInfo(RtUString("inputAOV"), k_RixSCInteger),
        RixSCParamInfo(RtUString("enable"), k_RixSCInteger),

        RixSCParamInfo()
    };
    return &s_ptable[0];
}

void
DxObjectAOV::Finalize(RixContext &ctx)
{
}

int
DxObjectAOV::CreateInstanceData(
    RixContext &ctx, RtUString const handle, RixParameterList const *plist, InstanceData *idata
)
{
    RixRenderState *rstate;
    rstate = (RixRenderState *) ctx.GetRixInterface(k_RixRenderState);
    RixRenderState::FrameInfo finfo;
    rstate->GetFrameInfo(&finfo);
    RixIntegratorEnvironment *integEnv;
    integEnv = (RixIntegratorEnvironment*) finfo.integratorEnv;

    // Get displays
    m_dispSize = integEnv->numDisplays;
    RixDisplayChannel const* displayChannels = integEnv->displays;

    size_t datalen = sizeof(displayData)*m_dispSize;
    displayData *disp = (displayData*)malloc(datalen);

    for( int dc=0; dc<m_dispSize; dc++ )
    {
        disp[dc].id     = displayChannels[dc].id;
        disp[dc].name   = displayChannels[dc].channel;
    }

    idata->data     = disp;
    idata->datalen  = datalen;
    idata->freefunc = free;

    return 0;
}

int
DxObjectAOV::ComputeOutputParams(
    RixShadingContext const *sctx, RtInt *noutputs, OutputSpec **outputs,
    RtPointer instanceData, RixSCParamInfo const *ignored
)
{
    if( !(sctx->scTraits.primaryHit && sctx->scTraits.eyePath) )
        return 1;

    displayData* disp = (displayData*)instanceData;

    bool uniform = false;
    RixSCType type;
    RixSCConnectionInfo cinfo;

    // Allocate and bind our outputs
    RixShadingContext::Allocator pool(sctx);

    // Input ==================================================
    // evaluate input parameters
    {
        RtInt const *inputAOVPtr;
        sctx->EvalParam( k_inputAOV, -1, &inputAOVPtr, &m_inputAOV, uniform );
    }

    RtInt const *enablePtr;
    sctx->EvalParam( k_enable, -1, &enablePtr, &m_enable, uniform );
    RtInt const enable( *enablePtr );

    if (!enable) return 1;

    // Get the render state to read attributes.
    RixRenderState::Type attrType;
    RtInt count;
    RixRenderState *rstate = (RixRenderState*) sctx->GetRixInterface(k_RixRenderState);
    RtInt errAttr; // 0 ok, -1 error

    // Find the number of outputs
    RixSCParamInfo const* paramTable = GetParamTable();
    int numOutputs = -1;
    while( paramTable[++numOutputs].access == k_RixSCOutput ) {}

    // Output ==================================================
    OutputSpec* out = pool.AllocForPattern<OutputSpec>(numOutputs);
    *outputs = out;
    *noutputs = numOutputs;
    RtInt* resultAOV = pool.AllocForPattern<RtInt>(1);

    out[0].paramId = k_resultAOV;
    out[0].detail = k_RixSCUniform;
    out[0].value = (RtPointer) resultAOV;

    // Uniform output
    resultAOV[0] = 19;

    // Get the display services to write out the AOV.
    RixDisplayServices *dispSvc = sctx->GetDisplayServices();

    for( int d=0; d<m_dispSize; d++ )
    {
        // motion vector
        if( strcmp("MV", disp[d].name.CStr()) == 0 ) {
            RtVector3 const *dPdtime;
            sctx->GetBuiltinVar( RixShadingContext::k_dPdtime, &dPdtime );

            RtFloat3 const *tmpP;
            sctx->GetBuiltinVar( RixShadingContext::k_P, &tmpP );

            RtPoint3 *rasterP = pool.AllocForPattern<RtPoint3>(sctx->numPts);
            memcpy( rasterP, tmpP, sctx->numPts*sizeof(RtFloat3) );
            sctx->Transform(RixShadingContext::k_AsPoints, Rix::k_current, Rix::k_raster, rasterP, NULL);

            RtVector3 *rasterPdPdtime = pool.AllocForPattern<RtVector3>(sctx->numPts);
            for( int i=0; i<sctx->numPts; i++ ) {
                rasterPdPdtime[i].x = tmpP[i].x + dPdtime[i].x;
                rasterPdPdtime[i].y = tmpP[i].y + dPdtime[i].y;
                rasterPdPdtime[i].z = tmpP[i].z + dPdtime[i].z;
            }
            sctx->Transform(RixShadingContext::k_AsVectors, Rix::k_current, Rix::k_raster, rasterPdPdtime, NULL);

            RtColorRGB *pixelMoved = pool.AllocForPattern<RtColorRGB>(sctx->numPts);
            for( int i=0; i<sctx->numPts; i++ ) {
                pixelMoved[i].r = rasterPdPdtime[i].x - rasterP[i].x;
                pixelMoved[i].g = (rasterPdPdtime[i].y - rasterP[i].y) * -1;
                pixelMoved[i].b = rasterPdPdtime[i].z - rasterP[i].z;

                dispSvc->Write( disp[d].id, sctx->integratorCtxIndex[i], (RtColorRGB)pixelMoved[i] );
            }
        }
    }

    return 0;
}

RIX_PATTERNCREATE
{
    return new DxObjectAOV();
}

RIX_PATTERNDESTROY
{
    delete ((DxObjectAOV*)pattern);
}
