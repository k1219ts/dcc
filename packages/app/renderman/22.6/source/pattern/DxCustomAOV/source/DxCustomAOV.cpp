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

struct displayData {
    RtInt       id;
    RtUString   name;
};

const static RtUString US_Nn("Nnx");
const static RtUString US_MV("MV");

class DxCustomAOV : public RixPattern
{
    public:

        DxCustomAOV();
        virtual ~DxCustomAOV();

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
        RtInt       m_inputAOV;
        RtInt       m_enable;
        RtColorRGB  m_aovColor;
        RtString    m_aovName;
        RtInt       m_dispSize;
        RixMessages *m_msg;
};

DxCustomAOV::DxCustomAOV() :
    m_inputAOV(19),
    m_enable(1),
    m_aovColor(RtColorRGB(1.f,1.f,1.f)),
    m_aovName(NULL),
    m_dispSize(1),
    m_msg(NULL)
{
}

DxCustomAOV::~DxCustomAOV()
{
}

int
DxCustomAOV::Init(RixContext &ctx, RtUString const pluginpath)
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
    k_aovColor,
    k_aovName,
    k_numParams
};

RixSCParamInfo const *
DxCustomAOV::GetParamTable()
{
    static RixSCParamInfo s_ptable[] =
    {
        // outputs
        RixSCParamInfo(RtUString("resultAOV"), k_RixSCInteger, k_RixSCOutput),

        // inputs
        RixSCParamInfo(RtUString("inputAOV"), k_RixSCInteger),
        RixSCParamInfo(RtUString("enable"), k_RixSCInteger),
        RixSCParamInfo(RtUString("aovColor"), k_RixSCColor),
        RixSCParamInfo(RtUString("aovName"), k_RixSCString),

        RixSCParamInfo()
    };
    return &s_ptable[0];
}

void
DxCustomAOV::Finalize(RixContext &ctx)
{
}

int
DxCustomAOV::CreateInstanceData(
    RixContext &ctx, RtUString const handle, RixParameterList const *plist, InstanceData *idata
)
{
    // m_msg->Info( "InstanceData Start" );
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
        disp[dc].id   = displayChannels[dc].id;
        disp[dc].name = displayChannels[dc].channel;
    }
    idata->data = disp;
    idata->datalen = datalen;
    idata->freefunc = free;
    // m_msg->Info( "InstanceData End" );
    return 0;
}

int
DxCustomAOV::ComputeOutputParams(
    RixShadingContext const *sctx, RtInt *noutputs, OutputSpec **outputs,
    RtPointer instanceData, RixSCParamInfo const *ignored
)
{
    // Only execute on camera hits
    if (!sctx->scTraits.primaryHit || !sctx->scTraits.eyePath || sctx->scTraits.shadingMode != k_RixSCScatterQuery)
        return 1;

    // m_msg->Info( "ComputeOutput Start" );
    displayData* disp = (displayData*)instanceData;

    bool uniform = false;
    RixSCType type;
    RixSCConnectionInfo cinfo;

    // Allocate and bind our outputs
    RixShadingContext::Allocator pool(sctx);

    // Input ==================================================
    RtUString const* aovname = NULL;
    sctx->EvalParam( k_aovName, -1, &aovname );

    RtColorRGB const* aovColor;
    sctx->EvalParam(k_aovColor, -1, &aovColor, &m_aovColor, true);

    // evaluate input parameters
    {
        RtInt const *inputAOVPtr;
        sctx->EvalParam(k_inputAOV, -1, &inputAOVPtr, &m_inputAOV, uniform);
    }

    RtInt const *enablePtr;
    sctx->EvalParam(k_enable, -1, &enablePtr, &m_enable, uniform);
    RtInt const enable(*enablePtr);

    if( !enable )
        return 1;

    // Get the render state to read attributes.
    RixRenderState *rstate;
    rstate = (RixRenderState *) sctx->GetRixInterface( k_RixRenderState );
    RixRenderState::Type attrType;
    RtInt attrCount;

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
        if (aovname)
        {
            // m_msg->Info( "dispaly channel : %s -> %d", disp[d].name, disp[d].id );
            if( strcmp(aovname->CStr(), disp[d].name.CStr()) == 0 )
            {
                // m_msg->Warning( "render channel : %s", disp[d].name );
                for( int i=0; i<sctx->numPts; i++ )
                {
                    dispSvc->Write( disp[d].id, sctx->integratorCtxIndex[i], aovColor[i] );
                }
            }
        }

        // For Denoise normal
        if (strcmp(US_Nn.CStr(), disp[d].name.CStr()) == 0)
        {
            // geometry check
            RtInt hairNormal = 0;
            RtFloat const *hairNormal_pvr;
            if (sctx->GetPrimVar(RtUString("useHairNormal"), 0.0f, &hairNormal_pvr) != k_RixSCInvalidDetail)
            {
                hairNormal = round(hairNormal_pvr[0]);
            }

            RtNormal3 *aovOut = pool.AllocForPattern<RtNormal3>(sctx->numPts);

            RtNormal3 const* ValN;
            if (hairNormal == 1) {
                sctx->GetBuiltinVar(RixShadingContext::k_Tn, &ValN);
                memcpy(aovOut, ValN, sctx->numPts*sizeof(RtFloat3));
            } else {
                sctx->GetBuiltinVar(RixShadingContext::k_Nn, &ValN);
                memcpy(aovOut, ValN, sctx->numPts*sizeof(RtFloat3));
                sctx->Transform(RixShadingContext::k_AsNormals, Rix::k_current, Rix::k_camera, aovOut, NULL);
            }

            for (int i=0; i<sctx->numPts; i++)
            {
                dispSvc->Write(disp[d].id, sctx->integratorCtxIndex[i], (RtColorRGB)aovOut[i]);
            }
        }

        if (strcmp("MV", disp[d].name.CStr()) == 0)
        {
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
    // m_msg->Info( "ComputeOutput End" );
    return 0;
}

RIX_PATTERNCREATE
{
    return new DxCustomAOV();
}

RIX_PATTERNDESTROY
{
    delete ((DxCustomAOV*)pattern);
}
