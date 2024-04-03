#include <string>
#include <iostream>
#include <sys/stat.h>
using namespace std;

#include <RixPredefinedStrings.hpp>
#include <RixPattern.h>
#include <RixShadingUtils.h>

#include "DxImageFile.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Class Declaration
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
class DxImageFile : public RixPattern
{
public:

    DxImageFile();
    virtual ~DxImageFile();

    virtual int Init(RixContext &, RtUString const pluginpath);
    virtual RixSCParamInfo const *GetParamTable();
    virtual void Synchronize(
        RixContext&, RixSCSyncMsg, RixParameterList const*
    )
    {
    }

    virtual int CreateInstanceData(
        RixContext&, RtUString const, RixParameterList const*, InstanceData*
    );

    virtual void Finalize(RixContext &);

    virtual int ComputeOutputParams(
        RixShadingContext const *, RtInt *noutputs, OutputSpec **outputs,
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
    RtInt m_linearize;
    bool failed = false;
    RixTexture *m_tex;
    RixMessages *m_msg {nullptr};
};

DxImageFile::DxImageFile() :
    m_linearize(1),
    m_tex(NULL),
    m_msg(NULL)
{
}

DxImageFile::~DxImageFile()
{
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DxImageFile::Init()
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

int DxImageFile::Init( RixContext& ctx, RtUString const pluginpath )
{
    // RixMessages* msg = (RixMessages*)ctx.GetRixInterface( k_RixMessages );
    // if( !msg ) { return 1; }
    m_msg = (RixMessages*)ctx.GetRixInterface(k_RixMessages);
    m_tex = (RixTexture*)ctx.GetRixInterface( k_RixTexture );
    // if( !m_tex ) { return 1; }
    //
    // return 0;
    if (!m_tex || !m_msg)
        return 1;
    else
        return 0;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Parameter Table: Inputs & Outputs
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum ParameterIDs
{
    k_resultRGB = 0,    // color output
    k_resultA,          // float output

    // Inputs
    k_filename,
    k_linearize,
    k_manifold,
    k_manifoldQ,
    k_manifoldQradius,
    k_manifoldEnd,
    k_numParams
};

RixSCParamInfo const* DxImageFile::GetParamTable()
{
    static RixSCParamInfo s_ptable[] =
    {
        // outputs
        RixSCParamInfo( RtUString("resultRGB"), k_RixSCColor, k_RixSCOutput ),
        RixSCParamInfo( RtUString("resultA"),   k_RixSCFloat, k_RixSCOutput ),

        // inputs
        RixSCParamInfo(RtUString("filename"), k_RixSCString),
        RixSCParamInfo(RtUString("linearize"), k_RixSCInteger),

        RixSCParamInfo(RtUString("PxrManifold"), RtUString("manifold"), k_RixSCStructBegin),
            RixSCParamInfo(RtUString("Q"), k_RixSCPoint),
            RixSCParamInfo(RtUString("Qradius"), k_RixSCFloat),
        RixSCParamInfo(RtUString("PxrManifold"), RtUString("manifold"), k_RixSCStructEnd),

        RixSCParamInfo() // end of table
    };

    return &s_ptable[0];
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DxImageFile::Finalize()
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DxImageFile::Finalize( RixContext& ctx )
{
    // nothing to do
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DxImageFile::CreateInstanceData()
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

static void CleanUpFunc( void* imageData )
{
    Image* image = (Image*)imageData;
    delete image;
    image = NULL;
}

int DxImageFile::CreateInstanceData( RixContext& ctx, RtUString handle, RixParameterList const* params, InstanceData* instance )
{
    RtUString filename;
    params->EvalParam( k_filename, -1, &filename );
    const string inputFileStr = filename.CStr();

    // RixRenderState* rs = (RixRenderState*)ctx.GetRixInterface( k_RixRenderState );
    // RixRenderState::FrameInfo finfo;
    // rs->GetFrameInfo( &finfo );
    // const int currentFrame = (int)finfo.frame;
    RixRenderState* rstate = (RixRenderState*)ctx.GetRixInterface(k_RixRenderState);
    RixRenderState::Type attrType;
    RtInt count;
    RtInt errAttr;

    RtUString txPath;
    errAttr = rstate->GetAttribute(RtUString("user:txPath"), &txPath, sizeof(RtUString), &attrType, &count);
    m_msg->Warning("[txPath] : %s", txPath.CStr());
    // const string txPathStr = txPath->CStr();

    Image* image = new Image;

    // if (image->load(txPathStr.c_str()))
    if( image->load( inputFileStr.c_str() ) )
    {

        failed = false;
    }
    else
    {
        cout << "[ImageFile] Failed to open file: " << inputFileStr << endl;
        failed = true;
    }

    instance->datalen  = sizeof(Image*);
    instance->data     = (void*)image;
    instance->freefunc = CleanUpFunc;

    return 0;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DxImageFile::ComputeOutputParams(): The Main Process
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

int DxImageFile::ComputeOutputParams(
    RixShadingContext const* sctx,
    RtInt*                   howManyOutputs,
    OutputSpec**             outputSpec,
    RtPointer                instanceData,
    RixSCParamInfo const*    instanceTable
)
{
    if( failed ) { return 1; }

    RixRenderState::Type attrType;
    RtInt count;
    RixRenderState* rstate = (RixRenderState*)sctx->GetRixInterface(k_RixRenderState);
    RtInt errAttr;

    // Input
    RtInt const* ival;

    sctx->EvalParam(k_linearize, -1, &ival, &m_linearize, false);
    PxrLinearizeMode linearize = PxrLinearizeMode(*ival);

    RtInt acescg = 0;
    errAttr = rstate->GetOption(RtUString("user:ACEScg"), (void*)&acescg, sizeof(RtInt), &attrType, &count);

    // RtUString txPath;
    // errAttr = rstate->GetAttribute(RtUString("user:txPath"), &txPath, sizeof(RtUString), &attrType, &count);
    // m_msg->Warning("[txPath] : %s", txPath.CStr());

    RixSCType type;
    RixSCConnectionInfo cinfo;

    const RtInt numOutputs = NumOutputs( GetParamTable() );

    const RtInt numShadedPoints = sctx->numPts;
    if( numShadedPoints <= 0 ) { return 1; }

    RixShadingContext::Allocator pool( sctx );

    OutputSpec* outputData = Malloc<OutputSpec>( pool, numOutputs );
    *outputSpec = outputData;
    *howManyOutputs = numOutputs;

    for( RtInt i=0; i<numOutputs; ++i )
    {
        outputData[i].paramId = i;
        outputData[i].detail  = k_RixSCInvalidDetail;
        outputData[i].value   = NULL;

        sctx->GetParamInfo( i, &type, &cinfo );

        if( cinfo == k_RixSCNetworkValue )
        {
            if( type == k_RixSCColor )
            {
                outputData[i].detail = k_RixSCVarying;
                outputData[i].value = Malloc<RtColorRGB>( pool, numShadedPoints );
            }
            else if( type == k_RixSCFloat )
            {
                outputData[i].detail = k_RixSCVarying;
                outputData[i].value = Malloc<RtFloat>( pool, numShadedPoints );
            }
        }
    }

    RtColorRGB* resultRGB = (RtColorRGB*)outputData[k_resultRGB].value;
    if( !resultRGB ) { resultRGB = Malloc<RtColorRGB>( pool, numShadedPoints ); }

    RtFloat* resultA = (RtFloat*)outputData[k_resultA].value;
    if( !resultA ) { resultA = Malloc<RtFloat>( pool, numShadedPoints ); }

    // Either st, or Q, will be non-NULL depending on a connected manifold.
    // RtFloat2 const* st = NULL;
    RtFloat2* st = pool.AllocForPattern<RtFloat2>(sctx->numPts);
    RtFloat3 const* Q  = NULL;
    RtFloat const* QRadius = NULL;

    // check for manifold input
    sctx->GetParamInfo(k_manifold, &type, &cinfo);
    if(cinfo != k_RixSCNetworkValue)
    {
        RtFloat2 const* stIn, defaultST(0.0f, 0.0f);
        sctx->GetPrimVar(Rix::k_st, defaultST, &stIn, &QRadius);
        for (int i=0; i<sctx->numPts; ++i)
        {
            st[i].x = stIn[i].x;
            st[i].y = 1.0f - stIn[i].y;
        }
    }
    else
    {
        sctx->EvalParam(k_manifoldQ, -1, &Q);
        sctx->EvalParam(k_manifoldQradius, -1, &QRadius);
    }

    RtFloat *stRadius = pool.AllocForPattern<RtFloat>(sctx->numPts);
    memcpy(stRadius, QRadius, sizeof(RtFloat) *sctx->numPts);

    // const Image* image = (Image*)instanceData;
    DxReadTexture dxtex((Image*)instanceData, linearize);
    int err = dxtex.Texture(sctx->numPts, st, resultRGB, resultA);

    if (linearize == k_linearizeEnabled || linearize == k_linearizeAutomatic)
    {
        if (acescg == 1) {
            for (int i=0; i<sctx->numPts; i++)
            {
                resultRGB[i] = linRec709ToLinAP1(resultRGB[i]);
            }
        }
    }


    return 0;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Creator & Destroyer
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

RIX_PATTERNCREATE
{
    return new DxImageFile();
}

RIX_PATTERNDESTROY
{
    delete ( (DxImageFile*)pattern );
}
