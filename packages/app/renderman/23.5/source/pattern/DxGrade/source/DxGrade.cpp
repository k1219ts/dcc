/*  $Date: 2018/08/13 $  $Revision: #1 $
# ------------------------------------------------------------------------------
#
#   Dexter
#
#       sanghun.kim
#
# ------------------------------------------------------------------------------
*/
#include "RixPattern.h"
#include "RixShadingUtils.h"

class DxGrade : public RixPattern
{
    public:
        DxGrade();
        virtual ~DxGrade();

        virtual int Init(RixContext &, RtUString const pluginpath) override;
        virtual RixSCParamInfo const *GetParamTable() override;
        virtual void Synchronize(
            RixContext&, RixSCSyncMsg, RixParameterList const*
        ) override
        {
        }

        virtual void Finalize(RixContext &) override;

        virtual int ComputeOutputParams(
            RixShadingContext const *,
            RtInt *n, OutputSpec **outputs,
            RtPointer instanceData, RixSCParamInfo const*
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
        RtColorRGB  m_inputRGB;
        RtColorRGB  m_gainColor;
        RtFloat     m_multiply;
        RtFloat     m_gamma;
        RtFloat     m_saturation;
        RtFloat     m_offset;
        RtInt       m_clampOutput;
        RtFloat     m_outputMin;
        RtFloat     m_outputMax;
        RixMessages *m_msg;
};

DxGrade::DxGrade() :
    m_inputRGB( 0.0f, 0.0f, 0.0f ),
    m_gainColor( 1.0f, 1.0f, 1.0f ),
    m_multiply( 1.0f ),
    m_gamma( 1.0f ),
    m_saturation( 1.0f ),
    m_offset( 0.0f ),
    m_clampOutput( 0 ),
    m_outputMin( 0.0f ),
    m_outputMax( 1.0f ),
    m_msg(NULL)
{
}

DxGrade::~DxGrade()
{
}

int
DxGrade::Init(RixContext &ctx, RtUString const pluginpath)
{
    PIXAR_ARGUSED(pluginpath);

    m_msg = (RixMessages*)ctx.GetRixInterface(k_RixMessages);
    if(!m_msg)
        return 1;
    return 0;
}

enum paramId
{
    // Outputs
    k_resultRGB=0,
    k_resultR,
    k_resultG,
    k_resultB,

    // Inputs
    k_inputRGB,
    k_gainColor,
    k_multiply,
    k_gamma,
    k_saturation,
    k_offset,
    k_clampOutput,
    k_outputMin,
    k_outputMax,
    k_numParams
};

RixSCParamInfo const *
DxGrade::GetParamTable()
{
    static RixSCParamInfo s_ptable[] =
    {
        // outputs
        RixSCParamInfo(RtUString("resultRGB"), k_RixSCColor, k_RixSCOutput),
        RixSCParamInfo(RtUString("resultR"),   k_RixSCFloat, k_RixSCOutput),
        RixSCParamInfo(RtUString("resultG"),   k_RixSCFloat, k_RixSCOutput),
        RixSCParamInfo(RtUString("resultB"),   k_RixSCFloat, k_RixSCOutput),

        // inputs
        RixSCParamInfo(RtUString("inputRGB"), k_RixSCColor),
        RixSCParamInfo(RtUString("gainColor"), k_RixSCColor),
        RixSCParamInfo(RtUString("multiply"), k_RixSCFloat),
        RixSCParamInfo(RtUString("gamma"), k_RixSCFloat),
        RixSCParamInfo(RtUString("saturation"), k_RixSCFloat),
        RixSCParamInfo(RtUString("offset"), k_RixSCFloat),
        RixSCParamInfo(RtUString("clampOutput"), k_RixSCInteger),
        RixSCParamInfo(RtUString("outputMin"), k_RixSCFloat),
        RixSCParamInfo(RtUString("outputMax"), k_RixSCFloat),

        RixSCParamInfo(),
    };
    return &s_ptable[0];
}

void
DxGrade::Finalize(RixContext &ctx)
{
    PIXAR_ARGUSED(ctx);
}

RtColorRGB
saturationCompute(RtColorRGB input, RtFloat value)
{
    RtColorRGB result;
    RtFloat lum = 0.2125f * input.r + 0.7154 * input.g + 0.0721 * input.b;
    result = RixMix( RtColorRGB(lum), input, value );
    return result;
}

int
DxGrade::ComputeOutputParams(
    RixShadingContext const *sctx, RtInt *noutputs, OutputSpec **outputs,
    RtPointer instanceData, RixSCParamInfo const *ignored
)
{
    PIXAR_ARGUSED(instanceData);
    PIXAR_ARGUSED(ignored);

    RixSCType type;
    RixSCConnectionInfo cinfo;
    bool varying = true;
    bool uniform = false;

    // Input =======================================================
    RtColorRGB const *inputC;
    sctx->EvalParam( k_inputRGB, -1, &inputC, &m_inputRGB, varying );

    RtColorRGB const *gainC;
    sctx->EvalParam( k_gainColor, -1, &gainC, &m_gainColor, varying );

    RtFloat const *multiplyPtr;
    sctx->EvalParam( k_multiply, -1, &multiplyPtr, &m_multiply, uniform );
    RtFloat const multiply(*multiplyPtr);

    RtFloat const *gamma;
    sctx->EvalParam( k_gamma, -1, &gamma, &m_gamma, varying );

    RtFloat const *saturationPtr;
    sctx->EvalParam( k_saturation, -1, &saturationPtr, &m_saturation, uniform );
    RtFloat const saturation(*saturationPtr);

    RtFloat const *offsetPtr;
    sctx->EvalParam( k_offset, -1, &offsetPtr, &m_offset, uniform );
    RtFloat const offset(*offsetPtr);

    RtInt const *clampOutputPtr;
    sctx->EvalParam( k_clampOutput, -1, &clampOutputPtr, &m_clampOutput, uniform );
    RtInt const clampOutput(*clampOutputPtr);

    RtFloat const *outputMinPtr;
    sctx->EvalParam( k_outputMin, -1, &outputMinPtr, &m_outputMin, uniform );
    RtFloat const outputMin(*outputMinPtr);

    RtFloat const *outputMaxPtr;
    sctx->EvalParam( k_outputMax, -1, &outputMaxPtr, &m_outputMax, uniform );
    RtFloat const outputMax(*outputMaxPtr);

    // Find the number of outputs
    RixSCParamInfo const* paramTable = GetParamTable();
    int numOutputs = -1;
    while (paramTable[++numOutputs].access == k_RixSCOutput) {}

    // Allocate and bind our outputs
    RixShadingContext::Allocator pool(sctx);
    OutputSpec* out = pool.AllocForPattern<OutputSpec>(numOutputs);
    *outputs = out;
    *noutputs = numOutputs;

    // looping through the different output ids
    for( int i=0; i<numOutputs; ++i )
    {
        out[i].paramId = i;
        out[i].detail = k_RixSCInvalidDetail;
        out[i].value = NULL;

        type = paramTable[i].type;

        sctx->GetParamInfo( i, &type, &cinfo );
        if( cinfo == k_RixSCNetworkValue )
        {
            if( type == k_RixSCColor )
            {
                out[i].detail = k_RixSCVarying;
                out[i].value = pool.AllocForPattern<RtColorRGB>(sctx->numPts);
            }
            else if( type == k_RixSCFloat )
            {
                out[i].detail = k_RixSCVarying;
                out[i].value = pool.AllocForPattern<RtFloat>(sctx->numPts);
            }
        }
    }

    RtColorRGB* resultRGB = (RtColorRGB*) out[k_resultRGB].value;
    if( !resultRGB )
    {
        resultRGB = pool.AllocForPattern<RtColorRGB>(sctx->numPts);
    }
    RtFloat* resultR = (RtFloat*) out[k_resultR].value;
    RtFloat* resultG = (RtFloat*) out[k_resultG].value;
    RtFloat* resultB = (RtFloat*) out[k_resultB].value;

    RtFloat invg = 0.0f;
    if( sctx->GetParamInfo( k_gamma, &type, &cinfo ) == 0 )
    {
        if( cinfo != k_RixSCNetworkValue )
            invg = 1.0f/RixMax( 1e-4f, gamma[0] );
    }

    RtInt cg = 0;
    if( sctx->GetParamInfo( k_gainColor, &type, &cinfo ) == 0 )
    {
        if( cinfo != k_RixSCNetworkValue )  // if not connected
        {
            if( gainC[0].r != 1.0f || gainC[0].g != 1.0f || gainC[0].b != 1.0f )
                cg = 1;
        }
        else
        {
            cg = 1;
        }
    }

    for( int i=0; i<sctx->numPts; i++ )
    {
        if( inputC[i].IsBlack() )
        {
            resultRGB[i].r = inputC[i].r;
            resultRGB[i].g = inputC[i].g;
            resultRGB[i].b = inputC[i].b;
        }
        else
        {
            resultRGB[i].r = inputC[i].r;
            resultRGB[i].g = inputC[i].g;
            resultRGB[i].b = inputC[i].b;

            // gain compute
            if( cg )
            {
                resultRGB[i].r *= gainC[i].r;
                resultRGB[i].g *= gainC[i].g;
                resultRGB[i].b *= gainC[i].b;
            }

            // multiply compute
            if( multiply != 1.0f )
            {
                resultRGB[i].r *= multiply;
                resultRGB[i].g *= multiply;
                resultRGB[i].b *= multiply;
            }

            if( invg == 0.0f )
                invg = 1.0f/RixMax(1e-4f, gamma[i]);
            // gamma compute
            if( invg != 1.0f )
            {
                resultRGB[i].r = powf( resultRGB[i].r, invg );
                resultRGB[i].g = powf( resultRGB[i].g, invg );
                resultRGB[i].b = powf( resultRGB[i].b, invg );
            }

            // saturation compute
            if( saturation != 1.0f )
            {
                resultRGB[i] = saturationCompute( resultRGB[i], saturation );
            }
        }

        // offset compute
        if( offset != 0.0f )
        {
            resultRGB[i].r += offset;
            resultRGB[i].g += offset;
            resultRGB[i].b += offset;
        }

        // clamp compute
        if( clampOutput )
        {
            resultRGB[i].r = RixClamp( resultRGB[i].r, outputMin, outputMax );
            resultRGB[i].g = RixClamp( resultRGB[i].g, outputMin, outputMax );
            resultRGB[i].b = RixClamp( resultRGB[i].b, outputMin, outputMax );
        }

        if( resultR )
        {
            resultR[i] = resultRGB[i].r;
        }
        if( resultG )
        {
            resultG[i] = resultRGB[i].g;
        }
        if( resultB )
        {
            resultB[i] = resultRGB[i].b;
        }
    }

    return 0;
}

RIX_PATTERNCREATE
{
    PIXAR_ARGUSED(hint);

    return new DxGrade();
}

RIX_PATTERNDESTROY
{
    delete ((DxGrade*)pattern);
}
