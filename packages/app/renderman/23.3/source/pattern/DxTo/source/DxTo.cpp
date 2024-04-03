#include "RixPattern.h"
#include "RixShadingUtils.h"

class DxTo : public RixPattern
{
public:
    DxTo();
    virtual ~DxTo();

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
        RtPointer instanceData, RixSCParamInfo const *
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
    // Defaults
    RtFloat const m_inputF;
    RtColorRGB const m_inputRGB;
    RtNormal3 const m_inputN;

    RixMessages *m_msg;
};

DxTo::DxTo() :
    m_inputF(0.0f),
    m_inputRGB(0.0f, 0.0f, 0.0f),
    m_inputN(0.0f, 0.0f, 0.0f),
    m_msg(NULL)
{
}

DxTo::~DxTo()
{
}

int
DxTo::Init(RixContext &ctx, RtUString const pluginpath)
{
    PIXAR_ARGUSED(pluginpath);

    m_msg = (RixMessages*)ctx.GetRixInterface(k_RixMessages);
    if (!m_msg)
        return 1;
    return 0;
}
enum paramId
{
    k_resultRGB=0,
    k_resultR,
    k_resultG,
    k_resultB,

    k_inputF,
    k_inputRGB,
    k_inputN,
    k_numParams
};

RixSCParamInfo const *
DxTo::GetParamTable()
{
    static RixSCParamInfo s_ptable[] =
    {
        // outputs
        RixSCParamInfo(RtUString("resultRGB"), k_RixSCColor, k_RixSCOutput),
        RixSCParamInfo(RtUString("resultR"), k_RixSCFloat, k_RixSCOutput),
        RixSCParamInfo(RtUString("resultG"), k_RixSCFloat, k_RixSCOutput),
        RixSCParamInfo(RtUString("resultB"), k_RixSCFloat, k_RixSCOutput),

        // inputs
        RixSCParamInfo(RtUString("inputF"), k_RixSCFloat),
        RixSCParamInfo(RtUString("inputRGB"), k_RixSCColor),
        RixSCParamInfo(RtUString("inputN"), k_RixSCNormal),

        RixSCParamInfo(),
    };
    return &s_ptable[0];
}

void
DxTo::Finalize(RixContext &ctx)
{
    PIXAR_ARGUSED(ctx);
}

int
DxTo::ComputeOutputParams(
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
    RtFloat const *inputF = NULL;
    RtColorRGB const *inputRGB = NULL;
    RtNormal3 const *inputN  = NULL;

    sctx->EvalParam(k_inputF, -1, &inputF, &m_inputF, varying);
    sctx->EvalParam(k_inputRGB, -1, &inputRGB, &m_inputRGB, varying);
    sctx->EvalParam(k_inputN, -1, &inputN, &m_inputN, varying);

    // Find the number of outputs
    RixSCParamInfo const* paramTable = GetParamTable();
    int numOutputs = -1;
    while (paramTable[++numOutputs].access == k_RixSCOutput) {}

    // Allocate and bind our output
    RixShadingContext::Allocator pool(sctx);
    OutputSpec* out = pool.AllocForPattern<OutputSpec>(numOutputs);
    *outputs = out;
    *noutputs= numOutputs;

    for (int i=0; i<numOutputs; ++i)
    {
        out[i].paramId = i;
        out[i].detail  = k_RixSCInvalidDetail;
        out[i].value   = NULL;

        type = paramTable[i].type;

        sctx->GetParamInfo(i, &type, &cinfo);
        if (cinfo == k_RixSCNetworkValue)
        {
            if (type == k_RixSCColor)
            {
                out[i].detail = k_RixSCVarying;
                out[i].value = pool.AllocForPattern<RtColorRGB>(sctx->numPts);
            }
            else if (type == k_RixSCFloat)
            {
                out[i].detail = k_RixSCVarying;
                out[i].value  = pool.AllocForPattern<RtFloat>(sctx->numPts);
            }
        }
    }

    RtColorRGB* resultRGB = (RtColorRGB*) out[k_resultRGB].value;
    if (!resultRGB)
    {
        resultRGB = pool.AllocForPattern<RtColorRGB>(sctx->numPts);
    }
    RtFloat* resultR = (RtFloat*) out[k_resultR].value;
    RtFloat* resultG = (RtFloat*) out[k_resultG].value;
    RtFloat* resultB = (RtFloat*) out[k_resultB].value;

    if (inputN)
    {
        memcpy(resultRGB, inputN, sctx->numPts * sizeof(float) * 3);
    }

    return 0;
}

RIX_PATTERNCREATE
{
    PIXAR_ARGUSED(hint);
    return new DxTo();
}

RIX_PATTERNDESTROY
{
    delete ((DxTo*)pattern);
}
