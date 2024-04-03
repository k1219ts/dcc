/* $Revision: #1 $ */

#include "RixPredefinedStrings.hpp"
#include "RixPattern.h"
#include "RixShadingUtils.h"
#include "PxrTextureAtlas.h"
#include "RixColorUtils.h"
#include <cstring>

inline RtColorRGB linRec709ToLinAP1(const RtColorRGB c)
{
    const static RtMatrix4x4 rec709toACEScg(0.610277f ,  0.0688436f , 0.0241673f, 0.0f,
                                            0.345424f ,  0.934974f  , 0.121814f , 0.0f,
                                            0.0443001f, -0.00381805f, 0.854019f , 0.0f,
                                            0.0f,        0.0f,        0.0f,       1.0f);

    // convert rec709 primaries to ACES AP1
    RtVector3 dir = rec709toACEScg.vTransform(RtVector3(c.r, c.g, c.b));

    return RtColorRGB(dir.x, dir.y, dir.z);
}

class DxTexture : public RixPattern
{
public:

    DxTexture();
    virtual ~DxTexture();

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
    void GetFilename(RixShadingContext const*, RtInt &, std::string &);

    RtInt const m_firstChannel;
    RtInt const m_computeMode;
    RtInt const m_txVarNum;
    RtInt const m_atlasStyle;
    RtInt const m_invertT;
    RtInt const m_filter;
    RtInt const m_lerp;
    RtColorRGB const m_missingColor;
    RtFloat const m_missingAlpha;
    RtInt const m_linearize;
    RtInt const m_mipBias;
    RtFloat const m_maxResolution;
    RtInt const m_optimizeIndirect;
    RtColorRGB const m_colorScale;
    RtColorRGB const m_colorOffset;
    RtFloat const m_alphaScale;
    RtFloat const m_alphaOffset;
    RtFloat const m_saturation;

    RixTexture *m_tex;
    RixMessages *m_msg;
};

DxTexture::DxTexture() :
      m_firstChannel(0),
      m_computeMode(1),
      m_txVarNum(1),
      m_atlasStyle(RixTexture::AtlasNone),
      m_invertT(1),
      m_filter(RixTexture::TxParams::Box),
      m_lerp(1),
      m_missingColor(RtColorRGB(1.f,0.f,1.f)),
      m_missingAlpha(1.f),
      m_linearize(1),
      m_mipBias(0),
      m_maxResolution(0.f),
      m_optimizeIndirect(1),
      m_colorScale(1.f, 1.f, 1.f),
      m_colorOffset(0.f, 0.f, 0.f),
      m_alphaScale(1.f),
      m_alphaOffset(0.f),
      m_saturation(1.f),
      m_tex(NULL),
      m_msg(NULL)
{
}

DxTexture::~DxTexture()
{
}

int
DxTexture::Init(RixContext &ctx, RtUString pluginpath)
{
    PIXAR_ARGUSED(pluginpath);

    m_tex = (RixTexture*)ctx.GetRixInterface(k_RixTexture);
    m_msg = (RixMessages*)ctx.GetRixInterface(k_RixMessages);

    if (!m_tex || !m_msg)
        return 1;
    else
        return 0;
}

enum paramId
{
    k_resultRGB = 0, // color output
    k_resultR,       // float output
    k_resultG,       // float output
    k_resultB,       // float output
    k_resultA,       // float output

    // Inputs
    k_computeMode,
    k_txChannel,
    k_txVarNum,

    // Inputs
    k_filename,
    k_firstChannel,
    k_atlasStyle,
    k_invertT,
    k_filter,
    k_blur,
    k_lerp,
    k_missingColor,
    k_missingAlpha,
    k_linearize,
    k_mipBias,
    k_maxResolution,
    k_optimizeIndirect,
    k_colorScale,
    k_colorOffset,
    k_alphaScale,
    k_alphaOffset,
    k_saturation,
    k_manifold,
    k_manifoldQ,
    k_manifoldQradius,
    k_manifoldEnd,
    k_numParams
};

void
DxTexture::GetFilename(
    RixShadingContext const *sctx, RtInt &atlas, std::string &filename
)
{
    RtInt const* computeMode;
    sctx->EvalParam(k_computeMode, -1, &computeMode, &m_computeMode, false);
    RtInt const* txVarNumEnabled;
    sctx->EvalParam(k_txVarNum, -1, &txVarNumEnabled, &m_txVarNum, false);
    if (*computeMode == 1)
    {
        RtUString const* txBasePath_pvr = NULL;
        RtUString const* txLayerName_pvr= NULL;
        RtUString const* txVersion_pvr  = NULL;
        RixSCDetail txBasePath_det = sctx->GetPrimVar(RtUString("txBasePath"), &txBasePath_pvr);
        RixSCDetail txVersion_det  = sctx->GetPrimVar(RtUString("txVersion"), &txVersion_pvr);
        if (txVersion_det == k_RixSCInvalidDetail) {
            RixRenderState *rstate = (RixRenderState *)sctx->GetRixInterface(k_RixRenderState);
            RixShadingContext::Allocator pool(sctx);
            RixRenderState::Type type;
            char **ver = pool.AllocForPattern<char*>(1);
            RtInt count= 0;
            RtInt errVer = rstate->GetAttribute(RtUString("user:txVersion"), (void*)ver, sizeof(char*), &type, &count);
            if (!errVer) {
                // m_msg->Warning("version %s, %d", *ver, errVer);
                // 1.
                // const static RtUString txver(*ver);
                // txVersion_pvr = &txver;
                // 2.
                txVersion_pvr = (RtUString const*)(ver);
                txVersion_det = k_RixSCVarying;
            }
        }
        RixSCDetail txLayerName_det= sctx->GetPrimVar(RtUString("txLayerName"), &txLayerName_pvr);
        if (txBasePath_det != k_RixSCInvalidDetail) {
            // node parameter - txChannel
            RtUString const* txChannel = NULL;
            sctx->EvalParam(k_txChannel, -1, &txChannel);
            // atlas primvar
            RtFloat const *txmultiUV_pvr;
            if (sctx->GetPrimVar(RtUString("txmultiUV"), 0.0f, &txmultiUV_pvr) != k_RixSCInvalidDetail) {
                atlas = round(txmultiUV_pvr[0]);
            }

            // case 1. - txLayerName ( with txVersion )
            if (txLayerName_det != k_RixSCInvalidDetail) {
                filename = txBasePath_pvr->CStr() + std::string("/tex/");
                if (txVersion_det != k_RixSCInvalidDetail) {
                    filename += txVersion_pvr->CStr() + std::string("/");
                }
                filename += txLayerName_pvr->CStr();
            } else {
            // case 2. - not txLayerName (it's vendor data)
                filename = txBasePath_pvr->CStr();
            }

            if (txChannel) {
                filename += std::string("_") + txChannel->CStr();
            }

            // texture variation
            if (*txVarNumEnabled == 1)
            {
                RtFloat const *txVarNum_pvr;
                if (sctx->GetPrimVar(RtUString("txVarNum"), 0.0f, &txVarNum_pvr) != k_RixSCInvalidDetail) {
                    RtInt txVarNum = round(txVarNum_pvr[0]);
                    if (txVarNum > 0) {
                        char intStr[16];
                        std::sprintf(intStr, "%d", txVarNum);
                        filename += std::string("_") + std::string(intStr);
                    }
                }
            }

            if (atlas == 1) {
                filename += std::string("._MAPID_");
            }
            filename += std::string(".tex");
        } else {
            m_msg->Error("Not found texture primvar : txBasePath");
        }
    }
    else
    {
        RtUString const* inputfile = NULL;
        sctx->EvalParam(k_filename, -1, &inputfile);
        if (inputfile) filename = inputfile->CStr();
    }
}

RixSCParamInfo const *
DxTexture::GetParamTable()
{
    static RixSCParamInfo s_ptable[] =
    {
        // outputs
        RixSCParamInfo(RtUString("resultRGB"), k_RixSCColor, k_RixSCOutput),
        RixSCParamInfo(RtUString("resultR"),  k_RixSCFloat, k_RixSCOutput),
        RixSCParamInfo(RtUString("resultG"),  k_RixSCFloat, k_RixSCOutput),
        RixSCParamInfo(RtUString("resultB"),  k_RixSCFloat, k_RixSCOutput),
        RixSCParamInfo(RtUString("resultA"),  k_RixSCFloat, k_RixSCOutput),

        // inputs
        RixSCParamInfo(RtUString("computeMode"), k_RixSCInteger),
        RixSCParamInfo(RtUString("txChannel"), k_RixSCString),
        RixSCParamInfo(RtUString("txVarNum"), k_RixSCInteger),

        // inputs
        RixSCParamInfo(RtUString("filename"), k_RixSCString),
        RixSCParamInfo(RtUString("firstChannel"), k_RixSCInteger),
        RixSCParamInfo(RtUString("atlasStyle"), k_RixSCInteger),
        RixSCParamInfo(RtUString("invertT"), k_RixSCInteger),
        RixSCParamInfo(RtUString("filter"), k_RixSCInteger),
        RixSCParamInfo(RtUString("blur"), k_RixSCFloat),
        RixSCParamInfo(RtUString("lerp"), k_RixSCInteger),
        RixSCParamInfo(RtUString("missingColor"), k_RixSCColor),
        RixSCParamInfo(RtUString("missingAlpha"), k_RixSCFloat),
        RixSCParamInfo(RtUString("linearize"), k_RixSCInteger),
        RixSCParamInfo(RtUString("mipBias"), k_RixSCInteger),
        RixSCParamInfo(RtUString("maxResolution"), k_RixSCFloat),
        RixSCParamInfo(RtUString("optimizeIndirect"), k_RixSCInteger),

        RixSCParamInfo(RtUString("colorScale"), k_RixSCColor),
        RixSCParamInfo(RtUString("colorOffset"), k_RixSCColor),
        RixSCParamInfo(RtUString("alphaScale"), k_RixSCFloat),
        RixSCParamInfo(RtUString("alphaOffset"), k_RixSCFloat),
        RixSCParamInfo(RtUString("saturation"), k_RixSCFloat),

        RixSCParamInfo(RtUString("PxrManifold"), RtUString("manifold"), k_RixSCStructBegin),
            RixSCParamInfo(RtUString("Q"), k_RixSCPoint),
            RixSCParamInfo(RtUString("Qradius"), k_RixSCFloat),
        RixSCParamInfo(RtUString("PxrManifold"), RtUString("manifold"), k_RixSCStructEnd),

            RixSCParamInfo() // end of table
        };
    return &s_ptable[0];
}

void
DxTexture::Finalize(RixContext &ctx)
{
    PIXAR_ARGUSED(ctx);
}

int
DxTexture::ComputeOutputParams(
    RixShadingContext const *sctx,
    RtInt *noutputs, OutputSpec **outputs,
    RtPointer instanceData,
    RixSCParamInfo const *ignored
)
{
    PIXAR_ARGUSED(instanceData);
    PIXAR_ARGUSED(ignored);

    bool indirectHit = !sctx->scTraits.primaryHit || !sctx->scTraits.eyePath ||
                       sctx->scTraits.shadingMode != k_RixSCScatterQuery;

    // Input ==================================================
    RtInt atlas_pvr = 0;
    RtUString const* filename = NULL;
    std::string filename_pvr  = std::string("");
    GetFilename(sctx, atlas_pvr, filename_pvr);
    RtUString const fn = RtUString(filename_pvr.c_str());
    filename = &fn;

    // Get our Rix interface for looking up attributes
    RixRenderState::Type attrType;
    RtInt count;
    RixRenderState* rstate = (RixRenderState*)sctx->GetRixInterface(k_RixRenderState);
    RtInt errAttr;  // 0 ok, -1 error
    // aces color workflow
    RtInt acescg = 0;
    errAttr = rstate->GetOption(RtUString("user:ACEScg"), (void*)&acescg, sizeof(RtInt), &attrType, &count);

    RtInt const* ival;
    RixTexture::TxParams txParams;
    sctx->EvalParam(k_firstChannel, -1, &ival, &m_firstChannel, false);
    txParams.firstchannel = *ival;

    sctx->EvalParam(k_atlasStyle, -1, &ival, &m_atlasStyle, false);
    RtInt const* atlas;
    if (atlas_pvr == 1) {
        atlas = &atlas_pvr;
    } else {
        atlas = ival;
    }
    RixTexture::TxAtlasStyle const atlasStyle = RixTexture::TxAtlasStyle(*atlas);

    sctx->EvalParam(k_invertT, -1, &ival, &m_invertT, false);
    txParams.invertT = *ival;

    sctx->EvalParam(k_filter, -1, &ival, &m_filter, false);
    txParams.filter = RixTexture::TxParams::FilterType(*ival);

    bool blurVarying;
    RtFloat const *blur;
    blurVarying = (sctx->EvalParam(k_blur, -1, &blur, NULL, true) == k_RixSCVarying);
    txParams.sblurVarying = txParams.tblurVarying = blurVarying;
    txParams.sblur = txParams.tblur = blur;

    sctx->EvalParam(k_lerp, -1, &ival, &m_lerp, false);
    txParams.lerp = ((*ival) != 0);

    // 0 - off, 1 - on, 2 - auto
    sctx->EvalParam(k_linearize, -1, &ival, &m_linearize, false);
    PxrLinearizeMode linearize = PxrLinearizeMode(*ival);

    RtInt mipBias = PxrGetMipBias(sctx, k_mipBias, m_mipBias);
    RtInt maxResolution = PxrGetMaxResolution(sctx, k_maxResolution, m_maxResolution);
    RtInt const *optimizeIndirect;
    sctx->EvalParam(k_optimizeIndirect, -1, &optimizeIndirect, &m_optimizeIndirect, false);

    // opt-in for now.
    if (optimizeIndirect[0] && indirectHit)
    {
        txParams.filter = RixTexture::TxParams::FilterType::Box;
    }

    RtColorRGB const *colorScale;
    sctx->EvalParam(k_colorScale, -1, &colorScale, &m_colorScale, true);
    RtColorRGB const *colorOffset;
    sctx->EvalParam(k_colorOffset, -1, &colorOffset, &m_colorOffset, true);
    RtFloat const *alphaScale;
    sctx->EvalParam(k_alphaScale, -1, &alphaScale, &m_alphaScale, true);
    RtFloat const *alphaOffset;
    sctx->EvalParam(k_alphaOffset, -1, &alphaOffset, &m_alphaOffset, true);
    RtFloat const *saturation;
    sctx->EvalParam(k_saturation, -1, &saturation, &m_saturation, true);

    // Output ==================================================
    RixSCType type;
    RixSCConnectionInfo cinfo;

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
    for (int i = 0; i < numOutputs; ++i)
    {
        out[i].paramId = i;
        out[i].detail = k_RixSCInvalidDetail;
        out[i].value = NULL;

        type = paramTable[i].type;    // we know this

        sctx->GetParamInfo(i, &type, &cinfo);
        if(cinfo == k_RixSCNetworkValue)
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
    if(!resultRGB)
    {
        // make sure the resultRGB space is allocated because it
        // will store the composite color results.
        resultRGB = pool.AllocForPattern<RtColorRGB>(sctx->numPts);
    }
    RtFloat* resultR = (RtFloat*) out[k_resultR].value;
    RtFloat* resultG = (RtFloat*) out[k_resultG].value;
    RtFloat* resultB = (RtFloat*) out[k_resultB].value;
    RtFloat* resultA = (RtFloat*) out[k_resultA].value;

    // Either st, or Q, will be non-NULL depending on a connected manifold.
    RtFloat2 const* st = NULL;
    RtPoint3 const* Q = NULL;
    RtFloat const* QRadius = NULL;

    // check for manifold input
    sctx->GetParamInfo(k_manifold, &type, &cinfo);
    if(cinfo != k_RixSCNetworkValue)
    {
        RtFloat2 const defaultST(0.0f, 0.0f);
        sctx->GetPrimVar(Rix::k_st, defaultST, &st, &QRadius);

        // Alternatively: call four-deriv version of GetPrimVar() to
        // get dsdu etc and then do the mapping from du,dv to ds,dt.
        // This would take stretching and/or rotation in s and t into
        // account.  However, the results of such calls are not
        // currently cached, so even though the results would be
        // better, we cannot afford the performance degradation.
        // (OSL has a separate cache that handles this, and also the
        // proper mapping from du,dv to ds,dt.)
        //sctx->GetPrimVar(Rix::k_st, &st, &dsdu, &dtdu, &dsdv, &dtdv);
        //ds[i] = dsdu[i] * du[i] + dsdv[i] * dv[i];
        //dt[i] = dtdu[i] * du[i] + dtdv[i] * dv[i];

        if (atlasStyle != RixTexture::AtlasNone) txParams.invertT = true;
    }
    else
    {
        sctx->EvalParam(k_manifoldQ, -1, &Q);
        sctx->EvalParam(k_manifoldQradius, -1, &QRadius);
        // We don't invert manifolds (since the upstream node can invert)
        txParams.invertT = false;
    }

    // Optionally clamp the filter width to prevent the texture system
    // from accessing too detailed mip levels
    RtFloat *stRadius;
    if (mipBias != 0 || maxResolution > 0.f)
    {
        stRadius = pool.AllocForPattern<RtFloat>(sctx->numPts);
        memcpy(stRadius, QRadius, sizeof(RtFloat) * sctx->numPts);
        PxrTxMipControls(sctx, mipBias, maxResolution, stRadius);
    }
    else
    {
        stRadius = const_cast<RtFloat*>(QRadius);
    }

    // Do the actual texture map lookup
    PxrReadTexture rtex(m_tex, *filename, atlasStyle, linearize);
    int err = rtex.Texture(txParams, sctx->numPts, st, Q, stRadius,
                           resultRGB, resultA); // results

    // Handle failed lookup
    if (err == RixTexture::FileNotFound)
    {
        if ((atlasStyle == RixTexture::AtlasNone) && !filename->Empty())
            m_msg->Error("DxTexFile could not open \"%s\"", filename->CStr());

        RtColorRGB const* missingColor;
        RtFloat const* missingAlpha;
        sctx->EvalParam(k_missingColor, -1, &missingColor, &m_missingColor, true);
        sctx->EvalParam(k_missingAlpha, -1, &missingAlpha, &m_missingAlpha, true);

        rtex.FillMissingTexture(sctx->numPts, missingColor, resultRGB);
        rtex.FillMissingTexture(sctx->numPts, missingAlpha, resultA);
    }

    if (linearize == k_linearizeEnabled || linearize == k_linearizeAutomatic)
    {
        if (acescg == 1) {
            for (int i=0; i<sctx->numPts; i++)
            {
                resultRGB[i] = linRec709ToLinAP1(resultRGB[i]);
            }
        }
    }

    // Reorder the outputs
    PxrInterleavedToPlanar(sctx->numPts, resultRGB, resultR, resultG, resultB);

    for (unsigned i=0; i<sctx->numPts; i++)
    {
        resultRGB[i] = resultRGB[i] * colorScale[i] + colorOffset[i];
        RixSaturation(resultRGB[i], saturation[i]);
        if (resultA) resultA[i] = resultA[i] * alphaScale[i] + alphaOffset[i];
        if (resultR) resultR[i] = resultRGB[i].r;
        if (resultG) resultG[i] = resultRGB[i].g;
        if (resultB) resultB[i] = resultRGB[i].b;
    }

    return 0;
}


RIX_PATTERNCREATE
{
    PIXAR_ARGUSED(hint);

    return new DxTexture();
}


RIX_PATTERNDESTROY
{
    delete ((DxTexture*)pattern);
}
