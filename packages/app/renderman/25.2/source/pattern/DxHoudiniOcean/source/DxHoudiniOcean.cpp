
#include "RixPredefinedStrings.hpp"
#include "RixPattern.h"
#include "RixShadingUtils.h"
#include "PxrTextureAtlas.h"
#include "RixColorUtils.h"

#include <cstring>
#include <iostream>
#include <iomanip>

#include <json-c/json.h>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;


class DxHoudiniOcean : public RixPattern
{
    public:

        DxHoudiniOcean();
        virtual ~DxHoudiniOcean();

        virtual int Init(RixContext &, RtUString const pluginpath ) override;
        virtual RixSCParamInfo const *GetParamTable() override;
        virtual void Synchronize(
            RixContext&,
            RixSCSyncMsg,
            RixParameterList const*) override
        {
        }

        virtual void CreateInstanceData(
            RixContext&,
            RtUString const,
            RixParameterList const*, InstanceData*) override;
        virtual void SynchronizeInstanceData(
            RixContext&,
            RtUString const,
            RixParameterList const*,
            uint32_t editHints, InstanceData*) override;

        virtual void Finalize(RixContext &) override;

        virtual int ComputeOutputParams(
            RixShadingContext const *,
            RtInt *n,
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
        void initializeInstanceData(RixContext& ctx, RtUString const handle,
                                    RixParameterList const* plist, InstanceData* idata);

        bool failed = false;

        RtFloat const m_globalScale;
        RtInt const m_frameOffset;
        RtInt const m_enableLoop;
        RtInt const m_texcoord;
        RtFloat const m_angle;
        RtFloat const m_scaleS;
        RtFloat const m_scaleT;

        RixTexture *m_tex;
        RixMessages *m_msg;
};

// Parameter Default Value
DxHoudiniOcean::DxHoudiniOcean() :
    m_globalScale(1.0f),
    m_frameOffset(0),
    m_enableLoop(1),
    m_texcoord(1),
    m_angle(0.0f),
    m_scaleS(1.0f),
    m_scaleT(1.0f),
    m_tex(NULL),
    m_msg(NULL)
{
}

DxHoudiniOcean::~DxHoudiniOcean()
{
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Parameter Table: Inputs & Outputs
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
enum paramId
{
    k_resultRGB = 0,
    k_resultA,
    k_filename,
    k_globalScale,
    k_frameOffset,
    k_enableLoop,
    k_texcoord,
    k_angle,
    k_scaleS,
    k_scaleT,
    k_numParams
};

enum texcoordId
{
    k_textureuv=0,
    k_world,
    k_WPref
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Utility Functions
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct InstData
{
    RtUString filename;
    RtFloat   L;
};

std::vector<std::string> Split(const std::string& s, char seperator)
{
    std::vector<std::string> output;
    std::string::size_type prev_pos = 0, pos = 0;

    while((pos = s.find(seperator, pos)) != std::string::npos)
    {
        std::string substring(s.substr(prev_pos, pos-prev_pos));
        output.push_back(substring);
        prev_pos = ++pos;
    }

    output.push_back(s.substr(prev_pos, pos-prev_pos));

    return output;
}

std::vector<std::string> GetFileList(const std::string d, const std::string ext)
{
    std::vector<std::string> output;

    fs::path directory(d.c_str());
    fs::directory_iterator iter(directory), end;

    for(; iter != end; ++iter)
    {

        if(iter->path().extension() == ext)
        {
            std::stringstream tmp;
            tmp << d << "/" << iter->path().filename();
            // std::cout << d << "/" << iter->path().filename() << std::endl;
            output.push_back(tmp.str());
        }
    }

    return output;
}

std::string MakePadding(int n, int p)
{
    std::string output;
    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(p) << n;

    output = oss.str();
    return output;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DxHoudiniOcean::Init()
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

int DxHoudiniOcean::Init(RixContext &ctx, RtUString pluginpath)
{
    PIXAR_ARGUSED(pluginpath);

    m_tex = (RixTexture*)ctx.GetRixInterface(k_RixTexture);
    m_msg = (RixMessages*)ctx.GetRixInterface(k_RixMessages);

    if (!m_tex || !m_msg)
        return 1;
    else
        return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DxHoudiniOcean::CreateInstanceData()
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DxHoudiniOcean::CreateInstanceData(
    RixContext& ctx, RtUString const handle, RixParameterList const* plist, InstanceData* idata
)
{
    PIXAR_ARGUSED(ctx);
    PIXAR_ARGUSED(handle);
    PIXAR_ARGUSED(plist);

    idata->synchronizeHints = RixShadingPlugin::SynchronizeHints::k_All;
}

void DxHoudiniOcean::initializeInstanceData(
    RixContext& ctx, RtUString const handle, RixParameterList const* params, InstanceData* instance
)
{
    PIXAR_ARGUSED(handle);
    PIXAR_ARGUSED(params);

    if (instance->data && instance->freefunc)
    {
        (instance->freefunc)(instance->data);
    }
    instance->data = nullptr;
    instance->paramtable = nullptr;
    instance->datalen = 0;
    instance->freefunc = nullptr;

    // .spectrum file path & name
    RtUString inputFile;
    params->EvalParam(k_filename, -1, &inputFile);

    struct json_object *jDATA;
    jDATA = json_object_from_file(inputFile.CStr());

    if(jDATA == NULL) {
        m_msg->Error("DxHoudiniOcean - Failed to get the filename.");
        failed = true;
        return;
    }

    std::string filePath;
    {
        std::vector<std::string> tokens;
        tokens = Split(std::string(inputFile.CStr()), '/');
        for(size_t i=1; i<tokens.size()-1; ++i)
        {
            filePath += "/" + tokens[i];
        }
    }

    struct json_object *jpxr, *jval;
    json_object_object_get_ex(jDATA, "pxr", &jpxr);

    // fileName
    json_object_object_get_ex(jpxr, "fileName", &jval);
    std::string fileName = json_object_get_string(jval);

    // gridsize
    json_object_object_get_ex(jpxr, "gridsize", &jval);
    float gridsize = json_object_get_double(jval);
    const float L = gridsize;


    // the current frame number (currentFrame)
    RixRenderState* rs = (RixRenderState*)ctx.GetRixInterface( k_RixRenderState );
    RixRenderState::FrameInfo finfo;
    rs->GetFrameInfo( &finfo );
    const int currentFrame = (int)finfo.frame;

    int frameOffset = 0;
    params->EvalParam(k_frameOffset, -1, &frameOffset);

    int enableLoop = 0;
    params->EvalParam(k_enableLoop, -1, &enableLoop);

    int frameNo = currentFrame + frameOffset;
    if( enableLoop )
    {
        std::vector<std::string> exr_files;
        exr_files = GetFileList( filePath, ".tex" );

        std::sort( exr_files.begin(), exr_files.end() );
        const int num_exr_files = (int)exr_files.size();

        int the_1st_frameNo = 0;
        if(num_exr_files > 0)
        {
            std::vector<std::string> tokens;
            tokens = Split( exr_files[0], '.' );

            the_1st_frameNo = atoi( tokens[tokens.size()-2].c_str() );
            frameNo = ( frameNo - the_1st_frameNo ) % num_exr_files + the_1st_frameNo;
        }
    }

    const std::string paddedFrame = MakePadding( frameNo, 4 );

    std::stringstream ss;
    ss << filePath << "/" << fileName << "." << paddedFrame << ".tex";

    // set instanceData
    InstData iData;

    std::string tmp = ss.str();
    iData.filename = RtUString(tmp.c_str());
    iData.L = L;

    instance->datalen  = sizeof(InstData);
    instance->data     = malloc(instance->datalen);
    instance->freefunc = free;
    *reinterpret_cast<InstData*>(instance->data) = iData;

    return;
}

void DxHoudiniOcean::SynchronizeInstanceData(RixContext& rixCtx, RtUString const handle,
                                               RixParameterList const* instanceParams,
                                               uint32_t editHints, InstanceData* instanceData)
{
    PIXAR_ARGUSED(editHints);
    assert(instanceData);

    initializeInstanceData(rixCtx, handle, instanceParams, instanceData);

    return;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DxHoudiniOcean::GetParamTable()
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

RixSCParamInfo const* DxHoudiniOcean::GetParamTable()
{
    static RixSCParamInfo s_ptable[] =
    {
        // outputs
        RixSCParamInfo(RtUString("resultRGB"), k_RixSCColor, k_RixSCOutput),
        RixSCParamInfo(RtUString("resultA"), k_RixSCFloat, k_RixSCOutput),

        // inputs
        RixSCParamInfo(RtUString("filename"), k_RixSCString),
        RixSCParamInfo(RtUString("globalScale"), k_RixSCFloat),
        RixSCParamInfo(RtUString("frameOffset"), k_RixSCInteger),
        RixSCParamInfo(RtUString("enableLoop"), k_RixSCInteger),
        RixSCParamInfo(RtUString("texcoord"), k_RixSCInteger),
        RixSCParamInfo(RtUString("angle"), k_RixSCFloat),
        RixSCParamInfo(RtUString("scaleS"), k_RixSCFloat),
        RixSCParamInfo(RtUString("scaleT"), k_RixSCFloat),

        RixSCParamInfo() // end of table
    };
    return &s_ptable[0];
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DxHoudiniOcean::Finalize()
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DxHoudiniOcean::Finalize(RixContext& ctx)
{
    // nothing to do
    PIXAR_ARGUSED(ctx);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DxHoudiniOcean::ComputeOutputParams(): The Main Process
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

int DxHoudiniOcean::ComputeOutputParams(
    RixShadingContext const *sctx,
    RtInt *noutputs, OutputSpec **outputs,
    RtPointer instanceData,
    RixSCParamInfo const *instanceTable
)
{
    if( failed ) { return 1; }

    // -------------------------------------------------------------------------
    // params
    RtFloat const* globalScale;
    sctx->EvalParam(k_globalScale, -1, &globalScale, &m_globalScale, false);

    RtInt const *texcoordPtr;
    sctx->EvalParam(k_texcoord, -1, &texcoordPtr, &m_texcoord, false);
    RtInt const texcoord(*texcoordPtr);

    RtFloat const* rotationAngle;
    sctx->EvalParam(k_angle, -1, &rotationAngle, &m_angle, false);

    RtFloat const* scaleInX;
    sctx->EvalParam(k_scaleS, -1, &scaleInX, &m_scaleS, false );

    RtFloat const* scaleInZ;
    sctx->EvalParam(k_scaleT, -1, &scaleInZ, &m_scaleT, false );

    // get instanceData
    InstData const* iData = (InstData*)instanceData;

    // -------------------------------------------------------------------------
    // outputs
    RixSCType type;
    RixSCConnectionInfo cinfo;

    // Find the number of outputs
    RixSCParamInfo const* paramTable = GetParamTable();
    int numOutputs = -1;
    while (paramTable[++numOutputs].access == k_RixSCOutput) {}

    // Allocate and bind outpus
    RixShadingContext::Allocator pool(sctx);
    OutputSpec* out = pool.AllocForPattern<OutputSpec>(numOutputs);
    *outputs = out;
    *noutputs = numOutputs;

    // looping through the different output ids
    for( RtInt i=0; i<numOutputs; ++i )
    {
        out[i].paramId = i;
        out[i].detail  = k_RixSCInvalidDetail;
        out[i].value   = NULL;

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

    RtColorRGB* resultRGB = (RtColorRGB*)out[k_resultRGB].value;
    if(!resultRGB)
    {
        // make sure the resultRGB space is allocated because it
        // will store the composite color results.
        resultRGB = pool.AllocForPattern<RtColorRGB>(sctx->numPts);
    }

    RtFloat* resultA = (RtFloat*)out[k_resultA].value;
    if(!resultA)
    {
        resultA = pool.AllocForPattern<RtFloat>(sctx->numPts);
    }

    // Do the actual texture map lookup
    PxrLinearizeMode linearize = PxrLinearizeMode(0);
    RixTexture::TxAtlasStyle const atlasStyle = RixTexture::TxAtlasStyle(0);
    // AtlasNone=0,
    // AtlasUdim,
    // AtlasUvTile1,
    // AtlasUvTile0

    RixTexture::TxParams txParams;

    txParams.firstchannel = 0;
    txParams.invertT = 0;
    txParams.filter = RixTexture::TxParams::FilterType(6); // Gaussian
    // > FilterTypes
    // Nearest = 0,  // *WARNING* disables MIPMAP! No filtering
    // Box,          // Fast "box" like filter
    // Bilinear,     // Slower, supports derivatives
    // Bspline,      // Slower bicubic filter, supports derivatives
    // Mitchell,     // Negative lobed, fast, bicubic filter - PTEX ONLY
    // Catmullrom,   // Negative lobed, fast, bicubic filter - PTEX ONLY
    // Gaussian,     // Gaussian fast filter
    // Lagrangian,   // Fastest (SSE based) filter
    // BsplineAniso  // Same as Bspline, but supports anisotropic filtering

    // -------------------------------------------------------------------------
    // compute waveST, waveRadius
    RtPoint3 const *Q = NULL;
    RtFloat2 *waveST = pool.AllocForPattern<RtFloat2>(sctx->numPts);
    RtFloat  *waveRadius = pool.AllocForPattern<RtFloat>(sctx->numPts);
    memset(waveRadius, 0, sctx->numPts*sizeof(RtFloat));

    if(texcoord == k_textureuv)
    {
        RtFloat2 const defaultST(0.0f, 0.0f);
        RtFloat2 const *st = NULL;
        RtFloat const *QRadius = NULL;
        sctx->GetPrimVar(Rix::k_st, defaultST, &st, &QRadius);

        memcpy(waveST, st, sctx->numPts * sizeof(RtFloat2));
        memcpy(waveRadius, QRadius, sctx->numPts * sizeof(RtFloat));
    }
    else
    {
        RtPoint3 *wP = pool.AllocForPattern<RtPoint3>(sctx->numPts);
        RtFloat3 const *pv3;
        RtFloat const *pv3Width;
        if(texcoord == k_world)
        {
            sctx->GetBuiltinVar(RixShadingContext::k_Po, &pv3);
            sctx->GetBuiltinVar(RixShadingContext::k_PRadius, &pv3Width);

            memcpy(wP, pv3, sctx->numPts * sizeof(RtPoint3));
            memcpy(waveRadius, pv3Width, sctx->numPts * sizeof(RtFloat));

            sctx->Transform(RixShadingContext::k_AsPoints, Rix::k_current, Rix::k_world, wP, waveRadius);
        }
        else
        {
            RtFloat3 fill(0, 0, 0);
            if(sctx->GetPrimVar(RtUString("__WPref"), fill, &pv3, &pv3Width) != k_RixSCInvalidDetail)
            {
                memcpy(wP, pv3, sctx->numPts * sizeof(RtPoint3));
                memcpy(waveRadius, pv3Width, sctx->numPts * sizeof(RtFloat));
            }
        }

        for(int i=0; i < sctx->numPts; ++i)
        {
            waveST[i].x = (wP[i].x / iData->L * *globalScale) - 0.5;
            waveST[i].y = (wP[i].z / iData->L * *globalScale) - 0.5;
            waveST[i].y *= -1.0;
            waveRadius[i] = RixMin(waveRadius[i] / iData->L * *globalScale, waveRadius[i] / iData->L * *globalScale);
        }
    }

    // for rotation
    float sn=0.f, cs=0.f;
    RixSinCos(-RixDegreesToRadians(*rotationAngle), &sn, &cs);

    for(unsigned i=0; i < sctx->numPts; ++i)
    {
        // map scale
        waveST[i].x *= (*scaleInX + 1e-6f);
        waveST[i].y *= (*scaleInZ + 1e-6f);
        waveRadius[i] = RixMin(waveRadius[i] * (*scaleInX + 1e-6f), waveRadius[i] * (*scaleInZ + 1e-6f));

        // map rotation
        if(*rotationAngle > 1e-6f) {
            float rx = (waveST[i].x * cs) - (waveST[i].y * sn);
            float ry = (waveST[i].x * sn) + (waveST[i].y * cs);
            waveST[i].x = rx;
            waveST[i].y = ry;
        }
    }


    PxrReadTexture rtex(m_tex, iData->filename, atlasStyle, linearize);
    int err = rtex.Texture(txParams, sctx->numPts, waveST, Q, waveRadius,
                           resultRGB, resultA); // results

    // Handle failed lookup
    if (err == RixTexture::FileNotFound)
    {
       if ((atlasStyle == RixTexture::AtlasNone) && !iData->filename.Empty())
       {
           m_msg->Error("DxHoudiniOcean could not open \"%s\"", iData->filename.CStr());
           return 1;
       }

       RtColorRGB missingColor = RtColorRGB(0, 0, 0);
       RtFloat    missingAlpha = 1.0;

       rtex.FillMissingTexture(sctx->numPts, &missingColor, resultRGB);
       rtex.FillMissingTexture(sctx->numPts, &missingAlpha, resultA);
    }

    // globalScale
    for (unsigned i=0; i<sctx->numPts; i++)
    {
        resultRGB[i] = resultRGB[i] * (1 / *globalScale);
    }


    // for(unsigned i=0; i<sctx->numPts; ++i)
    // {
    //     RtVector3 displacement = RtVector3(resultRGB[i]);
    //
    //     // displacement vector
    //     if(resultRGB)
    //     {
    //         resultRGB[i].r = displacement.x;    // * (*horizontalScale);
    //         resultRGB[i].g = displacement.y;    // * (*verticalScale);
    //         resultRGB[i].b = displacement.z;    // * (*horizontalScale);
    //     }
    //
    // }

    return 0;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Creator & Destroyer
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

RIX_PATTERNCREATE
{
    PIXAR_ARGUSED(hint);

    return new DxHoudiniOcean();
}

RIX_PATTERNDESTROY
{
    delete ( (DxHoudiniOcean*)pattern );
}
