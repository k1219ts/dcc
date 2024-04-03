
#include "RixPredefinedStrings.hpp"
#include "RixPattern.h"
#include "RixShadingUtils.h"
#include "PxrTextureAtlas.h"
#include "RixColorUtils.h"

// #include <cstring>
// #include <iostream>
#include <string>
#include <iomanip>
#include <sstream>

#include <json-c/json.h>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;


class DxBoraOcean : public RixPattern
{
    public:

        DxBoraOcean() {};

        virtual int Init( RixContext &, RtUString const pluginpath );
        virtual RixSCParamInfo const *GetParamTable();
        virtual void Synchronize(
            RixContext&, RixSCSyncMsg, RixParameterList const*
        )
        {
        }
        virtual int CreateInstanceData(
            RixContext&, RtUString const, RixParameterList const*, InstanceData*
        );
        virtual void Finalize( RixContext& );
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

        bool failed = false;

        RixTexture* m_tex = NULL;
        RixMessages *m_msg;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Creator & Destroyer
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

RIX_PATTERNCREATE
{
    PIXAR_ARGUSED(hint);

    return new DxBoraOcean();
}

RIX_PATTERNDESTROY
{
    delete ( (DxBoraOcean*)pattern );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Parameter Table: Inputs & Outputs
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum ParameterIDs
{
    outputRGB_ID,
    outputA_ID,
    inputFile_ID,
    frameOffset_ID,
    rotationAngle_ID,
    scaleInX_ID,
    scaleInZ_ID,
    verticalScale_ID,
    horizontalScale_ID,
    crestGain_ID,
    crestBias_ID,
    enableLoop_ID,
    k_numParams
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
// DxBoraOcean::Init()
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

int DxBoraOcean::Init(RixContext &ctx, RtUString pluginpath)
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
// DxBoraOcean::CreateInstanceData()
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

int DxBoraOcean::CreateInstanceData(
    RixContext& ctx, RtUString const handle, RixParameterList const* params, InstanceData* instance
)
{
    PIXAR_ARGUSED(handle);
    PIXAR_ARGUSED(params);

    // .oceanParams file path & name
    RtUString inputFile;
    params->EvalParam( inputFile_ID, -1, &inputFile );


    // get json objects
    struct json_object *jobj;
    struct json_object *jstrs, *jints, *jfloats;
    jobj = json_object_from_file(inputFile.CStr());

    if( jobj == NULL )
    {
        m_msg->Error("DxBoraOcean - Failed to get the input file.");
        failed = true;
        return 1;
    }
    else
    {
        if(
            !json_object_object_get_ex(jobj, "int",    &jints) ||
            !json_object_object_get_ex(jobj, "float",  &jfloats) ||
            !json_object_object_get_ex(jobj, "string", &jstrs)
        )
        {
            m_msg->Error("DxBoraOcean - Failed to load json file.");
            failed = true;
            return 1;
        }
    }

    std::string filePath;
    {
        std::vector<std::string> tokens;
        tokens = Split( std::string(inputFile.CStr()), '/');

        for( size_t i=1; i<tokens.size()-1; ++i )
        {
            filePath += "/" + tokens[i];
        }
    }


    // get fileName form json file
    if(!json_object_object_get_ex(jstrs, "fileName", &jobj))
    {
        m_msg->Error("DxBoraOcean - Failed to read 'fileName' in json file.");
        failed = true;
        return 1;
    }
    std::string fileName = json_object_get_string(jobj);

    // get physicalLength form json file
    if(!json_object_object_get_ex(jfloats, "physicalLength", &jobj))
    {
        m_msg->Error("DxBoraOcean - Failed to read 'physicalLength' in json file.");
        failed = true;
        return 1;
    }
    float physicalLength = json_object_get_double(jobj);

    // get sceneConvertingScale form json file
    if(!json_object_object_get_ex(jfloats, "sceneConvertingScale", &jobj))
    {
        m_msg->Error("DxBoraOcean - Failed to read 'sceneConvertingScale' in json file.");
        failed = true;
        return 1;
    }
    float sceneConvertingScale = json_object_get_double(jobj);

    // geometric length
    const float L = physicalLength * sceneConvertingScale;
    // std::cout << ">>>>>>>>>>>>>>>> : " << L << std::endl;

    // the current frame number (currentFrame)
    RixRenderState* rs = (RixRenderState*)ctx.GetRixInterface( k_RixRenderState );
    RixRenderState::FrameInfo finfo;
    rs->GetFrameInfo( &finfo );
    const int currentFrame = (int)finfo.frame;

    int frameOffset = 0;
    params->EvalParam( frameOffset_ID, -1, &frameOffset );

    int enableLoop = 0;
    params->EvalParam( enableLoop_ID, -1, &enableLoop );

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
    std::string tmp = ss.str();

    // set instanceData
    size_t datalen = sizeof(InstData);
    InstData *iData = (InstData*)malloc(datalen);

    iData->filename = RtUString(tmp.c_str());
    iData->L = L;

    instance->datalen = datalen;
    instance->data = iData;
    instance->freefunc = free;

    return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DxBoraOcean::GetParamTable()
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

RixSCParamInfo const* DxBoraOcean::GetParamTable()
{
    static RixSCParamInfo s_ptable[] =
    {
        // outputs
        RixSCParamInfo( RtUString("outputRGB"), k_RixSCColor, k_RixSCOutput ),
        RixSCParamInfo( RtUString("outputA"),   k_RixSCFloat, k_RixSCOutput ),

        // inputs
        RixSCParamInfo( RtUString("inputFile"),       k_RixSCString  ),
        RixSCParamInfo( RtUString("frameOffset"),     k_RixSCInteger ),
        RixSCParamInfo( RtUString("rotationAngle"),   k_RixSCFloat   ),
        RixSCParamInfo( RtUString("scaleInX"),        k_RixSCFloat   ),
        RixSCParamInfo( RtUString("scaleInZ"),        k_RixSCFloat   ),
        RixSCParamInfo( RtUString("verticalScale"),   k_RixSCFloat   ),
        RixSCParamInfo( RtUString("horizontalScale"), k_RixSCFloat   ),
        RixSCParamInfo( RtUString("crestGain"),       k_RixSCFloat   ),
        RixSCParamInfo( RtUString("crestBias"),       k_RixSCFloat   ),
        RixSCParamInfo( RtUString("enableLoop"),      k_RixSCInteger ),

        RixSCParamInfo() // end of table
    };

    return &s_ptable[0];
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DxBoraOcean::Finalize()
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DxBoraOcean::Finalize( RixContext& ctx )
{
    // nothing to do
    PIXAR_ARGUSED(ctx);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DxBoraOcean::ComputeOutputParams(): The Main Process
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

int DxBoraOcean::ComputeOutputParams
(
    RixShadingContext const* sctx,
    RtInt*                   noutputs,
    OutputSpec**             outputs,
    RtPointer                instanceData,
    RixSCParamInfo const*    instanceTable
)
{
    if( failed ) { return 1; }

    // -------------------------------------------------------------------------
    // params
    RtFloat const* rotationAngle;
    const RtFloat rotationAngleDefault = 0.f;
    sctx->EvalParam( rotationAngle_ID, -1, &rotationAngle, &rotationAngleDefault, false );

    RtFloat const* verticalScale;
    const RtFloat verticalScaleDefault = 1.f;
    sctx->EvalParam( verticalScale_ID, -1, &verticalScale, &verticalScaleDefault, false );

    RtFloat const* scaleInX;
    const RtFloat scaleInXDefault = 1.f;
    sctx->EvalParam( scaleInX_ID, -1, &scaleInX, &scaleInXDefault, false );

    RtFloat const* scaleInZ;
    const RtFloat scaleInZDefault = 1.f;
    sctx->EvalParam( scaleInZ_ID, -1, &scaleInZ, &scaleInZDefault, false );

    RtFloat const* horizontalScale;
    const RtFloat horizontalScaleDefault = 1.f;
    sctx->EvalParam( horizontalScale_ID, -1, &horizontalScale, &horizontalScaleDefault, false );

    RtFloat const* crestGain;
    const RtFloat crestGainDefault = 1.f;
    sctx->EvalParam( crestGain_ID, -1, &crestGain, &crestGainDefault, false );

    RtFloat const* crestBias;
    const RtFloat crestBiasDefault = 0.f;
    sctx->EvalParam( crestBias_ID, -1, &crestBias, &crestBiasDefault, false );

    // get instanceData
    InstData* iData = (InstData*)instanceData;

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
                out[i].value = pool.AllocForPattern<RtColorRGB>( sctx->numPts );
            }
            else if( type == k_RixSCFloat )
            {
                out[i].detail = k_RixSCVarying;
                out[i].value = pool.AllocForPattern<RtFloat>( sctx->numPts );
            }
        }
    }

    RtColorRGB* outputRGB = (RtColorRGB*)out[outputRGB_ID].value;
    if( !outputRGB )
    {
        // make sure the resultRGB space is allocated because it
        // will store the composite color results.
        outputRGB = pool.AllocForPattern<RtColorRGB>( sctx->numPts );
    }

    RtFloat* outputA = (RtFloat*)out[outputA_ID].value;
    if( !outputA )
    {
        outputA = pool.AllocForPattern<RtFloat>( sctx->numPts );
    }

    // -------------------------------------------------------------------------
    // manifold
    // Either st, or Q, will be non-NULL depending on a connected manifold.
    RtFloat2 const* st = NULL;
    RtPoint3 const* Q = NULL;
    RtFloat const* QRadius = NULL;

    RtFloat2 const defaultST(0.0f, 0.0f);
    sctx->GetPrimVar(Rix::k_st, defaultST, &st, &QRadius);

    // RtFloat *stRadius;
    // stRadius = const_cast<RtFloat*>(waveRadius);

    // get world positions
    RtVector3* wP = NULL;
    {
        RtPoint3 const* P;
        RtFloat const* pWidth;
        sctx->GetBuiltinVar( RixShadingContext::k_Po, &P );
        sctx->GetBuiltinVar( RixShadingContext::k_PRadius, &pWidth);

        wP = pool.AllocForPattern<RtVector3>( sctx->numPts );
        memcpy( wP, P, sctx->numPts*sizeof(RtPoint3) );
        RtFloat *wpWidth;
        wpWidth = pool.AllocForPattern<RtFloat>(sctx->numPts);
        memcpy( wpWidth, pWidth, sctx->numPts*sizeof(RtFloat));

        sctx->Transform( RixShadingContext::k_AsPoints, Rix::k_current, Rix::k_world, (RtPoint3*)wP, wpWidth );
    }

    // map scale
    for( RtInt i=0; i<sctx->numPts; ++i )
    {
        wP[i].x /= ( *scaleInX + 1e-6f );
        wP[i].z /= ( *scaleInZ + 1e-6f );
    }

    // map rotation
    float sn=0.f, cs=0.f;
    RixSinCos( -RixDegreesToRadians(*rotationAngle), &sn, &cs );

    if( *rotationAngle > 1e-6f )
    {
        for( RtInt i=0; i<sctx->numPts; ++i )
        {
            float& x = wP[i].x;
            float& z = wP[i].z;

            const float rx = ( x * cs ) - ( z * sn );
            const float rz = ( x * sn ) + ( z * cs );

            x = rx;
            z = rz;
        }
    }
    sn = -sn;

    // set wp to waveST
    RtFloat2 *waveST     = pool.AllocForPattern<RtFloat2>(sctx->numPts);
    for(RtInt i=0; i < sctx->numPts; ++i)
    {
        waveST[i].x = wP[i].x / iData->L;
        waveST[i].y = wP[i].z / iData->L;
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

    // std::cout << ">> : " << iData->filename.CStr() << " << " << std::endl;
    // std::cout << ">> : " << " << " << std::endl;

    PxrReadTexture rtex(m_tex, iData->filename, atlasStyle, linearize);
    int err = rtex.Texture(txParams, sctx->numPts, waveST, Q, QRadius,
                           outputRGB, outputA); // results
    // m_msg->Warning("FilePath : %s", iData->filename.CStr());

    // Handle failed lookup
    if (err == RixTexture::FileNotFound)
    {
       if ((atlasStyle == RixTexture::AtlasNone) && !iData->filename.Empty())
       {
           m_msg->Error("DxBoraOcean could not open \"%s\"", iData->filename.CStr());
           return 1;
       }

       RtColorRGB missingColor = RtColorRGB(0, 0, 0);
       RtFloat    missingAlpha = 1.0;

       rtex.FillMissingTexture(sctx->numPts, &missingColor, outputRGB);
       rtex.FillMissingTexture(sctx->numPts, &missingAlpha, outputA);
    }


    for( RtInt i=0; i < sctx->numPts; ++i )
    {
        // RtVector3 displacement = wP[i] - outputRGB[i];
        RtVector3 displacement = RtVector3(outputRGB[i]);

        // displacement rotation
        if( *rotationAngle > 1e-6f )
        {
            float& x = displacement.x;
            float& z = displacement.z;

            const float rx = ( x * cs ) - ( z * sn );
            const float rz = ( x * sn ) + ( z * cs );

            x = rx;
            z = rz;
        }

        // displacement vector
        if( outputRGB )
        {
            outputRGB[i].r = displacement.x * (*horizontalScale);
            outputRGB[i].g = displacement.y * (*verticalScale);
            outputRGB[i].b = displacement.z * (*horizontalScale);
        }

        // crest value
        if( outputA )
        {
            outputA[i] = ( outputA[i] * (*crestGain) ) + (*crestBias);
        }
    }

    return 0;
}
