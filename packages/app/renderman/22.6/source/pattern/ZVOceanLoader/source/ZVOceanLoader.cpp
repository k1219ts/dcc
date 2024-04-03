/*
Dexter $Date: 2018/09/05 $  $Revision: #1 $
# ------------------------------------------------------------------------------
#
#   Dexter ZarVis OceanWave json loader
#
#       sanghun.kim
#
# ------------------------------------------------------------------------------
*/

#include "RixPredefinedStrings.hpp"
#include "RixPattern.h"
#include "RixShadingUtils.h"
#include "PxrTextureAtlas.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sstream>
#include <iomanip>

#include "json.h"

class ZVOceanLoader : public RixPattern
{
    public:
        ZVOceanLoader();
        virtual ~ZVOceanLoader();

        virtual int Init(RixContext &, RtUString const pluginpath);
        virtual RixSCParamInfo const *GetParamTable();
        virtual void Synchronize(
            RixContext&, RixSCSyncMsg, RixParameterList const*
        )
        {
        }

        virtual int CreateInstanceData(
            RixContext&, RtUString const, RixParameterList const*, InstanceData*
        )
        {
            return -1;
        }

        virtual void Finalize(RixContext &);

        virtual int ComputeOutputParams( RixShadingContext const *,
                                         RtInt *noutputs,
                                         OutputSpec **outputs,
                                         RtPointer instaceData,
                                         RixSCParamInfo const * );

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
        int  getRenderFrame( int duration, int frame );
        void getTexFileName( const char *filename, int frame, std::string &out );
        RtInt       cframe;
        RtInt       m_frameOffset;
        RtFloat     m_waveScale;
        RtFloat     m_angle;
        RtFloat     m_scaleS;
        RtFloat     m_scaleT;
        RtFloat     m_offsetS;
        RtFloat     m_offsetT;
        RixTexture  *m_tex;
        RixMessages *m_msg;
};

int
ZVOceanLoader::getRenderFrame( int duration, int frame )
{
    int rframe = (frame-1) % duration + 1;
    return rframe;
}

void
ZVOceanLoader::getTexFileName( const char *filename, int frame, std::string &out )
{
    std::string s = filename;
    s.replace( s.find_last_of(".")+1, s.length(), "tex" );

    std::size_t pos = s.find( "$F4" );
    if( pos != std::string::npos ) {
        std::stringstream stream;
        stream << (frame < 0 ? "-" : "") << std::setfill('0') << std::setw(4) << (frame < 0 ? -frame : frame);
        s.replace( pos, 3, stream.str() );
    }
    out = s;
    return;
}

ZVOceanLoader::ZVOceanLoader() :
    m_frameOffset(0),
    m_waveScale(1.0f),
    m_angle(0.0f),
    m_scaleS(1.0f),
    m_scaleT(1.0f),
    m_offsetS(0.0f),
    m_offsetT(0.0f),
    m_tex( NULL ),
    m_msg( NULL )
{
}

ZVOceanLoader::~ZVOceanLoader()
{
}

int
ZVOceanLoader::Init(RixContext &ctx, RtUString pluginpath)
{
    RixRenderState *rstate;
    rstate = (RixRenderState *) ctx.GetRixInterface(k_RixRenderState);
    RixRenderState::FrameInfo finfo;
    rstate->GetFrameInfo( &finfo );
    cframe = (RtInt) finfo.frame;

    m_tex = (RixTexture*)ctx.GetRixInterface(k_RixTexture);
    m_msg = (RixMessages*)ctx.GetRixInterface(k_RixMessages);

    if( !m_tex || !m_msg )
        return 1;
    else
        return 0;
}

enum paramId
{
    k_resultVector=0,
    k_resultX,
    k_resultY,
    k_resultZ,
    k_resultFoam,
    k_resultWeight,
    // input
    k_jsonFile,
    k_frameOffset,
    k_waveScale,
    k_angle,
    k_scaleS,
    k_scaleT,
    k_offsetS,
    k_offsetT,
    k_numParams
};

RixSCParamInfo const *
ZVOceanLoader::GetParamTable()
{
    static RixSCParamInfo s_ptable[] =
    {
        // outputs
        RixSCParamInfo(RtUString("resultVector"), k_RixSCColor, k_RixSCOutput),
        RixSCParamInfo(RtUString("resultX"), k_RixSCFloat, k_RixSCOutput),
        RixSCParamInfo(RtUString("resultY"), k_RixSCFloat, k_RixSCOutput),
        RixSCParamInfo(RtUString("resultZ"), k_RixSCFloat, k_RixSCOutput),
        RixSCParamInfo(RtUString("resultFoam"), k_RixSCFloat, k_RixSCOutput),
        RixSCParamInfo(RtUString("resultWeight"), k_RixSCFloat, k_RixSCOutput),

        // inputs
        RixSCParamInfo(RtUString("jsonFile"), k_RixSCString),
        RixSCParamInfo(RtUString("frameOffset"), k_RixSCInteger),
        RixSCParamInfo(RtUString("waveScale"), k_RixSCFloat),
        RixSCParamInfo(RtUString("angle"), k_RixSCFloat),
        RixSCParamInfo(RtUString("scaleS"), k_RixSCFloat),
        RixSCParamInfo(RtUString("scaleT"), k_RixSCFloat),
        RixSCParamInfo(RtUString("offsetS"), k_RixSCFloat),
        RixSCParamInfo(RtUString("offsetT"), k_RixSCFloat),

        RixSCParamInfo()
    };
    return &s_ptable[0];
}

void
ZVOceanLoader::Finalize( RixContext &ctx )
{
}

int
ZVOceanLoader::ComputeOutputParams(RixShadingContext const *sctx,
                                    RtInt *noutputs, OutputSpec **outputs,
                                    RtPointer instanceData,
                                    RixSCParamInfo const *ignored)
{
    bool indirectHit = !sctx->scTraits.primaryHit || !sctx->scTraits.eyePath ||
                   sctx->scTraits.shadingMode != k_RixSCScatterQuery;

    // Input -------------------------------------------------------------------
    RtUString const* jsonFile = NULL;
    sctx->EvalParam(k_jsonFile, -1, &jsonFile);
    if( !jsonFile )
    {
        m_msg->Error( "ZVOceanLoader failed: filename is missing" );
        return 1;
    }

    RtInt const *frameOffsetPtr;
    sctx->EvalParam( k_frameOffset, -1, &frameOffsetPtr, &m_frameOffset, false );
    RtInt const frameOffset( *frameOffsetPtr );

    RtFloat const *waveScale;
    sctx->EvalParam( k_waveScale, -1, &waveScale, &m_waveScale, true );
    RtFloat const *angle;
    sctx->EvalParam( k_angle, -1, &angle, &m_angle, true );
    RtFloat const *scaleS;
    sctx->EvalParam( k_scaleS, -1, &scaleS, &m_scaleS, true );
    RtFloat const *scaleT;
    sctx->EvalParam( k_scaleT, -1, &scaleT, &m_scaleT, true );
    RtFloat const *offsetS;
    sctx->EvalParam( k_offsetS, -1, &offsetS, &m_offsetS, true );
    RtFloat const *offsetT;
    sctx->EvalParam( k_offsetT, -1, &offsetT, &m_offsetT, true );

    RixTexture::TxParams txParams;
    RixTexture::TxAtlasStyle const atlasStyle = RixTexture::TxAtlasStyle(0);
    PxrLinearizeMode linearize = PxrLinearizeMode(0);

    RtInt const optimizeIndirect = 1;
    if( optimizeIndirect && indirectHit ) {
        txParams.filter = RixTexture::TxParams::FilterType::Box;
    }

    txParams.filter = RixTexture::TxParams::FilterType(6);
    // txParams.sblurVarying = txParams.tblurVarying = true;
    // RtFloat const blur = 0.f;
    // txParams.sblur = txParams.tblur = &blur;

    // Output ==================================================================
    RixSCType type;
    RixSCConnectionInfo cinfo;

    // Find the number of outputs
    RixSCParamInfo const* paramTable = GetParamTable();
    int numOutputs = -1;
    while( paramTable[++numOutputs].access == k_RixSCOutput ) {}

    // Allocate and bind our outputs
    RixShadingContext::Allocator pool(sctx);
    OutputSpec *out = pool.AllocForPattern<OutputSpec>(numOutputs);
    *outputs        = out;
    *noutputs       = numOutputs;

    // looping through the different output ids
    for( int i=0; i<numOutputs; ++i )
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
                out[i].value  = pool.AllocForPattern<RtColorRGB>(sctx->numPts);
            }
            else if( type == k_RixSCFloat )
            {
                out[i].detail = k_RixSCVarying;
                out[i].value  = pool.AllocForPattern<RtFloat>(sctx->numPts);
            }
        }
    }

    RtColorRGB* resultVector = (RtColorRGB*) out[k_resultVector].value;
    if( !resultVector )
        resultVector = pool.AllocForPattern<RtColorRGB>(sctx->numPts);
    RtFloat *resultX = (RtFloat*) out[k_resultX].value;
    RtFloat *resultY = (RtFloat*) out[k_resultY].value;
    RtFloat *resultZ = (RtFloat*) out[k_resultZ].value;
    RtFloat *resultFoam   = (RtFloat*) out[k_resultFoam].value;
    RtFloat *resultWeight = (RtFloat*) out[k_resultWeight].value;

    // Either st, or Q, will be non-NULL depending on a connected manifold.
    RtFloat2 const* st = NULL;
    RtPoint3 const* Q  = NULL;
    RtFloat const* QRadius = NULL;

    RtFloat2 const defaultST(0.0f, 0.0f);
    sctx->GetPrimVar(Rix::k_st, defaultST, &st, &QRadius);

    RtFloat *stRadius;
    stRadius = const_cast<RtFloat*>(QRadius);

    // allocate space for out remapped worldspace P
    RtFloat3 const *p;
    RtFloat const *pWidth;
    sctx->GetBuiltinVar(RixShadingContext::k_Po, &p);
    sctx->GetBuiltinVar(RixShadingContext::k_PRadius, &pWidth);

    RtFloat3 *wp;
    RtFloat *wpWidth;
    wp = pool.AllocForPattern<RtFloat3>(sctx->numPts);
    memcpy(wp, p, sctx->numPts*sizeof(RtFloat3));
    wpWidth = pool.AllocForPattern<RtFloat>(sctx->numPts);
    memcpy(wpWidth, pWidth, sctx->numPts*sizeof(RtFloat));
    sctx->Transform(RixShadingContext::k_AsPoints, Rix::k_current, Rix::k_world, wp, wpWidth);

    // initialize
    RtColorRGB *waveCompute  = pool.AllocForPattern<RtColorRGB>(sctx->numPts);
    RtFloat    *foamCompute  = pool.AllocForPattern<RtFloat>(sctx->numPts);
    RtFloat    *weightCompute= pool.AllocForPattern<RtFloat>(sctx->numPts);
    for( int i=0; i<sctx->numPts; ++i )
    {
        waveCompute[i].r = 0.f;
        waveCompute[i].g = 0.f;
        waveCompute[i].b = 0.f;
        foamCompute[i]   = 0.f;
        weightCompute[i] = 0.f;
    }

    // read json
    json_object *jobj = json_object_from_file( jsonFile->CStr() );
    if( jobj != NULL )
    {
        struct json_object_iterator it;
        struct json_object_iterator itEnd;

        it    = json_object_iter_begin( jobj );
        itEnd = json_object_iter_end( jobj );
        while( !json_object_iter_equal(&it, &itEnd) ) {

            RtFloat waveScale      = 1.0;
            RtFloat weightMin      = 1.0f;
            RtFloat weightMax      = 1.0f;
            RtFloat weightScaleS   = 1.0f;
            RtFloat weightScaleT   = 1.0f;
            RtFloat weightOffsetS  = 0.0f;
            RtFloat weightOffsetT  = 0.0f;

            const char  *key = json_object_iter_peek_name( &it );
            json_object *obj = json_object_object_get( jobj, key );

            json_object *waveScaleObj = json_object_object_get( obj, "waveScale" );
            if( waveScaleObj != NULL )
                waveScale = json_object_get_double( waveScaleObj );
            const char *weightCoordsys = "worldP";
            json_object *weightCoordsysObj = json_object_object_get( obj, "weightCoordsys" );
            if( weightCoordsysObj != NULL )
                weightCoordsys = json_object_get_string( weightCoordsysObj );
            json_object *weightManifoldObj = json_object_object_get( obj, "weightManifold" );
            if( weightManifoldObj != NULL )
            {
                json_object *weightScaleSObj  = json_object_array_get_idx( weightManifoldObj, 0 );
                weightScaleS = json_object_get_double( weightScaleSObj );
                json_object *weightScaleTObj  = json_object_array_get_idx( weightManifoldObj, 1 );
                weightScaleT = json_object_get_double( weightScaleTObj );
                json_object *weightOffsetSObj = json_object_array_get_idx( weightManifoldObj, 2 );
                weightOffsetS= json_object_get_double( weightOffsetSObj );
                json_object *weightOffsetTObj = json_object_array_get_idx( weightManifoldObj, 3 );
                weightOffsetT= json_object_get_double( weightOffsetTObj );
            }

            // wavemap
            json_object *waveMapObj = json_object_object_get( obj, "waveMap" );
            if( waveMapObj != NULL )
            {
                RtColorRGB *wave   = pool.AllocForPattern<RtColorRGB>(sctx->numPts);
                RtFloat    *foam   = pool.AllocForPattern<RtFloat>(sctx->numPts);
                RtColorRGB *wnull  = pool.AllocForPattern<RtColorRGB>(sctx->numPts);
                RtFloat    *weight = pool.AllocForPattern<RtFloat>(sctx->numPts);

                // patchSize
                RtInt patchSize = 1;
                json_object *patchSizeObj = json_object_object_get( obj, "patchSize" );
                if( patchSizeObj != NULL )
                    patchSize = json_object_get_int( patchSizeObj );

                // Alloc wave map member variables & initialize
                RtFloat2 *waveST     = pool.AllocForPattern<RtFloat2>(sctx->numPts);
                RtFloat  *waveRadius = pool.AllocForPattern<RtFloat>(sctx->numPts);
                memset(waveRadius, 0, sctx->numPts*sizeof(RtFloat));

                // Alloc weight map member variables & initialize
                RtFloat2 *weightST     = pool.AllocForPattern<RtFloat2>(sctx->numPts);
                RtFloat  *weightRadius = pool.AllocForPattern<RtFloat>(sctx->numPts);
                memset(weightRadius, 0, sctx->numPts*sizeof(RtFloat));

                // initialize ...
                for( int i=0; i<sctx->numPts; ++i )
                {
                    // WAVE
                    waveST[i].x  = ( (wp[i].x-offsetS[i]) / patchSize * scaleS[i] );
                    waveST[i].y  = ( (wp[i].z-offsetT[i]) / patchSize * scaleT[i] );
                    waveST[i].y *= -1;
                    waveRadius[i] = RixMin( wpWidth[i] / patchSize * scaleS[i], wpWidth[i] / patchSize * scaleT[i] );

                    if( angle[i] != 0.f ) {
                        float rx, ry;
                        float cs, sn;
                        RixSinCos( RixDegreesToRadians(angle[i]), &sn, &cs );
                        rx = waveST[i].x * cs - waveST[i].y * sn;
                        ry = waveST[i].x * sn + waveST[i].y * cs;
                        waveST[i].x = rx;
                        waveST[i].y = ry;
                    }

                    // WEIGHT
                    if( !strcmp(weightCoordsys, "uv") ) {
                        weightST[i].x = weightScaleS * st[i].x + weightOffsetS;
                        weightST[i].y = weightScaleT * st[i].y + weightOffsetT;
                        weightRadius[i] = RixMin( weightScaleS * stRadius[i], weightScaleT * stRadius[i] );
                    } else {
                        weightST[i].x = (wp[i].x - (patchSize * 0.5)) / patchSize;
                        weightST[i].y = (wp[i].y - (patchSize * 0.4)) / patchSize;
                        // weightST[i].y*= -1;
                        weightRadius[i] = RixMin( wpWidth[i] / patchSize, wpWidth[i] / patchSize);
                    }

                    wave[i].r = 0.f;
                    wave[i].g = 0.f;
                    wave[i].b = 0.f;
                    foam[i]   = 0.f;
                    weight[i] = 1.f;
                }

                RtInt renderFrame = cframe + frameOffset;
                json_object *durationObj = json_object_object_get( obj, "duration" );
                if( durationObj != NULL ) {
                    int duration = json_object_get_int( durationObj );
                    renderFrame = getRenderFrame( duration, renderFrame );
                }

                // wavemap read pxrTexture =====================================
                std::string waveFile;
                const char *waveMap = json_object_get_string( waveMapObj );
                getTexFileName( waveMap, renderFrame, waveFile );

                PxrReadTexture rtex(m_tex, RtUString(waveFile.c_str()), atlasStyle, linearize);
                int err = rtex.Texture(txParams, sctx->numPts, waveST, Q, waveRadius, wave, foam);

                // weightmap read pxrTexture ===================================
                json_object *weightMapObj = json_object_object_get( obj, "weightMap" );
                if( weightMapObj != NULL )
                {
                    std::string weightFile;
                    const char *weightMap = json_object_get_string( weightMapObj );
                    getTexFileName( weightMap, renderFrame, weightFile );

                    PxrReadTexture weightTex(m_tex, RtUString(weightFile.c_str()), atlasStyle, linearize);
                    int weightErr = weightTex.Texture(txParams, sctx->numPts, weightST, Q, weightRadius, wnull, weight);

                    weightMin = 0.0f;
                    weightMax = 1.0f;
                    json_object *weightMixObj = json_object_object_get( obj, "weightMix" );
                    if( weightMixObj != NULL )
                    {
                        json_object *wminObj = json_object_array_get_idx( weightMixObj, 0 );
                        weightMin = json_object_get_double( wminObj );
                        json_object *wmaxObj = json_object_array_get_idx( weightMixObj, 1 );
                        weightMax = json_object_get_double( wmaxObj );
                    }
                }

                // compute ...
                for( int i=0; i<sctx->numPts; ++i )
                {
                    RtFloat weightFactor = RixMix( weightMin, weightMax, weight[i] );
                    waveCompute[i].r += wave[i].r * weightFactor * waveScale;
                    waveCompute[i].g += wave[i].g * weightFactor * waveScale;
                    waveCompute[i].b += wave[i].b * weightFactor * waveScale;

                    foamCompute[i]   += foam[i] * weightFactor;
                    weightCompute[i] += weightFactor;
                }
            }

            json_object_iter_next( &it );
        }

        json_object_put( jobj );
    }

    for(int i=0; i<sctx->numPts; ++i)
    {
        if(resultVector)
        {
            resultVector[i].r = waveCompute[i].r * waveScale[i];
            resultVector[i].g = waveCompute[i].g * waveScale[i];
            resultVector[i].b = waveCompute[i].b * waveScale[i];
        }
        if(resultX)
            resultX[i] = waveCompute[i].r * waveScale[i];
        if(resultY)
            resultY[i] = waveCompute[i].g * waveScale[i];
        if(resultZ)
            resultZ[i] = waveCompute[i].b * waveScale[i];
        if(resultFoam)
            resultFoam[i] = foamCompute[i];
        if(resultWeight)
            resultWeight[i] = weightCompute[i];
    }

    return 0;
}

RIX_PATTERNCREATE
{
    return new ZVOceanLoader();
}

RIX_PATTERNDESTROY
{
    delete ((ZVOceanLoader*)pattern);
}
