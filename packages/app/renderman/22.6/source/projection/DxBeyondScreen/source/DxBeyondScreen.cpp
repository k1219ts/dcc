/*
# ------------------------------------------------------------------------------
#
# Author: Wonchul.kang
#
# This camera projection renders the scene as angular fisheye.
# It is intended to be rendered in a square format.
#

####### GetProperty Usage ######################################################

    // -------------------------------------------------------------------------
    // Refer to  RixProjection.h

    // Dicing hint value, to be returned by GetProperty when
    /// ProjectionProperty == k_DicingHint. Helps the
    /// renderer decide how to dice geometry
    enum DicingHint
    {
        k_Orthographic,
        k_Perspective,
        k_Spherical
    };

    // Projections can be queried at the beginning of rendering for
    // general properties interesting to the renderer. They should
    // return k_RixSCInvalidDetail if a particular property is
    // unsupported.  If a property is supported, k_RixSCUniform or
    // k_RixSCVarying should be returned according to the detail of
    // the result. A trivial implementation that supports no
    // properties of any sort should simply return
    // k_RixSCInvalidDetail.
    enum ProjectionProperty
    {
        // enum DicingHint - see above
        k_DicingHint,

        // float: field of view of the plugin for perspective and
        // spherical. Used by the renderer as hint on how to dice
        // geometry.
        k_FieldOfView,

        // float: depth of field (defocus) for perspective projections
        k_FStop,
        k_FocalLength,
        k_FocalDistance

    };

    // -------------------------------------------------------------------------
    // Add variable in the projection class as private

    private:

        DicingHint const dicingHint;
        RtFloat const fovHint;


    // -------------------------------------------------------------------------
    // Set the parameter values the initializing function


    DxBeyondScreen::DxBeyondScreen(
        RixContext& ctx,
        RixProjectionEnvironment& env,
        RtUString const handle,
        RixParameterList const* parms)
        : dicingHint(k_Spherical),    << you can set in here
          m_handle(handle)
    {
        fovHint = 90.0;               << or like this

    }

# ------------------------------------------------------------------------------
*/

#include <RixProjection.h>
#include <RixIntegrator.h>
#include <RixShadingUtils.h>

#include <algorithm>
#include <cstring>

#include <BeyondScreen.h>
using namespace BeyondScreen;

static RtUString const US_DATAFILE("datafile");
static RtUString const US_USESCREENUV("useScreenUV");


class DxBeyondScreen : public RixProjection
{
public:

    DxBeyondScreen(
        RixContext& ctx,
        RixProjectionEnvironment& env,
        RtUString const handle,
        RixParameterList const* parms);

    ~DxBeyondScreen() override;

    RixSCDetail GetProperty(
        ProjectionProperty property,
        void const** result) const override
    {
        switch (property)
        {
            case k_DicingHint:
                *result = &dicingHint;
                return k_RixSCUniform;
                break;
            case k_FieldOfView:
                *result = &fovHint;
                return k_RixSCUniform;
                break;
            default:
                return k_RixSCInvalidDetail;
        }
    }

    void Project(
        RixProjectionContext &pCtx ) override;

    RtFloat3 getBeyondScreenVector(float u, float v, bool &found);

private:

    DicingHint const dicingHint;
    RtFloat const fovHint;
    RtUString m_handle;

    BeyondScreen::Manager manager;
    BeyondScreen::String  beyondscreen_cache_path;
    bool loadManager = false;
    bool useScreenUV;

    float uMin;
    float uMax;
    float vMin;
    float vMax;
    float uLen;
    float uCenter;
    float vCenter;
    float uvAspect;
    Vector camPos;
    Vector camX;
    Vector camY;
    Vector camZ;

    float fov = 90.0f;

    RtUString datafile;
    RtFloat xyStep;
    RtFloat spread;
};


DxBeyondScreen::DxBeyondScreen(
    RixContext& ctx,
    RixProjectionEnvironment& env,
    RtUString const handle,
    RixParameterList const* parms)
    : dicingHint(k_Spherical),
      fovHint(180.0),
      m_handle(handle)
{
    RixRenderState &state = *reinterpret_cast< RixRenderState * >(ctx.GetRixInterface( k_RixRenderState ) );
    RixRenderState::Type fmtType;

    RixRenderState::FrameInfo frameInfo;
    state.GetFrameInfo(&frameInfo);

    float screenWidth = env.screenWindowRight - env.screenWindowLeft;
    float screenHeight = env.screenWindowTop - env.screenWindowBottom;
    if ( screenWidth == 0.0f || screenHeight == 0.0f )
    {
        float aspect = env.width * env.pixelAspectRatio / env.height;
        screenWidth = std::max( 2.0f, 2.0f * aspect );
        screenHeight = std::max( 2.0f, 2.0f / aspect );
    }
    float xStep = 0.25f * screenWidth / env.width;
    float yStep = 0.25f * screenHeight / env.height;
    xyStep = std::max( xStep, yStep );

    env.deepMetric = RixProjectionEnvironment::k_rayLength;


    // get parameters
    RtInt paramId;
    int _useScreenUV;
    RtUString file;

    if( parms->GetParamId(US_DATAFILE, &paramId) == 0 )
        parms->EvalParam( paramId, 0, &file );

    if( parms->GetParamId(US_USESCREENUV, &paramId) == 0 )
        parms->EvalParam( paramId, 0, &_useScreenUV );

    beyondscreen_cache_path = (BeyondScreen::String)file.CStr();
    useScreenUV = bool(_useScreenUV);


    // beyondscreen setup
    loadManager = manager.load(beyondscreen_cache_path.asChar());
    if(loadManager)
    {
        uMin = manager.uMin();
        uMax = manager.uMax();
        vMin = manager.vMin();
        vMax = manager.vMax();

        uLen = uMax - uMin;
        uCenter = 0.5f*(uMin + uMax);
        vCenter = 0.5f*(vMin + vMax);
        uvAspect = (vMax - vMin) / (uMax - uMin);

        int    frame = frameInfo.frame - manager.animationFrames[0];
        Vector _pos  = manager.worldCameraPositions[frame];
        camPos = Vector(_pos.x, _pos.y, _pos.z);

        // get camera x,y,z axis
        camY = manager.worldCameraUpvectors[frame];
        camZ = camPos - manager.worldAimingPoints[frame];
        camZ.normalize();
        camX = camY ^ camZ;

        manager.setDrawingData(frame);

        // find fov
        bool found;
        RtFloat3 v1 = getBeyondScreenVector(uMin, vCenter, found);
        RtFloat3 v2 = getBeyondScreenVector(uMax, vCenter, found);

        fov = acosf( (v1.x*v2.x) + (v1.y*v2.y) + (v1.z*v2.z) );
    }

    spread = atanf( tanf(0.5 * fov * F_DEGTORAD) * xyStep );
}

DxBeyondScreen::~DxBeyondScreen()
{
}


RtFloat3 DxBeyondScreen::getBeyondScreenVector(float u, float v, bool &found)
{
    found = false;
    const Vector uv( u, v, 0.0 );
    // world position on screen object
    Vector wp;
    const IntArray& candidates = manager.hashGrid.candidates(u, v);

    const int numCandidates = candidates.length();
    for( int iCandidate=0; iCandidate<numCandidates; ++iCandidate )
    {
        // triangle index being queried
        const int& iTriangle = candidates[iCandidate];

        // the vertex index of the triangle
        const int& v0 = manager.worldScreenMesh.t[3*iTriangle  ];
        const int& v1 = manager.worldScreenMesh.t[3*iTriangle+1];
        const int& v2 = manager.worldScreenMesh.t[3*iTriangle+2];

         // the vertex position of the triangle in world space
        const Vector& p0 = manager.worldScreenMesh.p[v0];
        const Vector& p1 = manager.worldScreenMesh.p[v1];
        const Vector& p2 = manager.worldScreenMesh.p[v2];

        // the vertex position of the triangle in uv space
        const Vector& uv0 = manager.worldScreenMesh.uv[v0];
        const Vector& uv1 = manager.worldScreenMesh.uv[v1];
        const Vector& uv2 = manager.worldScreenMesh.uv[v2];

        // the barycentric coordinates of the point being queried
        Double3 weights;

        // point inside triangle test in 2D uv space
        if( PointInTriangle2D( uv, uv0, uv1, uv2, weights ) )
        {
            found = true;
            // the world position of the point being queried
            wp = WeightedSum( p0, p1, p2, weights );
            break;
        }
    }


    if(!found)
        return RtFloat3(0.0f, 0.0f, 0.0f);

    Vector rv = wp - camPos;

    RtFloat3 res =  RtFloat3( rv.x*camX.x + rv.y*camX.y + rv.z*camX.z,
                              rv.x*camY.x + rv.y*camY.y + rv.z*camY.z,
                             -rv.x*camZ.x - rv.y*camZ.y - rv.z*camZ.z
                            );
    res.Normalize();
    return res;
}


void DxBeyondScreen::Project(
    RixProjectionContext& pCtx )
{
    for ( int index = 0; index < pCtx.numRays; ++index )
    {
        RtPoint2 const& screen( pCtx.screen[ index ] );
        RtRayGeometry &ray( pCtx.rays[ index ] );


        if(!loadManager)
        {
            ray.direction = RtFloat3( 0.0f, 0.0f, 0.0f );
            continue;
        }

        // screen position to UV
        float u, v;
        u = uCenter + 0.5f*uLen*screen.x;
        v = vCenter + 0.5f*uLen*screen.y*((useScreenUV)? 1.f : uvAspect);

        if(useScreenUV && (u < uMin || u > uMax || v < vMin || v > vMax))
        {
            ray.direction = RtFloat3( 0.0f, 0.0f, 0.0f );
            continue;
        }
        // projection
        bool found;
        ray.direction = getBeyondScreenVector(u, v, found);

        if(!found)
            continue;

        ray.origin = RtFloat3( 0.0f, 0.0f, 0.0f );
        ray.originRadius = 0.0f;

        // std::cout << ray.direction << std::endl;
        ray.raySpread = spread * ray.direction.z;
    }
}

// ===============================================================

class DxBeyondScreenFactory : public RixProjectionFactory
{
public:

    DxBeyondScreenFactory() {};

    int Init(RixContext& ctx, RtUString const pluginPath) override
    {
        return 0;
    }

    void Finalize(RixContext& ctx) override
    {
    }

    int CreateInstanceData(
        RixContext&,
        RtUString handle,
        RixParameterList const*,
        InstanceData* result) override
    {
        return -1;
    }

    RixSCParamInfo const* GetParamTable() override;

    void Synchronize(
        RixContext&,
        RixSCSyncMsg,
        RixParameterList const*) override
    {
    }

    RixProjection* CreateProjection(
        RixContext& ctx,
        RixProjectionEnvironment& env,
        RtUString const handle,
        RixParameterList const* pList) override;

    void DestroyProjection(RixProjection const* projection) override;
};


RixSCParamInfo const* DxBeyondScreenFactory::GetParamTable()
{
    static RixSCParamInfo s_ptable[] =
    {
        RixSCParamInfo( US_DATAFILE, k_RixSCString ),
        RixSCParamInfo( US_USESCREENUV, k_RixSCInteger ),

        RixSCParamInfo() // end of table
    };
    return &s_ptable[ 0 ];
}

RixProjection* DxBeyondScreenFactory::CreateProjection(
    RixContext& ctx,
    RixProjectionEnvironment& env,
    RtUString const handle,
    RixParameterList const* pList)
{
    return new DxBeyondScreen(ctx, env, handle, pList);
}

void DxBeyondScreenFactory::DestroyProjection(RixProjection const* projection)
{
    delete (DxBeyondScreen*)projection;
}

RIX_PROJECTIONFACTORYCREATE
{
    PIXAR_ARGUSED(hint);
    return new DxBeyondScreenFactory();
}

RIX_PROJECTIONFACTORYDESTROY
{
    delete reinterpret_cast< DxBeyondScreenFactory * >( factory );
}
