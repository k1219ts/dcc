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


    DxOmniDirectionalStereo::DxOmniDirectionalStereo(
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

static RtUString const US_INTERPUPILARYDISTANCE("interpupilaryDistance");
static RtUString const US_WHICHCAMERA("whichCamera");


class DxOmniDirectionalStereo : public RixProjection
{
public:

    DxOmniDirectionalStereo(
        RixContext& ctx,
        RtUString const handle,
        RixParameterList const* parms);

    ~DxOmniDirectionalStereo() override;

    void RenderBegin(
        RixContext& ctx, RixProjectionEnvironment const& env, RixParameterList const* parms
    ) override;

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
            case k_DeepMetric:
                *result = &deepMetric;
                return k_RixSCUniform;
                break;
            default:
                return k_RixSCInvalidDetail;
        }
    }

    void Project(RixProjectionContext &pCtx) override;

private:

    DicingHint const dicingHint;
    DeepMetric const deepMetric;
    RtUString m_handle;

    RtFloat xyStep;
    RtFloat spread;
    RtFloat ratio;

    RtFloat interpupilaryDistance;
    RtUString whichCam;
    bool isL = true;
};


DxOmniDirectionalStereo::DxOmniDirectionalStereo(
    RixContext& ctx,
    RtUString const handle,
    RixParameterList const* parms)
    : dicingHint(k_Perspective),
      deepMetric(k_rayLength),
      m_handle(handle),
      whichCam("L")
{
    PIXAR_ARGUSED(ctx);
    PIXAR_ARGUSED(parms);
}

void DxOmniDirectionalStereo::RenderBegin(
    RixContext& ctx, RixProjectionEnvironment const& env, RixParameterList const* parms
)
{
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

    // env.deepMetric = RixProjectionEnvironment::k_rayLength;
    interpupilaryDistance = 0.0635f;

    RtInt paramId;
    if ( parms->GetParamId( US_INTERPUPILARYDISTANCE, &paramId ) == 0 )
        parms->EvalParam( paramId, 0, &interpupilaryDistance );

    if ( parms->GetParamId( US_WHICHCAMERA, &paramId ) == 0 )
        parms->EvalParam( paramId, 0, &whichCam );

    spread = 2.0f * F_PI * xyStep;
    isL = whichCam.Compare(RtUString("L"));
}

DxOmniDirectionalStereo::~DxOmniDirectionalStereo()
{
}


void DxOmniDirectionalStereo::Project(
    RixProjectionContext& pCtx )
{
    for ( int index = 0; index < pCtx.numRays; ++index )
    {
        RtPoint2 const& screen( pCtx.screen[ index ] );
        RtRayGeometry &ray( pCtx.rays[ index ] );

        float rx = screen.x * F_PI;
        float ry = screen.y * F_PI;
        float cr = interpupilaryDistance * 0.5;
        float offset = 0.5 * F_PI;

        if( isL )
            offset = rx + offset;
        else
            offset = rx - offset;

        ray.origin = RtFloat3( cr * -sinf(offset), 0.0f, cr * -cosf(offset) );
        ray.originRadius = 0.0f;

        float y = sinf(ry);
        float xz = cosf(ry);
        float x = sinf(rx) * xz;
        float z = cosf(rx) * xz;

        ray.direction = RtFloat3( x, y, z );
        ray.direction.Normalize();
        ray.raySpread = spread;// / (0.5f * 1.f + 0.5f);
    }
}

// ===============================================================

class DxOmniDirectionalStereoFactory : public RixProjectionFactory
{
public:

    DxOmniDirectionalStereoFactory() {};

    int Init(RixContext& ctx, RtUString const pluginPath) override
    {
        PIXAR_ARGUSED(ctx);
        PIXAR_ARGUSED(pluginPath);

        return 0;
    }

    void Finalize(RixContext& ctx) override
    {
        PIXAR_ARGUSED(ctx);
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
        RtUString const handle,
        RixParameterList const* pList) override;

    void DestroyProjection(RixProjection const* projection) override;
};


RixSCParamInfo const* DxOmniDirectionalStereoFactory::GetParamTable()
{
    static RixSCParamInfo s_ptable[] =
    {
        // $1 : Add camera parameters
        RixSCParamInfo( US_INTERPUPILARYDISTANCE, k_RixSCFloat ),
        RixSCParamInfo( US_WHICHCAMERA, k_RixSCString ),

        RixSCParamInfo() // end of table
    };
    return &s_ptable[ 0 ];
}

RixProjection* DxOmniDirectionalStereoFactory::CreateProjection(
    RixContext& ctx,
    RtUString const handle,
    RixParameterList const* pList)
{
    return new DxOmniDirectionalStereo(ctx, handle, pList);
}

void DxOmniDirectionalStereoFactory::DestroyProjection(RixProjection const* projection)
{
    delete (DxOmniDirectionalStereo*)projection;
}

RIX_PROJECTIONFACTORYCREATE
{
    PIXAR_ARGUSED(hint);
    return new DxOmniDirectionalStereoFactory();
}

RIX_PROJECTIONFACTORYDESTROY
{
    delete reinterpret_cast< DxOmniDirectionalStereoFactory * >( factory );
}
