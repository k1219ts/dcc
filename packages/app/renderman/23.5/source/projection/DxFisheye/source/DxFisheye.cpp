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


    DxFisheye::DxFisheye(
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

class DxFisheye : public RixProjection
{
public:

    DxFisheye(
        RixContext& ctx,
        RtUString const handle,
        RixParameterList const* parms);

    ~DxFisheye() override;

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
            case k_FieldOfView:
                *result = &fovHint;
                return k_RixSCUniform;
                break;
            default:
                return k_RixSCInvalidDetail;
        }
    }

    void Project(RixProjectionContext& pCtx) override;

private:

    DicingHint const dicingHint;
    DeepMetric const deepMetric;
    RtFloat const fovHint;
    RtUString m_handle;

    RtFloat xyStep;
    RtFloat spread;
};


DxFisheye::DxFisheye(
    RixContext& ctx,
    RtUString const handle,
    RixParameterList const* parms)
    : dicingHint(k_Spherical),
      deepMetric(k_rayLength),
      fovHint(170.0f),
      m_handle(handle)
{
    PIXAR_ARGUSED(ctx);
    PIXAR_ARGUSED(parms);
}

void DxFisheye::RenderBegin(
    RixContext& ctx, RixProjectionEnvironment const& env, RixParameterList const* parms
)
{
    PIXAR_ARGUSED(ctx);
    PIXAR_ARGUSED(parms);

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

    spread = atanf( tanf(58.4815f * F_DEGTORAD) * xyStep );
}

DxFisheye::~DxFisheye()
{
}


void DxFisheye::Project(RixProjectionContext& pCtx)
{
    for ( int index = 0; index < pCtx.numRays; ++index )
    {
        RtPoint2 const& screen( pCtx.screen[ index ] );
        RtRayGeometry &ray( pCtx.rays[ index ] );

        float len = screen.x * screen.x + screen.y * screen.y;
        float zSq = 1.0f - len;
        len = sqrtf(len);

        ray.origin = RtFloat3( 0.0f, 0.0f, 0.0f );
        ray.originRadius = 0.0f;
        if ( zSq < 0.0f )
        {
            ray.direction = RtFloat3( 0.0f, 0.0f, 0.0f );
            continue;
        }

        // tangent
        float dz = 1.0f;
        if (len != 0.0f)
        {
            dz = tanf((1.0f - len) * F_PI * 0.5f) * len;
        }

        ray.direction = RtFloat3(screen.x, screen.y, dz);

        ray.direction.Normalize();
        ray.raySpread = spread / (0.5f * ray.direction.z + 0.5f);
    }
}

// ===============================================================

class DxFisheyeFactory : public RixProjectionFactory
{
public:

    DxFisheyeFactory() {};

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


RixSCParamInfo const* DxFisheyeFactory::GetParamTable()
{
    static RixSCParamInfo s_ptable[] =
    {
        // $1 : Add camera parameters
        // RixSCParamInfo( US_INTERPUPILARYDISTANCE, k_RixSCFloat ),

        RixSCParamInfo() // end of table
    };
    return &s_ptable[ 0 ];
}

RixProjection* DxFisheyeFactory::CreateProjection(
    RixContext& ctx,
    RtUString const handle,
    RixParameterList const* pList)
{
    return new DxFisheye(ctx, handle, pList);
}

void DxFisheyeFactory::DestroyProjection(RixProjection const* projection)
{
    delete (DxFisheye*)projection;
}

RIX_PROJECTIONFACTORYCREATE
{
    PIXAR_ARGUSED(hint);
    return new DxFisheyeFactory();
}

RIX_PROJECTIONFACTORYDESTROY
{
    delete reinterpret_cast< DxFisheyeFactory * >( factory );
}
