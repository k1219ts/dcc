#include "RixPredefinedStrings.hpp"
#include "RixPattern.h"
#include "RixShadingUtils.h"
#include "ExtraExpr.h"

using namespace SeExpr2;

class DxFoam : public RixPattern
{
    public:

        DxFoam();
        virtual ~DxFoam();

        virtual int Init(RixContext &, RtUString const pluginpath) override;
        virtual RixSCParamInfo const *GetParamTable() override;
        virtual void Synchronize(
            RixContext&, RixSCSyncMsg, RixParameterList const*
        ) override
        {
        }

        virtual void Finalize(RixContext &) override;

        virtual int ComputeOutputParams(
            RixShadingContext const*,
            RtInt *noutputs, OutputSpec **outputs,
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
        // member variables
        RtInt m_softness;
        RtFloat m_frequency;
        RtFloat m_jitter;
        RtFloat m_fbmScale;
        RtInt m_fbmOctaves;
        RtFloat m_fbmLacunarity;
        RtFloat m_fbmGain;
        RtFloat m_seGamma;
        RixMessages *m_msg;
};

DxFoam::DxFoam() :
    m_softness(0),
    m_frequency(1.0f),
    m_jitter(0.2f),
    m_fbmScale(0.9f),
    m_fbmOctaves(16),
    m_fbmLacunarity(2.0f),
    m_fbmGain(0.5f),
    m_seGamma(1.0f),
    m_msg(NULL)
{
}

DxFoam::~DxFoam()
{
}

int
DxFoam::Init(RixContext &ctx, RtUString const pluginpath)
{
    PIXAR_ARGUSED(pluginpath);

    m_msg = (RixMessages*)ctx.GetRixInterface(k_RixMessages);
    if( !m_msg )
        return 1;
    return 0;
}

enum paramId
{
    k_resultF=0, // output
    k_softness,
    k_frequency,
    k_jitter,
    k_fbmScale,
    k_fbmOctaves,
    k_fbmLacunarity,
    k_fbmGain,
    k_seGamma,
    k_manifoldBegin,
    k_manifoldQ,
    k_manifoldQradius,
    k_manifoldEnd
};

RixSCParamInfo const *
DxFoam::GetParamTable()
{
    static RixSCParamInfo s_ptable[] =
    {
        // outputs
        RixSCParamInfo(RtUString("resultF"), k_RixSCFloat, k_RixSCOutput),
        // inputs
        RixSCParamInfo(RtUString("softness"), k_RixSCInteger),
        RixSCParamInfo(RtUString("frequency"), k_RixSCFloat),
        RixSCParamInfo(RtUString("jitter"), k_RixSCFloat),
        RixSCParamInfo(RtUString("fbmScale"), k_RixSCFloat),
        RixSCParamInfo(RtUString("fbmOctaves"), k_RixSCInteger),
        RixSCParamInfo(RtUString("fbmLacunarity"), k_RixSCFloat),
        RixSCParamInfo(RtUString("fbmGain"), k_RixSCFloat),
        RixSCParamInfo(RtUString("seGamma"), k_RixSCFloat),
        RixSCParamInfo(RtUString("PxrManifold"), RtUString("manifold"), k_RixSCStructBegin),
        RixSCParamInfo(RtUString("Q"), k_RixSCPoint),
        RixSCParamInfo(RtUString("Qradius"), k_RixSCNormal),
        RixSCParamInfo(RtUString("Qradius"), k_RixSCFloat),
        RixSCParamInfo(RtUString("PxrManifold"), RtUString("manifold"), k_RixSCStructEnd),
        RixSCParamInfo()
    };
    return &s_ptable[0];
}

void
DxFoam::Finalize(RixContext &ctx)
{
    PIXAR_ARGUSED(ctx);
}

int
DxFoam::ComputeOutputParams(
    RixShadingContext const *sctx, RtInt *noutputs, OutputSpec **outputs,
    RtPointer instanceData, RixSCParamInfo const *ignored
)
{
    PIXAR_ARGUSED(instanceData);
    PIXAR_ARGUSED(ignored);

    // read each parameter value and compute the output
    // EvalParam(id, arrayIndex, result, dflt, promoteToVarying)
    bool varying = true;
    bool uniform = false;
    RtInt const *softnessPtr;
    sctx->EvalParam(k_softness, -1, &softnessPtr, &m_softness, uniform);
    RtInt const softness(*softnessPtr);
    RtFloat const *frequency;
    sctx->EvalParam(k_frequency, -1, &frequency, &m_frequency, varying);
    RtFloat const *jitter;
    sctx->EvalParam(k_jitter, -1, &jitter, &m_jitter, varying);
    RtFloat const *fbmScale;
    sctx->EvalParam(k_fbmScale, -1, &fbmScale, &m_fbmScale, varying);
    RtInt const *fbmOctaves;
    sctx->EvalParam(k_fbmOctaves, -1, &fbmOctaves, &m_fbmOctaves, varying);
    RtFloat const *fbmLacunarity;
    sctx->EvalParam(k_fbmLacunarity, -1, &fbmLacunarity, &m_fbmLacunarity, varying);
    RtFloat const *fbmGain;
    sctx->EvalParam(k_fbmGain, -1, &fbmGain, &m_fbmGain, varying);
    RtFloat const *seGamma;
    sctx->EvalParam(k_seGamma, -1, &seGamma, &m_seGamma, varying);

    // Allocate and bind our outputs
    RixShadingContext::Allocator pool(sctx);
    OutputSpec *o = pool.AllocForPattern<OutputSpec>(2);
    *outputs = o;
    *noutputs = 1;
    RtFloat *resultF = NULL;

    resultF = pool.AllocForPattern<RtFloat>(sctx->numPts);

    o[0].paramId = k_resultF;
    o[0].detail = k_RixSCVarying;
    o[0].value = (RtPointer) resultF;

    // check for manifold input
    RixSCType type;
    RixSCConnectionInfo cinfo;
    sctx->GetParamInfo(k_manifoldBegin, &type, &cinfo);

    RtFloat2 *Q= pool.AllocForPattern<RtFloat2>(sctx->numPts);
    RtFloat const *Qradius;

    if(cinfo != k_RixSCNetworkValue)
    {
        RtFloat2 const* stIn, defaultST(0.0f, 0.0f);
        RtFloat const* stRadius;
        sctx->GetPrimVar(Rix::k_st, defaultST, &stIn, &stRadius);
        Qradius= stRadius;
        for(int i=0; i<sctx->numPts; i++)
        {
            Q[i].x= stIn[i].x;
            Q[i].y= 1.0f - stIn[i].y;
        }
    }
    else
    {
        RtPoint3 const *mQ;
        RtFloat const *mQradius;
        sctx->EvalParam(k_manifoldQ, -1, &mQ);
        sctx->EvalParam(k_manifoldQradius, -1, &mQradius);
        Qradius= mQradius;
        for(int i=0; i<sctx->numPts; i++)
        {
            Q[i].x= mQ[i].x;
            Q[i].y= mQ[i].y;
        }
    }

    for (int i=0; i<sctx->numPts; i++)
    {
        // Compute output values based on inputs
        // common parameter
        SeExpr2::VoronoiPointData pdata;
        pdata.points[0]= 1.0;// Q[i].x?
        pdata.points[1]= 1.0;// Q[i].y?
        pdata.points[2]= 1.0;// Q[i].z?
        Vec3d vtype, vjitter, vfbmScale, vfbmOctaves, vfbmLacunarity, vfbmGain;
        vtype[0]= 4;
        if(softness == 1){ vtype[0]= 3; }
        vjitter[0]= jitter[i];
        vfbmScale[0]= fbmScale[i];
        vfbmOctaves[0]= fbmOctaves[i];
        vfbmLacunarity[0]= fbmLacunarity[i];
        vfbmGain[0]= fbmGain[i];

        // vor1
        Vec3d p1;
        p1[0]= Q[i].x * 7 * 16 * frequency[i];
        p1[1]= Q[i].y * 7 * 16 * 0.8 * frequency[i];
        p1[2]= 1;
        Vec3d args1[7]= {p1, vtype, vjitter, vfbmScale, vfbmOctaves, vfbmLacunarity, vfbmGain};
        double vor1;
        vor1= 1 - SeExpr2::voronoiFn( pdata, 7, &args1[0] )[0];

        // vor2
        Vec3d p2;
        p2[0]= Q[i].x * 7 * 1 * frequency[i];
        p2[1]= Q[i].y * 7 * 1 * 0.8 * frequency[i];
        p2[2]= 1;
        Vec3d args2[7]= {p2, vtype, vjitter, vfbmScale, vfbmOctaves, vfbmLacunarity, vfbmGain};
        double vor2;
        vor2= 1 - SeExpr2::voronoiFn( pdata, 7, &args2[0] )[0];

        // vor3
        Vec3d p3;
        p3[0]= Q[i].x * 7 * 2 * frequency[i];
        p3[1]= Q[i].y * 7 * 2 * 0.8 * frequency[i];
        p3[2]= 1;
        Vec3d args3[7]= {p3, vtype, vjitter, vfbmScale, vfbmOctaves, vfbmLacunarity, vfbmGain};
        double vor3;
        vor3= 1 - SeExpr2::voronoiFn( pdata, 7, &args3[0] )[0];

        // vor4
        Vec3d p4;
        p4[0]= Q[i].x * 7 * 6 * frequency[i];
        p4[1]= Q[i].y * 7 * 6 * 0.8 * frequency[i];
        p4[2]= 1;
        Vec3d args4[7]= {p4, vtype, vjitter, vfbmScale, vfbmOctaves, vfbmLacunarity, vfbmGain};
        double vor4;
        vor4= 1 - SeExpr2::voronoiFn( pdata, 7, &args4[0] )[0];

//        // fbm1
//        SeVec3d p5;
//        p5[0]= Q[i].x * 7 * 100 * frequency[i];
//        p5[1]= Q[i].y * 7 * 100 * 0.8 * frequency[i];
//        p5[2]= 1;
//        SeVec3d args5[4]= {p5, 6, 2, 0.5};
//        double fbm1;
//        fbm1= SeExpr::fbm( 4, &args5[0] );

        if(softness == 0)
        {
            vor1= SeExpr2::gamma(vor1, seGamma[i]*0.05);
            vor2= SeExpr2::gamma(vor2, seGamma[i]*0.05);
            vor3= SeExpr2::gamma(vor3, seGamma[i]*0.05);
            vor4= SeExpr2::gamma(vor4, seGamma[i]*0.05);
            resultF[i]= vor1 + vor2 + vor3 + vor4;
            resultF[i]= SeExpr2::gaussstep(resultF[i], 0, 1.5);
            resultF[i]= SeExpr2::gamma(resultF[i], 0.5);
//            resultF[i]= resultF[i] * fbm1;
        }else{
            resultF[i]= vor1 + vor2 + vor3 + vor4;
            resultF[i]= SeExpr2::gaussstep(resultF[i], 0, 2.0);
            resultF[i]= SeExpr2::gamma(resultF[i], seGamma[i]*0.5);
        }
    }

    return 0;
}

RIX_PATTERNCREATE
{
    PIXAR_ARGUSED(hint);

    return new DxFoam();
}

RIX_PATTERNDESTROY
{
    delete ((DxFoam*)pattern);
}
