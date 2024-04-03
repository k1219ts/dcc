// USD Feather Op @ Dexter

#include <stdint.h>
#include <iostream>
#include <type_traits>

#include <FnGeolib/op/FnGeolibOp.h>
#include <FnGeolib/op/FnGeolibCookInterface.h>
#include <FnAttribute/FnAttribute.h>

#include <FnGeolibServices/FnGeolibCookInterfaceUtilsService.h>
#include <FnGeolibServices/FnBuiltInOpArgsUtil.h>

#include <OpenEXR/ImathVec.h>
#include <OpenEXR/ImathMatrix.h>

#include "Deformer.h"
#include "Utils.h"
#include "Data.h"

namespace // anonymous
{

std::vector<std::string> GetChildren(
    Foundry::Katana::GeolibCookInterface &interface,
    std::string location
)
{
    FnAttribute::StringAttribute    potentialChildren;
    FnAttribute::StringConstVector  childrenVec;
    potentialChildren = interface.getPotentialChildren(location);
    childrenVec       = potentialChildren.getNearestSample(0.f);

    std::vector<std::string> children;
    for(int i = 0; i < childrenVec.size(); i++)
    {
        children.push_back(childrenVec.data()[i]);
    }

    return children;
}

FnAttribute::Attribute GetAttr(
    Foundry::Katana::GeolibCookInterface &interface,
    std::string name, std::string path)
{
    FnAttribute::Attribute attr = interface.getAttr(name, path);
    if(attr.getType() == kFnKatAttributeTypeGroup)
    {
        FnAttribute::GroupAttribute grp(attr);
        attr = grp.getChildByName("value");
    }
    return attr;
}

std::vector<int32_t> GetIntVec(
    Foundry::Katana::GeolibCookInterface &interface,
    std::string name, std::string path, float t=0.f
)
{
    FnAttribute::IntAttribute attr(GetAttr(interface, name, path));
    FnAttribute::IntConstVector vecs(attr.getNearestSample(t));
    return std::vector<int32_t>(vecs.data(), vecs.data() + vecs.size());
}

int32_t GetInt(
    Foundry::Katana::GeolibCookInterface &interface,
    std::string name, std::string path, float t=0.f
)
{
    std::vector<int32_t> res = GetIntVec(interface, name, path, t);
    if(res.size())
        return res[0];
    else
        return 0;
}

std::vector<float> GetFloatVec(
    Foundry::Katana::GeolibCookInterface &interface,
    std::string name, std::string path, float t=0.f
)
{
    FnAttribute::FloatAttribute attr(GetAttr(interface, name, path));
    FnAttribute::FloatConstVector vecs(attr.getNearestSample(t));
    return std::vector<float>(vecs.data(), vecs.data() + vecs.size());
}

float GetFloat(
    Foundry::Katana::GeolibCookInterface &interface,
    std::string name, std::string path, float t=0.f
)
{
    std::vector<float> res =  GetFloatVec(interface, name, path, t);
    if(res.size())
        return res[0];
    else
        return 0.f;
}

std::vector<Imath::V2f> GetV2f(
    Foundry::Katana::GeolibCookInterface &interface,
    std::string name, std::string path, float t=0.f,
    bool v3to2=false
)
{
    std::vector<float> vs = GetFloatVec(interface, name, path, t);
    std::vector<Imath::V2f> res;
    int size = v3to2? 3 : 2;

    for(int i=0; i<vs.size()/size; i++)
    {
        Imath::V2f m(vs[i*size], vs[i*size+1]);
        res.push_back(m);
    }
    return res;
}

std::vector<Imath::V3f> GetV3f(
    Foundry::Katana::GeolibCookInterface &interface,
    std::string name, std::string path, float t=0.f
)
{
    std::vector<float> vs = GetFloatVec(interface, name, path, t);
    std::vector<Imath::V3f> res;
    for(int i=0; i<vs.size()/3; i++)
    {
        Imath::V3f m(vs[i*3], vs[i*3+1], vs[i*3+2]);
        res.push_back(m);
    }
    return res;
}

int32_t GetLamPoints(
    std::vector<Imath::V3f> *var,
    Foundry::Katana::GeolibCookInterface &interface,
    std::string name, std::string path, float t=0.f
)
{
    std::vector<Imath::V3f> Ps = GetV3f(interface, name, path, t);
    int num = Ps.size() / 3;
    for(int i=0; i<num; i++)
    {
        var[0].push_back(Ps[i]);
        var[1].push_back(Ps[i+num]);
        var[2].push_back(Ps[i+num+num]);
    }
    return Ps.size();
}

void UVGroupBuilder(
    FnAttribute::GroupBuilder &gb,
    const FnAttribute::FloatAttribute &v, bool isFace=false,
    bool is3Elms=false
)
{
    gb.reset();
    if(isFace)
        gb.set("scope", FnAttribute::StringAttribute("face"));
    else
    {
        gb.set("scope", FnAttribute::StringAttribute("point"));
        gb.set("interpolationType", FnAttribute::StringAttribute("subdiv"));
    }

    gb.set("inputType", FnAttribute::StringAttribute("float"));
    if(is3Elms)
      gb.set("elementSize", FnAttribute::IntAttribute(3));
    else
      gb.set("elementSize", FnAttribute::IntAttribute(2));

    gb.set("value", v);
}


void IDGroupBuilder(
    FnAttribute::GroupBuilder &gb,
    const FnAttribute::IntAttribute &v, bool isPrim=false
)
{
    gb.reset();
    if(isPrim)
        gb.set("scope", FnAttribute::StringAttribute("primitive"));
    else
        gb.set("scope", FnAttribute::StringAttribute("face"));

    gb.set("inputType", FnAttribute::StringAttribute("int"));
    gb.set("value", v);
}


void ColorGroupBuilder(
    FnAttribute::GroupBuilder &gb,
    const FnAttribute::FloatAttribute &v
)
{
    gb.reset();
    gb.set("scope", FnAttribute::StringAttribute("point"));
    gb.set("inputType", FnAttribute::StringAttribute("float"));
    gb.set("interpolationType", FnAttribute::StringAttribute("subdiv"));
    gb.set("elementSize", FnAttribute::IntAttribute(3));
    gb.set("value", v);
}


class DxUsdFeatherOp : public Foundry::Katana::GeolibOp
{
public:
    static void setup(Foundry::Katana::GeolibSetupInterface &interface)
    {
        interface.setThreading(
            Foundry::Katana::GeolibSetupInterface::ThreadModeConcurrent);
    }

    static void cook(Foundry::Katana::GeolibCookInterface &interface)
    {
        // CEL support
        FnAttribute::StringAttribute celAttr = interface.getOpArg("CEL");
        if (celAttr.isValid())
        {
            FnGeolibServices::FnGeolibCookInterfaceUtils::MatchesCELInfo info;
            FnGeolibServices::FnGeolibCookInterfaceUtils::matchesCEL(info,
                                                                     interface,
                                                                     celAttr);
            if (!info.canMatchChildren)
            {
                interface.stopChildTraversal();
            }
            if (!info.matches)
            {
                return;
            }
        }
        else
        {
            interface.stopChildTraversal();
            return;
        }

        // Get attributes
        float stemWMul;
        float barbWMul;
        float barbProb;
        float lamProb;
        FnAttribute::FloatAttribute stemWMulAttr;
        FnAttribute::FloatAttribute barbWMulAttr;
        FnAttribute::FloatAttribute bProbAttr;
        FnAttribute::FloatAttribute lProbAttr;
        stemWMulAttr  = interface.getOpArg("stemWidthMultiplier");
        barbWMulAttr  = interface.getOpArg("barbWidthMultiplier");
        bProbAttr = interface.getOpArg("barbProbability");
        lProbAttr = interface.getOpArg("laminationProbability");

        if(stemWMulAttr.isValid() && barbWMulAttr.isValid() &&
           bProbAttr.isValid() && lProbAttr.isValid())
        {
            stemWMul = stemWMulAttr.getValue();
            barbWMul = barbWMulAttr.getValue();
            barbProb = bProbAttr.getValue() * 100;
            lamProb  = lProbAttr.getValue() * 100;
        }
        else
        {
            interface.stopChildTraversal();
            return;
        }

        // ---------------------------------------------------------------------
        // Create Deformer
        DxUsdFeather::Deformer dfm;
        dfm.fthGrpPath = interface.getOutputLocationPath();

        // Set prim path
        if(!dfm.SetFeatherElements(GetChildren(interface, dfm.fthGrpPath)))
        {
            interface.stopChildTraversal();
            return;
        }

        // check children types
        std::string n = "geometry.arbitrary.fid";
        if(GetInputLocationType(interface, dfm.fthPath) != "curves" ||
           interface.getAttr(n, dfm.fthPath).getType() < 0)
        {
            std::cout << "#Warning : " << dfm.fthPath;
            std::cout << " is not available." << std::endl;
            interface.stopChildTraversal();
            return;
        }

        // ---------------------------------------------------------------------
        // Get "Feather" attrs
        dfm.slam.fid = GetInt(interface, n, dfm.fthPath);

        n = "userProperties.featherLamination";
        dfm.numv = GetLamPoints(dfm.slam.Ps, interface, n, dfm.fthPath);

        // ---------------------------------------------------------------------
        // Get attrs per curve
        n = "geometry.numVertices";
        std::vector<int> numVer = GetIntVec(interface, n, dfm.fthPath);
        n = "geometry.arbitrary.eid";
        std::vector<int> eid = GetIntVec(interface, n, dfm.fthPath);
        n = "geometry.arbitrary.featherREParams";
        std::vector<Imath::V2f> RE = GetV2f(interface, n, dfm.fthPath);

        // ---------------------------------------------------------------------
        // Get attrs per cvs
        // n = "geometry.arbitrary.nface";
        // int nface = GetInt(interface, n, dfm.fthPath);
        n = "geometry.point.P";
        std::vector<Imath::V3f> CV = GetV3f(interface, n, dfm.fthPath);
        n = "geometry.point.width";
        std::vector<float> W = GetFloatVec(interface, n, dfm.fthPath);
        n = "geometry.arbitrary.st";
        std::vector<Imath::V2f> ST = GetV2f(interface, n, dfm.fthPath);
        n = "geometry.arbitrary.fuv";
        std::vector<Imath::V3f> FUV = GetV3f(interface, n, dfm.fthPath);
        n = "geometry.arbitrary.featherUParams";
        std::vector<float> U = GetFloatVec(interface, n, dfm.fthPath);
        n = "geometry.arbitrary.displayColor";
        std::vector<Imath::V3f> Cd = GetV3f(interface, n, dfm.fthPath);

        int cv = 0;
        for(int b=0; b<numVer.size(); b++)
        {
            // barb probability
            srand(b);
            if(eid[b] > 0 && rand()%100 > barbProb)
            {
                cv += numVer[b];
            }
            else
            {
                DxUsdFeather::Barb barb;
                barb.num = numVer[b];
                barb.bid = b;
                barb.RE  = RE[b];
                barb.startCV = cv;

                for(int c=0; c<numVer[b]; c++)
                {
                    barb.CV.push_back(CV[cv]);
                    barb.ST.push_back(ST[cv]);
                    barb.FUV.push_back(FUV[cv]);
                    barb.U.push_back(U[cv]);
                    barb.W.push_back(W[cv] * ((eid[b])? barbWMul:stemWMul));
                    barb.Cd.push_back(Cd[cv]);
                    cv++;
                }
                dfm.barbs[eid[b]].push_back(barb);
            }
        }

        // ---------------------------------------------------------------------
        // Get time-sample from Lamination


        // ---------------------------------------------------------------------
        // Get "Lamination" attrs and deform the feathers
        n = "geometry.point.P";
        DxUsdFeather::Data data;
        FnAttribute::FloatAttribute PAttr = interface.getAttr(n, dfm.lamPath);
        data.SetTimeSamples(PAttr);

        // get bodyST
        std::vector<Imath::V2f> bST;
        std::string bstn = "geometry.arbitrary.bodyST";
        bST = GetV2f(interface, bstn, dfm.lamPath, data.samples[0]);

        // get face count of each lam

        for(int64_t t=0; t<data.numSamples; t++)
        {
            // clear lam vector
            dfm.lams.clear();

            // get points on lam by time-sample
            std::vector<Imath::V3f> P;
            P = GetV3f(interface, n, dfm.lamPath, data.samples[t]);


            // iter lamination
            int32_t fid = 0;
            int numlams = P.size() / dfm.numv;
            int nface   = bST.size() / numlams;

            for(int v=0; v<numlams; v++)
            {
                // lamination probability
                srand(v);
                if(rand()%100 <= lamProb)
                {
                    DxUsdFeather::Lamination lam;
                    lam.fid = fid;
                    lam.lid = v;
                    int32_t  numSideVtx = dfm.numv / 3;
                    for(int e=0; e<numSideVtx; e++)
                    {
                        lam.Ps[0].push_back(P[v*dfm.numv + e]);
                        lam.Ps[1].push_back(P[v*dfm.numv + e + numSideVtx]);
                        lam.Ps[2].push_back(P[v*dfm.numv + e + numSideVtx*2]);
                    }

                    if(t == 0 && bST.size() > 0) // not time-sampled
                    {
                        lam.bodyST = bST[v*nface];
                    }

                    dfm.lams.push_back(lam);
                }
                fid++;
            }

            // #################################################################
            // DEFORM
            // #################################################################

            // for debugging
            dfm.Deform(t, data, false);
            // dfm.Deform(t, data, false, 0, 80);

            // #################################################################
        }

        // ---------------------------------------------------------------------
        // Create feather prim and set attributes
        FnGeolibServices::StaticSceneCreateOpArgsBuilder sscb(false);
        FnAttribute::GroupBuilder gb;

        std::string outfth = "Feathers";
        sscb.createEmptyLocation("Feathers", "curves");

        n = "geometry.point.P";
        sscb.setAttrAtLocation(outfth, n, data.Get_P());

        n = "geometry.point.width";
        sscb.setAttrAtLocation(outfth, n, data.Get_W());

        n = "geometry.numVertices";
        sscb.setAttrAtLocation(outfth, n, data.Get_NV());

        n = "geometry.arbitrary.fid";
        IDGroupBuilder(gb, data.Get_FID(), true);
        sscb.setAttrAtLocation(outfth, n, gb.build());

        n = "geometry.arbitrary.fid2";
        IDGroupBuilder(gb, data.Get_FID2());
        sscb.setAttrAtLocation(outfth, n, gb.build());

        n = "geometry.arbitrary.bid";
        IDGroupBuilder(gb, data.Get_BID());
        sscb.setAttrAtLocation(outfth, n, gb.build());

        n = "geometry.arbitrary.st";
        UVGroupBuilder(gb, data.Get_ST());
        sscb.setAttrAtLocation(outfth, n, gb.build());

//        n = "geometry.arbitrary.fuv";
//        UVGroupBuilder(gb, data.Get_FUV(), false, true);
//        sscb.setAttrAtLocation(outfth, n, gb.build());

        n = "geometry.arbitrary.bodyST";
        UVGroupBuilder(gb, data.Get_BST(), true);
        sscb.setAttrAtLocation(outfth, n, gb.build());

        n = "geometry.arbitrary.Cd";
        ColorGroupBuilder(gb, data.Get_CD());
        sscb.setAttrAtLocation(outfth, n, gb.build());


        // ---------------------------------------------------------------------
        // Set other attributes
        n = "geometry.degree";
        sscb.setAttrAtLocation(outfth, n, interface.getAttr(n, dfm.fthPath));

        n = "geometry.basis";
        sscb.setAttrAtLocation(outfth, n, interface.getAttr(n, dfm.fthPath));

        n = "prmanStatements.basis.u";
        FnAttribute::StringAttribute catmul("catmull-rom");
        sscb.setAttrAtLocation(outfth, n, catmul);

        n = "prmanStatements.basis.v";
        sscb.setAttrAtLocation(outfth, n, catmul);

        // txBasePath
        n = "geometry.arbitrary.txBasePath";
        sscb.setAttrAtLocation(outfth, n, interface.getAttr(n, dfm.lamPath));

        // txLayerName
        n = "geometry.arbitrary.txLayerName";
        sscb.setAttrAtLocation(outfth, n, interface.getAttr(n, dfm.lamPath));

        // txmultiUV
        n = "geometry.arbitrary.txmultiUV";
        sscb.setAttrAtLocation(outfth, n, interface.getAttr(n, dfm.lamPath));

        // MaterialSet
        n = "userProperties.MaterialSet";
        sscb.setAttrAtLocation(outfth, n, interface.getAttr(n, dfm.lamPath));

        // MaterialSet
        n = "prmanStatements.attributes.user.txVersion";
        sscb.setAttrAtLocation(outfth, n, interface.getAttr(n, dfm.lamPath));


        // ---------------------------------------------------------------------
        // Build
        interface.execOp("StaticSceneCreate", sscb.build());
    }
};

DEFINE_GEOLIBOP_PLUGIN(DxUsdFeatherOp)


} // anonymous

void registerPlugins()
{
    REGISTER_PLUGIN(DxUsdFeatherOp, "DxUsdFeatherOp", 0, 1);
}
