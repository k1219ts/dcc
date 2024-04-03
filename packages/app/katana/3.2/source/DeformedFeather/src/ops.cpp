#include <pystring/pystring.h>
#include <FnGeolib/op/FnGeolibOp.h>

#include <FnGeolibServices/FnGeolibCookInterfaceUtilsService.h>
#include <FnGeolibServices/FnBuiltInOpArgsUtil.h>

#include <FnAttribute/FnAttribute.h>
#include <FnAttribute/FnGroupBuilder.h>
#include <FnPluginSystem/FnPlugin.h>

#include <stdint.h>
#include <iostream>
#include "DeformedFeather.h"

#include "DxNurbsCurve.h"
#include "DxFeather.h"
#include "DxPoint.h"
#include "DxVector.h"

using namespace std;

namespace
{
    class DxDeformedFeatherOp : public Foundry::Katana::GeolibOp
    {
    public:
        static void setup(FnKat::GeolibSetupInterface &interface)
        {
            interface.setThreading(FnKat::GeolibSetupInterface::ThreadModeConcurrent);
        }

        static void cook(FnKat::GeolibCookInterface &interface)
        {
            FnAttribute::StringAttribute celAttr = interface.getOpArg("CEL");
            if( !celAttr.isValid() )
            {
                interface.stopChildTraversal();
                return;
            }
            if( celAttr.isValid() )
            {
                FnGeolibServices::FnGeolibCookInterfaceUtils::MatchesCELInfo info;
                FnGeolibServices::FnGeolibCookInterfaceUtils::matchesCEL(info, interface, celAttr);

                if( !info.matches)
                {
                    return;
                }
            }

            /*
             Workflow
             1. Get Curve Information
                - geometry.point.P              : float3Array * timeSamples
                - geometry.numVertices          : floatArray * timeSamples
                - geometry.point.width          : floatArray * timeSamples
                - pimvars:st                    : float2Array

             2. Get DeformAttribute from Curve
                - primvars:featherLamination    : float3Array
                - primvars:featherREParams      : float2Array
                - primvars:featherUParams       : floatArray
                - primvars:featherSeperationIdx : float2Array

             3. Get Deformed Feather Mesh Information
                - geometry.points.P             : float3Array * timeSamples

             4. Deforming...
                - Set SourceFeather Data
             5. Create Deformed Curve. (basis : catmullRom)
                * geometry attributes
                    - geometry.point.P              : float3Array * timeSamples
                    - geometry.numVertices          : floatArray * timeSamples
                    - geometry.point.width          : floatArray * timeSamples
                * custom attributes
                    - primvars:uids                 : intArray
                    -
            */

            // last workflow
            /*
             * Loop By featherSource
             * Loop By Sampling
             */

            // *************** Workflow - 1 ***************
            std::string featherLocation = "";
            std::string deformMeshLocation = "";
            std::string renderOutputLocation = "render";
            FnAttribute::StringConstVector childrenVectorAttr   = interface.getPotentialChildren(interface.getOutputLocationPath()).getNearestSample(0.0f);
            size_t childrenSize                                 = childrenVectorAttr.size();
            // std::cout << childrenSize << std::endl;
            for(int i = 0; i < childrenSize; ++i)
            {
                if (strcmp(childrenVectorAttr.data()[i], "Prototypes") == 0)
                {
                    FnAttribute::StringConstVector protoVecAttr = interface.getPotentialChildren(interface.getOutputLocationPath() + "/Prototypes").getNearestSample(0.0f);
                    char featherTmpLocation[256] = {};
                    sprintf(featherTmpLocation, "Prototypes/%s", protoVecAttr.data()[0]);
                    featherLocation = std::string(featherTmpLocation);
                    sprintf(featherTmpLocation, "render/%s", protoVecAttr.data()[0]);
                    renderOutputLocation = std::string(featherTmpLocation);
                    // featherLocation = std::string("Prototypes/" + protoVecAttr.data()[0]);
                }
                else
                {
                    FnAttribute::StringAttribute typeAttr = interface.getAttr("type", childrenVectorAttr.data()[i]);
                    const std::string typeStr = typeAttr.getValue("", false);
                    if( !pystring::endswith(typeStr, "mesh") )
                        continue;
                    deformMeshLocation = std::string(childrenVectorAttr.data()[i]);
                }
            }

            // if(strcmp(featherLocation, "") == 0 || strcmp(deformMeshLocation, "") == 0)
                // return;

            /*
             motion blur list
             points
            */

            // get Attribute
            FnAttribute::FloatAttribute pointsAttr          = interface.getAttr("geometry.point.P", featherLocation);
            FnAttribute::IntAttribute numVerticesAttr       = interface.getAttr("geometry.numVertices", featherLocation);
            FnAttribute::FloatAttribute widthsAttr          = interface.getAttr("geometry.point.width", featherLocation);
            FnAttribute::GroupAttribute stGrpAttr           = interface.getAttr("geometry.arbitrary.st", featherLocation);
            FnAttribute::IntAttribute degreeAttr            = interface.getAttr("geometry.degree", featherLocation);

            // get Value
            FnAttribute::FloatConstVector pointsVectorAttr  = pointsAttr.getNearestSample(0.0f);
            size_t cvsSize                                  = pointsVectorAttr.size();
            std::vector<float> cvsArr(pointsVectorAttr.data(), pointsVectorAttr.data() + cvsSize);

            FnAttribute::IntConstVector numCVsVectorAttr    = numVerticesAttr.getNearestSample(0.0f);
            size_t numCVsSize                               = numCVsVectorAttr.size();
            std::vector<int> numCVsArr(numCVsVectorAttr.data(), numCVsVectorAttr.data() + numCVsSize);

            FnAttribute::FloatConstVector widthsVectorAttr  = widthsAttr.getNearestSample(0.0f);
            size_t widthsSize                               = widthsVectorAttr.size();
            std::vector<float> widthsArr(widthsVectorAttr.data(), widthsVectorAttr.data() + widthsSize);

            FnAttribute::FloatAttribute stAttr              = stGrpAttr.getChildByName("value");
            FnAttribute::FloatConstVector stVectorAttr      = stAttr.getNearestSample(0.0f);
            size_t stSize                                   = stVectorAttr.size();
            std::vector<float> stArr(stVectorAttr.data(), stVectorAttr.data() + stSize);

            // *************** Workflow - 2 ***************
            FnAttribute::FloatAttribute laminationAttr      = interface.getAttr("geometry.arbitrary.featherLamination.value", featherLocation);
            FnAttribute::FloatAttribute REParamsAttr        = interface.getAttr("geometry.arbitrary.featherREParams.value", featherLocation);
            FnAttribute::FloatAttribute UParamsAttr         = interface.getAttr("geometry.arbitrary.featherUParams.value", featherLocation);
            FnAttribute::FloatAttribute SeperationIdxAttr   = interface.getAttr("geometry.arbitrary.featherSeperationIdx.value", featherLocation);

            // get Value
            FnAttribute::FloatConstVector lamiVectorAttr    = laminationAttr.getNearestSample(0.0f);
            size_t lamiSize                                 = lamiVectorAttr.size();
            std::vector<float> laminationArr(lamiVectorAttr.data(), lamiVectorAttr.data() + lamiSize);

            FnAttribute::FloatConstVector REParamVectorAttr = REParamsAttr.getNearestSample(0.0f);
            size_t REParamSize                              = REParamVectorAttr.size();
            std::vector<float> REParamsArr(REParamVectorAttr.data(), REParamVectorAttr.data() + REParamSize);

            FnAttribute::FloatConstVector UParamVectorAttr  = UParamsAttr.getNearestSample(0.0f);
            size_t UParamSize                               = UParamVectorAttr.size();
            std::vector<float> UParamArr(UParamVectorAttr.data(), UParamVectorAttr.data() + UParamSize);

            FnAttribute::FloatConstVector SeperIdxVectorAttr= SeperationIdxAttr.getNearestSample(0.0f);
            size_t seperIdxSize                             = SeperIdxVectorAttr.size();
            std::vector<float> seperIdxArr(SeperIdxVectorAttr.data(), SeperIdxVectorAttr.data() + seperIdxSize);

            // *************** Workflow - 4 ***************
            std::vector<Dx::Point> dxCvsArr = std::vector<Dx::Point>();
            for (int i = 0; i < cvsArr.size(); i += 3)
            {
                dxCvsArr.push_back(Dx::Point(cvsArr[i], cvsArr[i + 1], cvsArr[i + 2]));
            }

            std::vector<Dx::Point> dxLamVtxArr = std::vector<Dx::Point>();
            for (int i = 0; i < laminationArr.size(); i += 3)
            {
                dxLamVtxArr.push_back(Dx::Point(laminationArr[i], laminationArr[i + 1], laminationArr[i + 2]));
            }

            Dx::SourceFeather sf = Dx::SourceFeather(dxLamVtxArr);

            int cvsIdx = 0;
            // rachis setup
            for (int rachisIdx = seperIdxArr[0]; rachisIdx < seperIdxArr[1]; ++rachisIdx)
            {
                for(int curveIdx = 0; curveIdx < numCVsArr[rachisIdx]; ++curveIdx)
                {
                    sf.m_vRachisCVs.push_back(dxCvsArr[cvsIdx]);
                    sf.m_vRachisUParams.push_back(UParamArr[cvsIdx]);
                    cvsIdx++;
                }
            }

            // barbs setup
            //  left
            int side = 0;
            for (int leftBarbsIdx = seperIdxArr[2]; leftBarbsIdx < seperIdxArr[3]; ++leftBarbsIdx)
            {
                sf.m_vBarbCVs[side].push_back(std::vector<Dx::Point>());
                sf.m_vBarbParams[side].push_back(Dx::Float2());
                sf.m_vBarbUParams[side].push_back(std::vector<float>());

                /*
                    barbsCvsIdx = leftBarbsIdx * curveIdx
                */
                for(int curveIdx = 0; curveIdx < numCVsArr[leftBarbsIdx]; curveIdx++)
                {
                    sf.m_vBarbCVs[side].back().push_back(dxCvsArr[cvsIdx]);
                    sf.m_vBarbUParams[side].back().push_back(UParamArr[cvsIdx]);
                    sf.m_vBarbParams[side].back() = Dx::Float2(REParamsArr[cvsIdx * 2 + 0], REParamsArr[cvsIdx * 2 + 1]);
                    cvsIdx++;
                }
            }

            side++;
            for (int rightBarbsCvsIdx = seperIdxArr[4]; rightBarbsCvsIdx < seperIdxArr[5]; ++rightBarbsCvsIdx)
            {
                sf.m_vBarbCVs[side].push_back(std::vector<Dx::Point>());
                sf.m_vBarbParams[side].push_back(Dx::Float2());
                sf.m_vBarbUParams[side].push_back(std::vector<float>());

                for(int curveIdx = 0; curveIdx < numCVsArr[rightBarbsCvsIdx]; curveIdx++)
                {
                    sf.m_vBarbCVs[side].back().push_back(dxCvsArr[cvsIdx]);
                    sf.m_vBarbUParams[side].back().push_back(UParamArr[cvsIdx]);
                    sf.m_vBarbParams[side].back() = Dx::Float2(REParamsArr[cvsIdx * 2 + 0], REParamsArr[cvsIdx * 2 + 1]);
                    cvsIdx++;
                }
            }
            sf.set();
            // Source Feather Set End

            // Deform & Motion blur
            // *************** Workflow - 3 ***************
            FnAttribute::FloatAttribute meshVertexAttr      = interface.getAttr("geometry.point.P", deformMeshLocation);

            // TimeSamples
            std::vector<float> timeSamples( FindTimeSamples(meshVertexAttr) );
            const size_t timeSamplesCount = timeSamples.size();

            std::vector<float> deformCombineCvsArr;
            size_t seperateMeshCount = 0;

            for(int i = 0; i < timeSamplesCount; i++)
            {
                // std::cout << "time Sample : " << timeSamples[i] << std::endl;
                // get Value
                FnAttribute::FloatConstVector meshVertexVectorAttr  = meshVertexAttr.getNearestSample(timeSamples[i]);
                size_t meshSize                                     = meshVertexVectorAttr.size();
                std::vector<float> meshVertexArr(meshVertexVectorAttr.data(), meshVertexVectorAttr.data() + meshSize);

                seperateMeshCount                                   = (meshSize / 3) / (lamiSize / 3); // total Mesh vertex Count / Lamination count

                // ======== Start Deforming ========

                for(int i = 0; i < seperateMeshCount; ++i)
                {
                    int startFaceIdx = ((lamiSize / 3) * i);
                    int endFaceIdx = ((lamiSize / 3) * (i + 1));
                    std::vector<Dx::Point> vtcs;

                    for(int j = startFaceIdx; j < endFaceIdx; j++)
                    {
                        vtcs.push_back(Dx::Point(meshVertexArr[j * 3 + 0], meshVertexArr[j * 3 + 1], meshVertexArr[j * 3 + 2]));
                    }

                    // $1
                    // std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ " << i << std::endl;

                    Dx::DeformFeather df = Dx::DeformFeather(sf, vtcs);

                    std::vector<float> deformedCvsArr;

                    df.deform(&deformedCvsArr);

                    deformCombineCvsArr.insert(deformCombineCvsArr.end(), deformedCvsArr.begin(), deformedCvsArr.end());
                }
            }

            int64_t pointsCount = deformCombineCvsArr.size() / timeSamplesCount;
            std::vector<const float*> pointPtrs;
            pointPtrs.reserve(timeSamplesCount);

            for (size_t i = 0; i < timeSamplesCount; ++i)
            {
                pointPtrs[i] = &(deformCombineCvsArr[i * pointsCount]);
            }

            // Etc Attribute
            // 1. Width
            // 2. st
            std::vector<float> combineWidthsArr;
            std::vector<float> combineStArr;
            std::vector<int> deformCombineNumCvsArr;

            for(int i = 0; i < seperateMeshCount; ++i)
            {
                int widthIdx = 0;
                for(int j = 0; j < numCVsSize; ++j)
                {
                    // root
                    combineWidthsArr.push_back(widthsArr[widthIdx]);
                    combineStArr.push_back(stArr[widthIdx * 2 + 0]);
                    combineStArr.push_back(stArr[widthIdx * 2 + 1]);

                    // org
                    for(int k = 0; k < numCVsArr[j]; ++k)
                    {
                        combineWidthsArr.push_back(widthsArr[widthIdx]);
                        combineStArr.push_back(stArr[widthIdx * 2 + 0]);
                        combineStArr.push_back(stArr[widthIdx * 2 + 1]);
                        widthIdx++;
                    }

                    // tip
                    combineWidthsArr.push_back(widthsArr[widthIdx - 1]);
                    combineStArr.push_back(stArr[(widthIdx - 1) * 2 + 0]);
                    combineStArr.push_back(stArr[(widthIdx - 1) * 2 + 1]);

                    deformCombineNumCvsArr.push_back(numCVsArr[j] + 2);
                }
            }

            FnAttribute::GroupBuilder gb;
            gb.set("scope", FnAttribute::StringAttribute("vertex"));
            gb.set("inputType", stGrpAttr.getChildByName("inputType"));
            gb.set("elementSize", stGrpAttr.getChildByName("elementSize"));
            gb.set("value", FnAttribute::FloatAttribute(combineStArr.data(), combineStArr.size(), 2));
            FnAttribute::GroupAttribute combineStGrpAttr = gb.build();

            FnGeolibServices::StaticSceneCreateOpArgsBuilder sscb(false);
            sscb.createEmptyLocation("render", "group");
            sscb.createEmptyLocation(renderOutputLocation, "curves");

            sscb.setAttrAtLocation(renderOutputLocation, "geometry.point.P", FnAttribute::FloatAttribute(timeSamples.data(), timeSamplesCount, pointPtrs.data(), pointsCount, 3));
            sscb.setAttrAtLocation(renderOutputLocation, "geometry.numVertices", FnAttribute::IntAttribute(deformCombineNumCvsArr.data(), deformCombineNumCvsArr.size(), 1));
            sscb.setAttrAtLocation(renderOutputLocation, "geometry.point.width", FnAttribute::FloatAttribute(combineWidthsArr.data(), combineWidthsArr.size(), 1));
            sscb.setAttrAtLocation(renderOutputLocation, "geometry.arbitrary.st", combineStGrpAttr);
            sscb.setAttrAtLocation(renderOutputLocation, "geometry.degree", degreeAttr);
            sscb.setAttrAtLocation(renderOutputLocation, "prmanStatements.basis.u", FnAttribute::StringAttribute("catmull-rom"));
            sscb.setAttrAtLocation(renderOutputLocation, "prmanStatements.basis.v", FnAttribute::StringAttribute("catmull-rom"));

            FnAttribute::GroupAttribute featherXformAttr = interface.getAttr("xform", deformMeshLocation);
            if (featherXformAttr.isValid())
                sscb.setAttrAtLocation(renderOutputLocation, "xform", featherXformAttr);

            // bodyST in mesh
            // but if rigged mesh, maybe don't have bodyST
            FnAttribute::GroupAttribute bodySTGrpAttr       = interface.getAttr("geometry.arbitrary.bodyST", deformMeshLocation);
            if (bodySTGrpAttr.isValid())
            {
                FnAttribute::FloatAttribute bodySTAttr      = bodySTGrpAttr.getChildByName("value");
                FnAttribute::FloatConstVector bodySTVectorAttr  = bodySTAttr.getNearestSample(0.0f);
                size_t bodySTSize                                   = bodySTVectorAttr.size();
                std::vector<float> bodySTArr(bodySTVectorAttr.data(), bodySTVectorAttr.data() + bodySTSize);
                std::vector<float> combineBodySTArr;

                // bodyST per strand
                int bodySTIdx = 0;
                int bodySTUnitCnt = 2 * ((lamiSize / 9) - 1);
                // std::cout << "-------bodyST : " << bodySTSize << ", " << bodySTUnitCnt << std::endl;
                for(int i = 0; i < deformCombineNumCvsArr.size(); i += numCVsSize)
                {
                    for(int j = 0; j < numCVsSize; ++j)
                    {
                        float bodyS = bodySTArr[bodySTIdx * 2 + 0];
                        float bodyT = bodySTArr[bodySTIdx * 2 + 1];
                        combineBodySTArr.push_back(bodyS);
                        combineBodySTArr.push_back(bodyT);
                    }
                    bodySTIdx += bodySTUnitCnt;
                }

                FnAttribute::GroupBuilder bodySTGroupBuilder;
                bodySTGroupBuilder.set("scope", FnAttribute::StringAttribute("face"));
                bodySTGroupBuilder.set("inputType", bodySTGrpAttr.getChildByName("inputType"));
                bodySTGroupBuilder.set("elementSize", bodySTGrpAttr.getChildByName("elementSize"));
                bodySTGroupBuilder.set("value", FnAttribute::FloatAttribute(combineBodySTArr.data(), combineBodySTArr.size(), 2));

                sscb.setAttrAtLocation(renderOutputLocation, "geometry.arbitrary.bodyST", bodySTGroupBuilder.build());
            }

            interface.execOp("StaticSceneCreate", sscb.build());
        }
    };

    DEFINE_GEOLIBOP_PLUGIN(DxDeformedFeatherOp)
}

void registerPlugins()
{
    REGISTER_PLUGIN(DxDeformedFeatherOp, "DxDeformedFeather", 0, 1);
}
