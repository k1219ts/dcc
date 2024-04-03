
#include "skinningUsdSkel.h"

#include <usdKatana/attrMap.h>
#include <usdKatana/readPrim.h>
#include <usdKatana/readGprim.h>
#include <usdKatana/readMesh.h>
#include <usdKatana/usdInPrivateData.h>
#include <usdKatana/readXformable.h>
#include <usdKatana/utils.h>

#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/mesh.h>

#include <pxr/usd/usdSkel/root.h>
#include <pxr/usd/usdSkel/skeleton.h>
#include <pxr/usd/usdSkel/skeletonQuery.h>
#include <pxr/usd/usdSkel/skinningQuery.h>
#include <vtKatana/array.h>

#include <FnAttribute/FnDataBuilder.h>
#include <FnGeolibServices/FnBuiltInOpArgsUtil.h>
#include <FnGeolib/util/Path.h>
#include <FnLogging/FnLogging.h>

#include <pystring/pystring.h>

#include <iostream>

PXR_NAMESPACE_OPEN_SCOPE

static FnKat::DoubleAttribute _MakeBoundsAttribute(
    const UsdPrim& prim,
    const PxrUsdKatanaUsdInPrivateData& data
) {
    if (prim.GetPath() == SdfPath::AbsoluteRootPath()) {
        return FnKat::DoubleAttribute();
    }
    const std::vector<double>& motionSampleTimes = data.GetMotionSampleTimes();
    std::vector<GfBBox3d> bounds = data.GetUsdInArgs()->ComputeBounds(prim, motionSampleTimes);

    bool hasInfiniteBound = false;
    bool isMotionBackward = motionSampleTimes.size() > 1 && motionSampleTimes.front() > motionSampleTimes.back();
    FnKat::DoubleAttribute boundsAttr = PxrUsdKatanaUtils::ConvertBoundsToAttribute(
        bounds, motionSampleTimes, isMotionBackward, &hasInfiniteBound
    );
    return boundsAttr;
}

void DxReadSkelRootGeom(
    FnKat::GeolibCookInterface& interface,
    const UsdPrim& rootPrim,
    const PxrUsdKatanaUsdInPrivateData& data,
    PxrUsdKatanaAttrMap& inputAttrMap
) {
    const UsdPrim& geoPrim = rootPrim.GetChild(TfToken("Geometry"));
    if (!geoPrim) return;

    const UsdSkelRoot& skelRoot = UsdSkelRoot(rootPrim);

    UsdSkelCache &_skelCache = data.GetUsdInArgs()->GetUsdSkelCache();
    _skelCache.Populate(skelRoot);

    std::vector<UsdSkelBinding> bindings;
    if (!_skelCache.ComputeSkelBindings(skelRoot, &bindings)) {
        return;
    }

    UsdStageRefPtr stage = UsdStage::CreateInMemory();
    UsdGeomXform tmpRoot = UsdGeomXform::Define(stage, SdfPath("/Geometry"));

    const auto currentTime = data.GetUsdInArgs()->GetCurrentTime();

    const std::string rootPath = skelRoot.GetPrim().GetPath().GetString();

    FnKat::GroupAttribute inputAttrs = inputAttrMap.build();
    const std::string katOutputPath  = FnKat::StringAttribute(inputAttrs.getChildByName("outputLocationPath")).getValue("", false);


    FnGeolibServices::StaticSceneCreateOpArgsBuilder sscb(false);
    sscb.setAttrAtLocation("", "usdPrimPath", FnKat::StringAttribute(geoPrim.GetPath().GetString()));
    sscb.setAttrAtLocation("", "usdPrimName", FnKat::StringAttribute("Geometry"));


    for (const UsdSkelBinding &binding : bindings) {
        if (binding.GetSkinningTargets().empty())
            continue;

        if (const UsdSkelSkeletonQuery &skelQuery = _skelCache.GetSkelQuery(binding.GetSkeleton())) {
            const std::vector<double>& motionSampleTimes = data.GetMotionSampleTimes();
            const bool isMotionBackward = data.IsMotionBackward();

            std::map<float, VtArray<GfMatrix4d>> xformsTimeToSampleMap;
            for (double relSampleTime : motionSampleTimes) {
                double time = currentTime + relSampleTime;

                VtArray<GfMatrix4d> xforms;
                if (skelQuery.ComputeSkinningTransforms(&xforms, time)) {
                    float correctedSampleTime =
                        isMotionBackward
                            ? PxrUsdKatanaUtils::ReverseTimeSample(relSampleTime)
                            : relSampleTime;
                    xformsTimeToSampleMap.insert({correctedSampleTime, xforms});
                }
            }

            if (!xformsTimeToSampleMap.empty()) {
                for (const auto &skinningQuery : binding.GetSkinningTargets()) {
                    const UsdPrim& skinnedPrim = skinningQuery.GetPrim();
                    const UsdGeomMesh& skinnedMesh = UsdGeomMesh(skinnedPrim);
                    //
                    UsdGeomMesh mesh = UsdGeomMesh::Define(stage, tmpRoot.GetPath().AppendChild(skinnedPrim.GetName()));
                    mesh.CreatePointsAttr();

                    std::map<float, VtArray<GfVec3f>> pointsTimeToSampleMap;
                    for (auto it=xformsTimeToSampleMap.begin(); it!=xformsTimeToSampleMap.end(); it++) {
                        VtArray<GfVec3f> points;
                        skinnedMesh.GetPointsAttr().Get(&points);
                        if (skinningQuery.ComputeSkinnedPoints(it->second, &points)) {
                            pointsTimeToSampleMap.insert({it->first, points});
                            //
                            const auto pointsAttr = mesh.GetPointsAttr();
                            pointsAttr.Set(points, it->first);
                        }
                    }

                    if (!pointsTimeToSampleMap.empty()) {
                        const std::string buildPath = skinnedPrim.GetPath().GetString();

                        std::string relBuildPath = pystring::replace(buildPath, rootPath + "/", "");
                        const std::string fullPath = katOutputPath + "/" + relBuildPath;

                        FnGeolibServices::AttributeSetOpArgsBuilder asb;
                        asb.deleteAttr("xform");
                        asb.setLocationPaths(fullPath);
                        asb.setAttr("geometry.point.P", VtKatanaMapOrCopy<GfVec3f>(pointsTimeToSampleMap));
                        asb.setAttr("bound", _MakeBoundsAttribute(mesh.GetPrim(), data));
                        sscb.addSubOpAtLocation(relBuildPath, "AttributeSet", asb.build());
                    }
                }
            }
        }
    }

    // Geometry Bound
    const std::string geofullPath = katOutputPath + "/Geometry";
    FnGeolibServices::AttributeSetOpArgsBuilder asb;
    asb.deleteAttr("xform");
    asb.setLocationPaths(geofullPath);
    FnKat::DoubleAttribute rootBound = _MakeBoundsAttribute(tmpRoot.GetPrim(), data);
    asb.setAttr("bound", rootBound);
    sscb.addSubOpAtLocation("Geometry", "AttributeSet", asb.build());

    PxrUsdKatanaUsdInArgsRefPtr usdInArgs = data.GetUsdInArgs();
    interface.execOp(
        "PxrUsdIn.BuildIntermediate",
        FnKat::GroupBuilder()
            .update(inputAttrs.getChildByName("opArgs"))
            .set("staticScene", sscb.build())
            .build(),
            new PxrUsdKatanaUsdInPrivateData(usdInArgs->GetRootPrim(), usdInArgs, &data),
            PxrUsdKatanaUsdInPrivateData::Delete
    );

    interface.setAttr("bound", rootBound);

    _skelCache.Clear();
}


PXR_NAMESPACE_CLOSE_SCOPE
