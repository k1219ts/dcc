#include <usdKatana/attrMap.h>
#include <usdKatana/usdInPrivateData.h>
#include <usdKatana/utils.h>

#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/pointBased.h>

#include <pxr/usd/usdSkel/root.h>
#include <pxr/usd/usdSkel/binding.h>
#include <pxr/usd/usdSkel/bindingAPI.h>
#include <pxr/usd/usdSkel/skeleton.h>
#include <pxr/usd/usdSkel/skeletonQuery.h>
#include <pxr/usd/usdSkel/skinningQuery.h>
#include <pxr/usd/usdSkel/blendShapeQuery.h>

#include <vtKatana/array.h>

#include <FnAttribute/FnDataBuilder.h>
#include <FnGeolibServices/FnBuiltInOpArgsUtil.h>
#include <FnGeolib/util/Path.h>
#include <FnLogging/FnLogging.h>

#include <pystring/pystring.h>

#include "readSkelRoot.h"

PXR_NAMESPACE_OPEN_SCOPE

void
UsdKatanaReadSkelBinding(
    const UsdSkelRoot &skelRoot,
    const PxrUsdKatanaUsdInPrivateData &data,
    PxrUsdKatanaAttrMap &geomsAttrMap,
    PxrUsdKatanaAttrMap &inputAttrMap
)
{
    std::vector<std::string> usdPrimPathValues;
    for (UsdPrim child : skelRoot.GetPrim().GetChildren()) {
        usdPrimPathValues.push_back(child.GetPath().GetText());
    }

    UsdSkelCache _skelCache;
    _skelCache.Populate(skelRoot);

    std::vector<UsdSkelBinding> bindings;
    if (!_skelCache.ComputeSkelBindings(skelRoot, &bindings)) {
        return;
    }

    const auto currentTime = data.GetUsdInArgs()->GetCurrentTime();
    const std::string rootPath = skelRoot.GetPrim().GetPath().GetString();

    FnKat::GroupAttribute inputAttrs = inputAttrMap.build();
    const std::string outputPath  = FnKat::StringAttribute(inputAttrs.getChildByName("outputLocationPath")).getValue("", false);

    FnGeolibServices::StaticSceneCreateOpArgsBuilder geomsBldr(false);
    geomsBldr.setAttrAtLocation("", "usdPrimPath", FnKat::StringAttribute(usdPrimPathValues));

    for (const UsdSkelBinding &binding : bindings)
    {
        if (binding.GetSkinningTargets().empty()) continue;

        if (const UsdSkelSkeletonQuery &skelQuery = _skelCache.GetSkelQuery(binding.GetSkeleton())) {
            const UsdSkelAnimQuery &animQuery = skelQuery.GetAnimQuery();

            const std::vector<double>& motionSampleTimes = data.GetMotionSampleTimes();
            const bool isMotionBackward = data.IsMotionBackward();
            bool hasInfiniteBound = false;

            std::map<float, VtArray<GfMatrix4d>> xformsTimeSampleMap;
            std::map<float, VtFloatArray> weightsTimeSampleMap;
            for (double relSampleTime : motionSampleTimes) {
                double time = currentTime + relSampleTime;
                float correctedSampleTime = isMotionBackward ? PxrUsdKatanaUtils::ReverseTimeSample(relSampleTime) : relSampleTime;

                VtArray<GfMatrix4d> xforms;
                if (skelQuery.ComputeSkinningTransforms(&xforms, time)) {
                    xformsTimeSampleMap.insert({correctedSampleTime, xforms});
                }
                VtFloatArray weights;
                if (animQuery.ComputeBlendShapeWeights(&weights, time)) {
                    weightsTimeSampleMap.insert({correctedSampleTime, weights});
                }
            }

            if (!xformsTimeSampleMap.empty())
            {
                for (UsdSkelSkinningQuery const &skinQuery : binding.GetSkinningTargets()) {
                    UsdPrim const &skinnedPrim = skinQuery.GetPrim();

                    UsdGeomMesh const &skinnedMesh = UsdGeomMesh(skinnedPrim);
                    TfToken scheme;
                    skinnedMesh.GetSubdivisionSchemeAttr().Get(&scheme);
                    bool isNormal = false;
                    if (scheme == UsdGeomTokens->none) {
                        TfToken interp = skinnedMesh.GetNormalsInterpolation();
                        if (interp == UsdGeomTokens->varying || interp == UsdGeomTokens->vertex) {
                            isNormal = true;
                        }
                    }

                    std::vector<GfBBox3d> bounds;

                    std::map<float, VtArray<GfVec3f>> pointsTimeSampleMap, normalsTimeSampleMap;
                    for (auto it = xformsTimeSampleMap.begin(); it != xformsTimeSampleMap.end(); it++) {
                        VtArray<GfVec3f> points, normals;
                        skinnedMesh.GetPointsAttr().Get(&points);
                        if (isNormal) skinnedMesh.GetNormalsAttr().Get(&normals);
                        // Compute BlendShape
                        if (skinQuery.HasBlendShapes()) {
                            VtFloatArray blendShapeWeights = weightsTimeSampleMap.find(it->first)->second;
                            const UsdSkelBlendShapeQuery &blendShapeQuery = UsdSkelBlendShapeQuery(UsdSkelBindingAPI(skinnedPrim));
                            VtFloatArray weights;
                            if (skinQuery.GetBlendShapeMapper()->Remap(blendShapeWeights, &weights)) {
                                VtFloatArray subShapeWeights;
                                VtUIntArray blendShapeIndices, subShapeIndices;
                                if (blendShapeQuery.ComputeSubShapeWeights(weights, &subShapeWeights, &blendShapeIndices, &subShapeIndices)) {
                                    std::vector<VtIntArray> blendShapePointIndices;
                                    blendShapePointIndices = blendShapeQuery.ComputeBlendShapePointIndices();
                                    std::vector<VtVec3fArray> subShapePointOffsets;
                                    subShapePointOffsets = blendShapeQuery.ComputeSubShapePointOffsets();
                                    blendShapeQuery.ComputeDeformedPoints(
                                        subShapeWeights, blendShapeIndices, subShapeIndices, blendShapePointIndices, subShapePointOffsets, points
                                    );
                                    if (isNormal) {
                                        std::vector<VtVec3fArray> subShapeNormalOffsets;
                                        subShapeNormalOffsets = blendShapeQuery.ComputeSubShapeNormalOffsets();
                                        blendShapeQuery.ComputeDeformedNormals(
                                            subShapeWeights, blendShapeIndices, subShapeIndices, blendShapePointIndices, subShapeNormalOffsets, normals
                                        );
                                    }
                                }
                            }
                        }
                        if (skinQuery.ComputeSkinnedPoints(it->second, &points)) {
                            pointsTimeSampleMap.insert({it->first, points});
                            // Compute BoundingBox
                            VtArray<GfVec3f> extent(2);
                            UsdGeomPointBased::ComputeExtent(points, &extent);
                            bounds.push_back(GfBBox3d(GfRange3d(extent[0], extent[1])));
                        }
                        if (isNormal && skinQuery.ComputeSkinnedNormals(it->second, &normals)) {
                            // XXX RfK currently doesn't support uniform normals for polymeshes.
                            //  convert to faceVarying
                            VtIntArray vertsArray;
                            skinnedMesh.GetFaceVertexIndicesAttr().Get(&vertsArray);

                            VtArray<GfVec3f> faceVaryingNormals;
                            faceVaryingNormals.resize(vertsArray.size());
                            unsigned int i = 0;
                            for (VtArray<int>::iterator vi = vertsArray.begin(), end = vertsArray.end(); vi != end; ++vi, ++i) {
                                faceVaryingNormals[i][0] = normals[*vi][0];
                                faceVaryingNormals[i][1] = normals[*vi][1];
                                faceVaryingNormals[i][2] = normals[*vi][2];
                            }
                            normalsTimeSampleMap.insert({it->first, faceVaryingNormals});
                        }
                    }
                    if (!pointsTimeSampleMap.empty())
                    {
                        const std::string buildPath = skinnedPrim.GetPath().GetString();
                        std::string relBuildPath = pystring::replace(buildPath, rootPath + "/", "");
                        const std::string fullPath = outputPath + "/" + relBuildPath;

                        FnKat::DoubleAttribute boundsAttr = PxrUsdKatanaUtils::ConvertBoundsToAttribute(bounds, motionSampleTimes, isMotionBackward, &hasInfiniteBound);

                        FnGeolibServices::AttributeSetOpArgsBuilder asb;
                        asb.deleteAttr("xform");
                        asb.setLocationPaths(fullPath);
                        asb.setAttr("geometry.point.P", VtKatanaMapOrCopy<GfVec3f>(pointsTimeSampleMap));
                        if (isNormal) {
                            asb.deleteAttr("geometry.point.N");
                            asb.setAttr("geometry.vertex.N", VtKatanaMapOrCopy<GfVec3f>(normalsTimeSampleMap));
                        }
                        asb.setAttr("bound", boundsAttr);
                        geomsBldr.addSubOpAtLocation(relBuildPath, "AttributeSet", asb.build());
                    }
                }
            }
        }
    }

    FnKat::GroupAttribute geomsAttrs = geomsBldr.build();
    for (int64_t i = 0; i < geomsAttrs.getNumberOfChildren(); ++i)
    {
        geomsAttrMap.set(
            geomsAttrs.getChildName(i),
            geomsAttrs.getChildByIndex(i)
        );
    }

    _skelCache.Clear();
}

PXR_NAMESPACE_CLOSE_SCOPE
