#include <readNurbsCurves.h>

#include <usdKatana/attrMap.h>
#include <usdKatana/readPrim.h>
#include <usdKatana/readGprim.h>
#include <usdKatana/readXformable.h>
#include <usdKatana/utils.h>
#include <vtKatana/array.h>

#include <pxr/usd/usdGeom/nurbsCurves.h>

#include <FnAttribute/FnDataBuilder.h>


PXR_NAMESPACE_OPEN_SCOPE

template <typename T>
VtArray<T> ConvertNurbsAttr(const VtArray<T>& value) {
    VtArray<T> tempArray;

    tempArray.push_back(value.front());
    tempArray.push_back(value.front());
    for (typename VtArray<T>::const_iterator i = value.begin(), end = value.end(); i != end; ++i) {
        tempArray.push_back(*i);
    }
    tempArray.push_back(value.back());
    tempArray.push_back(value.back());

    return tempArray;
}

template <typename T_USD, typename T_ATTR> FnKat::Attribute
_ConvertNurbsAttr(
    const UsdAttribute& usdAttr,
    const int tupleSize,
    const UsdKatanaUsdInPrivateData& data
) {
    if (!usdAttr.HasValue())
    {
        return FnKat::Attribute();
    }

    const double currentTime = data.GetCurrentTime();
    const std::vector<double>& motionSampleTimes = data.GetMotionSampleTimes(usdAttr);

    bool varyingTopology = false;

    int arraySize = -1;

    const bool isMotionBackward = data.IsMotionBackward();

    std::map<float, VtArray<T_USD>> timeToSampleMap;
    for (double relSampleTime : motionSampleTimes) {
        double time = currentTime + relSampleTime;

        VtArray<T_USD> getVal;
        usdAttr.Get(&getVal, time);

        // add for null data
        if (arraySize == -1) {
            arraySize = getVal.size();
        } else if ( getVal.size() != static_cast<size_t>(arraySize) ) {
            varyingTopology = true;
            break;
        }

        VtArray<T_USD> attrArray = ConvertNurbsAttr<T_USD>(getVal);

        if (!timeToSampleMap.empty()) {
            if (timeToSampleMap.begin()->second.size() != attrArray.size()) {
                timeToSampleMap.clear();
                varyingTopology = true;
                break;
            }
        }
        float correctedSampleTime =
            isMotionBackward
                ? UsdKatanaUtils::ReverseTimeSample(relSampleTime)
                : relSampleTime;
        timeToSampleMap.insert({correctedSampleTime, attrArray});
    }

    if (varyingTopology) {
        VtArray<T_USD> getVal;
        usdAttr.Get(&getVal, currentTime);
        VtArray<T_USD> attrArray = ConvertNurbsAttr<T_USD>(getVal);
        return VtKatanaMapOrCopy<T_USD>(attrArray);
    } else {
        return VtKatanaMapOrCopy<T_USD>(timeToSampleMap);
    }
}

FnKat::Attribute NurbsGeomGetPAttr(
    const UsdGeomPointBased& points,
    const UsdKatanaUsdInPrivateData& data
) {
    return _ConvertNurbsAttr<GfVec3f, FnKat::FloatAttribute>(points.GetPointsAttr(), 3, data);
}


void readUSDNurbsCurves(
    FnKat::GeolibCookInterface& interface, FnKat::GroupAttribute opArgs,
    const UsdKatanaUsdInPrivateData& privateData
) {
    const auto prim = privateData.GetUsdPrim();
    const auto nurbsCurves = UsdGeomNurbsCurves(prim);

    UsdKatanaAttrMap attrs;
    UsdKatanaReadPrim(prim, privateData, attrs);

    double currentTime = privateData.GetCurrentTime();

    attrs.set("geometry.point.P", NurbsGeomGetPAttr(nurbsCurves, privateData));

    VtIntArray vtxCts;
    nurbsCurves.GetCurveVertexCountsAttr().Get(&vtxCts, currentTime);
    if (!vtxCts.empty())
        attrs.set("geometry.numVertices", FnKat::IntAttribute(vtxCts[0] + 4));

    VtArray<double> getKnots;
    nurbsCurves.GetKnotsAttr().Get(&getKnots, currentTime);
    if (!getKnots.empty()) {
        VtArray<double> setKnots = ConvertNurbsAttr<double>(getKnots);
        attrs.set("geometry.knots", VtKatanaMapOrCopy(setKnots));
    }

    // VtFloatArray widths;
    VtArray<float> widths;
    if (nurbsCurves.GetWidthsAttr().Get(&widths, currentTime))
    {
        size_t numWidths = widths.size();
        if (numWidths == 1)
        {
            attrs.set("geometry.constantWidth", FnKat::FloatAttribute(widths[0]));
        }
        else if (numWidths > 1)
        {
            VtArray<float> setWidths = ConvertNurbsAttr<float>(widths);
            attrs.set("geometry.point.width", VtKatanaMapOrCopy(setWidths));
        }
    }

    attrs.set("geometry.degree", FnKat::IntAttribute(3));
    attrs.set("geometry.basis", FnKat::IntAttribute(2));
    attrs.set("type", FnKat::StringAttribute("curves"));
    attrs.toInterface(interface);

    FnKat::GroupAttribute xform;
    if (UsdKatanaReadXformable(nurbsCurves, privateData, xform)) {
        interface.setAttr("xform", xform);
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
