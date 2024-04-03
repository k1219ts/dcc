#include <pystring/pystring.h>
#include <FnGeolib/op/FnGeolibOp.h>
#include <FnGeolibServices/FnGeolibCookInterfaceUtilsService.h>
#include <FnAttribute/FnAttribute.h>
#include <FnAttribute/FnGroupBuilder.h>
#include <FnPluginSystem/FnPlugin.h>

#include <stdint.h>
#include <iostream>
#include "CurvesUtils.h"

namespace
{
    class CurvesUtilsOp : public Foundry::Katana::GeolibOp
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
            // type
            FnAttribute::StringAttribute typeAttr = interface.getAttr("type");
            const std::string typeStr = typeAttr.getValue("", false);
            if( !pystring::startswith(typeStr, "curves") )
                return;

            //------------------------------------------------------------------
            // Get Parameters
            FnAttribute::FloatAttribute rootWidthScaleAttr = interface.getOpArg("rootWidthScale");
            const float rootWidthScale = rootWidthScaleAttr.getValue(1.0f, false);

            FnAttribute::FloatAttribute tipWidthScaleAttr = interface.getOpArg("tipWidthScale");
            const float tipWidthScale = tipWidthScaleAttr.getValue(1.0f, false);

            FnAttribute::FloatAttribute curvesRatioAttr = interface.getOpArg("curvesRatio");
            const float curvesRatio = curvesRatioAttr.getValue(1.0f, false);
            //------------------------------------------------------------------

            //
            // main compute
            //
            int interval = (int)(1 / curvesRatio);

            if( curvesRatio != 1.0f )
            {
                // compute ratio
                geomRatio(interface, interval, rootWidthScale, tipWidthScale);
            }
            else
            {
                // compute only width scale
                if( (rootWidthScale != 1.0f) || (tipWidthScale != 1.0f) )
                {
                    geomWidthScale(interface, rootWidthScale, tipWidthScale);
                }
            }
        }

    protected:

        static void geomWidthScale(FnKat::GeolibCookInterface &interface, const float rootScale, const float tipScale)
        {
            FnAttribute::FloatAttribute currWidthAttr = interface.getAttr("geometry.point.width");
            if( currWidthAttr.isValid() )
            {
                // numVertices
                FnAttribute::IntAttribute currNumVerticesAttr = interface.getAttr("geometry.numVertices");
                FnAttribute::IntConstVector currNumVerticesVec= currNumVerticesAttr.getNearestSample(0.0f);
                const int64_t currNumVerticesSize = currNumVerticesVec.size();
                const int* currNumVertices = currNumVerticesVec.data();

                // width scale
                FnAttribute::FloatConstVector currWidthVec = currWidthAttr.getNearestSample(0.0f);
                const float* currWidth = currWidthVec.data();

                std::vector<float> vec_width;

                int64_t vtxIndex = 0;
                for( int64_t i=0; i<currNumVerticesSize; i++ )
                {
                    int numVtx = currNumVertices[i];
                    IterSetScaleWidth(vec_width, currWidth, vtxIndex, numVtx, rootScale, tipScale);

                    vtxIndex += numVtx;
                }

                FnAttribute::FloatAttribute widthAttr(vec_width.data(), vec_width.size(), 1);
                interface.setAttr("geometry.point.width", widthAttr, false);
            }
        }

        static void geomRatio(FnKat::GeolibCookInterface &interface, int interval, const float rootScale, const float tipScale)
        {
            // numVertices
            FnAttribute::IntAttribute currNumVerticesAttr = interface.getAttr("geometry.numVertices");
            FnAttribute::IntConstVector currNumVerticesVec= currNumVerticesAttr.getNearestSample(0.0f);
            const int64_t currNumVerticesSize = currNumVerticesVec.size();
            const int* currNumVertices = currNumVerticesVec.data();

            // width
            FnAttribute::FloatAttribute currWidthAttr = interface.getAttr("geometry.point.width");
            if( currWidthAttr.isValid() )
            {
                widthRatio(interface, currWidthAttr, currNumVerticesSize, currNumVertices, interval, rootScale, tipScale);
            }

            // arbitrary
            FnAttribute::GroupBuilder arbBuilder;
            FnAttribute::GroupAttribute arbGrpAttr = interface.getAttr("geometry.arbitrary");
            if( arbGrpAttr.isValid() )
            {
                arbitraryRatio(arbBuilder, arbGrpAttr, currNumVerticesSize, interval);
                interface.setAttr("geometry.arbitrary", arbBuilder.build());
            }

            // P
            FnAttribute::FloatAttribute currPointsAttr = interface.getAttr("geometry.point.P");
            // TimeSamples
            std::vector<float> timeSamples( FindTimeSamples(currPointsAttr) );
            const size_t timeSamplesCount = timeSamples.size();
            std::vector<float> pointSamples;
            pointSamples.reserve(timeSamplesCount);

            // new values
            std::vector<int> vec_numVertices;
            std::vector<float> vec_points;
            std::vector<const float*> pointPtrs;
            pointPtrs.reserve(timeSamplesCount);

            for( std::vector<float>::const_iterator it = timeSamples.begin(); it != timeSamples.end(); ++it )
            {
                const float time = *it;
                pointSamples.push_back(time);

                FnAttribute::FloatConstVector currPointsVec = currPointsAttr.getNearestSample(time);
                const float* currPoints = currPointsVec.data();

                int64_t vtxIndex = 0;
                for( int64_t i=0; i<currNumVerticesSize; i++ )
                {
                    int numVtx = currNumVertices[i];
                    if( (i % interval) == 0 )
                    {
                        if( it == timeSamples.begin() ) vec_numVertices.push_back(numVtx);
                        for( int x=0; x<numVtx; x++ )
                        {
                            for( int y=0; y<3; y++ )
                            {
                                vec_points.push_back(currPoints[(vtxIndex*3) + (x*3+y)]);
                            }
                        }
                    }

                    vtxIndex += numVtx;
                }
            }

            int64_t pointsCount = vec_points.size() / timeSamplesCount;
            for( size_t i=0; i<timeSamplesCount; i++ )
            {
                pointPtrs[i] = &(vec_points[i*pointsCount]);
            }

            // Set
            //  numVertices
            FnAttribute::IntAttribute numVerticesAttr(vec_numVertices.data(), vec_numVertices.size(), 1);
            interface.setAttr("geometry.numVertices", numVerticesAttr);
            //  P
            FnAttribute::FloatAttribute pointsAttr(pointSamples.data(), timeSamplesCount, pointPtrs.data(), pointsCount, 3);
            interface.setAttr("geometry.point.P", pointsAttr);
        }

        static void widthRatio(FnKat::GeolibCookInterface &interface, FnAttribute::FloatAttribute currWidthAttr, int64_t count, const int* numVertices, int interval, const float rootScale, const float tipScale)
        {
            FnAttribute::FloatConstVector currWidthVec = currWidthAttr.getNearestSample(0.0f);
            const float* currWidth = currWidthVec.data();

            // new value
            std::vector<float> vec_width;

            int64_t vtxIndex = 0;
            for( int64_t i=0; i<count; i++ )
            {
                int numVtx = numVertices[i];
                if( (i % interval) == 0 )
                {
                    IterSetScaleWidth(vec_width, currWidth, vtxIndex, numVtx, rootScale, tipScale);
                }

                vtxIndex += numVtx;
            }

            FnAttribute::FloatAttribute widthAttr(vec_width.data(), vec_width.size(), 1);
            interface.setAttr("geometry.point.width", widthAttr, false);
        }

        static void arbitraryRatio(FnAttribute::GroupBuilder& builder, FnAttribute::GroupAttribute grpAttr, int64_t count, int interval)
        {
            int64_t childCount = grpAttr.getNumberOfChildren();
            for( int64_t c=0; c<childCount; c++ )
            {
                FnAttribute::GroupAttribute arb = grpAttr.getChildByIndex(c);

                // new group attribute
                FnAttribute::GroupBuilder gb;

                // name
                const std::string name = grpAttr.getChildName(c);
                // scope
                const std::string scope = ((FnAttribute::StringAttribute)arb.getChildByName("scope")).getValue();
                gb.set("scope", FnAttribute::StringAttribute(scope));
                // inputType
                const std::string inputType = ((FnAttribute::StringAttribute)arb.getChildByName("inputType")).getValue();
                gb.set("inputType", FnAttribute::StringAttribute(inputType));
                // elementSize
                int elementSize = 1;
                FnAttribute::IntAttribute elementSizeAttr = arb.getChildByName("elementSize");
                if( elementSizeAttr.isValid() )
                {
                    elementSize = elementSizeAttr.getValue();
                    gb.set("elementSize", FnAttribute::IntAttribute(elementSize));
                }

                // value
                if( pystring::startswith(inputType, "float") )
                {
                    std::vector<float> vec_value;
                    FnAttribute::FloatAttribute valueAttr = arb.getChildByName("value");
                    FnAttribute::FloatConstVector valueVec= valueAttr.getNearestSample(0.0f);
                    const float* value = valueVec.data();

                    for( int64_t i=0; i<count; i+=interval )
                    {
                        for( int x=0; x<elementSize; x++ )
                        {
                            vec_value.push_back(value[(i*elementSize)+x]);
                        }
                    }

                    FnAttribute::FloatAttribute nAttr(vec_value.data(), vec_value.size(), elementSize);
                    gb.set("value", nAttr);
                }
                else if( pystring::startswith(inputType, "string") )
                {
                    gb.set("value", arb.getChildByName("value"));
                }

                builder.set(name, gb.build());
            }
        }
    };

    DEFINE_GEOLIBOP_PLUGIN(CurvesUtilsOp)
}

void registerPlugins()
{
    REGISTER_PLUGIN(CurvesUtilsOp, "CurvesUtils", 0, 1);
}
