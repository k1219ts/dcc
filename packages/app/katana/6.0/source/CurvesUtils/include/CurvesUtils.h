#ifndef CURVESUTILS_H
#define CURVESUTILS_H

#include <FnAttribute/FnAttribute.h>

#include <vector>
#include <iostream>

std::vector<float> FindTimeSamples(const FnAttribute::DataAttribute& attr)
{
    std::vector<float> timeSamples;
    if( !attr.isValid() )
        return timeSamples;

    int64_t count = attr.getNumberOfTimeSamples();
    timeSamples.reserve(count);

    for( int64_t i=0; i<count; ++i )
    {
        timeSamples.push_back(attr.getSampleTime(i));
    }
    return timeSamples;
}


static void IterSetScaleWidth(
    std::vector<float>& dest, const float* src, int64_t startIndex,
    int numVtx, float rootScale, float tipScale
)
{
    if( rootScale != tipScale )
    {
        if( numVtx > 3 )
        {
            dest.push_back(src[startIndex] * rootScale);

            float step = (rootScale - tipScale) / (numVtx - 3);
            for( int i=1; i<numVtx-1; i++ )
            {
                int64_t index = startIndex + i;
                dest.push_back(src[index] * (rootScale - (step * (i-1))));
            }

            dest.push_back(src[startIndex+numVtx-1] * tipScale);
        }
        else
        {
            for( int i=0; i<numVtx; i++ )
            {
                dest.push_back(src[startIndex+i] * rootScale);
            }
        }
    }
    else
    {
        for( int i=0; i<numVtx; i++ )
        {
            dest.push_back(src[startIndex+1] * rootScale);
        }
    }
}

#endif
