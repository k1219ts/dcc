#ifndef _dxCombinationShapeNode
#define _dxCombinationShapeNode

// ==========================================================================
// Copyright Dexter Studios. All rights reserved.
// ==========================================================================

// File: dxCombinationShapeNode.h
//
// Dependency Graph Node: dxCombinationShape
//
// dxCombinationShapeNode


#include <maya/MPxNode.h>
#include <maya/MTypeId.h>
#include <maya/MPlug.h>
#include <maya/MDataBlock.h>


class dxCombinationShape : public MPxNode
{
public:
    dxCombinationShape();
    ~dxCombinationShape() override;

    MStatus compute( const MPlug& plug, MDataBlock& data ) override;

    static void*    creator();
    static MStatus  initialize();

public:
    static MObject  aInput;

    static MObject  aMinWeight;
    static MObject  aMaxWeight;
    static MObject  aMidWeight;
    static MObject  aInWeight;

    static MObject  aOutWeight;

    static MTypeId  id;

private:
    float getCurveValue(float w0, float w1, float f);
}
