// ==========================================================================
// Copyright Dexter Studios. All rights reserved.
// ==========================================================================

// File: dxCombinationShapeNode.h
// Created on: 2021. 3. 19.
// Author: taewan.kim

// Dependency Graph Node: dxCombinationShape

#ifndef INCLUDE_DXCOMBINATIONSHAPENODE_H_
#define INCLUDE_DXCOMBINATIONSHAPENODE_H_


#include <maya/MPxNode.h>


class dxCombinationShape : public MPxNode
{
public:
    dxCombinationShape();
    ~dxCombinationShape() override;

    MStatus compute( const MPlug& plug, MDataBlock& data ) override;

    static void*    creator();
    static MStatus  initialize();

    static MTypeId  id;
    static MObject  aInput;

    static MObject  aMinWeight;
    static MObject  aMaxWeight;
    static MObject  aMidWeight;
    static MObject  aInWeight;

    static MObject  aOutWeight;

    float getCurveValue(float w0, float w1, float f);
};

#endif /* INCLUDE_DXCOMBINATIONSHAPENODE_H_ */
