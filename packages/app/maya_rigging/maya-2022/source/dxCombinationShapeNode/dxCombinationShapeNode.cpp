// ==========================================================================
// Copyright Dexter Studios. All rights reserved.
// ==========================================================================

// File: dxCombinationShapeNode.cpp
// Created on: 2021. 3. 19.
// Author: taewan.kim
//
// Dependency Graph Node: dxCombinationShape


#include "dxCombinationShapeNode.h"

#include <maya/MFnNumericAttribute.h>
#include <maya/MFnCompoundAttribute.h>

#include <maya/MArrayDataHandle.h>
#include <maya/MFnPlugin.h>
#include <maya/MGlobal.h>


MTypeId dxCombinationShape::id( 0x10400001 );

// attributes

MObject  dxCombinationShape::aInput;

MObject  dxCombinationShape::aMinWeight;
MObject  dxCombinationShape::aMaxWeight;
MObject  dxCombinationShape::aMidWeight;
MObject  dxCombinationShape::aInWeight;

MObject  dxCombinationShape::aOutWeight;

dxCombinationShape::dxCombinationShape() {}
dxCombinationShape::~dxCombinationShape() {}


MStatus dxCombinationShape::compute( const MPlug& plug, MDataBlock& dataBlock )
{
    MStatus stat;

    if ( plug == aInput || plug == aOutWeight ) {
        MArrayDataHandle inArray = dataBlock.inputArrayValue(aInput, &stat);
        if( stat != MS::kSuccess ) {
            MGlobal::displayError( "ERROR: cannot get input\n" );
            return MS::kFailure;
        }

        float outVal = 1.0;
        for (unsigned int i = 0; i < inArray.elementCount(); i++) {
            MDataHandle inData = inArray.inputValue();
            float minVal = inData.child(aMinWeight).asFloat();
            float maxVal = inData.child(aMaxWeight).asFloat();
            float inVal = inData.child(aInWeight).asFloat();
            if ( minVal <= inVal && inVal <= maxVal ) {
                float midVal = inData.child(aMidWeight).asFloat();
                if ( inVal != midVal ) {
                    float w0 = (inVal < midVal) ? minVal : maxVal;
                    outVal *= getCurveValue(w0, midVal, inVal);
                }
                inArray.next();
            } else {
                outVal = 0.0f;
                break;
            }
        }

        MDataHandle outData = dataBlock.outputValue(aOutWeight);
        outData.setFloat(outVal);
    } else {
        return MS::kUnknownParameter;
    }

    dataBlock.setClean(plug);
    return MS::kSuccess;
}

float dxCombinationShape::getCurveValue(float w0, float w1, float f)
{
    float x = (f - w0) / (w1 - w0);
    return x * x * (-2.0f * x + 3.0f);
}


void* dxCombinationShape::creator() {
    return new dxCombinationShape;
}


MStatus dxCombinationShape::initialize()
{
    MFnNumericAttribute numAttr;
    aMinWeight = numAttr.create("minWeight", "min", MFnNumericData::kFloat);
    CHECK_MSTATUS( numAttr.setStorable(true) );
    CHECK_MSTATUS( numAttr.setKeyable(true) );

    aMaxWeight = numAttr.create("maxWeight", "max", MFnNumericData::kFloat, 1.0f);
    CHECK_MSTATUS( numAttr.setStorable(true) );
    CHECK_MSTATUS( numAttr.setKeyable(true) );

    aMidWeight = numAttr.create("midWeight", "mid", MFnNumericData::kFloat, .5f);
    CHECK_MSTATUS( numAttr.setStorable(true) );
    CHECK_MSTATUS( numAttr.setKeyable(true) );

    aInWeight = numAttr.create("inWeight", "in", MFnNumericData::kFloat);
    CHECK_MSTATUS( numAttr.setStorable(true) );
    CHECK_MSTATUS( numAttr.setKeyable(true) );
    CHECK_MSTATUS( numAttr.setWritable(true) );

    MFnCompoundAttribute compAttr;
    aInput = compAttr.create("input", "input");
    CHECK_MSTATUS( compAttr.addChild(aMinWeight) );
    CHECK_MSTATUS( compAttr.addChild(aMaxWeight) );
    CHECK_MSTATUS( compAttr.addChild(aMidWeight) );
    CHECK_MSTATUS( compAttr.addChild(aInWeight) );
    CHECK_MSTATUS( compAttr.setArray(true) );
    CHECK_MSTATUS( compAttr.setHidden(false) );
    CHECK_MSTATUS( compAttr.setDisconnectBehavior(MFnAttribute::kDelete) );

    aOutWeight = numAttr.create("outWeight", "out", MFnNumericData::kFloat);
    CHECK_MSTATUS( numAttr.setDefault(0.0f) );
    CHECK_MSTATUS( numAttr.setStorable(false) );
    CHECK_MSTATUS( numAttr.setWritable(false) );
    CHECK_MSTATUS( numAttr.setReadable(true) );

    CHECK_MSTATUS( addAttribute(aInput) );
    CHECK_MSTATUS( addAttribute(aOutWeight) );

    CHECK_MSTATUS( attributeAffects(aMinWeight, aOutWeight) );
    CHECK_MSTATUS( attributeAffects(aMaxWeight, aOutWeight) );
    CHECK_MSTATUS( attributeAffects(aMidWeight, aOutWeight) );
    CHECK_MSTATUS( attributeAffects(aInWeight, aOutWeight) );
    CHECK_MSTATUS( attributeAffects(aInput, aOutWeight) );

    return MS::kSuccess;
}


// -----------------------------------------------------------------------------

MStatus initializePlugin( MObject obj )
{
    MStatus status;
    const MString UserClassify( "utility/general" );
    MFnPlugin plugin( obj, "Dexter Studios", "1.0", "Any" );
    status = plugin.registerNode( "dxCombinationShape",
                                  dxCombinationShape::id,
                                  &dxCombinationShape::creator,
                                  &dxCombinationShape::initialize,
								  MPxNode::kDependNode,
								  &UserClassify );
    if (!status) {
        status.perror("register node");
        return status;
    }

    return status;
}

MStatus uninitializePlugin( MObject obj )
{
    MStatus     status;
    MFnPlugin   plugin( obj );

    status = plugin.deregisterNode( dxCombinationShape::id );
    if (!status) {
        status.perror("deregister node");
        return status;
    }

    return status;
}

