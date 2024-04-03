#ifndef _BS_MAYA_Common_h_
#define _BS_MAYA_Common_h_

#include <BeyondScreen.h>

#include <maya/MTime.h>
#include <maya/MGlobal.h>
#include <maya/MSyntax.h>
#include <maya/MMatrix.h>
#include <maya/MFnMesh.h>
#include <maya/MDagPath.h>
#include <maya/MPxCommand.h>
#include <maya/MPointArray.h>
#include <maya/MArgDatabase.h>
#include <maya/MThreadUtils.h>
#include <maya/MComputation.h>
#include <maya/MItMeshVertex.h>
#include <maya/MSelectionList.h>
#include <maya/MItMeshPolygon.h>
#include <maya/MPxLocatorNode.h>
#include <maya/MFnUnitAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnMatrixAttribute.h>
#include <maya/MFnNumericAttribute.h>

using namespace BeyondScreen;

bool GetWorldMatrix( const MObject& dagNodeObj, MMatrix& worldMat );

Vector Translation( const MMatrix& m );

Vector GetWorldPosition( const MString& xformNodeName );

Vector GetWorldUpvector( const MString& xformNodeName );

void ApplyXForm( const MMatrix& M, const MPoint& p, MPoint& q );

void Copy( VectorArray& to, const MPointArray& from );

void Copy( Matrix& to, const MMatrix& from );

bool Convert( ScreenMesh& mesh, MObject& meshObj, bool vPosOnly=false, const char* uvSetName=NULL );

MObject NodeNameToMObject( const MString& nodeName );

Vector Color( const MObject& nodeObj, const MObject& attrObj );

#endif

