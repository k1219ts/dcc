//-
// ==========================================================================
// Copyright 1995,2006,2008 Autodesk, Inc. All rights reserved.
//
// Use of this software is subject to the terms of the Autodesk
// license agreement provided at the time of installation or download,
// or which otherwise accompanies this software in either electronic
// or hard copy form.
// ==========================================================================
//+

////////////////////////////////////////////////////////////////////////
// 
// DESCRIPTION:
// 
// Produces the MEL command "getAttrAffects".
//
// This command takes the name of a node as an argument.
// It then iterates over each attribute of the node and prints a list of attributes that it affects
// and the ones that affect it. 
//
// To use it, issue the command "getAttrAffects nodeName", where "nodeName" is the name of the node
// whose attributes you want to display. If invoked with no arguments, "getAttrAffects" will
// display the attribute info of all the selected nodes. 
// 
////////////////////////////////////////////////////////////////////////

#include <maya/MPxCommand.h>
#include <maya/MArgList.h>
#include <maya/MSyntax.h>
#include <maya/MString.h>
#include <maya/MSelectionList.h>

#include <maya/MArgDatabase.h>
#include <maya/MGlobal.h>
#include <maya/MDagPath.h>
#include <maya/MPointArray.h>
#include <maya/MFnTransform.h>
#include <maya/MFnSingleIndexedComponent.h>
#include <maya/MItMeshVertex.h>
#include <maya/MItMeshEdge.h>

#include <maya/MFnNurbsCurve.h>
#include <maya/MFnNumericAttribute.h>

#include <maya/MFnMesh.h>
#include <maya/MItMeshFaceVertex.h>
#include <maya/MDGModifier.h>

#include <maya/MFnPlugin.h>

#define MStatError(status, msg)								\
    if ( MS::kSuccess != (status) ) {						\
		MPxCommand::displayError(							\
			(msg) + MString(":") + (status).errorString()); \
        return (status);									\
    }

#define MCheckReturn(expression)                \
    {                                           \
        MStatus status = (expression);          \
        if ( MS::kSuccess != (status) ) {       \
            return (status);                    \
        }                                       \
    }

#define vendor			"Dexter Studios"

#define kName			"-n"
#define kNameLong		"-name"
#define kVertexIds		"-ids"
#define kVertexIdsLong	"-vertexids"
#define kVectors		"-v"
#define kVectorsLong	"-vectors"
#define kScale			"-s"
#define kScaleLong		"-scale"

#define commandName		"createVectorCurves"


class createVectorCurves : public MPxCommand
{
public:
	createVectorCurves();
	~createVectorCurves() override;

	static MSyntax	createSyntax();
	static void*	creator();
	MStatus			doIt(const MArgList&) override;
	MStatus			redoIt() override;
	MStatus			undoIt() override;
	bool			isUndoable() const override;
private:
	MStatus			parseArgs(const MArgList&);
	MStatus			setupMesh();
	void			createCurve(unsigned int);

	// parameters
	MSelectionList			mSelection;
	MString					mName;
	MIntArray				mVertexIds;
	MPointArray				mVectors;
	MTransformationMatrix	mMatrix;
	MDagPath				mPathMesh;
	MVectorArray			mPoints;
	double					mScale;
	MObject					mTransform;
	bool					mDone;
};

createVectorCurves::createVectorCurves() : mScale(-1.0f), mDone(false)
{
}

createVectorCurves::~createVectorCurves()
{
}

void * createVectorCurves::creator()
{
	return (void *)(new createVectorCurves);
}

MSyntax createVectorCurves::createSyntax()
{
	MSyntax syntax;
	syntax.addArg(MSyntax::kString);
	syntax.setMaxObjects(1);

	syntax.addFlag(kName, kNameLong, MSyntax::kString);

	syntax.addFlag(kVectors, kVectorsLong, 
		MSyntax::kDouble, MSyntax::kDouble, MSyntax::kDouble);
	syntax.makeFlagMultiUse(kVectors);

	syntax.addFlag(kVertexIds, kVertexIdsLong, MSyntax::kLong);
	syntax.makeFlagMultiUse(kVertexIds);

	syntax.addFlag(kScale, kScaleLong, MSyntax::kDouble);

	return syntax;
}

MStatus createVectorCurves::parseArgs(const MArgList &args)
{
	MStatus status = MStatus::kSuccess;

	MArgDatabase data(syntax(), args, &status);
	MStatError(status, "data(), parseArgs()");

	status = data.getCommandArgument(0, mSelection);
	MStatError(status, "data.getObjects(), parseArgs()");


	if (!data.isFlagSet(kVectors))
	{
		MPxCommand::displayError("Vector Flag is not set.");
		return MStatus::kFailure;
	}

	unsigned int n = data.numberOfFlagUses(kVectors);
	if (data.isFlagSet(kVertexIds))
	{
		if (n != data.numberOfFlagUses(kVertexIds))
		{
			MPxCommand::displayError("Vector and component counts do not match.");
			return MStatus::kFailure;
		}

		for (unsigned int i = 0; i < n; i++)
		{
			MArgList argsFlag;
			status = data.getFlagArgumentList(kVertexIds, i, argsFlag);
			MStatError(status, "data.getFlagArgumentList(kVertexIds), parseArgs()");
			int id = argsFlag.asInt(0);
			MStatError(status, "argsFlag.asInt(), parseArgs()");
			status = mVertexIds.append(id);
			MStatError(status, "mVertexIds.append(argsFlagged.asInt(0)), parseArgs()");
		}
	}

	for (unsigned int i = 0; i < n; i++) {
		MArgList argsFlag;
		status = data.getFlagArgumentList(kVectors, i, argsFlag);
		MStatError(status, "data.getFlagArgumentList(kVectors), parseArgs");
		unsigned int index = 0;
		MPoint v(argsFlag.asPoint(index, 3, &status));
		MStatError(status, "data.argsFlag.asVector(), parseArgs");
		status = mVectors.append(v);
		MStatError(status, "data.append(argsFlagged.asVector(index, 3)), parseArgs");
	}

	if (data.isFlagSet(kName)) {
		status = data.getFlagArgument(kName, 0, mName);
		MStatError(status, "data.getFlagArgument(kName), parseArgs");
	}

	if (data.isFlagSet(kScale)) {
		status = data.getFlagArgument(kScale, 0, mScale);
		MStatError(status, "data.getFlagArgument(kScale), parseArgs")
	}

	return status;
}

MStatus createVectorCurves::setupMesh()
{
	MStatus status = MStatus::kSuccess;

	bool backupNeeded = true;
	if (mSelection.isEmpty())
	{
		MGlobal::getActiveSelectionList(mSelection);
		if (mSelection.length() != 1) {
			MPxCommand::displayError("Nothing is selected.");
			return MStatus::kFailure;
		}
		backupNeeded = false;
	}

	MObject transform;
	status = mSelection.getDagPath(0, mPathMesh);
	MStatError(status, "mSelection.getDagPath(), setupMesh()");
	if (mPathMesh.apiType() == MFn::kTransform)
	{
		transform = mPathMesh.node();
		status = mPathMesh.extendToShape();
		MStatError(status, "mPathMesh.extendToShape(), setupMesh()");
		if (mPathMesh.apiType() != MFn::kMesh) {
			MPxCommand::displayError("A Mesh Object is needed.");
			return MStatus::kFailure;
		}
	}
	else if (mPathMesh.apiType() == MFn::kMesh)
	{
		transform = mPathMesh.transform();
	}
	else
	{
		MPxCommand::displayError("A Mesh is needed.");
		return MStatus::kFailure;
	}

	if (mName.length() == 0)
	{
		MDagPath pathTransform = MDagPath::getAPathTo(transform);
		MString partialName = pathTransform.partialPathName();
		MStringArray splits;
		partialName.split('|', splits);
		mName = splits[splits.length() - 1] + "_curve";
	}

	MFnTransform fnTransform(transform);
	mMatrix = fnTransform.transformation();

	if (backupNeeded) {
		MGlobal::getActiveSelectionList(mSelection);
	}
	return status;
}

MStatus createVectorCurves::doIt(const MArgList& args)
{
	MCheckReturn(parseArgs(args));
	MCheckReturn(setupMesh());

	MFnSingleIndexedComponent fnIds;
	MObject comp = fnIds.create(MFn::kMeshVertComponent);
	if (mVertexIds.length() != 0) {
		fnIds.addElements(mVertexIds);
	}
	else {
		fnIds.setCompleteData(mVectors.length());
	}
	for (MItMeshVertex itVtx(mPathMesh, comp); !itVtx.isDone(); itVtx.next())
	{
		MVector position(itVtx.position());
 		mPoints.append(position);
	}

	return redoIt();
}

void createVectorCurves::createCurve(unsigned int i)
{
	MFnTransform fnTransform;
	MObject transform = fnTransform.create(mTransform);
	fnTransform.setName(mName + i);

	MFnNurbsCurve fnCurve;
	MPointArray points;
	points.append(MPoint()); 
	points.append(mVectors[i]);
	MObject curve = fnCurve.createWithEditPoints(
		points, 1, MFnNurbsCurve::kOpen, false, false, false, transform);
	fnCurve.setName(mName + "Shape");

	fnTransform.setTranslation(mPoints[i], MSpace::kObject);
	if (mScale > .0f) {
		const double scale[3] = {mScale, mScale, mScale};
		fnTransform.setScale(scale);
	}
}

MStatus createVectorCurves::redoIt()
{
	MFnTransform fnTransform;
	mTransform = fnTransform.create();
	fnTransform.setName(mName);

	if (mVertexIds.length() == 0) {
		for (unsigned int i = 0; i < mPoints.length(); i++) {
			createCurve(i);
		}
	}
	else {
		for (unsigned int i = 0; i < mVertexIds.length(); i++) {
			createCurve(mVertexIds[i]);
		}
	}

	fnTransform.set(mMatrix);
	MString name = fnTransform.partialPathName();
	clearResult();
	setResult(name);

	mDone = true;

	return MGlobal::selectByName(name, MGlobal::kReplaceList);
}

MStatus createVectorCurves::undoIt()
{
	MGlobal::setActiveSelectionList(mSelection);

	return MGlobal::deleteNode(mTransform);
}

bool createVectorCurves::isUndoable() const
{
	return mDone;
}

//////////////////////////////////////////////////////////////////////////////////////

//	Register the command

MStatus initializePlugin(MObject obj)
{
	MStatus   status;
	MFnPlugin plugin(obj, vendor, "1.0", "Any");

	status = plugin.registerCommand(commandName,
		createVectorCurves::creator,
		createVectorCurves::createSyntax);
	if (!status) {
		status.perror("registerCommand");
	}
	return status;
}

MStatus uninitializePlugin(MObject obj)
{
	MStatus	  status;
	MFnPlugin plugin(obj);

	status = plugin.deregisterCommand(commandName);
	if (!status) {
		status.perror("deregisterCommand");
	}
	return status;
}

