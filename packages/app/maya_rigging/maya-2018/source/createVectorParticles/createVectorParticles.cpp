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

#include <maya/MFnParticleSystem.h>
#include <maya/MFnNurbsCurve.h>
#include <maya/MFnNumericAttribute.h>

#include <maya/MFnMesh.h>

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

#define kIdAttr			"vid"
#define kIdAttrLong		"vertexId"

#define commandName		"createVectorParticles"


class createVectorParticles : public MPxCommand
{
public:
	createVectorParticles();
	~createVectorParticles() override;

	static MSyntax	createSyntax();
	static void*	creator();
	MStatus			doIt(const MArgList&) override;
	MStatus			redoIt() override;
	MStatus			undoIt() override;
	bool			isUndoable() const override;
private:
	MStatus			parseArgs(const MArgList&);
	MStatus			setupMesh();

	// parameters
	MSelectionList			mSelection;
	MString					mName;
	MIntArray				mVertexIds;
	MVectorArray			mVectors;
	MTransformationMatrix	mMatrix;
	MDagPath				mPathMesh;
	MPointArray				mPoints;
	MObject					mTransform;
	bool					mDone;
};

createVectorParticles::createVectorParticles() : mDone(false)
{
}

createVectorParticles::~createVectorParticles()
{
}

void * createVectorParticles::creator()
{
	return (void *)(new createVectorParticles);
}

MSyntax createVectorParticles::createSyntax()
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
	return syntax;
}

MStatus createVectorParticles::parseArgs(const MArgList &args)
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

	unsigned n = data.numberOfFlagUses(kVectors);
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

	if (data.isFlagSet(kName)) {
		status = data.getFlagArgument(kName, 0, mName);
		MStatError(status, "data.getFlagArgument(kName), parseArgs");
	}

	for (unsigned int i = 0; i < n; i++) {
		MArgList argsFlag;
		status = data.getFlagArgumentList(kVectors, i, argsFlag);
		MStatError(status, "data.getFlagArgumentList(kVectors), parseArgs");
		unsigned int index = 0;
		MVector v = argsFlag.asVector(index, 3, &status);
		MStatError(status, "data.argsFlag.asVector(), parseArgs");
		status = mVectors.append(v * -30.0f);
		MStatError(status, "data.append(argsFlagged.asVector(index, 3)), parseArgs");
	}

	return status;
}

inline MStatus createVectorParticles::setupMesh()
{
	MStatus status = MStatus::kSuccess;

	if (mSelection.isEmpty())
	{
		MGlobal::getActiveSelectionList(mSelection);
		if (mSelection.length() != 1) {
			MPxCommand::displayError("Nothing is selected.");
			return MStatus::kFailure;
		}
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
		mName = splits[splits.length() - 1] + "_vectors";
	}

	MFnTransform fnTransform(transform);
	mMatrix = fnTransform.transformation();
	return status;
}

MStatus createVectorParticles::doIt(const MArgList& args)
{
	MCheckReturn(parseArgs(args));
	MCheckReturn(setupMesh());

	MFnSingleIndexedComponent fnIds;
	MObject comp = fnIds.create(MFn::kMeshVertComponent);
	if (mVertexIds.length() != 0) {
		fnIds.addElements(mVertexIds);
	}
	else {
		fnIds.setComplete(true);
	}
	for (MItMeshVertex itVtx(mPathMesh, comp); !itVtx.isDone(); itVtx.next())
	{
		//MPoint position = itVtx.position();
		mPoints.append(itVtx.position());
	}

	return redoIt();
}

MStatus createVectorParticles::redoIt()
{
	MFnTransform fnTransform;
	mTransform = fnTransform.create();
	fnTransform.setName(mName);

	MFnParticleSystem fnParticle;
	fnParticle.create(mTransform);
	fnParticle.emit(mPoints, mVectors);
	fnParticle.setName(mName + "Shape");

	fnTransform.set(mMatrix);

	clearResult();
	setResult(fnParticle.partialPathName());

	mDone = true;

	return MS::kSuccess;
}

MStatus createVectorParticles::undoIt()
{
	return MGlobal::deleteNode(mTransform);
}

bool createVectorParticles::isUndoable() const
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
		createVectorParticles::creator,
		createVectorParticles::createSyntax);
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

