#include <maya/MPxCommand.h>
#include <maya/MArgList.h>
#include <maya/MSyntax.h>
#include <maya/MString.h>
#include <maya/MSelectionList.h>

#include <maya/MArgDatabase.h>
#include <maya/MGlobal.h>
#include <maya/MDagPath.h>
#include <maya/MFnMesh.h>
#include <maya/MItMeshFaceVertex.h>

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

#define kUvSetFlag		"-uvs"
#define kUvSetFlagLong	"-uvSet"

#define commandName		"getVertexTangents"


class getVertexTangents : public MPxCommand
{
public:
	getVertexTangents();
	~getVertexTangents() override;

	MStatus			doIt(const MArgList& args) override;
	static MSyntax	createSyntax();
	static void*	creator();
private:
	MStatus			parseArgs(const MArgList& args);
	MStatus			setupMesh();
	MStatus			setupUVSetName(MFnMesh& fnMesh);

	// parameters
	MSelectionList	mSelection;
	MDagPath		mPath;
	MString			mUVSetName;
};

getVertexTangents::getVertexTangents()
{
}

getVertexTangents::~getVertexTangents()
{
}

void * getVertexTangents::creator()
{
	return (void *)(new getVertexTangents);
}

MSyntax getVertexTangents::createSyntax()
{
	MSyntax syntax;
	syntax.addArg(MSyntax::kString);
	syntax.setMaxObjects(1);
	syntax.addFlag(kUvSetFlag, kUvSetFlagLong, MSyntax::kString);
	return syntax;
}

MStatus getVertexTangents::parseArgs(const MArgList & args)
{
	MStatus status = MStatus::kSuccess;
	MArgDatabase argData(syntax(), args, &status);
	MStatError(status, "argData()");

	status = argData.getCommandArgument(0, mSelection);
	MStatError(status, "argData.getCommandArgument()");

	if (argData.isFlagSet(kUvSetFlag)) {
		argData.getFlagArgument(kUvSetFlag, 0, mUVSetName);
	}

	return MStatus();
}

MStatus getVertexTangents::setupMesh()
{
	MStatus status = MStatus::kSuccess;

	if (mSelection.length() == 0)
	{
		MGlobal::getActiveSelectionList(mSelection);
		if (mSelection.length() != 1) {
			MPxCommand::displayError("Nothing is selected.");
			return MStatus::kFailure;
		}
	}
	status = mSelection.getDagPath(0, mPath);
	MStatError(status, "mSelection.getDependNode()");
	if (mPath.apiType() == MFn::kTransform)
	{
		status = mPath.extendToShape();
		MStatError(status, "mPath.extendToTshape()");
	}
	if (mPath.apiType() != MFn::kMesh)
	{
		MPxCommand::displayError("A Mesh is needed.");
		return MStatus::kFailure;
	}

	return status;
}

MStatus getVertexTangents::setupUVSetName(MFnMesh& fnMesh)
{
	if (mUVSetName.length() == 0)
	{
		mUVSetName = fnMesh.currentUVSetName();
		if (mUVSetName.length() == 0)
		{
			MPxCommand::displayError("A UVSet is needed.");
			return MStatus::kFailure;
		}
	}
	else
	{
		MStringArray setNames;
		fnMesh.getUVSetNames(setNames);
		if (setNames.indexOf(mUVSetName) == -1)
		{
			MPxCommand::displayError("Given UVSet does not exist.");
			return MStatus::kFailure;
		}
	}
	return MStatus::kSuccess;
}

MStatus getVertexTangents::doIt(const MArgList& args)
{
	MCheckReturn(parseArgs(args));
	MCheckReturn(setupMesh());
	MFnMesh fnMesh(mPath);
	MCheckReturn(setupUVSetName(fnMesh));

//	int numVertices = fnMesh.numVertices();
//	MFloatVectorArray tangents(numVertices);
//	for (MItMeshFaceVertex itFaceVert(mPath); !itFaceVert.isDone(); itFaceVert.next())
//	{
//		int vertId = itFaceVert.vertId();
//		MVector tangent = itFaceVert.getTangent(MSpace::kObject, &mUVSetName);
//		tangents[vertId] += tangent;
//	}
//
//	MFloatVectorArray normals;
//	fnMesh.getVertexNormals(false, normals);
//	MFloatVectorArray binormals;
//
//	MDoubleArray result;
//	for (int i = 0; i < numVertices; i++)
//	{
//		MVector binormal = normals[i] ^ tangents[i];
//		MVector tangent = binormal ^ normals[i];
//		tangent.normalize();
//		result.append(tangent.x);
//		result.append(tangent.y);
//		result.append(tangent.z);
//	}

//	int numVertices = fnMesh.numVertices();
//	MFloatVectorArray tangents(numVertices);
//	for (MItMeshFaceVertex itFaceVert(mPath); !itFaceVert.isDone(); itFaceVert.next())
//	{
//		int vertId = itFaceVert.vertId();
//		if (tangents[vertId].isEquivalent(MFloatVector::zero)) {
//			tangents[vertId] = itFaceVert.getTangent(MSpace::kObject, &mUVSetName);
//		}
//	}
//
//	MDoubleArray result;
//	for (int i = 0; i < numVertices; i++)
//	{
//		result.append(tangents[i].x);
//		result.append(tangents[i].y);
//		result.append(tangents[i].z);
//	}

	int numVertices = fnMesh.numVertices();
	MFloatVectorArray binormals(numVertices);
	for (MItMeshFaceVertex itFaceVert(mPath); !itFaceVert.isDone(); itFaceVert.next())
	{
		int vertId = itFaceVert.vertId();
		if (binormals[vertId].isEquivalent(MFloatVector::zero)) {
			binormals[vertId] = itFaceVert.getBinormal(MSpace::kObject, &mUVSetName);
		}
	}

	MFloatVectorArray normals;
	fnMesh.getVertexNormals(false, normals);
	MDoubleArray result;
	for (int i = 0; i < numVertices; i++)
	{
		MVector tangent = binormals[i] ^ normals[i];
		tangent.normalize();
		result.append(tangent.x);
		result.append(tangent.y);
		result.append(tangent.z);
	}
	setResult(result);
	return MS::kSuccess;
}

//////////////////////////////////////////////////////////////////////////////////////

//	Register the command

MStatus initializePlugin(MObject obj)
{
	MStatus   status;
	MFnPlugin plugin(obj, vendor, "1.0", "Any");

	status = plugin.registerCommand(commandName
			, getVertexTangents::creator
			, getVertexTangents::createSyntax);
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

