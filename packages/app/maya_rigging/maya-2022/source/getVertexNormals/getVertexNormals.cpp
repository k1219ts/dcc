#include <maya/MPxCommand.h>
#include <maya/MArgList.h>
#include <maya/MSyntax.h>
#include <maya/MString.h>
#include <maya/MSelectionList.h>

#include <maya/MArgDatabase.h>
#include <maya/MGlobal.h>
#include <maya/MDagPath.h>
#include <maya/MFnMesh.h>
#include <maya/MFloatVectorArray.h>

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
#define commandName		"getVertexNormals"


class getVertexNormals : public MPxCommand
{
public:
	getVertexNormals();
	~getVertexNormals() override;

	MStatus			doIt(const MArgList& args) override;
	static MSyntax	createSyntax();
	static void*	creator();
private:
	MStatus			parseArgs(const MArgList& args);
	MStatus			setupMesh();

	// parameters
	MSelectionList	mSelection;
	MDagPath		mPath;
};

getVertexNormals::getVertexNormals()
{
}

getVertexNormals::~getVertexNormals()
{
}

void * getVertexNormals::creator()
{
	return (void *)(new getVertexNormals);
}

MSyntax getVertexNormals::createSyntax()
{
	MSyntax syntax;
	syntax.addArg(MSyntax::kString);
	syntax.setMaxObjects(1);
	return syntax;
}

MStatus getVertexNormals::parseArgs(const MArgList & args)
{
	MStatus status = MStatus::kSuccess;
	MArgDatabase argData(syntax(), args, &status);
	MStatError(status, "argData()");

	status = argData.getCommandArgument(0, mSelection);
	MStatError(status, "argData.getCommandArgument()");

	return MStatus();
}

MStatus getVertexNormals::setupMesh()
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

MStatus getVertexNormals::doIt(const MArgList& args)
{
	MCheckReturn(parseArgs(args));
	MCheckReturn(setupMesh());
	MFnMesh fnMesh(mPath);

	MFloatVectorArray normals;
	fnMesh.getVertexNormals(false, normals);
	MDoubleArray result;
	for (int i = 0; i < fnMesh.numVertices(); i++)
	{
		MVector normal = normals[i];
		result.append(normal.x);
		result.append(normal.y);
		result.append(normal.z);
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
			, getVertexNormals::creator
			, getVertexNormals::createSyntax);
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

