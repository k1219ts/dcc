#include <vector>
using namespace std;

#include <maya/MPxCommand.h>
#include <maya/MArgList.h>
#include <maya/MSyntax.h>
#include <maya/MString.h>
#include <maya/MSelectionList.h>

#include <maya/MArgDatabase.h>
#include <maya/MGlobal.h>
#include <maya/MRichSelection.h>
#include <maya/MItSelectionList.h>

#include <maya/MDagPath.h>
#include <maya/MFnSingleIndexedComponent.h>
#include <maya/MItMeshVertex.h>

#include <maya/MFnMesh.h>

#include <maya/MFnPlugin.h>


#define MStatError(status, msg)								\
    if ( MS::kSuccess != (status) ) {						\
		MPxCommand::displayError(							\
			(msg) + MString(":") + (status).errorString()); \
        return (status);									\
    }

#define MCheckReturn(expression)			\
    {                                       \
        MStatus status = (expression);      \
        if ( MS::kSuccess != (status) ) {   \
            return (status);                \
        }                                   \
    }

#define CheckReturn(expression, msg)		\
    if ( (expression) ) {					\
		MPxCommand::displayError( (msg) );	\
        return MS::kFailure;				\
    }

#define vendor			"Dexter Studios"

#define kVertexIds		"-ids"
#define kVertexIdsLong	"-vertexids"
#define kTarget			"-t"
#define kTargetLong		"-target"
#define kDoubles		"-d"
#define kDoublesLong	"-doubles"
#define kVectors		"-v"
#define kVectorsLong	"-vectors"
#define kIteration		"-iter"
#define kIterationLong	"-iteration"
#define kWeight			"-w"
#define kWeightLong		"-weight"

#define commandName		"smoothenVertexValues"


class smoothenVertexValues : public MPxCommand
{
public:
	smoothenVertexValues() {}
	~smoothenVertexValues() override {}

	static MSyntax	createSyntax();
	static void*	creator();
	MStatus			doIt(const MArgList&) override;
private:
	MStatus				parseArgs(const MArgList&);
	MStatus				setupMesh();
	void				setupWeights();
	template<class T> T getSource(T);
	bool				getComponents();

	// parameters
	MSelectionList	mSelection;
	MIntArray		mVertexIds;
	MIntArray		mTargetIds;
	MDoubleArray	mDoubles;
	MVectorArray	mVectors;
	int				mIteration = 1;
	double			mCenterWeight = 1.0f;
	MDoubleArray	mWeights;
	MDagPath		mPathMesh;
	MObject			mComponents;
	unsigned int	mVertexCount = 0;
};

void * smoothenVertexValues::creator()
{
	return (void *)(new smoothenVertexValues);
}

MSyntax smoothenVertexValues::createSyntax()
{
	MSyntax syntax;
	syntax.addArg(MSyntax::kString);
	syntax.setMaxObjects(1);

	syntax.addFlag(kDoubles, kDoublesLong, MSyntax::kDouble);
	syntax.makeFlagMultiUse(kDoubles);

	syntax.addFlag(kVectors, kVectorsLong,
		MSyntax::kDouble, MSyntax::kDouble, MSyntax::kDouble);
	syntax.makeFlagMultiUse(kVectors);

	syntax.addFlag(kIteration, kIterationLong, MSyntax::kLong);

	syntax.addFlag(kVertexIds, kVertexIdsLong, MSyntax::kLong);
	syntax.makeFlagMultiUse(kVertexIds);

	syntax.addFlag(kTarget, kTargetLong, MSyntax::kLong);
	syntax.makeFlagMultiUse(kTarget);

	syntax.addFlag(kWeight, kWeightLong, MSyntax::kDouble);

	return syntax;
}

MStatus smoothenVertexValues::parseArgs(const MArgList &args)
{
	MStatus status = MStatus::kSuccess;

	MArgDatabase data(syntax(), args, &status);
	MStatError(status, "data(), parseArgs()");

	MStringArray commandArgs;
	data.getObjects(commandArgs);
	if (commandArgs.length() == 1) {
		status = data.getCommandArgument(0, mSelection);
		MStatError(status, "data.getObjects(), parseArgs()");
	}

	bool isD = data.isFlagSet(kDoubles);
	bool isV = data.isFlagSet(kVectors);
	CheckReturn((isD && isV) || !(isD || isV), "too many data flags");

	unsigned int n = 0;
	if (isD) {
		n = data.numberOfFlagUses(kDoubles);
		for (unsigned int i = 0; i < n; i++)
		{
			MArgList argsFlag;
			status = data.getFlagArgumentList(kDoubles, i, argsFlag);
			MStatError(status, "data.getFlagArgumentList(kDoubles), parseArgs()");
			mDoubles.append(argsFlag.asDouble(0));
		}
	}
	else if (isV) {
		n = data.numberOfFlagUses(kVectors);
		for (unsigned int i = 0; i < n; i++)
		{
			MArgList argsFlag;
			status = data.getFlagArgumentList(kVectors, i, argsFlag);
			MStatError(status, "data.getFlagArgumentList(kVectors), parseArgs()");
			unsigned int index = 0;
			MVector v = argsFlag.asVector(index, 3, &status);
			MStatError(status, "data.argsFlag.asVector(), parseArgs");
			mVectors.append(v);
		}
	}

	if (data.isFlagSet(kVertexIds))
	{
		CheckReturn(n != data.numberOfFlagUses(kVertexIds), "Value and component counts mismatch.");
		CheckReturn(mSelection.isEmpty(), "No mesh is given.");

		for (unsigned int i = 0; i < n; i++)
		{
			MArgList argsFlag;
			status = data.getFlagArgumentList(kVertexIds, i, argsFlag);
			MStatError(status, "data.getFlagArgumentList(kVertexIds), parseArgs()");
			status = mVertexIds.append(argsFlag.asInt(0));
			MStatError(status, "mVertexIds.append(argsFlagged.asInt(0)), parseArgs()");
		}
	}

	if (data.isFlagSet(kTarget))
	{
		for (unsigned int i = 0; i < data.numberOfFlagUses(kTarget); i++)
		{
			MArgList argsFlag;
			status = data.getFlagArgumentList(kTarget, i, argsFlag);
			MStatError(status, "data.getFlagArgumentList(kTarget), parseArgs()");
			status = mTargetIds.append(argsFlag.asInt(0));
			MStatError(status, "mTargetIds.append(argsFlagged.asInt()), parseArgs()");
		}
	}

	if (data.isFlagSet(kWeight)) {
		data.getFlagArgument(kWeight, 0, mCenterWeight);
	}

	return status;
}

void smoothenVertexValues::setupWeights()
{
	mWeights = MDoubleArray(mVertexCount, 1.0f);
	if (!mSelection.isEmpty()) {
		return;
	}

	MRichSelection richSelection;
	MGlobal::getRichSelection(richSelection);
	MSelectionList selection;
	richSelection.getSelection(selection);

	for (MItSelectionList it(selection); !it.isDone(); it.next())
	{
		MDagPath path;
		MObject comp;
		it.getDagPath(path, comp);
		MFnSingleIndexedComponent fn(comp);
		if (fn.hasWeights()) {
			for (int i = 0; i < fn.elementCount(); i++) {
				mWeights[fn.element(i)] = fn.weight(i).influence();
			}
		}
	}
}

MStatus smoothenVertexValues::setupMesh()
{
	MStatus status = MStatus::kSuccess;

	if (mSelection.isEmpty())
	{
		MSelectionList selection;
		MGlobal::getActiveSelectionList(selection);
		for (MItSelectionList it(selection); !it.isDone(); it.next())
		{
			MDagPath path;
			MObject component;
			status = it.getDagPath(path, component);

			if (component.isNull() || component.apiType() != MFn::kMeshVertComponent) {
				displayError("non-vertex selection");
				return MStatus::kFailure;
			}

			if (!mPathMesh.isValid()) {
				if (path.apiType() != MFn::kMesh) {
					displayError("non-vertex selection");
					return MStatus::kFailure;
				}
				mPathMesh = path;
			}
			else if (mPathMesh == path) {
			}
			else {
				displayError("selection error");
				return MStatus::kFailure;
			}
		}
	}
	else {
		status = mSelection.getDagPath(0, mPathMesh);
		MStatError(status, "mSelection.getDagPath(), setupMesh()");
	}
	MFnMesh fnMesh(mPathMesh);
	mVertexCount = fnMesh.numVertices();
	setupWeights();

	return status;
}

template<class T>
T smoothenVertexValues::getSource(T source)
{
	T meshValues(mVertexCount);
	if (mVertexIds.length()) {
		for (unsigned int i = 0; i < source.length(); i++) {
			meshValues[mVertexIds[i]] = source[i];
		}
	}
	else {
		meshValues.copy(source);
	}

	return meshValues;
}

bool smoothenVertexValues::getComponents()
{
	MFnSingleIndexedComponent fnIds;
	mComponents = fnIds.create(MFn::kMeshVertComponent);
	unsigned int n = mTargetIds.length();

	if (n && n != mVertexCount) {
		fnIds.addElements(mTargetIds);
		return true;
	}
	else {
		fnIds.setComplete(true);
		return false;
	}
}

MStatus smoothenVertexValues::doIt(const MArgList& args)
{
	MStatus status = MStatus::kSuccess;
	clearResult();

	MCheckReturn(parseArgs(args));
	MCheckReturn(setupMesh());
	getComponents();

	MItMeshVertex it(mPathMesh, mComponents);
	if (mDoubles.length())
	{
		vector<MDoubleArray> results;
		MDoubleArray source = getSource(mDoubles);
		for (int i = 0; i < mIteration; i++)
		{
			MDoubleArray target(source);
			for (it.reset(); !it.isDone(); it.next())
			{
				int index = it.index();
				double weight = mWeights[index];
				double value = source[index] * weight;
				MIntArray neighbors;
				it.getConnectedVertices(neighbors);
				for (int j = 0; j < neighbors.length(); j++) {
					int idx = neighbors[j];
					double w = mWeights[idx];
					value += source[idx] * w;
					weight += w;
				}
				value /= weight;
				target[index] = value;
			}
			results.push_back(target);
			source = target;
		}
		for (auto array : results)
		{
			unsigned int j = 0;
			for (unsigned int i = 0; i < mVertexCount; i++)
			{
				for (; j < mVertexIds.length(); j++) {
					if (mVertexIds[j] == i) {
						appendToResult(array[i]);
						break;
					}
				}
			}
		}
	}
	else
	{
		vector<MVectorArray> results;
		MVectorArray source = getSource(mVectors);
		for (int i = 0; i < mIteration; i++)
		{
			MVectorArray target(source);
			for (it.reset(); !it.isDone(); it.next())
			{
				int index = it.index();
				double weight = mWeights[index];
				MVector value = source[index] * weight;
				MIntArray neighbors;
				it.getConnectedVertices(neighbors);
				for (int j = 0; j < neighbors.length(); j++) {
					int idx = neighbors[j];
					double w = mWeights[idx];
					value += source[idx] * w;
					weight += w;
				}
				value /= weight;
				target[index] = value;
			}
			results.push_back(target);
			source = target;
		}
		for (auto array : results)
		{
			unsigned int j = 0;
			for (unsigned int i = 0; i < mVertexCount; i++)
			{
				for (; j < mVertexIds.length(); j++) {
					if (mVertexIds[j] == i) {
						MVector value = array[i];
						appendToResult(value.x);
						appendToResult(value.y);
						appendToResult(value.z);
						break;
					}
				}
			}
		}
	}
	return status;
}


//////////////////////////////////////////////////////////////////////////////////////

//	Register the command

MStatus initializePlugin(MObject obj)
{
	MStatus   status;
	MFnPlugin plugin(obj, vendor, "1.0", "Any");

	status = plugin.registerCommand(commandName,
		smoothenVertexValues::creator,
		smoothenVertexValues::createSyntax);
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

