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

#define vendor				"Dexter Studios"

#define kAttr				"-at"
#define kAttrLong			"-attribute"
#define kIds				"-ids"
#define kIdsLong			"-particleIds"
#define kDoubleValue		"-d"
#define kDoubleValueLong	"-double"
#define kDoubleArray		"-dar"
#define kDoubleArrayLong	"-doubleArray"
#define kVectorValue		"-v"
#define kVectorValueLong	"-vector"
#define kVectorArray		"-var"
#define kVectorArrayLong	"-vectorArray"

#define commandName			"setPPAttr"

using namespace std;


class setPPAttr : public MPxCommand
{
public:
	setPPAttr() {}
	~setPPAttr() override {};

	static MSyntax	createSyntax();
	static void*	creator();
	MStatus			doIt(const MArgList&) override;
	bool			isUndoable() const override;
private:
	MStatus			parseArgs(const MArgList&);
	template<class T> MStatus updateAttributeData(T, T);

	// parameters
	MSelectionList			mSelection;
	MString					mAttrName;
	MIntArray				mIds;
	unsigned int			mCount = 0;
	bool					mIsDouble = false;
	double					mDouble = 0.0f;
	MDoubleArray			mDoubles;
	bool					mIsVector = false;
	MVector					mVector;
	MVectorArray			mVectors;

	MDagPath				mPath;

	MDoubleArray			mOriginalDoubles;
	MVectorArray			mOriginalVectors;
};

void * setPPAttr::creator()
{
	return (void *)(new setPPAttr);
}

MSyntax setPPAttr::createSyntax()
{
	MSyntax syntax;
	syntax.addArg(MSyntax::kString);
	syntax.setMaxObjects(1);

	syntax.addFlag(kAttr, kAttrLong, MSyntax::kString);

	syntax.addFlag(kIds, kIdsLong, MSyntax::kLong);
	syntax.makeFlagMultiUse(kIds);

	syntax.addFlag(kDoubleValue, kDoubleValueLong, MSyntax::kDouble);
	syntax.addFlag(kDoubleArray, kDoubleArrayLong, MSyntax::kDouble);
	syntax.makeFlagMultiUse(kDoubleArray);

	syntax.addFlag(kVectorValue, kVectorValueLong,
		MSyntax::kDouble, MSyntax::kDouble, MSyntax::kDouble);
	syntax.addFlag(kVectorArray, kVectorArrayLong,
		MSyntax::kDouble, MSyntax::kDouble, MSyntax::kDouble);
	syntax.makeFlagMultiUse(kVectorArray);

	return syntax;
}

MStatus setPPAttr::parseArgs(const MArgList &args)
{
	MStatus status = MStatus::kSuccess;

	MArgDatabase data(syntax(), args, &status);
	MStatError(status, "data(), parseArgs()");

	status = data.getCommandArgument(0, mSelection);
	MStatError(status, "data.getObjects(), parseArgs()");


	if (data.isFlagSet(kAttr)) {
		status = data.getFlagArgument(kAttr, 0, mAttrName);
		MStatError(status, "data.getFlagArgument(kAttr), parseArgs");
	}
	else {
		displayError("Attribute is not given.");
		return MStatus::kFailure;
	}

	if (data.isFlagSet(kIds)) {
		mCount = data.numberOfFlagUses(kIds);
		for (unsigned int i = 0; i < mCount; i++)
		{
			MArgList argsFlag;
			status = data.getFlagArgumentList(kIds, i, argsFlag);
			MStatError(status, "data.getFlagArgumentList(kIds), parseArgs()");
			mIds.append(argsFlag.asInt(0));
		}
	}

	mIsDouble = data.isFlagSet(kDoubleValue);
	bool isDoubleArray = data.isFlagSet(kDoubleArray);
	mIsVector = data.isFlagSet(kVectorValue);
	bool isVectorArray = data.isFlagSet(kVectorArray);
	if (int(mIsDouble) + int(isDoubleArray) + int(mIsVector) + int(isVectorArray) != 1) {
		displayError("too many data flags");
		return MStatus::kFailure;
	}

	if (mIsDouble) {
		data.getFlagArgument(kDoubleValue, 0, mDouble);
	}
	else if (isDoubleArray) {
		if (!mCount) {
			mCount = data.numberOfFlagUses(kDoubleArray);
		}
		else if (mCount != data.numberOfFlagUses(kDoubleArray)) {
			displayError("numbers of ids and data do not meet.");
			return MStatus::kFailure;
		}
		for (unsigned int i = 0; i < mCount; i++)
		{
			MArgList argsFlag;
			status = data.getFlagArgumentList(kDoubleArray, i, argsFlag);
			MStatError(status, "data.getFlagArgumentList(kDoubleArray), parseArgs()");
			mDoubles.append(argsFlag.asDouble(0));
		}
	}
	else if (mIsVector) {
		double x, y, z;
		data.getFlagArgument(kVectorValue, 0, x);
		data.getFlagArgument(kVectorValue, 1, y);
		data.getFlagArgument(kVectorValue, 2, z);
		mVector = MVector(x, y, z);
	}
	else if (isVectorArray) {
		if (!mCount) {
			mCount = data.numberOfFlagUses(kVectorArray);
		}
		else if (mCount != data.numberOfFlagUses(kVectorArray)) {
			displayError("numbers of ids and data do not meet.");
			return MStatus::kFailure;
		}
		for (unsigned int i = 0; i < mCount; i++)
		{
			MArgList argsFlag;
			status = data.getFlagArgumentList(kVectorArray, i, argsFlag);
			MStatError(status, "data.getFlagArgumentList(kVectorArray), parseArgs()");
			unsigned int index = 0;
			MVector v = argsFlag.asVector(index, 3, &status);
			MStatError(status, "data.argsFlag.asVector(), parseArgs");
			mVectors.append(v);
		}
	}

	return status;
}

template<class T>
MStatus setPPAttr::updateAttributeData(T inputData, T originalData)
{
	MStatus status = MStatus::kSuccess;
	T data;
	MFnParticleSystem fnParticle(mPath);

	fnParticle.getPerParticleAttribute(mAttrName, data, &status);
	MStatError(status, "fn.getPerParticleAttr()");

	originalData.copy(data);
	unsigned int n = mIds.length();
	if (n) {
		for (unsigned int i = 0; i < n; i++) {
			data[mIds[i]] = inputData[i];
		}
	}
	fnParticle.setPerParticleAttribute(mAttrName, data, &status);
	MStatError(status, "fnParticle.setPerParticleAttribute(MDoubleArray)");
	return status;
}

MStatus setPPAttr::doIt(const MArgList& args)
{
	MCheckReturn(parseArgs(args));

	MStatus status = MStatus::kSuccess;
	status = mSelection.getDagPath(0, mPath);
	MStatError(status, "mSelection.getDagPath()");

	MFnParticleSystem fnParticle(mPath);
	if (!mCount) {
		mCount = fnParticle.count();
	}
	if (mIsDouble) {
		mDoubles = MDoubleArray(mCount, mDouble);
	}
	else if (mIsVector) {
		mVectors = MVectorArray(mCount, mVector);
	}

	if (mDoubles.length()) {
		MCheckReturn(updateAttributeData(mDoubles, mOriginalDoubles));
	}
	else if (mVectors.length()) {
		MCheckReturn(updateAttributeData(mVectors, mOriginalVectors));
	}

	return status;
}

bool setPPAttr::isUndoable() const
{
	return false;
}

//////////////////////////////////////////////////////////////////////////////////////

//	Register the command

MStatus initializePlugin(MObject obj)
{
	MStatus   status;
	MFnPlugin plugin(obj, vendor, "1.0", "Any");

	status = plugin.registerCommand(commandName,
		setPPAttr::creator,
		setPPAttr::createSyntax);
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
