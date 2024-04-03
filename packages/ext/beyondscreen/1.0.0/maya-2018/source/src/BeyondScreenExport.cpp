#include <BeyondScreenExport.h>

MString BeyondScreenExport::name( "BeyondScreenExport" );

#define startFrameFlag     "-sF"
#define startFrameLongFlag "-startFrame"
#define endFrameFlag       "-eF"
#define endFrameLongFlag   "-endFrame"
#define screenFlag         "-sc"
#define screenLongFlag     "-screen"
#define aimFlag            "-ap"
#define aimLongFlag        "-aimPoint"
#define cameraFlag         "-cm"
#define cameraLongFlag     "-camera"
#define filePathFlag       "-fP"
#define filePathLongFlag   "-filePath"
#define fileNameFlag       "-fN"
#define fileNameLongFlag   "-fileName"

MSyntax BeyondScreenExport::newSyntax()
{
	MSyntax syntax;

    syntax.addFlag( startFrameFlag, startFrameLongFlag, MSyntax::kUnsigned );
	syntax.addFlag( endFrameFlag,   endFrameLongFlag,   MSyntax::kUnsigned );
	syntax.addFlag( screenFlag,     screenLongFlag,     MSyntax::kString   );
	syntax.addFlag( aimFlag,        aimLongFlag,        MSyntax::kString   );
	syntax.addFlag( cameraFlag,     cameraLongFlag,     MSyntax::kString   );
	syntax.addFlag( filePathFlag,   filePathLongFlag,   MSyntax::kString   );
	syntax.addFlag( fileNameFlag,   fileNameLongFlag,   MSyntax::kString   );

	return syntax;
}

MStatus BeyondScreenExport::doIt( const MArgList& args )
{
	MStatus stat = MS::kSuccess;

	MArgDatabase argData( syntax(), args, &stat );
	if( !stat ) { return MS::kFailure; }

    int     startFrame    = getStartFrame    ( argData );
    int     endFrame      = getEndFrame      ( argData );
	MString screenName    = getScreenName    ( argData );
	MString aimName       = getAimName       ( argData );
	MString cameraName    = getCameraName    ( argData );
	MString filePath      = getFilePath      ( argData );
	MString fileName      = getFileName      ( argData );

    BeyondScreenExport::execute
    (
        startFrame,
        endFrame,
        screenName,
        aimName,
        cameraName,
        filePath,
        fileName
    );

	return MS::kSuccess;
}

int BeyondScreenExport::getStartFrame( const MArgDatabase& argData )
{
    unsigned int startFrame = 0;

    if( argData.isFlagSet( startFrameFlag ) )
    {
        if( !argData.getFlagArgument( startFrameFlag, 0, startFrame ) )
        {
            return 0; // default startFrameue
        }
    }

    return (int)startFrame;
}

int BeyondScreenExport::getEndFrame( const MArgDatabase& argData )
{
    unsigned int endFrame = 0;

    if( argData.isFlagSet( endFrameFlag ) )
    {
        if( !argData.getFlagArgument( endFrameFlag, 0, endFrame ) )
        {
            return 0; // default endFrameue
        }
    }

    return (int)endFrame;
}

MString BeyondScreenExport::getScreenName( const MArgDatabase& argData )
{
	MString screen;

	if( argData.isFlagSet( screenFlag ) )
	{
		if( !argData.getFlagArgument( screenFlag, 0, screen ) )
		{
			MGlobal::displayError( name + ": No -screen flag." );
			return screen;
		}
	}

	return screen;
}

MString BeyondScreenExport::getAimName( const MArgDatabase& argData )
{
	MString aim;

	if( argData.isFlagSet( aimFlag ) )
	{
		if( !argData.getFlagArgument( aimFlag, 0, aim ) )
		{
			MGlobal::displayError( name + ": No -aim flag." );
			return aim;
		}
	}

	return aim;
}

MString BeyondScreenExport::getCameraName( const MArgDatabase& argData )
{
	MString camera;

	if( argData.isFlagSet( cameraFlag ) )
	{
		if( !argData.getFlagArgument( cameraFlag, 0, camera ) )
		{
			MGlobal::displayError( name + ": No -camera flag." );
			return camera;
		}
	}

	return camera;
}

MString BeyondScreenExport::getFilePath( const MArgDatabase& argData )
{
	MString filePath;

	if( argData.isFlagSet( filePathFlag ) )
	{
		if( !argData.getFlagArgument( filePathFlag, 0, filePath ) )
		{
			MGlobal::displayError( name + ": No -filePath flag." );
			return filePath;
		}
	}

	return filePath;
}

MString BeyondScreenExport::getFileName( const MArgDatabase& argData )
{
	MString fileName;

	if( argData.isFlagSet( fileNameFlag ) )
	{
		if( !argData.getFlagArgument( fileNameFlag, 0, fileName ) )
		{
			MGlobal::displayError( name + ": No -fileName flag." );
			return fileName;
		}
	}

	return fileName;
}

void BeyondScreenExport::execute
(
    const int      startFrame,
    const int      endFrame,
    const MString& screenName,
    const MString& aimName,
    const MString& cameraName,
    const MString& filePath,
    const MString& fileName
)
{
    Manager manager;

    IntArray&    animationFrames       = manager.animationFrames;
    ScreenMesh&  objectScreenMesh      = manager.objectScreenMesh;
    Int4&        cornerIndices         = manager.cornerIndices;
    MatrixArray& objectToWorldMatrices = manager.objectToWorldMatrices;
    VectorArray& worldAimingPoints     = manager.worldAimingPoints;
    VectorArray& worldCameraPositions  = manager.worldCameraPositions;
    VectorArray& worldCameraUpvectors  = manager.worldCameraUpvectors;

    MObject screenShapeNodeObj = NodeNameToMObject( screenName );
    Convert( objectScreenMesh, screenShapeNodeObj, false, "currentUVSet" );

    manager.computeFourCorners();

    int animationFrame = startFrame;

    MComputation computation;
    computation.beginComputation();
    {
        for( ; animationFrame <= endFrame; ++animationFrame )
        {
            if( computation.isInterruptRequested() ) { return; }

            MGlobal::viewFrame( animationFrame );

            Matrix objectToWorldMatrix;
            {
                MMatrix m;
                GetWorldMatrix( screenShapeNodeObj, m );
                Copy( objectToWorldMatrix, m );
            }

            const Vector worldAimingPoint    = GetWorldPosition( aimName       );
            const Vector worldCameraPosition = GetWorldPosition( cameraName    );
            const Vector worldCameraUpvector = GetWorldUpvector( cameraName    );

            animationFrames       .append( animationFrame      );
            objectToWorldMatrices .append( objectToWorldMatrix );
            worldAimingPoints     .append( worldAimingPoint    );
            worldCameraPositions  .append( worldCameraPosition );
            worldCameraUpvectors  .append( worldCameraUpvector );
        }
    }
    computation.endComputation();

    // if you did not stop intemediately, it exports a cache file.
    if( (--animationFrame) == endFrame )
    {
        manager.save( ( filePath + "/" + fileName ).asChar() );
    }
}

