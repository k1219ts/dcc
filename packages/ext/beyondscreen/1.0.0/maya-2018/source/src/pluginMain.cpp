#include <maya/MGlobal.h>
#include <maya/MFnPlugin.h>

//#include <BeyondScreenTest.h>
#include <BeyondScreenCacheViewer.h>
#include <BeyondScreenExport.h>

MStatus initializePlugin( MObject obj )
{
    MStatus s = MS::kSuccess;    

    MFnPlugin pluginFn( obj, "Dexter Studios", "1.0", "Any" );

    //pluginFn.registerNode( BeyondScreenTest::name, BeyondScreenTest::id, BeyondScreenTest::creator, BeyondScreenTest::initialize, MPxNode::kLocatorNode );
    pluginFn.registerNode( BeyondScreenCacheViewer::name, BeyondScreenCacheViewer::id, BeyondScreenCacheViewer::creator, BeyondScreenCacheViewer::initialize, MPxNode::kLocatorNode );
    pluginFn.registerCommand( BeyondScreenExport::name, BeyondScreenExport::creator, BeyondScreenExport::newSyntax );

    MGlobal::sourceFile( "BeyondScreen.mel" );
    MGlobal::sourceFile( "BeyondScreenMenu.mel" );

    pluginFn.registerUI( "CreateBeyondScreenMenu", "DeleteBeyondScreenMenu" );

    return MS::kSuccess;
}

MStatus uninitializePlugin( MObject obj )
{
    MStatus s = MS::kSuccess;
	
    MFnPlugin pluginFn( obj );

//	pluginFn.deregisterNode( BeyondScreenTest::id );
    pluginFn.deregisterNode( BeyondScreenCacheViewer::id );
    pluginFn.deregisterCommand( BeyondScreenExport::name );

    return MS::kSuccess;
}

