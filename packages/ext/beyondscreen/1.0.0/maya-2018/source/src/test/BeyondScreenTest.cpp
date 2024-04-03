#include <BeyondScreenTest.h>

MTypeId BeyondScreenTest::id( 0x71034 );
MString BeyondScreenTest::name( "BeyondScreenTest" );

MObject BeyondScreenTest::cameraPositionObj;
MObject BeyondScreenTest::aimPointObj;
MObject BeyondScreenTest::projectorPositionObj;
MObject BeyondScreenTest::screenMeshObj;
MObject BeyondScreenTest::screenXFormObj;
MObject BeyondScreenTest::imageFilePathNameObj;
MObject BeyondScreenTest::outputObj;

void* BeyondScreenTest::creator()
{
	return new BeyondScreenTest();
}

MStatus BeyondScreenTest::initialize()
{
    MStatus stat = MS::kSuccess;

    MFnTypedAttribute   tAttr;
    MFnMatrixAttribute  mAttr;
    MFnNumericAttribute nAttr;

	cameraPositionObj = nAttr.createPoint( "cameraPosition", "cameraPosition" );
    nAttr.setHidden(1);
    addAttribute( cameraPositionObj );

	aimPointObj = nAttr.createPoint( "aimPoint", "aimPoint" );
    nAttr.setHidden(1);
    addAttribute( aimPointObj );

	projectorPositionObj = nAttr.createPoint( "projectorPosition", "projectorPosition" );
    nAttr.setHidden(1);
    addAttribute( projectorPositionObj );

	screenMeshObj = tAttr.create( "screenMesh", "screenMesh", MFnData::kMesh );
    tAttr.setHidden(1);
    addAttribute( screenMeshObj );

    screenXFormObj = mAttr.create( "screenXForm", "screenXForm", MFnMatrixAttribute::kDouble );
    mAttr.setHidden(1);
    addAttribute( screenXFormObj );

	imageFilePathNameObj = tAttr.create( "imageFilePathName", "imageFilePathName", MFnData::kString );
    tAttr.setUsedAsFilename(1);
    addAttribute( imageFilePathNameObj );

	outputObj = nAttr.create( "output", "output", MFnNumericData::kFloat, 0.f );
    nAttr.setHidden(1);
    CHECK_MSTATUS( addAttribute( outputObj ) );

    attributeAffects( cameraPositionObj,    outputObj );
    attributeAffects( aimPointObj,          outputObj );
    attributeAffects( projectorPositionObj, outputObj );
    attributeAffects( screenMeshObj,        outputObj );
    attributeAffects( screenXFormObj,       outputObj );
    attributeAffects( imageFilePathNameObj, outputObj );

    return stat;
}

MStatus BeyondScreenTest::compute( const MPlug& plug, MDataBlock& block )
{
    if( plug != outputObj ) { return MS::kUnknownParameter; }
    MThreadUtils::syncNumOpenMPThreads();

    Vector  worldCameraPosition    = block.inputValue( cameraPositionObj    ).asFloat3();
    Vector  worldAimingPoint       = block.inputValue( aimPointObj          ).asFloat3();
    Vector  worldProjectorPosition = block.inputValue( projectorPositionObj ).asFloat3();
    MObject screenMeshShapeObj     = block.inputValue( screenMeshObj        ).asMesh();
    MMatrix screenXForm            = block.inputValue( screenXFormObj       ).asMatrix();
    MString imageFilePathName      = block.inputValue( imageFilePathNameObj ).asString();

    // load necessary data
    {
        Convert( manager.objectScreenMesh, screenMeshShapeObj );
        manager.objectScreenCenter = manager.objectScreenMesh.center();
        manager.worldAimingPoint = worldAimingPoint;
        manager.worldProjectorPosition = worldProjectorPosition;

        manager.animationFrames.append( 0 );

        manager.objectToWorldMatrices.resize(1);
        Copy( manager.objectToWorldMatrices[0], screenXForm );

        manager.worldCameraPositions.resize(1);
        manager.worldCameraPositions[0] = worldCameraPosition;

        manager.computeDerivedData( 0 );
    }

    // debugging by loading cache data
    //manager.load( "/home/wanho.choi/BeyondScreen.data" );
    //manager.computeDerivedData( 0 );

    if( ( manager.imageFilePathName != imageFilePathName.asChar() ) && ( imageFilePathName != "" ) )
    {
        manager.imageFilePathName = imageFilePathName.asChar();
        manager.image.load( manager.imageFilePathName.asChar() );
    }

    const int imgWidth  = manager.image.width();
    const int imgHeight = manager.image.height();

    const int nVerts = manager.localScreenMesh.numVertices();

    vertexColors.setLength( nVerts );

    if( imgWidth*imgHeight > 0 )
    {
        const Vector viewDirection = ( manager.localScreenCenter - manager.localCameraPosition ).normalize();

        const Vector verticalBase( 0.0, 1.0, 0.0 );
        const Vector horizontalBase = viewDirection.cross( verticalBase ).normalize();

        #pragma omp parallel for
        for( int i=0; i<nVerts; ++i )
        {
            const Vector pixelDirection = ( manager.localScreenMesh.p[i] - manager.localCameraPosition ).normalize();

            // -1.0 ~ +1.0
            const double x =  pixelDirection.dot( horizontalBase );
            const double y = -pixelDirection.dot( verticalBase   );

            // 0.0 ~ +1.0
            const double s = Clamp( 0.5*x+0.5, 0.0, 1.0 );
            const double t = Clamp( 0.5*y+0.5, 0.0, 1.0 );

            const int pixel_i = Clamp( s * (imgWidth -1), 0, (imgWidth -1) );
            const int pixel_j = Clamp( t * (imgHeight-1), 0, (imgHeight-1) );

            const BeyondScreen::Pixel& pixel = manager.image( pixel_i, pixel_j );

            vertexColors[i].set( pixel.r, pixel.g, pixel.b );
        }
    }

    block.outputValue( outputObj ).set( 0.f );
    block.setClean( plug );

    return MS::kSuccess;
}

void BeyondScreenTest::draw( M3dView& view, const MDagPath& path, M3dView::DisplayStyle style, M3dView::DisplayStatus displayStatus )
{
	view.beginGL();
	glPushAttrib( GL_ALL_ATTRIB_BITS );
    {
//        manager.worldScreenMesh.draw( &vertexColors );

//        glColor3f(1,0,0);
//        manager.worldScreenMesh.draw(); //( &vertexColors );

//        glPointSize( 10 );
//        glColor( 1, 0, 0 );
//        glBegin( GL_POINTS );
//            glVertex( manager.worldCornerPoint( 0 ) );
//            glVertex( manager.worldCornerPoint( 1 ) );
//            glVertex( manager.worldCornerPoint( 2 ) );
//            glVertex( manager.worldCornerPoint( 3 ) );
//        glEnd();

        glColor(1,0,0);
        manager.worldScreenMesh.draw();

        glColor(0,0,1);
        manager.localScreenMesh.draw();
        manager.localScreenAABB.draw();
    }
	glPopAttrib();
	view.endGL();
}

bool BeyondScreenTest::isBounded() const
{
	return false;
}

