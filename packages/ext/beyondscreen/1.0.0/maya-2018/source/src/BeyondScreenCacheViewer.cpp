#include <BeyondScreenCacheViewer.h>

MTypeId BeyondScreenCacheViewer::id( 0x71035 );
MString BeyondScreenCacheViewer::name( "BeyondScreenCacheViewer" );

MObject BeyondScreenCacheViewer::timeObj;
MObject BeyondScreenCacheViewer::cacheFileStrObj; // BeyondScreen info cache path
MObject BeyondScreenCacheViewer::imgPlaneXFormObj;
MObject BeyondScreenCacheViewer::worldViewPointObj;
MObject BeyondScreenCacheViewer::outputObj;
MObject BeyondScreenCacheViewer::frameRangeObj;
MObject BeyondScreenCacheViewer::worldScreenCenterObj;
MObject BeyondScreenCacheViewer::drawScreenMeshObj;
MObject BeyondScreenCacheViewer::screenMeshColorObj;
MObject BeyondScreenCacheViewer::drawAimingPointObj;
MObject BeyondScreenCacheViewer::aimingPointColorObj;
MObject BeyondScreenCacheViewer::drawCameraPositionObj;
MObject BeyondScreenCacheViewer::cameraPositionColorObj;

void* BeyondScreenCacheViewer::creator()
{
	return new BeyondScreenCacheViewer();
}

void BeyondScreenCacheViewer::postConstructor()
{
    MPxNode::postConstructor();

    nodeObj = thisMObject();
    nodeFn.setObject( nodeObj );
    dagNodeFn.setObject( nodeObj );
    nodeFn.setName( "BeyondScreenCacheViewer#" );
}

MStatus BeyondScreenCacheViewer::initialize()
{
    MStatus stat = MS::kSuccess;

    MFnUnitAttribute    uAttr;
    MFnTypedAttribute   tAttr;
    MFnMatrixAttribute  mAttr;
    MFnNumericAttribute nAttr;

	timeObj = uAttr.create( "time", "time", MFnUnitAttribute::kTime, 0.0 );
	uAttr.setHidden(1);
	CHECK_MSTATUS( addAttribute( timeObj ) );

	cacheFileStrObj = tAttr.create( "cacheFile", "cacheFile", MFnData::kString );
    CHECK_MSTATUS( addAttribute( cacheFileStrObj ) );

    imgPlaneXFormObj = mAttr.create( "imgPlaneXForm", "imgPlaneXForm", MFnMatrixAttribute::kDouble );
    mAttr.setHidden(1);
    CHECK_MSTATUS( addAttribute( imgPlaneXFormObj ) );

	worldViewPointObj = nAttr.createPoint( "worldViewPoint", "worldViewPoint" );
    nAttr.setHidden(1);
    CHECK_MSTATUS( addAttribute( worldViewPointObj ) );

	outputObj = nAttr.create( "output", "output", MFnNumericData::kFloat, 0.f );
    nAttr.setHidden(1);
    CHECK_MSTATUS( addAttribute( outputObj ) );

    frameRangeObj = nAttr.create( "frameRange", "frameRange", MFnNumericData::k2Int, 0 );
    nAttr.setWritable(0); nAttr.setStorable(0);
    CHECK_MSTATUS( addAttribute( frameRangeObj ) );

    worldScreenCenterObj = nAttr.createPoint( "worldScreenCenter", "worldScreenCenter" );
    nAttr.setHidden(1);
    CHECK_MSTATUS( addAttribute( worldScreenCenterObj ) );

    drawScreenMeshObj = nAttr.create( "drawScreenMesh", "drawScreenMesh", MFnNumericData::kBoolean, false );
    CHECK_MSTATUS( addAttribute( drawScreenMeshObj ) );

    screenMeshColorObj = nAttr.createColor( "screenMeshColor", "screenMeshColor" );
    nAttr.setHidden(1); nAttr.setChannelBox(1); nAttr.setDefault(0.75,0.75,0.75);
    CHECK_MSTATUS( addAttribute( screenMeshColorObj ) );

    drawAimingPointObj = nAttr.create( "drawAimingPoint", "drawAimingPoint", MFnNumericData::kBoolean, false );
    CHECK_MSTATUS( addAttribute( drawAimingPointObj ) );

    aimingPointColorObj = nAttr.createColor( "aimingPointColor", "aimingPointColor" );
    nAttr.setHidden(1); nAttr.setChannelBox(1); nAttr.setDefault(1.0,0.0,0.0);
    CHECK_MSTATUS( addAttribute( aimingPointColorObj ) );

    drawCameraPositionObj = nAttr.create( "drawCameraPosition", "drawCameraPosition", MFnNumericData::kBoolean, false );
    CHECK_MSTATUS( addAttribute( drawCameraPositionObj ) );

    cameraPositionColorObj = nAttr.createColor( "cameraPositionColor", "cameraPositionColor" );
    nAttr.setHidden(1); nAttr.setChannelBox(1); nAttr.setDefault(0.0,1.0,0.0);
    CHECK_MSTATUS( addAttribute( cameraPositionColorObj ) );

    attributeAffects( timeObj,           outputObj );
    attributeAffects( cacheFileStrObj,   outputObj );
    attributeAffects( imgPlaneXFormObj,  outputObj );
    attributeAffects( worldViewPointObj, outputObj );

    return stat;
}

MStatus BeyondScreenCacheViewer::compute( const MPlug& plug, MDataBlock& block )
{
    if( plug != outputObj ) { return MS::kUnknownParameter; }
    MThreadUtils::syncNumOpenMPThreads();

	const int frame = (int)block.inputValue( timeObj ).asTime().as( MTime::uiUnit() );

    const MString cacheFileStr   = block.inputValue( cacheFileStrObj   ).asString();
    const MMatrix imgPlaneMatrix = block.inputValue( imgPlaneXFormObj  ).asMatrix();
    const Vector  worldViewPoint = block.inputValue( worldViewPointObj ).asFloat3();

    Copy( imgPlaneXForm, imgPlaneMatrix );
    imgPlaneXForm.transpose();

    failed = false;

    if( cacheFile != cacheFileStr )
    {
        if( manager.load( cacheFileStr.asChar() ) == false )
        {
            failed = true;
            return MS::kSuccess;
        }

        {
            startFrame =  100000000;
            endFrame   = -100000000;

            const IntArray& animationFrames = manager.animationFrames;

            for( size_t i=0; i<animationFrames.size(); ++i )
            {
                startFrame = MIN( startFrame, animationFrames[i] );
                endFrame   = MAX( endFrame,   animationFrames[i] );
            }
        }

        cacheFile = cacheFileStr;
    }

    if( ( frame < startFrame ) || ( frame > endFrame ) )
    {
        failed = true;
    }
    else
    {
        manager.setDrawingData( frame - startFrame );

        const Float2 st = manager.worldToST( worldViewPoint );

        s = st[0];
        t = st[1];

        P = manager.worldCameraPosition;
        Q = worldViewPoint;

        R = manager.worldScreenMesh.intersectionPoint( P, Q );
    }

    const Vector wc = manager.worldScreenMesh.center();
    block.outputValue( worldScreenCenterObj ).set( (float)wc.x, (float)wc.y, (float)wc.z );
    block.outputValue( frameRangeObj ).set( startFrame, endFrame );
    block.outputValue( outputObj ).set( 0.f );
    block.setClean( plug );

    return MS::kSuccess;
}

void BeyondScreenCacheViewer::draw( M3dView& view, const MDagPath& path, M3dView::DisplayStyle style, M3dView::DisplayStatus displayStatus )
{
    if( failed ) { return; }

    bool drawScreenMesh = MPlug( nodeObj, drawScreenMeshObj ).asBool();

	view.beginGL();
	glPushAttrib( GL_ALL_ATTRIB_BITS );
    {

		glEnable( GL_BLEND );
		glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
		glEnable( GL_POINT_SMOOTH );
		glEnable( GL_LINE_SMOOTH );

        glPointSize( 10 );
        glLineWidth( 3 );

        if( MPlug( nodeObj, drawScreenMeshObj ).asBool() )
        {
            glColor( Color( nodeObj, screenMeshColorObj ) );
            Draw( manager.worldScreenMesh );
        }

        glBegin( GL_POINTS );

        if( MPlug( nodeObj, drawAimingPointObj ).asBool() )
        {
            glColor( Color( nodeObj, aimingPointColorObj ) );
            glVertex( manager.worldAimingPoint );
        }

        if( MPlug( nodeObj, drawCameraPositionObj ).asBool() )
        {
            glColor( Color( nodeObj, cameraPositionColorObj ) );
            glVertex( manager.worldCameraPosition );
        }

        glEnd();

//        glBegin( GL_POINTS );
//        {
//            glColor(1,0,0);
//            glVertex( manager.worldAimingPoint );
//
//            glColor(0,1,0);
//            glVertex( manager.worldCameraPosition );
//        }

        glBegin( GL_LINES );
        {
            glColor( 1, 1, 0 );
            glVertex( P );
            glVertex( Q );
        }
        glEnd();

        glPushMatrix();
        glMultMatrixd( &imgPlaneXForm._00 );
        {
            glPushMatrix();
            glTranslatef( -0.5, 0.0, -0.5 );
            {
                glColor(1,0,0);
                glBegin( GL_POINTS );
                    glVertex( s, 0.0, 1-t );
                glEnd();
            }
            glPopMatrix();
        }
        glPopMatrix();

        //glColor( 1, 1, 1 );
        //Draw( manager.worldScreenMesh );
    }
	glPopAttrib();
	view.endGL();
}

MBoundingBox BeyondScreenCacheViewer::boundingBox() const
{
    MBoundingBox bBox;

    const BoundingBox aabb = manager.worldScreenMeshAABB;

    if( aabb.initialized() )
    {
        bBox.expand( MPoint( aabb.min().x, aabb.min().y, aabb.min().z ) );
        bBox.expand( MPoint( aabb.max().x, aabb.max().y, aabb.max().z ) );
    }

    return bBox;
}

bool BeyondScreenCacheViewer::isBounded() const
{
	return true;
}

