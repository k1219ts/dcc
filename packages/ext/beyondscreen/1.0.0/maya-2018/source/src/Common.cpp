#include <Common.h>

bool GetWorldMatrix( const MObject& dagNodeObj, MMatrix& worldMat )
{
    MStatus status = MS::kSuccess;

    MFnDagNode dagFn( dagNodeObj, &status );
    if( status != MS::kSuccess ) { return false; } // Don't print any error messages here!

    MDagPath dagPath;
    status = dagFn.getPath( dagPath );
    if( status != MS::kSuccess ) { return false; } // Don't print any error messages here!

    worldMat = dagPath.inclusiveMatrix();

    return true;
}

Vector Translation( const MMatrix& m )
{
    return Vector( m(3,0), m(3,1), m(3,2) );
}

Vector GetWorldPosition( const MString& xformNodeName )
{
    MDoubleArray ret;
    MGlobal::executeCommand( MString("WorldPosition ") + xformNodeName, ret );
    return Vector( ret[0], ret[1], ret[2] );
}

Vector GetWorldUpvector( const MString& xformNodeName )
{
    MDoubleArray ret;
    MGlobal::executeCommand( MString("WorldUpvector ") + xformNodeName, ret );
    return Vector( ret[0], ret[1], ret[2] );
}

void ApplyXForm( const MMatrix& M, const MPoint& p, MPoint& q )
{
    const double (*matrix)[4] = M.matrix;
    const double &x=p.x, &y=p.y, &z=p.z;

    q.x = matrix[0][0]*x + matrix[1][0]*y + matrix[2][0]*z + matrix[3][0];
    q.y = matrix[0][1]*x + matrix[1][1]*y + matrix[2][1]*z + matrix[3][1];
    q.z = matrix[0][2]*x + matrix[1][2]*y + matrix[2][2]*z + matrix[3][2];
}

void Copy( VectorArray& to, const MPointArray& from )
{
	const int n = (int)from.length();
	if( !n ) { to.clear(); return; }

	to.setLength( n );

	for( int i=0; i<n; ++i )
	{
		const MPoint& p = from[i];

		to[i].set( p.x, p.y, p.z );
	}
}

void Copy( Matrix& to, const MMatrix& from )
{
    for( int i=0; i<4; ++i )
    for( int j=0; j<4; ++j )
    {{
        to(i,j) = from(j,i);
    }}
}

bool Convert( ScreenMesh& mesh, MObject& meshObj, bool vPosOnly, const char* uvSetName )
{
    MStatus status = MS::kSuccess;

    if( !vPosOnly ) { mesh.clear(); }

    MFnMesh        meshFn ( meshObj );
    MItMeshVertex  vItr   ( meshObj );
    MItMeshPolygon fItr   ( meshObj );

    const int numVertices = meshFn.numVertices();

	//////////////////////
	// vertex positions //

    int         vIdx = 0;
    MPoint      localPos;
    MMatrix     localToWorld;
    MPointArray vP( numVertices ); // vertex positions

    for( vItr.reset(); !vItr.isDone(); vItr.next(), ++vIdx )
    {
        localPos = vItr.position( MSpace::kObject );
        ApplyXForm( localToWorld, localPos, vP[vIdx] );
    }

	Copy( mesh.p, vP ); // mesh.p <- vP

    if( vPosOnly ) { return true; }

	/////////////////
	// UV-set name //

    bool toConvertUV = true;
    MString uvSetNameStr;
    MFloatArray vU, vV;

    if( !meshFn.numUVSets() || !meshFn.numUVs() )
    {
        toConvertUV = false;
    }
    else
    {
        MString inUVSetName( uvSetName );

        if( inUVSetName.length() == 0 )
        {
            toConvertUV = false;
        }
        else if( inUVSetName == MString("currentUVSet") )
        {
            uvSetNameStr = meshFn.currentUVSetName();
        }
        else
        {
            MStringArray uvSetNames;
            meshFn.getUVSetNames( uvSetNames );
            const int numUVSets = (int)uvSetNames.length();

            for( int i=0; i<numUVSets; ++i )
            {
                if( inUVSetName == uvSetNames[i] )
                {
                    uvSetNameStr = inUVSetName;
                    break;
                }
            }
        }
    }

    if( toConvertUV )
    {
        if( !meshFn.getUVs( vU, vV, &uvSetNameStr ) )
        {
            toConvertUV = false;
        }
    }

    if( toConvertUV == false )
    {
        return false;
    }

	//////////////////////
	// triangle indices //

    UIntArray& triangles = mesh.t;
    triangles.reserve( meshFn.numPolygons()*2*3 );

    MIntArray vList;

    for( fItr.reset(); !fItr.isDone(); fItr.next() )
    {
        fItr.getVertices( vList );
        const int vCount = (int)vList.length();

        if( vCount < 3 ) { continue; } // invalid case

        for( int i=0; i<vCount-2; ++i )
        {
            triangles.append( vList[0]   );
            triangles.append( vList[i+1] );
            triangles.append( vList[i+2] );
        }
    }

    VectorArray& uv = mesh.uv;
    uv.resize( mesh.numVertices() );

    if( toConvertUV )
    {
        int triangleIndex = 0;

        for( fItr.reset(); !fItr.isDone(); fItr.next() )
        {
            fItr.getVertices( vList );
            const int vCount = (int)vList.length();

            if( vCount < 3 ) { continue; } // invalid case

            MIntArray uvIndices;
            uvIndices.setLength( vCount );

            for( int i=0; i<vCount; ++i )
            {
                fItr.getUVIndex( i, uvIndices[i], &uvSetNameStr );
            }

            for( int i=0; i<vCount-2; ++i )
            {
                const int index = 3*triangleIndex;

                const int& vrt0 = mesh.t[index  ];
                const int& vrt1 = mesh.t[index+1];
                const int& vrt2 = mesh.t[index+2];

                uv[vrt0].x = vU[uvIndices[0  ]];
                uv[vrt0].y = vV[uvIndices[0  ]];

                uv[vrt1].x = vU[uvIndices[i+1]];
                uv[vrt1].y = vV[uvIndices[i+1]];

                uv[vrt2].x = vU[uvIndices[i+2]];
                uv[vrt2].y = vV[uvIndices[i+2]];

                ++triangleIndex;
            }
        }
    }

	return true;
}

MObject NodeNameToMObject( const MString& nodeName )
{
    MObject obj;
    MSelectionList sList;
    MStatus stat = MGlobal::getSelectionListByName( nodeName, sList );
    if( !stat ) { MGlobal::displayError( "Error@DagPath(): Failed." ); return MObject::kNullObj; }
    sList.getDependNode( 0, obj );
    return obj;
}

MDagPath NodeNameToDagPath( const MString& dagNodeName )
{
    MDagPath dagPath;
    MSelectionList sList;
    MStatus stat = MGlobal::getSelectionListByName( dagNodeName, sList );
    if( !stat ) { MGlobal::displayError("Error@DagPath(): Failed."); return MDagPath(); }
    sList.getDagPath( 0, dagPath );
    return dagPath;
}

Vector Color( const MObject& nodeObj, const MObject& attrObj )
{
    Vector c;

    MPlug plg( nodeObj, attrObj );

    plg.child(0).getValue( c.x );
    plg.child(1).getValue( c.y );
    plg.child(2).getValue( c.z );

    return c;
}

