//---------//
// ZMesh.h //
//-------------------------------------------------------//
// author: Taeyong Kim @ nVidia                          //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2016.11.14                               //
//-------------------------------------------------------//

#ifndef _ZMesh_h_
#define _ZMesh_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

/// @brief Polygonal mesh.
class ZMesh
{
	protected:

		ZPointArray       _points;			// vertex positions (x,y,z)
		ZPointArray       _uvs;				// texture coordinates (u,v,w)
		ZMeshElementArray _elements;		// elements

	public:

		ZMesh();
		ZMesh( const ZMesh& source );
		ZMesh( const char* filePathName );

		void reset();

		ZMesh& operator=( const ZMesh& mesh );

		int numVertices() const;
		int numUVs() const;
		int numElements() const;

		bool empty() const;

		// the position of the i-th vertex
		ZPoint& operator[]( int i ) { return _points[i]; }
		const ZPoint& operator[]( int i ) const { return _points[i]; }

		bool create( const ZPointArray& vertexPositions, const ZIntArray& polyCounts, const ZIntArray& polyConnections, ZMeshElementType::MeshElementType type );

		bool assignUVs( const ZPointArray& uvs, const ZIntArray& uvIndices );

		int addPoint( const ZPoint& point );
		int addPoints( const ZPointArray& points );

		int addUV( const ZPoint& uv );
		int addUVs( const ZPointArray& uvs );

		ZMeshElement& addElement( const ZMeshElement& element );

		ZMeshElement& addLine( int v0, int v1, int id=0 );
		ZMeshElement& addLine( const ZPoint& p0, const ZPoint& p1, int id=0 );
		ZMeshElement& addLine( int v0, int v1, int uv0, int uv1, int id=0 );

		ZMeshElement& addTri( int v0, int v1, int v2, int id=0 );
		ZMeshElement& addTri( const ZPoint& p0, const ZPoint& p1, const ZPoint& p2, int id=0 );
		ZMeshElement& addTri( int v0, int v1, int v2, int uv0, int uv1, int uv2, int id=0 );

		ZMeshElement& addTet( int v0, int v1, int v2, int v3, int id=0 );
		ZMeshElement& addTet( const ZPoint& p0, const ZPoint& p1, const ZPoint& p2, const ZPoint& p3, int id=0 );
		ZMeshElement& addTet( int v0, int v1, int v2, int v3, int uv0, int uv1, int uv2, int uv3, int id=0 );

		ZMeshElement& addQuad( int v00, int v10, int v11, int v01, int id=0 );
		ZMeshElement& addQuad( const ZPoint& p00, const ZPoint& p10, const ZPoint& p11, const ZPoint& p01, int id=0 );
		ZMeshElement& addQuad( int v00, int v10, int v11, int v01, int uv00, int uv10, int uv11, int uv01, int id=0 );

		ZMeshElement& addCube( int v000, int v100, int v101, int v001, int v010, int v110, int v111, int v011, int id=0 );
		ZMeshElement& addCube( const ZPoint& p000, const ZPoint& p100, const ZPoint& p101, const ZPoint& p001, const ZPoint& p010, const ZPoint& p110, const ZPoint& p111, const ZPoint& p011, int id=0 );
		ZMeshElement& addCube( int v000, int v100, int v101, int v001, int v010, int v110, int v111, int v011, int uv000, int uv100, int uv101, int uv001, int uv010, int uv110, int uv111, int uv011, int id=0 );

		void append( const ZMesh& other );

		bool deleteElement( int i );
		int deleteElements( const ZIntArray& list );

		void deleteUnusedPointsAndUVs();

		ZBoundingBox boundingBox() const;

		ZMeshElementArray& elements() { return _elements; }
		const ZMeshElementArray& elements() const { return _elements; }

		ZMeshElement& element( int i ) { return _elements[i]; }
		const ZMeshElement& element( int i ) const { return _elements[i]; }

		const ZPointArray& points() const { return _points; }
		ZPointArray& points() { return _points; }

		ZPointArray& uvs() { return _uvs; }
		const ZPointArray& uvs() const { return _uvs; }

		void getMeshInfo( ZIntArray& polyCounts, ZIntArray& polyConnections ) const;

		void getConnections( ZIntArray& polyConnections ) const;

		void getVertexNormals( ZVectorArray& normals ) const;
		void getElementNormals( ZVectorArray& normals ) const;

		ZMesh& triangulate();
		bool isTriangulated() const;
		void getTriangleIndices( ZInt3Array& triConnections ) const;
		void getTriangleCenters( ZPointArray& centers, int computingMethod, bool useOpenMP=false ) const;

		void getQuadrilateralIndices( ZInt4Array& quadConnections ) const;
		void getTetrahedronIndices( ZInt4Array& tetConnections ) const;

		void weld( float epsilon=Z_EPS );

		void reverse();

		void exchange( ZMesh& mesh );

		void transform( const ZMatrix& matrix, bool useOpenMP=false );

		const ZString dataType() const;

		void write( ofstream& fout ) const;
		void read( ifstream& fin );

		bool save( const char* filePathName ) const;
		bool load( const char* filePathName );

		void draw( ZMeshDisplayMode::MeshDisplayMode mode, const ZColor& lineColor=ZColor(0.5f), const ZColor& surfaceColor=ZColor(0.8f), float opacity=1.f ) const;
		void drawVertexNormals( const ZVectorArray& normals, const ZColor& lineColor=ZColor(1,0,0), float scale=1.f ) const;
		void drawUVs( const ZColor& lineColor=ZColor(0,0,1) ) const;

		void drawVertices() const;
		void drawWireframe() const;
		void drawSurface( ZVectorArray* vNrmPtr=NULL ) const;
		void drawWireSurface( const ZColor& wireColor=ZColor::gray(), ZVectorArray* vNrmPtr=NULL ) const;

	private:

		int _findUnusedPoints( ZIntArray& unused );
		int _findUnusedUVs( ZIntArray& unused );

		void _deletePoints( const ZIntArray& toBeDeleted );
		void _deleteUVs( const ZIntArray& toBeDeleted );

		void _updateElementVertices( ZIntArray& newVertexTable );
		void _updateElementUVs( ZIntArray& newUVTable );
};

inline int
ZMesh::numVertices() const
{
	return (int)_points.length();
}

inline int
ZMesh::numUVs() const
{
	return (int)_uvs.size();
}

inline int
ZMesh::numElements() const
{
	return (int)_elements.size();
}

inline bool
ZMesh::empty() const
{
	if( !_points.length() ) { return true; }
	if( !_elements.length() ) { return true; }
	return false;
}

ostream&
operator<<( ostream& os, const ZMesh& object );

ZELOS_NAMESPACE_END

#endif

