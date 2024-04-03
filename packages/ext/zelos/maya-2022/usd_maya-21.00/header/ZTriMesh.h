//------------//
// ZTriMesh.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.11.14                               //
//-------------------------------------------------------//

#ifndef _ZTriMesh_h_
#define _ZTriMesh_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

/// @brief Triangle mesh.
class ZTriMesh
{
	public:

		ZPointArray  p;		// vertex positions; length = (# of vertices)
		ZInt3Array   v012;	// triangle vertices; length = (# of triangles)
		ZPointArray  uv;	// uv coordinates; length = (# of triangles) x 3

	public:

		ZTriMesh();
		ZTriMesh( const ZTriMesh& mesh );
		ZTriMesh( const char* filePathName );

		void reset();

		int numVertices() const;
		int numTriangles() const;
		int numUVs() const;

		bool empty() const;

		ZTriMesh& operator=( const ZTriMesh& mesh );

		void transform( const ZMatrix& matrix, bool useOpenMP=false );
		
		void transform( const ZMatrix& matrix, const ZPoint& pivot, bool useOpenMP=false );

		ZPoint center( int triIdx ) const;

		ZBoundingBox boundingBox( bool useOpenMP=true ) const;

		double volume( bool useOpenMP=true ) const;
		double centerOfMass( ZPoint& cm, bool useOpenMP=true ) const;

		void reverse();

		void getPositions( const ZIntArray& triIndices, const ZFloat3Array& baryCoords, ZPointArray& positions, bool useOpenMP=true ) const;
		void getUVs( const ZIntArray& triIndices, const ZFloat3Array& baryCoords, ZPointArray& uvs, bool useOpenMP=true ) const;
		void getNormals( const ZIntArray& triIndices, const ZFloat3Array& baryCoords, ZPointArray& normals, bool useOpenMP=true ) const;

		void getTriangleCenters( ZPointArray& centers, bool useOpenMP=true ) const;
		void getVertexNormals( ZVectorArray& normals, bool useOpenMP=true ) const;
		void getTriangleNormals( ZVectorArray& normals, bool useOpenMP=true ) const;

		double area( bool useOpenMP=true ) const;
		void getTriangleAreas( ZFloatArray& areas, bool useOpenMP=true ) const;

		void getMinMaxEdgeLength( float& min, float& max ) const;
		void getMinMaxUVEdgeLength( float& min, float& max ) const;

		void getTriangleCenterValue( const char* densityMapFilePathName, ZFloatArray& values, int channel=-1, bool useOpenMP=true ) const;

		void combine( const ZTriMesh& mesh );

		void deleteTriangles( const ZIntArray& indicesToBeDeleted );

		double usedMemorySize( ZDataUnit::DataUnit dataUnit ) const;

		const ZString dataType() const;

		void write( ofstream& fout ) const;
		void read( ifstream& fin );
	
		void exchange( ZTriMesh& mesh );

		bool save( const char* filePathName ) const;
		bool load( const char* filePathName );

		void drawVertices() const;
		void drawWireframe() const;
		void drawSurface( bool withNormal=false ) const;

		void draw( ZMeshDisplayMode::MeshDisplayMode mode, const ZColor& lineColor=ZColor(0.5f), const ZColor& surfaceColor=ZColor(0.8f), float opacity=1.f ) const;
		void draw( const ZVectorArray& vNrm ) const;
		void drawVertexNormals( const ZVectorArray& vNrm, float scale=1.f ) const;
		void drawUVs() const;
		void drawWithTexture( bool flipInY=false ) const;
};

inline int
ZTriMesh::numVertices() const
{
	return (int)p.size();
	}

inline int
ZTriMesh::numTriangles() const
{
	return (int)v012.size();
	}

inline int
ZTriMesh::numUVs() const
{
	return (int)uv.size();
}

inline bool
ZTriMesh::empty() const
{
	if( !p.size() ) { return true; }
	if( !v012.size() ) { return true; }
	return false;
}

inline void
ZTriMesh::exchange( ZTriMesh& mesh )
{
	p.swap( mesh.p );
	v012.swap( mesh.v012 );
	uv.swap( mesh.uv );
}

ostream&
operator<<( ostream& os, const ZTriMesh& object );

ZELOS_NAMESPACE_END

#endif

