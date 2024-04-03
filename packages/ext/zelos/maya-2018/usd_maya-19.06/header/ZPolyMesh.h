//-------------//
// ZPolyMesh.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.12.23                               //
//-------------------------------------------------------//

#ifndef _ZPolyMesh_h_
#define _ZPolyMesh_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

// @brief Polygonal mesh.
class ZPolyMesh
{
	public:

		ZPointArray  vertexPositions;
		ZIntArray    polygonCounts;
		ZIntArray    polygonConnects;
		ZBoundingBox boundingBox;
		ZVectorArray vertexNormals;
		ZFloatArray  u;
		ZFloatArray  v;
		ZIntArray    uvIndices;

	public:

		ZPolyMesh();
		ZPolyMesh( const ZPolyMesh& mesh );
		ZPolyMesh( const char* filePathName );

		void reset();

		int numVertices() const;
		int numPolygons() const;
		int numUVs() const;

		bool empty() const;
		bool hasVertexNormals() const;
		bool hasBoundingBox() const;

		ZPolyMesh& operator=( const ZPolyMesh& mesh );

		void transform( const ZMatrix& matrix, bool useOpenMP=false );

		void append( const ZPolyMesh& other, bool uv=false );

		void computeVertexNormals();

		void computeBoundingBox();

		void reverse();

		void convertTo( ZTriMesh& mesh );

		double usedMemorySize( ZDataUnit::DataUnit dataUnit ) const;

		const ZString dataType() const;

		void write( ofstream& fout ) const;
		void read( ifstream& fin );

		bool save( const char* filePathName ) const;
		bool load( const char* filePathName );

		void drawVertices() const;

		void drawWireframe() const;

		void drawSurface( bool withNormal=false ) const;
};

inline int
ZPolyMesh::numVertices() const
{
	return (int)vertexPositions.size();
}

inline int
ZPolyMesh::numPolygons() const
{
	return (int)polygonCounts.size();
}

inline int
ZPolyMesh::numUVs() const
{
	return (int)u.size();
}

inline bool
ZPolyMesh::empty() const
{
	if( !vertexPositions.size() ) { return true; }
	if( !polygonCounts.size() ) { return true; }
	return false;
}

inline bool
ZPolyMesh::hasVertexNormals() const
{
	if( !vertexNormals.size() ) { return false; }
	if( vertexNormals.size() != vertexPositions.size() ) { return false; }
	return true;
}

inline bool
ZPolyMesh::hasBoundingBox() const
{
	return boundingBox.initialized();
}

ostream&
operator<<( ostream& os, const ZPolyMesh& object );

ZELOS_NAMESPACE_END

#endif

