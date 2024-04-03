//-------------//
// ZQuadMesh.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.28                               //
//-------------------------------------------------------//

#ifndef _ZQuadMesh_h_
#define _ZQuadMesh_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

// @brief Quadrilateral mesh.
class ZQuadMesh
{
	public:

		ZPointArray  p;		// vertex positions; length = (# of vertices)
		ZInt4Array   v0123;	// quad vertices; length = (# of quads)
		ZPointArray  uv;	// uv coordinates; length = (# of quads) x 4

	public:

		ZQuadMesh();
		ZQuadMesh( const ZQuadMesh& m );
		ZQuadMesh( const char* filePathName );

		void reset();

		int numVertices() const;
		int numQuads() const;
		int numUVs() const;

		ZQuadMesh& operator=( const ZQuadMesh& mesh );

		void transform( const ZMatrix& matrix, bool useOpenMP=false );

		ZBoundingBox boundingBox() const;

		void reverse();

		void getVertexNormals( ZVectorArray& normals ) const;

		void combine( const ZQuadMesh& mesh );

		void getTriangleIndices( ZInt3Array& triangles ) const;

		double usedMemorySize( ZDataUnit::DataUnit dataUnit ) const;

		const ZString dataType() const;

		void write( ofstream& fout ) const;
		void read( ifstream& fin );

		bool save( const char* filePathName ) const;
		bool load( const char* filePathName );

		void drawVertices() const;
		void drawWireframe() const;
		void drawSurface( bool withNormal=false ) const;

		void draw( ZMeshDisplayMode::MeshDisplayMode mode, const ZColor& lineColor=ZColor(0.5f), const ZColor& surfaceColor=ZColor(0.8f), float opacity=1.f ) const;
		void drawVertexNormals( const ZVectorArray& vNrm, float scale=1.f ) const;
		void drawUVs() const;
};

inline int
ZQuadMesh::numVertices() const
{ return p.length(); }

inline int
ZQuadMesh::numQuads() const
{ return v0123.length(); }

inline int
ZQuadMesh::numUVs() const
{ return (4*uv.length()); }

ostream&
operator<<( ostream& os, const ZQuadMesh& object );

ZELOS_NAMESPACE_END

#endif

