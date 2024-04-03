//--------------------------//
// ZTriMeshConnectionInfo.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.12.04                               //
//-------------------------------------------------------//

#ifndef _ZTriMeshConnectionInfo_h_
#define _ZTriMeshConnectionInfo_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZTriMeshConnectionInfo
{
	private:

		int  _numVertices;
		int  _numTriangles;

	public: // v:vertex, e:edge, t:triangle

		ZIntArrayList v2v;	// one-ring vertices
		ZIntArrayList e2v;	// edges-to-vertex list
		ZIntArrayList t2v;	// triangles-to-vertex list

		ZInt2Array    v2e;	// edge list
		ZIntArrayList e2e;	// not necessary
		ZInt2Array    t2e;	// triangles-to-edge list

		ZInt3Array    v2t;	// v012
		ZInt3Array    e2t;	// edges-to-triangle list
		ZInt3Array    t2t;	// triangles-to-triangle list

	public:

		ZTriMeshConnectionInfo();
		ZTriMeshConnectionInfo( const ZTriMesh& mesh );

		void reset();

		void set( const ZTriMesh& mesh );

		void calculate_v2v(); // from v2t
		void calculate_e2v(); // from v2e
		void calculate_t2v(); // from v2t

		void calculate_v2e(); // from t2t
		void calculate_e2e(); // none
		void calculate_t2e(); // from t2e 

		void calculate_v2t(); // none (v2t = mesh.v012)
		void calculate_e2t(); // from e2v
		void calculate_t2t(); // fromo v2t
};

ostream&
operator<<( ostream& os, const ZTriMeshConnectionInfo& object );

ZELOS_NAMESPACE_END

#endif

