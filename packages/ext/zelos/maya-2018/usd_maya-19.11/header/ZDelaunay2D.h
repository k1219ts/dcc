//---------------//
// ZDelaunay2D.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// modified from Magic Software library                  //
// last update: 2017.02.14                               //
//-------------------------------------------------------//

/// @brief 2D Delaunay triangulator

// The number of triangles in the Delaunay triangulation is returned in 'numTrifaces'.
// The array 'connectionInfo' stores numTrifaces triples of indices into the vertex array vertices.
// The i-th triangle has vertices vertices[connectionInfo[3*i]], vertices[connectionInfo[3*i+1]], and vertices[connectionInfo[3*i+2]].
// No point in throwing away information obtained during the construction:
// 'neighborInfo' stores 'numTrifaces' triples of indices, each triple consisting of indices to adjacent triangles.
//  The i-th triangle has edge index pairs
//   edge0 = <connectionInfo[3*i  ],connectionInfo[3*i+1]>
//   edge1 = <connectionInfo[3*i+1],connectionInfo[3*i+2]>
//   edge2 = <connectionInfo[3*i+2],connectionInfo[3*i  ]>
// The triangle adjacent to these edges have indices
//   adj0 = neighborInfo[3*i  ]
//   adj1 = neighborInfo[3*i+1]
//   adj2 = neighborInfo[3*i+2]
// If there is no adjacent triangle, the index in neighborInfo is -1.
//
// The caller is responsible for deleting the input and output arrays.

#ifndef _ZDelaunay2D_h_
#define _ZDelaunay2D_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZDelaunay2D
{
	public:

		ZDelaunay2D( ZFloat2Array& vertices, int& numTrifaces, ZIntArray& connectionInfo, ZIntArray& neighborInfo );

		~ZDelaunay2D();

	protected:

		// for sorting to remove duplicate input points
		class ZD_SortedVertex
		{
			public:

				ZFloat2 m_kV;
				int m_iIndex;

			public:

				ZD_SortedVertex() {}

				ZD_SortedVertex( const ZFloat2& rkV, int iIndex ): m_kV(rkV)
				{ m_iIndex = iIndex; }

				bool operator==( const ZD_SortedVertex& rkSV ) const
				{ return m_kV == rkSV.m_kV; }

				bool operator!=( const ZD_SortedVertex& rkSV ) const
				{ return !( m_kV == rkSV.m_kV ); }

				bool operator<( const ZD_SortedVertex& rkSV ) const
				{
					if( m_kV[0] < rkSV.m_kV[0] ) { return true;  }
					if( m_kV[0] > rkSV.m_kV[0] ) { return false; }
					return ( m_kV[1] < rkSV.m_kV[1] );
				}
		};

		class ZD_Triangle
		{
			public:

				// vertices, listed in counterclockwise order
				int m_aiV[3];

				// adjacent triangles,
				//   a[0] points to triangle sharing edge (v[0],v[1])
				//   a[1] points to triangle sharing edge (v[1],v[2])
				//   a[2] points to triangle sharing edge (v[2],v[0])
				ZD_Triangle* m_apkAdj[3];

			public:

				ZD_Triangle ()
				{	
					FOR(i,0,3) {
						m_aiV[i] = -1;
						m_apkAdj[i] = (ZD_Triangle*)NULL;
					}
				}

				ZD_Triangle( int iV0, int iV1, int iV2, ZD_Triangle* pkA0, ZD_Triangle* pkA1, ZD_Triangle* pkA2 )
				{
					m_aiV[0] = iV0;
					m_aiV[1] = iV1;
					m_aiV[2] = iV2;

					m_apkAdj[0] = pkA0;
					m_apkAdj[1] = pkA1;
					m_apkAdj[2] = pkA2;
				}

				bool pointInCircle( const ZFloat2& rkP, const vector<ZD_SortedVertex>& rkVertex ) const
				{
					// assert: <V0,V1,V2> is counterclockwise ordered
					const ZFloat2& rkV0 = rkVertex[m_aiV[0]].m_kV;
					const ZFloat2& rkV1 = rkVertex[m_aiV[1]].m_kV;
					const ZFloat2& rkV2 = rkVertex[m_aiV[2]].m_kV;

					float dV0x = rkV0[0];
					float dV0y = rkV0[1];
					float dV1x = rkV1[0];
					float dV1y = rkV1[1];
					float dV2x = rkV2[0];
					float dV2y = rkV2[1];
					float dV3x = rkP[0];
					float dV3y = rkP[1];

					float dR0Sqr = dV0x*dV0x + dV0y*dV0y;
					float dR1Sqr = dV1x*dV1x + dV1y*dV1y;
					float dR2Sqr = dV2x*dV2x + dV2y*dV2y;
					float dR3Sqr = dV3x*dV3x + dV3y*dV3y;

					float dDiff1x = dV1x - dV0x;
					float dDiff1y = dV1y - dV0y;
					float dRDiff1 = dR1Sqr - dR0Sqr;
					float dDiff2x = dV2x - dV0x;
					float dDiff2y = dV2y - dV0y;
					float dRDiff2 = dR2Sqr - dR0Sqr;
					float dDiff3x = dV3x - dV0x;
					float dDiff3y = dV3y - dV0y;
					float dRDiff3 = dR3Sqr - dR0Sqr;

					float dDet = dDiff1x * ( dDiff2y*dRDiff3 - dRDiff2*dDiff3y )
							   - dDiff1y * ( dDiff2x*dRDiff3 - dRDiff2*dDiff3x )
							   + dRDiff1 * ( dDiff2x*dDiff3y - dDiff2y*dDiff3x );

					return ( dDet <= 0 );
				}

				bool pointLeftOfEdge( const ZFloat2& rkP, const vector<ZD_SortedVertex>& rkVertex, int i0, int i1) const
				{
					const ZFloat2& rkV0 = rkVertex[m_aiV[i0]].m_kV;
					const ZFloat2& rkV1 = rkVertex[m_aiV[i1]].m_kV;

					float dV0x = rkV0[0];
					float dV0y = rkV0[1];
					float dV1x = rkV1[0];
					float dV1y = rkV1[1];
					float dV2x = rkP[0];
					float dV2y = rkP[1];

					float dEdgex = dV1x - dV0x;
					float dEdgey = dV1y - dV0y;
					float dDiffx = dV2x - dV0x;
					float dDiffy = dV2y - dV0y;

					float dKross = dEdgex*dDiffy - dEdgey*dDiffx;
					return ( dKross >= 0 );
				}

				bool pointInTriangle( const ZFloat2& rkP, const vector<ZD_SortedVertex>& rkVertex ) const
				{
					// assert: <V0,V1,V2> is counterclockwise ordered
					const ZFloat2& rkV0 = rkVertex[m_aiV[0]].m_kV;
					const ZFloat2& rkV1 = rkVertex[m_aiV[1]].m_kV;
					const ZFloat2& rkV2 = rkVertex[m_aiV[2]].m_kV;

					float dV0x = rkV0[0];
					float dV0y = rkV0[1];
					float dV1x = rkV1[0];
					float dV1y = rkV1[1];
					float dV2x = rkV2[0];
					float dV2y = rkV2[1];
					float dV3x = rkP[0];
					float dV3y = rkP[1];

					float dEdgex = dV1x - dV0x;
					float dEdgey = dV1y - dV0y;
					float dDiffx = dV3x - dV0x;
					float dDiffy = dV3y - dV0y;

					float dKross = dEdgex*dDiffy - dEdgey*dDiffx;
					// If P right of edge <V0,V1>, so outside the triangle
					if( dKross < 0 ) { return false; }

					dEdgex = dV2x - dV1x;
					dEdgey = dV2y - dV1y;
					dDiffx = dV3x - dV1x;
					dDiffy = dV3y - dV1y;
					dKross = dEdgex*dDiffy - dEdgey*dDiffx;
					// If P right of edge <V1,V2>, outside the triangle
					if( dKross < 0 ) { return false; }

					dEdgex = dV0x - dV2x;
					dEdgey = dV0y - dV2y;
					dDiffx = dV3x - dV2x;
					dDiffy = dV3y - dV2y;
					dKross = dEdgex*dDiffy - dEdgey*dDiffx;
					// If P right of edge <V2,V0>, so outside the triangle
					if( dKross < 0 ) { return false; }

					// P left of all edges, so inside the triangle
					return true;
				}
		};

		// edges (to support constructing the insertion polygon)
		class ZD_Edge
		{
			public:

				int m_iV0, m_iV1;		// ordered vertices
				ZD_Triangle* m_pkT;		// insertion polygon triangle
				ZD_Triangle* m_pkA;		// triangle adjacent to insertion polygon

			public:

				ZD_Edge( int iV0=-1, int iV1=-1, ZD_Triangle* pkT=(ZD_Triangle*)NULL, ZD_Triangle* pkA=(ZD_Triangle*)NULL )
				{
					m_iV0 = iV0;
					m_iV1 = iV1;
					m_pkT = pkT;
					m_pkA = pkA;
				}
		};

		// sorted input vertices for processing
		vector<ZD_SortedVertex> m_kVertex;

		// indices for the supertriangle vertices
		int m_aiSuperV[3];

		// triangles that contain a supertriangle edge
		ZD_Triangle* m_apkSuperT[3];

		// the current triangulation
		set<ZD_Triangle*> m_kTriangle;

	public:

		ZD_Triangle* getContaining( const ZFloat2& rkP ) const;
		bool isInsertionComponent( const ZFloat2& rkV, ZD_Triangle* pkTri ) const;
		void GetInsertionPolygon( const ZFloat2& rkV, set<ZD_Triangle*>& rkPolyTri ) const;
		void GetInsertionPolygonEdges( set<ZD_Triangle*>& rkPolyTri, vector<ZD_Edge>& rkPoly ) const;
		void AddTriangles( int iV2, const vector<ZD_Edge>& rkPoly );
		void RemoveInsertionPolygon( set<ZD_Triangle*>& rkPolyTri );
		void RemoveTriangles();
};

ZELOS_NAMESPACE_END

#endif

