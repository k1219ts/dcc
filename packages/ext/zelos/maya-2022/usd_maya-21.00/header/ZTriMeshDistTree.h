//--------------------//
// ZTriMeshDistTree.h //
//-------------------------------------------------------//
// author: Taeyong Kim @ nVidia                          //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2016.02.04                               //
//-------------------------------------------------------//

#ifndef _ZTriMeshDistTree_h_
#define _ZTriMeshDistTree_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

/// @brief Bounding box hierarchy for accelerating point-triangle distance query.
class ZTriMeshDistTree
{
	private:

		/// Distance information used by KDTree structures
		struct Data
		{
			int    id;		///< triangle id.
			float  dist;	///< computed distance.
			ZPoint pt;		///< computed point.
			float  a, b;	///< barycentric coordinates

			Data()
			: id(-1), dist(Z_LARGE), a(0), b(0) {}

			bool valid()   const { return (id>=0); }
			bool invalid() const { return (id< 0); }
		};

		/// Each cell of spatial partitioning scheme
		struct Cell
		{
			int          level;		///< suvdivision level of the cell
			int          parent;	///< index of the parent
			int          left;		///< index of the left child
			int          right;		///< index of the right child
			ZIntArray    ids;		///< element ID list
			ZBoundingBox bBox;		///< bounding box of the cell

			Cell( int subdLevel= -1 )
			: level(subdLevel), parent(-1), left(-1), right(-1) {}

			bool isLeaf() const { return ( (left<0) || (right<0) ); }
		};

	private:

		ZTriMesh*                           _meshPtr;
		int                                 _maxLevel;		///< The max. level of subdivision.
		int                                 _maxElements;	///< The max. number of elements per each leaf node.
		ZBoundingBoxArray                   _bBoxes;		///< The bounding boxes of triangles.
		std::vector<ZTriMeshDistTree::Cell> _cells;			///< The cells.

	public:

		ZTriMeshDistTree( int maxLevel=10, int maxElements=10 );
		ZTriMeshDistTree( const ZTriMesh& mesh, int maxLevel=10, int maxElements=10 );

		void reset();

		bool set( const ZTriMesh& mesh );

		const ZTriMesh& mesh() const;

		// Find the closest point on the mesh and the triangle including it from the given position.
		float getClosestPoint( const ZPoint& pt, ZPoint& closestPt, int& closestTriangle, float maxDist=Z_LARGE ) const;
		float getClosestPoint( const ZPoint& pt, int& closestTriangle, float& a, float& b, float maxDist=Z_LARGE ) const;
		float getClosestPoint( const ZPoint& pt, ZPoint& closestPt, int& closestTriangle, float& a, float& b, float maxDist=Z_LARGE ) const;

		// Find all triangles of which bounding boxes are colliding with the given bounding box.
		void getTriangles( const ZBoundingBox& bBox, ZIntArray& triangles, bool accurate=true ) const;

		int numVertices() const;
		int numTriangles() const;

		int maxLevel() const;
		int numCells() const;
		int numLeafCells() const;
		const ZBoundingBoxArray& boundingBoxes() const;
		float averageNumTriangles() const;

	private:

		bool _initialize();
		bool _subdividable( int cellId ) const;
		void _splitCell( int cellId );
		int _addCell( int level );

		// Recursively find the closest triangle in the cell of the given index.
		void _compute( int cellId, const ZPoint& point, ZTriMeshDistTree::Data& data ) const;

		// Recursively find the cell index which is the deepest cell including the given bounding box.
		int _findDeepestCellIncludingBox( int cellId, const ZBoundingBox& bBox ) const;
};

inline int
ZTriMeshDistTree::_addCell( int level )
{
	_cells.push_back( ZTriMeshDistTree::Cell( level ) );
	return ( (int)_cells.size()-1 );
}

inline bool
ZTriMeshDistTree::_subdividable( int cellId ) const
{
	if( cellId >= (int)_cells.size() ) { return false; }
	if( _cells[cellId].level >= _maxLevel ) { return false; }
	if( _cells[cellId].ids.length() <= _maxElements ) { return false; }
	return true;
}

ZELOS_NAMESPACE_END

#endif

