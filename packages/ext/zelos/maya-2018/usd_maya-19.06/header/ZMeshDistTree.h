//-----------------//
// ZMeshDistTree.h //
//-------------------------------------------------------//
// author: Taeyong Kim @ nVidia                          //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZMeshDistTree_h_
#define _ZMeshDistTree_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

/// @brief The bounding box hierarchy class for accelerating point-triangle distance query.
class ZMeshDistTree
{
	protected:

		/// Distance information used by KDTree structures
		struct Data
		{
			int    id;		///< The triangle id.
			float  dist;	///< The computed distance.
			ZPoint pt;		///< The computed point.

			Data()
			: id(-1), dist(Z_LARGE) {}

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

	protected:

		int                              _maxLevel;		///< The maximum level of subdivision.
		int                              _maxElements;	///< The maximum number of elements per each leaf node.
		ZPointArray                      _points;		///< The triangle vertex positions.
		ZInt3Array                       _triangles;	///< The triangle vertices.
		std::vector<ZMeshDistTree::Cell> _cells;		///< The cells.
		ZBoundingBoxArray                _bBoxes;		///< The bounding boxes of triangles.

	public:

		/**
			Default constructor.
			Create a new empty tree.
			@param[in] maxLevel The maximum subdivision level.
			@param[in] maxElements The maximum number of elements in leaf cell.
			@note This structure should be initialized using initialize() function.
		*/
		ZMeshDistTree( int maxLevel=10, int maxElements=10 );

		/**
			Class constructor.
			Create a new tree and initialize it from the given mesh.
			@param[in] mesh The given mesh to initialize from it.
			@param[in] maxLevel The maximum subdivision level.
			@param[in] maxElements The maximum number of elements in leaf cell.
		*/
		ZMeshDistTree( const ZMesh& mesh, int maxLevel=10, int maxElements=10 );

		/**
			Reset the current distance tree.
		*/
		void reset();

		/**
			Reinitialize the tree from the given mesh.
			Set points and triangles from the given shape.
			@param[in] mesh The input shape to extract points and triangles.
		*/
		bool set( const ZMesh& mesh );

		/**
			Return the maximum subdivision level.
			@return The maximum subdivision level.
		*/
		int maxLevel() const;

		/**
			Return the total number of cells.
			@return The total number of cells.
		*/
		int numCells() const;

		/**
			Return the total number of leaf cells.
			@return The total number of leaf cells.
		*/
		int numLeafCells() const;

		/**
			Return the reference to the mesh vertex positions.
			@return The reference to the mesh vertex positions.
		*/
		const ZPointArray& points() const;

		/**
			Return the reference to the mesh triangle indices.
			@return The reference to the mesh triangle indices
		*/
		const ZInt3Array& triangles() const;

		/**
			Return the reference to the triangle bounding boxes.
			@return The reference to the triangle bounding boxes.
		*/
		const ZBoundingBoxArray& boundingBoxes() const;

		/**
			Return the average number of triangles in a cell.
			@return The average number of triangles in a cell.
		*/
		float averageNumTriangles() const;

		/**
			Find the closest point on the mesh and the triangle including it from the given position.
			@param[in] point The point being queried.
			@param[out] closestPoint The closest point on the mesh from the given position.
			@param[out] closestTriangle The triangle index which includes the closest point.
			@param[in] maxDist The maximum range of computation.
			@return The closest distance from the given position.
		*/
		float getClosestPoint( const ZPoint& point, ZPoint& closestPoint, int& closestTriangle, float maxDist=Z_LARGE ) const;

		/**
			Find all triangles of which bounding boxes are colliding with the given bounding box.
			@param[in] bBox The given bounding box being queried.
			@param[out] triangles The resulting triangle index list.
			@param[in] accurate If it is on, this routine returns all the triangle which are exactrly colliding with the given bounding box.
		*/
		void getTriangles( const ZBoundingBox& bBox, ZIntArray& triangles, bool accurate=true ) const;

	protected:

		/// @brief The variables initializer.
		/** 
			This function initializes all the internal data, including suvdivision.
			@return True if success and false otherwise.
		*/
		bool _initialize();

		/**
			Check if we can subdivide the cell of the given index.
			@param cellId The cell index.
			@return True if the cell can be further subdivided and false otherwise.
		*/
		bool _subdividable( int cellId ) const;

		/** 
			Split the cell of the given index by k-D subdivision.
			@param cellId The cell id being splited into two children cells.
		*/
		void _splitCell( int cellId );

		/**
			Add a new cell with a given level.
			@param[in] level The level of the cell being added.
			@return The index of the cell being added.
		*/
		int _addCell( int level );

		/**
			Recursively find the closest triangle in the cell of the given index.
			@param[in] cellId The cell index being queried.
			@param[in] point The point to compute distance from it.
			@param[out] distData The distance data coming in and gets updated in this function.
		*/
		void _compute( int cellId, const ZPoint& point, ZMeshDistTree::Data& distData ) const;

		/**
			Recursively find the cell index which is the deepest cell including the given bounding box.
			@param[in] cellId The cell index being queried.
			@param[in] bBox The bounding box being queried.
			@return The deepest cell index including the given bounding box.
		*/
		int _findDeepestCellIncludingBox( int cellId, const ZBoundingBox& bBox ) const;
};

ZELOS_NAMESPACE_END

#endif

