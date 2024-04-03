//-------------------//
// ZPointsDistTree.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.10.30                               //
//-------------------------------------------------------//

#ifndef _ZPointsDistTree_h_
#define _ZPointsDistTree_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

/// @brief A balanced k-D tree (k=3).
class ZPointsDistTree
{
	private:

		struct QueryData
		{
			int*   result;
			float* dist2;           // 2 = squared
			ZPoint point;           // querying point
			int    maxCount;
			int    foundCount;
			float  maxRadius2;      // 2 = squared

			QueryData( int* _result, float* _dist2, const ZPoint& _point, int _maxCount, float _maxRadius2 )
			: result(_result), point(_point), dist2(_dist2), maxCount(_maxCount), foundCount(0), maxRadius2(_maxRadius2)
			{}
		};

		struct ComparePointsById
		{
			float* points;
			ComparePointsById( float* p ): points(p) {}
			bool operator()( int a, int b ) { return points[a*3] < points[b*3]; }
		};

		ZIntArray    _ids;
		ZPointArray  _points;
		ZBoundingBox _bBox;

	public:

		ZPointsDistTree();
		ZPointsDistTree( const ZPointArray& points );

		void clear();

		const ZPointArray& points() const;
		const ZBoundingBox& boundingBox() const;

		void setPoints( const ZPointArray& points, const ZIntArray* ids=NULL );

		void addPoints( const ZPointArray& points, const ZIntArray* ids=NULL );
		void finalizeAddingPoints();

    	float findNPoints( const ZPoint& p, int nPoints, float maxRadius, ZIntArray& pointIds, ZFloatArray& dist2 ) const;
	    void  findPoints( const ZBoundingBox& box, ZIntArray& pointIds ) const;
		void  findClosestPoint( const ZPoint& p, int& closestPointId, float& closestDist2 ) const;

	private:

		void  _sort();
		void  _sortSubtree( int n, int count, int j );
		void  _computeSubtreeSizes( int size, int& left, int& right ) const;
		float _insertToHeap( int* result, float* dist2, int heap_size, int new_id, float new_dist2 ) const;
		float _buildHeap( int* result, float* dist2, int heap_size ) const;
		void  _findPoints( const ZBoundingBox& box, int n, int size, int j, ZIntArray& result ) const;
		int   _findNPoints( const ZPoint& p, int nPoints, float maxRadius, int* result, float* dist2, float& finalSearchRadius ) const;
		void  _findNPoints( QueryData& query, int n, int size, int j ) const;
};

ostream& operator<<( ostream& os, const ZPointsDistTree& object );

ZELOS_NAMESPACE_END

#endif

