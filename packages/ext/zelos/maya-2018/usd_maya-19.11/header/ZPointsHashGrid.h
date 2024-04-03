//-------------------//
// ZPointsHashGrid.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZPointsHashGrid_h_
#define _ZPointsHashGrid_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

/// @brief Unbounded spatial partitioning hash grid.
/**
	This grid divides space into a number of regions (grid cells) of equal size.
	Each grid cell maps into a hash table of a fixed set of n buckets.
	The buckets contain the linked lists of objects.
	The grid itself is conceptual and does not use any memory.
*/
class ZPointsHashGrid
{
	private:

		struct Item
		{
			int              id;
			ZPoint           position;
			ZPointsHashGrid::Item* next;

			Item()
			: id(-1), next((ZPointsHashGrid::Item*)NULL)
			{}

			Item( int inId, const ZPoint& inPosition, ZPointsHashGrid::Item* inNext )
			: id(inId), position(inPosition), next(inNext)
			{}
		};

	public:

		int               _numBuckets;			///< The number of buckets.
		float             _h;			///< The grid cell size.
		ZPointsHashGrid::Item** _list;				///< The linked list of each bucket.

		// from "Real-time Collision Detection" p.288
		static const int32_t _h1 = 0x8da6b343;	///< (= -1918454973) The large multiplicative constants
		static const int32_t _h2 = 0xd8163841;	///< (= -669632447) The arbitrarily chosen primes
		static const int32_t _h3 = 0xcb1ab31f;  ///< (= -887442657)

	public:

		ZPointsHashGrid();
		ZPointsHashGrid( int numBuckets, float voxelSize );

		virtual ~ZPointsHashGrid();

		void reset();

		float voxelSize() const;

		int numItems( int i ) const;
		int numTotalItems() const;

		int index( int i, int j, int k ) const; // hash function

		void add( int id, const ZPoint& pos );

		int findPoints( ZIntArray& neighbors, const ZPoint& p, float maxDistance, bool removeRedundancy, bool asAppending ) const;
};

inline int
ZPointsHashGrid::index( int i, int j, int k ) const
{
	int32_t n = _h1*i + _h2*j + _h3*k;
	n %= _numBuckets;
	if( n < 0 ) { n += _numBuckets; }
	return (int)n;
}

inline void
ZPointsHashGrid::add( int id, const ZPoint& p )
{
	const int idx = index( int(p.x/_h), int(p.y/_h), int(p.z/_h) );
	ZPointsHashGrid::Item* newItem = new ZPointsHashGrid::Item( id, p, _list[idx] );
	_list[idx] = newItem;
}

ostream&
operator<<( ostream& os, const ZPointsHashGrid& object );

ZELOS_NAMESPACE_END

#endif

