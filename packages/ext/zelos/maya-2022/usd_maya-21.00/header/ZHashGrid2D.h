//---------------//
// ZHashGrid2D.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.06.29                               //
//-------------------------------------------------------//

#ifndef _ZHashGrid2D_h_
#define _ZHashGrid2D_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZHashGrid2D
{
	private:

		ZBox2d  _domainAABB;			// domain (axis aligned bounding box)
		double  _cellSize;				// cell size (the size of one cell)
		int     _nx, _ny;				// grid resolution
		int     _hashTableSize;			// hash table size
		int     _maxHashTableSize;		// maximum hash table size

		vector<vector<int> > _items;	// per cell index list

	public:

		ZHashGrid2D();
		ZHashGrid2D( const ZBox2f& domainAABB, double cellSize, int hashTableSize=ZPow2(512), int maxHashTableSize=ZPow2(1024) );
		ZHashGrid2D( const ZBox2d& domainAABB, double cellSize, int hashTableSize=ZPow2(512), int maxHashTableSize=ZPow2(1024) );

		void reset();

		void set( const ZBox2f& domainAABB, double cellSize, int hashTableSize=ZPow2(512), int maxHashTableSize=ZPow2(1024) );
		void set( const ZBox2d& domainAABB, double cellSize, int hashTableSize=ZPow2(512), int maxHashTableSize=ZPow2(1024) );

		// to add a 2D point
		void add( int id, const float& x, const float& y );
		void add( int id, const double& x, const double& y );
		void add( int id, const ZFloat2& p );
		void add( int id, const ZDouble2& p );

		// to add a bounding box
		void add( int id, const ZBox2f& aabb );
		void add( int id, const ZBox2d& aabb );

		// to add bounding boxes
		void add( const std::vector<ZBox2f >& AABBs );
		void add( const std::vector<ZBox2d >& AABBs );

		// 2D index -> 1D array index
		int hashFunc( const int& i, const int& j ) const;
		int hashFunc( const ZInt2& ij ) const;

		ZInt2 cellIndex( const float& x, const float& y ) const;
		ZInt2 cellIndex( const double& x, const double& y ) const;
		ZInt2 cellIndex( const ZFloat2& p ) const;
		ZInt2 cellIndex( const ZDouble2& p ) const;

		double cellSize() const;

		int hashTableSize() const;

		int maxHashTableSize() const;

		const vector<vector<int> >& items() const;

		const vector<int>& items( int i, int j ) const;

		const vector<int>& items( const float& x, const float& y ) const;
		const vector<int>& items( const double& x, const double& y ) const;
		const vector<int>& items( const ZFloat2& p ) const;
		const vector<int>& items( const ZDouble2& p ) const;
};

inline
ZHashGrid2D::ZHashGrid2D()
{
	ZHashGrid2D::reset();
}

inline
ZHashGrid2D::ZHashGrid2D( const ZBox2f& domainAABB, double cellSize, int hashTableSize, int maxHashTableSize )
{
	ZHashGrid2D::set( domainAABB, cellSize, hashTableSize, maxHashTableSize );
}

inline
ZHashGrid2D::ZHashGrid2D( const ZBox2d& domainAABB, double cellSize, int hashTableSize, int maxHashTableSize )
{
	ZHashGrid2D::set( domainAABB, cellSize, hashTableSize, maxHashTableSize );
}

inline void
ZHashGrid2D::reset()
{
	_domainAABB.reset();

	_cellSize = 0.0;

	_nx = _ny = 0;

	_hashTableSize = 0;
	_maxHashTableSize = 0;

	_items.clear();
}

inline void
ZHashGrid2D::set( const ZBox2f& domainAABB, double cellSize, int hashTableSize, int maxHashTableSize )
{
	ZHashGrid2D::reset();

	const double Lx = (double)domainAABB.xWidth();
	const double Ly = (double)domainAABB.xWidth();

	const ZFloat2& minPt = domainAABB.minPoint();
	const ZFloat2& maxPt = domainAABB.maxPoint();

	_domainAABB.set
	(
		ZDouble2( (double)minPt[0], (double)minPt[1] ),
		ZDouble2( (double)maxPt[0], (double)maxPt[1] )
	);

	_cellSize = cellSize;

	_nx = (int)( ( Lx / _cellSize ) + 0.5 );
	_ny = (int)( ( Ly / _cellSize ) + 0.5 );

	_maxHashTableSize = ZClamp( maxHashTableSize, ZPow2(32), ZPow2(1024) );
	_hashTableSize = ZMin( hashTableSize, _maxHashTableSize );

	_items.clear();
	_items.resize( _hashTableSize );
}

inline void
ZHashGrid2D::set( const ZBox2d& domainAABB, double cellSize, int hashTableSize, int maxHashTableSize )
{
	ZHashGrid2D::reset();

	const double Lx = domainAABB.xWidth();
	const double Ly = domainAABB.xWidth();

	const ZDouble2& minPt = domainAABB.minPoint();
	const ZDouble2& maxPt = domainAABB.maxPoint();

	_domainAABB.set( minPt, maxPt );

	_cellSize = cellSize;

	_nx = (int)( ( Lx / _cellSize ) + 0.5 );
	_ny = (int)( ( Ly / _cellSize ) + 0.5 );

	_maxHashTableSize = ZClamp( maxHashTableSize, ZPow2(32), ZPow2(1024) );
	_hashTableSize = ZMin( hashTableSize, _maxHashTableSize );

	_items.clear();
	_items.resize( _hashTableSize );
}

inline void
ZHashGrid2D::add( int id, const float& x, const float& y )
{
	_items[ hashFunc( cellIndex(x,y) ) ].push_back( id );
}

inline void
ZHashGrid2D::add( int id, const double& x, const double& y )
{
	_items[ hashFunc( cellIndex(x,y) ) ].push_back( id );
}

inline void
ZHashGrid2D::add( int id, const ZFloat2& p )
{
	_items[ hashFunc( cellIndex(p) ) ].push_back( id );
}

inline void
ZHashGrid2D::add( int id, const ZDouble2& p )
{
	_items[ hashFunc( cellIndex(p) ) ].push_back( id );
}

inline void
ZHashGrid2D::add( int id, const ZBox2f& aabb )
{
	const ZFloat2& minPt = aabb.minPoint();
	const ZFloat2& maxPt = aabb.maxPoint();

	const ZInt2 minIJK = cellIndex( (double)minPt.data[0], (double)minPt.data[1] );
	const ZInt2 maxIJK = cellIndex( (double)maxPt.data[0], (double)maxPt.data[1] );

	for( int i=minIJK.data[0]; i<=maxIJK.data[0]; ++i )
	for( int j=minIJK.data[1]; j<=maxIJK.data[1]; ++j )
	{{
		_items[ hashFunc(i,j) ].push_back( id );
	}}
}

inline void
ZHashGrid2D::add( int id, const ZBox2d& aabb )
{
	const ZDouble2& minPt = aabb.minPoint();
	const ZDouble2& maxPt = aabb.maxPoint();

	const ZInt2 minIJK = cellIndex( minPt.data[0], minPt.data[1] );
	const ZInt2 maxIJK = cellIndex( maxPt.data[0], maxPt.data[1] );

	for( int i=minIJK.data[0]; i<=maxIJK.data[0]; ++i )
	for( int j=minIJK.data[1]; j<=maxIJK.data[1]; ++j )
	{{
		_items[ hashFunc(i,j) ].push_back( id );
	}}
}

inline void
ZHashGrid2D::add( const std::vector<ZBox2f>& AABBs )
{
	const int n = (int)AABBs.size();

	FOR( id, 0, n )
	{
		const ZBox2f& aabb = AABBs[id];

		const ZFloat2& minPt = aabb.minPoint();
		const ZFloat2& maxPt = aabb.maxPoint();

		const ZInt2 minIJK = cellIndex( (double)minPt.data[0], (double)minPt.data[1] );
		const ZInt2 maxIJK = cellIndex( (double)maxPt.data[0], (double)maxPt.data[1] );

		for( int i=minIJK.data[0]; i<=maxIJK.data[0]; ++i )
		for( int j=minIJK.data[1]; j<=maxIJK.data[1]; ++j )
		{{
			_items[ hashFunc(i,j) ].push_back( id );
		}}
	}
}

inline void
ZHashGrid2D::add( const std::vector<ZBox2d >& AABBs )
{
	const int n = (int)AABBs.size();

	FOR( id, 0, n )
	{
		const ZBox2d& aabb = AABBs[id];

		const ZDouble2& minPt = aabb.minPoint();
		const ZDouble2& maxPt = aabb.maxPoint();

		const ZInt2 minIJK = cellIndex( minPt.data[0], minPt.data[1] );
		const ZInt2 maxIJK = cellIndex( maxPt.data[0], maxPt.data[1] );

		for( int i=minIJK.data[0]; i<=maxIJK.data[0]; ++i )
		for( int j=minIJK.data[1]; j<=maxIJK.data[1]; ++j )
		{{
			_items[ hashFunc(i,j) ].push_back( id );
		}}
	}
}

inline int
ZHashGrid2D::hashFunc( const int& i, const int& j ) const
{
	return ( ( i*_ny + j ) % _hashTableSize );
}

inline int
ZHashGrid2D::hashFunc( const ZInt2& ij ) const
{
	return ( ( ij.data[0]*_ny + ij.data[1] ) % _hashTableSize );
}

inline ZInt2
ZHashGrid2D::cellIndex( const float& x, const float& y ) const
{
	const ZDouble2& minPt = _domainAABB.minPoint();

	return ZInt2
	(
		ZClamp( (int)( ( x - minPt.data[0] ) / _cellSize ), 0, _nx ),
		ZClamp( (int)( ( y - minPt.data[1] ) / _cellSize ), 0, _ny )
	);
}

inline ZInt2
ZHashGrid2D::cellIndex( const double& x, const double& y ) const
{
	const ZDouble2& minPt = _domainAABB.minPoint();

	return ZInt2
	(
		ZClamp( (int)( ( x - minPt.data[0] ) / _cellSize ), 0, _nx ),
		ZClamp( (int)( ( y - minPt.data[1] ) / _cellSize ), 0, _ny )
	);
}

inline ZInt2
ZHashGrid2D::cellIndex( const ZFloat2& p ) const 
{
	const ZDouble2& minPt = _domainAABB.minPoint();

	return ZInt2
	(
		ZClamp( (int)( ( p.data[0] - minPt.data[0] ) / _cellSize ), 0, _nx ),
		ZClamp( (int)( ( p.data[1] - minPt.data[1] ) / _cellSize ), 0, _ny )
	);
}

inline ZInt2
ZHashGrid2D::cellIndex( const ZDouble2& p ) const 
{
	const ZDouble2& minPt = _domainAABB.minPoint();

	return ZInt2
	(
		ZClamp( (int)( ( p.data[0] - minPt.data[0] ) / _cellSize ), 0, _nx ),
		ZClamp( (int)( ( p.data[1] - minPt.data[1] ) / _cellSize ), 0, _ny )
	);
}

inline double
ZHashGrid2D::cellSize() const
{
	return _cellSize;
}

inline int
ZHashGrid2D::hashTableSize() const
{
	return _hashTableSize;
}

inline int
ZHashGrid2D::maxHashTableSize() const
{
	return _maxHashTableSize;
}

inline const vector<vector<int> >&
ZHashGrid2D::items() const
{
	return _items;
}

inline const vector<int>&
ZHashGrid2D::items( int i, int j ) const
{
	return _items[ hashFunc(i,j) ];
}

inline const vector<int>&
ZHashGrid2D::items( const float& x, const float& y ) const
{
	return _items[ hashFunc( cellIndex(x,y) ) ];
}

inline const vector<int>&
ZHashGrid2D::items( const double& x, const double& y ) const
{
	return _items[ hashFunc( cellIndex(x,y) ) ];
}

inline const vector<int>&
ZHashGrid2D::items( const ZFloat2& p ) const
{
	return _items[ hashFunc( cellIndex(p) ) ];
}

inline const vector<int>&
ZHashGrid2D::items( const ZDouble2& p ) const
{
	return _items[ hashFunc( cellIndex(p) ) ];
}

ZELOS_NAMESPACE_END

#endif

