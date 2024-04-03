//---------------//
// ZHashGrid3D.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.06.29                               //
//-------------------------------------------------------//

#ifndef _ZHashGrid3D_h_
#define _ZHashGrid3D_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZHashGrid3D
{
	private:

		ZBox3d _domainAABB;				// domain (axis aligned bounding box)
		double _cellSize;				// cell size (the size of one cell)
		int    _nx, _ny, _nz;			// grid resolution
		int    _nynz;					// = _ny*_nz (for efficient computation)
		int    _hashTableSize;			// hash table size
		int    _maxHashTableSize;		// maximum hash table size

		vector<vector<int> > _items;	// per cell index list

	public:

		ZHashGrid3D();
		ZHashGrid3D( const ZBox3f& domainAABB, double cellSize, int hashTableSize=ZPow3(256), int maxHashTableSize=ZPow3(512) );
		ZHashGrid3D( const ZBox3d& domainAABB, double cellSize, int hashTableSize=ZPow3(256), int maxHashTableSize=ZPow3(512) );
		ZHashGrid3D( const ZBoundingBox& domainAABB, double cellSize, int hashTableSize=ZPow3(256), int maxHashTableSize=ZPow3(512) );

		void reset();

		void set( const ZBox3f& domainAABB, double cellSize, int hashTableSize=ZPow3(256), int maxHashTableSize=ZPow3(512) );
		void set( const ZBox3d& domainAABB, double cellSize, int hashTableSize=ZPow3(256), int maxHashTableSize=ZPow3(512) );
		void set( const ZBoundingBox& domainAABB, double cellSize, int hashTableSize=ZPow3(256), int maxHashTableSize=ZPow3(512) );

		// to add a 3D point
		void add( int id, const float& x, const float& y, const float& z );
		void add( int id, const double& x, const double& y, const double& z );
		void add( int id, const ZFloat3& p );
		void add( int id, const ZDouble3& p );
		void add( int id, const ZPoint& p );

		// to add a bounding box
		void add( int id, const ZBox3f& aabb );
		void add( int id, const ZBox3d& aabb );
		void add( int id, const ZBoundingBox& aabb );

		// to add bounding boxes
		void add( const std::vector<ZBox3f >& AABBs );
		void add( const std::vector<ZBox3d >& AABBs );
		void add( const ZBoundingBoxArray& AABBs );

		// 3D index -> 1D array index
		int hashFunc( const int& i, const int& j, const int& k ) const;
		int hashFunc( const ZInt3& ijk ) const;

		ZInt3 cellIndex( const float& x, const float& y, const float& z ) const;
		ZInt3 cellIndex( const double& x, const double& y, const double& z ) const;
		ZInt3 cellIndex( const ZFloat3& p ) const; 
		ZInt3 cellIndex( const ZDouble3& p ) const; 
		ZInt3 cellIndex( const ZPoint& p ) const; 

		double cellSize() const;

		int hashTableSize() const;

		int maxHashTableSize() const;

		const vector<vector<int> >& items() const;

		const vector<int>& items( int i, int j, int k ) const;

		const vector<int>& items( const float& x, const float& y, const float& z ) const;
		const vector<int>& items( const double& x, const double& y, const double& z ) const;
		const vector<int>& items( const ZFloat3& p ) const;
		const vector<int>& items( const ZDouble3& p ) const;
		const vector<int>& items( const ZPoint& p ) const;
};

inline
ZHashGrid3D::ZHashGrid3D()
{
	ZHashGrid3D::reset();
}

inline
ZHashGrid3D::ZHashGrid3D( const ZBox3f& domainAABB, double cellSize, int hashTableSize, int maxHashTableSize )
{
	ZHashGrid3D::set( domainAABB, cellSize, hashTableSize, maxHashTableSize );
}

inline
ZHashGrid3D::ZHashGrid3D( const ZBox3d& domainAABB, double cellSize, int hashTableSize, int maxHashTableSize )
{
	ZHashGrid3D::set( domainAABB, cellSize, hashTableSize, maxHashTableSize );
}

inline
ZHashGrid3D::ZHashGrid3D( const ZBoundingBox& domainAABB, double cellSize, int hashTableSize, int maxHashTableSize )
{
	ZHashGrid3D::set( domainAABB, cellSize, hashTableSize, maxHashTableSize );
}

inline void
ZHashGrid3D::reset()
{
	_domainAABB.reset();

	_cellSize = 0.0;

	_nx = _ny = _nz = 0;
	_nynz = 0;

	_hashTableSize = 0;
	_maxHashTableSize = 0;

	_items.clear();
}

inline void
ZHashGrid3D::set( const ZBox3f& domainAABB, double cellSize, int hashTableSize, int maxHashTableSize )
{
	ZHashGrid3D::reset();

	const double Lx = (double)domainAABB.xWidth();
	const double Ly = (double)domainAABB.xWidth();
	const double Lz = (double)domainAABB.xWidth();

	const ZFloat3& minPt = domainAABB.minPoint();
	const ZFloat3& maxPt = domainAABB.maxPoint();

	_domainAABB.set
	(
		ZDouble3( (double)minPt[0], (double)minPt[1], (double)minPt[2] ),
		ZDouble3( (double)maxPt[0], (double)maxPt[1], (double)maxPt[2] )
	);

	_cellSize = cellSize;

	_nx = (int)( ( Lx / _cellSize ) + 0.5 );
	_ny = (int)( ( Ly / _cellSize ) + 0.5 );
	_nz = (int)( ( Lz / _cellSize ) + 0.5 );

	_nynz = _ny * _nz;

	_maxHashTableSize = ZClamp( maxHashTableSize, ZPow3(32), ZPow3(512) );
	_hashTableSize = ZMin( hashTableSize, _maxHashTableSize );

	_items.clear();
	_items.resize( _hashTableSize );
}

inline void
ZHashGrid3D::set( const ZBox3d& domainAABB, double cellSize, int hashTableSize, int maxHashTableSize )
{
	ZHashGrid3D::reset();

	const double Lx = domainAABB.xWidth();
	const double Ly = domainAABB.xWidth();
	const double Lz = domainAABB.xWidth();

	const ZDouble3& minPt = domainAABB.minPoint();
	const ZDouble3& maxPt = domainAABB.maxPoint();

	_domainAABB.set( minPt, maxPt );

	_cellSize = cellSize;

	_nx = (int)( ( Lx / _cellSize ) + 0.5 );
	_ny = (int)( ( Ly / _cellSize ) + 0.5 );
	_nz = (int)( ( Lz / _cellSize ) + 0.5 );

	_nynz = _ny * _nz;

	_maxHashTableSize = ZClamp( maxHashTableSize, ZPow3(32), ZPow3(512) );
	_hashTableSize = ZMin( hashTableSize, _maxHashTableSize );

	_items.clear();
	_items.resize( _hashTableSize );
}

inline void
ZHashGrid3D::set( const ZBoundingBox& domainAABB, double cellSize, int hashTableSize, int maxHashTableSize )
{
	ZHashGrid3D::reset();

	const double Lx = (double)domainAABB.xWidth();
	const double Ly = (double)domainAABB.xWidth();
	const double Lz = (double)domainAABB.xWidth();

	const ZPoint& minPt = domainAABB.minPoint();
	const ZPoint& maxPt = domainAABB.maxPoint();

	_domainAABB.set
	(
		ZDouble3( (double)minPt.x, (double)minPt.y, (double)minPt.z ),
		ZDouble3( (double)maxPt.x, (double)maxPt.y, (double)maxPt.z )
	);

	_cellSize = cellSize;

	_nx = (int)( ( Lx / _cellSize ) + 0.5 );
	_ny = (int)( ( Ly / _cellSize ) + 0.5 );
	_nz = (int)( ( Lz / _cellSize ) + 0.5 );

	_nynz = _ny * _nz;

	_maxHashTableSize = ZClamp( maxHashTableSize, ZPow3(32), ZPow3(512) );
	_hashTableSize = ZMin( hashTableSize, _maxHashTableSize );

	_items.clear();
	_items.resize( _hashTableSize );
}

inline void
ZHashGrid3D::add( int id, const float& x, const float& y, const float& z )
{
	_items[ hashFunc( cellIndex(x,y,z) ) ].push_back( id );
}

inline void
ZHashGrid3D::add( int id, const double& x, const double& y, const double& z )
{
	_items[ hashFunc( cellIndex(x,y,z) ) ].push_back( id );
}

inline void
ZHashGrid3D::add( int id, const ZFloat3& p )
{
	_items[ hashFunc( cellIndex(p) ) ].push_back( id );
}

inline void
ZHashGrid3D::add( int id, const ZDouble3& p )
{
	_items[ hashFunc( cellIndex(p) ) ].push_back( id );
}

inline void
ZHashGrid3D::add( int id, const ZPoint& p )
{
	_items[ hashFunc( cellIndex(p) ) ].push_back( id );
}

inline void
ZHashGrid3D::add( int id, const ZBox3f& aabb )
{
	const ZInt3 minIJK = cellIndex( aabb.minPoint() );
	const ZInt3 maxIJK = cellIndex( aabb.maxPoint() );

	for( int i=minIJK.data[0]; i<=maxIJK.data[0]; ++i )
	for( int j=minIJK.data[1]; j<=maxIJK.data[1]; ++j )
	for( int k=minIJK.data[2]; k<=maxIJK.data[2]; ++k )
	{{{
		_items[ hashFunc(i,j,k) ].push_back( id );
	}}}
}

inline void
ZHashGrid3D::add( int id, const ZBox3d& aabb )
{
	const ZInt3 minIJK = cellIndex( aabb.minPoint() );
	const ZInt3 maxIJK = cellIndex( aabb.maxPoint() );

	for( int i=minIJK.data[0]; i<=maxIJK.data[0]; ++i )
	for( int j=minIJK.data[1]; j<=maxIJK.data[1]; ++j )
	for( int k=minIJK.data[2]; k<=maxIJK.data[2]; ++k )
	{{{
		_items[ hashFunc(i,j,k) ].push_back( id );
	}}}
}

inline void
ZHashGrid3D::add( int id, const ZBoundingBox& aabb )
{
	const ZInt3 minIJK = cellIndex( aabb.minPoint() );
	const ZInt3 maxIJK = cellIndex( aabb.maxPoint() );

	for( int i=minIJK.data[0]; i<=maxIJK.data[0]; ++i )
	for( int j=minIJK.data[1]; j<=maxIJK.data[1]; ++j )
	for( int k=minIJK.data[2]; k<=maxIJK.data[2]; ++k )
	{{{
		_items[ hashFunc(i,j,k) ].push_back( id );
	}}}
}

inline void
ZHashGrid3D::add( const std::vector<ZBox3f>& AABBs )
{
	const int n = (int)AABBs.size();

	FOR( id, 0, n )
	{
		const ZBox3f& aabb = AABBs[id];

		const ZInt3 minIJK = cellIndex( aabb.minPoint() );
		const ZInt3 maxIJK = cellIndex( aabb.maxPoint() );

		for( int i=minIJK.data[0]; i<=maxIJK.data[0]; ++i )
		for( int j=minIJK.data[1]; j<=maxIJK.data[1]; ++j )
		for( int k=minIJK.data[2]; k<=maxIJK.data[2]; ++k )
		{{{
			_items[ hashFunc(i,j,k) ].push_back( id );
		}}}
	}
}

inline void
ZHashGrid3D::add( const std::vector<ZBox3d >& AABBs )
{
	const int n = (int)AABBs.size();

	FOR( id, 0, n )
	{
		const ZBox3d& aabb = AABBs[id];

		const ZInt3 minIJK = cellIndex( aabb.minPoint() );
		const ZInt3 maxIJK = cellIndex( aabb.maxPoint() );

		for( int i=minIJK.data[0]; i<=maxIJK.data[0]; ++i )
		for( int j=minIJK.data[1]; j<=maxIJK.data[1]; ++j )
		for( int k=minIJK.data[2]; k<=maxIJK.data[2]; ++k )
		{{{
			_items[ hashFunc(i,j,k) ].push_back( id );
		}}}
	}
}

inline void
ZHashGrid3D::add( const ZBoundingBoxArray& AABBs )
{
	const int n = (int)AABBs.size();

	FOR( id, 0, n )
	{
		const ZBoundingBox& aabb = AABBs[id];

		const ZInt3 minIJK = cellIndex( aabb.minPoint() );
		const ZInt3 maxIJK = cellIndex( aabb.maxPoint() );

		for( int i=minIJK.data[0]; i<=maxIJK.data[0]; ++i )
		for( int j=minIJK.data[1]; j<=maxIJK.data[1]; ++j )
		for( int k=minIJK.data[2]; k<=maxIJK.data[2]; ++k )
		{{{
			_items[ hashFunc(i,j,k) ].push_back( id );
		}}}
	}
}

inline int
ZHashGrid3D::hashFunc( const int& i, const int& j, const int& k ) const
{
	return ( ( i*_nynz + j*_nz + k ) % _hashTableSize );
}

inline int
ZHashGrid3D::hashFunc( const ZInt3& ijk ) const
{
	return ( ( ijk.data[0]*_nynz + ijk.data[1]*_nz + ijk.data[2] ) % _hashTableSize );
}

inline ZInt3
ZHashGrid3D::cellIndex( const float& x, const float& y, const float& z ) const
{
	const ZDouble3& minPt = _domainAABB.minPoint();

	return ZInt3
	(
		ZClamp( (int)( ( x - minPt.data[0] ) / _cellSize ), 0, _nx ),
		ZClamp( (int)( ( y - minPt.data[1] ) / _cellSize ), 0, _ny ),
		ZClamp( (int)( ( z - minPt.data[2] ) / _cellSize ), 0, _nz )
	);
}

inline ZInt3
ZHashGrid3D::cellIndex( const double& x, const double& y, const double& z ) const
{
	const ZDouble3& minPt = _domainAABB.minPoint();

	return ZInt3
	(
		ZClamp( (int)( ( x - minPt.data[0] ) / _cellSize ), 0, _nx ),
		ZClamp( (int)( ( y - minPt.data[1] ) / _cellSize ), 0, _ny ),
		ZClamp( (int)( ( z - minPt.data[2] ) / _cellSize ), 0, _nz )
	);
}

inline ZInt3
ZHashGrid3D::cellIndex( const ZFloat3& p ) const
{
	const ZDouble3& minPt = _domainAABB.minPoint();

	return ZInt3
	(
		ZClamp( (int)( ( p.data[0] - minPt.data[0] ) / _cellSize ), 0, _nx ),
		ZClamp( (int)( ( p.data[1] - minPt.data[1] ) / _cellSize ), 0, _ny ),
		ZClamp( (int)( ( p.data[2] - minPt.data[2] ) / _cellSize ), 0, _nz )
	);
}

inline ZInt3
ZHashGrid3D::cellIndex( const ZDouble3& p ) const
{
	const ZDouble3& minPt = _domainAABB.minPoint();

	return ZInt3
	(
		ZClamp( (int)( ( p.data[0] - minPt.data[0] ) / _cellSize ), 0, _nx ),
		ZClamp( (int)( ( p.data[1] - minPt.data[1] ) / _cellSize ), 0, _ny ),
		ZClamp( (int)( ( p.data[2] - minPt.data[2] ) / _cellSize ), 0, _nz )
	);
}

inline ZInt3
ZHashGrid3D::cellIndex( const ZPoint& p ) const
{
	const ZDouble3& minPt = _domainAABB.minPoint();

	return ZInt3
	(
		ZClamp( (int)( ( p.x - minPt.data[0] ) / _cellSize ), 0, _nx ),
		ZClamp( (int)( ( p.y - minPt.data[1] ) / _cellSize ), 0, _ny ),
		ZClamp( (int)( ( p.z - minPt.data[2] ) / _cellSize ), 0, _nz )
	);
}

inline double
ZHashGrid3D::cellSize() const
{
	return _cellSize;
}

inline int
ZHashGrid3D::hashTableSize() const
{
	return _hashTableSize;
}

inline int
ZHashGrid3D::maxHashTableSize() const
{
	return _maxHashTableSize;
}

inline const vector<vector<int> >&
ZHashGrid3D::items() const
{
	return _items;
}

inline const vector<int>&
ZHashGrid3D::items( int i, int j, int k ) const
{
	return _items[ hashFunc(i,j,k) ];
}

inline const vector<int>&
ZHashGrid3D::items( const float& x, const float& y, const float& z ) const
{
	return _items[ hashFunc( cellIndex(x,y,z) ) ];
}

inline const vector<int>&
ZHashGrid3D::items( const double& x, const double& y, const double& z ) const
{
	return _items[ hashFunc( cellIndex(x,y,z) ) ];
}

inline const vector<int>&
ZHashGrid3D::items( const ZFloat3& p ) const
{
	return _items[ hashFunc( cellIndex(p) ) ];
}

inline const vector<int>&
ZHashGrid3D::items( const ZDouble3& p ) const
{
	return _items[ hashFunc( cellIndex(p) ) ];
}

inline const vector<int>&
ZHashGrid3D::items( const ZPoint& p ) const
{
	return _items[ hashFunc( cellIndex(p) ) ];
}

ZELOS_NAMESPACE_END

#endif

