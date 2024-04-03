//-----------//
// ZGrid2D.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZGrid2D_h_
#define _ZGrid2D_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

/// @brief 2D axis-aligned uniform grid on xz-plane.
class ZGrid2D
{
	protected:

		int    _nx, _nz;				// resolution
		float  _lx, _lz;				// dimension
		float  _dx, _dz;				// cell size

		ZPoint _minPt, _maxPt;			// two corner points of AABB
		float  _y;						// y-value

		// frequently asked values
		int    _nxp1, _nzp1;			// nx+1, nz+1 (# of nodes per each axis)
		int    _nxm1, _nzm1;			// nx-1, nz-1 (# of cells per each axis)
		float  _dxd2, _dzd2;			// dx/2, dz/2
		float  _ddx,  _ddz;				// 1/dx, 1/dz

	public:

		ZGrid2D();
		ZGrid2D( const ZGrid2D& source );
		ZGrid2D( int Nx, int Nz, float Lx=1, float Lz=1 );
		ZGrid2D( int Nx, int Nz, const ZBoundingBox& bBox );
		ZGrid2D( int subdivision, const ZBoundingBox& bBox );
		ZGrid2D( float h, int maxSubdivision, const ZBoundingBox& bBox );

		void reset();

		void set( int Nx, int Nz, float Lx=1, float Lz=1 );
		void set( int Nx, int Nz, const ZBoundingBox& bBox );
		void set( int subdivision, const ZBoundingBox& bBox );
		void set( float h, int maxSubdivision, const ZBoundingBox& bBox );

		float getY() const { return _y; }
		void setY( float y ) { _minPt.y = _maxPt.y = _y = y; }

		ZGrid2D& operator=( const ZGrid2D& other );

		bool operator==( const ZGrid2D& other );
		bool operator!=( const ZGrid2D& other );

		void drawGrid() const;

		void write( ofstream& fout ) const;
		void read( ifstream& fin );

		// grid resolution
		int nx() const;
		int nz() const;

		// grid dimension
		float lx() const;
		float lz() const;

		// cell size
		float dx() const;
		float dz() const;

		ZPoint minPoint() const;
		ZPoint maxPoint() const;

		float diagonalLength() const;
		float cellDiagonalLength() const;
		float avgCellSize() const;

		int numCells() const;
		int numNodes() const;

		int cell( int i, int k ) const;
		int node( int i, int k ) const;

		ZPoint worldToVoxel( ZPoint p ) const;
		ZPoint voxelToWorld( ZPoint p ) const;

		ZPoint cellWorldPos( int i, int k ) const;
		ZPoint cellVoxelPos( int i, int k ) const;

		ZPoint nodeWorldPos( int i, int k ) const;
		ZPoint nodeVoxelPos( int i, int k ) const;

		void getCellIndex( const ZPoint& p, int& i, int& k ) const;
		void getCellRange( const ZBoundingBox& b, int& i0, int& i1, int& k0, int& k1 ) const;

		void clampCellIndex( int& i, int& k ) const;
		void clampNodeIndex( int& i, int& k ) const;

		bool whichCell( const ZPoint& p, int& i, int& k, bool doClamp ) const;

		bool inside( const ZPoint& p ) const;
		bool outside( const ZPoint& p ) const;

		// four node indices of (i,k) cell
		void getNodesOfCell( int i, int k, ZInt4& nodes ) const;

	protected:

		void _calcFreqAskedVariables();
};

inline int   ZGrid2D::nx() const { return _nx; }
inline int   ZGrid2D::nz() const { return _nz; }
inline float ZGrid2D::lx() const { return _lx; }
inline float ZGrid2D::lz() const { return _lz; }
inline float ZGrid2D::dx() const { return _dx; }
inline float ZGrid2D::dz() const { return _dz; }

inline ZPoint ZGrid2D::minPoint() const { return _minPt; }
inline ZPoint ZGrid2D::maxPoint() const { return _maxPt; }

inline float
ZGrid2D::diagonalLength() const
{
	return sqrtf( ZPow2(_lx) + ZPow2(_lz) );
}

inline float
ZGrid2D::cellDiagonalLength() const
{
	return sqrtf( ZPow2(_dx) + ZPow2(_dz) );
}

inline float
ZGrid2D::avgCellSize() const
{
	return ( _dx + _dz ) / 2.f;
}

inline int
ZGrid2D::numCells() const
{
	return ( _nx * _nz );
}

inline int
ZGrid2D::numNodes() const
{
	return ( _nxp1 * _nzp1 );
}

inline int
ZGrid2D::cell( int i, int k ) const
{
	return ( i + _nx*k );
}

inline int
ZGrid2D::node( int i, int k ) const
{
	return ( i + _nxp1*k );
}

inline ZPoint
ZGrid2D::worldToVoxel( ZPoint p ) const
{
	p.x = ( p.x - _minPt.x ) * _ddx;
	p.y = _y;
	p.z = ( p.z - _minPt.z ) * _ddz;
	return p;
}

inline ZPoint
ZGrid2D::voxelToWorld( ZPoint p ) const
{
	p.x = ( p.x * _dx ) + _minPt.x;
	p.y = _y;
	p.z = ( p.z * _dz ) + _minPt.z;
	return p;
}

inline ZPoint
ZGrid2D::cellWorldPos( int i, int k ) const
{
	ZPoint p( _minPt );
	p.x += (i+0.5f)*_dx;
	p.y = _y;
	p.z += (k+0.5f)*_dz;
	return p;
}

inline ZPoint
ZGrid2D::cellVoxelPos( int i, int k ) const
{
	return ZPoint( (i+0.5f)*_dx, _y, (k+0.5f)*_dz );
}

inline ZPoint
ZGrid2D::nodeWorldPos( int i, int k ) const
{
	ZPoint p( _minPt );
	p.x += i*_dx;
	p.y = _y;
	p.z += k*_dz;
	return p;
}

inline ZPoint
ZGrid2D::nodeVoxelPos( int i, int k ) const
{
	return ZPoint( i*_dx, _y, k*_dz );
}

inline void
ZGrid2D::getCellIndex( const ZPoint& p, int& i, int& k ) const 
{
	i = int( ( p.x - _minPt.x ) * _ddx );
	k = int( ( p.z - _minPt.z ) * _ddz );
}

inline void
ZGrid2D::getCellRange( const ZBoundingBox& b, int& i0, int& i1, int& k0, int& k1 ) const
{
	ZGrid2D::getCellIndex( b.minPoint(), i0, k0 );
	ZGrid2D::getCellIndex( b.maxPoint(), i1, k1 );
}

inline void
ZGrid2D::clampCellIndex( int& i, int& k ) const
{
	i = ZClamp( i, 0, _nxm1 );
	k = ZClamp( k, 0, _nzm1 );
}

inline void
ZGrid2D::clampNodeIndex( int& i, int& k ) const
{
	i = ZClamp( i, 0, _nx );
	k = ZClamp( k, 0, _nz );
}

inline bool
ZGrid2D::whichCell( const ZPoint& p, int& i, int& k, bool doClamp ) const
{
	i = int( ( p.x - _minPt.x ) * _ddx );
	k = int( ( p.z - _minPt.z ) * _ddz );

	bool contains = true;
	if( i<   0  ) { if(doClamp){i=0;    } contains=false; }
	if( i>_nxm1 ) { if(doClamp){i=_nxm1;} contains=false; }
	if( k<   0  ) { if(doClamp){k=0;    } contains=false; }
	if( k>_nzm1 ) { if(doClamp){k=_nzm1;} contains=false; }

	return contains;
}

inline bool
ZGrid2D::inside( const ZPoint& p ) const
{
	if( p.x <  _minPt.x ) { return false; }
	if( p.x >= _maxPt.x ) { return false; }
	if( p.z <  _minPt.z ) { return false; }
	if( p.z >= _maxPt.z ) { return false; }
	return true;
}

inline bool
ZGrid2D::outside( const ZPoint& p ) const
{
	if( p.x <  _minPt.x ) { return true; }
	if( p.x >= _maxPt.x ) { return true; }
	if( p.z <  _minPt.z ) { return true; }
	if( p.z >= _maxPt.z ) { return true; }
	return false;
}

inline void
ZGrid2D::getNodesOfCell( int i, int k, ZInt4& nodes ) const
{
	const int& n0 = nodes[0] = i+_nxp1*k;	// base (lowest) node
	const int& n1 = nodes[1] = n0+1;		// i1(n0)
	const int& n2 = nodes[2] = n1+_nxp1;	// k1(n1)
	                nodes[3] = n2-1;		// i0(n2)
}

ostream&
operator<<( ostream& os, const ZGrid2D& object );

ZELOS_NAMESPACE_END

#endif

