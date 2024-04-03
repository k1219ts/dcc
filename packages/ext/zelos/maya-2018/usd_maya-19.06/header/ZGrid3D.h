//-----------//
// ZGrid3D.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZGrid3D_h_
#define _ZGrid3D_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

/// @brief 3D axis-aligned uniform grid on xz-plane.
class ZGrid3D
{
	protected:

		int    _nx, _ny, _nz;			// resolution
		float  _lx, _ly, _lz;			// dimension
		float  _dx, _dy, _dz;			// cell size

		ZPoint _minPt, _maxPt;			// two corner points of AABB

		// frequently asked values
		int    _nxp1, _nyp1, _nzp1;		// nx+1, ny+1, nz+1 (# of nodes per each axis)
		int    _nxm1, _nym1, _nzm1;		// nx-1, ny-1, nz-1 (# of cells per each axis)
		int    _nxny;					// nx*ny
		int    _nxp1nyp1;				// (nx+1)*(ny+1)
		float  _dxd2, _dyd2, _dzd2;		// dx/2, dy/2, dz/2
		float  _ddx,  _ddy,  _ddz;		// 1/dx, 1/dy, 1/dz

	public:

		ZGrid3D();
		ZGrid3D( const ZGrid3D& source );
		ZGrid3D( int Nx, int Ny, int Nz, float Lx=1, float Ly=1, float Lz=1 );
		ZGrid3D( int Nx, int Ny, int Nz, const ZBoundingBox& bBox );
		ZGrid3D( int subdivision, const ZBoundingBox& bBox );
		ZGrid3D( float h, int maxSubdivision, const ZBoundingBox& bBox );

		void reset();

		void set( int Nx, int Ny, int Nz, float Lx=1, float Ly=1, float Lz=1 );
		void set( int Nx, int Ny, int Nz, const ZBoundingBox& bBox );
		void set( int subdivision, const ZBoundingBox& bBox );
		void set( float h, int maxSubdivision, const ZBoundingBox& bBox );

		ZGrid3D& operator=( const ZGrid3D& other );

		bool operator==( const ZGrid3D& other );
		bool operator!=( const ZGrid3D& other );

		void drawGrid( bool x0, bool x1, bool y0, bool y1, bool z0, bool z1 ) const;

		void write( ofstream& fout ) const;
		void read( ifstream& fin );

		// grid resolution
		int nx() const;
		int ny() const;
		int nz() const;

		// grid dimension
		float lx() const;
		float ly() const;
		float lz() const;

		// cell size
		float dx() const;
		float dy() const;
		float dz() const;

		ZPoint minPoint() const;
		ZPoint maxPoint() const;

		float diagonalLength() const;
		float cellDiagonalLength() const;
		float avgCellSize() const;

		int numCells() const;
		int numNodes() const;

		int cell( int i, int j, int k ) const;
		int node( int i, int j, int k ) const;

		ZPoint worldToVoxel( ZPoint p ) const;
		ZPoint voxelToWorld( ZPoint p ) const;

		ZPoint cellWorldPos( int i, int j, int k ) const;
		ZPoint cellVoxelPos( int i, int j, int k ) const;

		ZPoint nodeWorldPos( int i, int j, int k ) const;
		ZPoint nodeVoxelPos( int i, int j, int k ) const;

		void getCellIndex( const ZPoint& p, int& i, int& j, int& k ) const;
		void getCellRange( const ZBoundingBox& b, int& i0, int& i1, int& j0, int& j1, int& k0, int& k1 ) const;

		void clampCellIndex( int& i, int& j, int& k ) const;
		void clampNodeIndex( int& i, int& j, int& k ) const;

		bool whichCell( const ZPoint& p, int& i, int& j, int& k, bool doClamp ) const;

		bool inside( const ZPoint& p ) const;
		bool outside( const ZPoint& p ) const;
		bool inside( int i, int j, int k ) const;

		// eight node indices of (i,j,k) cell
		void getNodesOfCell( int i, int j, int k, ZInt8& nodes ) const;

	protected:

		void _calcFreqAskedVariables();

		void _drawXSlice( int i ) const;
		void _drawYSlice( int j ) const;
		void _drawZSlice( int k ) const;
};

inline int   ZGrid3D::nx() const { return _nx; }
inline int   ZGrid3D::ny() const { return _ny; }
inline int   ZGrid3D::nz() const { return _nz; }
inline float ZGrid3D::lx() const { return _lx; }
inline float ZGrid3D::ly() const { return _ly; }
inline float ZGrid3D::lz() const { return _lz; }
inline float ZGrid3D::dx() const { return _dx; }
inline float ZGrid3D::dy() const { return _dy; }
inline float ZGrid3D::dz() const { return _dz; }

inline ZPoint ZGrid3D::minPoint() const { return _minPt; }
inline ZPoint ZGrid3D::maxPoint() const { return _maxPt; }

inline float
ZGrid3D::diagonalLength() const
{
	return sqrtf( ZPow2(_lx) + ZPow2(_ly) + ZPow2(_lz) );
}

inline float
ZGrid3D::cellDiagonalLength() const
{
	return sqrtf( ZPow2(_dx) + ZPow2(_dy) + ZPow2(_dz) );
}

inline float
ZGrid3D::avgCellSize() const
{
	return ( _dx + _dy + _dz ) / 3.f;
}

inline int
ZGrid3D::numCells() const
{
	return ( _nx * _ny * _nz );
}

inline int
ZGrid3D::numNodes() const
{
	return ( _nxp1 * _nyp1 * _nzp1 );
}

inline int
ZGrid3D::cell( int i, int j, int k ) const
{
	return ( i + _nx*j + _nxny*k );
}

inline int
ZGrid3D::node( int i, int j, int k ) const
{
	return ( i + _nxp1*j + _nxp1nyp1*k );
}

inline ZPoint
ZGrid3D::worldToVoxel( ZPoint p ) const
{
	p.x = ( p.x - _minPt.x ) * _ddx;
	p.y = ( p.y - _minPt.y ) * _ddy;
	p.z = ( p.z - _minPt.z ) * _ddz;
	return p;
}

inline ZPoint
ZGrid3D::voxelToWorld( ZPoint p ) const
{
	p.x = ( p.x * _dx ) + _minPt.x;
	p.y = ( p.y * _dy ) + _minPt.y;
	p.z = ( p.z * _dz ) + _minPt.z;
	return p;
}

inline ZPoint
ZGrid3D::cellWorldPos( int i, int j, int k ) const
{
	ZPoint p( _minPt );
	p.x += (i+0.5f)*_dx;
	p.y += (j+0.5f)*_dy;
	p.z += (k+0.5f)*_dz;
	return p;
}

inline ZPoint
ZGrid3D::cellVoxelPos( int i, int j, int k ) const
{
	return ZPoint( (i+0.5f)*_dx, (j+0.5f)*_dy, (k+0.5f)*_dz );
}

inline ZPoint
ZGrid3D::nodeWorldPos( int i, int j, int k ) const
{
	ZPoint p( _minPt );
	p.x += i*_dx;
	p.y += j*_dy;
	p.z += k*_dz;
	return p;
}

inline ZPoint
ZGrid3D::nodeVoxelPos( int i, int j, int k ) const
{
	return ZPoint( i*_dx, j*_dy, k*_dz );
}

inline void
ZGrid3D::getCellIndex( const ZPoint& p, int& i, int& j, int& k ) const
{
	i = int( ( p.x - _minPt.x ) * _ddx );
	j = int( ( p.y - _minPt.y ) * _ddy );
	k = int( ( p.z - _minPt.z ) * _ddz );
}

inline void
ZGrid3D::getCellRange( const ZBoundingBox& b, int& i0, int& i1, int& j0, int& j1, int& k0, int& k1 ) const
{
	ZGrid3D::getCellIndex( b.minPoint(), i0, j0, k0 );
	ZGrid3D::getCellIndex( b.maxPoint(), i1, j1, k1 );
}

inline void
ZGrid3D::clampCellIndex( int& i, int& j, int& k ) const
{
	i = ZClamp( i, 0, _nxm1 );
	j = ZClamp( j, 0, _nym1 );
	k = ZClamp( k, 0, _nzm1 );
}

inline void
ZGrid3D::clampNodeIndex( int& i, int& j, int& k ) const
{
	i = ZClamp( i, 0, _nx );
	j = ZClamp( j, 0, _ny );
	k = ZClamp( k, 0, _nz );
}

inline bool
ZGrid3D::whichCell( const ZPoint& p, int& i, int& j, int& k, bool doClamp ) const
{
	i = int( ( p.x - _minPt.x ) * _ddx );
	j = int( ( p.y - _minPt.y ) * _ddy );
	k = int( ( p.z - _minPt.z ) * _ddz );

	bool contains = true;
	if( i<   0  ) { if(doClamp){i=0;    } contains=false; }
	if( i>_nxm1 ) { if(doClamp){i=_nxm1;} contains=false; }
	if( j<   0  ) { if(doClamp){j=0;    } contains=false; }
	if( j>_nym1 ) { if(doClamp){j=_nym1;} contains=false; }
	if( k<   0  ) { if(doClamp){k=0;    } contains=false; }
	if( k>_nzm1 ) { if(doClamp){k=_nzm1;} contains=false; }

	return contains;
}

inline bool
ZGrid3D::inside( const ZPoint& p ) const
{
	if( p.x <  _minPt.x ) { return false; }
	if( p.x >= _maxPt.x ) { return false; }
	if( p.y <  _minPt.y ) { return false; }
	if( p.y >= _maxPt.y ) { return false; }
	if( p.z <  _minPt.z ) { return false; }
	if( p.z >= _maxPt.z ) { return false; }
	return true;
}

inline bool
ZGrid3D::outside( const ZPoint& p ) const
{
	if( p.x <  _minPt.x ) { return true; }
	if( p.x >= _maxPt.x ) { return true; }
	if( p.y <  _minPt.y ) { return true; }
	if( p.y >= _maxPt.y ) { return true; }
	if( p.z <  _minPt.z ) { return true; }
	if( p.z >= _maxPt.z ) { return true; }
	return false;
}

inline bool 
ZGrid3D::inside( int i, int j, int k) const
{
	if (i < 0) return false;
	if (j < 0) return false;
	if (k < 0) return false;

	if (i >= _nx) return false;
	if (j >= _ny) return false;
	if (k >= _nz) return false;

	return true;
}

inline void
ZGrid3D::getNodesOfCell( int i, int j, int k, ZInt8& nodes ) const
{

	clampCellIndex( i,j,k );
	int stride0 = _nx;
	int stride1 = _nx*_ny;
	const int& n0 = nodes[0] = cell(i,j,k);	 				// base
	const int& n1 = nodes[1] = n0 + 1;						// i1(n0)
	const int& n2 = nodes[2] = n1 + stride1;				// k1(n1)
					nodes[3] = n2 - 1;						// i0(n2)
	const int& n4 = nodes[4] = n0 + stride0;				// j1(n0)
	const int& n5 = nodes[5] = n4 + 1;						// i1(n4)
	const int& n6 = nodes[6] = n5 + stride1;				// k1(n5)
					nodes[7] = n6 - 1;						// i0(n6)

//	const int& n0 = nodes[0] = i+_nxp1*j+_nxp1nyp1*k;	// base (lowest) node
//	const int& n1 = nodes[1] = n0+1;					// i1(n0)
//	const int& n2 = nodes[2] = n1+_nxp1nyp1;			// k1(n1)
//	                nodes[3] = n2-1;					// i0(n2)
//	const int& n4 = nodes[4] = n0+_nxp1;				// j1(n0)
//	const int& n5 = nodes[5] = n4+1;					// i1(n4)
//	const int& n6 = nodes[6] = n5+_nxp1nyp1;			// k1(n5)
//	                nodes[7] = n6-1;					// i0(n6)
}

ostream&
operator<<( ostream& os, const ZGrid3D& object );

ZELOS_NAMESPACE_END

#endif

