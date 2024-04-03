//------------------//
// ZVectorField3D.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.07.05                               //
//-------------------------------------------------------//

#ifndef _ZVectorField3D_h_
#define _ZVectorField3D_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZVectorField3D : public ZField3DBase, public ZVectorArray
{
	public:

		ZVectorField3D();
		ZVectorField3D( const ZVectorField3D& source );
		ZVectorField3D( const ZGrid3D& grid, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );
		ZVectorField3D( int Nx, int Ny, int Nz, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );
		ZVectorField3D( int Nx, int Ny, int Nz, float Lx, float Ly, float Lz, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );
		ZVectorField3D( const char* filePathName );

		void set( const ZGrid3D& grid, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );
		void set( int Nx, int Ny, int Nz, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );
		void set( int Nx, int Ny, int Nz, float Lx, float Ly, float Lz, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );

		void reset();

		ZVectorField3D& operator=( const ZVectorField3D& other );

		bool exchange( const ZVectorField3D& other );

		ZVector& operator()( const int& i, const int& j, const int& k );
		const ZVector& operator()( const int& i, const int& j, const int& k ) const;

		ZVector lerp( const ZPoint& p ) const;

		ZVector mcerp( const ZPoint& p ) const;

		const ZString dataType() const;

		bool save( const char* filePathName ) const;
		bool load( const char* filePathName );
};

inline ZVector&
ZVectorField3D::operator()( const int& i, const int& j, const int& k )
{
	return std::vector<ZVector>::operator[]( i + _stride0*j + _stride1*k );
}

inline const ZVector&
ZVectorField3D::operator()( const int& i, const int& j, const int& k ) const
{
	return std::vector<ZVector>::operator[]( i + _stride0*j + _stride1*k );
}

inline ZVector
ZVectorField3D::lerp( const ZPoint& p ) const
{
	float x=p.x, y=p.y, z=p.z;
	if( _location==ZFieldLocation::zCell ){ x-=_dxd2; y-=_dyd2; z-=_dzd2; }

	int i=int(x=((x-_minPt.x)*_ddx)); float fx=x-i;
	int j=int(y=((y-_minPt.y)*_ddy)); float fy=y-j;
	int k=int(z=((z-_minPt.z)*_ddz)); float fz=z-k;

	if(i<0) {i=0;fx=0;} else if(i>=_iMax) {i=_iMax-1;fx=1;}
	if(j<0) {j=0;fy=0;} else if(j>=_jMax) {j=_jMax-1;fy=1;}
	if(k<0) {k=0;fz=0;} else if(k>=_kMax) {k=_kMax-1;fz=1;}

	int idx[8];
	idx[0]=i+_stride0*j+_stride1*k; idx[1]=idx[0]+1; idx[2]=idx[1]+_stride1; idx[3]=idx[2]-1;
	idx[4]=idx[0]+_stride0;         idx[5]=idx[4]+1; idx[6]=idx[5]+_stride1; idx[7]=idx[6]-1;

	const float _fx=1-fx, _fy=1-fy, _fz=1-fz;
	const float wgt[8] = { _fx*_fy*_fz, fx*_fy*_fz, fx*_fy*fz, _fx*_fy*fz, _fx*fy*_fz, fx*fy*_fz, fx*fy*fz, _fx*fy*fz };

	const ZVector* _data = (const ZVector*)ZVectorArray::pointer();

	x = y = z = 0.f;

	FOR( l, 0, 8 )
	{
		const float& w = wgt[l];
		const ZVector& v = _data[ idx[l] ];

		x += w * v.x;
		y += w * v.y;
		z += w * v.z;
	}

	return ZVector(x,y,z);
}

inline ZVector
ZVectorField3D::mcerp( const ZPoint& p ) const
{
	float x=p.x, y=p.y, z=p.z;
	if( _location==ZFieldLocation::zCell ) { x-=_dxd2; y-=_dyd2; z-=_dzd2; }

	x = std::max( std::min(x, _maxPt.x), _minPt.x );
	y = std::max( std::min(y, _maxPt.y), _minPt.y );
	z = std::max( std::min(z, _maxPt.z), _minPt.z );

	int i = int(x=((x-_minPt.x)*_ddx)); float fx=x-i;
	int j = int(y=((y-_minPt.y)*_ddy)); float fy=y-j;
	int k = int(z=((z-_minPt.z)*_ddz)); float fz=z-k;

	if(i<0) {i=0;fx=0;} else if(i>=_iMax) {i=_iMax-1;fx=1;}
	if(j<0) {j=0;fy=0;} else if(j>=_jMax) {j=_jMax-1;fy=1;}
	if(k<0) {k=0;fz=0;} else if(k>=_kMax) {k=_kMax-1;fz=1;}

	int is[4] =
		{
			std::max(i-1, 0),
			i,
			i+1,
			std::min(i+2, _iMax)
		};

	int js[4] = 
		{
			std::max(j-1,0),
			j,
			j+1,
			std::min(j+2, _jMax)
		};

	int ks[4] =
		{
			std::max(k-1,0),
			k,
			k+1,
			std::min(k+2, _kMax)
		};

	float kValues[3][4];

	for( int kk=0; kk<4; ++kk )
	{
		float jValues[3][4];

		for( int jj=0; jj<4; ++jj )
		{
			for( int m=0; m<3; ++m )
			{
				jValues[m][jj] = ZMCerp(
					(*this)(is[0], js[jj], ks[kk])[m],
					(*this)(is[1], js[jj], ks[kk])[m],
					(*this)(is[2], js[jj], ks[kk])[m],
					(*this)(is[3], js[jj], ks[kk])[m],
					fx);
			}
		}

		for( int m=0; m<3; ++m )
		{
			kValues[m][kk] = ZMCerp( jValues[m][0], jValues[m][1],
									 jValues[m][2], jValues[m][3],
									 fy );
		}
	}

	return ZVector( ZMCerp( kValues[0][0], kValues[0][1], kValues[0][2], kValues[0][3], fz),
					ZMCerp( kValues[1][0], kValues[1][1], kValues[1][2], kValues[1][3], fz),
					ZMCerp( kValues[2][0], kValues[2][1], kValues[2][2], kValues[2][3], fz) );
					
}

ostream&
operator<<( ostream& os, const ZVectorField3D& object );

ZELOS_NAMESPACE_END

#endif

