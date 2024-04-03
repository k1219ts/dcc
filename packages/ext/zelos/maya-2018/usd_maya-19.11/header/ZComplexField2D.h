//-------------------//
// ZComplexField2D.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.07.05                               //
//-------------------------------------------------------//

#ifndef _ZComplexField2D_h_
#define _ZComplexField2D_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZComplexField2D : public ZField2DBase, public ZComplexArray
{
	public:

		ZComplexField2D();
		ZComplexField2D( const ZComplexField2D& source );
		ZComplexField2D( const ZGrid2D& grid, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );
		ZComplexField2D( int Nx, int Nz, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );
		ZComplexField2D( int Nx, int Nz, float Lx, float Lz, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );
		ZComplexField2D( const char* filePathName );

		void set( const ZGrid2D& grid, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );
		void set( int Nx, int Nz, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );
		void set( int Nx, int Nz, float Lx, float Lz, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );

		void reset();

		ZComplexField2D& operator=( const ZComplexField2D& other );

		bool exchange( const ZComplexField2D& other );

		ZComplex& operator()( const int& i, const int& k );
		const ZComplex& operator()( const int& i, const int& k ) const;

		ZComplex lerp( const ZPoint& p ) const;

		const ZString dataType() const;

		bool save( const char* filePathName ) const;
		bool load( const char* filePathName );
};

inline ZComplex&
ZComplexField2D::operator()( const int& i, const int& k )
{
	return std::vector<ZComplex>::operator[]( i + _stride*k );
}

inline const ZComplex&
ZComplexField2D::operator()( const int& i, const int& k ) const
{
	return std::vector<ZComplex>::operator[]( i + _stride*k );
}

inline ZComplex
ZComplexField2D::lerp( const ZPoint& p ) const
{
	float x=p.x, z=p.z;
	if( _location==ZFieldLocation::zCell ) { x-=_dxd2; z-=_dzd2; }

	int i=int(x=((x-_minPt.x)*_ddx)); float fx=x-i;
	int k=int(z=((z-_minPt.z)*_ddz)); float fz=z-k;

	if(i<0) {i=0;fx=0;} else if(i>_iMax) {i=_iMax-1;fx=1;}
	if(k<0) {k=0;fz=0;} else if(k>_kMax) {k=_kMax-1;fz=1;}

	int idx[4];
	idx[0]=i+_stride*k; idx[1]=idx[0]+1; idx[2]=idx[1]+_stride; idx[3]=idx[2]-1;

	const float _fx=1-fx, _fz=1-fz;
	const float wgt[4] = { _fx*_fz, fx*_fz, fx*fz, _fx*fz };

	const ZComplex* _data = (const ZComplex*)ZComplexArray::pointer();

	x = z = 0.f;

	FOR( l, 0, 4 )
	{
		const float& w = wgt[l];
		const ZComplex& v = _data[ idx[l] ];

		x += w * v.r;
		z += w * v.i;
	}

	return ZComplex(x,z);
}

ostream&
operator<<( ostream& os, const ZComplexField2D& object );

ZELOS_NAMESPACE_END

#endif

