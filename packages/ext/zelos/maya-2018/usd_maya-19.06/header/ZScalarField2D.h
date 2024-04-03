//------------------//
// ZScalarField2D.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.07.05                               //
//-------------------------------------------------------//

#ifndef _ZScalarField2D_h_
#define _ZScalarField2D_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZScalarField2D : public ZField2DBase, public ZFloatArray
{
	public:

		float minValue, maxValue;

	public:

		ZScalarField2D();
		ZScalarField2D( const ZScalarField2D& source );
		ZScalarField2D( const ZGrid2D& grid, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );
		ZScalarField2D( int Nx, int Nz, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );
		ZScalarField2D( int Nx, int Nz, float Lx, float Lz, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );
		ZScalarField2D( const char* filePathName );

		void set( const ZGrid2D& grid, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );
		void set( int Nx, int Nz, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );
		void set( int Nx, int Nz, float Lx, float Lz, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );

		void reset();

		ZScalarField2D& operator=( const ZScalarField2D& other );

		void fill( float valueForAll );
		bool exchange( const ZScalarField2D& other );

		float& operator()( const int& i, const int& k );
		const float& operator()( const int& i, const int& k ) const;

		float lerp( const ZPoint& p ) const;
		float catrom( const ZPoint& p ) const;

		float min( bool useOpenMP=false ) const;
		float max( bool useOpenMP=false ) const;
		float absMax( bool useOpenMP=false ) const;
		void setMinMax( bool useOpenMP=false );

		const ZString dataType() const;

		bool save( const char* filePathName ) const;
		bool load( const char* filePathName );

		void setNoise( const ZSimplexNoise& noise, float time, bool useOpenMP=true );

		void drawHeightMesh( ZMeshDisplayMode::MeshDisplayMode mode, const ZColor& lineColor=ZColor(0.5f), const ZColor& surfaceColor=ZColor(0.8f) ) const;

		void draw( bool smoothPosArea=true, const ZColor& farPos=ZColor::blue(), const ZColor& nearPos=ZColor::yellow(),
                   bool smoothNegArea=true, const ZColor& farNeg=ZColor::red(),  const ZColor& nearNeg=ZColor::yellow(),
                   float elementSize=1.f ) const;
};

inline float&
ZScalarField2D::operator()( const int& i, const int& k )
{
	return std::vector<float>::operator[]( i + _stride*k );
}

inline const float&
ZScalarField2D::operator()( const int& i, const int& k ) const
{
	return std::vector<float>::operator[]( i + _stride*k );
}

inline float
ZScalarField2D::lerp( const ZPoint& p ) const
{
	float x=p.x, z=p.z;
	if( _location==ZFieldLocation::zCell ) { x-=_dxd2; z-=_dzd2; }

	int i=int(x=((x-_minPt.x)*_ddx)); float fx=x-i;
	int k=int(z=((z-_minPt.z)*_ddz)); float fz=z-k;

	if(i<0) {i=0;fx=0;} else if(i>=_iMax) {i=_iMax-1;fx=1;}
	if(k<0) {k=0;fz=0;} else if(k>=_kMax) {k=_kMax-1;fz=1;}

	int idx[4];
	idx[0]=i+_stride*k; idx[1]=idx[0]+1; idx[2]=idx[1]+_stride; idx[3]=idx[2]-1;

	const float _fx=1-fx, _fz=1-fz;
	const float wgt[4] = { _fx*_fz, fx*_fz, fx*fz, _fx*fz };


	const float* _data = (const float*)ZFloatArray::pointer();
	float est = wgt[0] * _data[ idx[0] ];
	FOR(l,1,4) { est += wgt[l] * _data[ idx[l] ]; }

	return est;
}

inline float
ZScalarField2D::catrom( const ZPoint& p ) const
{
	float x=p.x, z=p.z;
	if( _location==ZFieldLocation::zCell ) { x-=_dxd2; z-=_dzd2; }
	
	int i1=int(x); i1 = ZClamp( i1, 0, _iMax );
	int k1=int(z); k1 = ZClamp( k1, 0, _kMax );

	int i0 = i1-1; i0 = ZClamp( i0, 0, _iMax );
	int i2 = i1+1; i2 = ZClamp( i2, 0, _iMax );
	int i3 = i2+1; i3 = ZClamp( i3, 0, _iMax );
	
	int k0 = k1-1; k0 = ZClamp( k0, 0, _kMax );
	int k2 = k1+1; k2 = ZClamp( k2, 0, _kMax );
	int k3 = k2+1; k3 = ZClamp( k3, 0, _kMax );

	const int idx[4][4] = { { index(i0,k0), index(i1,k0), index(i2,k0), index(i3,k0) },
							{ index(i0,k1), index(i1,k1), index(i2,k1), index(i3,k1) },
							{ index(i0,k2), index(i1,k2), index(i2,k2), index(i3,k2) },
							{ index(i0,k3), index(i1,k3), index(i2,k3), index(i3,k3) } };

	const float fx = x-i1;
	const float fz = z-k1;
	
	float t   = fx;
	float tt  = t*t;
	float ttt = tt*t;
	const float xw[4] = { 0.5f*(-t+2*tt-ttt), 0.5f*(2-5*tt+3*ttt), 0.5f*(t+4*tt-3*ttt), 0.5f*(-tt+ttt) };
	
	t   = fz;
	tt  = t*t;
	ttt = tt*t;
	const float zw[4] = { 0.5f*(-t+2*tt-ttt), 0.5f*(2-5*tt+3*ttt), 0.5f*(t+4*tt-3*ttt), 0.5f*(-tt+ttt) };
	
	float est=0;
	const float* _data = (const float*)ZFloatArray::pointer();
	FOR(k,0,4) { t=0.f; FOR(i,0,4) { t+=xw[i]*_data[idx[k][i]]; } est+=zw[k]*t; }
		
	return est;
}

ostream&
operator<<( ostream& os, const ZScalarField2D& object );

ZELOS_NAMESPACE_END

#endif

