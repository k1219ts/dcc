//------------------//
// ZScalarField3D.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.07.05                               //
//-------------------------------------------------------//

#ifndef _ZScalarField3D_h_
#define _ZScalarField3D_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZScalarField3D : public ZField3DBase, public ZFloatArray
{
	public:

		float minValue, maxValue;

	public:

		ZScalarField3D();
		ZScalarField3D( const ZScalarField3D& source );
		ZScalarField3D( const ZGrid3D& grid, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );
		ZScalarField3D( int Nx, int Ny, int Nz, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );
		ZScalarField3D( int Nx, int Ny, int Nz, float Lx, float Ly, float Lz, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );
		ZScalarField3D( const char* filePathName );

		void set( const ZGrid3D& grid, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );
		void set( int Nx, int Ny, int Nz, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );
		void set( int Nx, int Ny, int Nz, float Lx, float Ly, float Lz, ZFieldLocation::FieldLocation loc=ZFieldLocation::zCell );

		void reset();

		ZScalarField3D& operator=( const ZScalarField3D& other );

		void fill( float valueForAll );
		bool exchange( const ZScalarField3D& other );

		float& operator()( const int& i, const int& j, const int& k );
		const float& operator()( const int& i, const int& j, const int& k ) const;

		float lerp( const ZPoint& p ) const;

		float mcerp( const ZPoint& p ) const;

		ZVector gradient( const ZPoint& p ) const;

		float min( bool useOpenMP=false ) const;
		float max( bool useOpenMP=false ) const;
		void setMinMax( bool useOpenMP=false );
		float absMax( bool useOpenMP=false ) const;

		const ZString dataType() const;

		void drawSlice( const ZInt3& whichSlice, const ZFloat3& sliceRatio,
                        bool smoothPosArea=true, const ZColor& farPos=ZColor::blue(), const ZColor& nearPos=ZColor::yellow(),
                        bool smoothNegArea=true, const ZColor& farNeg=ZColor::red(),  const ZColor& nearNeg=ZColor::yellow(),
                        float elementSize=1.f ) const;

		void drawSliceRangeOnly( const ZInt3& whichSlice, const ZFloat3& sliceRatio,
 		                       	 bool smoothPosArea=true, const ZColor& farPos=ZColor::blue(), const ZColor& nearPos=ZColor::yellow(),
 		                         bool smoothNegArea=true, const ZColor& farNeg=ZColor::red(),  const ZColor& nearNeg=ZColor::yellow(),
 		                         float elementSize=1.f, float dispMin=-Z_LARGE, float dispMax=Z_LARGE ) const;

		bool save( const char* filePathName ) const;
		bool load( const char* filePathName );

	private:

		void _drawXSlice( int i,
                          bool smoothPosArea, const ZColor& farPos, const ZColor& nearPos,
                          bool smoothNegArea, const ZColor& farNeg, const ZColor& nearNeg,
                          float elementSize ) const;

		void _drawYSlice( int j,
                          bool smoothPosArea, const ZColor& farPos, const ZColor& nearPos,
                          bool smoothNegArea, const ZColor& farNeg, const ZColor& nearNeg,
                          float elementSize ) const;

		void _drawZSlice( int k,
                          bool smoothPosArea, const ZColor& farPos, const ZColor& nearPos,
                          bool smoothNegArea, const ZColor& farNeg, const ZColor& nearNeg,
                          float elementSize ) const;

		void _drawXSliceRangeOnly( int i,
                          bool smoothPosArea, const ZColor& farPos, const ZColor& nearPos,
                          bool smoothNegArea, const ZColor& farNeg, const ZColor& nearNeg,
                          float elementSize, float dispMin, float dispMax ) const;

		void _drawYSliceRangeOnly( int j,
                          bool smoothPosArea, const ZColor& farPos, const ZColor& nearPos,
                          bool smoothNegArea, const ZColor& farNeg, const ZColor& nearNeg,
                          float elementSize, float dispMin, float dispMax ) const;

		void _drawZSliceRangeOnly( int k,
                          bool smoothPosArea, const ZColor& farPos, const ZColor& nearPos,
                          bool smoothNegArea, const ZColor& farNeg, const ZColor& nearNeg,
                          float elementSize, float dispMin, float dispMax ) const;
};

inline float&
ZScalarField3D::operator()( const int& i, const int& j, const int& k )
{
	return std::vector<float>::operator[]( i + _stride0*j + _stride1*k );
}

inline const float&
ZScalarField3D::operator()( const int& i, const int& j, const int& k ) const
{
	return std::vector<float>::operator[]( i + _stride0*j + _stride1*k );
}

inline float
ZScalarField3D::lerp( const ZPoint& p ) const
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

	const float* _data = (const float*)ZFloatArray::pointer();
	float est = wgt[0] * _data[ idx[0] ];
	FOR(l,1,8) { est += wgt[l] * _data[ idx[l] ]; }

	return est;
}

inline float
ZScalarField3D::mcerp( const ZPoint& p ) const
{
	float x=p.x, y=p.y, z=p.z;
	if( _location==ZFieldLocation::zCell ) { x-=_dxd2; y-=_dyd2; z-=_dzd2; }
	
	x = std::max(std::min(x, _maxPt.x), _minPt.x );
	y = std::max(std::min(y, _maxPt.y), _minPt.y );
	z = std::max(std::min(z, _maxPt.z), _minPt.z );

	int i=int(x=((x-_minPt.x)*_ddx)); float fx=x-i;
	int j=int(y=((y-_minPt.y)*_ddy)); float fy=y-j;
	int k=int(z=((z-_minPt.z)*_ddz)); float fz=z-k;

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
		std::max(j-1, 0),
		j,
		j+1,
		std::min(j+2, _jMax)
	};
	int ks[4] = 
	{
		std::max(k-1, 0),
		k,
		k+1,
		std::min(k+2, _kMax)
	};

	float kValues[4];
		
	for (int kk = 0; kk < 4; ++kk )
	{
		float jValues[4];

		for( int jj=0; jj<4; ++jj )
		{
			jValues[jj] = ZMCerp( 
							(*this)(is[0], js[jj], ks[kk]),
							(*this)(is[1], js[jj], ks[kk]),
							(*this)(is[2], js[jj], ks[kk]),
							(*this)(is[3], js[jj], ks[kk]),
							fx);
		}
		
		kValues[kk] = ZMCerp( jValues[0], jValues[1], jValues[2], jValues[3], fy );
	}

	return ZMCerp( kValues[0], kValues[1], kValues[2], kValues[3], fz );
}

inline ZVector
ZScalarField3D::gradient( const ZPoint& p ) const
{
	float x=p.x, y=p.y, z=p.z;
	if( _location==ZFieldLocation::zCell ){ x-=_dxd2; y-=_dyd2; z-=_dzd2; }

	int i=int(x=((x-_minPt.x)*_ddx)); float fx=x-i;
	int j=int(y=((y-_minPt.y)*_ddy)); float fy=y-j;
	int k=int(z=((z-_minPt.z)*_ddz)); float fz=z-k;

	if(i<0) {i=0;fx=0;} else if(i>_iMax) {i=_iMax-1;fx=1;}
	if(j<0) {j=0;fy=0;} else if(j>_jMax) {j=_jMax-1;fy=1;}
	if(k<0) {k=0;fz=0;} else if(k>_kMax) {k=_kMax-1;fz=1;}

	int idx[8];
	idx[0]=i+_stride0*j+_stride1*k; idx[1]=idx[0]+1; idx[2]=idx[1]+_stride1; idx[3]=idx[2]-1;
	idx[4]=idx[0]+_stride0;         idx[5]=idx[4]+1; idx[6]=idx[5]+_stride1; idx[7]=idx[6]-1;

	float val[8];
	const float* _data = ZArray<float>::data();
	FOR( l, 0, 8 ) { val[l] = _data[ idx[l] ]; }

	return ZVector( ZLerp( ZLerp( val[1]-val[0], val[2]-val[3], fz ), ZLerp( val[5]-val[4], val[6]-val[7], fz ), fy ),
					ZLerp( ZLerp( val[4]-val[0], val[7]-val[3], fz ), ZLerp( val[5]-val[1], val[6]-val[2], fz ), fx ),
					ZLerp( ZLerp( val[3]-val[0], val[2]-val[1], fx ), ZLerp( val[7]-val[4], val[6]-val[5], fx ), fy ) ).normalize();
}

ostream&
operator<<( ostream& os, const ZScalarField3D& object );

ZELOS_NAMESPACE_END

#endif

