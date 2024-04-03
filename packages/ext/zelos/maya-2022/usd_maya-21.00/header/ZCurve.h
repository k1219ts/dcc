//----------//
// ZCurve.h //
//-------------------------------------------------------//
// author: Taeyong Kim @ Rhythm & Hues                   //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZCurve_h_
#define _ZCurve_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

// cv(i) = position(t=i/(nCVs-1))
class ZCurve : public ZPointArray
{
	public:

		ZCurve();
		ZCurve( const ZCurve& curve );

		ZCurve& operator=( const ZCurve& other );

		void reset();

		int numCVs() const;

		void setNumCVs( int nCVs );

		void addCV( const ZPoint& cv );

		ZPoint& root();
		const ZPoint& root() const;

		ZPoint& tip();
		const ZPoint& tip() const;

		ZPoint  position( float t ) const;
		ZVector tangent ( float t ) const;
		ZVector normal  ( float t ) const;
		ZVector biNormal( float t ) const;
		float   speed   ( float t ) const;

		void getPositionAndTangent( int i, float t, ZPoint& P, ZVector& T ) const;

		float curveLength() const;
		float curveLength( float t0, float t1 ) const;
		float lineLength() const;

	private:

		void _whereIsIt( float& t, int index[4] ) const;

		ZPoint  _zeroDerivative  ( float t, const ZPoint& P0, const ZPoint& P1, const ZPoint& P2, const ZPoint& P3 ) const;
		ZVector _firstDerivative ( float t, const ZPoint& P0, const ZPoint& P1, const ZPoint& P2, const ZPoint& P3 ) const;
		ZVector _secondDerivative( float t, const ZPoint& P0, const ZPoint& P1, const ZPoint& P2, const ZPoint& P3 ) const;
		ZVector _thirdDerivative ( float t, const ZPoint& P0, const ZPoint& P1, const ZPoint& P2, const ZPoint& P3 ) const;
};

inline int
ZCurve::numCVs() const
{
	return (int)std::vector<ZPoint>::size();
}

inline ZPoint&
ZCurve::root()
{
	return std::vector<ZPoint>::operator[](0);
}

inline const
ZPoint& ZCurve::root() const
{
	return std::vector<ZPoint>::operator[](0);
}

inline ZPoint&
ZCurve::tip()
{
	return std::vector<ZPoint>::operator[](std::vector<ZPoint>::size()-1);
}

inline const
ZPoint& ZCurve::tip() const
{
	return std::vector<ZPoint>::operator[](std::vector<ZPoint>::size()-1);
}

inline void
ZCurve::_whereIsIt( float& t, int idx[4] ) const
{
	const int nCVs   = (int)std::vector<ZPoint>::size();
	const int nCVs_1 = nCVs-1;

	t = ZClamp( t, 0.f, 1.f );

	const float k = t * nCVs_1;
	const int start = int(k);

	int& i0 = idx[0] = start-1;
	int& i1 = idx[1] = ( i0 >= nCVs_1 ) ? i0 : (i0+1);
	int& i2 = idx[2] = ( i1 >= nCVs_1 ) ? i1 : (i1+1);
	          idx[3] = ( i2 >= nCVs_1 ) ? i2 : (i2+1);

	if( i0 < 0 ) { i0 = 0; }

	t = k - start;
}

inline ZPoint
ZCurve::_zeroDerivative( float t, const ZPoint& P0, const ZPoint& P1, const ZPoint& P2, const ZPoint& P3 ) const
//	return 0.5f * ( (       2*P1           )
//				  + ( -1*P0     +1*P2      ) * t
//				  + (  2*P0-5*P1+4*P2-1*P3 ) * tt
//				  + ( -1*P0+3*P1-3*P2+1*P3 ) * ttt );
{
	const float tt  = t*t;
	const float ttt = tt*t;
	return ( WeightedSum( P0, P1, P2, P3, -t+2*tt-ttt, 2-5*tt+3*ttt, t+4*tt-3*ttt, -tt+ttt ) *= 0.5f );
}

inline ZVector
ZCurve::_firstDerivative( float t, const ZPoint& P0, const ZPoint& P1, const ZPoint& P2, const ZPoint& P3 ) const
//	return 0.5f * (   ( -1*P0     +1*P2      )
//				  + 2*(  2*P0-5*P1+4*P2-1*P3 )*t
//				  + 3*( -1*P0+3*P1-3*P2+1*P3 )*tt );
{
	const float tt = t*t;
	return ( WeightedSum( P0, P1, P2, P3, -1+4*t-3*tt, -10*t+9*tt, 1+8*t-9*tt, -2*t+3*tt ) *= 0.5f );
}

inline ZVector
ZCurve::_secondDerivative( float t, const ZPoint& P0, const ZPoint& P1, const ZPoint& P2, const ZPoint& P3 ) const
//	return 0.5f * ( 2*(  2*P0-5*P1+4*P2-1*P3 )
//				  + 6*( -1*P0+3*P1-3*P2+1*P3 )*t );
{
	return ( WeightedSum( P0, P1, P2, P3, 4-6*t, -10+18*t, 8-18*t, -2+6*t ) *= 0.5f );
}

inline ZVector
ZCurve::_thirdDerivative( float t, const ZPoint& P0, const ZPoint& P1, const ZPoint& P2, const ZPoint& P3 ) const
//	return 0.5f * ( 6*( -1*P0+3*P1-3*P2+1*P3 ) );
{
	return ( WeightedSum( P0, P1, P2, P3, -6.f, 18.f, -18.f, 6.f ) *= 0.5f );
}

ZELOS_NAMESPACE_END

#endif

