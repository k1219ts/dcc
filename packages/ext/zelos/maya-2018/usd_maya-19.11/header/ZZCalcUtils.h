//---------------//
// ZZCalcUtils.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.04.17                               //
//-------------------------------------------------------//

#ifndef _ZZCalcUtils_h_
#define _ZZCalcUtils_h_

#include <ZelosCudaBase.h>

__device__
inline ZZVector
WeightedSum( const ZZVector& A, const ZZVector& B, float a, float b )
{
	return ZZVector( (a*A.x)+(b*B.x), (a*A.y)+(b*B.y), (a*A.z)+(b*B.z) );
}

__device__
inline ZZVector
WeightedSum( const ZZVector& A, const ZZVector& B, const ZZVector& C, float a, float b, float c )
{
	return ZZVector( (a*A.x)+(b*B.x)+(c*C.x), (a*A.y)+(b*B.y)+(c*C.y), (a*A.z)+(b*B.z)+(c*C.z) );
}

__device__
inline float
Angle( const ZZVector& A, const ZZVector& B )
{
	return atan2f( (A^B).length(), A*B );
}

// for point
__device__
inline ZZPoint
Rotate( const ZZPoint& p, const ZZVector& unitAxis, float angleInRadians, const ZZPoint& pivot )
{
	ZZPoint P( p - pivot );

	const float c = cosf( angleInRadians );
	const float s = sinf( angleInRadians );

	const ZZVector cross( unitAxis ^ P );
	const float dot = unitAxis * P;

	P.x = c*P.x + s*cross.x + (1-c)*dot*unitAxis.x + pivot.x;
	P.y = c*P.y + s*cross.y + (1-c)*dot*unitAxis.y + pivot.y;
	P.z = c*P.z + s*cross.z + (1-c)*dot*unitAxis.z + pivot.z;

	return P;
}

// for vector
__device__
inline ZZVector
Rotate( const ZZVector& v, const ZZVector& unitAxis, float angleInRadians )
{
	ZZVector V( v );

	const float c = cosf( angleInRadians );
	const float s = sinf( angleInRadians );

	const ZZVector cross( unitAxis ^ V );
	const float dot = unitAxis * V;

	V.x = c*V.x + s*cross.x + (1-c)*dot*unitAxis.x;
	V.y = c*V.y + s*cross.y + (1-c)*dot*unitAxis.y;
	V.z = c*V.z + s*cross.z + (1-c)*dot*unitAxis.z;

	return V;
}

__device__
inline ZZVector
CVPosition( int nCVs, const ZZVector* cvs, float t )
{
	int idx[4];
	const int nCVs_1 = nCVs - 1;
	t = ZZClamp( t, 0.f, 1.f );
	const float k = t * nCVs_1;
	const int start = int(k);
	int& i0 = idx[0] = start-1;
	int& i1 = idx[1] = ( i0 >= nCVs_1 ) ? i0 : (i0+1);
	int& i2 = idx[2] = ( i1 >= nCVs_1 ) ? i1 : (i1+1);
	int& i3 = idx[3] = ( i2 >= nCVs_1 ) ? i2 : (i2+1);
	if( i0 < 0 ) { i0 = 0; }
	t = k - start;

	const ZZVector& p0 = cvs[i0];
	const ZZVector& p1 = cvs[i1];
	const ZZVector& p2 = cvs[i2];
	const ZZVector& p3 = cvs[i3];

	const float tt  = t*t;
	const float ttt = tt*t;

	const float a = 0.5f * ( -t+2*tt-ttt );
	const float b = 0.5f * ( 2-5*tt+3*ttt );
	const float c = 0.5f * ( t+4*tt-3*ttt );
	const float d = 0.5f * ( -tt+ttt );

	ZZVector cv;

	cv.x = a*p0.x + b*p1.x + c*p2.x + d*p3.x;
	cv.y = a*p0.y + b*p1.y + c*p2.y + d*p3.y;
	cv.z = a*p0.z + b*p1.z + c*p2.z + d*p3.z;

	return cv;
}

#endif

