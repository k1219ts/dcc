//---------------------//
// ZDenseMatrixUtils.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.06.21                               //
//-------------------------------------------------------//

#ifndef _ZDenseMatrixUtils_h_
#define _ZDenseMatrixUtils_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

template <typename T>
inline ZDenseMatrix<3,3,T>
OrthoProjector( const ZTuple<3,T>& v )
{
	const T& x = v.data[0];
	const T& y = v.data[1];
	const T& z = v.data[2];

	const T _d = (T)1 / ( ZPow2(x) + ZPow2(y) + ZPow2(z) + (T)1e-30 );

	const T xx_d = (x*x) * _d;
	const T yy_d = (y*y) * _d;
	const T zz_d = (z*z) * _d;

	const T xy_d = (x*y) * _d;
	const T yz_d = (y*z) * _d;
	const T zx_d = (z*x) * _d;

	return ZDenseMatrix<3,3,T>
	(
		xx_d, xy_d, zx_d,
		xy_d, yy_d, yz_d,
		zx_d, yz_d, zz_d
	);
}

template <typename T>
inline ZDenseMatrix<3,3,T>
OuterProduct( const ZTuple<3,T>& a, const ZTuple<3,T>& b )
{
	const T& ax = a.data[0];
	const T& ay = a.data[1];
	const T& az = a.data[2];

	const T& bx = b.data[0];
	const T& by = b.data[1];
	const T& bz = b.data[2];

	return ZDenseMatrix<3,3,T>
	(
		ax * bx, ax * by, ax * bz,
		ay * bx, ay * by, ay * bz,
		az * bx, az * by, az * bz
	);
}

ZELOS_NAMESPACE_END

#endif

