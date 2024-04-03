//---------------//
// ZTupleUtils.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.06.16                               //
//-------------------------------------------------------//

#ifndef _ZTupleUtils_h_
#define _ZTupleUtils_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

template <typename T>
inline T
DOT( const ZTuple<3,T>& a, const ZTuple<3,T>& b )
{
	return ( a.data[0]*b.data[0] + a.data[1]*b.data[1] + a.data[2]*b.data[2] );
}

template <typename T>
inline ZTuple<3,T>
CRS( const ZTuple<3,T>& a, const ZTuple<3,T>& b )
{
	return ZTuple<3,T>( a.data[1]*b.data[2]-a.data[2]*b.data[1], a.data[2]*b.data[0]-a.data[0]*b.data[2], a.data[0]*b.data[1]-a.data[1]*b.data[0] );
}

template <typename T>
inline T
CRS( const ZTuple<2,T>& a, const ZTuple<2,T>& b )
{
	return ( a.data[0] * b.data[1] - b.data[0] * a.data[1] );
}

template <int N, typename T>
inline void
ADD( std::vector<ZTuple<N,T> >& a, const std::vector<ZTuple<N,T> >& b, const std::vector<ZTuple<N,T> >& c, bool useOpenMP=true )
{
	#pragma omp parallel for
	FOR( i, 0, N )
	{
		a[i] = b[i] + c[i];
	}
}

template <int N, typename T>
inline void
SUB( std::vector<ZTuple<N,T> >& a, const std::vector<ZTuple<N,T> >& b, const std::vector<ZTuple<N,T> >& c, bool useOpenMP=true )
{
	#pragma omp parallel for
	FOR( i, 0, N )
	{
		a[i] = b[i] - c[i];
	}
}

template <int N, typename T>
inline void
MUL( std::vector<ZTuple<N,T> >& a, const std::vector<ZTuple<N,T> >& b, const std::vector<ZTuple<N,T> >& c, bool useOpenMP=true )
{
	#pragma omp parallel for
	FOR( i, 0, N )
	{
		a[i] = b[i] * c[i];
	}
}

template <int N, typename T>
inline void
MUL( std::vector<ZTuple<N,T> >& a, T b, const std::vector<ZTuple<N,T> >& c, bool useOpenMP=true )
{
	#pragma omp parallel for
	FOR( i, 0, N )
	{
		a[i] = b * c[i];
	}
}

template <int N, typename T>
inline void
INC( std::vector<ZTuple<N,T> >& a, const std::vector<ZTuple<N,T> >& b, bool useOpenMP=true )
{
	#pragma omp parallel for
	FOR( i, 0, N )
	{
		a[i] += b[i];
	}
}

template <int N, typename T>
inline void
INCMUL( std::vector<ZTuple<N,T> >& a, T s, const std::vector<ZTuple<N,T> >& b, bool useOpenMP=true )
{
	const int M = (int)a.size();

	#pragma omp parallel for
	FOR( i, 0, M )
	{
		a[i] += s * b[i];
	}
}

template <int N, typename T>
inline double
DOT( const std::vector<ZTuple<N,T> >& a, const std::vector<ZTuple<N,T> >& b, bool useOpenMP=true )
{
	const int M = (int)a.size();

	double s = 0.0;

	#pragma omp parallel for reduction( +: s ) if( useOpenMP )
	FOR( i, 0, M )
	{
		s += a[i] * b[i];
	}

	return s;
}

ZELOS_NAMESPACE_END

#endif

