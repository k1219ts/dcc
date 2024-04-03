//-----------//
// ZRandom.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.11.25                               //
//-------------------------------------------------------//

#ifndef _ZRandom_h_
#define _ZRandom_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

/// @brief Pseudo random number generator
/**
	This function generate a pseudo-random number from a uniform distribution with zero mean.
	@param[in] seed The random seed.
	@return A float-precision floating-point value over the interval (0.0, 1.0).
*/
inline float
ZRand( unsigned int seed )
{
	unsigned int i=(seed^12345391u)*2654435769u;
	i^=(i<<6)^(i>>26);
	i*=2654435769u;
	i+=(i<<5)^(i>>12);
	return (float)(i*Z_UINTMAX_INV);
}

// 0 ~ x
inline float
ZRand( unsigned int seed, float x )
{
	unsigned int i=(seed^12345391u)*2654435769u;
	i^=(i<<6)^(i>>26);
	i*=2654435769u;
	i+=(i<<5)^(i>>12);
	return ( x*(float)(i*Z_UINTMAX_INV) );
}

// a ~ b
inline float
ZRand( unsigned int seed, float a, float b )
{
	unsigned int i=(seed^12345391u)*2654435769u;
	i^=(i<<6)^(i>>26);
	i*=2654435769u;
	i+=(i<<5)^(i>>12);
	return ((b-a)*(float)(i*Z_UINTMAX_INV)+a);
}

inline void
ZRandSeed( int seed=time(NULL) )
{
	srand48( seed );
}

// 0 ~ (n-1)
inline int
ZRandInt0( unsigned int n )
{
	return ( std::rand() % n );
}

// 0 ~ n
inline int
ZRandInt1( unsigned int n )
{
	return ( std::rand() % (n+1) );
}

// min ~ max
inline int
ZRandInt2( unsigned int min, unsigned int max )
{
	return ( std::rand() % (max-min) + min );
}

/// @brief Pseudo random number generator, with mean 0.0 and standard deviation 1.0.
/**
	This function generate a pseudo-random number from a Gaussian distribution (also known as a normal distribution) with zero mean and a standard deviation of one.
	@note It is not thread safe because drand48() depends on time.
	@return A double-precision floating-point value over the interval (-1.0, 1.0).
*/
inline double
ZRandGaussian()
{
	double x=0.0, y=0.0, s=0.0;

	do {

		x = 2.0 * drand48() - 1.0; // -1~+1
		y = 2.0 * drand48() - 1.0; // -1~+1

		s = x*x + y*y;

	} while( s >= 1 || s == 0 );

	s = sqrt( ( -2.0 * log(s) ) / s );

	return ( x*s );
}

ZELOS_NAMESPACE_END

#endif

