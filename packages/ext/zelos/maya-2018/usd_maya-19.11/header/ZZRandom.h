//------------//
// ZZRandom.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.04.16                               //
//-------------------------------------------------------//

#ifndef _ZZRandom_h_
#define _ZZRandom_h_

#include <ZelosBase.h>
using namespace Zelos;

// 0 ~ 1
__device__
inline float
ZZRand( unsigned int seed )
{
	unsigned int i=(seed^12345391u)*2654435769u;
	i^=(i<<6)^(i>>26);
	i*=2654435769u;
	i+=(i<<5)^(i>>12);
	return (i*Z_UINTMAX_INV);
}

// 0 ~ x
__device__
inline float
ZZRand( unsigned int seed, float x )
{
	unsigned int i=(seed^12345391u)*2654435769u;
	i^=(i<<6)^(i>>26);
	i*=2654435769u;
	i+=(i<<5)^(i>>12);
	return ( x*(i*Z_UINTMAX_INV) );
}

// a ~ b
__device__
inline float
ZZRand( unsigned int seed, float a, float b )
{
	unsigned int i=(seed^12345391u)*2654435769u;
	i^=(i<<6)^(i>>26);
	i*=2654435769u;
	i+=(i<<5)^(i>>12);
	return ((b-a)*(i*Z_UINTMAX_INV)+a);
}

#endif

