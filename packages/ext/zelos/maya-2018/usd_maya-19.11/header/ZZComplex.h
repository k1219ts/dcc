//-------------//
// ZZComplex.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.04.16                               //
//-------------------------------------------------------//

#ifndef _ZZComplex_h_
#define _ZZComplex_h_

#include <ZelosCudaBase.h>

class ZZComplex
{
	public:

		float r;	// The real component.
		float i;	// The imaginary component.

	public:

		__device__
		ZZComplex()
		: r(0), i(0) {}

		__device__
		ZZComplex( float a, float b )
		:r(a), i(b)  {}

		__device__
		float squaredLength()
		{
			return (r*r+i*i);
		}

		__device__
		float length()
		{
			return sqrtf(r*r+i*i);
		}

		__device__
		ZZComplex operator*( const ZZComplex& c )
		{
			return ZZComplex( r*c.r-i*c.i, i*c.r+r*c.i );
		}

		__device__
		ZZComplex operator+( const ZZComplex& c )
		{
			return ZZComplex( r+c.r, i+c.i );
		}
};

#endif

