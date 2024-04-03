//------------------//
// ZZSimplexNoise.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios					 //
// last update: 2015.04.16                               //
//-------------------------------------------------------//

#ifndef _ZZSimplexNoise_h_
#define _ZZSimplexNoise_h_

#include <ZelosCudaBase.h>

class ZZSimplexNoise
{
	private:

		unsigned char _perm[512];

		int _seed;

	public:

		ZZVector offset;

		ZZVector sFreq;
		float    tFreq;

		float    scale;
		float    lift;

	public:

		__device__
		ZZSimplexNoise( unsigned int seed=0 )
		{
			_seed = -1;

			sFreq = ZZVector(0.1f,0.1f,0.1f);
			tFreq = 1.0f;

			scale = 1.0f;
			lift  = 0.0f;

			shuffle( seed );
		}

		__device__
		void shuffle( unsigned int seed )
		{
			if( _seed == (int)seed ) { return; }

			FOR( i, 0, 256 )
			{
				_perm[i] = _perm[i+256] = ZClamp( (int)ZZRand(i+seed,256), 0, 255 );
			}

			_seed = (int)seed;
		}

		__device__
		float pureValue
		(
			float x,
			float* dx=NULL
		) const
		{
			int i0 = ZFloor(x);
			int i1 = i0+1;
			float x0 = x-i0;
			float x1 = x0-1.f;

			float gx0, gx1;
			float n0, n1;
			float t20, t40, t21, t41;

			float x20 = x0*x0;
			float t0 = 1.f - x20;
			t20 = t0 * t0;
			t40 = t20 * t20;
			_grad( _perm[i0&0xff], &gx0 );
			n0 = t40 * gx0 * x0;

			float x21 = x1*x1;
			float t1 = 1.f - x21;
			t21 = t1 * t1;
			t41 = t21 * t21;
			_grad(_perm[i1&0xff], &gx1 );
			n1 = t41 * gx1 * x1;

			if( dx )
			{
				*dx = t20 * t0 * gx0 * x20;
				*dx += t21 * t1 * gx1 * x21;
				*dx *= -8.f;
				*dx += t40 * gx0 + t41 * gx1;
				*dx *= 0.25f;
			}

			float noise = 0.25f * ( n0 + n1 );

			return noise;
		}

	private:

		__device__
		void _grad( int hash, float* gx ) const
		{
			const int h = hash&15;
			*gx = 1.0f + (h&7);
			if( h&8 ) { *gx = -(*gx); }
		}
};

#endif

