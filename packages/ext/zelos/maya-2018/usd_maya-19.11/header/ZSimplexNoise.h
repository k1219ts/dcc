//-----------------//
// ZSimplexNoise.h //
//-------------------------------------------------------//
// author: Jinhyuk Bae @ Dexter Studios                  //
//         Wanho Choi @ Dexter Studios					 //
//         Nayoung Kim @ Dexter Studios                  //
// last update: 2016.03.15                               //
//-------------------------------------------------------//

#ifndef _ZSimplexNoise_h_
#define _ZSimplexNoise_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

/// @brief Simplex noise with true analytic derivative in 1D to 4D.
/**
	This is an implementation of Perlin "simplex noise" over one dimension (x), two dimensions (x,y), three dimensions (x,y,z), and four dimensions (x,y,z,t).
	The analytic derivative is returned, to make it possible to do lots of fun stuff like flow animations, curl noise, analytic antialiasing and such.
*/
class ZSimplexNoise
{
	private:

		// Gradient tables. 
		// These could be programmed the Ken Perlin way with some clever bit-twiddling, but this is more clear, and not really slower.
		static const float _grad2lut[ 8][2];		// Gradient directions for 2D.
		static const float _grad3lut[16][3];		// Gradient directions for 3D.
		static const float _grad4lut[32][4];		// Gradient directions for 4D.

		// A lookup table to traverse the simplex around a given point in 4D.
		static const unsigned char _simplex[64][4];

		// Permutation table. This is just a random jumble of all numbers 0-255,
		// repeated twice to avoid wrapping the index at 255 for each lookup.
		// 1D uniformly distributed PRN(pseudo-random numbers) from 0 to 255 of length 512.
		unsigned char _perm[512];

		// Skewing factors for 2D simplex grid:
		static const float _F2;
		static const float _G2;

		// Skewing factors for 3D simplex grid:
		static const float _F3;
		static const float _G3;

		// The skewing and unskewing factors are hairy again for the 4D case
		static const float _F4;
		static const float _G4;

		// random seed
		int _seed;

	public:

		// for input
		ZVector offset;
		ZVector sFreq;		// spatial frequency
		float   tFreq;		// temporal frequency

		// for output
		float   scale;
		float   lift;

	public:

		ZSimplexNoise( unsigned int seed=0 );
		ZSimplexNoise( const ZSimplexNoise& noise );

		ZSimplexNoise& operator=( const ZSimplexNoise& obj );

		void shuffle( unsigned int seed );

		// If the last argument(dx,dy,dz,dz) is not null, the analytic derivative is also calculated and returned.

		float pureValue
		(
			float x,
			float* dx=NULL
		) const;

		float pureValue
		(
			float x, float y,
			float* dx=NULL, float* dy=NULL
		) const;

		float pureValue
		(
			float x, float y, float z,
			float* dx=NULL, float* dy=NULL, float* dz=NULL
		) const;

		float pureValue
		(
			float x, float y, float z, float w,
			float* dx=NULL, float* dy=NULL, float* dz=NULL, float* dw=NULL
		) const;

		float value
		(
			float x,
			float* dx=NULL
		) const
		{
			x = sFreq.x * x + offset.x;
			return ( scale * pureValue( x, dx ) + lift );
		}

		float value
		(
			float x, float y,
			float* dx=NULL, float* dy=NULL
		) const
		{
			x = sFreq.x * x + offset.x;
			y = sFreq.y * y + offset.y;
			return ( scale * pureValue( x,y, dx,dy ) + lift );
		}

		float value
		(
			float x, float y, float z,
			float* dx=NULL, float* dy=NULL, float* dz=NULL
		) const
		{
			x = sFreq.x * x + offset.x;
			y = sFreq.y * y + offset.y;
			z = sFreq.z * z + offset.z;
			return ( scale * pureValue( x,y,z, dx,dy,dz ) + lift );
		}

		float value
		(
			float x, float y, float z, float w,
			float* dx=NULL, float* dy=NULL, float* dz=NULL, float* dw=NULL
		) const
		{
			x = sFreq.x * x + offset.x;
			y = sFreq.y * y + offset.y;
			z = sFreq.z * z + offset.z;
			w *= tFreq;
			return ( scale * pureValue( x,y,z,w, dx,dy,dz,dz ) + lift );
		}

		ZVector vector( float x, float y, float z, float w ) const
		{
			return ZVector
			(
				value( x,       y,        z,       w ),
				value( y-19.1f, z+33.4f,  x+47.2f, w ),
				value( z+74.2f, x-124.5f, y+99.4f, w )
			);
		}

		void getPerm( unsigned char perm[512] ) const
		{
			memcpy( perm, _perm, sizeof(unsigned char)*512 );
		}

		float turbulence( float x, float y, float z, float t, int numOctaves ) const;

		float fBm( float x, float y, float z, float t, int numOctaves, float amp, float freq, float rough ) const;

		// marble()
		// cloud()
		// fractal()
		// ...

	private:

		// Helper functions to compute gradients in 1D to 4D,
		// and gradients-dot-residualvectors in 2D to 4D.

		void _grad( int hash, float* gx ) const;
		void _grad( int hash, float* gx, float* gy ) const;
		void _grad( int hash, float* gx, float* gy, float* gz ) const;
		void _grad( int hash, float* gx, float* gy, float* gz, float* gw ) const;
};

inline void
ZSimplexNoise::_grad( int hash, float* gx ) const
{
	const int h = hash&15;
	*gx = 1.0f + (h&7);			// Gradient value is one of 1.0, 2.0, ..., 8.0
	if( h&8 ) { *gx = -(*gx); }	// Make half of the gradients negative
}

inline void
ZSimplexNoise::_grad( int hash, float* gx, float* gy ) const
{
	const int h = hash&7;
	*gx = _grad2lut[h][0];
	*gy = _grad2lut[h][1];
}

inline void
ZSimplexNoise::_grad( int hash, float* gx, float* gy, float* gz ) const
{
	const int h = hash&15;
	*gx = _grad3lut[h][0];
	*gy = _grad3lut[h][1];
	*gz = _grad3lut[h][2];
}

inline void
ZSimplexNoise::_grad( int hash, float* gx, float* gy, float* gz, float* gw ) const
{
	const int h = hash&31;
	*gx = _grad4lut[h][0];
	*gy = _grad4lut[h][1];
	*gz = _grad4lut[h][2];
	*gw = _grad4lut[h][3];
}

ostream&
operator<<( ostream& os, const ZSimplexNoise& object );

ZELOS_NAMESPACE_END

#endif

