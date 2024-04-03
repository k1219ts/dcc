//------------//
// ZZVector.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.03.08                               //
//-------------------------------------------------------//

#ifndef _ZZVector_h_
#define _ZZVector_h_

#include <ZelosCudaBase.h>

class ZZVector
{
	public:

		float x;
		float y;
		float z;

	public:

		__host__ __device__
		ZZVector()
		: x(0.f), y(0.f), z(0.f)
		{}

		__host__ __device__
		ZZVector( float X, float Y, float Z )
		: x(X), y(Y), z(Z)
		{}

		__device__
		void zeroize()
		{
			x = y = z = 0.f;
		}

		__device__
		ZZVector& operator=( const ZZVector& v )
		{
			x=v.x; y=v.y; z=v.z;
			return (*this);
		}

		__device__
		ZZVector& operator+=( const ZZVector& v )
		{
			x+=v.x; y+=v.y; z+=v.z;
			return (*this);
		}

		__device__
		ZZVector& operator-=( const ZZVector& v )
		{
			x-=v.x; y-=v.y; z-=v.z;
			return (*this);
		}

		__device__
		ZZVector& operator*=( float s )
		{
			x*=s; y*=s; z*=s;
			return (*this);
		}

		__device__
		ZZVector operator+( const ZZVector& v ) const
		{
			return ZZVector( x+v.x, y+v.y, z+v.z );
		}

		__device__
		ZZVector operator-( const ZZVector& v ) const
		{
			return ZZVector( x-v.x, y-v.y, z-v.z );
		}

		__device__
		ZZVector operator-() const
		{
			return ZZVector( -x, -y, -z );
		}

		__device__
		ZZVector& negate()
		{
			x=-x; y=-y, z=-z;
			return (*this);
		}

		__device__
		ZZVector negated() const
		{
			return ZZVector( -x, -y, -z );
		}

		__device__
		ZZVector& reverse()
		{
			x=-x; y=-y, z=-z;
			return (*this);
		}

		__device__
		ZZVector reversed() const
		{
			return ZZVector( -x, -y, -z );
		}

		__device__
		float operator*( const ZZVector& v ) const // inner(dot) product
		{
			return ( x*v.x + y*v.y + z*v.z );
		}

		__device__
		ZZVector operator*( const float v ) const 
		{
			return ZZVector(x*v, y*v, z*v);
		}

		__device__
		ZZVector operator/( const float v ) const
		{
			return ZZVector(x/v, y/v, z/v);
		}

		__device__
		ZZVector operator^( const ZZVector& v ) const // outer(cross) product
		{
			return ZZVector( y*v.z-z*v.y, z*v.x-x*v.z, x*v.y-y*v.x );
		}

		__device__
		float length() const
		{
			return sqrtf( x*x + y*y + z*z );
		}

		__device__
		float squaredLength() const
		{
			return ( x*x + y*y + z*z );
		}

		__device__
		ZZVector& normalize()
		{
			const float d = 1.f / sqrtf( x*x + y*y + z*z );
			x*=d; y*=d; z*=d;
			return (*this);
		}

		__device__
		ZZVector direction() const
		{
			const float d = 1.f / sqrtf( x*x + y*y + z*z );
			return ZZVector( x*d, y*d, z*d );
		}

		__device__
		float distanceTo( const ZZVector& p ) const
		{
			return sqrtf( ZZPow2(x-p.x) + ZZPow2(y-p.y) + ZZPow2(z-p.z) );
		}

		__device__
		float squaredDistanceTo( const ZZVector& p ) const
		{
			return ( ZZPow2(x-p.x) + ZZPow2(y-p.y) + ZZPow2(z-p.z) );
		}
};

typedef ZZVector ZZPoint;

#endif

