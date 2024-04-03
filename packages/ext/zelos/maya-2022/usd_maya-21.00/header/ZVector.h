//-----------//
// ZVector.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.03.08                               //
//-------------------------------------------------------//

#ifndef _ZVector_h_
#define _ZVector_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

/// @brief 3-dimensional vector (or point).
class ZVector
{
	public:

		float x;	///< the x-component of the vector
		float y;	///< the y-component of the vector
		float z;	///< the z-component of the vector

	public:

		ZVector();
		ZVector( const ZVector& v );
		ZVector( const ZFloat3& v );
		ZVector( const float& X, const float& Y, const float& Z=0.f );
		ZVector( const float* v );
		ZVector( const float& s );

		ZVector& set( const float* v );
		ZVector& set( const float& s );
		ZVector& set( const float& X, const float& Y, const float& Z=0.f );

		void get( float* v ) const;
		void get( float& X, float& Y, float& Z ) const;

		ZVector& fill( const float& s );

		void zeroize();
		ZVector& zeroizeExcept( const int& i );

		float& operator[]( const int& i );
		float& operator()( const int& i );

		const float& operator[]( const int& i ) const;
		const float& operator()( const int& i ) const;

		ZVector& operator=( const ZVector& v );
		ZVector& operator=( const ZFloat3& v );
		ZVector& operator=( float* v );
		ZVector& operator=( const float& s );

		bool operator==( const ZVector& v ) const;
		bool operator!=( const ZVector& v ) const;

		bool operator<( const ZVector& v ) const;
		bool operator>( const ZVector& v ) const;

		bool operator<=( const ZVector& v ) const;
		bool operator>=( const ZVector& v ) const;

		ZVector& operator+=( const int& s );
		ZVector& operator+=( const float& s );
		ZVector& operator+=( const double& s );

		ZVector& operator-=( const int& s );
		ZVector& operator-=( const float& s );
		ZVector& operator-=( const double& s );

		ZVector& operator+=( const ZVector& v );
		ZVector& operator-=( const ZVector& v );

		ZVector operator+( const ZVector& v ) const;
		ZVector operator-( const ZVector& v ) const;

		ZVector& operator*=( const int& s );
		ZVector& operator*=( const float& s );
		ZVector& operator*=( const double& s );

		ZVector& operator/=( const int& s );
		ZVector& operator/=( const float& s );
		ZVector& operator/=( const double& s );

		ZVector operator*( const int& s ) const;
		ZVector operator*( const float& s ) const;
		ZVector operator*( const double& s ) const;

		ZVector operator/( const int& s ) const;
		ZVector operator/( const float& s ) const;
		ZVector operator/( const double& s ) const;

		ZVector operator-() const;

		ZVector& negate();
		ZVector negated() const;

		ZVector& reverse();
		ZVector reversed() const;

		ZVector& abs();
		ZVector& clamp( const float& minValue, const float& maxValue );
		ZVector& cycle( bool toLeft=true );

		// dot(inner) product
		float operator*( const ZVector& v ) const;
		float dot( const ZVector& v ) const;

		// cross(outer) product
		ZVector operator^( const ZVector& v ) const;
		ZVector cross( const ZVector& v ) const;

		float length() const;
		float squaredLength() const;

		ZVector& normalize( bool accurate=false );
		ZVector normalized( bool accurate=false ) const;

		ZVector& robustNormalize();
		ZVector robustNormalized() const;

		ZVector direction( bool accurate=false ) const;
		ZVector robustDirection() const;
		ZVector orthogonalDirection( bool accurate=false ) const;

		ZVector& limitLength( const float& targetLength );

		float distanceTo( const ZVector& p ) const;
		float squaredDistanceTo( const ZVector& p ) const;

		bool isEquivalent( const ZVector& v, float epsilon=Z_EPS ) const;
		bool isParallel( const ZVector& v, float epsilon=Z_EPS ) const;

		float min() const;
		float max() const;

		float absMin() const;
		float absMax() const;

		int minIndex() const;
		int maxIndex() const;

		int absMinIndex() const;
		int absMaxIndex() const;

		float sum() const;

		ZVector& setComponentwiseMin( const ZVector& v );
		ZVector& setComponentwiseMax( const ZVector& v );

		ZVector componentwiseMin( const ZVector& v ) const;
		ZVector componentwiseMax( const ZVector& v ) const;

		ZFloat3 asZFloat3() const;

		bool isZero() const;
		bool isAlmostZero( float eps=Z_EPS ) const;

		// norm
		float l1Norm() const;
		float l2Norm() const;
		float lpNorm( int p ) const;
		float infNorm() const;

		void write( ofstream& fout ) const;
		void read( ifstream& fin );

		// static functions
		static ZVector zero();
		static ZVector one();

		static ZVector unitX();
		static ZVector unitY();
		static ZVector unitZ();

		static ZVector xPosAxis();
		static ZVector yPosAxis();
		static ZVector zPosAxis();

		static ZVector xNegAxis();
		static ZVector yNegAxis();
		static ZVector zNegAxis();
};

inline
ZVector::ZVector()
: x(0.f), y(0.f), z(0.f)
{
	// nothing to do
}

inline
ZVector::ZVector( const ZVector& v )
: x(v.x), y(v.y), z(v.z)
{
	// nothing to do
}

inline
ZVector::ZVector( const ZFloat3& v )
: x(v.data[0]), y(v.data[1]), z(v.data[2])
{
	// nothing to do
}

inline
ZVector::ZVector( const float& X, const float& Y, const float& Z )
: x(X), y(Y), z(Z)
{
	// nothing to do
}

inline
ZVector::ZVector( const float* v )
: x(v[0]), y(v[1]), z(v[2])
{
	// nothing to do
}

inline
ZVector::ZVector( const float& s )
: x(s), y(s), z(s)
{
	// nothing to do
}

inline ZVector&
ZVector::set( const float* v )
{
	if( v )
	{
		x = v[0];
		y = v[1];
		z = v[2];
	}

	return (*this);
}

inline ZVector&
ZVector::set( const float& s )
{
	x = y = z = s;

	return (*this);
}

inline ZVector&
ZVector::set( const float& X, const float& Y, const float& Z )
{
	x = X;
	y = Y;
	z = Z;

	return (*this);
}

inline void
ZVector::get( float* v ) const
{
	v[0] = x;
	v[1] = y;
	v[2] = z;
}

inline void
ZVector::get( float& X, float& Y, float& Z ) const
{
	X = x;
	Y = y;
	Z = z;
}

inline ZVector&
ZVector::fill( const float& s )
{
	x = y = z = s;

	return (*this);
}

inline void
ZVector::zeroize()
{
	x = y = z = 0.f;
}

inline ZVector&
ZVector::zeroizeExcept( const int& i )
{
	if( i == 0 ) { y = z = 0.f; }
	if( i == 1 ) { z = x = 0.f; }
	if( i == 2 ) { x = y = 0.f; }

	return (*this);
}

inline float&
ZVector::operator[]( const int& i )
{
	switch( i )
	{
		default:
		case 0: { return x; }
		case 1: { return y; }
		case 2: { return z; }
	}
}

inline float&
ZVector::operator()( const int& i )
{
	switch( i )
	{
		default:
		case 0: { return x; }
		case 1: { return y; }
		case 2: { return z; }
	}
}

inline const float&
ZVector::operator[]( const int& i ) const
{
	switch( i )
	{
		default:
		case 0: { return x; }
		case 1: { return y; }
		case 2: { return z; }
	}
}

inline const float&
ZVector::operator()( const int& i ) const
{
	switch( i )
	{
		default:
		case 0: { return x; }
		case 1: { return y; }
		case 2: { return z; }
	}
}

inline ZVector&
ZVector::operator=( const ZVector& v )
{
	x = v.x;
	y = v.y;
	z = v.z;

	return (*this);
}

inline ZVector&
ZVector::operator=( const ZFloat3& v )
{
	x = v[0];
	y = v[1];
	z = v[2];

	return (*this);
}

inline ZVector&
ZVector::operator=( float* v )
{
	if( v )
	{
		x = v[0];
		y = v[1];
		z = v[2];
	}

	return (*this);
}

inline ZVector&
ZVector::operator=( const float& s )
{
	x = y = z = s;

	return (*this);
}

inline bool
ZVector::operator==( const ZVector& v ) const
{
	if( x != v.x ) { return false; }
	if( y != v.y ) { return false; }
	if( z != v.z ) { return false; }

	return true;
}

inline bool
ZVector::operator!=( const ZVector& v ) const
{
	if( x != v.x ) { return true; }
	if( y != v.y ) { return true; }
	if( z != v.z ) { return true; }

	return false;
}

inline bool
ZVector::operator<( const ZVector& v ) const
{
	if( x >= v.x ) { return false; }
	if( y >= v.y ) { return false; }
	if( z >= v.z ) { return false; }

	return true;
}

inline bool
ZVector::operator>( const ZVector& v ) const
{
	if( x <= v.x ) { return false; }
	if( y <= v.y ) { return false; }
	if( z <= v.z ) { return false; }

	return true;
}

inline bool
ZVector::operator<=( const ZVector& v ) const
{
	if( x > v.x ) { return false; }
	if( y > v.y ) { return false; }
	if( z > v.z ) { return false; }

	return true;
}

inline bool
ZVector::operator>=( const ZVector& v ) const
{
	if( x < v.x ) { return false; }
	if( y < v.y ) { return false; }
	if( z < v.z ) { return false; }

	return true;
}

inline ZVector&
ZVector::operator+=( const int& s )
{
	x += (float)s;
	y += (float)s;
	z += (float)s;

	return (*this);
}

inline ZVector&
ZVector::operator+=( const float& s )
{
	x += s;
	y += s;
	z += s;

	return (*this);
}

inline ZVector&
ZVector::operator+=( const double& s )
{
	x += (float)s;
	y += (float)s;
	z += (float)s;

	return (*this);
}

inline ZVector&
ZVector::operator-=( const int& s )
{
	x -= (float)s;
	y -= (float)s;
	z -= (float)s;

	return (*this);
}

inline ZVector&
ZVector::operator-=( const float& s )
{
	x -= s;
	y -= s;
	z -= s;

	return (*this);
}

inline ZVector&
ZVector::operator-=( const double& s )
{
	x -= (float)s;
	y -= (float)s;
	z -= (float)s;

	return (*this);
}

inline ZVector&
ZVector::operator+=( const ZVector& v )
{
	x += v.x;
	y += v.y;
	z += v.z;

	return (*this);
}

inline ZVector&
ZVector::operator-=( const ZVector& v )
{
	x -= v.x;
	y -= v.y;
	z -= v.z;

	return (*this);
}

inline ZVector
ZVector::operator+( const ZVector& v ) const
{
	return ZVector( x+v.x, y+v.y, z+v.z );
}

inline ZVector
ZVector::operator-( const ZVector& v ) const
{
	return ZVector( x-v.x, y-v.y, z-v.z );
}

inline ZVector&
ZVector::operator*=( const int& s )
{
	x *= (float)s;
	y *= (float)s;
	z *= (float)s;

	return (*this);
}

inline ZVector&
ZVector::operator*=( const float& s )
{
	x *= s;
	y *= s;
	z *= s;

	return (*this);
}

inline ZVector&
ZVector::operator*=( const double& s )
{
	x *= (float)s;
	y *= (float)s;
	z *= (float)s;

	return (*this);
}

inline ZVector&
ZVector::operator/=( const int& s )
{
	const float _s = 1.f / ( (float)s + Z_EPS );

	x *= _s;
	y *= _s;
	z *= _s;

	return (*this);
}

inline ZVector&
ZVector::operator/=( const float& s )
{
	const float _s = 1.f / ( s + Z_EPS );

	x *= _s;
	y *= _s;
	z *= _s;

	return (*this);
}

inline ZVector&
ZVector::operator/=( const double& s )
{
	const float _s = 1.f / ( (float)s + Z_EPS );

	x *= _s;
	y *= _s;
	z *= _s;

	return (*this);
}

inline ZVector
ZVector::operator*( const int& s ) const
{
	return ZVector( x*(float)s, y*(float)s, z*(float)s );
}

inline ZVector
ZVector::operator*( const float& s ) const
{
	return ZVector( x*s, y*s, z*s );
}

inline ZVector
ZVector::operator*( const double& s ) const
{
	return ZVector( x*(float)s, y*(float)s, z*(float)s );
}

inline ZVector
ZVector::operator/( const int& s ) const
{
	const float _s = 1.f / ( (float)s + Z_EPS );
	return ZVector( x*_s, y*_s, z*_s );
}

inline ZVector
ZVector::operator/( const float& s ) const
{
	const float _s = 1.f / ( s + Z_EPS );
	return ZVector( x*_s, y*_s, z*_s );
}

inline ZVector
ZVector::operator/( const double& s ) const
{
	const float _s = 1.f / ( (float)s + Z_EPS );
	return ZVector( x*_s, y*_s, z*_s );
}

inline ZVector
ZVector::operator-() const
{
	return ZVector( -x, -y, -z );
}

inline ZVector&
ZVector::negate()
{
	x = -x;
	y = -y;
	z = -z;

	return (*this);
}

inline ZVector
ZVector::negated() const
{
	return ZVector( -x, -y, -z );
}

inline ZVector&
ZVector::reverse()
{
	x = -x;
	y = -y;
	z = -z;

	return (*this);
}

inline ZVector
ZVector::reversed() const
{
	return ZVector( -x, -y, -z );
}

inline ZVector&
ZVector::abs()
{
	x = ZAbs( x );
	y = ZAbs( y );
	z = ZAbs( z );

	return (*this);
}

inline ZVector&
ZVector::clamp( const float& minValue, const float& maxValue )
{
	x = ZClamp( x, minValue, maxValue );
	y = ZClamp( y, minValue, maxValue );
	z = ZClamp( z, minValue, maxValue );

	return (*this);
}

inline ZVector&
ZVector::cycle( bool toLeft )
{
	if( toLeft ) {

		const float x0 = x;

		x = y;
		y = z;
		z = x0;

	} else { // toRight

		const float z0 = z;

		z = y;
		y = x;
		x = z0;

	}

	return (*this);
}

inline float
ZVector::operator*( const ZVector& v ) const
{
	return ( ( x * v.x ) + ( y * v.y ) + ( z * v.z ) );
}

inline float
ZVector::dot( const ZVector& v ) const
{
	return ( ( x * v.x ) + ( y * v.y ) + ( z * v.z ) );
}

inline ZVector
ZVector::operator^( const ZVector& v ) const
{
	return ZVector
	(
		( y * v.z ) - ( z * v.y ),
		( z * v.x ) - ( x * v.z ),
		( x * v.y ) - ( y * v.x )
	);
}

inline ZVector
ZVector::cross( const ZVector& v ) const
{
	return ZVector
	(
		( y * v.z ) - ( z * v.y ),
		( z * v.x ) - ( x * v.z ),
		( x * v.y ) - ( y * v.x )
	);
}

inline float
ZVector::length() const
{
	return sqrtf( x*x + y*y + z*z );
}

inline float
ZVector::squaredLength() const
{
	return ( x*x + y*y + z*z );
}

inline ZVector&
ZVector::normalize( bool accurate )
{
	float d = x*x + y*y + z*z + Z_EPS;
	d = accurate ? (1.f/sqrtf(d)) : ZFastInvSqrt(d);

	x *= d;
	y *= d;
	z *= d;

	return (*this);
}

inline ZVector
ZVector::normalized( bool accurate ) const
{
	ZVector tmp( *this );
	tmp.normalize( accurate );
	return tmp;
}

inline ZVector&
ZVector::robustNormalize()
{
	double d = ZPow2(double(x)) + ZPow2(double(y)) + ZPow2(double(z)) + Z_EPS;
	d = 1.0 / sqrt( d );

	x = (float)( x * d );
	y = (float)( y * d );
	z = (float)( z * d );

	return (*this);
}

inline ZVector
ZVector::robustNormalized() const
{
	ZVector tmp( *this );
	tmp.robustNormalize();
	return tmp;
}

inline ZVector
ZVector::direction( bool accurate ) const
{
	ZVector tmp( *this );
	tmp.normalize( accurate );
	return tmp;
}

inline ZVector
ZVector::robustDirection() const
{
	ZVector v( *this );
	v.robustNormalize();
	return v;
}

inline ZVector
ZVector::orthogonalDirection( bool accurate ) const
{
	ZVector v;
	{
		if( x > y ) {

			if( x > z ) { v.y = 1.f; } // x-direction is max.
			else        { v.x = 1.f; } // z-direction is max.

		} else {

			if( y > z ) { v.z = 1.f; } // y-direction is max.
			else        { v.x = 1.f; } // z-direction is max.

		}
	}

	return ZVector::operator^(v).normalize( accurate );
}

inline ZVector&
ZVector::limitLength( const float& targetLength )
{
	const float lenSQ = x*x + y*y + z*z;

	if( lenSQ > ZPow2(targetLength) )
	{
		const float d = targetLength / ( sqrtf(lenSQ) + Z_EPS );

		x *= d;
		y *= d;
		z *= d;
	}

	return (*this);
}

inline float
ZVector::distanceTo( const ZVector& p ) const
{
	return sqrtf( ZPow2(x-p.x) + ZPow2(y-p.y) + ZPow2(z-p.z) );
}

inline float
ZVector::squaredDistanceTo( const ZVector& p ) const
{
	return ( ZPow2(x-p.x) + ZPow2(y-p.y) + ZPow2(z-p.z) );
}

inline bool
ZVector::isEquivalent( const ZVector& v, float e ) const
{
	const ZVector diff( *this - v );
	const float lenSQ = diff.squaredLength();
	return ( lenSQ < (e*e) );
}

inline bool
ZVector::isParallel( const ZVector& v, float e ) const
{
	return ZAlmostSame( (*this)*v, 1.f, e );
}

inline float
ZVector::min() const
{
	return ZMin( x, y, z );
}

inline float
ZVector::max() const
{
	return ZMax( x, y, z );
}

inline float
ZVector::absMin() const
{
	return ZAbsMin( x, y, z );
}

inline float
ZVector::absMax() const
{
	return ZAbsMax( x, y, z );
}

inline int
ZVector::minIndex() const
{
	return ( (x<y) ? ( (x<z) ? 0 : 2 ) : ( (y<z) ? 1 : 2 ) );
}

inline int
ZVector::maxIndex() const
{
	return ( (x>y) ? ( (x>z) ? 0 : 2 ) : ( (y>z) ? 1 : 2 ) );
}

inline int
ZVector::absMinIndex() const
{
	const float X = ZAbs(x);
	const float Y = ZAbs(y);
	const float Z = ZAbs(z);

	return ( (X<Y) ? ( (X<Z) ? 0 : 2 ) : ( (Y<Z) ? 1 : 2 ) );
}

inline int
ZVector::absMaxIndex() const
{
	const float X = ZAbs(x);
	const float Y = ZAbs(y);
	const float Z = ZAbs(z);

	return ( (X>Y) ? ( (X>Z) ? 0 : 2 ) : ( (Y>Z) ? 1 : 2 ) );
}

inline float
ZVector::sum() const
{
	return ( x + y + z );
}

inline ZVector&
ZVector::setComponentwiseMin( const ZVector& v )
{
	if( v.x < x ) { x = v.x; }
	if( v.y < y ) { y = v.y; }
	if( v.z < z ) { z = v.z; }

	return (*this);
}

inline ZVector&
ZVector::setComponentwiseMax( const ZVector& v )
{
	if( v.x > x ) { x = v.x; }
	if( v.y > y ) { y = v.y; }
	if( v.z > z ) { z = v.z; }

	return (*this);
}

inline ZVector
ZVector::componentwiseMin( const ZVector& v ) const
{
	return ZVector( ZMin(x,v.x), ZMin(y,v.y), ZMin(z,v.z) );
}

inline ZVector
ZVector::componentwiseMax( const ZVector& v ) const
{
	return ZVector( ZMax(x,v.x), ZMax(y,v.y), ZMax(z,v.z) );
}

inline ZFloat3
ZVector::asZFloat3() const
{
	return ZFloat3( x, y, z );
}

inline bool
ZVector::isZero() const
{
	if( x != 0.f ) { return false; }
	if( y != 0.f ) { return false; }
	if( z != 0.f ) { return false; }

	return true;
}

inline bool
ZVector::isAlmostZero( float eps ) const
{
	if( x < -eps ) { return false; }
	if( x >  eps ) { return false; }

	if( y < -eps ) { return false; }
	if( y >  eps ) { return false; }

	if( z < -eps ) { return false; }
	if( z >  eps ) { return false; }

	return true;
}

inline float
ZVector::l1Norm() const
{
	return ( ZAbs(x) + ZAbs(y) + ZAbs(z) );
}

inline float
ZVector::l2Norm() const
{
	return sqrtf( ZPow2(x) + ZPow2(y) + ZPow2(z) );
}

inline float
ZVector::lpNorm( int p ) const
{
	if( p <= 0 ) { return 0.f; }

	const float sum = pow(ZAbs(x),p) + pow(ZAbs(y),p) + pow(ZAbs(z),p);
	return powf( sum, 1/(float)p );
}

inline float
ZVector::infNorm() const
{
	return ZAbsMax( x, y, z );
}

inline void
ZVector::write( ofstream& fout ) const
{
	fout.write( (char*)&x, sizeof(float)*3 );
}

inline void
ZVector::read( ifstream& fin )
{
	fin.read( (char*)&x, sizeof(float)*3 );
}

inline ZVector
ZVector::zero()
{
	return ZVector( 0.f );
}

inline ZVector
ZVector::one()
{
	return ZVector( 1.f );
}

inline ZVector
ZVector::unitX()
{
	return ZVector( 1.f, 0.f, 0.f );
}

inline ZVector
ZVector::unitY()
{
	return ZVector( 0.f, 1.f, 0.f );
}

inline ZVector
ZVector::unitZ()
{
	return ZVector( 0.f, 0.f, 1.f );
}

inline ZVector
ZVector::xPosAxis()
{
	return ZVector( 1.f, 0.f, 0.f );
}

inline ZVector
ZVector::yPosAxis()
{
	return ZVector( 0.f, 1.f, 0.f );
}

inline ZVector
ZVector::zPosAxis()
{
	return ZVector( 0.f, 0.f, 1.f );
}

inline ZVector
ZVector::xNegAxis()
{
	return ZVector( -1.f, 0.f, 0.f );
}

inline ZVector
ZVector::yNegAxis()
{
	return ZVector( 0.f, -1.f, 0.f );
}

inline ZVector
ZVector::zNegAxis()
{
	return ZVector( 0.f, 0.f, -1.f );
}

/////////////////////////////////////////
// simple non-member utility funcitons //
/////////////////////////////////////////

inline ZVector
operator*( const int& s, const ZVector& v )
{
	return ZVector( v.x*(float)s, v.y*(float)s, v.z*(float)s );
}

inline ZVector
operator*( const float& s, const ZVector& v )
{
	return ZVector( v.x*s, v.y*s, v.z*s );
}

inline ZVector
operator*( const double& s, const ZVector& v )
{
	return ZVector( float(v.x*s), float(v.y*s), float(v.z*s) );
}

inline ostream&
operator<<( ostream& os, const ZVector& v )
{
	os << "( " << v.x << ", " << v.y << ", " << v.z << " ) ";
	return os;
}

inline istream&
operator>>( istream& istr, ZVector& v )
{
	istr >> v.x >> v.y >> v.z;
	return istr;
}

inline void
PrintLocator( const ZVector& v )
{
	cout << "spaceLocator -p " << v.x << " " << v.y << " " << v.z << ";" << endl;
}

inline float
MAG( const ZVector& A )
{
	return sqrtf( ZPow2(A.x) + ZPow2(A.y) + ZPow2(A.z) );
}

inline float
LEN( const ZVector& A )
{
	return sqrtf( ZPow2(A.x) + ZPow2(A.y) + ZPow2(A.z) );
}

inline float
DST( const ZVector& A, const ZVector& B )
{
	return sqrtf( ZPow2(A.x-B.x) + ZPow2(A.y-B.y) + ZPow2(A.z-B.z) );
}

// apploximated geodesic distance
inline float
DST( const ZVector& p0, const ZVector& p1, const ZVector& n0, const ZVector& n1 )
{
	ZVector dir( p1 - p0 );

	const float de = dir.length();

	dir *= 1.f / ( de + Z_EPS ); // normalize

	const float c0 = ZClamp( dir*n0, -1.f, 1.f );
	const float c1 = ZClamp( dir*n1, -1.f, 1.f );

	if( ZAlmostSame( c0, c1 ) )
	{
		return ( de / ( sqrtf( 1.f - ( c0 *c1 ) ) + 1e-30f ) );
	}

	const float dg = ( ( asinf(c1) - asinf(c0) ) / ( c1 - c0 ) );

	return ( dg * de );
}

inline void
ADD( ZVector& A, const ZVector& B, const ZVector& C )
{
	A.x = B.x + C.x;
	A.y = B.y + C.y;
	A.z = B.z + C.z;
}

inline void
SUB( ZVector& A, const ZVector& B, const ZVector& C )
{
	A.x = B.x - C.x;
	A.y = B.y - C.y;
	A.z = B.z - C.z;
}

inline void
INC( ZVector& A, const ZVector& B )
{
	A.x += B.x;
	A.y += B.y;
	A.z += B.z;
}

inline void
MUL( ZVector& A, float b, const ZVector& B )
{
	A.x = b * B.x;
	A.y = b * B.y;
	A.z = b * B.z;
}

inline void
INCMUL( ZVector& A, float b, const ZVector& B )
{
	A.x += b * B.x;
	A.y += b * B.y;
	A.z += b * B.z;
}

inline void
SUBMUL( ZVector& A, float b, const ZVector& B )
{
	A.x -= b * B.x;
	A.y -= b * B.y;
	A.z -= b * B.z;
}

inline float
DOT( const ZVector& a, const ZVector& b )
{
	return ( (a.x*b.x) + (a.y*b.y) + (a.z*b.z) );
}

inline ZVector
CRS( const ZVector& a, const ZVector& b )
{
	return ZVector
	(
		( a.y * b.z ) - ( a.z * b.y ),
		( a.z * b.x ) - ( a.x * b.z ),
		( a.x * b.y ) - ( a.y * b.x )
	);
}

////////////////
// data types //
////////////////

typedef ZVector ZPoint;

ZELOS_NAMESPACE_END

#endif

