//------------//
// ZComplex.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.03.08                               //
//-------------------------------------------------------//

#ifndef _ZComplex_h_
#define _ZComplex_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

/// @brief A comple number.
class ZComplex
{
	public:

		float r;	// The real component.
		float i;	// The im component.

	public:

		ZComplex();
		ZComplex( const ZComplex& source );
		ZComplex( const float& re, const float& im );
		ZComplex( const float source[2] );

		ZComplex& set( const float& re, const float& im=0 );
		ZComplex& set( const float source[2] );
		ZComplex& setByPolar( const float& radius, const float& theta );

		void zeroize();

		float& operator[]( const int& idx );
		float& operator()( const int& idx );

		const float& operator[]( const int& idx ) const;
		const float& operator()( const int& idx ) const;

		ZComplex& operator=( const float& realPart );
		ZComplex& operator=( const ZComplex& source );

		bool operator==( const ZComplex& other ) const;
		bool operator!=( const ZComplex& other ) const;

		ZComplex& operator+=( const ZComplex& other );
		ZComplex& operator-=( const ZComplex& other );
		ZComplex& operator*=( const ZComplex& right );

		ZComplex& operator*=( const int& scalar );
		ZComplex& operator*=( const float& scalar );
		ZComplex& operator*=( const double& scalar );

		ZComplex& operator/=( const int& scalar );
		ZComplex& operator/=( const float& scalar );
		ZComplex& operator/=( const double& scalar );

		ZComplex operator+( const ZComplex& other ) const;
		ZComplex operator-( const ZComplex& other ) const;
		ZComplex operator*( const ZComplex& right ) const;

		ZComplex operator*( const int& scalar ) const;
		ZComplex operator*( const float& scalar ) const;
		ZComplex operator*( const double& scalar ) const;

		ZComplex operator/( const int& scalar ) const;
		ZComplex operator/( const float& scalar ) const;
		ZComplex operator/( const double& scalar ) const;

		ZComplex operator-() const;

		ZComplex& conjugate();
		ZComplex conjugated() const;

		ZComplex& inverse();
		ZComplex inversed() const;

		float radius() const;
		float squaredRadius() const;

		float angle( bool zeroTo2Pi=true ) const;

		ZPoint rotate( const ZPoint& p ) const;

		ZComplex& rotate_counterClockwise_90();
		ZComplex  rotated_counterClockwise_90() const;

		ZComplex& rotate_clockwise_90();
		ZComplex  rotated_clockwise_90() const;

		static ZComplex plus_i();  // 0+i.
		static ZComplex minus_i(); // 0-i.

		static ZComplex polar( const float& radius, const float& theta );
		static ZComplex polar( const float& theta );

		static ZComplex exp( const float& exponent );

		void write( ofstream& fout ) const;
		void read( ifstream& fin );
};

inline
ZComplex::ZComplex()
: r(0.f), i(0.f)
{}

inline
ZComplex::ZComplex( const ZComplex& c )
: r(c.r), i(c.i)
{}

inline
ZComplex::ZComplex( const float& re, const float& im )
: r(re), i(im)
{}

inline
ZComplex::ZComplex( const float c[2] )
: r(c[0]), i(c[1])
{}

inline ZComplex&
ZComplex::set( const float& re, const float& im )
{
	r = re;
	i = im;

	return (*this);
}

inline ZComplex&
ZComplex::set( const float c[2] )
{
	r = c[0];
	i = c[1];

	return (*this);
}

inline ZComplex&
ZComplex::setByPolar( const float& radius, const float& theta )
{
	r = radius * cosf(theta);
	i = radius * sinf(theta);

	return (*this);
}

inline void
ZComplex::zeroize()
{
	r = i = 0.f;
}

inline const float&
ZComplex::operator[]( const int& idx ) const
{
	switch( idx )
	{
		default:
		case 0: { return r; }
		case 1: { return i; }
	}
}

inline float&
ZComplex::operator[]( const int& idx )
{
	switch( idx )
	{
		default:
		case 0: { return r; }
		case 1: { return i; }
	}
}

inline const float&
ZComplex::operator()( const int& idx ) const
{
	switch( idx )
	{
		default:
		case 0: { return r; }
		case 1: { return i; }
	}
}

inline float&
ZComplex::operator()( const int& idx )
{
	switch( idx )
	{
		default:
		case 0: { return r; }
		case 1: { return i; }
	}
}

inline ZComplex&
ZComplex::operator=( const float& realPart )
{
	r = realPart;
	i = 0.f;

	return (*this);
}

inline ZComplex&
ZComplex::operator=( const ZComplex& c )
{
	r = c.r;
	i = c.i;

	return (*this);
}

inline bool
ZComplex::operator==( const ZComplex& c ) const
{
	if( r!=c.r ) { return false; }
	if( i!=c.i ) { return false; }

	return true;
}

inline bool
ZComplex::operator!=( const ZComplex& c ) const
{
	if( r!=c.r ) { return true; }
	if( i!=c.i ) { return true; }

	return false;
}

inline ZComplex&
ZComplex::operator+=( const ZComplex& c )
{
	r += c.r;
	i += c.i;

	return (*this);
}

inline ZComplex&
ZComplex::operator-=( const ZComplex& c )
{
	r -= c.r;
	i -= c.i;

	return (*this);
}

inline ZComplex&
ZComplex::operator*=( const ZComplex& c )
{
	const float r0 = r, i0 = i;

	r = (r0*c.r) - (i0*c.i);
	i = (r0*c.i) + (i0*c.r);

	return (*this);
}

inline ZComplex&
ZComplex::operator*=( const int& s )
{
	r *= (float)s;
	i *= (float)s;

	return (*this);
}

inline ZComplex&
ZComplex::operator*=( const float& s )
{
	r *= s;
	i *= s;

	return (*this);
}

inline ZComplex&
ZComplex::operator*=( const double& s )
{
	r *= (float)s;
	i *= (float)s;

	return (*this);
}

inline ZComplex&
ZComplex::operator/=( const int& s )
{
	const float _s = 1.f / ( (float)s + Z_EPS );

	r *= _s;
	i *= _s;

	return (*this);
}

inline ZComplex&
ZComplex::operator/=( const float& s )
{
	const float _s = 1.f / ( s + Z_EPS );

	r *= _s;
	i *= _s;

	return (*this);
}

inline ZComplex&
ZComplex::operator/=( const double& s )
{
	const float _s = 1.f / ( (float)s + Z_EPS );

	r *= _s;
	i *= _s;

	return (*this);
}

inline ZComplex
ZComplex::operator+( const ZComplex& c ) const
{
	return ZComplex( r+c.r, i+c.i );
}

inline ZComplex
ZComplex::operator-( const ZComplex& c ) const
{
	return ZComplex( r-c.r, i-c.i );
}

inline ZComplex
ZComplex::operator*( const ZComplex& c ) const
{
	return ZComplex( r*c.r-i*c.i, r*c.i + i*c.r );
}

inline ZComplex
ZComplex::operator*( const int& s ) const
{
	return ZComplex( r*(float)s, i*(float)s );
}

inline ZComplex
ZComplex::operator*( const float& s ) const
{
	return ZComplex( r*s, i*s );
}

inline ZComplex
ZComplex::operator*( const double& s ) const
{
	return ZComplex( r*(float)s, i*(float)s );
}

inline ZComplex
ZComplex::operator/( const int& s ) const
{
	const float _s = 1.f / ( (float)s + Z_EPS );
	return ZComplex( r*_s, i*_s );
}

inline ZComplex
ZComplex::operator/( const float& s ) const
{
	const float _s = 1.f / ( (float)s + Z_EPS );
	return ZComplex( r*_s, i*_s );
}

inline ZComplex
ZComplex::operator/( const double& s ) const
{
	const float _s = 1.f / ( (float)s + Z_EPS );
	return ZComplex( r*_s, i*_s );
}

inline ZComplex
ZComplex::operator-() const
{
	return ZComplex( -r, -i );
}

inline ZComplex&
ZComplex::conjugate()
{
	i = -i;
	return (*this);
}

inline ZComplex
ZComplex::conjugated() const
{
	return ZComplex( r, -i );
}

inline ZComplex&
ZComplex::inverse()
{
	const float _d = 1.f / ( r*r + i*i + Z_EPS );

	r *= _d;
	i *= _d;

	return (*this);
}

inline ZComplex
ZComplex::inversed() const
{
	ZComplex c( *this );
	c.inverse();
	return c;
}

inline float
ZComplex::radius() const
{
	return sqrtf( r*r + i*i );
}

inline float
ZComplex::squaredRadius() const
{
	return ( r*r + i*i );
}

inline float
ZComplex::angle( bool zeroTo2Pi ) const
{
	float ang = atan2f( i, r );
	if( zeroTo2Pi && ( ang<0.f ) ) { ang += Z_PIx2; }

	return ang;
}

inline ZPoint
ZComplex::rotate( const ZPoint& p ) const
{
	float R = r, I = i;
	float r2 = R*R+I*I;
	if( ZAlmostZero(r2) ) { return p; }
	if( !ZAlmostSame( r2, 1.f ) )
	{
		float r = sqrtf(r2);
		R /= r;
		I /= r;
	}

	return ZPoint( R*p.x-I*p.y, I*p.x+R*p.y );
}

inline ZComplex&
ZComplex::rotate_counterClockwise_90()
{
	ZSwap( r, i );
	r = -r;

	return (*this);
}

inline ZComplex
ZComplex::rotated_counterClockwise_90() const
{
	return ZComplex( -i, r );
}

inline ZComplex&
ZComplex::rotate_clockwise_90()
{
	ZSwap( r, i );
	i = -i;

	return (*this);
}

inline ZComplex
ZComplex::rotated_clockwise_90() const
{
	return ZComplex( i, -r );
}

inline ZComplex
ZComplex::plus_i()
{
	return ZComplex(0.f,1.f);
}

inline ZComplex
ZComplex::minus_i()
{
	return ZComplex(0.f,-1.f);
}

inline ZComplex
ZComplex::polar( const float& r, const float& a )
{
	return ZComplex( r*cosf(a), r*sinf(a) );
}

inline ZComplex
ZComplex::polar( const float& a )
{
	return ZComplex( cosf(a), sinf(a) );
}

inline ZComplex
ZComplex::exp( const float& e )
{
	return ZComplex( cosf(e), sinf(e) );
}

inline void
ZComplex::write( ofstream& fout ) const
{
	fout.write( (char*)&r, sizeof(float)*2 );
}

inline void
ZComplex::read( ifstream& fin )
{
	fin.read( (char*)&r, sizeof(float)*2 );
}

/////////////////////////////////////////
// simple non-member utility funcitons //
/////////////////////////////////////////

inline ZComplex
operator*( const int& s, const ZComplex& c )
{
	return ZComplex( c.r*s, c.i*s );
}

inline ZComplex
operator*( const float& s, const ZComplex& c )
{
	return ZComplex( c.r*s, c.i*s );
}

inline ZComplex
operator*( const double& s, const ZComplex& c )
{
	return ZComplex( (float)(c.r*s), (float)(c.i*s) );
}

inline ostream&
operator<<( ostream& os, const ZComplex& object )
{
	os << "( " << object.r << " + " << object.i << "i )" << endl;
	return os;
}

ZELOS_NAMESPACE_END

#endif

