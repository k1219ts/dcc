//----------//
// ZColor.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.03.08                               //
//-------------------------------------------------------//

#ifndef _ZColor_h_
#define _ZColor_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

/// @brief A color with four components.
class ZColor
{
	public:

		union
		{
			struct
			{
				union { float r; float h; float x; };
				union { float g; float s; float y; };
				union { float b; float v; float z; };
			};
			float data[3];
		};

		float a;	///< The alpha component.

	public:

		ZColor();
		ZColor( const ZColor& c );
		ZColor( const ZFloat3& c );
		ZColor( const float& red, const float& green, const float& blue, const float& alpha=1.0f );
		ZColor( const float& gray, const float& alpha=1.0f );

		ZColor& set( const float& red, const float& green, const float& blue, const float& alpha=1.0f );
		ZColor& set( const float& gray, const float& alpha=1.0f );

		ZColor& fill( const float& s );

		void zeroize();

		float& operator[]( const int& i );
		float& operator()( const int& i );

		const float& operator[]( const int& i ) const;
		const float& operator()( const int& i ) const;

		ZColor& operator=( const ZColor& c );
		ZColor& operator=( const ZFloat3& c );

		bool operator==( const ZColor& c ) const;
		bool operator!=( const ZColor& c ) const;

		ZColor& operator+=( const int& s );
		ZColor& operator+=( const float& s );
		ZColor& operator+=( const double& s );

		ZColor& operator-=( const int& s );
		ZColor& operator-=( const float& s );
		ZColor& operator-=( const double& s );

		ZColor& operator+=( const ZColor& c );
		ZColor& operator-=( const ZColor& c );

		ZColor operator+( const ZColor& c ) const;
		ZColor operator-( const ZColor& c ) const;

		ZColor& operator*=( const int& scalar );
		ZColor& operator*=( const float& scalar );
		ZColor& operator*=( const double& scalar );

		ZColor& operator/=( const int& scalar );
		ZColor& operator/=( const float& scalar );
		ZColor& operator/=( const double& scalar );

		ZColor operator*( const int& s ) const;
		ZColor operator*( const float& s ) const;
		ZColor operator*( const double& s ) const;

		ZColor operator/( const int& s ) const;
		ZColor operator/( const float& s ) const;
		ZColor operator/( const double& s ) const;

		ZColor operator-() const;

		bool isEquivalent( const ZColor& c, float epsilon=Z_EPS ) const;

		float intensity() const;

		void rgb2hsv();
		void hsv2rgb();

		void write( ofstream& fout ) const;
		void read( ifstream& fin );

		// static functions
		static ZColor black();
		static ZColor white();
		static ZColor red();
		static ZColor green();
		static ZColor blue();
		static ZColor yellow();
		static ZColor magenta();
		static ZColor cyan();
		static ZColor orange();
		static ZColor gray();
		static ZColor grey();
};

inline
ZColor::ZColor()
: r(0.f), g(0.f), b(0.f), a(1.f)
{}

inline
ZColor::ZColor( const ZColor& c )
: r(c.r), g(c.g), b(c.b), a(c.a)
{}

inline
ZColor::ZColor( const ZFloat3& c )
: r(c[0]), g(c[1]), b(c[2]), a(1.f)
{}

inline
ZColor::ZColor( const float& red, const float& green, const float& blue, const float& alpha )
{
	r = red;
	g = green;
	b = blue;
	a = alpha;
}

inline
ZColor::ZColor( const float& gray, const float& alpha )
{
	r = g = b = gray;
	a = alpha;
}

inline ZColor&
ZColor::set( const float& red, const float& green, const float& blue, const float& alpha )
{
	r = red;
	g = green;
	b = blue;
	a = alpha;

	return (*this);
}

inline ZColor&
ZColor::set( const float& gray, const float& alpha )
{
	r = g = b = gray;
	a = alpha;

	return (*this);
}

// except alpha
inline ZColor&
ZColor::fill( const float& s )
{
	r = g = b = s;

	return (*this);
}

// except alpha
inline void
ZColor::zeroize()
{
	r = g = b = 0.f;
}

inline float&
ZColor::operator[]( const int& i )
{
	switch( i )
	{
		default:
		case 0: { return r; }
		case 1: { return g; }
		case 2: { return b; }
		case 3: { return a; }
	}
}

inline float&
ZColor::operator()( const int& i )
{
	switch( i )
	{
		default:
		case 0: { return r; }
		case 1: { return g; }
		case 2: { return b; }
		case 3: { return a; }
	}
}

inline const float&
ZColor::operator[]( const int& i ) const
{
	switch( i )
	{
		default:
		case 0: { return r; }
		case 1: { return g; }
		case 2: { return b; }
		case 3: { return a; }
	}
}

inline const float&
ZColor::operator()( const int& i ) const
{
	switch( i )
	{
		default:
		case 0: { return r; }
		case 1: { return g; }
		case 2: { return b; }
		case 3: { return a; }
	}
}

inline ZColor&
ZColor::operator=( const ZColor& c )
{
	r = c.r;
	g = c.g;
	b = c.b;
	a = c.a;

	return (*this);
}

inline ZColor&
ZColor::operator=( const ZFloat3& c )
{
	r = c[0];
	g = c[1];
	b = c[2];

	return (*this);
}

// except alpha
inline bool
ZColor::operator==( const ZColor& c ) const
{
	if( r != c.r ) { return false; }
	if( g != c.g ) { return false; }
	if( b != c.b ) { return false; }

	return true;
}

// except alpha
inline bool
ZColor::operator!=( const ZColor& c ) const
{
	if( r != c.r ) { return true; }
	if( g != c.g ) { return true; }
	if( b != c.b ) { return true; }

	return false;
}

// except alpha
inline ZColor&
ZColor::operator+=( const int& s )
{
	r += (float)s;
	g += (float)s;
	b += (float)s;

	return (*this);
}

// except alpha
inline ZColor&
ZColor::operator+=( const float& s )
{
	r += s;
	g += s;
	b += s;

	return (*this);
}

// except alpha
inline ZColor&
ZColor::operator+=( const double& s )
{
	r += (float)s;
	g += (float)s;
	b += (float)s;

	return (*this);
}

// except alpha
inline ZColor&
ZColor::operator-=( const int& s )
{
	r -= (float)s;
	g -= (float)s;
	b -= (float)s;

	return (*this);
}

// except alpha
inline ZColor&
ZColor::operator-=( const float& s )
{
	r -= s;
	g -= s;
	b -= s;

	return (*this);
}

// except alpha
inline ZColor&
ZColor::operator-=( const double& s )
{
	r -= (float)s;
	g -= (float)s;
	b -= (float)s;

	return (*this);
}

// except alpha
inline ZColor&
ZColor::operator+=( const ZColor& c )
{
	r += c.r;
	g += c.g;
	b += c.b;

	return (*this);
}

// except alpha
inline ZColor&
ZColor::operator-=( const ZColor& c )
{
	r -= c.r;
	g -= c.g;
	b -= c.b;

	return (*this);
}

// except alpha
inline ZColor
ZColor::operator+( const ZColor& c ) const
{
	return ZColor( r+c.r, g+c.g, b+c.b );
}

// except alpha
inline ZColor
ZColor::operator-( const ZColor& c ) const
{
	return ZColor( r-c.r, g-c.g, b-c.b );
}

// except alpha
inline ZColor&
ZColor::operator*=( const int& s )
{
	r *= (float)s;
	g *= (float)s;
	b *= (float)s;

	return (*this);
}

// except alpha
inline ZColor&
ZColor::operator*=( const float& s )
{
	r *= s;
	g *= s;
	b *= s;

	return (*this);
}

// except alpha
inline ZColor&
ZColor::operator*=( const double& s )
{
	r *= (float)s;
	g *= (float)s;
	b *= (float)s;

	return (*this);
}

// except alpha
inline ZColor&
ZColor::operator/=( const int& s )
{
	const float _s = 1.f / ( (float)s + Z_EPS );

	r *= _s;
	g *= _s;
	b *= _s;

	return (*this);
}

// except alpha
inline ZColor&
ZColor::operator/=( const float& s )
{
	const float _s = 1.f / ( s + Z_EPS );

	r *= _s;
	g *= _s;
	b *= _s;

	return (*this);
}

// except alpha
inline ZColor&
ZColor::operator/=( const double& s )
{
	const float _s = 1.f / ( (float)s + Z_EPS );

	r *= _s;
	g *= _s;
	b *= _s;

	return (*this);
}

// except alpha
inline ZColor
ZColor::operator*( const int& s ) const
{
	return ZColor( r*(float)s, g*(float)s, b*(float)s, a );
}

// except alpha
inline ZColor
ZColor::operator*( const float& s ) const
{
	return ZColor( r*s, g*s, b*s, a );
}

// except alpha
inline ZColor
ZColor::operator*( const double& s ) const
{
	return ZColor( r*(float)s, g*(float)s, b*(float)s, a );
}

// except alpha
inline ZColor
ZColor::operator/( const int& s ) const
{
	const float _s = 1.f / ( (float)s + Z_EPS );
	return ZColor( r*_s, g*_s, b*_s, a );
}

// except alpha
inline ZColor
ZColor::operator/( const float& s ) const
{
	const float _s = 1.f / ( s + Z_EPS );
	return ZColor( r*_s, g*_s, b*_s, a );
}

// except alpha
inline ZColor
ZColor::operator/( const double& s ) const
{
	const float _s = 1.f / ( (float)s + Z_EPS );
	return ZColor( r*_s, g*_s, b*_s, a );
}

// except alpha
inline ZColor
ZColor::operator-() const
{
	return ZColor( -r, -g, -b, a );
}

inline bool
ZColor::isEquivalent( const ZColor& c, float e ) const
{
	if( !ZAlmostSame( r - c.r, e ) ) { return false; }
	if( !ZAlmostSame( g - c.g, e ) ) { return false; }
	if( !ZAlmostSame( b - c.b, e ) ) { return false; }

	return true;
}

inline float
ZColor::intensity() const
{
	return ( 0.299f*r + 0.587f*g + 0.114f*b );
}

// from: www.rapidtables.com/convert/color/rgb-to-hsv.htm
//
// r: 0.0 ~ 1.0
// g: 0.0 ~ 1.0
// b: 0.0 ~ 1.0
//
// h: 0.0 ~ 360.0 
// s: 0.0 ~ 1.0
// v: 0.0 ~ 1.0
//
inline void
ZColor::rgb2hsv()
{
	float R=r, G=g, B=b;

	ZFloat3 c( R, G, B );

	const int minIdx = c.minIndex();
	const int maxIdx = c.maxIndex();

	const float& min = (minIdx==0) ? R : ( (minIdx==1) ? G : B );
	const float& max = (maxIdx==0) ? R : ( (maxIdx==1) ? G : B );
	const float delta = max - min;

	if( ZAlmostZero(delta) ) { h = 0.f;                                   }
	else if( maxIdx == 0 )   { h = 60.f * ZRemainder( (G-B)/delta, 6.f ); }
	else if( maxIdx == 1 )   { h = 60.f * ( (B-R)/delta + 2.f );          }
	else                     { h = 60.f * ( (R-G)/delta + 4.f );          }

	if( ZAlmostZero(max) )   { s = 0.f;       }
	else                     { s = delta/max; }

	v = max;
}

// from: www.rapidtables.com/convert/color/hsv-to-rgb.htm
//
// h: 0.0 ~ 360.0 
// s: 0.0 ~ 1.0
// v: 0.0 ~ 1.0
//
// r: 0.0 ~ 1.0
// g: 0.0 ~ 1.0
// b: 0.0 ~ 1.0
//
inline void
ZColor::hsv2rgb()
{
	float H=h, S=s, V=v;

	// H: 0.0 ~ 360.0
	H = ZRemainder( H, 360.f );
	if( H < 0.f ) { H += 360.f; }

	const float C = V * S;
	const float X = C * (1.f-ZAbs(ZRemainder(H/60.f,2.f)-1.f));
	const float m = V - C;

	     if( H <  60.f ) { r=C;   g=X;   b=0.f; }
	else if( H < 120.f ) { r=X;   g=C;   b=0.f; }
	else if( H < 180.f ) { r=0.f; g=C;   b=X;   }
	else if( H < 240.f ) { r=0.f; g=X;   b=C;   }
	else if( H < 300.f ) { r=X;   g=0.f; b=C;   }
	else                 { r=C;   g=0.f; b=X;   }

	r += m;
	g += m;
	b += m;
}

inline void
ZColor::write( ofstream& fout ) const
{
	fout.write( (char*)&r, sizeof(float)*4 );
}

inline void
ZColor::read( ifstream& fin )
{
	fin.read( (char*)&r, sizeof(float)*4 );
}

inline ZColor
ZColor::black()
{
	return ZColor( 0.f, 0.f, 0.f, 1.f );
}

inline ZColor
ZColor::white()
{
	return ZColor( 1.f, 1.f, 1.f, 1.f );
}

inline ZColor
ZColor::red()
{
	return ZColor( 1.f, 0.f, 0.f, 1.f );
}

inline ZColor
ZColor::green()
{
	return ZColor( 0.f, 1.f, 0.f, 1.f );
}

inline ZColor
ZColor::blue()
{
	return ZColor( 0.f, 0.f, 1.f, 1.f );
}

inline ZColor
ZColor::yellow()
{
	return ZColor( 1.f, 1.f, 0.f, 1.f );
}

inline ZColor
ZColor::magenta()
{
	return ZColor( 1.f, 0.f, 1.f, 1.f );
}

inline ZColor
ZColor::cyan()
{
	return ZColor( 0.f, 1.f, 1.f, 1.f );
}

inline ZColor
ZColor::orange()
{
	return ZColor( 1.f, 0.6f, 0.f, 1.f );
}

inline ZColor
ZColor::gray()
{
	return ZColor( 0.5f, 0.5f, 0.5f, 1.f );
}

inline ZColor
ZColor::grey()
{
	return ZColor( 0.5f, 0.5f, 0.5f, 1.f );
}

/////////////////////////////////////////
// simple non-member utility funcitons //
/////////////////////////////////////////

// except alpha
inline ZColor
operator*( const int& s, const ZColor& c )
{
	return ZColor( c.r*(float)s, c.g*(float)s, c.b*(float)s );
}

// except alpha
inline ZColor
operator*( const float& s, const ZColor& c )
{
	return ZColor( c.r*s, c.g*s, c.b*s );
}

// except alpha
inline ZColor
operator*( const double& s, const ZColor& c )
{
	return ZColor( float(c.r*s), float(c.g*s), float(c.b*s) );
}

inline ostream&
operator<<( ostream& os, const ZColor& object )
{
	os << "( " << object.r << ", " << object.g << ", " << object.b << ", " << object.a << " )";
	return os;
}

ZELOS_NAMESPACE_END

#endif

