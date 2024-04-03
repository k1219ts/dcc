#ifndef _BS_MathUtils_h_
#define _BS_MathUtils_h_

#include <BeyondScreen.h>

BS_NAMESPACE_BEGIN

#define RadToDeg(x) ((x)*57.295779513082322864647721871733665466309f)
#define DegToRad(x) ((x)*0.0174532925199432954743716805978692718782f)

inline double ABS( const double& x )
{
    return ( ( x > 0 ) ? x : -x );
}

inline double SQR( const double& x )
{
    return (x*x);
}

inline bool SAME( const double& a, const double& b, const double epsilon=1e-10 )
{
    return ( ( ABS(a-b) < epsilon ) ? true : false );
}

inline double MIN( const double& a, const double& b )
{
    return ( ( a < b ) ? a : b );
}

inline double MIN( const double& a, const double& b, const double& c )
{
	return ( (a<b) ? ( (a<c) ? a : c ) : ( (b<c) ? b : c ) );
}

inline double MAX( const double& a, const double& b )
{
    return ( ( a > b ) ? a : b );
}

inline double MAX( const double& a, const double& b, const double& c )
{
	return ( (a>b) ? ( (a>c) ? a : c ) : ( (b>c) ? b : c ) );
}

inline void GetMinMax( const double& a, const double& b, double& min, double& max )
{
	if( a < b ) { min = a; max = b; }
	else        { min = b; max = a; }
}

inline int Clamp( const int& x, const int& low, const int& high )
{
    if( x < low  ) { return low;  }
    if( x > high ) { return high; }

    return x;
}

inline double Clamp( const double& x, const double& low, const double& high )
{
    if( x < low  ) { return low;  }
    if( x > high ) { return high; }

    return x;
}

template <class T>
inline void Swap( T& a, T& b )
{
    T c = a;

    a = b;
    b = c;    
}

template <typename T>
inline int Sign( const T& x )
{
    return ( ( x < 0 ) ? -1 : ( x > 0 ) ? 1 : 0 );
}

template <typename T>
inline bool AlmostZero( const T& x, const T eps=EPSILON )
{
    return ( ( ABS(x) <= eps ) ? true : false );
}

template <typename T>
inline bool AlmostSame( const T& a, const T& b, const T eps=EPSILON )
{
    return ( ( ABS(a-b) <= eps ) ? true : false );
}

BS_NAMESPACE_END

#endif

