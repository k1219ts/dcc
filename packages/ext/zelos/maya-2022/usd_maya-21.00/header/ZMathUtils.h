//--------------//
// ZMathUtils.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.09.04                               //
//-------------------------------------------------------//

#ifndef _ZMathUtils_h_
#define _ZMathUtils_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

// Is finite number or not?
#define ZFinite(x)				(((x)-(x))==0)

// Is not a number of not?
#define ZNan(x)					((!((x)<0))&&(!((x)>=0)))

// Is infinite number or not?
#define ZInfinite(x)			(!ZNan(x)&&ZNan(x-x))

// Is odd number or not?
#define ZOdd(x)					((x)&1)

// Is even number or not?
#define ZEven(x)				(!((x)&1))

// = |x|
#define ZAbs(x)					(((x)<0)?-(x):(x))

// This function returns the input x clamped to the range [min,max].
// This function is useful to constrain a value to a certain range.
#define ZClamp(x,min,max)		(((x)<(min))?((min)):(((x)>(max))?((max)):(x)))

// This function returns the absolute input clamped to the range [0,1].
// It is a special case of the ZClamp function.
#define ZSaturate(x)			(((x)<0)?(0):(((x)>1)?(1):(x)))

// This function converts the input x from radians to degrees.
#define ZRadToDeg(x)			((x)*Z_RADtoDEG)

// This function converts the input x from degrees to radians.
#define ZDegToRad(x)			((x)*Z_DEGtoRAD)

// The nearest integer not less than x.
#define ZCeil(x)				((int)(x)+(((x)>0)&&((x)!=(int)(x))))
//#define ZCeil(x)				std::ceil(x)

// The nearest integer not greater than x.
#define ZFloor(x)				((int)(x)-(((x)<0)&&((x)!=(int)(x))))
//#define ZFloor(x)				std::floor(x)

// The nearest integer not greater in magnitude than x.
#define ZTrunc(x)				((int)(x))

// The nearest integer to arg. Number is rounded away from zero in halfway cases.
#define ZRound(x)				((x)>0?(int)((x)+.5):-(int)(.5-(x)))

// Returns the fractional part of the input x such that the result will be between zero and one.
#define ZFrac(x)				((x)-(int)(x))

// = (x>=y) ? 1 : 0.
#define ZStep(y,x)				(((x)>=(y))?1:0)

// Return -1 if the input is negative and +1 if the input is positive.
#define ZSgn(x)					(((x)>0)?+1:-1)

// Return -1 if the input is negative, +1 if the input is positive, and 0 if the input is zero.
#define ZSign(x)				(((x)<0)?-1:((x)>0)?1:0)

//////////////////////////////////////////////////////////////
// Caution: Do not write below functions as macro function. //
//////////////////////////////////////////////////////////////

// = remainder of a/b
template <typename T>
inline T
ZRemainder( T a, T b )
{
	const int quotient = (int)(a/b);
	return ( a - ( quotient * b ) );
}

////////////////
// comparison //

// Is exactly zero or not?
template <class T>
inline bool ZIsZero( const T& v )
{
	const int size = sizeof(T);
	if( !size ) { return false; }

	char* x = (char*)&v;

	FOR( i, 0, size )
	{
		if( x[i] )
		{
			return false;
		}
	}

	return true;
}

// Is almost zero or not?
template <typename T>
inline bool ZAlmostZero( const T& x, const T eps=Z_EPS )
{
	return ( ( ZAbs(x) <= eps ) ? true : false );
}

// Are almost same or not?
template <typename T>
inline bool ZAlmostSame( const T& a, const T& b, const T eps=Z_EPS )
{
	return ( ( ZAbs(a-b) <= eps ) ? true : false );
}

template <typename T>
inline bool ZInside( const T& x, const T& low, const T& high )
{
	if( x < low  ) { return false; }
	if( x > high ) { return false; }
	return true;
}

template <typename T>
inline bool ZOutside( const T& x, const T& low, const T& high )
{
	if( x < low  ) { return true; }
	if( x > high ) { return true; }
	return false;
}

/////////////////////////////
// trigonometric functions //

inline float  Zsqrt ( const float&  x ) { return sqrtf (x); }
inline double Zsqrt ( const double& x ) { return sqrt  (x); }

inline float  Zsin  ( const float&  x ) { return sinf  (x); }
inline double Zsin  ( const double& x ) { return sin   (x); }

inline float  Zcos  ( const float&  x ) { return cosf  (x); }
inline double Zcos  ( const double& x ) { return cos   (x); }

inline float  Ztan  ( const float&  x ) { return tanf  (x); }
inline double Ztan  ( const double& x ) { return tan   (x); }

inline float  Zasin ( const float&  x ) { return asinf (x); }
inline double Zasin ( const double& x ) { return asin  (x); }

inline float  Zacos ( const float&  x ) { return acosf (x); }
inline double Zacos ( const double& x ) { return acos  (x); }

inline float  Zatan ( const float&  x ) { return atanf (x); }
inline double Zatan ( const double& x ) { return atan  (x); }

///////////////////////
// minimum / maximum //

template <typename T>
inline T ZMin( const T& a, const T& b )
{
	return ( (a<b) ? a : b );
}

template <typename T>
inline T ZMin( const T& a, const T& b, const T& c )
{
	return ( (a<b) ? ( (a<c) ? a : c ) : ( (b<c) ? b : c ) );
}

template <typename T>
inline T ZMin( const T& a, const T& b, const T& c, const T& d )
{
	return ( (a<b) ? ( (a<c) ? ( (a<d) ? a : d ) : ( (c<d) ? c : d ) ) : ( (b<c) ? ( (b<d) ? b : d ) : ( (c<d) ? c : d ) ) );
}

template <typename T>
inline T ZMax( const T& a, const T& b )
{
	return ( (a>b) ? a : b );
}

template <typename T>
inline T ZMax( const T& a, const T& b, const T& c )
{
	return ( (a>b) ? ( (a>c) ? a : c ) : ( (b>c) ? b : c ) );
}

template <typename T>
inline T ZMax( const T& a, const T& b, const T& c, const T& d )
{
	return ( (a>b) ? ( (a>c) ? ( (a>d) ? a : d ) : ( (c>d) ? c : d ) ) : ( (b>c) ? ( (b>d) ? b : d ) : ( (c>d) ? c : d ) ) );
}

template <typename T>
inline T ZAbsMin( const T& a, const T& b )
{
	const T A = ZAbs(a);
	const T B = ZAbs(b);

	return ( (A<B) ? A : B );
}

template <typename T>
inline T ZAbsMin( const T& a, const T& b, const T& c )
{
	const T A = ZAbs(a);
	const T B = ZAbs(b);
	const T C = ZAbs(c);

	return ( (A<B) ? ( (A<C) ? A : C ) : ( (B<C) ? B : C ) );
}

template <typename T>
inline T ZAbsMin( const T& a, const T& b, const T& c, const T& d )
{
	const T A = ZAbs(a);
	const T B = ZAbs(b);
	const T C = ZAbs(c);
	const T D = ZAbs(d);

	return ( (A<B) ? ( (A<C) ? ( (A<D) ? A : D ) : ( (C<D) ? C : D ) ) : ( (B<C) ? ( (B<D) ? B : D ) : ( (C<D) ? C : D ) ) );
}

template <typename T>
inline T ZAbsMax( const T& a, const T& b )
{
	const T A = ZAbs(a);
	const T B = ZAbs(b);

	return ( (A>B) ? A : B );
}

template <typename T>
inline T ZAbsMax( const T& a, const T& b, const T& c )
{
	const T A = ZAbs(a);
	const T B = ZAbs(b);
	const T C = ZAbs(c);

	return ( (A>B) ? ( (A>C) ? A : C ) : ( (B>C) ? B : C ) );
}

template <typename T>
inline T ZAbsMax( const T& a, const T& b, const T& c, const T& d )
{
	const T A = ZAbs(a);
	const T B = ZAbs(b);
	const T C = ZAbs(c);
	const T D = ZAbs(d);

	return ( (A>B) ? ( (A>C) ? ( (A>D) ? A : D ) : ( (C>D) ? C : D ) ) : ( (B>C) ? ( (B>D) ? B : D ) : ( (C>D) ? C : D ) ) );
}

template <typename T>
inline void ZMinMax( const T& a, const T& b, T& min, T& max )
{
	if( a < b ) { min = a; max = b; }
	else        { min = b; max = a; }
}

template <typename T>
inline void ZMinMax( const T& a, const T& b, const T& c, T& min, T& max )
{
	min = max = a;

	if( b < min ) { min = b; }
	if( b > max ) { max = b; }

	if( c < min ) { min = c; }
	if( c > max ) { max = c; }
}

template <typename T>
inline void ZMinMax( const T& a, const T& b, const T& c, const T& d, T& min, T& max )
{
	min = max = a;

	if( b < min ) { min = b; }
	if( b > max ) { max = b; }

	if( c < min ) { min = c; }
	if( c > max ) { max = c; }

	if( d < min ) { min = d; }
	if( d > max ) { max = d; }
}

///////////
// power //

// = x^2
template <typename T>
inline T ZPow2( const T& x )
{
	return (x*x);
}

// = x^3
template <typename T>
inline T ZPow3( const T& x )
{
	return (x*x*x);
}

// = x^4
template <typename T>
inline T ZPow4( const T& x )
{
	const T xx = x*x;
	return (xx*xx);
}

// = x^5
template <typename T>
inline T ZPow5( const T& x )
{
	const T xx = x*x;
	return (xx*xx*x);
}

// = x^6
template <typename T>
inline T ZPow6( const T& x )
{
	const T xxx = x*x*x;
	return (xxx*xxx);
}

// Is power of two or not?
inline bool
ZPowersOfTwo( const int& n )
{
	return ( n&(n-1) ? false : true );
}

// Returns the n-powers of two (2^n).
inline int
ZGetPowersOfTwo( int n )
{
	if( n <= 0 ) { return -1; }
	if( !ZPowersOfTwo(n) ) { return -1; }
	int count = 0;
	while(1) { ++count; n=(int)(n/2); if(n==1) { break; } }
	return count;
}

// Returns the smallest powers of two number larger than n.
// If n=1, it returns 1(=2^0).
inline int
ZRoundUpToPowerOfTwo( int n )
{
	if( n <= 0 ) { return 1; }
	int exponent = 0;
	--n;
	while(n) { ++exponent; n>>=1; }
	return (1<<exponent);
}

// Returns the largest powers of two number smaller than n.
// If n=1, it will return 1 (=2^0).
inline int
ZRoundDownToPowerOfTwo( int n )
{
	if( n <= 0 ) { return 1; }
	int exponent = 0;
	while(n>1) { ++exponent; n>>=1; }
	return (1<<exponent);
}

// Returns the natural logarithm of the given number.
inline float
ZLogarithm( const float& x )
{
	return logf( Z_Eminus1*x + 1 );
}

// Returns the given power of the natural logarithm of the given number.
inline float
ZLogarithm( const float& x, const float& power )
{
	return powf( logf( Z_Eminus1*x + 1 ), power );
}

//////////////////
// modification //

// Returns the mapping value by the comma curve.
// g: The gamma value. A value less than 1 will push the input value down and one greater than 1 will pull the input value up.
inline float  ZGamma( const float&  g, const float&  x ) { return powf( x, 1.f/g ); }
inline double ZGamma( const double& g, const double& x ) { return pow ( x, 1.0/g ); }

// Reeturns the mapping value by the bias curve.
// b: The bias value. A value increases or decreases the bias of the given value. 0.5 is no change.
inline float  ZBias( const float&  b, const float&  x ) { return powf( x, logf(b)/logf(0.5f) ); }
inline double ZBias( const double& b, const double& x ) { return pow ( x, log (b)/log (0.50) ); }

// Returns the mapping value by the gain curve.
// g: The gain value. A value increases or decreases the gain of the given value. 0.5f is no change.
inline float  ZGain( const float&  g, const float&  x ) { return ( (x<0.5f) ? (0.5f*ZBias(1.f-g,2.f*x)) : (1.f-0.5f*ZBias(1.f-g,2.f-2.f*x)) ); }
inline double ZGain( const double& g, const double& x ) { return ( (x<0.50) ? (0.50*ZBias(1.0-g,2.0*x)) : (1.0-0.50*ZBias(1.0-g,2.0-2.0*x)) ); }

///////////////////
// interpolation //

template <class T>
inline T
ZLerp( const T& a, const T& b, const float& t )
{
	if( t <= 0 ) { return a; }
	if( t >= 1 ) { return b; }

	return ( (1-t)*a + t*b );
}

// Return a value between 0 and 1 that represents the relationship of the input value to the min and max values.
template <typename T>
inline T
ZSmoothStep( const T& x, const T& xMin=0, const T& xMax=1 )
{
	if( x < xMin ) { return 0; }
	if( x > xMax ) { return 1; }

	const T t = (x-xMin) / (xMax-xMin); // Normalize x.
	return ( t*(t*(-2*t+3)) );
}

// Returns a value between 0 and 1 that represents the relationship of the input value to the min and max values.
template <typename T>
inline T
ZFade( const T& x, const T& xMin=0, const T& xMax=1 )
{
	if( x < xMin ) { return 0; }
	if( x > xMax ) { return 1; }

	const T t = (x-xMin) / (xMax-xMin);
	return ( t*t*t*(t*(t*6-15)+10) );
}

// Gaussian-like function.
// Return a value between 0 and 1 that represents the relationship of the input values.
template <typename T>
inline T
ZBellShape( const T& x, const T& xMin, const T& xMax )
{
	const T mid = (T)0.5 * ( xMin + xMax );
	return ( (x<mid)?(ZSmoothStep(x,xMin,mid)):((T)1-ZSmoothStep(x,mid,xMax)) );
}

// Flattened Gaussian-like function.
// Return a value between 0 and 1 that represents the relationship of the input values.
template <typename T>
inline T
ZHillShape( const T& x, const T& x0, const T& x1, const T& x2, const T& x3 )
{
	if( x < x1 ) { return ZSmoothStep( x, x0, x1 ); }
	if( x > x2 ) { return (T)1 - ZSmoothStep( x, x2, x3 ); }
	return (T)1;
}

template <class T>
inline T
ZCatRom( const T& P0, const T& P1, const T& P2, const T& P3, const float& t )
{
	const float tt  = t*t;
	const float ttt = tt*t;

	T tmp = (-t+2*tt-ttt)*P0;
	tmp += (2-5*tt+3*ttt)*P1;
	tmp += (t+4*tt-3*ttt)*P2;
	tmp +=      (-tt+ttt)*P3;

	return (tmp*=0.5f);
}
//{
//	return 0.5f * ( (       2*P1           )
//				  + ( -1*P0     +1*P2      ) * t
//				  + (  2*P0-5*P1+4*P2-1*P3 ) * tt
//				  + ( -1*P0+3*P1-3*P2+1*P3 ) * ttt );
//}

template <class T>
inline T
ZMCerp( const T& v0, const T& v1, const T& v2, const T& v3, const float& f )
{
	T d1 = (v2 - v0) / 2;
	T d2 = (v3 - v1) / 2;
	T D1 = v2 - v1;

	if( fabs(D1) < Z_EPS ||
		D1 * d1 < 0.0 ||
		D1 * d2 < 0.0 )
	{
		d1 = d2 = 0;
	}
	
	T a3 = d1 + d2 - 2 * D1;
	T a2 = 3 * D1 - 2 * d1 - d2;
	T a1 = d1;
	T a0 = v1;

	return ( (a3*f*f*f) + (a2*f*f) + (a1*f) + a0 );
}

// Fits the given value from on range to another.
// It the given value is outside the old range, it wiil be clamped to the new range.
// Returns a value between newMin and newMax that is relative to value in the range between oldMin and oldMax.
template <typename T>
inline T
ZFit( const T& oldValue, const T& oldMin, const T& oldMax, const T& newMin, const T& newMax )
{
	if( oldValue < oldMin ) { return newMin; }
	if( oldValue > oldMax ) { return newMax; }

	return ( (oldValue-oldMin)*((newMax-newMin)/(oldMax-oldMin)) + newMin );
}

// unit-height zero-mean Gaussian function
// mu: mean
// deviation: sigma (sigma^2: variance)
inline float
ZGaussian( const float& x, const float& mu, const float& sigma )
{
	//return ( (sigma/sqrtf(Z_PIx2)) * expf(-0.5f*ZPow2((x-mu)/sigma)) );
	return ( (sigma*0.398942291736602783203125f) * expf(-0.5f*ZPow2((x-mu)/sigma)) );
}

inline double
ZGaussian( const double& x, const double& mu, const double& sigma )
{
	//return ( (sigma/sqrtf(Z_PIx2)) * expf(-0.5f*ZPow2((x-mu)/sigma)) );
	return ( (sigma*0.398942291736602783203125) * exp(-0.5*ZPow2((x-mu)/sigma)) );
}


// diff = x - mu
inline float
ZGaussian( const float& diff, const float& sigma )
{
	//return ( (sigma/sqrtf(Z_PIx2)) * expf(-0.5f*ZPow2((x-mu)/sigma)) );
	return ( (sigma*0.398942291736602783203125f) * expf(-0.5f*ZPow2(diff/sigma)) );
}

inline double
ZGaussian( const double& diff, const double& sigma )
{
	//return ( (sigma/sqrtf(Z_PIx2)) * expf(-0.5f*ZPow2((x-mu)/sigma)) );
	return ( (sigma*0.398942291736602783203125) * exp(-0.5*ZPow2(diff/sigma)) );
}

template <typename T>
inline void
ZSort( T& a, T& b, bool increasingOrder=true )
{
	if( increasingOrder )
    {
		if( a > b ) { ZSwap( a, b ); }
	}
    else
    {
		if( a < b ) { ZSwap( a, b ); }
	}
}

template <typename T>
inline void
ZSort( T& a, T& b, T& c, bool increasingOrder=true )
{
	if( increasingOrder )
    {
		if( a > b ) { ZSwap( a, b ); }
		if( a > c ) { ZSwap( a, c ); }
		if( b > c ) { ZSwap( b, c ); }
	}
    else
    {
		if( a < b ) { ZSwap( a, b ); }
		if( a < c ) { ZSwap( a, c ); }
		if( b < c ) { ZSwap( b, c ); }
	}
}

inline float
ZFastInvSqrt( float x )
{
	const float xHalf = 0.5f * x;
	int i = *(int*)&x;			// get bits for floating value
	i = 0x5f3759df - (i>>1);	// gives initial guess (y0)
	x = *(float*)&i;			// convert bits back to float
	x = x*(1.5f-xHalf*x*x);		// Newton step, repeating increases accuracy
	//x = x*(1.5f-xHalf*x*x);	// 2nd iteration, this can be removed;w

	return x;
}

// angle -> [0, 360) or [0, 2pi)
template <typename T>
inline T
ZFitTo360Range( const T& angle, bool asDegrees=true )
{
    const T max = asDegrees ? 360 : Z_PIx2;

    T a = angle;

    if( a > 0 )
    {
        while( a > max )
        {
            a -= max;
        }
    }
    else if( a < 0 )
    {
        while( a < 0 )
        {
            a += max;
        }
    }

    return a;
}

template <typename FUNC>
float NumericalQuadrature( FUNC f, const float a, const float b, const int n=100 )
{
    const float dx = ( b - a ) / (float)n;

    float sum = 0.f;

    for( int i=1; i<n; ++i )
    {
        const float x = a + i*dx;
        sum += f( x );
    }

    return ( dx * ( 0.5f*f(a) + sum + 0.5f*f(b) ) );
}

// the length of the hypotenuse of a right-angle triangle
inline float  Hypot( const float&  x, const float&  y ) { return hypotf(x,y); }
inline double Hypot( const double& x, const double& y ) { return hypot(x,y);  }

// square root
inline float  Sqrt( const float&  x ) { return sqrtf(x); }
inline double Sqrt( const double& x ) { return sqrt(x);  }

// exponential
inline float  Exp( const float&  x ) { return expf(x); }
inline double Exp( const double& x ) { return exp(x);  }

// logarithm
inline float  Log( const float&  x ) { return logf(x); }
inline double Log( const double& x ) { return log(x);  }

// power
inline float  Pow( const float&  x, const float&  y ) { return powf(x,y); }
inline double Pow( const double& x, const double& y ) { return pow(x,y);  }

// gamma
inline float  TGamma( const float&  x ) { return tgammaf(x); }
inline double TGamma( const double& x ) { return tgamma(x);  }

// floor
inline float  Floor( const float&  x ) { return floorf(x); }
inline double Floor( const double& x ) { return floor(x);  }

// ceil
inline float  Ceil( const float&  x ) { return ceilf(x); }
inline double Ceil( const double& x ) { return ceil(x);  }

// round
inline float  Round( const float&  x ) { return roundf(x); }
inline double Round( const double& x ) { return round(x);  }

// sine
inline float  Sin( const float&  x ) { return sinf(x); }
inline double Sin( const double& x ) { return sin(x);  }

// cosine
inline float  Cos( const float&  x ) { return cosf(x); }
inline double Cos( const double& x ) { return cos(x);  }

// tangent
inline float  Tan( const float&  x ) { return tanf(x); }
inline double Tan( const double& x ) { return tan(x);  }

// hyperbolic sine
inline float  Sinh( const float&  x ) { return sinhf(x); }
inline double Sinh( const double& x ) { return sinh(x);  }

// hyperbolic cosine
inline float  Cosh( const float&  x ) { return coshf(x); }
inline double Cosh( const double& x ) { return cosh(x);  }

// hyperbolic tangent
inline float  Tanh( const float&  x ) { return tanhf(x); }
inline double Tanh( const double& x ) { return tanh(x);  }

// hyperbolic cosecant
inline float  Csch( const float&  x ) { return 1/(sinhf(x)+Z_EPS); }
inline double Csch( const double& x ) { return 1/(sinh(x)+Z_EPS);  }

// hyperbolic secant
inline float  Sech( const float&  x ) { return 1/(coshf(x)+Z_EPS); }
inline double Sech( const double& x ) { return 1/(cosh(x)+Z_EPS);  }

// hyperblic cotangent
inline float  Coth( const float&  x ) { return coshf(x)/(sinhf(x)+Z_EPS); }
inline double Coth( const double& x ) { return cosh(x)/(sinh(x)+Z_EPS);   }

// arcsine: sin^-1(x) (inverse sine)
inline float  ASin( const float&  x ) { return asinf(ZClamp(x,-1.f,1.f)); }
inline double ASin( const double& x ) { return asin(ZClamp(x,-1.0,1.0));  }

// arccosine: cos^-1(x) (inverse cosine)
inline float  ACos( const float&  x ) { return acosf(ZClamp(x,-1.f,1.f)); }
inline double ACos( const double& x ) { return acos(ZClamp(x,-1.0,1.0));  }

// arctangent: tan^-1(x) (inverse tangent)
inline float  ATan( const float&  x ) { return atanf(x); }
inline double ATan( const double& x ) { return atan(x);  }

// 1/tan(y/x)
inline float  ATan2( const float&  y, const float&  x ) { return atan2f(y,x+Z_EPS); }
inline double ATan2( const double& y, const double& x ) { return atan2(y,x+Z_EPS);  }

ZELOS_NAMESPACE_END

#endif

