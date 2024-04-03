//---------------//
// ZFoundation.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.08.25                               //
//-------------------------------------------------------//

#include <ZelosBase.h>

#define ZELOS_NAMESPACE_BEGIN namespace Zelos {
#define ZELOS_NAMESPACE_END   }

ZELOS_NAMESPACE_BEGIN

const bool Z_NONE    = (bool)0;
const bool Z_ALL     = (bool)1;

const bool Z_FALSE   = (bool)0;
const bool Z_TRUE    = (bool)1;
const bool Z_SUCCESS = (bool)1;

/// A very small floaging value
const float Z_EPS    = 1e-30f;

/// A maximum floating value which can be regarded as an infinite number
const float Z_LARGE  = 1e+30f;

#ifndef Z_INTMAX
#   ifdef INT_MAX
#       define Z_INTMAX		INT_MAX
#   else
#       define Z_INTMAX		2147483647
#   endif
#endif

#ifndef Z_UINTMAX
#   ifdef UINT_MAX
#       define Z_UINTMAX	UINT_MAX
#   else
#       define Z_UINTMAX	4294967295
#   endif
#endif

#ifndef Z_FLTMAX
#   ifdef FLT_MAX
#       define Z_FLTMAX		FLT_MAX
#   else
#       define Z_FLTMAX		3.402823466e+38f
#   endif
#endif

#ifndef Z_FLTEPS
#   ifdef FLT_EPSILON
#       define Z_FLTEPS		FLT_EPSILON
#   else
#       define Z_FLTEPS		Z_EPS
#   endif
#endif

#ifndef Z_DBLMAX
#   ifdef DBL_MAX
#       define Z_DBLMAX		DBL_MAX
#   else
#       define Z_DBLMAX		1.7976931348623158e+308
#   endif
#endif

#ifndef Z_DBLEPS
#   ifdef DBL_EPSILON
#       define Z_DBLEPS		DBL_EPSILON
#   else
#       define Z_DBLEPS		Z_EPS
#   endif
#endif

const double Z_UINTMAX_INV = 2.3283064365386962890625e-10f;				///< 1/UINT_MAX

/// the ratio of the circumference of a circle to its diameter
const float Z_PI		= 3.1415926535897931159979634685441851615906f;	///< pi
const float Z_PI_INV	= 0.3183098861837906912164442019275156781077f;	///< 1/pi
const float Z_PIx2		= 6.2831853071795862319959269370883703231812f;	///< 2*pi
const float Z_PIx3		= 9.4247779607693793479938904056325554847717f;	///< 3*pi
const float Z_PI_2		= 1.5707963267948965579989817342720925807953f;	///< pi/2
const float Z_PI_3		= 1.0471975511965976313177861811709590256214f;	///< pi/3
const float Z_PI_4		= 0.7853981633974482789994908671360462903976f;	///< pi/4
const float Z_PI_6		= 0.5235987755982988156588930905854795128107f;	///< pi/6

/// the base of the natural logarithm (Euler's number)
const float Z_E			= 2.718281828459045090795598298428f;			///< e
const float Z_Eplus1	= 3.718281828459045090795598298428f;			///< e+1
const float Z_Eminus1	= 1.718281828459045090795598298428f;			///< e-1
const float Z_Ex2		= 5.436563656918090181591196596855f;			///< 2*e
const float Z_Ex3		= 8.154845485377135716476004745346f;			///< 3*e
const float Z_E_2		= 1.359140914229522545397799149214f;			///< e/2
const float Z_E_3		= 0.906093942819681696931866099476f;			///< e/3
const float Z_E_INV		= 0.367879441171442334024277442950f;			///< 1/e
const float Z_E_SQ		= 1.648721270700128194164335582172f;			///< sqrt(e)

/// the square root of N
const float Z_SQRT2		= 1.4142135623730951454746218587388284504414f;	///< sqrt(2)
const float Z_SQRT3		= 1.7320508075688771931766041234368458390236f;	///< sqrt(3)

const float Z_RADtoDEG	= 57.2957795130823228646477218717336654663086f;	///< 180/pi: radian -> degree
const float Z_DEGtoRAD	=  0.0174532925199432954743716805978692718782f;	///< pi/180: degree -> radian

const float Z_One_Over_Three = 1.f / 3.f;
const float Z_One_Over_Five  = 1.f / 5.f;
const float Z_One_Over_Six   = 1.f / 6.f;
const float Z_One_Over_Seven = 1.f / 7.f;
const float Z_One_Over_Nine  = 1.f / 9.f;

///////////////////////
// keyboard ASC code //
#define Z_KEY_ESC 27

//////////////
// for loop //
#define FOR(i,iStart,iEnd) for( int i=iStart; i<(int)iEnd; ++i )

///////////////////
// for debugging //
inline void
ZVerify( bool x )
{
	assert( x );

	if( !x )
	{
		cout << "ERROR detected." << endl;
		//exit(0);
	}
}

/////////////////////////
// charactor utilities //
inline bool
ZIsUpper( char c )
{
	if( c < 'A' ) { return false; }
	if( c > 'Z' ) { return false; }
	return true;
}

inline bool
ZIsLower( char c )
{
	if( c < 'a' ) { return false; }
	if( c > 'z' ) { return false; }
	return true;
}

inline bool
ZIsDigit( char c )
{
	if( c < '0' ) { return false; }
	if( c > '9' ) { return false; }
	return true;
}

inline char
ZToUpper( char c )
{
	if( c < 'a' ) { return c; }
	if( c > 'z' ) { return c; }
	return (c-32);
}

inline char
ZToLower( char c )
{
	if( c < 'A' ) { return c; }
	if( c > 'Z' ) { return c; }
	return (c+32);
}

ZELOS_NAMESPACE_END

