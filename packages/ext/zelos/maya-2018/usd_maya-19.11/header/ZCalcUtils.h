//--------------//
// ZCalcUtils.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2019.03.26                               //
//-------------------------------------------------------//

#ifndef _ZCalcUtils_h_
#define _ZCalcUtils_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

/////////////////////
// short type name //

#define  ZP       ZPoint
#define cZP const ZPoint

#define  ZV       ZVector
#define cZV const ZVector

/////////////////////
// macro operators //

// ex) ZOP1(A,+=,B)
#define ZOP1(A,op,B)\
((A.x) op (B.x));\
((A.y) op (B.y));\
((A.z) op (B.z));

// ex) ZOP2(A,+=,B,+,C)
#define ZOP2(A,op1,B,op2,C)\
((A.x op1 B.x op2 C.x));\
((A.y op1 B.y op2 C.y));\
((A.z op1 B.z op2 C.z));

// ex) ZOP3(A,+=,b,*,B)
#define ZOP3(A,op1,b,op2,B)\
((A.x op1 b op2 B.x));\
((A.y op1 b op2 B.y));\
((A.z op1 b op2 B.z));

////////////////////
// transformation //

inline ZPoint
ZSphericalToCartesian( const float& azimuth, const float& elevation, const float& radius )
{
	return ZPoint( radius*cosf(elevation)*cosf(azimuth), radius*cosf(elevation)*sinf(azimuth), radius*sinf(elevation) );
}

////////////////////////////
// reflection, refraction //

// surfNormal must be a unit vector.
inline ZVector
Reflect( cZV& t, cZV& n )
{
	ZVector nn( n );
	if( (t*n) > 0 ) { nn.negate(); }
	return (t-nn*(2*(nn*t)));
}

// Heckbert's method.
// surfNormal must be a unit vector.
inline ZVector
Refract( cZV& t, cZV& n, float inside, float outside, bool& totalRefraction )
{
	const float eta = outside / inside;
	const float tn = -(t*n);
	float tt = 1 + ZPow2(eta) * ( ZPow2(tn) - 1 );
	if( tt < 0 ) {	// total  reflection
		totalRefraction = true;
    	return t;
	} else {		// normal reflection
		totalRefraction = false;
        tt = eta*tn - sqrtf( tt );
        return ( t*eta + n*tt );
    }
}

inline float
ZFresnel( float angle, float n1, float n2 )
{
	float r = 1;
	float a = (n1/n2)*sinf(angle);
	a *= a;

	if ( a <= 1 )
	{
		const float b = n2*sqrtf(1-a);
		const float c = n1*cosf(angle);

		r =  ( c - b ) / ( c + b );
		r *= r;
		r = ZMin( 1.f, r );
	}

	return r;
}

/////////////////
// = a*A + b*B //

inline float
ZWeightedSum( float A, float B, float a, float b )
{
	return ( a*A + b*B );
}

inline float
ZWeightedSum( float A, float B, const ZFloat2& coeff )
{
	const float &a=coeff[0], &b=coeff[1];
	return ( a*A + b*B );
}

inline ZVector
WeightedSum( cZV& A, cZV& B, float a, float b )
{
	return ZVector( (a*A.x)+(b*B.x), (a*A.y)+(b*B.y), (a*A.z)+(b*B.z) );
}

inline ZVector
WeightedSum( cZV& A, cZV& B, const ZFloat2& coeff )
{
	const float &a=coeff[0], &b=coeff[1];
	return ZVector( (a*A.x)+(b*B.x), (a*A.y)+(b*B.y), (a*A.z)+(b*B.z) );
}

///////////////////////
// = a*A + b*B + c*C //

inline float
ZWeightedSum( float A, float B, float C, float a, float b, float c )
{
	return ( a*A + b*B + c*C );
}

inline float
ZWeightedSum( float A, float B, float C, const ZFloat3& coeff )
{
	const float &a=coeff[0], &b=coeff[1], &c=coeff[2];
	return ( a*A + b*B + c*C );
}

inline ZVector
WeightedSum( cZV& A, cZV& B, cZV& C, float a, float b, float c )
{
	return ZVector( (a*A.x)+(b*B.x)+(c*C.x), (a*A.y)+(b*B.y)+(c*C.y), (a*A.z)+(b*B.z)+(c*C.z) );
}

inline ZVector
WeightedSum( cZV& A, cZV& B, cZV& C, const ZFloat3& coeff )
{
	const float &a=coeff[0], &b=coeff[1], &c=coeff[2];
	return ZVector( (a*A.x)+(b*B.x)+(c*C.x), (a*A.y)+(b*B.y)+(c*C.y), (a*A.z)+(b*B.z)+(c*C.z) );
}

inline ZAxis
WeightedSum( const ZAxis& A, const ZAxis& B, const ZAxis& C, float a, float b, float c )
{
	ZAxis R;
	R.origin = WeightedSum( A.origin, B.origin, C.origin, a, b, c );
	R.xAxis  = WeightedSum( A.xAxis,  B.xAxis,  C.xAxis,  a, b, c );
	R.yAxis  = WeightedSum( A.yAxis,  B.yAxis,  C.yAxis,  a, b, c );
	R.zAxis  = WeightedSum( A.zAxis,  B.zAxis,  C.zAxis,  a, b, c );
	R.normalize( true );
	return R;
}

inline ZAxis
WeightedSum( const ZAxis& A, const ZAxis& B, const ZAxis& C, const ZFloat3& coeff )
{
	const float &a=coeff[0], &b=coeff[1], &c=coeff[2];
	ZAxis R;
	R.origin = WeightedSum( A.origin, B.origin, C.origin, a, b, c );
	R.xAxis  = WeightedSum( A.xAxis,  B.xAxis,  C.xAxis,  a, b, c );
	R.yAxis  = WeightedSum( A.yAxis,  B.yAxis,  C.yAxis,  a, b, c );
	R.zAxis  = WeightedSum( A.zAxis,  B.zAxis,  C.zAxis,  a, b, c );
	R.normalize( true );
	return R;
}

/////////////////////////////
// = a*A + b*B + c*C + d*D //

inline float
ZWeightedSum( float A, float B, float C, float D, float a, float b, float c, float d )
{
	return ( a*A + b*B + c*C + d*D );
}

inline float
ZWeightedSum( float A, float B, float C, float D, const ZFloat4& coeff )
{
	const float &a=coeff[0], &b=coeff[1], &c=coeff[2], &d=coeff[3];
	return ( a*A + b*B + c*C + d*D );
}

inline ZVector
WeightedSum( cZV& A, cZV& B, cZV& C, cZV& D, float a, float b, float c, float d )
{
	return ZVector( (a*A.x)+(b*B.x)+(c*C.x)+(d*D.x), (a*A.y)+(b*B.y)+(c*C.y)+(d*D.y), (a*A.z)+(b*B.z)+(c*C.z)+(d*D.z) );
}

inline ZVector
WeightedSum( cZV& A, cZV& B, cZV& C, cZV& D, const ZFloat4& coeff )
{
	const float &a=coeff[0], &b=coeff[1], &c=coeff[2], &d=coeff[3];
	return ZVector( (a*A.x)+(b*B.x)+(c*C.x)+(d*D.x), (a*A.y)+(b*B.y)+(c*C.y)+(d*D.y), (a*A.z)+(b*B.z)+(c*C.z)+(d*D.z) );
}

/////////////////////
// center position //

inline ZPoint
Center( cZP& A, cZP& B )
{
	return ZPoint( 0.5f*(A.x+B.x), 0.5f*(A.y+B.y), 0.5f*(A.z+B.z) );
}

inline ZPoint
Center( cZP& A, cZP& B, cZP& C )
{
	return ZPoint((A.x+B.x+C.x)/3.f, (A.y+B.y+C.y)/3.f, (A.z+B.z+C.z)/3.f );
}

inline ZPoint
Center( cZP& A, cZP& B, cZP& C, cZP& D )
{
	return ZPoint( 0.25f*(A.x+B.x+C.x+D.x), 0.25f*(A.y+B.y+C.y+D.y), 0.25f*(A.z+B.z+C.z+D.z) );
}

///////////////////
// angle, volume //

// A, B: non-zero, unit length (normalized)
inline float
Angle( cZV& A, cZV& B )
{
	return atan2f( (A^B).length(), A*B );
	//return acosf( ZClamp( A*B, -1.f, 1.f ) ); // atan() is more accurate than acos() for small angles
}

// the volume of a tetrahedron ABCD
inline float
Volume( cZP& A, cZP& B, cZP& C, cZP& D )
{
	const ZVector N( (B-A)^(C-A) );
	return ZAbs( (N*(D-A)) / 6.f );
}

//////////////
// rotation //

// for point
inline ZPoint
Rotate( cZP& p, cZV& unitAxis, float angleInRadians, cZP& pivot )
{
	ZPoint P( p - pivot );

	const float c = cosf( angleInRadians );
	const float s = sinf( angleInRadians );

	const ZVector cross( unitAxis ^ P );
	const float dot = unitAxis * P;

	const float alpha = (1-c) * dot;

	P.x = c*P.x + s*cross.x + alpha*unitAxis.x + pivot.x;
	P.y = c*P.y + s*cross.y + alpha*unitAxis.y + pivot.y;
	P.z = c*P.z + s*cross.z + alpha*unitAxis.z + pivot.z;

	return P;
}

inline ZPoint
RotateOnY( cZP& p, float angleInRadians, cZP& pivot )
{
	ZPoint P;

	const float c = cosf( angleInRadians );
	const float s = sinf( angleInRadians );

	P.x = c*(p.x-pivot.x) - s*(p.z-pivot.z) + pivot.x;
	P.y = p.y;
	P.z = s*(p.x-pivot.x) + s*(p.z-pivot.z) + pivot.z;

	return P;
}

inline ZPoint
RotateOnZ( cZP& p, float angleInRadians, cZP& pivot )
{
	ZPoint P;

	const float c = cosf( angleInRadians );
	const float s = sinf( angleInRadians );

	P.x = c*(p.x-pivot.x) - s*(p.y-pivot.y) + pivot.x;
	P.y = s*(p.x-pivot.x) + s*(p.y-pivot.y) + pivot.y;
	P.z = p.z;

	return P;
}

// for vector
inline ZVector
Rotate( cZV& v, cZV& unitAxis, float angleInRadians )
{
	ZVector V( v );

	const float c = cosf( angleInRadians );
	const float s = sinf( angleInRadians );

	const ZVector cross( unitAxis ^ V );
	const float dot = unitAxis * V;

	const float alpha = (1-c) * dot;

	V.x = c*V.x + s*cross.x + alpha*unitAxis.x;
	V.y = c*V.y + s*cross.y + alpha*unitAxis.y;
	V.z = c*V.z + s*cross.z + alpha*unitAxis.z;

	return V;
}

//////////
// line //

inline ZPoint
ClosestPointOnLine( cZP& P, cZP& A, cZP& B, float& t )
{
	const ZVector AB(B-A);
	const float ab = AB.squaredLength();
	if( ab < Z_EPS ) { t=0.f; return A; }
	t = ZClamp( ((B-P)*AB)/ab, 0.f, 1.f );
	return WeightedSum( A,B, t,1-t );
}

inline float
DistanceFromPointToLine( cZP& P, cZP& A, cZP& B )
{
	float t = 0.f;
	return P.distanceTo( ClosestPointOnLine( P, A,B, t ) );
}

/////////
// ray //

// D: unit direction vector
inline ZPoint
ClosestPointOnRay( cZP& P, cZP& O, cZV& D, float& t )
{
	t = (P-O)*D;
	return (O+t*D);
}

inline float
DistanceFromPointToRay( cZP& P, cZP& O, cZV& D )
{
	float t = 0.f;
	return P.distanceTo( ClosestPointOnRay( P, O,D, t ) );
}

///////////
// plane //

// N: unit normal vector
inline ZPoint
ClosestPointOnPlane( cZP& P, cZP& O, cZV& N )
{
	return ( P-N*((P-O)*N) );
}

inline float
DistanceFromPointToPlane( cZP& P, cZP& O, cZV& N )
{
	return P.distanceTo( ClosestPointOnPlane( P, O,N ) );
}

inline bool
LinePlaneIntersection( cZP& A, cZP& B, cZP& O, cZV& N, ZPoint& P )
{
    const ZVector AO = O - A;
    const ZVector AB = B - A;
    const float t = (AO*N) / (AB*N);
    P = A + t * AB;
    if( t < 0.f ) { return false; }
    if( t > 1.f ) { return false; }
    return true;
}

/////////////////
// triangle 2D //

// perp dot: 2D version of cross product
// given two vectors: A(x0,y0), B(x1,y1)
//  0: A and B are parallel each other.
// -1: B is counter-clockwise from A
// +1: B is clockwise from A
#define ZPerpDot(x0,y0,x1,y1)\
((x0)*(y1)-(x1)*(y0))

inline float
PerpDot( cZP& A, cZP& B, int whichPlane )
{
	const float& x0 = (whichPlane==0) ? A.x: ( (whichPlane==1) ? A.y : A.z );
	const float& y0 = (whichPlane==0) ? A.y: ( (whichPlane==1) ? A.z : A.x );

	const float& x1 = (whichPlane==0) ? B.x: ( (whichPlane==1) ? B.y : B.z );
	const float& y1 = (whichPlane==0) ? B.y: ( (whichPlane==1) ? B.z : B.x );

	return ZPerpDot( x0,y0, x1,y1 );
}

inline float
Area( cZP& A, cZP& B, cZP& C, int whichPlane )
{
	const float& x0 = (whichPlane==0) ? A.x: ( (whichPlane==1) ? A.y : A.z );
	const float& y0 = (whichPlane==0) ? A.y: ( (whichPlane==1) ? A.z : A.x );

	const float& x1 = (whichPlane==0) ? B.x: ( (whichPlane==1) ? B.y : B.z );
	const float& y1 = (whichPlane==0) ? B.y: ( (whichPlane==1) ? B.z : B.x );

	const float& x2 = (whichPlane==0) ? C.x: ( (whichPlane==1) ? C.y : C.z );
	const float& y2 = (whichPlane==0) ? C.y: ( (whichPlane==1) ? C.z : C.x );

	return ( 0.5f * ZPerpDot( x1-x0, y1-y0, x2-x0, y2-y0 ) );
}

// just inside-outside test without returning barycentric coordinates
inline bool
IsPointInsideTriangle( cZP& P, cZP& A, cZP& B, cZP& C, int whichPlane, float e=Z_EPS )
{
	const ZPoint PP( P.x+e, P.y+e, P.z+e ); // to diminish numerical error
	// When using P and it is on the sharing edge of two neighbor triangles,
	// this function may return false for both triangles.

	const float c0 = PerpDot( B-A, PP-A, whichPlane );
	const float c1 = PerpDot( C-B, PP-B, whichPlane );
	const float c2 = PerpDot( A-C, PP-C, whichPlane );

	return ( ( c0>0 && c1>0 && c2>0 ) || ( c0<0 && c1<0 && c2<0 ) );
}

inline bool
IsInsideTriangle( const ZFloat3& baryCoords, float e=Z_EPS )
{
	const float& s = baryCoords[0];
	const float& t = baryCoords[1];

	if( s < -e ) { return false; }
	if( t < -e ) { return false; }
	if( (s+t) > (1+e) ) { return false; }

	return true;
}

// -1: the triangle is singular
//  0: P is outside of ABC
// +1: P is inside of ABC
// P = WeightedSum( A,B,C, baryCoords )
inline int
BaryCoords( cZP& P, cZP& A, cZP& B, cZP& C, int whichPlane, ZFloat3& baryCoords, float e=Z_EPS )
{
	const float& x  = (whichPlane==0) ? P.x: ( (whichPlane==1) ? P.y : P.z );
	const float& y  = (whichPlane==0) ? P.y: ( (whichPlane==1) ? P.z : P.x );

	const float& x0 = (whichPlane==0) ? A.x: ( (whichPlane==1) ? A.y : A.z );
	const float& y0 = (whichPlane==0) ? A.y: ( (whichPlane==1) ? A.z : A.x );

	const float& x1 = (whichPlane==0) ? B.x: ( (whichPlane==1) ? B.y : B.z );
	const float& y1 = (whichPlane==0) ? B.y: ( (whichPlane==1) ? B.z : B.x );

	const float& x2 = (whichPlane==0) ? C.x: ( (whichPlane==1) ? C.y : C.z );
	const float& y2 = (whichPlane==0) ? C.y: ( (whichPlane==1) ? C.z : C.x );

	const float denom = (y1-y2)*(x0-x2)+(x2-x1)*(y0-y2);

	if( ZAlmostZero( denom, e ) )
	{
		baryCoords.zeroize();
		return -1;
	}

	const float& s = baryCoords[0] = ( (y1-y2)*(x-x2)+(x2-x1)*(y-y2) ) / denom;
	const float& t = baryCoords[1] = ( (y2-y0)*(x-x2)+(x0-x2)*(y-y2) ) / denom;
	baryCoords[2] = 1-s-t;

	return int( IsInsideTriangle( baryCoords ) );
}

/////////////////
// triangle 3D //

// triangle normal (counter-clockwise)
inline ZVector
Normal( cZP& A, cZP& B, cZP& C, bool doNormalize=true )
{
	ZVector N( (B-A)^(C-A) );
	if( doNormalize ) { N.normalize(); }
	return N;
}

inline ZVector
RobustNormal( cZP& A, cZP& B, cZP& C )
{
	const double ABx = (double)B.x - (double)A.x;
	const double ABy = (double)B.y - (double)A.y;
	const double ABz = (double)B.z - (double)A.z;

	const double ACx = (double)C.x - (double)A.x;
	const double ACy = (double)C.y - (double)A.y;
	const double ACz = (double)C.z - (double)A.z;

	double x = ABy*ACz - ABz*ACy;
	double y = ABz*ACx - ABx*ACz;
	double z = ABx*ACy - ABy*ACx;

	const double _l = 1.0 / sqrt( x*x + y*y + z*z + 1e-20 );

	x *= _l;
	y *= _l;
	z *= _l;

	return ZVector( (float)x, (float)y, (float)z );
}

inline float
Area( cZP& A, cZP& B, cZP& C )
{
	const ZVector N( (B-A)^(C-A) );
	return ( 0.5f * N.length() );
}

inline bool
Colinear( cZP& A, cZP& B, cZP& C, float e=Z_EPS )
{
	const ZVector N( (B-A)^(C-A) );
	return ( N.squaredLength() < ZPow2(e) );
}

// -1: the triangle is singular
//  0: P is outside of ABC
// +1: P is inside of ABC
// P = WeightedSum( A,B,C, baryCoords )
inline int
BaryCoords( cZP& P, cZP& A, cZP& B, cZP& C, ZFloat3& baryCoords, float e=Z_EPS )
{
	const ZVector v0(C-A), v1(B-A), v2(P-A);
	const float d00=v0*v0, d01=v0*v1, d02=v0*v2, d11=v1*v1, d12=v1*v2;
	const float denom = (d00*d11-d01*d01);
	if( ZAlmostZero( denom, e ) )
	{
		baryCoords.zeroize();
		return -1;
	}
	const float& s = baryCoords[1] = (d00*d12-d01*d02) / denom;
	const float& t = baryCoords[2] = (d11*d02-d01*d12) / denom;
	baryCoords[0] = 1-s-t;

	return int( IsInsideTriangle( baryCoords ) );
}

// This function is similar to BaryCoords, but different.
// If P is outside the triangle ABC, the position reproduced by baryCoords is on the triangle
// while BaryCoords()'s baryCoords represents the projected point on the plane of the triangle.
inline ZPoint
ClosestPointOnTriangle( cZP& P, cZP& A, cZP& B, cZP& C, ZFloat3& baryCoords )
{
	const ZVector AB(B-A), AC(C-A), PA(A-P);
	const float a=AB*AB, b=AB*AC, c=AC*AC, d=AB*PA, e=AC*PA;
	float det=a*c-b*b, s=b*e-c*d, t=b*d-a*e;

	if( s+t < det ) {
		if( s < 0.f ) {
			if( t < 0.f ) {
				if( d < 0.f ) { s=ZClamp(-d/a,0.f,1.f); t=0.f; }
				else          { s=0.f; t=ZClamp(-e/c,0.f,1.f); }
			} else {
				s=0.f; t=ZClamp( -e/c, 0.f, 1.f );
			}
		} else if( t < 0.f ) {
			s=ZClamp(-d/a,0.f,1.f); t=0.f;
		} else {
			det=1.f/det; s*=det; t*=det;
		}
	} else {
		if( s < 0.f ) {
			const float tmp0 = b+d;
			const float tmp1 = c+e;
			if( tmp1 > tmp0 ) {
				const float numer = tmp1 - tmp0;
				const float denom = a-2*b+c;
				s = ZClamp( numer/denom, 0.f, 1.f );
				t = 1-s;
			} else {
				t = ZClamp( -e/c, 0.f, 1.f );
				s = 0.f;
			}
		} else if( t < 0.f ) {
			if( a+d > b+e ) {
				const float numer = c+e-b-d;
				const float denom = a-2*b+c;
				s = ZClamp( numer/denom, 0.f, 1.f );
				t = 1-s;
			} else {
				s = ZClamp( -e/c, 0.f, 1.f );
				t = 0.f;
			}
		} else {
			const float numer = c+e-b-d;
			const float denom = a-2*b+c;
			s = ZClamp( numer/denom, 0.f, 1.f );
			t = 1.f - s;
		}
	}

	baryCoords.set( (1-s-t), s, t );

	return ZPoint( (A.x+s*AB.x+t*AC.x), (A.y+s*AB.y+t*AC.y), (A.z+s*AB.z+t*AC.z) );
}

inline bool
RayTriangleTest( cZP& o, cZV& dir, cZP& A, cZP& B, cZP& C, ZFloat3& baryCoords, float& t, float e=Z_EPS )
{
	bool colliding = true;

	ZVector unitDir( dir.direction() );

	baryCoords.zeroize();
	t = 0;

	const ZVector AB(B-A), AC(C-A), s1(unitDir^AC);
	float divisor = s1 * AB;
	if( ZAlmostZero(divisor,e) ) { colliding = false; }
	divisor = 1 / divisor;

	ZVector d( o-A );
	float b1 = ( d * s1 ) * divisor;
	if( (b1<-e) || (b1>(1+e)) ) { colliding = false; }

	ZVector s2( d ^ AB );
	float b2 = ( unitDir * s2 ) * divisor;
	if( (b2<-e) || (b1+b2>(1+e)) ) { colliding = false; }

	baryCoords[0] = (float)( 1 - b1 - b2 );
	baryCoords[1] = (float)b1;
	baryCoords[2] = (float)b2;

	t = (float)( (AC*s2) * divisor );

	return colliding;
}

inline bool
LineTriangleTest( cZP& P, cZP& Q, cZP& A, cZP& B, cZP& C, ZFloat3& baryCoords, float e=Z_EPS )
{
	float t = 0;

	if( !RayTriangleTest( P, Q-P, A, B, C, baryCoords, t, e ) )
	{
		return false;
	}

	if( t < Z_EPS )
	{
		return false;
	}

	if( t > P.distanceTo(Q) )
	{
		return false;
	}

	return true;
}

inline float
DistanceFromPointToTriangle( cZP& P, cZP& A, cZP& B, cZP& C )
{
	ZFloat3 baryCoords;
	return P.distanceTo( ClosestPointOnTriangle( P, A,B,C, baryCoords ) );
}

/////////////////
// tetrahedron //

inline bool
Coplanar( cZP& A, cZP& B, cZP& C, cZP& D, float tol )
{
	if( Colinear(A,B,C) ) { return true; }
	return ( Volume(A,B,C,D) < tol );
}

inline bool
IsInsideTetrahedron( const ZFloat4& baryCoords, float e=Z_EPS )
{
	const float& s = baryCoords[0];
	const float& t = baryCoords[1];
	const float& u = baryCoords[2];

	if( s < -e ) { return false; }
	if( t < -e ) { return false; }
	if( u < -e ) { return false; }
	if( (s+t+u) > (1+e) ) { return false; }

	return true;
}

// -1: the tetrahedron is singular
//  0: P is outside of ABCD
// +1: P is inside of ABCD
// P = WeightedSum( A,B,C,D, baryCoords )
inline int
BaryCoords( cZP& P, cZP& A, cZP& B, cZP& C, cZP& D, ZFloat4& baryCoords, float e=Z_EPS )
{
	const ZMatrix X( A-D, B-D, C-D );
	const ZMatrix invX( X.inversed3x3() );
	const float det = invX.det3x3();

	if( ZAlmostZero(det) )
	{
		baryCoords.zeroize();
		return -1;
	}

	const ZVector p( P - D );

	const float& a = baryCoords[0] = invX._00*p.x + invX._01*p.y + invX._02*p.z;
	const float& b = baryCoords[1] = invX._10*p.x + invX._11*p.y + invX._12*p.z;
	const float& c = baryCoords[2] = invX._20*p.x + invX._21*p.y + invX._22*p.z;
	baryCoords[3] = 1-a-b-c;

	return (int)IsInsideTetrahedron( baryCoords, e );
}

ZELOS_NAMESPACE_END

#endif

