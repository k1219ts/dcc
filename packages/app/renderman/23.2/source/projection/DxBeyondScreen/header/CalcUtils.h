#ifndef _BS_CalcUtils_h_
#define _BS_CalcUtils_h_

#include <BeyondScreen.h>

BS_NAMESPACE_BEGIN

// assumption) A and B were normalized.
inline double Angle( const Vector& A, const Vector& B )
{
	return atan2( A.cross(B).magnitude(), A.dot(B) );
}

inline double Length( const Vector& A )
{
    return A.magnitude();
}

inline double Area( const Vector& A, const Vector& B, const Vector& C )
{
	const Vector N( (B-A)^(C-A) );
	return ( 0.5 * N.magnitude() );
}
inline Vector Normalize( const Vector& A )
{
    return A.direction();
}

inline double Dot( const Vector& A, const Vector& B )
{
    return A.dot(B);
}

inline Vector Cross( const Vector& A, const Vector& B )
{
    return A.cross(B);
}

inline Vector RotatePoint( const Vector& p, const Vector& unitAxis, double radians, const Vector& pivot )
{
    Vector P( p - pivot );

    const double c = cos( radians );
    const double s = sin( radians );

    const Vector cross = unitAxis.cross( P );
    const double dot = unitAxis.dot( P );

    const double alpha = ( 1.0 - c ) * dot;

    P.x = c*P.x + s*cross.x + alpha*unitAxis.x + pivot.x;
    P.y = c*P.y + s*cross.y + alpha*unitAxis.y + pivot.y;
    P.z = c*P.z + s*cross.z + alpha*unitAxis.z + pivot.z;

    return P;
}

inline Vector RotateVector( const Vector& v, const Vector& unitAxis, double radians )
{
    Vector V( v );

    const double c = cos( radians );
    const double s = sin( radians );

    const Vector cross = unitAxis.cross( V );
    const double dot = unitAxis.dot( V );

    const double alpha = ( 1.0 - c ) * dot;

    V.x = c*V.x + s*cross.x + alpha*unitAxis.x;
    V.y = c*V.y + s*cross.y + alpha*unitAxis.y;
    V.z = c*V.z + s*cross.z + alpha*unitAxis.z;

    return V;
}

inline bool RayTriangleTest
(
    const Vector& o, const Vector& dir,
    const Vector& A, const Vector& B, const Vector& C,
    Double3& baryCoords, double& t,
    double e=EPSILON
)
{
    bool colliding = true;

    Vector unitDir( dir.direction() );

    baryCoords.zeroize();
    t = 0;

    const Vector AB(B-A), AC(C-A), s1(unitDir^AC);
    double divisor = s1 * AB;
    if( AlmostZero(divisor,e) ) { colliding = false; }
    divisor = 1 / divisor;

    Vector d( o-A );
    double b1 = ( d * s1 ) * divisor;
    if( (b1<-e) || (b1>(1+e)) ) { colliding = false; }

    Vector s2( d ^ AB );
    double b2 = ( unitDir * s2 ) * divisor;
    if( (b2<-e) || (b1+b2>(1+e)) ) { colliding = false; }

    baryCoords[0] = ( 1 - b1 - b2 );
    baryCoords[1] = b1;
    baryCoords[2] = b2;

    t = ( (AC*s2) * divisor );

    return colliding;
}

#define ZPerpDot(x0,y0,x1,y1)\
((x0)*(y1)-(x1)*(y0))

inline double PerpDot( const Vector& A, const Vector& B, int whichPlane )
{
	const double& x0 = (whichPlane==0) ? A.x: ( (whichPlane==1) ? A.y : A.z );
	const double& y0 = (whichPlane==0) ? A.y: ( (whichPlane==1) ? A.z : A.x );

	const double& x1 = (whichPlane==0) ? B.x: ( (whichPlane==1) ? B.y : B.z );
	const double& y1 = (whichPlane==0) ? B.y: ( (whichPlane==1) ? B.z : B.x );

	return ZPerpDot( x0,y0, x1,y1 );
}

// just inside-outside test without returning barycentric coordinates
inline bool IsPointInsideTriangle( const Vector& P, const Vector& A, const Vector& B, const Vector& C, int whichPlane, double e=EPSILON )
{
	const Vector PP( P.x+e, P.y+e, P.z+e ); // to diminish numerical error
	// When using P and it is on the sharing edge of two neighbor triangles,
	// this function may return false for both triangles.

	const double c0 = PerpDot( B-A, PP-A, whichPlane );
	const double c1 = PerpDot( C-B, PP-B, whichPlane );
	const double c2 = PerpDot( A-C, PP-C, whichPlane );

	return ( ( c0>0 && c1>0 && c2>0 ) || ( c0<0 && c1<0 && c2<0 ) );
}

inline bool IsInsideTriangle( const Double3& baryCoords, double e=EPSILON )
{
	const double& s = baryCoords[0];
	const double& t = baryCoords[1];

	if( s < -e ) { return false; }
	if( t < -e ) { return false; }
	if( (s+t) > (1+e) ) { return false; }

	return true;
}
// -1: the triangle is singular
//  0: P is outside of ABC
// +1: P is inside of ABC
// P = WeightedSum( A,B,C, baryCoords )
inline int BaryCoords( const Vector& P, const Vector& A, const Vector& B, const Vector& C, int whichPlane, Double3& baryCoords, double e=EPSILON )
{
	const double& x  = (whichPlane==0) ? P.x: ( (whichPlane==1) ? P.y : P.z );
	const double& y  = (whichPlane==0) ? P.y: ( (whichPlane==1) ? P.z : P.x );

	const double& x0 = (whichPlane==0) ? A.x: ( (whichPlane==1) ? A.y : A.z );
	const double& y0 = (whichPlane==0) ? A.y: ( (whichPlane==1) ? A.z : A.x );

	const double& x1 = (whichPlane==0) ? B.x: ( (whichPlane==1) ? B.y : B.z );
	const double& y1 = (whichPlane==0) ? B.y: ( (whichPlane==1) ? B.z : B.x );

	const double& x2 = (whichPlane==0) ? C.x: ( (whichPlane==1) ? C.y : C.z );
	const double& y2 = (whichPlane==0) ? C.y: ( (whichPlane==1) ? C.z : C.x );

	const double denom = (y1-y2)*(x0-x2)+(x2-x1)*(y0-y2);

	if( AlmostZero( denom, e ) )
	{
		baryCoords.zeroize();
		return -1;
	}

	const double& s = baryCoords[0] = ( (y1-y2)*(x-x2)+(x2-x1)*(y-y2) ) / denom;
	const double& t = baryCoords[1] = ( (y2-y0)*(x-x2)+(x0-x2)*(y-y2) ) / denom;
	baryCoords[2] = 1-s-t;

	return int( IsInsideTriangle( baryCoords ) );
}

inline void BaryCentricCoordinates( const Vector& P, const Vector& A, const Vector& B, const Vector& C, double& w0, double& w1, double& w2 )
{
    const double area = Area( A, B, C );

    w0 = Area( B, C, P ) / area;
    w1 = Area( C, A, P ) / area;
    w2 = Area( A, B, P ) / area;
}

inline bool PointInTriangle2D( const Vector& P, const Vector& A, const Vector& B, const Vector& C, Double3& barycentricCoordinates )
{
    const Vector AB( B - A );
    const Vector AC( C - A );

    const Vector Q( P - A );

    const double determinant = ( AB.x * AC.y ) - ( AC.x * AB.y );

    const double alpha = ( ( AC.y * Q.x ) - ( AC.x * Q.y ) ) / ( determinant + EPSILON );
    const double beta  = ( ( AB.x * Q.y ) - ( AB.y * Q.x ) ) / ( determinant + EPSILON );

    barycentricCoordinates[0] = 1 - alpha - beta;
    barycentricCoordinates[1] = alpha;
    barycentricCoordinates[2] = beta;

    return ( (beta >= 0.0) && (beta <= 1.0) && (alpha >= 0.0) && (alpha + beta <= 1.0) );
}

inline Vector WeightedSum( const Vector& A, const Vector& B, const Vector& C, const Double3& w )
{
    const double &a=w[0], &b=w[1], &c=w[2];
    return Vector( (a*A.x)+(b*B.x)+(c*C.x), (a*A.y)+(b*B.y)+(c*C.y), (a*A.z)+(b*B.z)+(c*C.z) );
}

BS_NAMESPACE_END

#endif
