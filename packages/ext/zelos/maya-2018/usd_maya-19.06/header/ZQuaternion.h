//---------------//
// ZQuaternion.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.10.18                               //
//-------------------------------------------------------//

// Caution) Don't use double type for quaternions!

#ifndef _ZQuaternion_h_
#define _ZQuaternion_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

/// @brief A quaternion math class.
class ZQuaternion
{
	public:

		double w;	///< The w component of the vector
		double x;	///< The x component of the vector
		double y;	///< The y component of the vector
		double z;	///< The z component of the vector

	public:

		ZQuaternion();
		ZQuaternion( const ZQuaternion& q );
		ZQuaternion( const double& W, const double& X, const double& Y, const double& Z );
		ZQuaternion( const double q[4] );
		ZQuaternion( const double& W, const ZVector& v );
		ZQuaternion( const ZVector& v );
		ZQuaternion( const ZVector& from, const ZVector& to );
		ZQuaternion( const ZMatrix& rotationMatrix );
		ZQuaternion( const ZVector& rotationAxis, const double& rotationAngle );
		ZQuaternion( const double& eulerRotationX, const double& eulerRotationY, const double& eulerRotationZ );

		// Set the quaternion which describes the arc between the two specified points on the unit sphere.
		ZQuaternion& setArcRotation( const ZPoint& from, const ZPoint& to );

		ZQuaternion& fromRotationMatrix( const ZMatrix& rotationMatrix );
		ZQuaternion& fromAxisAngle( const ZVector& rotationAxis, double rotationAngle );
		ZQuaternion& fromEulerAngle( const double& eulerRotationX, const double& eulerRotationY, const double& eulerRotationZ );

		void toRotationMatrix( ZMatrix& rot ) const;
		void toAxisAngle( ZVector& unitAxis, double& angle ) const;

		// as radians
		void toEulerAngle( double& rx, double& ry, double& rz ) const;

		void zeroize();

		double real() const;
		ZVector imaginaries() const;

		double rotationAngle() const;
		ZVector rotationAxis() const;

		ZQuaternion& operator=( const ZQuaternion& q );

		bool operator==( const ZQuaternion& q ) const;
		bool operator!=( const ZQuaternion& q ) const;

		ZQuaternion& operator+=( const ZQuaternion& q );
		ZQuaternion& operator-=( const ZQuaternion& q );

		ZQuaternion& operator*=( const ZQuaternion& q );

		ZQuaternion& operator*=( const double& s );
		ZQuaternion& operator/=( const double& s );

		ZQuaternion operator+( const ZQuaternion& q ) const;
		ZQuaternion operator-( const ZQuaternion& q ) const;

		ZQuaternion operator*( const ZQuaternion& q ) const;

		ZQuaternion operator*( const int& s ) const;
		ZQuaternion operator*( const double& s ) const;

		ZQuaternion operator/( const int& s ) const;
		ZQuaternion operator/( const double& s ) const;

		ZQuaternion operator-() const;
		ZQuaternion& negate();

		double dot( const ZQuaternion& q ) const;

		bool isEquivalent( const ZQuaternion& other, double tolerance=Z_EPS ) const;

		double length() const;
		double squaredLength() const;

		ZQuaternion& normalize();
		ZQuaternion normalized() const;

		ZQuaternion& conjugate();
		ZQuaternion conjugated() const;

		ZQuaternion& inverse();
		ZQuaternion inversed() const;

		ZQuaternion derivative( const ZVector& omega ) const;

		ZQuaternion log() const;
		ZQuaternion exp() const;

		ZPoint rotate( const ZPoint& p ) const;

		static ZQuaternion zero();
		static ZQuaternion identity();
};

inline
ZQuaternion::ZQuaternion()
: w(1.0), x(0.0), y(0.0), z(0.0)
{}

inline
ZQuaternion::ZQuaternion( const ZQuaternion& q )
: w(q.w), x(q.x), y(q.y), z(q.z)
{}

inline
ZQuaternion::ZQuaternion( const double& W, const double& X, const double& Y, const double& Z )
: w(W), x(X), y(Y), z(Z)
{}

inline
ZQuaternion::ZQuaternion( const double q[4] )
: w(q[0]), x(q[1]), y(q[2]), z(q[3])
{}

inline
ZQuaternion::ZQuaternion( const double& W, const ZVector& v )
: w(W), x(v.x), y(v.y), z(v.z)
{}

inline
ZQuaternion::ZQuaternion( const ZVector& v )
: w(0.0), x(v.x), y(v.y), z(v.z)
{}

inline
ZQuaternion::ZQuaternion( const ZVector& from, const ZVector& to )
{
	const double fromLenSq = from.squaredLength();
	const double toLenSq   = to.squaredLength();

	if( ZAlmostZero(fromLenSq) || ZAlmostZero(toLenSq) )
	{
		w = 1.0;
		x = y = z = 0.0;
		return;
	}

	const ZVector axis( from ^ to );
	const double axisLenSq = axis.squaredLength();

	double angle = asin( sqrt( axisLenSq/(fromLenSq*toLenSq ) ) );
	if( from*to < 0 ) { angle = Z_PI - angle; }

	fromAxisAngle( axis, angle );
}

inline
ZQuaternion::ZQuaternion( const ZMatrix& R )
{
	fromRotationMatrix( R );
}

inline
ZQuaternion::ZQuaternion( const ZVector& axis, const double& angle )
{
	fromAxisAngle( axis, angle );
}

inline
ZQuaternion::ZQuaternion( const double& Rx, const double& Ry, const double& Rz )
{
	fromEulerAngle( Rx, Ry, Rz );
}

inline ZQuaternion&
ZQuaternion::setArcRotation( const ZPoint& from, const ZPoint& to )
{
	w = from.x*to.x + from.y*to.y + from.z*to.z;
	x = from.y*to.z - from.z*to.y;
	y = from.z*to.x - from.x*to.z;
	z = from.x*to.y - from.y*to.x;
	return (*this);
}

inline void
ZQuaternion::zeroize()
{
	w = x = y = z = 0.0;
}

inline double
ZQuaternion::real() const
{
	return w;
}

inline ZVector
ZQuaternion::imaginaries() const
{
	return ZVector( x, y, z );
}

inline double
ZQuaternion::rotationAngle() const
{
	const double len2 = squaredLength();
	if( ZAlmostZero(len2) ) { return 0.0; }
	const double wc = ZClamp( w, -1.0, 1.0 );
	return ( 2 * acos(wc) );
}

inline ZVector
ZQuaternion::rotationAxis() const
{
	return ZVector( x, y, z ).normalize();
}

inline ZQuaternion&
ZQuaternion::operator=( const ZQuaternion& q )
{
	w=q.w; x=q.x; y=q.y; z=q.z;
	return (*this);
}

inline bool
ZQuaternion::operator==( const ZQuaternion& q ) const
{
	if( w != q.w ) { return false; }
	if( x != q.x ) { return false; }
	if( y != q.y ) { return false; }
	if( z != q.z ) { return false; }
	return true;
}

inline bool
ZQuaternion::operator!=( const ZQuaternion& q ) const
{
	if( w != q.w ) { return true; }
	if( x != q.x ) { return true; }
	if( y != q.y ) { return true; }
	if( z != q.z ) { return true; }
	return false;
}

inline ZQuaternion&
ZQuaternion::operator+=( const ZQuaternion& q )
{
	w+=q.w; x+=q.x; y+=q.y; z+=q.z;
	return (*this);
}

inline ZQuaternion&
ZQuaternion::operator-=( const ZQuaternion& q )
{
	w-=q.w; x-=q.x; y-=q.y; z-=q.z;
	return (*this);
}

inline ZQuaternion&
ZQuaternion::operator*=( const ZQuaternion& q )
{
	const double W=w, X=x, Y=y, Z=z;
	w = W*q.w - X*q.x - Y*q.y - Z*q.z;
	x = W*q.x + X*q.w + Y*q.z - Z*q.y;
	y = W*q.y - X*q.z + Y*q.w + Z*q.x;
	z = W*q.z + X*q.y - Y*q.x + Z*q.w;
	return (*this);
}

inline ZQuaternion&
ZQuaternion::operator*=( const double& s )
{
	w*=s; x*=s; y*=s; z*=s;
	return (*this);
}

inline ZQuaternion&
ZQuaternion::operator/=( const double& s )
{
	if( ZAlmostZero(s) ) {
		w = x = y = z = Z_LARGE;
	} else {
		const double d = 1/s;
		w*=d; x*=d; y*=d; z*=d;
	}
	return (*this);
}

inline ZQuaternion
ZQuaternion::operator+( const ZQuaternion& q ) const
{
	return ZQuaternion( w+q.w, x+q.x, y+q.y, z+q.z );
}

inline ZQuaternion
ZQuaternion::operator-( const ZQuaternion& q ) const
{
	return ZQuaternion( w-q.w, x-q.x, y-q.y, z-q.z );
}

inline ZQuaternion
ZQuaternion::operator*( const ZQuaternion& q ) const
{
	return ZQuaternion( w*q.w-x*q.x-y*q.y-z*q.z, w*q.x+x*q.w+y*q.z-z*q.y, w*q.y-x*q.z+y*q.w+z*q.x, w*q.z+x*q.y-y*q.x+z*q.w );
}

inline ZQuaternion
ZQuaternion::operator*( const int& s ) const
{
	return ZQuaternion( w*s, x*s, y*s, z*s );
}

inline ZQuaternion
ZQuaternion::operator*( const double& s ) const
{
	return ZQuaternion( w*s, x*s, y*s, z*s );
}

inline ZQuaternion
ZQuaternion::operator/( const int& s ) const
{
	const double _s = 1.0 / ( (double)s + Z_EPS );
	return ZQuaternion( w*_s, x*_s, y*_s, z*_s );
}

inline ZQuaternion
ZQuaternion::operator/( const double& s ) const
{
	const double _s = 1.0 / ( s + Z_EPS );
	return ZQuaternion( w*_s, x*_s, y*_s, z*_s );
}

inline ZQuaternion
ZQuaternion::operator-() const
{
	return ZQuaternion( -w, -x, -y, -z );
}

inline ZQuaternion&
ZQuaternion::negate()
{
	w=-w; x=-x; y=-y; z=-z;
	return (*this);
}

inline double
ZQuaternion::dot( const ZQuaternion& q ) const
{
	return ( w*q.w + x*q.x + y*q.y + z*q.z );
}

inline bool
ZQuaternion::isEquivalent( const ZQuaternion& q, double tol ) const
{
	if( !ZAlmostSame( w, q.w, tol ) ) { return false; }
	if( !ZAlmostSame( x, q.x, tol ) ) { return false; }
	if( !ZAlmostSame( y, q.y, tol ) ) { return false; }
	if( !ZAlmostSame( z, q.z, tol ) ) { return false; }
	return true;
}

inline double
ZQuaternion::length() const
{
	return sqrt( w*w + x*x + y*y + z*z );
}

inline double
ZQuaternion::squaredLength() const
{
	return ( w*w + x*x + y*y + z*z );
}

inline ZQuaternion&
ZQuaternion::normalize()
{
	const double lenSq = squaredLength();
	if( ZAlmostZero(lenSq) ) { w=1.0; x=y=z=0.0; return (*this); }
	(*this) *= 1/sqrt(lenSq);
	return (*this);
}

inline ZQuaternion
ZQuaternion::normalized() const
{
	ZQuaternion tmp( *this );
	tmp.normalize();
	return tmp;
}

inline ZQuaternion&
ZQuaternion::conjugate()
{
	x=-x; y=-y; z=-z;
	return (*this);
}

inline ZQuaternion
ZQuaternion::conjugated() const
{
	return ZQuaternion( w, -x, -y, -z );
}

inline ZQuaternion&
ZQuaternion::inverse()
{
	conjugate();
	return ( (*this) *= 1/squaredLength() );
}

inline ZQuaternion
ZQuaternion::inversed() const
{
	ZQuaternion q( *this );
	q.inverse();
	return q;
}

inline ZQuaternion
ZQuaternion::derivative( const ZVector& omega ) const
{
	return ZQuaternion
	(
		-x*omega.x - y*omega.y - z*omega.z,
		 w*omega.x - z*omega.y + y*omega.z,
		 z*omega.x + w*omega.y - x*omega.z,
		-y*omega.x + x*omega.y + w*omega.z
	);
}

inline ZQuaternion
ZQuaternion::log() const
{
	double length = sqrt( x*x + y*y + z*z );
	if( ZAlmostZero(w) ) { length = Z_PI_2; }
	else { length = atan( length / w ); }
	return ZQuaternion( 0.0, x*length, y*length, z*length );
}

inline ZQuaternion
ZQuaternion::exp() const
{
	double len1 = sqrt( x*x + y*y + z*z ), len2=0.0;
	if( ZAlmostZero(len1) ) { len2 = 1.0; }
	else { len2 = sin(len1)/len1; }
	return ZQuaternion( cos(len1), x*len2, y*len2, z*len2 );
}

inline ZPoint
ZQuaternion::rotate( const ZPoint& p ) const
{
	ZQuaternion Qp( 0.0, p.x, p.y, p.z );
	ZQuaternion Qr = (*this) * Qp * (*this).inversed();
	return ZPoint( Qr.x, Qr.y, Qr.z );
}

inline ZQuaternion
ZQuaternion::zero()
{
	return ZQuaternion( 0.0, 0.0, 0.0, 0.0 );
}

inline ZQuaternion
ZQuaternion::identity()
{
	return ZQuaternion( 1.0, 0.0, 0.0, 0.0 );
}

/////////////////////////////////////////
// simple non-member utility funcitons //
/////////////////////////////////////////

inline ZQuaternion
operator*( const int& s, const ZQuaternion& q )
{
	return ZQuaternion( s*q.w, s*q.x, s*q.y, s*q.z );
}

inline ZQuaternion
operator*( const double& s, const ZQuaternion& q )
{
	return ZQuaternion( s*q.w, s*q.x, s*q.y, s*q.z );
}

inline ostream&
operator<<( ostream& os, const ZQuaternion& q )
{
	cout << "( " << q.w << " + " << q.x << "i + " << q.y << "j + " << q.z << "k )";
	return os;
}

inline istream&
operator>>( istream& istr, ZQuaternion& q )
{
	istr >> q.w >> q.x >> q.y >> q.z;
	return istr;
}

ZELOS_NAMESPACE_END

#endif

