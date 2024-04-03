//-----------//
// ZMatrix.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
//         Jinhyuk Bae @ Dexter Studios                  //
// last update: 2018.03.08                               //
//-------------------------------------------------------//

#ifndef _ZMatrix_h_
#define _ZMatrix_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

/// @brief 4x4 matrix.
//
// A class for row-major ordering based 4x4 matrix
// , which has consecutive elements of the row of the array contiguous in memory.
// On the other hand, OpenGL and Maya use column-major ordering.
//
// _ij = data[i][j] = (i,j)
// _data = { _00,_01,_02,_03, _10,_11,_12,_13, _20,_21,_22,_23, _30,_31,_32,_33 };
//
// Caution)
// ZMatrix( const ZVector& c0, const ZVector& c1, const ZVector& c2 )
// == set( const ZVector& c0, const ZVector& c1, const ZVector& c2 )
// != gluLookAt( const ZPoint& eye, const ZPoint& center, const ZVector& up )
class ZMatrix
{
	public:

		union
		{
			struct
			{
				float _00, _01, _02, _03;
				float _10, _11, _12, _13;
				float _20, _21, _22, _23;
				float _30, _31, _32, _33;
			};
			float data[4][4];
		};

	public:

		ZMatrix();
		ZMatrix( const ZMatrix& m );
		ZMatrix( const float& m00, const float& m01, const float& m02,
				 const float& m10, const float& m11, const float& m12,
				 const float& m20, const float& m21, const float& m22 );
		ZMatrix( const float& m00, const float& m01, const float& m02, const float& m03,
				 const float& m10, const float& m11, const float& m12, const float& m13,
				 const float& m20, const float& m21, const float& m22, const float& m23,
				 const float& m30, const float& m31, const float& m32, const float& m33 );
		ZMatrix( const float source[16] );
		ZMatrix( const float source[4][4] );
		ZMatrix( const float& s );
		ZMatrix( const ZVector& c0, const ZVector& c1, const ZVector& c2 );

		ZMatrix& set( const float& m00, const float& m01, const float& m02,
					  const float& m10, const float& m11, const float& m12,
					  const float& m20, const float& m21, const float& m22 );
		ZMatrix& set( const float& m00, const float& m01, const float& m02, const float& m03,
					  const float& m10, const float& m11, const float& m12, const float& m13,
					  const float& m20, const float& m21, const float& m22, const float& m23,
					  const float& m30, const float& m31, const float& m32, const float& m33 );
		ZMatrix& set( const float source[16] );
		ZMatrix& set( const float source[4][4] );
		ZMatrix& set( const float& s );
		ZMatrix& set( const ZVector& c0, const ZVector& c1, const ZVector& c2 );

		static ZMatrix gluLookAt( const ZPoint& eye, const ZPoint& center, const ZVector& up );

		bool get( float dest[16] ) const; // vectorization in row-major ordering
		bool get( float dest[4][4] ) const;
		void get( ZVector& c0, ZVector& c1, ZVector& c2 ) const;
		ZVector column( const int& i ) const;

		void zeroize();

		float& operator()( const int& row, const int& col );
		const float& operator()( const int& row, const int& col ) const;

		const float* operator[]( const int& row ) const;

		ZMatrix& operator=( const ZMatrix& m );
		ZMatrix& operator=( const float& s );

		bool operator==( const ZMatrix& m ) const;
		bool operator!=( const ZMatrix& m ) const;

		ZMatrix& operator+=( const ZMatrix& m );
		ZMatrix& operator-=( const ZMatrix& m );

		ZMatrix& operator*=( const ZMatrix& m );

		ZMatrix& operator*=( const int& s );
		ZMatrix& operator*=( const float& s );
		ZMatrix& operator*=( const double& s );

		ZMatrix& operator/=( const int& s );
		ZMatrix& operator/=( const float& s );
		ZMatrix& operator/=( const double& s );

		ZMatrix operator+( const ZMatrix& m ) const;
		ZMatrix operator-( const ZMatrix& m ) const;

		ZMatrix operator*( const ZMatrix& m ) const;
		ZVector operator*( const ZPoint& p ) const;

		ZMatrix operator*( const int& s ) const;
		ZMatrix operator*( const float& s ) const;
		ZMatrix operator*( const double& s ) const;

		ZMatrix operator/( const int& s ) const;
		ZMatrix operator/( const float& s ) const;
		ZMatrix operator/( const double& s ) const;

		ZVector transform( const ZVector& v, bool asVector ) const;
		ZVector transform( const ZVector& v, const ZPoint& pivot, bool asVector ) const;

		ZMatrix& transpose();
		ZMatrix transposed() const;

		bool isEquivalent( const ZMatrix& other, float tolerance=Z_EPS ) const;
		bool isIdentity( float tolerance=Z_EPS ) const;
		bool isSingular( float tolerance=Z_EPS ) const;
		bool isInvertible( float tolerance=Z_EPS ) const;
		bool isSymmetric( float tolerance=Z_EPS ) const;
		bool isUnitary( float tolerance=Z_EPS ) const;
		bool isDiagonal( float tolerance=Z_EPS ) const;

		static ZMatrix identity();

		void setToIdentity();
		ZMatrix& setToOuterProduct( const ZVector& p, const ZVector& q );
		ZMatrix& setToStar( const ZVector& w );

		double det3x3() const;
		double det() const;

		ZMatrix inversed3x3( bool doublePrecision=false ) const;

		ZMatrix& inverse( bool doublePrecision=false );
		ZMatrix inversed( bool doublePrecision=false ) const;

		void setTranslation( const float& tx, const float& ty, const float& tz );
		void setTranslation( const ZVector& t );
		void addTranslation( const float& tx, const float& ty, const float& tz );
		void addTranslation( const ZVector& t );

		ZVector translation() const;
		void getTranslation( float& tx, float& ty, float& tz ) const;
		void getTranslation( ZVector& translation ) const;

		void setEulerRotation( float angle, int axis, bool isRadian=false );
		void setEulerRotation( const ZVector& eulerAngle, ZRotationOrder::RotationOrder order=ZRotationOrder::zZYX, bool isRadian=false );
		void setRotation( const float& rx, const float& ry, const float& rz, bool isRadian=false );

		// eliminate scale effects
		void eliminateScaleEffects();

		// as radians
		// note) the current 3x3 matrix must be orthogonal.
		ZVector rotation() const;

		// note) the current 3x3 matrix must be orthogonal.
		void getRotation( float& rx, float& ry, float& rz, bool asDegrees=true ) const;

		void setScale( const float& sx, const float& sy, const float& sz, bool preserveRotation=true );
		void addScale( float sx, float sy, float sz );

		ZVector scale() const;
		void getScale( float& sx, float& sy, float& sz ) const;

		void decompose( ZVector& translation, ZVector& rotation, ZVector& scale ) const;

		void setTransform( const ZVector& translation, const ZVector& rotation, const ZVector& scale );

		float trace3x3() const;
		float trace() const;

		bool eigen3x3( ZFloat3& eigenValues, ZMatrix& eigenVectors, bool isSymmetric=true );

		void write( ofstream& fout ) const;
		void read( ifstream& fin );
};

inline
ZMatrix::ZMatrix()
{
	_00=1.f; _01=0.f; _02=0.f; _03=0.f;
	_10=0.f; _11=1.f; _12=0.f; _13=0.f;
	_20=0.f; _21=0.f; _22=1.f; _23=0.f;
	_30=0.f; _31=0.f; _32=0.f; _33=1.f;
}

inline
ZMatrix::ZMatrix( const ZMatrix& m )
{
	memcpy( (char*)data, (char*)m.data, 16*sizeof(float) );
}

inline
ZMatrix::ZMatrix
(
	const float& m00, const float& m01, const float& m02,
	const float& m10, const float& m11, const float& m12,
	const float& m20, const float& m21, const float& m22
)
{
	_00=m00;  _01=m01;  _02=m02;  _03=0.f;
	_10=m10;  _11=m11;  _12=m12;  _13=0.f;
	_20=m20;  _21=m21;  _22=m22;  _23=0.f;
	_30=0.f;  _31=0.f;  _32=0.f;  _33=1.f;
}

inline
ZMatrix::ZMatrix
(
	const float& m00, const float& m01, const float& m02, const float& m03,
	const float& m10, const float& m11, const float& m12, const float& m13,
	const float& m20, const float& m21, const float& m22, const float& m23,
	const float& m30, const float& m31, const float& m32, const float& m33
)
{
	_00=m00; _01=m01; _02=m02; _03=m03;
	_10=m10; _11=m11; _12=m12; _13=m13;
	_20=m20; _21=m21; _22=m22; _23=m23;
	_30=m30; _31=m31; _32=m32; _33=m33;
}

inline
ZMatrix::ZMatrix( const float m[16] )
{
	memcpy( (char*)data, (char*)m, 16*sizeof(float) );
}

inline
ZMatrix::ZMatrix( const float m[4][4] )
{
	memcpy( (char*)data, (char*)m, 16*sizeof(float) );
}

inline
ZMatrix::ZMatrix( const float& s )
{
	FOR(i,0,4) FOR(j,0,4) { data[i][j] = s; }
}

inline
ZMatrix::ZMatrix( const ZVector& c0, const ZVector& c1, const ZVector& c2 )
{
	_00=c0.x; _01=c1.x; _02=c2.x; _03=0.f;
	_10=c0.y; _11=c1.y; _12=c2.y; _13=0.f;
	_20=c0.z; _21=c1.z; _22=c2.z; _23=0.f;
	_30=0.f;  _31=0.f;  _32=0.f;  _33=1.f;
}

inline ZMatrix&
ZMatrix::set
(
	const float& m00, const float& m01, const float& m02,
	const float& m10, const float& m11, const float& m12,
	const float& m20, const float& m21, const float& m22
)
{
	_00=m00;  _01=m01;  _02=m02;  _03=0.f;
	_10=m10;  _11=m11;  _12=m12;  _13=0.f;
	_20=m20;  _21=m21;  _22=m22;  _23=0.f;
	_30=0.f;  _31=0.f;  _32=0.f;  _33=1.f;

	return (*this);
}

inline ZMatrix&
ZMatrix::set
(
	const float& m00, const float& m01, const float& m02, const float& m03,
	const float& m10, const float& m11, const float& m12, const float& m13,
	const float& m20, const float& m21, const float& m22, const float& m23,
	const float& m30, const float& m31, const float& m32, const float& m33
)
{
	_00=m00; _01=m01; _02=m02; _03=m03;
	_10=m10; _11=m11; _12=m12; _13=m13;
	_20=m20; _21=m21; _22=m22; _23=m23;
	_30=m30; _31=m31; _32=m32; _33=m33;

	return (*this);
}

inline ZMatrix&
ZMatrix::set( const float m[16] )
{
	if( !m ) { memset( (char*)data, 0, 16*sizeof(float) ); }
	else { memcpy( (char*)data, (char*)m, 16*sizeof(float) ); }

	return (*this);
}

inline ZMatrix&
ZMatrix::set( const float m[4][4] )
{
	if( !m ) { memset( (char*)data, 0, 16*sizeof(float) ); }
	else { memcpy( (char*)data, (char*)m, 16*sizeof(float) ); }

	return (*this);
}

inline ZMatrix&
ZMatrix::set( const float& s )
{
	FOR( i, 0, 4 )
	FOR( j, 0, 4 )
	{{
		data[i][j] = s;
	}}

	return (*this);
}

inline ZMatrix&
ZMatrix::set( const ZVector& c0, const ZVector& c1, const ZVector& c2 )
{
	_00=c0.x; _01=c1.x; _02=c2.x;
	_10=c0.y; _11=c1.y; _12=c2.y;
	_20=c0.z; _21=c1.z; _22=c2.z;

	return (*this);
}

inline ZMatrix
ZMatrix::gluLookAt( const ZPoint& eye, const ZPoint& center, const ZVector& up )
{
	ZVector F(center-eye), f(F.normalize()), s((f^up).normalize()), u(s^f);

	return ZMatrix
	(
		 s.x,  s.y,  s.z, 0.f,
		 u.x,  u.y,  u.z, 0.f,
		-f.x, -f.y, -f.z, 0.f,
		 0.f,  0.f,  0.f, 1.f
	);
}

inline bool
ZMatrix::get( float m[16] ) const
{
	if( !m ) { return false; }
	memcpy( (char*)m, (char*)data, 16*sizeof(float) );
	return true;
}

inline bool
ZMatrix::get( float m[4][4] ) const
{
	if( !m ) { return false; }
	memcpy( (char*)m, (char*)data, 16*sizeof(float) );
	return true;
}

inline void
ZMatrix::get( ZVector& c0, ZVector& c1, ZVector& c2 ) const
{
	c0.set( _00, _10, _20 );
	c1.set( _01, _11, _21 );
	c2.set( _02, _12, _22 );
}

inline ZVector
ZMatrix::column( const int& i ) const
{
	return ZVector( data[0][i], data[1][i], data[2][i] );
}

inline void
ZMatrix::zeroize()
{
	memset( (char*)data, 0, 16*sizeof(float) );
}

inline const float&
ZMatrix::operator()( const int& i, const int& j ) const
{
	return data[i][j];
}

inline float&
ZMatrix::operator()( const int& i, const int& j )
{
	return data[i][j];
}

inline const float*
ZMatrix::operator[]( const int& i ) const
{
	return data[i];
}

inline ZMatrix&
ZMatrix::operator=( const ZMatrix& m )
{
	memcpy( (char*)data, (char*)m.data, 16*sizeof(float) );

	return (*this);
}

inline ZMatrix&
ZMatrix::operator=( const float& s )
{
	FOR( i, 0, 4 )
	FOR( j, 0, 4 )
	{{
		data[i][j] = s;
	}}

	return (*this);
}

inline bool
ZMatrix::operator==( const ZMatrix& m ) const
{
	FOR( i, 0, 4 )
	FOR( j, 0, 4 )
	{{
		if( data[i][j] != m.data[i][j] )
		{
			return false;
		}
	}}

	return true;
}

inline bool
ZMatrix::operator!=( const ZMatrix& m ) const
{
	FOR( i, 0, 4 )
	FOR( j, 0, 4 )
	{{
		if( data[i][j] != m.data[i][j] )
		{
			return true;
		}
	}}

	return false;
}

inline ZMatrix&
ZMatrix::operator+=( const ZMatrix& m )
{
	FOR( i, 0, 4 )
	FOR( j, 0, 4 )
	{{
		data[i][j] += m[i][j];
	}}

	return (*this);
}

inline ZMatrix&
ZMatrix::operator-=( const ZMatrix& m )
{
	FOR( i, 0, 4 )
	FOR( j, 0, 4 )
	{{
		data[i][j] -= m[i][j];
	}}

	return (*this);
}

inline ZMatrix&
ZMatrix::operator*=( const ZMatrix& m )
{
	float tmp[4][4];
	get( tmp );

	FOR( i, 0, 4 )
	FOR( j, 0, 4 )
	{{
		float& d = data[i][j] = 0.f;

		FOR( k, 0, 4 )
		{
			d += tmp[i][k] * m.data[k][j];
		}
	}}

	return (*this);
}

inline ZMatrix&
ZMatrix::operator*=( const int& s )
{
	const float ss = (float)s;

	FOR( i, 0, 4 )
	FOR( j, 0, 4 )
	{{
		data[i][j] *= ss;
	}}

	return (*this);
}

inline ZMatrix&
ZMatrix::operator*=( const float& s )
{
	FOR( i, 0, 4 )
	FOR( j, 0, 4 )
	{{
		data[i][j] *= s;
	}}

	return (*this);
}

inline ZMatrix&
ZMatrix::operator*=( const double& s )
{
	const float ss = (float)s;

	FOR( i, 0, 4 )
	FOR( j, 0, 4 )
	{{
		data[i][j] *= ss;
	}}

	return (*this);
}

inline ZMatrix&
ZMatrix::operator/=( const int& s )
{
	if( ZAlmostZero(s) )
	{
		ZMatrix::set( Z_LARGE );
		return (*this);
	}

	const float _s = 1.f / ( s + Z_EPS );

	FOR( i, 0, 4 )
	FOR( j, 0, 4 )
	{{
		data[i][j] *= _s;
	}}

	return (*this);
}

inline ZMatrix&
ZMatrix::operator/=( const float& s )
{
	if( ZAlmostZero(s) )
	{
		ZMatrix::set( Z_LARGE );
		return (*this);
	}

	const float _s = 1.f / ( s + Z_EPS );

	FOR( i, 0, 4 )
	FOR( j, 0, 4 )
	{{
		data[i][j] *= _s;
	}}

	return (*this);
}

inline ZMatrix&
ZMatrix::operator/=( const double& s )
{
	if( ZAlmostZero(s) )
	{
		ZMatrix::set( Z_LARGE );
		return (*this);
	}

	const float _s = 1.f / ( (float)s + Z_EPS );

	FOR( i, 0, 4 )
	FOR( j, 0, 4 )
	{{
		data[i][j] *= _s;
	}}

	return (*this);
}

inline ZMatrix
ZMatrix::operator+( const ZMatrix& m ) const
{
	return ( ZMatrix(*this) += m );
}

inline ZMatrix
ZMatrix::operator-( const ZMatrix& m ) const
{
	return ( ZMatrix(*this) -= m );
}

inline ZMatrix
ZMatrix::operator*( const ZMatrix& m ) const
{
	return ( ZMatrix(*this) *= m );
}

inline ZPoint
ZMatrix::operator*( const ZPoint& p ) const
{
	const float& x = p.x;
	const float& y = p.y;
	const float& z = p.z;

	return ZVector( _00*x+_01*y+_02*z+_03, _10*x+_11*y+_12*z+_13, _20*x+_21*y+_22*z+_23 );
}

inline ZMatrix
ZMatrix::operator*( const int& s ) const
{
	return ( ZMatrix(*this) *= s );
}

inline ZMatrix
ZMatrix::operator*( const float& s ) const
{
	return ( ZMatrix(*this) *= s );
}

inline ZMatrix
ZMatrix::operator*( const double& s ) const
{
	return ( ZMatrix(*this) *= s );
}

inline ZMatrix
ZMatrix::operator/( const int& s ) const
{
	return ( ZMatrix(*this) /= s );
}

inline ZMatrix
ZMatrix::operator/( const float& s ) const
{
	return ( ZMatrix(*this) /= s );
}

inline ZMatrix
ZMatrix::operator/( const double& s ) const
{
	return ( ZMatrix(*this) /= s );
}

inline ZVector
ZMatrix::transform( const ZVector& v, bool asVector ) const
{
	const float& x = v.x;
	const float& y = v.y;
	const float& z = v.z;

	ZVector tmp( _00*x+_01*y+_02*z, _10*x+_11*y+_12*z, _20*x+_21*y+_22*z );

	if( asVector ) { return tmp; } // no consideration for translation

	tmp.x += _03;
	tmp.y += _13;
	tmp.z += _23;

	return tmp;
}

inline ZVector
ZMatrix::transform( const ZVector& v, const ZPoint& pivot, bool asVector ) const
{
	ZVector tmp( v.x-pivot.x, v.y-pivot.y, v.z-pivot.z );
	transform( tmp, asVector );
	tmp += pivot;
	return tmp;
}

inline ZMatrix&
ZMatrix::transpose()
{
	FOR( i, 0, 4 )
	FOR( j, i, 4 )
	{{
		if( i != j )
		{
			ZSwap( data[i][j], data[j][i] );
		}
	}}

	return (*this);
}

inline ZMatrix
ZMatrix::transposed() const
{
	return ( ZMatrix(*this).transpose() );
}

inline bool
ZMatrix::isEquivalent( const ZMatrix& m, float tol ) const
{
	FOR( i, 0, 4 )
	FOR( j, 0, 4 )
	{{
		if( !ZAlmostSame( data[i][j], m.data[i][j], tol ) )
		{
			return false;
		}
	}}

	return true;
}

inline bool
ZMatrix::isIdentity( float tol ) const
{
	return (*this).isEquivalent( identity(), tol );
}

inline bool
ZMatrix::isSingular( float tol ) const
{
	return ZAlmostZero( (float)det(), tol );
}

inline bool
ZMatrix::isInvertible( float tol ) const
{
	return !ZAlmostZero( (float)det(), tol );
}

inline bool
ZMatrix::isSymmetric( float tol ) const
{
	return (*this).isEquivalent( (*this).transposed(), tol );
}

inline bool
ZMatrix::isUnitary( float tol ) const
{
	if( !ZAlmostSame( (float)det(), 1.f, tol ) ) { return false; }
	// check if orthogonal
	ZMatrix tmp( (*this) * (*this).transposed() );
	return tmp.isIdentity( tol );
}

inline bool
ZMatrix::isDiagonal( float tol ) const
{
	float sum = 0.f;

	FOR( i, 0, 4 )
	FOR( j, 0, 4 )
	{{
		if( i != j )
		{
			sum += data[i][j];
		}
	}}

	return ZAlmostZero( sum, tol );
}

inline ZMatrix
ZMatrix::identity()
{
	return ZMatrix
	(
		1.f, 0.f, 0.f, 0.f,
		0.f, 1.f, 0.f, 0.f,
		0.f, 0.f, 1.f, 0.f,
		0.f, 0.f, 0.f, 1.f
	);
}

inline void
ZMatrix::setToIdentity()
{
	FOR( i, 0, 4 )
	FOR( j, 0, 4 )
	{{
		data[i][j] = (i==j) ? 1.f : 0.f;
	}}
}

inline ZMatrix&
ZMatrix::setToOuterProduct( const ZVector& p, const ZVector& q )
{
	_00=p.x*q.x; _01=p.x*q.y; _02=p.x*q.z; _03=0.f;
	_10=p.y*q.x; _11=p.y*q.y; _12=p.y*q.z; _13=0.f;
	_20=p.z*q.x; _21=p.z*q.y; _22=p.z*q.z; _23=0.f;
	_30=0.f;     _31=0.f;     _32=0.f;     _33=1.f;

	return (*this);
}

inline ZMatrix&
ZMatrix::setToStar( const ZVector& w )
{
	_00=0.f;  _01=-w.z; _02= w.y; _03=0.f;
	_10= w.z; _11=0.f;  _12=-w.x; _13=0.f;
	_20=-w.y; _21= w.x; _22=0.f;  _23=0.f;
	_30=0.f;  _31=0.f;  _32=0.f;  _33=1.f;

	return (*this);
}

inline double
ZMatrix::det3x3() const
{
	return double
	(
		_00 * ( _11 * _22 - _21 * _12 ) - _01 * ( _10 * _22 - _12 * _20 ) + _02 * ( _10 * _21 - _11 * _20 )
	);

}

inline double
ZMatrix::det() const
{
	return double
	(
		_03 * _12 * _21 * _30 - _02 * _13 * _21 * _30 -
		_03 * _11 * _22 * _30 + _01 * _13 * _22 * _30 +
		_02 * _11 * _23 * _30 - _01 * _12 * _23 * _30 -
		_03 * _12 * _20 * _31 + _02 * _13 * _20 * _31 +
		_03 * _10 * _22 * _31 - _00 * _13 * _22 * _31 -
		_02 * _10 * _23 * _31 + _00 * _12 * _23 * _31 +
		_03 * _11 * _20 * _32 - _01 * _13 * _20 * _32 -
		_03 * _10 * _21 * _32 + _00 * _13 * _21 * _32 +
		_01 * _10 * _23 * _32 - _00 * _11 * _23 * _32 -
		_02 * _11 * _20 * _33 + _01 * _12 * _20 * _33 +
		_02 * _10 * _21 * _33 - _00 * _12 * _21 * _33 -
		_01 * _10 * _22 * _33 + _00 * _11 * _22 * _33
	);
}

inline ZMatrix
ZMatrix::inversed3x3( bool doublePrecision ) const
{
	ZMatrix inv;

	const double _det = 1.0 / ( det3x3() + 1e-30 );

	inv.data[0][0] = (float)( _11*_22 - _12*_21 ) * _det;
	inv.data[0][1] = (float)( _02*_21 - _01*_22 ) * _det;
	inv.data[0][2] = (float)( _01*_12 - _02*_11 ) * _det;

	inv.data[1][0] = (float)( _12*_20 - _10*_22 ) * _det;
	inv.data[1][1] = (float)( _00*_22 - _02*_20 ) * _det;
	inv.data[1][2] = (float)( _02*_10 - _00*_12 ) * _det;

	inv.data[2][0] = (float)( _10*_21 - _11*_20 ) * _det;
	inv.data[2][1] = (float)( _01*_20 - _00*_21 ) * _det;
	inv.data[2][2] = (float)( _00*_11 - _01*_10 ) * _det;

	return inv;
}

inline ZMatrix&
ZMatrix::inverse( bool doublePrecision )
{
	const double _det = 1.0 / ( det() + 1e-30 );

	const double m00=_00, m01=_01, m02=_02, m03=_03;
	const double m10=_10, m11=_11, m12=_12, m13=_13;
	const double m20=_20, m21=_21, m22=_22, m23=_23;
	const double m30=_30, m31=_31, m32=_32, m33=_33;

	_00 = (float)(  ( m11*(m22*m33-m23*m32) - m12*(m21*m33-m23*m31) + m13*(m21*m32-m22*m31) ) * _det );
	_01 = (float)( -( m01*(m22*m33-m23*m32) - m02*(m21*m33-m23*m31) + m03*(m21*m32-m22*m31) ) * _det );
	_02 = (float)(  ( m01*(m12*m33-m13*m32) - m02*(m11*m33-m13*m31) + m03*(m11*m32-m12*m31) ) * _det );
	_03 = (float)( -( m01*(m12*m23-m13*m22) - m02*(m11*m23-m13*m21) + m03*(m11*m22-m12*m21) ) * _det );

	_10 = (float)( -( m10*(m22*m33-m23*m32) - m12*(m20*m33-m23*m30) + m13*(m20*m32-m22*m30) ) * _det );
	_11 = (float)(  ( m00*(m22*m33-m23*m32) - m02*(m20*m33-m23*m30) + m03*(m20*m32-m22*m30) ) * _det );
	_12 = (float)( -( m00*(m12*m33-m13*m32) - m02*(m10*m33-m13*m30) + m03*(m10*m32-m12*m30) ) * _det );
	_13 = (float)(  ( m00*(m12*m23-m13*m22) - m02*(m10*m23-m13*m20) + m03*(m10*m22-m12*m20) ) * _det );

	_20 = (float)(  ( m10*(m21*m33-m23*m31) - m11*(m20*m33-m23*m30) + m13*(m20*m31-m21*m30) ) * _det );
	_21 = (float)( -( m00*(m21*m33-m23*m31) - m01*(m20*m33-m23*m30) + m03*(m20*m31-m21*m30) ) * _det );
	_22 = (float)(  ( m00*(m11*m33-m13*m31) - m01*(m10*m33-m13*m30) + m03*(m10*m31-m11*m30) ) * _det );
	_23 = (float)( -( m00*(m11*m23-m13*m21) - m01*(m10*m23-m13*m20) + m03*(m10*m21-m11*m20) ) * _det );

	_30 = (float)( -( m10*(m21*m32-m22*m31) - m11*(m20*m32-m22*m30) + m12*(m20*m31-m21*m30) ) * _det );
	_31 = (float)(  ( m00*(m21*m32-m22*m31) - m01*(m20*m32-m22*m30) + m02*(m20*m31-m21*m30) ) * _det );
	_32 = (float)( -( m00*(m11*m32-m12*m31) - m01*(m10*m32-m12*m30) + m02*(m10*m31-m11*m30) ) * _det );
	_33 = (float)(  ( m00*(m11*m22-m12*m21) - m01*(m10*m22-m12*m20) + m02*(m10*m21-m11*m20) ) * _det );

	return (*this);
}

inline ZMatrix
ZMatrix::inversed( bool doublePrecision ) const
{
	return ( ZMatrix(*this).inverse( doublePrecision ) );
}

inline void
ZMatrix::setTranslation( const float& tx, const float& ty, const float& tz )
{
	_03 = tx;
	_13 = ty;
	_23 = tz;
}

inline void
ZMatrix::setTranslation( const ZVector& t )
{
	_03 = t.x;
	_13 = t.y;
	_23 = t.z;
}

inline void
ZMatrix::addTranslation( const float& tx, const float& ty, const float& tz )
{
	_03 += tx;
	_13 += ty;
	_23 += tz;
}

inline void
ZMatrix::addTranslation( const ZVector& t )
{
	_03 += t.x;
	_13 += t.y;
	_23 += t.z;
}

inline ZVector
ZMatrix::translation() const
{
	return ZVector( _03, _13, _23 );
}

inline void
ZMatrix::getTranslation( float& tx, float& ty, float& tz ) const
{
	tx = _03;
	ty = _13;
	tz = _23;
}

inline void
ZMatrix::getTranslation( ZVector& t ) const
{
	t.x = _03;
	t.y = _13;
	t.z = _23;
}

// Don't touch other elements (*3 or 3*).
inline void
ZMatrix::setEulerRotation( float a, int axis, bool isRadian )
{
	if( !isRadian ) { a = ZDegToRad(a); }

	const float s = sinf(a);
	const float c = cosf(a);

	switch( axis )
	{
		default:
		case 0: // about x-axis
		{
			_00=1.f; _01=0.f; _02=0.f;
			_10=0.f; _11=c;   _12=-s;
			_20=0.f; _21=s;   _22=c;
			break;
		}

		case 1: // about y-axis
		{
			_00=c;   _01=0.f; _02=s;
			_10=0.f; _11=1.f; _12=0.f;
			_20=-s;  _21=0.f; _22=c;
			break;
		}

		case 2: // about z-axix
		{
			_00=c;   _01=-s;  _02=0.f;
			_10=s;   _11=c;   _12=0.f;
			_20=0.f; _21=0.f; _22=1.f;
			break;
		}
	}
}

// Don't touch other elements (*3 or 3*).
inline void
ZMatrix::setEulerRotation( const ZVector& eulerAngle, ZRotationOrder::RotationOrder order, bool isRadian )
{
	const float rx = isRadian ? eulerAngle.x : ZDegToRad(eulerAngle.x);
	const float ry = isRadian ? eulerAngle.y : ZDegToRad(eulerAngle.y);
	const float rz = isRadian ? eulerAngle.z : ZDegToRad(eulerAngle.z);

	const float sx=sinf(rx), cx=cosf(rx);
	const float sy=sinf(ry), cy=cosf(ry);
	const float sz=sinf(rz), cz=cosf(rz);

	switch( order )
	{
		case ZRotationOrder::zXYZ:
		{
			_00=cy*cz;          _01=-cy*sz;         _02=sy;
			_10=cx*sz+sx*sy*cz; _11=cx*cz-sx*sy*sz; _12=-sx*cy;
			_20=sx*sz-cx*sy*cz; _21=sx*cz+cx*sy*sz; _22=cx*cy;

			break;
		}

		case ZRotationOrder::zYZX:
		{
			_00=cy*cz;          _01=sx*sy-cx*cy*sz; _02=sx*cy*sz+cx*sy;
			_10=sz;             _11=cx*cz;          _12=-sx*cz;
			_20=-sy*cz;         _21=cx*sy*sz+sx*cy; _22=cx*cy-sx*sy*sz;

			break;
		}

		case ZRotationOrder::zZXY:
		{
			_00=cy*cz-sx*sy*sz; _01=-cx*sz;         _02=sy*cz+sx*cy*sz;
			_10=cy*sz+sx*sy*cz; _11=cx*cz;          _12=sy*sz-sx*cy*cz;
			_20=-cx*sy;         _21=sx;             _22=cx*cy;

			break;
		}

		case ZRotationOrder::zXZY:
		{
			_00=cy*cz;          _01=-sz;            _02=sy*cz;
			_10=cx*cy*sz+sx*sy; _11=cx*cz;          _12=cx*sy*sz-sx*cy;
			_20=sx*cy*sz-cx*sy; _21=sx*cz;          _22=sx*sy*sz+cx*cy;

			break;
		}

		case ZRotationOrder::zYXZ:
		{
			_00=cy*cz+sx*sy*sz; _01=sx*sy*cz-cy*sz; _02=cx*sy;
			_10=cx*sz;          _11=cx*cz;          _12=-sx;
			_20=sx*cy*sz-sy*cz; _21=sy*sz+sx*cy*cz; _22=cx*cy;

			break;
		}

		case ZRotationOrder::zZYX:
		{
			_00=cy*cz;          _01=sx*sy*cz-cx*sz; _02=cx*sy*cz+sx*sz;
			_10=cy*sz;          _11=sx*sy*sz+cx*cz; _12=cx*sy*sz-sx*cz;
			_20=-sy;            _21=sx*cy;          _22=cx*cy;

			break;
		}

		default:
		{
			cout << "Error@ZMatrix::setRotation(): Invalid order." << endl;
			return;
		}
	}
}

inline void
ZMatrix::setRotation( const float& rx, const float& ry, const float& rz, bool isRadian )
{
	float xR=0.f, yR=0.f, zR=0.f;

	if( isRadian )
    {
		xR = rx;
		yR = ry;
		zR = rz;
	}
    else
    {
		xR = ZDegToRad( rx );
		yR = ZDegToRad( ry );
		zR = ZDegToRad( rz );
	}

	const float sx = sinf( xR );
	const float cx = cosf( xR );
	const float sy = sinf( yR );
	const float cy = cosf( yR );
	const float sz = sinf( zR );
	const float cz = cosf( zR );

	_00=cy*cz; _01=sy*sx*cz-cx*sz; _02=sy*cx*cz+sx*sz;
	_10=cy*sz; _11=sy*sx*sz+cx*cz; _12=sy*cx*sz-sx*cz;
	_20=-sy;   _21=cy*sx;          _22=cy*cx;
}

inline void
ZMatrix::eliminateScaleEffects()
{
	// scale
	ZVector c0, c1, c2;
	get( c0, c1, c2 );

	const float sx = c0.length();
	const float sy = c1.length();
	const float sz = c2.length();

	// rotation
	c0 /= ( sx + Z_EPS );
	c1 /= ( sy + Z_EPS );
	c2 /= ( sz + Z_EPS );

	ZMatrix::set( c0, c1, c2 );
}

inline ZVector
ZMatrix::rotation() const
{
	float rx=0.f, ry=0.f, rz=0.f;

	ZMatrix::getRotation( rx, ry, rz, false );

	return ZVector( rx, ry, rz );
}

// Ref.) "Extracting Euler Angles from a Rotation Matrix" by Mike Day, Insomniac Games
// note) the current 3x3 matrix must be orthogonal.
inline void
ZMatrix::getRotation( float& rx, float& ry, float& rz, bool asDegrees ) const
{
	const float c2 = sqrtf( _00*_00 + _10*_10 );

	if( c2 < Z_EPS ) // singular case
	{
		rx = ry = rz = 0.f;
	}
	else // regular case
	{
		rx = atan2( _21, _22 );

		ry = atan2( -_20, c2 );

		const float s = sinf( rx );
		const float c = cosf( rx );

		rz = atan2( s*_02-c*_01, c*_11-s*_12 );
	}

	if( asDegrees )
	{
		rx = ZRadToDeg( rx );
		ry = ZRadToDeg( ry );
		rz = ZRadToDeg( rz );
	}
}

// assumption) No shear.
inline void
ZMatrix::setScale( const float& sx, const float& sy, const float& sz, bool preserveRotation )
{
	if( preserveRotation ) {

		float rx=0.f, ry=0.f, rz=0.f;

		ZMatrix::eliminateScaleEffects();
		ZMatrix::getRotation( rx, ry, rz, false );
		ZMatrix::setRotation( rx, ry, rz );

		_00*=sx;  _01*=sy;  _02*=sz;
		_10*=sx;  _11*=sy;  _12*=sz;
		_20*=sx;  _21*=sy;  _22*=sz;

	} else {

		_00=sx;   _01=sy;   _02=sz;
		_10=sx;   _11=sy;   _12=sz;
		_20=sx;   _21=sy;   _22=sz;

	}
}

inline void
ZMatrix::addScale( float sx, float sy, float sz )
{
	_00*=sx;  _01*=sy;  _02*=sz;
	_10*=sx;  _11*=sy;  _12*=sz;
	_20*=sx;  _21*=sy;  _22*=sz;
}

inline ZVector
ZMatrix::scale() const
{
	ZVector s;
	float& sx = s.x;
	float& sy = s.y;
	float& sz = s.z;

	ZVector c0, c1, c2;
	get( c0, c1, c2 );

	sx = c0.length();
	sy = c1.length();
	sz = c2.length();

	return s;
}

inline void
ZMatrix::getScale( float& sx, float& sy, float& sz ) const
{
	ZVector c0, c1, c2;
	get( c0, c1, c2 );

	sx = c0.length();
	sy = c1.length();
	sz = c2.length();
}

inline void
ZMatrix::decompose( ZVector& t, ZVector& r, ZVector& s) const
{
	ZMatrix tmp( *this );

	tmp.getTranslation( t.x, t.y, t.z );

	tmp.getScale( s.x, s.y, s.z );

	tmp.eliminateScaleEffects();

	tmp.getRotation( r.x, r.y, r.z, false );
}

inline void
ZMatrix::setTransform( const ZVector& t, const ZVector& r, const ZVector& s )
{
	ZMatrix::setRotation( r.x, r.y, r.z, true );
	ZMatrix::addScale( s.x, s.y, s.z );
	ZMatrix::addTranslation( t.x, t.y, t.z );
}

inline float
ZMatrix::trace3x3() const
{
	return (_00 + _11 + _22);
}

inline float
ZMatrix::trace() const
{
	return ( _00 + _11 + _22 + _33 );
}

inline void
ZMatrix::write( ofstream& fout ) const
{
	fout.write( (char*)data, 16*sizeof(float) );
}

inline void
ZMatrix::read( ifstream& fin )
{
	fin.read( (char*)data, 16*sizeof(float) );
}

/////////////////////////////////////////
// simple non-member utility funcitons //
/////////////////////////////////////////

inline ZMatrix
operator*( const int& s, const ZMatrix& m )
{
	ZMatrix ret(m);
	ret *= s;
	return ret;
}

inline ZMatrix
operator*( const float& s, const ZMatrix& m )
{
	ZMatrix ret(m);
	ret *= s;
	return ret;
}

inline ZMatrix
operator*( const double& s, const ZMatrix& m )
{
	ZMatrix ret(m);
	ret *= s;
	return ret;
}

inline ostream&
operator<<( ostream& os, const ZMatrix& m )
{
    std::string ret;
    std::string indent;

    const int indentation = 0;
    indent.append( indentation+1, ' ' );

    ret.append( "[" );

    for( int i=0; i<4; ++i )
    {
        ret.append( "[" );

        for( int j=0; j<4; ++j )
        {
            if( j ) { ret.append(", "); }
            ret.append( std::to_string( m(i,j) ) );
        }

        ret.append("]");

        if( i< 4-1 )
        {
            ret.append( ",\n" );
            ret.append( indent );
        }
    }

    ret.append( "]" );

    os << ret;

    return os;
}

ZELOS_NAMESPACE_END

#endif

