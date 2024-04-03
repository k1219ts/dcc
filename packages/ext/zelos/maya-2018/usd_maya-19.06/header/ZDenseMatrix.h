//----------------//
// ZDenseMatrix.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
//         Nayoung Kim @ Dexter Studios                  //
// last update: 2018.03.08                               //
//-------------------------------------------------------//

#ifndef _ZDenseMatrix_h_
#define _ZDenseMatrix_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

/// @brief MxN matrix.
//
// A class for row-major ordering based MxN matrix
// , which has consecutive elements of the row of the array contiguous in memory.
template <int M, int N, typename T>
class ZDenseMatrix
{
	public:

		T data[M*N]; // data

	public:

		ZDenseMatrix();
		ZDenseMatrix( const ZDenseMatrix<M,N,T>& m );
		ZDenseMatrix( const T& m00, const T& m01,
					  const T& m10, const T& m11 );
		ZDenseMatrix( const T& m00, const T& m01, const T& m02,
					  const T& m10, const T& m11, const T& m12,
					  const T& m20, const T& m21, const T& m22 );
		ZDenseMatrix( const T& m00, const T& m01, const T& m02, const T& m03,
					  const T& m10, const T& m11, const T& m12, const T& m13,
					  const T& m20, const T& m21, const T& m22, const T& m23,
					  const T& m30, const T& m31, const T& m32, const T& m33 );
		ZDenseMatrix( const T* m );
		ZDenseMatrix( const T& s );

		ZDenseMatrix& set( const T& m00, const T& m01,
						   const T& m10, const T& m11 );
		ZDenseMatrix& set( const T& m00, const T& m01, const T& m02,
						   const T& m10, const T& m11, const T& m12,
						   const T& m20, const T& m21, const T& m22 );
		ZDenseMatrix& set( const T& m00, const T& m01, const T& m02, const T& m03,
						   const T& m10, const T& m11, const T& m12, const T& m13,
						   const T& m20, const T& m21, const T& m22, const T& m23,
						   const T& m30, const T& m31, const T& m32, const T& m33 );
		ZDenseMatrix& set( const T* m );
		ZDenseMatrix& set( const T& s );

		ZDenseMatrix& fill( const T& s );

		void zeroize();

		T& operator[]( const int& i );
		const T& operator[]( const int& i ) const;

		T& operator()( const int& i, int j );
		const T& operator()( const int& i, int j ) const;

		ZDenseMatrix& operator=( const ZDenseMatrix<M,N,T>& m );
		ZDenseMatrix& operator=( const T* m );
		ZDenseMatrix& operator=( const T& s );

		bool operator==( const ZDenseMatrix<M,N,T>& m ) const;
		bool operator!=( const ZDenseMatrix<M,N,T>& m ) const;

		ZDenseMatrix& operator+=( const int& s );
		ZDenseMatrix& operator+=( const float& s );
		ZDenseMatrix& operator+=( const double& s );

		ZDenseMatrix& operator-=( const int& s );
		ZDenseMatrix& operator-=( const float& s );
		ZDenseMatrix& operator-=( const double& s );

		ZDenseMatrix& operator+=( const ZDenseMatrix<M,N,T>& m );
		ZDenseMatrix& operator-=( const ZDenseMatrix<M,N,T>& m );

		ZDenseMatrix operator+( const ZDenseMatrix<M,N,T>& m ) const;
		ZDenseMatrix operator-( const ZDenseMatrix<M,N,T>& m ) const;

		ZDenseMatrix& operator*=( const int& s );
		ZDenseMatrix& operator*=( const float& s );
		ZDenseMatrix& operator*=( const double& s );

		ZDenseMatrix& operator/=( const int& s );
		ZDenseMatrix& operator/=( const float& s );
		ZDenseMatrix& operator/=( const double& s );

		ZDenseMatrix operator*( const int& s ) const;
		ZDenseMatrix operator*( const float& s ) const;
		ZDenseMatrix operator*( const double& s ) const;

		ZDenseMatrix operator/( const int& s ) const;
		ZDenseMatrix operator/( const float& s ) const;
		ZDenseMatrix operator/( const double& s ) const;

		ZDenseMatrix operator-() const;

		ZDenseMatrix& operator*=( const ZDenseMatrix<M,N,T>& m );

		// (MxP matrix) = (MxN matrix) x (NxP matrix)
		template <int P>
		ZDenseMatrix<M,P,T> operator*( const ZDenseMatrix<N,P,T>& m ) const;

		ZTuple<M,T> operator*( const ZTuple<N,T>& v ) const;
		ZVector operator*( const ZVector& v ) const;

		ZDenseMatrix& transpose();
		ZDenseMatrix transposed() const;

		void setToIdentity();

		ZDenseMatrix& setToOrthoProjector( const ZTuple<N,T>& v );
		ZDenseMatrix& setToOrthoProjector( const ZVector& v );

		ZDenseMatrix& setToOuterProduct( const ZTuple<N,T>& a, const ZTuple<N,T>& b );
		ZDenseMatrix& setToOuterProduct( const ZVector& a, const ZVector& b );

		// when a is a vector or 3-tuple, a ^ p = setToStar(a) * p
		ZDenseMatrix& setToStar( const ZTuple<N,T>& a );
		ZDenseMatrix& setToStar( const ZVector& a );

		double determinant() const;

		ZDenseMatrix& inverse();
		ZDenseMatrix inversed() const;
		void getInverse( ZDenseMatrix& m ) const;

		bool eigen( ZTuple<N,T>& eigenValues, ZDenseMatrix<M,N,T>& eigenVectors, int maxIterations=10 ) const;

		T min() const;
		T max() const;

		T absMin() const;
		T absMax() const;

		T l1Norm() const;

		T trace() const;

		ZTuple<N,T> translation() const;
		ZTuple<N,T> rotation( bool inDegrees=false ) const;
		ZTuple<N,T> scale() const;

		ZDenseMatrix& setTranslation( const ZTuple<N,T>& t );
		ZDenseMatrix& setRotation( const ZTuple<N,T>& r, bool inDegrees=false );
		ZDenseMatrix& setScale( const ZTuple<N,T>& s, bool preserveRotation=true );

	private:

		double _determinant( const T* m, int n ) const;
};

template <int M, int N, typename T>
inline
ZDenseMatrix<M,N,T>::ZDenseMatrix()
{
	memset( data, 0, (M*N)*sizeof(T) );

	if( M == N ) // to identity matrix
	{
		FOR( i, 0, N )
		{
			data[i*(N+1)] = (T)1;
		}
	}
}

template <int M, int N, typename T>
inline
ZDenseMatrix<M,N,T>::ZDenseMatrix( const ZDenseMatrix<M,N,T>& m )
{
	memcpy( data, m.data, (M*N)*sizeof(T) );
}

template <int M, int N, typename T>
inline
ZDenseMatrix<M,N,T>::ZDenseMatrix
(
	const T& m00, const T& m01,
	const T& m10, const T& m11
)
{
	data[0] = m00;   data[1] = m01;
	data[2] = m10;   data[3] = m11;
}

template <int M, int N, typename T>
inline
ZDenseMatrix<M,N,T>::ZDenseMatrix
(
	const T& m00, const T& m01, const T& m02,
	const T& m10, const T& m11, const T& m12,
	const T& m20, const T& m21, const T& m22
)
{
	data[0] = m00;   data[1] = m01;   data[2] = m02;
	data[3] = m10;   data[4] = m11;   data[5] = m12;
	data[6] = m20;   data[7] = m21;   data[8] = m22;
}

template <int M, int N, typename T>
inline
ZDenseMatrix<M,N,T>::ZDenseMatrix
(
	const T& m00, const T& m01, const T& m02, const T& m03,
	const T& m10, const T& m11, const T& m12, const T& m13,
	const T& m20, const T& m21, const T& m22, const T& m23,
	const T& m30, const T& m31, const T& m32, const T& m33
)
{
	data[ 0] = m00;   data[ 1] = m01;   data[ 2] = m02;   data[ 3] = m03;
	data[ 4] = m10;   data[ 5] = m11;   data[ 6] = m12;   data[ 7] = m13;
	data[ 8] = m20;   data[ 9] = m21;   data[10] = m22;   data[11] = m23;
	data[12] = m30;   data[13] = m31;   data[14] = m32;   data[15] = m33;
}

template <int M, int N, typename T>
inline
ZDenseMatrix<M,N,T>::ZDenseMatrix( const T* m )
{
	memcpy( data, m, (M*N)*sizeof(T) );
}

template <int M, int N, typename T>
inline
ZDenseMatrix<M,N,T>::ZDenseMatrix( const T& s )
{
	const int MN = M*N;

	FOR( i, 0, MN )
	{
		data[i] = s;
	}
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>&
ZDenseMatrix<M,N,T>::set
(
	const T& m00, const T& m01,
	const T& m10, const T& m11
)
{
	data[0] = m00;   data[1] = m01;
	data[2] = m10;   data[3] = m11;

	return (*this);
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>&
ZDenseMatrix<M,N,T>::set
(
	const T& m00, const T& m01, const T& m02,
	const T& m10, const T& m11, const T& m12,
	const T& m20, const T& m21, const T& m22
)
{
	data[0] = m00;   data[1] = m01;   data[2] = m02;
	data[3] = m10;   data[4] = m11;   data[5] = m12;
	data[6] = m20;   data[7] = m21;   data[8] = m22;

	return (*this);
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>&
ZDenseMatrix<M,N,T>::set
(
	const T& m00, const T& m01, const T& m02, const T& m03,
	const T& m10, const T& m11, const T& m12, const T& m13,
	const T& m20, const T& m21, const T& m22, const T& m23,
	const T& m30, const T& m31, const T& m32, const T& m33
)
{
	data[ 0] = m00;   data[ 1] = m01;   data[ 2] = m02;   data[ 3] = m03;
	data[ 4] = m10;   data[ 5] = m11;   data[ 6] = m12;   data[ 7] = m13;
	data[ 8] = m20;   data[ 9] = m21;   data[10] = m22;   data[11] = m23;
	data[12] = m30;   data[13] = m31;   data[14] = m32;   data[15] = m33;

	return (*this);
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>&
ZDenseMatrix<M,N,T>::set( const T* m )
{
	memcpy( data, m, (M*N)*sizeof(T) );

	return (*this);
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>&
ZDenseMatrix<M,N,T>::set( const T& s )
{
	const int MN = M*N;

	FOR( i, 0, MN )
	{
		data[i] = s;
	}

	return (*this);
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>&
ZDenseMatrix<M,N,T>::fill( const T& s )
{
	const int MN = M*N;

	FOR( i, 0, MN )
	{
		data[i] = s;
	}

	return (*this);
}

template <int M, int N, typename T>
inline void
ZDenseMatrix<M,N,T>::zeroize()
{
	memset( data, 0, (M*N)*sizeof(T) );
}

template <int M, int N, typename T>
inline T&
ZDenseMatrix<M,N,T>::operator[]( const int& i )
{
	return (*(data+i));
}

template <int M, int N, typename T>
inline const T&
ZDenseMatrix<M,N,T>::operator[]( const int& i ) const
{
	return (*(data+i));
}

template <int M, int N, typename T>
inline T&
ZDenseMatrix<M,N,T>::operator()( const int& i, int j )
{
	return (*(data+j+i*N)); // = return data[j+i*N];
}

template <int M, int N, typename T>
inline const T&
ZDenseMatrix<M,N,T>::operator()( const int& i, int j ) const
{
	return (*(data+j+i*N)); // = return data[j+i*N];
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>&
ZDenseMatrix<M,N,T>::operator=( const ZDenseMatrix<M,N,T>& m )
{
	memcpy( data, m.data, (M*N)*sizeof(T) );
	return (*this);
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>&
ZDenseMatrix<M,N,T>::operator=( const T* m )
{
	memcpy( data, m, (M*N)*sizeof(T) );
	return (*this);
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>&
ZDenseMatrix<M,N,T>::operator=( const T& s )
{
	const int MN = M*N;

	FOR( i, 0, MN )
	{
		data[i] = s;
	}

	return (*this);
}

template <int M, int N, typename T>
inline bool
ZDenseMatrix<M,N,T>::operator==( const ZDenseMatrix<M,N,T>& m ) const
{
	const int MN = M*N;

	FOR( i, 0, MN )
	{
		if( data[i] != m.data[i] ) { return false; }
	}

	return true;
}

template <int M, int N, typename T>
inline bool
ZDenseMatrix<M,N,T>::operator!=( const ZDenseMatrix<M,N,T>& m ) const
{
	const int MN = M*N;

	FOR( i, 0, MN )
	{
		if( data[i] != m.data[i] ) { return true; }
	}

	return false;
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>&
ZDenseMatrix<M,N,T>::operator+=( const int& s )
{
	const int MN = M*N;

	FOR( i, 0, MN )
	{
		data[i] += (T)s;
	}

	return (*this);
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>&
ZDenseMatrix<M,N,T>::operator+=( const float& s )
{
	const int MN = M*N;

	FOR( i, 0, MN )
	{
		data[i] += (T)s;
	}

	return (*this);
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>&
ZDenseMatrix<M,N,T>::operator+=( const double& s )
{
	const int MN = M*N;

	FOR( i, 0, MN )
	{
		data[i] += (T)s;
	}

	return (*this);
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>&
ZDenseMatrix<M,N,T>::operator-=( const int& s )
{
	const int MN = M*N;

	FOR( i, 0, MN )
	{
		data[i] -= (T)s;
	}

	return (*this);
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>&
ZDenseMatrix<M,N,T>::operator-=( const float& s )
{
	const int MN = M*N;

	FOR( i, 0, MN )
	{
		data[i] -= (T)s;
	}

	return (*this);
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>&
ZDenseMatrix<M,N,T>::operator-=( const double& s )
{
	const int MN = M*N;

	FOR( i, 0, MN )
	{
		data[i] -= (T)s;
	}

	return (*this);
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>&
ZDenseMatrix<M,N,T>::operator+=( const ZDenseMatrix<M,N,T>& m )
{
	const int MN = M*N;

	FOR( i, 0, MN )
	{
		data[i] += m.data[i];
	}

	return (*this);
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>&
ZDenseMatrix<M,N,T>::operator-=( const ZDenseMatrix<M,N,T>& m )
{
	const int MN = M*N;

	FOR( i, 0, MN )
	{
		data[i] -= m.data[i];
	}

	return (*this);
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>
ZDenseMatrix<M,N,T>::operator+( const ZDenseMatrix<M,N,T>& m ) const
{
	ZDenseMatrix<M,N,T> tmp( *this );
	return ( tmp += m );
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>
ZDenseMatrix<M,N,T>::operator-( const ZDenseMatrix<M,N,T>& m ) const
{
	ZDenseMatrix<M,N,T> tmp( *this );
	return ( tmp -= m );
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>&
ZDenseMatrix<M,N,T>::operator*=( const int& s )
{
	const int MN = M*N;

	const T ss = (T)s;

	FOR( i, 0, MN )
	{
		data[i] *= ss;
	}

	return (*this);
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>&
ZDenseMatrix<M,N,T>::operator*=( const float& s )
{
	const int MN = M*N;

	const T ss = (T)s;

	FOR( i, 0, MN )
	{
		data[i] *= ss;
	}

	return (*this);
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>&
ZDenseMatrix<M,N,T>::operator*=( const double& s )
{
	const int MN = M*N;

	const T ss = (T)s;

	FOR( i, 0, MN )
	{
		data[i] *= ss;
	}

	return (*this);
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>&
ZDenseMatrix<M,N,T>::operator/=( const int& s )
{
	const int MN = M*N;

	const T _s = (T)1 / ( (T)s + (T)Z_EPS );

	FOR( i, 0, MN )
	{
		data[i] *= _s;
	}

	return (*this);
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>&
ZDenseMatrix<M,N,T>::operator/=( const float& s )
{
	const int MN = M*N;

	const T _s = (T)1 / ( (T)s + (T)Z_EPS );

	FOR( i, 0, MN )
	{
		data[i] *= _s;
	}

	return (*this);
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>&
ZDenseMatrix<M,N,T>::operator/=( const double& s )
{
	const int MN = M*N;

	const T _s = (T)1 / ( (T)s + (T)Z_EPS );

	FOR( i, 0, MN )
	{
		data[i] *= _s;
	}

	return (*this);
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>
ZDenseMatrix<M,N,T>::operator*( const int& s ) const
{
	ZDenseMatrix<M,N,T> tmp( *this );
	return ( tmp *= s );
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>
ZDenseMatrix<M,N,T>::operator*( const float& s ) const
{
	ZDenseMatrix<M,N,T> tmp( *this );
	return ( tmp *= s );
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>
ZDenseMatrix<M,N,T>::operator*( const double& s ) const
{
	ZDenseMatrix<M,N,T> tmp( *this );
	return ( tmp *= s );
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>
ZDenseMatrix<M,N,T>::operator/( const int& s ) const
{
	ZDenseMatrix<M,N,T> tmp( *this );
	return ( tmp /= s );
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>
ZDenseMatrix<M,N,T>::operator/( const float& s ) const
{
	ZDenseMatrix<M,N,T> tmp( *this );
	return ( tmp /= s );
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>
ZDenseMatrix<M,N,T>::operator/( const double& s ) const
{
	ZDenseMatrix<M,N,T> tmp( *this );
	return ( tmp /= s );
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>
ZDenseMatrix<M,N,T>::operator-() const
{
	const int MN = M*N;

	ZDenseMatrix<M,N,T> tmp;

	FOR( i, 0, MN )
	{
		tmp.data[i] = -data[i];
	}

	return tmp;
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>&
ZDenseMatrix<M,N,T>::operator*=( const ZDenseMatrix<M,N,T>& m )
{
	if( M == N ) {

		ZDenseMatrix<M,N,T> tmp( *this );

		FOR( i, 0, M )
		FOR( j, 0, N )
		{{
			T& s = (*this)(i,j) = (T)0;

			FOR( k, 0, N )
			{
				s += tmp(i,k) * m(k,j);
			}
		}}

	} else {

		cout << "Error@ZDenseMatrix::operator*=(): Invalid dimension." << endl;

	}

	return (*this);
}

template <int M, int N, typename T>
template <int P>
inline ZDenseMatrix<M,P,T>
ZDenseMatrix<M,N,T>::operator*( const ZDenseMatrix<N,P,T>& m ) const
{
	ZDenseMatrix<M,P,T> tmp;

	FOR( i, 0, M )
	FOR( j, 0, P )
	{{
		T& s = tmp(i,j) = (T)0;

		FOR( k, 0, N )
		{
			s += (*this)(i,k) * m(k,j);
		}
	}}

	return tmp;
}

template <int M, int N, typename T>
inline ZTuple<M,T>
ZDenseMatrix<M,N,T>::operator*( const ZTuple<N,T>& v ) const
{
	ZTuple<M,T> tmp;

	FOR( i, 0, M )
	{
		const int iN = i*N;

		T& s = tmp.data[i] = (T)0;

		FOR( j, 0, N )
		{
			s += data[j+iN] * v.data[j];
		}
	}

	return tmp;
}

template <int M, int N, typename T>
inline ZVector
ZDenseMatrix<M,N,T>::operator*( const ZVector& v ) const
{
	assert( M==3 && N==3 );

	const float& x = v.x;
	const float& y = v.y;
	const float& z = v.z;

	return ZVector
	(
		( data[0] * x ) + ( data[1] * y ) + ( data[2] * z ),
		( data[3] * x ) + ( data[4] * y ) + ( data[5] * z ),
		( data[6] * x ) + ( data[7] * y ) + ( data[8] * z )
	);
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>&
ZDenseMatrix<M,N,T>::transpose()
{
	if( M == N ) {

		FOR( i, 0,   M )
		FOR( j, i+1, N )
		{{
			ZSwap( (*this)(i,j), (*this)(j,i) );
		}}

	} else {

		cout << "Error@ZDenseMatrix::transpose(): Invalid dimension." << endl;

	}

	return (*this);
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>
ZDenseMatrix<M,N,T>::transposed() const
{
	ZDenseMatrix<M,N,T> tmp( *this );
	tmp.transpose();
	return tmp;
}

template <int M, int N, typename T>
inline void
ZDenseMatrix<M,N,T>::setToIdentity()
{
	memset( data, 0, (M*N)*sizeof(T) );

	if( M == N ) // to identity matrix
	{
		FOR( i, 0, N )
		{
			data[i*(N+1)] = (T)1;
		}
	}
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>&
ZDenseMatrix<M,N,T>::setToOrthoProjector( const ZTuple<N,T>& v )
{
	assert( M==3 && N==3 );

	const T& x = v.data[0];
	const T& y = v.data[1];
	const T& z = v.data[2];

	const T _d = (T)1 / ( ZPow2(x) + ZPow2(y) + ZPow2(z) + (T)Z_EPS );

	data[0] = (x*x) * _d;
	data[4] = (y*y) * _d;
	data[8] = (z*z) * _d;

	data[1] = data[3] = (x*y) * _d;
	data[2] = data[6] = (z*x) * _d;
	data[5] = data[7] = (y*z) * _d;

	return (*this);
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>&
ZDenseMatrix<M,N,T>::setToOrthoProjector( const ZVector& v )
{
	assert( M==3 && N==3 );

	const T& x = v.x;
	const T& y = v.y;
	const T& z = v.z;

	const T _d = (T)1 / ( ZPow2(x) + ZPow2(y) + ZPow2(z) + (T)Z_EPS );

	data[0] = (x*x) * _d;
	data[4] = (y*y) * _d;
	data[8] = (z*z) * _d;

	data[1] = data[3] = (x*y) * _d;
	data[2] = data[6] = (z*x) * _d;
	data[5] = data[7] = (y*z) * _d;

	return (*this);
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>&
ZDenseMatrix<M,N,T>::setToOuterProduct( const ZTuple<N,T>& a, const ZTuple<N,T>& b )
{
	assert( M==3 && N==3 );

	const T& ax = a[0];
	const T& ay = a[1];
	const T& az = a[2];

	const T& bx = b[0];
	const T& by = b[1];
	const T& bz = b[2];

	data[0] = ax * bx;
	data[1] = ax * by;
	data[2] = ax * bz;
	data[3] = ay * bx;
	data[4] = ay * by;
	data[5] = ay * bz;
	data[6] = az * bx;
	data[7] = az * by;
	data[8] = az * bz;

	return (*this);
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>&
ZDenseMatrix<M,N,T>::setToOuterProduct( const ZVector& a, const ZVector& b )
{
	assert( M==3 && N==3 );

	const T& ax = a.x;
	const T& ay = a.y;
	const T& az = a.z;

	const T& bx = b.x;
	const T& by = b.y;
	const T& bz = b.z;

	data[0] = ax * bx;
	data[1] = ax * by;
	data[2] = ax * bz;
	data[3] = ay * bx;
	data[4] = ay * by;
	data[5] = ay * bz;
	data[6] = az * bx;
	data[7] = az * by;
	data[8] = az * bz;

	return (*this);
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>&
ZDenseMatrix<M,N,T>::setToStar( const ZTuple<N,T>& v )
{
	assert( M==3 && N==3 );

	const T& x = v.data[0];
	const T& y = v.data[1];
	const T& z = v.data[2];

	data[0] =  (T)0;
	data[1] = -(T)z;
	data[2] =  (T)y;
	data[3] =  (T)z;
	data[4] =  (T)0;
	data[5] = -(T)x;
	data[6] = -(T)y;
	data[7] =  (T)x;
	data[8] =  (T)0;

	return (*this);
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>&
ZDenseMatrix<M,N,T>::setToStar( const ZVector& v )
{
	assert( M==3 && N==3 );

	const T& x = v.x;
	const T& y = v.y;
	const T& z = v.z;

	data[0] =  (T)0;
	data[1] = -(T)z;
	data[2] =  (T)y;
	data[3] =  (T)z;
	data[4] =  (T)0;
	data[5] = -(T)x;
	data[6] = -(T)y;
	data[7] =  (T)x;
	data[8] =  (T)0;

	return (*this);
}

template <int M, int N, typename T>
inline double
ZDenseMatrix<M,N,T>::determinant() const
{
	if( M != N )
	{
		cout << "Error@ZDenseMatrix::determinant(): Invalid dimension." << endl;
		return 0.0;
	}

	if( N < 1 )
	{
		cout << "Error@ZDenseMatrix::determinant(): Invalid dimension." << endl;
		return 0.0;
	}

	if( N == 1 )
	{
		return ( 1.0 / (double)data[0] );
	}

	if( N == 2 )
	{

		const T& a = data[0];
		const T& b = data[1];
		const T& c = data[2];
		const T& d = data[3];

		return double( a*d - b*c );
	}

	if( N == 3 )
	{
		const T& _00 = data[0];
		const T& _01 = data[1];
		const T& _02 = data[2];

		const T& _10 = data[3];
		const T& _11 = data[4];
		const T& _12 = data[5];

		const T& _20 = data[6];
		const T& _21 = data[7];
		const T& _22 = data[8];

		return
		double(
			_00 * ( _11 * _22 - _21 * _12 ) -
			_01 * ( _10 * _22 - _12 * _20 ) +
			_02 * ( _10 * _21 - _11 * _20 )
		);
	}

	if( N == 4 )
	{
		const T& _00 = data[ 0];
		const T& _01 = data[ 1];
		const T& _02 = data[ 2];
		const T& _03 = data[ 3];

		const T& _10 = data[ 4];
		const T& _11 = data[ 5];
		const T& _12 = data[ 6];
		const T& _13 = data[ 7];

		const T& _20 = data[ 8];
		const T& _21 = data[ 9];
		const T& _22 = data[10];
		const T& _23 = data[11];

		const T& _30 = data[12];
		const T& _31 = data[13];
		const T& _32 = data[14];
		const T& _33 = data[15];

		return
		double(
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

	return _determinant( data, N );
}

template <int M, int N, typename T>
double
ZDenseMatrix<M,N,T>::_determinant( const T* m, int n ) const
{
	double det = 0.0;

	if( n < 1 ) {

		cout << "Error@ZDenseMatrix::determinant(): Invalid dimension." << endl;
		return 0.0;

	} else if( n == 1 ) {

		det = ( 1.0 / (double)m[0] );

	} else if( n == 2 ) {

		const T& a = m[0]; //m(0,0)
		const T& b = m[1]; //m(0,1)
		const T& c = m[2]; //m(1,0)
		const T& d = m[3]; //m(1,1)

		det = double( a*d - b*c );

	} else {

		T* mm = new T[(n-1)*(n-1)];

		FOR( j1, 0, n )
		{
			FOR( i, 1, n )
			{
				int j2 = 0;

				FOR( j, 0, n )
				{
					if( j == j1 ) { continue; }

					mm[j2+(n-1)*(i-1)] = (double)m[j+n*i];
					++j2;
				}
			}

			det += pow(-1.0,j1+2.0) * m[j1] * _determinant(mm,n-1);
		}

		delete[] mm;

	}

	return det;
}

template <int M, int N, typename T>
ZDenseMatrix<M,N,T>&
ZDenseMatrix<M,N,T>::inverse()
{
	if( M != N )
	{
		cout << "Error@ZDenseMatrix::inverse(): Invalid dimension." << endl;
		return (*this);
	}

	if( N < 1 )
	{
		cout << "Error@ZDenseMatrix::determinant(): Invalid dimension." << endl;
		return (*this);
	}

	if( N == 1 )
	{
		data[0] = (T)1 / data[0];
		return (*this);
	}

	if( N == 2 )
	{
		const double _det = 1.0 / ( determinant() + Z_EPS );

		T& a = data[0];
		T& b = data[1];
		T& c = data[2];
		T& d = data[3];

		ZSwap( a, d );

		b = -b;
		c = -c;

		a = T( a * _det );
		b = T( a * _det );
		c = T( a * _det );
		d = T( a * _det );

		return (*this);
	}

	if( N == 3 )
	{
		const double _det = 1.0 / ( determinant() + Z_EPS );

		const double _00 = (double)data[0];
		const double _01 = (double)data[1];
		const double _02 = (double)data[2];

		const double _10 = (double)data[3];
		const double _11 = (double)data[4];
		const double _12 = (double)data[5];

		const double _20 = (double)data[6];
		const double _21 = (double)data[7];
		const double _22 = (double)data[8];

		data[0] = T( ( _11*_22 - _12*_21 ) * _det) ;
		data[1] = T( ( _02*_21 - _01*_22 ) * _det );
		data[2] = T( ( _01*_12 - _02*_11 ) * _det );

		data[3] = T( ( _12*_20 - _10*_22 ) * _det );
		data[4] = T( ( _00*_22 - _02*_20 ) * _det );
		data[5] = T( ( _02*_10 - _00*_12 ) * _det );

		data[6] = T( ( _10*_21 - _11*_20 ) * _det );
		data[7] = T( ( _01*_20 - _00*_21 ) * _det );
		data[8] = T( ( _00*_11 - _01*_10 ) * _det );

		return (*this);
	}

	if( N == 4 )
	{
		const double _det = 1.0 / ( determinant() + Z_EPS );

		const double _00 = (double)data[ 0];
		const double _01 = (double)data[ 1];
		const double _02 = (double)data[ 2];
		const double _03 = (double)data[ 3];

		const double _10 = (double)data[ 4];
		const double _11 = (double)data[ 5];
		const double _12 = (double)data[ 6];
		const double _13 = (double)data[ 7];

		const double _20 = (double)data[ 8];
		const double _21 = (double)data[ 9];
		const double _22 = (double)data[10];
		const double _23 = (double)data[11];

		const double _30 = (double)data[12];
		const double _31 = (double)data[13];
		const double _32 = (double)data[14];
		const double _33 = (double)data[15];

		data[ 0] =  T( ( _11*(_22*_33-_23*_32) - _12*(_21*_33-_23*_31) + _13*(_21*_32-_22*_31) ) * _det );
		data[ 1] = -T( ( _01*(_22*_33-_23*_32) - _02*(_21*_33-_23*_31) + _03*(_21*_32-_22*_31) ) * _det );
		data[ 2] =  T( ( _01*(_12*_33-_13*_32) - _02*(_11*_33-_13*_31) + _03*(_11*_32-_12*_31) ) * _det );
		data[ 3] = -T( ( _01*(_12*_23-_13*_22) - _02*(_11*_23-_13*_21) + _03*(_11*_22-_12*_21) ) * _det );

		data[ 4] = -T( ( _10*(_22*_33-_23*_32) - _12*(_20*_33-_23*_30) + _13*(_20*_32-_22*_30) ) * _det );
		data[ 5] =  T( ( _00*(_22*_33-_23*_32) - _02*(_20*_33-_23*_30) + _03*(_20*_32-_22*_30) ) * _det );
		data[ 6] = -T( ( _00*(_12*_33-_13*_32) - _02*(_10*_33-_13*_30) + _03*(_10*_32-_12*_30) ) * _det );
		data[ 7] =  T( ( _00*(_12*_23-_13*_22) - _02*(_10*_23-_13*_20) + _03*(_10*_22-_12*_20) ) * _det );

		data[ 8] =  T( ( _10*(_21*_33-_23*_31) - _11*(_20*_33-_23*_30) + _13*(_20*_31-_21*_30) ) * _det );
		data[ 9] = -T( ( _00*(_21*_33-_23*_31) - _01*(_20*_33-_23*_30) + _03*(_20*_31-_21*_30) ) * _det );
		data[10] =  T( ( _00*(_11*_33-_13*_31) - _01*(_10*_33-_13*_30) + _03*(_10*_31-_11*_30) ) * _det );
		data[11] = -T( ( _00*(_11*_23-_13*_21) - _01*(_10*_23-_13*_20) + _03*(_10*_21-_11*_20) ) * _det );

		data[12] = -T( ( _10*(_21*_32-_22*_31) - _11*(_20*_32-_22*_30) + _12*(_20*_31-_21*_30) ) * _det );
		data[13] =  T( ( _00*(_21*_32-_22*_31) - _01*(_20*_32-_22*_30) + _02*(_20*_31-_21*_30) ) * _det );
		data[14] = -T( ( _00*(_11*_32-_12*_31) - _01*(_10*_32-_12*_30) + _02*(_10*_31-_11*_30) ) * _det );
		data[15] =  T( ( _00*(_11*_22-_12*_21) - _01*(_10*_22-_12*_20) + _02*(_10*_21-_11*_20) ) * _det );

		return (*this);
	}

	// from now, N > 4
	// Gauss Jordan method

	T* A = data;

	T det = (T)1, factor = (T)0;

	T* B = new T[N*N];
	memcpy( B, A, N*N*sizeof(T) );

	T* C = new T[N*N];
	memcpy( C, A, N*N*sizeof(T) );

	T* invA = A;
	memset( (char*)invA, 0, N*N*sizeof(T) );
	FOR( i, 0, N ) { invA[N*i+i] = (T)1; }

	// The current pivot row is iPass.  
	// For each pass, first find the maximum element in the pivot column.
	FOR( iPass, 0, N )
	{
		int imx = iPass;

		FOR( iRow, iPass, N )
		{
			if( ZAbs(B[N*iRow+iPass]) > ZAbs(B[N*imx+iPass]) ) { imx = iRow; }
		}

		// Interchange the elements of row iPass and row imx in both B and invA.
		if( imx != iPass )
		{
			FOR( iCol, 0, N )
			{
				T temp = invA[N*iPass+iCol];
				invA[N*iPass+iCol] = invA[N*imx+iCol];
				invA[N*imx+iCol] = temp;

				if( iCol >= iPass )
				{
					temp = B[N*iPass+iCol];
					B[N*iPass+iCol] = B[N*imx+iCol];
					B[N*imx+iCol] = temp;
				}
			}
		}

		// The current pivot is now B[iPass][iPass].
		// The determinant is the product of the pivot elements.
		const T pivot = B[N*iPass+iPass];
		det = det * pivot;
		if( ZAlmostZero(det) )
		{
			delete[] B;
			delete[] C;

			return (*this);
		}

		FOR( iCol, 0, N )
		{
			// Normalize the pivot row by dividing by the pivot element.
			invA[N*iPass+iCol] = invA[N*iPass+iCol] / pivot;
			if( iCol >= iPass ) { B[N*iPass+iCol] = B[N*iPass+iCol] / pivot; }
		}

		// Bdd a multiple of the pivot row to each row.  The multiple factor 
		// is chosen so that the element of B on the pivot column is 0.
		FOR( iRow, 0, N )
		{
			if( iRow != iPass ) { factor = B[N*iRow+iPass]; }

			FOR( iCol, 0, N )
			{
				if( iRow != iPass )
				{
					invA[N*iRow+iCol] -= factor * invA[N*iPass+iCol];
					B[N*iRow+iCol] -= factor * B[N*iPass+iCol];
				}
			}
		}
	}

	delete[] B;
	delete[] C;

	return (*this);
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>
ZDenseMatrix<M,N,T>::inversed() const
{
	ZDenseMatrix<M,N,T> inv( *this );
	inv.inverse();
	return inv;
}

template <int M, int N, typename T>
inline void
ZDenseMatrix<M,N,T>::getInverse( ZDenseMatrix<M,N,T>& m ) const
{
	m = *this;
	m.inverse();
}

template <int M, int N, typename T>
bool
ZDenseMatrix<M,N,T>::eigen( ZTuple<N,T>& eigenValues, ZDenseMatrix<M,N,T>& eigenVectors, int maxIterations ) const
{
	if( M != N )
	{
		cout << "Error@ZDenseMatrix::eigen(): Invalid dimension." << endl;
		return false;
	}

	if( N < 2 )
	{
		cout << "Error@ZDenseMatrix::eigen(): Invalid dimension." << endl;
		return false;
	}

	eigenValues.zeroize();
	eigenVectors = *this;

	ZTuple<N,T> tmp;

	const int numIters = maxIterations;//ZMin( N, maxIterations );
	bool isRotation = false;

	////////////////////////
	// tri-digonalization //
	{
		int i0, i1, i2, i3;

		for( i0=N-1, i3=N-2; i0>=1; --i0, --i3 )
		{
			T fH = (T)0, fScale = (T)0;

			if( i3 > 0 ) {

				for( i2=0; i2<=i3; ++i2 )
				{
					fScale += ZAbs( eigenVectors(i0,i2) );
				}

				if( fScale == (T)0 ) {

					tmp[i0] = eigenVectors(i0,i3);

				} else {

					T fInvScale = 1 / fScale;

					for( i2=0; i2<=i3; ++i2 )
					{
						eigenVectors(i0,i2) *= fInvScale;
						fH += eigenVectors(i0,i2) * eigenVectors(i0,i2);
					}

					T fF = eigenVectors(i0,i3);
					T fG = (T)sqrt( fH );

					if( fF > 0 ) { fG = -fG; }

					tmp[i0] = fScale*fG;
					fH -= fF * fG;
					eigenVectors(i0,i3) = fF - fG;
					fF = (T)0;
					T fInvH = 1 / fH;

					for( i1=0; i1<=i3; ++i1 )
					{
						eigenVectors(i1,i0) = eigenVectors(i0,i1) * fInvH;
						fG = (T)0;

						for( i2=0; i2<=i1; ++i2 )
						{
							fG += eigenVectors(i1,i2) * eigenVectors(i0,i2);
						}

						for( i2=i1+1; i2<=i3; ++i2 )
						{
							fG += eigenVectors(i2,i1) * eigenVectors(i0,i2);
						}

						tmp[i1] = fG * fInvH;
						fF += tmp[i1] * eigenVectors(i0,i1);
					}

					T fHalfFdivH = (T)0.5 * fF * fInvH;

					for( i1=0; i1<=i3; ++i1 )
					{
						fF = eigenVectors(i0,i1);
						fG = tmp[i1] - fHalfFdivH * fF;
						tmp[i1] = fG;

						for( i2=0; i2<=i1; ++i2 )
						{
							eigenVectors(i1,i2) -= fF*tmp[i2] + fG * eigenVectors(i0,i2);
						}
					}
				}

			} else {

				tmp[i0] = eigenVectors(i0,i3);

			}

			eigenValues[i0] = fH;
		}

		eigenValues[0] = (T)0;
		tmp[0] = (T)0;

		for( i0=0, i3=-1; i0<=N-1; ++i0, ++i3 )
		{
			if( eigenValues[i0] != 0 )
			{
				for( i1=0; i1<=i3; ++i1 )
				{
					T fSum = (T)0;

					for( i2=0; i2<=i3; ++i2 ) {
						fSum += eigenVectors(i0,i2) * eigenVectors(i2,i1);
					}

					for( i2=0; i2<=i3; ++i2 ) {
						eigenVectors(i2,i1) -= fSum * eigenVectors(i2,i0);
					}
				}
			}

			eigenValues[i0] = eigenVectors(i0,i0);
			eigenVectors(i0,i0) = (T)1;

			for( i1=0; i1<=i3; ++i1 )
			{
				eigenVectors(i1,i0) = (T)0;
				eigenVectors(i0,i1) = (T)0;
			}
		}

		// re-ordering if Eigen<T>::QLAlgorithm is used subsequently
		for( i0=1, i3=0; i0<N; ++i0, ++i3 )
		{
			tmp[i3] = tmp[i0];
		}

		tmp[N-1] = (T)0;
	}

	//////////////////
	// QL algorithm //
	{
		for(int i0=0; i0<N; ++i0 )
		{
			int i1;
			for( i1=0; i1<numIters; ++i1 )
			{
				int i2;
				for( i2=i0; i2<=N-2; ++i2 )
				{
					T fTmp = ZAbs( eigenValues[i2] ) + ZAbs( eigenValues[i2+1] );
					if( ZAbs(tmp[i2]) + fTmp == fTmp ) { break; }
				}

				if( i2 == i0 ) { break; }

				T fG = ( eigenValues[i0+1] - eigenValues[i0] ) / ( 2 * tmp[i0] );
				T fR = (T)sqrt( ZPow2(fG) + 1 );

				if( fG < 0 ) {
					fG = eigenValues[i2] - eigenValues[i0] + tmp[i0] / ( fG - fR );
				} else {
					fG = eigenValues[i2] - eigenValues[i0] + tmp[i0] / ( fG + fR );
				}

				T fSin = (T)1, fCos = (T)1, fP = (T)0;

				for( int i3=i2-1; i3>=i0; --i3 )
				{
					T fF = fSin * tmp[i3];
					T fB = fCos * tmp[i3];

					if( ZAbs(fF) >= ZAbs(fG) ) {

						fCos = fG / fF;
						fR = (T)sqrt( ZPow2(fCos) + 1 );
						tmp[i3+1] = fF * fR;
						fSin = 1 / fR;
						fCos *= fSin;

					} else {

						fSin = fF / fG;
						fR = (T)sqrt( ZPow2(fSin)+ 1 );
						tmp[i3+1] = fG * fR;
						fCos = 1 / fR;
						fSin *= fCos;

					}

					fG = eigenValues[i3+1] - fP;
					fR = ( eigenValues[i3] - fG ) * fSin + 2 * fB * fCos;
					fP = fSin * fR;
					eigenValues[i3+1] = fG + fP;
					fG = fCos * fR - fB;

					for( int i4=0; i4<N; ++i4 )
					{
						fF = eigenVectors(i4,i3+1);
						eigenVectors(i4,i3+1) = fSin * eigenVectors(i4,i3) + fCos * fF;
						eigenVectors(i4,i3  ) = fCos * eigenVectors(i4,i3) - fSin * fF;
					}
				}

				eigenValues[i0] -= fP;
				tmp[i0] = fG;
				tmp[i2] = (T)0;
			}

			if( i1 == numIters ) { break; }
		}
	}

	//////////////////////////////////////
	// sorting eigenValues increasingly //
	{
		for( int i0=0, i1; i0<=N-2; ++i0 )
		{
			// locate minimum eigenvalue
			i1 = i0;
			T fMin = eigenValues[i1];

			for( int i2=i0+1; i2<N; ++i2 )
			{
				if( eigenValues[i2] < fMin )
				{
					i1 = i2;
					fMin = eigenValues[i1];
				}
			}

			if( i1 != i0 )
			{
				// swap eigenValues
				eigenValues[i1] = eigenValues[i0];
				eigenValues[i0] = fMin;

				// swap eigenVectors
				for( int i2=0; i2<N; ++i2 )
				{
					T fTmp = eigenVectors(i2,i0);
					eigenVectors(i2,i0) = eigenVectors(i2,i1);
					eigenVectors(i2,i1) = fTmp;
					isRotation = !isRotation;
				}
			}
		}
	}

	////////////////////////
	// guarantee rotation //
	{
		if( !isRotation )
		{
			// change sign on the first column
			for( int i=0; i<N; ++i )
			{
				eigenVectors(i,0) = -eigenVectors(i,0);
			}
		}
	}


	return true;
}

template <int M, int N, typename T>
inline T
ZDenseMatrix<M,N,T>::min() const
{
	const int MN = M*N;

	T min = data[0];

	FOR( i, 1, MN )
	{
		min = ZMin( min, data[i] );
	}

	return min;
}

template <int M, int N, typename T>
inline T
ZDenseMatrix<M,N,T>::max() const
{
	const int MN = M*N;

	T min = data[0];

	FOR( i, 1, MN )
	{
		min = ZMax( min, data[i] );
	}

	return min;
}

template <int M, int N, typename T>
inline T
ZDenseMatrix<M,N,T>::absMin() const
{
	const int MN = M*N;

	T min = data[0];

	FOR( i, 1, MN )
	{
		min = ZAbsMin( min, data[i] );
	}

	return min;
}

template <int M, int N, typename T>
inline T
ZDenseMatrix<M,N,T>::absMax() const
{
	const int MN = M*N;

	T min = data[0];

	FOR( i, 1, MN )
	{
		min = ZAbsMax( min, data[i] );
	}

	return min;
}

template <int M, int N, typename T>
inline T
ZDenseMatrix<M,N,T>::l1Norm() const
{
	T max = (T)0;

	FOR( j, 0, N )
	{
		T sum = (T)0;

		FOR( i, 0, M )
		{
			sum += ZAbs( (*this)(i,j) );
		}

		max = ZMax( max, sum );
	}

	return max;
}

template <int M, int N, typename T>
inline T
ZDenseMatrix<M,N,T>::trace() const
{
	T sum = (T)0;

	if( M == N ) {

		FOR( i, 0, N )
		{
			sum += (*this)(i,i);
		}

	} else {

		cout << "Error@ZDenseMatrix::trace(): Invalid dimension." << endl;

	}

	return sum;
}

template <int M, int N, typename T>
inline ZTuple<N,T>
ZDenseMatrix<M,N,T>::translation() const
{
	assert( M==4 && N==4 );

	const T& tx = data[3    ];
	const T& ty = data[3+  N];
	const T& tz = data[3+2*N];

	return ZTuple<N,T>( data[3], data[7], data[11] );
}

template <int M, int N, typename T>
inline ZTuple<N,T>
ZDenseMatrix<M,N,T>::rotation( bool inDegrees ) const
{
	assert( M>=3 && N>=3 && M==N );

	ZTuple<N,T> tmp;

	T& rx = tmp.data[0];
	T& ry = tmp.data[1];
	T& rz = tmp.data[2];

	const int N2 = N*2;

	const T &_00=data[0],  &_01=data[1],    &_02=data[2];
	const T &_10=data[N],  &_11=data[N+1],  &_12=data[N+2];
	const T &_20=data[N2], &_21=data[N2+1], &_22=data[N2+2];

	if( ZAlmostZero(_00) && ZAlmostZero(_01) ) { rz = (T)0; }
	else { rz = (T)atan2( _10, _00 ); }

	const T c = (T)cos( rz );
	const T s = (T)sin( rz );

	rx = (T)atan2( _02*s-_12*c, _11*c-_01*s );
	ry = (T)atan2( -_20, _00*c+_10*s );

	if( inDegrees )
	{
		rx = ZRadToDeg( rx );
		ry = ZRadToDeg( ry );
		rz = ZRadToDeg( rz );
	}

	return tmp;
}

template <int M, int N, typename T>
inline ZTuple<N,T>
ZDenseMatrix<M,N,T>::scale() const
{
	assert( M>=3 && N>=3 && M==N );

	const int N2 = N*2;

	const T &_00=data[0],  &_01=data[1],    &_02=data[2];
	const T &_10=data[N],  &_11=data[N+1],  &_12=data[N+2];
	const T &_20=data[N2], &_21=data[N2+1], &_22=data[N2+2];

	return ZTuple<N,T>
	(
		(T)sqrt( ZPow2(_00) + ZPow2(_10) + ZPow2(_20) ),
		(T)sqrt( ZPow2(_01) + ZPow2(_11) + ZPow2(_21) ),
		(T)sqrt( ZPow2(_02) + ZPow2(_12) + ZPow2(_22) )
	);
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>&
ZDenseMatrix<M,N,T>::setTranslation( const ZTuple<N,T>& t )
{
	assert( M==4 && N==4 );

	data[3    ] = t.data[0];
	data[3+  N] = t.data[1];
	data[3+2*N] = t.data[2];

	return (*this);
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>&
ZDenseMatrix<M,N,T>::setRotation( const ZTuple<N,T>& r, bool inDegrees )
{
	assert( M>=3 && N>=3 && M==N );

	T rx = r.data[0];
	T ry = r.data[1];
	T rz = r.data[2];

	if( inDegrees )
	{
		rx = ZDegToRad( rx );
		ry = ZDegToRad( ry );
		rz = ZDegToRad( rz );
	}

	const T sx = (T)sin( rx );
	const T cx = (T)cos( rx );
	const T sy = (T)sin( ry );
	const T cy = (T)cos( ry );
	const T sz = (T)sin( rz );
	const T cz = (T)cos( rz );

	const int N2 = N*2;

	data[0]   = cy*cz;
	data[1]   = sy*sx*cz-cx*sz;
	data[2]   = sy*cx*cz+sx*sz;

	data[N]   = cy*sz;
	data[N+1] = sy*sx*sz+cx*cz;
	data[N+2] = sy*cx*sz-sx*cz;

	data[N2]   = -sy;
	data[N2+1] = cy*sx;
	data[N2+2] = cy*cx;

	return (*this);
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>&
ZDenseMatrix<M,N,T>::setScale( const ZTuple<N,T>& s, bool preserveRotation )
{
	assert( M>=3 && N>=3 && M==N );

	T sx = s.data[0];
	T sy = s.data[1];
	T sz = s.data[2];

	const int N2 = N*2;

	T &_00=data[0],  &_01=data[1],    &_02=data[2];
	T &_10=data[N],  &_11=data[N+1],  &_12=data[N+2];
	T &_20=data[N2], &_21=data[N2+1], &_22=data[N2+2];

	if( preserveRotation ) {
	
		setRotation( rotation() );

		_00 *= sx;   _01 *= sy;   _02 *= sz;
		_10 *= sx;   _11 *= sy;   _12 *= sz;
		_20 *= sx;   _21 *= sy;   _22 *= sz;

	} else {

		_00 = sx;    _01 = sy;    _02 = sz;
		_10 = sx;    _11 = sy;    _12 = sz;
		_20 = sx;    _21 = sy;    _22 = sz;

	}

	return (*this);
}

/////////////////////////////////////////
// simple non-member utility funcitons //
/////////////////////////////////////////

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>
operator*( const int& s, const ZDenseMatrix<M,N,T>& m )
{
	ZDenseMatrix<M,N,T> tmp( m );
	return ( tmp *= s );
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>
operator*( const float& s, const ZDenseMatrix<M,N,T>& m )
{
	ZDenseMatrix<M,N,T> tmp( m );
	return ( tmp *= s );
}

template <int M, int N, typename T>
inline ZDenseMatrix<M,N,T>
operator*( const double& s, const ZDenseMatrix<M,N,T>& m )
{
	ZDenseMatrix<M,N,T> tmp( m );
	return ( tmp *= s );
}

template <int M, int N, typename T>
inline ostream&
operator<<( ostream& os, const ZDenseMatrix<M,N,T>& m )
{
	FOR( i, 0, M )
	{
		FOR( j, 0, N )
		{
			os << " " << m(i,j);
		}
		os << endl;
	}
	return os;
}

template <int M, int N, typename T>
inline istream&
operator>>( istream& is, ZDenseMatrix<M,N,T>& m )
{
	const int MN = M*N;
	FOR( i, 0, MN )
	{
		is >> m[i];
	}
	return is;
}

template <int M, int N, typename T>
inline void
ADD( ZDenseMatrix<M,N,T>& A, const ZDenseMatrix<M,N,T>& B, const ZDenseMatrix<M,N,T>& C, bool useOpenMP=false )
{
	const int MN = M*N;

	#pragma omp parallel for if( useOpenMP )
	FOR( i, 0, MN )
	{
		A.data[i] = B.data[i] + C.data[i];
	}
}

template <int M, int N, typename T>
inline void
SUB( ZDenseMatrix<M,N,T>& A, const ZDenseMatrix<M,N,T>& B, const ZDenseMatrix<M,N,T>& C, bool useOpenMP=false )
{
	const int MN = M*N;

	#pragma omp parallel for if( useOpenMP )
	FOR( i, 0, MN )
	{
		A.data[i] = B.data[i] - C.data[i];
	}
}

// b = Ax
template <typename T>
inline void
MUL( ZTuple<3,T>& b, const ZDenseMatrix<3,3,T>& A, const ZTuple<3,T>& x )
{
	int i3 = 0;

	FOR( i, 0, 3 )
	{
		i3 = i*3;

		T& s = b.data[i] = (T)0;

		FOR( j, 0, 3 )
		{
			s += A.data[j+i3] * x.data[j];
		}
	}
}

template <int M, int N, typename T>
inline void
MUL( ZDenseMatrix<M,N,T>& A, const T& b, const ZDenseMatrix<M,N,T>& B, bool useOpenMP=false )
{
	const int MN = M*N;

	#pragma omp parallel for if( useOpenMP )
	FOR( i, 0, MN )
	{
		A.data[i] = b * B.data[i];
	}
}

template <int M, int N, typename T>
inline void
INCMUL( ZDenseMatrix<M,N,T>& A, const T& b, const ZDenseMatrix<M,N,T>& B, bool useOpenMP=false )
{
	const int MN = M*N;

	#pragma omp parallel for if( useOpenMP )
	FOR( i, 0, MN )
	{
		A.data[i] += b * B.data[i];
	}
}

template <int M, int N, typename T>
inline void
SUBMUL( ZDenseMatrix<M,N,T>& A, const T& b, const ZDenseMatrix<M,N,T>& B, bool useOpenMP=false )
{
	const int MN = M*N;

	#pragma omp parallel for if( useOpenMP )
	FOR( i, 0, MN )
	{
		A.data[i] -= b * B.data[i];
	}
}

////////////////
// data types //
////////////////

typedef ZDenseMatrix<2,2,float>  ZMat2x2f;
typedef ZDenseMatrix<3,3,float>  ZMat3x3f;
typedef ZDenseMatrix<4,4,float>  ZMat4x4f;
typedef ZDenseMatrix<9,4,float>  ZMat9x4f;

typedef ZDenseMatrix<2,2,double> ZMat2x2d;
typedef ZDenseMatrix<3,3,double> ZMat3x3d;
typedef ZDenseMatrix<4,4,double> ZMat4x4d;
typedef ZDenseMatrix<9,4,double> ZMat9x4d;

typedef ZDenseMatrix<2,2,float>  ZMat2x2;
typedef ZDenseMatrix<3,3,float>  ZMat3x3;
typedef ZDenseMatrix<4,4,float>  ZMat4x4;
typedef ZDenseMatrix<9,4,float>  ZMat9x4;

ZELOS_NAMESPACE_END

#endif

