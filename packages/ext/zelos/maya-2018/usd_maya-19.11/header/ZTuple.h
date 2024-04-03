//----------//
// ZTuple.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
//         Nayoung Kim @ Dexter Studios                  //
// last update: 2018.06.07                               //
//-------------------------------------------------------//

#ifndef _ZTuple_h_
#define _ZTuple_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

/// @brief N-tuple to be used for N-dimensional vector (or point).
template <int N, typename T>
class ZTuple
{
	public:

		T data[N]; // data

	public:

		ZTuple();
		ZTuple( const ZTuple& v );
		ZTuple( const T* v );
		ZTuple( const T& s );												// for ND
		ZTuple( const T& v0, const T& v1 );									// for 2D
		ZTuple( const T& v0, const T& v1, const T& v2 );					// for 3D
		ZTuple( const T& v0, const T& v1, const T& v2, const T& v3 );		// for 4D

		ZTuple& set( const T* v );
		ZTuple& set( const T& s );											// for ND
		ZTuple& set( const T& v0, const T& v1 );							// for 2D
		ZTuple& set( const T& v0, const T& v1, const T& v2 );				// for 3D
		ZTuple& set( const T& v0, const T& v1, const T& v2, const T& v3 );	// for 4D

		void get( T* v ) const;
		void get( T& X, T& Y ) const;
		void get( T& X, T& Y, T& Z ) const;
		void get( T& X, T& Y, T& Z, T& W ) const;

		ZTuple& fill( const T& s );

		void zeroize();
		void zeroizeExcept( const int& i );

		T& operator[]( const int& i );
		T& operator()( const int& i );

		const T& operator[]( const int& i ) const;
		const T& operator()( const int& i ) const;

		T& x();
		T& y();
		T& z();
		T& w();

		const T& x() const;
		const T& y() const;
		const T& z() const;
		const T& w() const;

		ZTuple& operator=( const ZTuple& v );
		ZTuple& operator=( T* a );
		ZTuple& operator=( const T& s );

		bool operator==( const ZTuple& v ) const;
		bool operator!=( const ZTuple& v ) const;

		bool operator<( const ZTuple& v ) const;
		bool operator>( const ZTuple& v ) const;

		bool operator<=( const ZTuple& v ) const;
		bool operator>=( const ZTuple& v ) const;

		ZTuple& operator+=( const int& s );
		ZTuple& operator+=( const float& s );
		ZTuple& operator+=( const double& s );

		ZTuple& operator-=( const int& s );
		ZTuple& operator-=( const float& s );
		ZTuple& operator-=( const double& s );

		ZTuple& operator+=( const ZTuple& v );
		ZTuple& operator-=( const ZTuple& v );

		ZTuple operator+( const ZTuple& v ) const;
		ZTuple operator-( const ZTuple& v ) const;

		ZTuple& operator*=( const int& s );
		ZTuple& operator*=( const float& s );
		ZTuple& operator*=( const double& s );

		ZTuple& operator/=( const int& s );
		ZTuple& operator/=( const float& s );
		ZTuple& operator/=( const double& s );

		ZTuple operator*( const int& s ) const;
		ZTuple operator*( const float& s ) const;
		ZTuple operator*( const double& s ) const;

		ZTuple operator/( const int& s ) const;
		ZTuple operator/( const float& s ) const;
		ZTuple operator/( const double& s ) const;

		ZTuple operator-() const;

		ZTuple& negate();
		ZTuple negated() const;

		ZTuple& reverse();
		ZTuple reversed() const;

		ZTuple& abs();
		ZTuple& clamp( const T& minValue, const T& maxValue );
		ZTuple& cycle( bool toLeft=true );
		ZTuple& sort( bool increasingOrder=true );

		// dot(inner) product
		T operator*( const ZTuple& v ) const;
		T dot( const ZTuple& v ) const;

		// cross(outer) product (valid only for N=3)
		ZTuple operator^( const ZTuple& v ) const;
		ZTuple cross( const ZTuple& v ) const;

		T length() const;
		T squaredLength() const;

		ZTuple& normalize();
		ZTuple normalized() const;

		ZTuple direction() const;

		ZTuple& limitLength( const T& targetLength );

		T distanceTo( const ZTuple& v ) const;
		T squaredDistanceTo( const ZTuple& v ) const;

		bool isEquivalent( const ZTuple& v, T epsilon=(T)Z_EPS ) const;
		bool isParallel( const ZTuple& v, T epsilon=(T)Z_EPS ) const;

		T min() const;
		T max() const;

		T absMin() const;
		T absMax() const;

		int minIndex() const;
		int maxIndex() const;

		int absMinIndex() const;
		int absMaxIndex() const;

		T sum() const;
		T avg() const;

		ZTuple& setComponentwiseMin( const ZTuple& v );
		ZTuple& setComponentwiseMax( const ZTuple& v );

		ZTuple componentwiseMin( const ZTuple& v ) const;
		ZTuple componentwiseMax( const ZTuple& v ) const;

		bool isZero() const;
		bool isAlmostZero( T eps=(T)Z_EPS ) const;

		// norm
		T l1Norm() const;
		T l2Norm() const;
		T lpNorm( int p ) const;
		T infNorm() const;

		void write( ofstream& fout ) const;
		void read( ifstream& fin );

		// static functions
		static ZTuple zero();
		static ZTuple one();

		static ZTuple unitX();	// set to (1,......)
		static ZTuple unitY();	// set to (0,1,....)
		static ZTuple unitZ();	// set to (0,0,1,..)

		static ZTuple xPosAxis();	// set to (1,......)
		static ZTuple yPosAxis();	// set to (0,1,....)
		static ZTuple zPosAxis();	// set to (0,0,1,..)

		static ZTuple xNegAxis();	// set to (-1,......)
		static ZTuple yNegAxis();	// set to (0,-1,....)
		static ZTuple zNegAxis();	// set to (0,0,-1,..)
};

template <int N, typename T>
inline
ZTuple<N,T>::ZTuple()
{
	memset( data, 0, N*sizeof(T) );
}

template <int N, typename T>
inline
ZTuple<N,T>::ZTuple( const ZTuple<N,T>& v )
{
	memcpy( data, v.data, N*sizeof(T) );
}

template <int N, typename T>
inline
ZTuple<N,T>::ZTuple( const T* v )
{
	memcpy( data, v, N*sizeof(T) );
}

template <int N, typename T>
inline
ZTuple<N,T>::ZTuple( const T& s )
{
	FOR( i, 0, N ) { data[i] = s; }
}

template <int N, typename T>
inline
ZTuple<N,T>::ZTuple( const T& v0, const T& v1 )
{
	data[0] = v0;
	data[1] = v1;
}

template <int N, typename T>
inline
ZTuple<N,T>::ZTuple( const T& v0, const T& v1, const T& v2 )
{
	data[0] = v0;
	data[1] = v1;
	data[2] = v2;
}

template <int N, typename T>
inline
ZTuple<N,T>::ZTuple( const T& v0, const T& v1, const T& v2, const T& v3 )
{
	data[0] = v0;
	data[1] = v1;
	data[2] = v2;
	data[3] = v3;
}

template <int N, typename T>
inline ZTuple<N,T>&
ZTuple<N,T>::set( const T* v )
{
	memcpy( data, v, N*sizeof(T) );
	return (*this);
}

template <int N, typename T>
inline ZTuple<N,T>&
ZTuple<N,T>::set( const T& s )
{
	FOR( i, 0, N )
	{
		data[i] = s;
	}

	return (*this);
}

template <int N, typename T>
inline ZTuple<N,T>&
ZTuple<N,T>::set( const T& v0, const T& v1 )
{
	data[0] = v0;
	data[1] = v1;

	return (*this);
}

template <int N, typename T>
inline ZTuple<N,T>&
ZTuple<N,T>::set( const T& v0, const T& v1, const T& v2 )
{
	data[0] = v0;
	data[1] = v1;
	data[2] = v2;

	return (*this);
}

template <int N, typename T>
inline ZTuple<N,T>&
ZTuple<N,T>::set( const T& v0, const T& v1, const T& v2, const T& v3 )
{
	data[0] = v0;
	data[1] = v1;
	data[2] = v2;
	data[3] = v3;

	return (*this);
}

template <int N, typename T>
inline void
ZTuple<N,T>::get( T* v ) const
{
	memcpy( v, data, N*sizeof(T) );
}

template <int N, typename T>
inline void
ZTuple<N,T>::get( T& X, T& Y ) const
{
	X = data[0];
	Y = data[1];
}

template <int N, typename T>
inline void
ZTuple<N,T>::get( T& X, T& Y, T& Z ) const
{
	X = data[0];
	Y = data[1];
	Z = data[2];
}

template <int N, typename T>
inline void
ZTuple<N,T>::get( T& X, T& Y, T& Z, T& W ) const
{
	X = data[0];
	Y = data[1];
	Z = data[2];
	W = data[3];
}

template <int N, typename T>
inline ZTuple<N,T>&
ZTuple<N,T>::fill( const T& s )
{
	FOR( i, 0, N )
	{
		data[i] = s;
	}

	return (*this);
}

template <int N, typename T>
inline void
ZTuple<N,T>::zeroize()
{
	memset( data, 0, sizeof(T)*N );
}

template <int N, typename T>
inline void
ZTuple<N,T>::zeroizeExcept( const int& i )
{
	const T vi = data[i];
	memset( data, 0, sizeof(T)*N );
	data[i] = vi;
}

template <int N, typename T>
inline T&
ZTuple<N,T>::operator[]( const int& i )
{
	return (*(data+i)); // = data[i];
}

template <int N, typename T>
inline T&
ZTuple<N,T>::operator()( const int& i )
{
	return (*(data+i)); // = data[i];
}

template <int N, typename T>
inline const T&
ZTuple<N,T>::operator[]( const int& i ) const
{
	return (*(data+i)); // = data[i];
}

template <int N, typename T>
inline const T&
ZTuple<N,T>::operator()( const int& i ) const
{
	return (*(data+i)); // = data[i];
}

template <int N, typename T>
inline T&
ZTuple<N,T>::x()
{
	return (*(data)); // = data[0]
}

template <int N, typename T>
inline T&
ZTuple<N,T>::y()
{
	return (*(data+1)); // = data[1]
}

template <int N, typename T>
inline T&
ZTuple<N,T>::z()
{
	return (*(data+2)); // = data[2]
}

template <int N, typename T>
inline T&
ZTuple<N,T>::w()
{
	return (*(data+3)); // = data[3]
}

template <int N, typename T>
inline const T&
ZTuple<N,T>::x() const
{
	return (*(data)); // = data[0]
}

template <int N, typename T>
inline const T&
ZTuple<N,T>::y() const
{
	return (*(data+1)); // = data[1]
}

template <int N, typename T>
inline const T&
ZTuple<N,T>::z() const
{
	return (*(data+2)); // = data[2]
}

template <int N, typename T>
inline const T&
ZTuple<N,T>::w() const
{
	return (*(data+3)); // = data[3]
}

template <int N, typename T>
inline ZTuple<N,T>&
ZTuple<N,T>::operator=( const ZTuple<N,T>& v )
{
	memcpy( data, v.data, N*sizeof(T) );

	return (*this);
}

template <int N, typename T>
inline ZTuple<N,T>&
ZTuple<N,T>::operator=( T* a )
{
	FOR( i, 0, N )
	{
		data[i] = a[i];
	}

	return (*this);
}

template <int N, typename T>
inline ZTuple<N,T>&
ZTuple<N,T>::operator=( const T& s )
{
	FOR( i, 0, N )
	{
		data[i] = s;
	}

	return (*this);
}

template <int N, typename T>
inline bool
ZTuple<N,T>::operator==( const ZTuple<N,T>& v ) const
{
	FOR( i, 0, N )
	{
		if( data[i] != v.data[i] )
		{
			return false;
		}
	}

	return true;
}

template <int N, typename T>
inline bool
ZTuple<N,T>::operator!=( const ZTuple<N,T>& v ) const
{
	FOR( i, 0, N )
	{
		if( data[i] != v.data[i] ) { return true; }
	}

	return false;
}

template <int N, typename T>
inline bool
ZTuple<N,T>::operator<( const ZTuple<N,T>& v ) const
{
	FOR( i, 0, N )
	{
		if( data[i] >= v.data[i] )
		{
			return false;
		}
	}

	return true;
}

template <int N, typename T>
inline bool
ZTuple<N,T>::operator>( const ZTuple<N,T>& v ) const
{
	FOR( i, 0, N )
	{
		if( data[i] <= v.data[i] )
		{
			return false;
		}
	}

	return true;
}

template <int N, typename T>
inline bool
ZTuple<N,T>::operator<=( const ZTuple<N,T>& v ) const
{
	FOR( i, 0, N )
	{
		if( data[i] > v.data[i] )
		{
			return false;
		}
	}

	return true;
}

template <int N, typename T>
inline bool
ZTuple<N,T>::operator>=( const ZTuple<N,T>& v ) const
{
	FOR( i, 0, N )
	{
		if( data[i] < v.data[i] )
		{
			return false;
		}
	}

	return true;
}

template <int N, typename T>
inline ZTuple<N,T>&
ZTuple<N,T>::operator+=( const int& s )
{
	FOR( i, 0, N )
	{
		data[i] += (float)s;
	}

	return (*this);
}

template <int N, typename T>
inline ZTuple<N,T>&
ZTuple<N,T>::operator+=( const float& s )
{
	FOR( i, 0, N )
	{
		data[i] += s;
	}

	return (*this);
}

template <int N, typename T>
inline ZTuple<N,T>&
ZTuple<N,T>::operator+=( const double& s )
{
	FOR( i, 0, N )
	{
		data[i] += (float)s;
	}

	return (*this);
}

template <int N, typename T>
inline ZTuple<N,T>&
ZTuple<N,T>::operator-=( const int& s )
{
	FOR( i, 0, N )
	{
		data[i] -= (float)s;
	}

	return (*this);
}

template <int N, typename T>
inline ZTuple<N,T>&
ZTuple<N,T>::operator-=( const float& s )
{
	FOR( i, 0, N )
	{
		data[i] -= s;
	}

	return (*this);
}

template <int N, typename T>
inline ZTuple<N,T>&
ZTuple<N,T>::operator-=( const double& s )
{
	FOR( i, 0, N )
	{
		data[i] -= (float)s;
	}

	return (*this);
}

template <int N, typename T>
inline ZTuple<N,T>&
ZTuple<N,T>::operator+=( const ZTuple<N,T>& v )
{
	FOR( i, 0, N )
	{
		data[i] += v.data[i];
	}

	return (*this);
}

template <int N, typename T>
inline ZTuple<N,T>&
ZTuple<N,T>::operator-=( const ZTuple<N,T>& v )
{
	FOR( i, 0, N )
	{
		data[i] -= v.data[i];
	}

	return (*this);
}

template <int N, typename T>
inline ZTuple<N,T>
ZTuple<N,T>::operator+( const ZTuple<N,T>& v ) const
{
	ZTuple<N,T> tmp( *this );
	return ( tmp += v );
}

template <int N, typename T>
inline ZTuple<N,T>
ZTuple<N,T>::operator-( const ZTuple<N,T>& v ) const
{
	ZTuple<N,T> tmp( *this );
	return ( tmp -= v );
}

template <int N, typename T>
inline ZTuple<N,T>&
ZTuple<N,T>::operator*=( const int& s )
{
	FOR( i, 0, N )
	{
		data[i] *= (T)s;
	}

	return (*this);
}

template <int N, typename T>
inline ZTuple<N,T>&
ZTuple<N,T>::operator*=( const float& s )
{
	FOR( i, 0, N )
	{
		data[i] *= (T)s;
	}

	return (*this);
}

template <int N, typename T>
inline ZTuple<N,T>&
ZTuple<N,T>::operator*=( const double& s )
{
	FOR( i, 0, N )
	{
		data[i] *= (T)s;
	}

	return (*this);
}

template <int N, typename T>
inline ZTuple<N,T>&
ZTuple<N,T>::operator/=( const int& s )
{
	const T _s = (T)1 / ( (T)s + (T)Z_EPS );

	FOR( i, 0, N )
	{
		data[i] *= _s;
	}

	return (*this);
}

template <int N, typename T>
inline ZTuple<N,T>&
ZTuple<N,T>::operator/=( const float& s )
{
	const T _s = (T)1 / ( (T)s + (T)Z_EPS );

	FOR( i, 0, N )
	{
		data[i] *= _s;
	}

	return (*this);
}

template <int N, typename T>
inline ZTuple<N,T>&
ZTuple<N,T>::operator/=( const double& s )
{
	const T _s = (T)1 / ( (T)s + (T)Z_EPS );

	FOR( i, 0, N )
	{
		data[i] *= _s;
	}

	return (*this);
}

template <int N, typename T>
inline ZTuple<N,T>
ZTuple<N,T>::operator*( const int& s ) const
{
	ZTuple<N,T> tmp( *this );
	return ( tmp *= (T)s );
}

template <int N, typename T>
inline ZTuple<N,T>
ZTuple<N,T>::operator*( const float& s ) const
{
	ZTuple<N,T> tmp( *this );
	return ( tmp *= (T)s );
}

template <int N, typename T>
inline ZTuple<N,T>
ZTuple<N,T>::operator*( const double& s ) const
{
	ZTuple<N,T> tmp( *this );
	return ( tmp *= (T)s );
}

template <int N, typename T>
inline ZTuple<N,T>
ZTuple<N,T>::operator/( const int& s ) const
{
	ZTuple<N,T> tmp( *this );
	return ( tmp /= s );
}

template <int N, typename T>
inline ZTuple<N,T>
ZTuple<N,T>::operator/( const float& s ) const
{
	ZTuple<N,T> tmp( *this );
	return ( tmp /= s );
}

template <int N, typename T>
inline ZTuple<N,T>
ZTuple<N,T>::operator/( const double& s ) const
{
	ZTuple<N,T> tmp( *this );
	return ( tmp /= s );
}

template <int N, typename T>
inline ZTuple<N,T>
ZTuple<N,T>::operator-() const
{
	ZTuple<N,T> tmp;

	FOR( i, 0, N )
	{
		tmp[i] = -data[i];
	}

	return tmp;
}

template <int N, typename T>
inline ZTuple<N,T>&
ZTuple<N,T>::negate()
{
	FOR( i, 0, N )
	{
		data[i] = -data[i];
	}

	return (*this);
}

template <int N, typename T>
inline ZTuple<N,T>
ZTuple<N,T>::negated() const
{
	ZTuple<N,T> tmp( *this );
	tmp.negate();

	return tmp;
}

template <int N, typename T>
inline ZTuple<N,T>&
ZTuple<N,T>::reverse()
{
	FOR( i, 0, N )
	{
		data[i] = -data[i];
	}

	return (*this);
}

template <int N, typename T>
inline ZTuple<N,T>
ZTuple<N,T>::reversed() const
{
	ZTuple<N,T> tmp( *this );
	tmp.reverse();

	return tmp;
}

template <int N, typename T>
inline ZTuple<N,T>&
ZTuple<N,T>::abs()
{
	FOR( i, 0, N )
	{
		data[i] = ZAbs( data[i] );
	}

	return (*this);
}

template <int N, typename T>
inline ZTuple<N,T>&
ZTuple<N,T>::clamp( const T& minValue, const T& maxValue )
{
	FOR( i, 0, N )
	{
		data[i] = ZClamp( data[i], minValue, maxValue );
	}

	return (*this);
}

template <int N, typename T>
inline ZTuple<N,T>&
ZTuple<N,T>::cycle( bool toLeft )
{
	if( toLeft ) {

		const T v0 = data[0];

		for( int i=1; i<N; ++i )
		{
			data[i-1] = data[i];
		}

		data[N-1] = v0;

	} else { // toRight

		const T v0 = data[N-1];

		for( int i=N-1; i>=0; --i )
		{
			data[i] = data[i-1];
		}

		data[0] = v0;

	}

	return (*this);
}

// sort elements by insertion sort
template <int N, typename T>
ZTuple<N,T>&
ZTuple<N,T>::sort( bool increasingOrder )
{
	int j = 0;
	T remember = 0;

	if( increasingOrder ) {

		FOR( i, 1, N )
		{
			remember = data[j=i];
			while( --j >=0 && remember < data[j] )
			{
				data[j+1] = data[j];
			}
			data[j+1] = remember;
		}

	} else { // decreasing order

		FOR( i, 1, N )
		{
			remember = data[j=i];
			while( --j >=0 && remember > data[j] )
			{
				data[j+1] = data[j];
			}
			data[j+1] = remember;
		}

	}

	return (*this);
}

template <int N, typename T>
inline T
ZTuple<N,T>::operator*( const ZTuple<N,T>& v ) const
{
	T sum = (T)0;

	FOR( i, 0, N )
	{
		sum += data[i] * v.data[i];
	}

	return sum;
}

template <int N, typename T>
inline T
ZTuple<N,T>::dot( const ZTuple<N,T>& v ) const
{
	T sum = (T)0;

	FOR( i, 0, N )
	{
		sum += data[i] * v.data[i];
	}

	return sum;
}

template <int N, typename T>
inline ZTuple<N,T>
ZTuple<N,T>::operator^( const ZTuple<N,T>& v ) const
{
	return ZTuple<N,T>
	(
		( data[1] * v[2] ) - ( data[2] * v[1] ),
		( data[2] * v[0] ) - ( data[0] * v[2] ),
		( data[0] * v[1] ) - ( data[1] * v[0] )
	);
}

template <int N, typename T>
inline ZTuple<N,T>
ZTuple<N,T>::cross( const ZTuple<N,T>& v ) const
{
	return ZTuple<N,T>
	(
		( data[1] * v[2] ) - ( data[2] * v[1] ),
		( data[2] * v[0] ) - ( data[0] * v[2] ),
		( data[0] * v[1] ) - ( data[1] * v[0] )
	);
}

template <int N, typename T>
inline T
ZTuple<N,T>::length() const
{
	double sum = ZPow2( data[0] );

	FOR( i, 1, N )
	{
		sum += ZPow2( data[i] );
	}

	return (T)sqrt( sum );
}

template <int N, typename T>
inline T
ZTuple<N,T>::squaredLength() const
{
	double sum = ZPow2( data[0] );

	FOR( i, 1, N )
	{
		sum += ZPow2( data[i] );
	}

	return (T)sqrt( sum );
}

template <int N, typename T>
inline ZTuple<N,T>&
ZTuple<N,T>::normalize()
{
	double d = Z_EPS;
	FOR( i, 0, N ) { d += ZPow2(data[i]); }
	d = 1.0 / sqrt( d );

	FOR( i, 0, N )
	{
		data[i] *= (T)d;
	}

	return (*this);
}

template <int N, typename T>
inline ZTuple<N,T>
ZTuple<N,T>::normalized() const
{
	ZTuple<N,T> tmp( *this );
	tmp.normalize();
	return tmp;
}

template <int N, typename T>
inline ZTuple<N,T>
ZTuple<N,T>::direction() const
{
	ZTuple<N,T> tmp( *this );
	tmp.normalize();
	return tmp;
}

template <int N, typename T>
inline ZTuple<N,T>&
ZTuple<N,T>::limitLength( const T& targetLength )
{
	double lenSQ = data[0];
	FOR( i, 1, N ) { lenSQ += ZPow2(data[i]); }

	if( lenSQ > ZPow2(targetLength) )
	{
		const T d = targetLength / (T)( sqrt(lenSQ) + Z_EPS );

		x *= d;
		y *= d;
		z *= d;
	}

	return (*this);
}

template <int N, typename T>
inline T
ZTuple<N,T>::distanceTo( const ZTuple<N,T>& v ) const
{
	T sum = ZPow2( data[0] - v.data[0] );

	FOR( i, 1, N )
	{
		sum += ZPow2( data[i] - v.data[i] );
	}

	return (T)sqrt( sum );
}

template <int N, typename T>
inline T
ZTuple<N,T>::squaredDistanceTo( const ZTuple<N,T>& v ) const
{
	T sum = ZPow2( data[0] - v.data[0] );

	FOR( i, 1, N )
	{
		sum += ZPow2( data[i] - v.data[i] );
	}

	return (T)sum;
}

template <int N, typename T>
inline bool
ZTuple<N,T>::isEquivalent( const ZTuple<N,T>& v, T e ) const
{
	const ZTuple<N,T> diff( *this - v );
	const float lenSQ = diff.squaredLength();
	return ( lenSQ < (e*e) );
}

template <int N, typename T>
inline bool
ZTuple<N,T>::isParallel( const ZTuple<N,T>& v, T e ) const
{
	return ZAlmostSame( (*this)*v, (T)1, e );
}

template <int N, typename T>
inline T
ZTuple<N,T>::min() const
{
	T minVal = data[0];
	FOR( i, 1, N ) { minVal = ZMin( minVal, data[i] ); }
	return minVal;
}

template <int N, typename T>
inline T
ZTuple<N,T>::max() const
{
	T maxVal = data[0];
	FOR( i, 1, N ) { maxVal = ZMax( maxVal, data[i] ); }
	return maxVal;
}

template <int N, typename T>
inline T
ZTuple<N,T>::absMin() const
{
	T minVal = data[0];
	FOR( i, 1, N ) { minVal = ZAbsMin( minVal, data[i] ); }
	return minVal;
}

template <int N, typename T>
inline T
ZTuple<N,T>::absMax() const
{
	T maxVal = data[0];
	FOR( i, 1, N ) { maxVal = ZAbsMax( maxVal, data[i] ); }
	return maxVal;
}

template <int N, typename T>
inline int
ZTuple<N,T>::minIndex() const
{
	int minIdx = 0;
	T minVal = data[0];

	FOR( i, 1, N )
	{
		if( data[i] < minVal )
		{
			minVal = data[i];
			minIdx = i;
		}
	}

	return minIdx;
}

template <int N, typename T>
inline int
ZTuple<N,T>::maxIndex() const
{
	int maxIdx = 0;
	T maxVal = data[0];

	FOR( i, 1, N )
	{
		if( data[i] > maxVal )
		{
			maxVal = data[i];
			maxIdx = i;
		}
	}

	return maxIdx;
}

template <int N, typename T>
inline int
ZTuple<N,T>::absMinIndex() const
{
	int minIdx = 0;
	T minVal = ZAbs(data[0]);

	FOR( i, 1, N )
	{
		const T D = ZABs(data[i]);

		if( D < minVal )
		{
			minVal = D;
			minIdx = i;
		}
	}

	return minIdx;
}

template <int N, typename T>
inline int
ZTuple<N,T>::absMaxIndex() const
{
	int maxIdx = 0;
	T maxVal = ZAbs(data[0]);

	FOR( i, 1, N )
	{
		const T D = ZABs(data[i]);

		if( D > maxVal )
		{
			maxVal = D;
			maxIdx = i;
		}
	}

	return maxIdx;
}

template <int N, typename T>
inline T
ZTuple<N,T>::sum() const
{
	T tmp = data[0];

	FOR( i, 1, N )
	{
		tmp += data[i];
	}

	return tmp;
}

template <int N, typename T>
inline T
ZTuple<N,T>::avg() const
{
	const T _N = (T)1 / (T)N;

	T tmp = data[0] * _N;

	FOR( i, 1, N )
	{
		tmp += data[i] * _N;
	}

	return tmp;
}

template <int N, typename T>
inline ZTuple<N,T>&
ZTuple<N,T>::setComponentwiseMin( const ZTuple<N,T>& v )
{
	FOR( i, 0, N )
	{
		if( v.data[i] < data[i] ) { data[i] = v.data[i]; }
	}

	return (*this);
}

template <int N, typename T>
inline ZTuple<N,T>&
ZTuple<N,T>::setComponentwiseMax( const ZTuple<N,T>& v )
{
	FOR( i, 0, N )
	{
		if( v.data[i] > data[i] ) { data[i] = v.data[i]; }
	}

	return (*this);
}

template <int N, typename T>
inline ZTuple<N,T>
ZTuple<N,T>::componentwiseMin( const ZTuple<N,T>& v ) const
{
	ZTuple<N,T> tmp;

	FOR( i, 0, N )
	{
		tmp[i] = ZMin( data[i], v.data[i] );
	}

	return tmp;
}

template <int N, typename T>
inline ZTuple<N,T>
ZTuple<N,T>::componentwiseMax( const ZTuple<N,T>& v ) const
{
	ZTuple<N,T> tmp;

	FOR( i, 0, N )
	{
		tmp[i] = ZMax( data[i], v.data[i] );
	}

	return tmp;
}

template <int N, typename T>
inline bool
ZTuple<N,T>::isZero() const
{
	if( data[0] != (T)0 ) { return false; }
	if( data[1] != (T)0 ) { return false; }
	if( data[2] != (T)0 ) { return false; }

	return true;
}

template <int N, typename T>
inline bool
ZTuple<N,T>::isAlmostZero( T eps ) const
{
	if( data[0] < -(T)Z_EPS ) { return false; }
	if( data[1] < -(T)Z_EPS ) { return false; }
	if( data[2] < -(T)Z_EPS ) { return false; }

	if( data[0] >  (T)Z_EPS ) { return false; }
	if( data[1] >  (T)Z_EPS ) { return false; }
	if( data[2] >  (T)Z_EPS ) { return false; }

	return true;
}

template <int N, typename T>
inline T
ZTuple<N,T>::l1Norm() const
{
	double sum = 0.0;
	FOR( i, 0, N ) { sum += ZAbs(data[i]); }
	return sum;
}

template <int N, typename T>
inline T
ZTuple<N,T>::l2Norm() const
{
	double sum = 0.0;
	FOR( i, 0, N ) { sum += ZPow2(data[i]); }
	return (T)sqrt(sum);
}

template <int N, typename T>
inline T
ZTuple<N,T>::lpNorm( int p ) const
{
	if( p <= 0 ) { return (T)0; }

	double sum = 0.0;
	FOR( i, 0, N ) { sum += pow( ZAbs(data[i]), p ); }
	return (T)pow( sum, 1/(double)p );
}

template <int N, typename T>
inline T
ZTuple<N,T>::infNorm() const
{
	T max = (T)0;
	FOR( i, 0, N ) { max = ZMax( max, ZAbs(data[i]) ); }
	return max;
}

template <int N, typename T>
inline void
ZTuple<N,T>::write( ofstream& fout ) const
{
	fout.write( (char*)&data, sizeof(T)*N );
}

template <int N, typename T>
inline void
ZTuple<N,T>::read( ifstream& fin )
{
	fin.read( (char*)&data, sizeof(T)*N );
}

template <int N, typename T>
inline ZTuple<N,T>
ZTuple<N,T>::zero()
{
	ZTuple<N,T> tmp;
	FOR( i, 0, N ) { tmp.data[i] = (T)0; }
	return tmp;
}

template <int N, typename T>
inline ZTuple<N,T>
ZTuple<N,T>::one()
{
	ZTuple<N,T> tmp;
	FOR( i, 0, N ) { tmp.data[i] = (T)1; }
	return tmp;
}

template <int N, typename T>
inline ZTuple<N,T>
ZTuple<N,T>::unitX()
{
	ZTuple<N,T> tmp;
	tmp[0] = (T)1;
	return tmp;
}

template <int N, typename T>
inline ZTuple<N,T>
ZTuple<N,T>::unitY()
{
	ZTuple<N,T> tmp;
	tmp[1] = (T)1;
	return tmp;
}

template <int N, typename T>
inline ZTuple<N,T>
ZTuple<N,T>::unitZ()
{
	ZTuple<N,T> tmp;
	tmp[2] = (T)1;
	return tmp;
}

template <int N, typename T>
inline ZTuple<N,T>
ZTuple<N,T>::xPosAxis()
{
	ZTuple<N,T> tmp;
	tmp[0] = (T)1;
	return tmp;
}

template <int N, typename T>
inline ZTuple<N,T>
ZTuple<N,T>::yPosAxis()
{
	ZTuple<N,T> tmp;
	tmp[1] = (T)1;
	return tmp;
}

template <int N, typename T>
inline ZTuple<N,T>
ZTuple<N,T>::zPosAxis()
{
	ZTuple<N,T> tmp;
	tmp[2] = (T)1;
	return tmp;
}

template <int N, typename T>
inline ZTuple<N,T>
ZTuple<N,T>::xNegAxis()
{
	ZTuple<N,T> tmp;
	tmp[0] = -(T)1;
	return tmp;
}

template <int N, typename T>
inline ZTuple<N,T>
ZTuple<N,T>::yNegAxis()
{
	ZTuple<N,T> tmp;
	tmp[1] = -(T)1;
	return tmp;
}

template <int N, typename T>
inline ZTuple<N,T>
ZTuple<N,T>::zNegAxis()
{
	ZTuple<N,T> tmp;
	tmp[2] = -(T)1;
	return tmp;
}

/////////////////////////////////////////
// simple non-member utility funcitons //
/////////////////////////////////////////

template <int N, typename T>
inline ZTuple<N,T>
operator*( const int& s, const ZTuple<N,T>& v )
{
	ZTuple<N,T> tmp( v );
	return ( tmp *= s );
}

template <int N, typename T>
inline ZTuple<N,T>
operator*( const float& s, const ZTuple<N,T>& v )
{
	ZTuple<N,T> tmp( v );
	return ( tmp *= s );
}

template <int N, typename T>
inline ZTuple<N,T>
operator*( const double& s, const ZTuple<N,T>& v )
{
	ZTuple<N,T> tmp( v );
	return ( tmp *= s );
}

template <int N, typename T>
inline ostream&
operator<<( ostream& os, const ZTuple<N,T>& v )
{
	os << "( " << v.data[0];
	FOR( i, 1, N ) { os << ", " << v.data[i]; } os<<" )";
	return os;
}

template <int N, typename T>
inline istream&
operator>>( istream& is, ZTuple<N,T>& v )
{
	FOR( i, 0, N ) { is >> v.data[i]; }
	return is;
}

template <int N, typename T>
inline T
MAG( const ZTuple<N,T>& A )
{
	double sum = ZPow2( A.data[0] );

	FOR( i, 1, N )
	{
		sum += ZPow2( A.data[i] );
	}

	return (T)sqrt( sum );
}

template <int N, typename T>
inline T
LEN( const ZTuple<N,T>& A )
{
	double sum = ZPow2( A.data[0] );

	FOR( i, 1, N )
	{
		sum += ZPow2( A.data[i] );
	}

	return (T)sqrt( sum );
}

template <int N, typename T>
inline T
DST( const ZTuple<N,T>& A, const ZTuple<N,T>& B )
{
	T sum = ZPow2( A.data[0] - B.data[0] );

	FOR( i, 1, N )
	{
		sum += ZPow2( A.data[i] - B.data[i] );
	}

	return (T)sqrt( sum );
}

template <int N, typename T>
inline void
ADD( ZTuple<N,T>& A, const ZTuple<N,T>& B, const ZTuple<N,T>& C, bool useOpenMP=false )
{
	#pragma omp parallel for if( useOpenMP )
	FOR( i, 0, N )
	{
		A.data[i] = B.data[i] + C.data[i];
	}
}

template <int N, typename T>
inline void
SUB( ZTuple<N,T>& A, const ZTuple<N,T>& B, const ZTuple<N,T>& C, bool useOpenMP=false )
{
	#pragma omp parallel for if( useOpenMP )
	FOR( i, 0, N )
	{
		A.data[i] = B.data[i] - C.data[i];
	}
}

template <int N, typename T>
inline void
INC( ZTuple<N,T>& A, const ZTuple<N,T>& B, bool useOpenMP=false )
{
	#pragma omp parallel for if( useOpenMP )
	FOR( i, 0, N )
	{
		A.data[i] += B.data[i];
	}
}

template <int N, typename T>
inline void
MUL( ZTuple<N,T>& A, T b, const ZTuple<N,T>& B, bool useOpenMP=false )
{
	#pragma omp parallel for if( useOpenMP )
	FOR( i, 0, N )
	{
		A.data[i] = b * B.data[i];
	}
}

template <int N, typename T>
inline void
INCMUL( ZTuple<N,T>& A, T b, const ZTuple<N,T>& B, bool useOpenMP=false )
{
	#pragma omp parallel for if( useOpenMP )
	FOR( i, 0, N )
	{
		A.data[i] += b * B.data[i];
	}
}

template <int N, typename T>
inline void
SUBMUL( ZTuple<N,T>& A, T b, const ZTuple<N,T>& B, bool useOpenMP=false )
{
	#pragma omp parallel for if( useOpenMP )
	FOR( i, 0, N )
	{
		A.data[i] -= b * B.data[i];
	}
}

template <int N, typename T>
inline T
DOT( const ZTuple<N,T>& a, const ZTuple<N,T>& b, bool useOpenMP=false )
{
	T sum = (T)0;

	#pragma omp parallel for reduction( +: sum ) if( useOpenMP )
	FOR( i, 0, N )
	{
		sum += a.data[i] * b.data[i];
	}

	return sum;

}

template <int N, typename T>
inline ZTuple<N,T>
CRS( const ZTuple<N,T>& a, const ZTuple<N,T>& b )
{
	return ZTuple<N,T>
	(
		( a.data[1] * b.data[2] ) - ( a.data[2] * b.data[1] ),
		( a.data[2] * b.data[0] ) - ( a.data[0] * b.data[2] ),
		( a.data[0] * b.data[1] ) - ( a.data[1] * b.data[0] )
	);
}

////////////////
// data types //
////////////////

typedef ZTuple<2,char>	 ZChar2;
typedef ZTuple<2,int>    ZInt2;
typedef ZTuple<2,float>  ZFloat2;
typedef ZTuple<2,double> ZDouble2;

typedef ZTuple<3,char>	 ZChar3;
typedef ZTuple<3,int>    ZInt3;
typedef ZTuple<3,float>  ZFloat3;
typedef ZTuple<3,double> ZDouble3;

typedef ZTuple<4,char>	 ZChar4;
typedef ZTuple<4,int>    ZInt4;
typedef ZTuple<4,float>  ZFloat4;
typedef ZTuple<4,double> ZDouble4;

typedef ZTuple<5,char>	 ZChar5;
typedef ZTuple<5,int>    ZInt5;
typedef ZTuple<5,float>  ZFloat5;
typedef ZTuple<5,double> ZDouble5;

typedef ZTuple<6,char>	 ZChar6;
typedef ZTuple<6,int>    ZInt6;
typedef ZTuple<6,float>  ZFloat6;
typedef ZTuple<6,double> ZDouble6;

typedef ZTuple<7,char>	 ZChar7;
typedef ZTuple<7,int>    ZInt7;
typedef ZTuple<7,float>  ZFloat7;
typedef ZTuple<7,double> ZDouble7;

typedef ZTuple<8,char>	 ZChar8;
typedef ZTuple<8,int>    ZInt8;
typedef ZTuple<8,float>  ZFloat8;
typedef ZTuple<8,double> ZDouble8;


typedef ZTuple<2,float>   ZVec2;
typedef ZTuple<3,float>   ZVec3;
typedef ZTuple<4,float>   ZVec4;
typedef ZTuple<5,float>   ZVec5;
typedef ZTuple<6,float>   ZVec6;
typedef ZTuple<7,float>   ZVec7;
typedef ZTuple<8,float>   ZVec8;
typedef ZTuple<9,float>   ZVec9;

typedef ZTuple<2,float>   ZVec2f;
typedef ZTuple<3,float>   ZVec3f;
typedef ZTuple<4,float>   ZVec4f;
typedef ZTuple<5,float>   ZVec5f;
typedef ZTuple<6,float>   ZVec6f;
typedef ZTuple<7,float>   ZVec7f;
typedef ZTuple<8,float>   ZVec8f;
typedef ZTuple<9,float>   ZVec9f;

typedef ZTuple<2,double>  ZVec2d;
typedef ZTuple<3,double>  ZVec3d;
typedef ZTuple<4,double>  ZVec4d;
typedef ZTuple<5,double>  ZVec5d;
typedef ZTuple<6,double>  ZVec6d;
typedef ZTuple<7,double>  ZVec7d;
typedef ZTuple<8,double>  ZVec8d;
typedef ZTuple<9,double>  ZVec9d;

ZELOS_NAMESPACE_END

#endif

