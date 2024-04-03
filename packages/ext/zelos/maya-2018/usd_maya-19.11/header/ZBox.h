//--------//
// ZBox.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.03.08                               //
//-------------------------------------------------------//

#ifndef _ZBox_h_
#define _ZBox_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

/// @brief A n-dimensional box class.
template <int N, typename T>
class ZBox
{
	public:

		bool _initialized;

		ZTuple<N,T> _min;			///< the minimum corner point
		ZTuple<N,T> _max;			///< the maximum corner point

	public:

		ZBox();
		ZBox( const ZBox& b );
		ZBox( const ZTuple<N,T>& p0 );
		ZBox( const ZTuple<N,T>& p1, const ZTuple<N,T>& p2 );
		ZBox( const ZTuple<N,T>& p1, const ZTuple<N,T>& p2, const ZTuple<N,T>& p3 );

		void reset();

		ZBox& set( const ZTuple<N,T>& p1 );
		ZBox& set( const ZTuple<N,T>& p1, const ZTuple<N,T>& p2 );
		ZBox& set( const ZTuple<N,T>& p1, const ZTuple<N,T>& p2, const ZTuple<N,T>& p3 );

		ZBox& operator=( const ZBox& b );

		ZBox& operator*=( T scale );

		ZBox& expand( ZTuple<N,T>& p );
		ZBox& expand( const ZBox<N,T>& b );
		ZBox& expand( T epsilon=(T)1e-30 );

		ZBox& merge( const ZBox& b0, const ZBox b1 );

		bool initialized() const;

		const ZTuple<N,T>& minPoint() const;
		const ZTuple<N,T>& maxPoint() const;

		ZTuple<N,T> center() const;

		T xWidth() const;
		T yWidth() const;
		T zWidth() const;

		T width( int dimension ) const;

		T minWidth() const;
		T maxWidth() const;

		int maxDimension() const;

		bool contains( const ZTuple<N,T>& p ) const;
		bool contains( const ZTuple<N,T>& p, T epsilon ) const;

		bool intersects( const ZBox<N,T>& b ) const;
};

template <int N, typename T>
inline
ZBox<N,T>::ZBox()
: _initialized(false)
{}

template <int N, typename T>
inline
ZBox<N,T>::ZBox( const ZBox<N,T>& b )
: _initialized(b._initialized), _min(b._min), _max(b._max)
{}

template <int N, typename T>
inline
ZBox<N,T>::ZBox( const ZTuple<N,T>& p1 )
{
	_initialized = true;

	_min = _max = p1;
}

template <int N, typename T>
inline
ZBox<N,T>::ZBox( const ZTuple<N,T>& p1, const ZTuple<N,T>& p2 )
{
	_initialized = true;

	FOR( i, 0, N )
	{
		ZMinMax( p1.data[i], p2.data[i], _min.data[i], _max.data[i] );
	}
}

template <int N, typename T>
inline
ZBox<N,T>::ZBox( const ZTuple<N,T>& p1, const ZTuple<N,T>& p2, const ZTuple<N,T>& p3 )
{
	_initialized = true;

	FOR( i, 0, N )
	{
		ZMinMax( p1.data[i], p2.data[i], p3.data[i], _min.data[i], _max.data[i] );
	}
}

template <int N, typename T>
inline ZBox<N,T>&
ZBox<N,T>::set( const ZTuple<N,T>& p1 )
{
	_initialized = true;

	_min = _max = p1;

	return (*this);
}

template <int N, typename T>
inline ZBox<N,T>&
ZBox<N,T>::set( const ZTuple<N,T>& p1, const ZTuple<N,T>& p2 )
{
	_initialized = true;

	FOR( i, 0, N )
	{
		ZMinMax( p1.data[i], p2.data[i], _min.data[i], _max.data[i] );
	}

	return (*this);
}

template <int N, typename T>
inline ZBox<N,T>&
ZBox<N,T>::set( const ZTuple<N,T>& p1, const ZTuple<N,T>& p2, const ZTuple<N,T>& p3 )
{
	_initialized = true;

	FOR( i, 0, N )
	{
		ZMinMax( p1.data[i], p2.data[i], p3.data[i], _min.data[i], _max.data[i] );
	}

	return (*this);
}

template <int N, typename T>
inline void
ZBox<N,T>::reset()
{
	_initialized = false;

	_min.zeroize();
	_max.zeroize();
}

template <int N, typename T>
inline ZBox<N,T>&
ZBox<N,T>::operator=( const ZBox& b )
{
	_initialized = b._initialized;

	_min = b._min;
	_max = b._max;

	return (*this);
}

template <int N, typename T>
inline ZBox<N,T>&
ZBox<N,T>::operator*=( T scale )
{
	if( _initialized )
	{
		FOR( i, 0, N )
		{
			const T c = (T)0.5 * ( _min.data[i] + _max.data[i] );

			_min.data[i] = ( scale * ( _min.data[i] - c ) ) + c;
			_max.data[i] = ( scale * ( _max.data[i] - c ) ) + c;
		}
	}

	return (*this);
}

template <int N, typename T>
inline ZBox<N,T>&
ZBox<N,T>::expand( ZTuple<N,T>& p )
{
	if( _initialized ) {

		FOR( i, 0, N )
		{
			_min.data[i] = ZMin( _min.data[i], p.data[i] );
			_max.data[i] = ZMax( _max.data[i], p.data[i] );
		}

	} else {

		_initialized = true;

		_min = _max = p;

	}

	return (*this);
}

template <int N, typename T>
inline ZBox<N,T>&
ZBox<N,T>::expand( const ZBox<N,T>& b )
{
	if( !b._initialized ) {

		return (*this);

	} else if( !_initialized ) {

		this->operator=( b );
		return (*this);

	} else {

		FOR( i, 0, N )
		{
			_min.data[i] = ZMin( _min.data[i], b._min.data[i] );
			_max.data[i] = ZMax( _max.data[i], b._max.data[i] );
		}

		return (*this);
	}
}

template <int N, typename T>
inline ZBox<N,T>&
ZBox<N,T>::expand( T e )
{
	if( _initialized )
	{
		FOR( i, 0, N )
		{
			_min.data[i] -= e;
			_max.data[i] += e;
		}
	}

	return (*this);
}

template <int N, typename T>
inline ZBox<N,T>&
ZBox<N,T>::merge( const ZBox<N,T>& b0, const ZBox<N,T> b1 )
{
	if( b0._initialized )
	{
		const ZTuple<N,T>& minPt = b0.minPoint();
		const ZTuple<N,T>& maxPt = b0.maxPoint();

		FOR( i, 0, 3 )
		{
			_min.data[i] = ZMin( _min.data[i], minPt.data[i] );
			_max.data[i] = ZMax( _max.data[i], maxPt.data[i] );
		}

		_initialized = true;
	}

	if( b1._initialized )
	{
		const ZTuple<N,T>& minPt = b1.minPoint();
		const ZTuple<N,T>& maxPt = b1.maxPoint();

		FOR( i, 0, 3 )
		{
			_min.data[i] = ZMin( _min.data[i], minPt.data[i] );
			_max.data[i] = ZMax( _max.data[i], maxPt.data[i] );
		}

		_initialized = true;
	}

	return (*this);
}

template <int N, typename T>
inline bool
ZBox<N,T>::initialized() const
{
	return _initialized;
}

template <int N, typename T>
inline const ZTuple<N,T>&
ZBox<N,T>::minPoint() const
{
	return _min;
}

template <int N, typename T>
inline const ZTuple<N,T>&
ZBox<N,T>::maxPoint() const
{
	return _max;
}

template <int N, typename T>
inline ZTuple<N,T>
ZBox<N,T>::center() const
{
	ZTuple<N,T> c;

	FOR( i, 0, N )
	{
		c.data[i] = (T)0.5 * ( _min.data[i] + _max.data[i] );
	}

	return c;
}

template <int N, typename T>
inline T
ZBox<N,T>::xWidth() const
{
	return ( _max.data[0] - _min.data[0] );
}

template <int N, typename T>
inline T
ZBox<N,T>::yWidth() const
{
	return ( _max.data[1] - _min.data[1] );
}

template <int N, typename T>
inline T
ZBox<N,T>::zWidth() const
{
	return ( _max.data[2] - _min.data[2] );
}

template <int N, typename T>
inline T
ZBox<N,T>::width( int i ) const
{
	return ( _max.data[i] - _min.data[i] );
}

template <int N, typename T>
inline T
ZBox<N,T>::minWidth() const
{
	T w = (T)1e-30;

	FOR( i, 0, N )
	{
		w = ZMin( w, _max.data[i] - _min.data[i] );
	}

	return w;
}

template <int N, typename T>
inline T
ZBox<N,T>::maxWidth() const
{
	T w = -(T)1e-30;

	FOR( i, 0, N )
	{
		w = ZMax( w, _max.data[i] - _min.data[i] );
	}

	return w;
}

template <int N, typename T>
inline int
ZBox<N,T>::maxDimension() const
{
	T wMax = -(T)1e-30, w;
	int idx = 0;

	FOR( i, 0, N )
	{
		w = _max.data[i] - _min.data[i];

		if( w > wMax )
		{
			wMax = w;
			idx = i;
		}
	}

	return idx;
}

template <int N, typename T>
inline bool
ZBox<N,T>::contains( const ZTuple<N,T>& p ) const
{
	if( !_initialized ) { return false; }

	FOR( i, 0, N )
	{
		if( p.data[i] < _min.data[i] ) { return false; }
		if( p.data[i] > _max.data[i] ) { return false; }
	}

	return true;
}

template <int N, typename T>
inline bool
ZBox<N,T>::contains( const ZTuple<N,T>& p, T e ) const
{
	if( !_initialized ) { return false; }

	FOR( i, 0, N )
	{
		if( p.data[i] < (_min.data[i]-e) ) { return false; }
		if( p.data[i] > (_max.data[i]+e) ) { return false; }
	}

	return true;
}

template <int N, typename T>
inline bool
ZBox<N,T>::intersects( const ZBox<N,T>& b ) const
{
	if( !_initialized ) { return false; }

	FOR( i, 0, N )
	{
		if( _max.data[i] < b._min.data[i] ) { return false; }
		if( _min.data[i] > b._max.data[i] ) { return false; }
	}

	return true;
}

/////////////////////////////////////////
// simple non-member utility funcitons //
/////////////////////////////////////////

template <int N, typename T>
inline ZTuple<N,T>
AverageCellSize( const vector<ZBox<N,T> >& boxes )
{
	const int n = (int)boxes.size();
	const T _n = (T)1 / (T)n;

	ZTuple<N,T> cellSize;

	FOR( i, 0, n )
	{
		FOR( j, 0, N )
		{
			cellSize.data[j] += boxes[i].width(j) * _n;
		}
	}

	return cellSize;
}

template <int N, typename T>
inline ostream&
operator<<( ostream& os, const ZBox<N,T>& object )
{
	os << "<ZBox>" << endl;
	os << " domain: " << object.minPoint() << " ~ " << object.maxPoint() << endl;

	if( N == 2 ) {

		os << " size  : " << object.width(0) << " x " << object.width(1) << " x " << object.width(2) << endl;

	} else if( N == 3 ) {

		os << " size  : " << object.width(0) << " x " << object.width(1) << endl;

	}

	return os;
}

////////////////
// data types //
////////////////

typedef ZBox<2,float>  ZBox2f;
typedef ZBox<2,double> ZBox2d;

typedef ZBox<3,float>  ZBox3f;
typedef ZBox<3,double> ZBox3d;

ZELOS_NAMESPACE_END

#endif

