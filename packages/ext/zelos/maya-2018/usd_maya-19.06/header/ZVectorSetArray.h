//-------------------//
// ZVectorSetArray.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.03.08                               //
//-------------------------------------------------------//

#ifndef _ZVectorSetArray_h_
#define _ZVectorSetArray_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZVectorSetArray
{
	private:

		ZIntArray    _n;	// # of elements           (to be saved)
		ZIntArray    _i;	// start index of each set (not to be saved)
		ZVectorArray _v;	// data                    (to be saved)

	public:

		ZVectorSetArray();
		ZVectorSetArray( const ZVectorSetArray& source );
		ZVectorSetArray( const ZIntArray& numElements );
		ZVectorSetArray( const char* filePathName );

		void reset();

		ZVectorSetArray& operator=( const ZVectorSetArray& source );

		void set( const ZIntArray& numElements );

		void fill( const ZVector& valueForAll );
		void zeroize();

		int numSets() const;
		int numTotalElements() const;
		int count( const int& i ) const;

		ZVector& operator[]( const int& i );
		const ZVector& operator[]( const int& i ) const;

		ZVector& operator()( const int& i, const int& j );
		const ZVector& operator()( const int& i, const int& j ) const;

		ZVector& start( const int& i );
		const ZVector& start( const int& i ) const;

		ZVector& end( const int& i );
		const ZVector& end( const int& i ) const;

		ZBoundingBox boundingBox( bool onlyEndPoints=false, bool useOpenMP=true ) const;

		void getStartElements( ZVectorArray& elements ) const;
		void getEndElements( ZVectorArray& elements ) const;

		void append( const ZVectorSetArray& other );

		void exchangeData( ZVectorSetArray& other );

		const ZIntArray& n() const;
		const ZIntArray& i() const;
		const ZVectorArray& v() const;

		void write( ofstream& fout ) const;
		void read( ifstream& fin );

		bool save( const char* filePathName ) const;
		bool load( const char* filePathName );

		double usedMemorySize( ZDataUnit::DataUnit dataUnit=ZDataUnit::zBytes ) const;

	private:

		void _init();
		void _allocate();
};

inline int
ZVectorSetArray::numSets() const
{
	return (int)_n.size();
}

inline int
ZVectorSetArray::numTotalElements() const
{
	return _v.length();
}

inline int
ZVectorSetArray::count( const int& i ) const
{
	return _n[i];
}

inline ZVector&
ZVectorSetArray::operator[]( const int& i )
{
	return _v[i];
}

inline const
ZVector& ZVectorSetArray::operator[]( const int& i ) const
{
	return _v[i];
}

inline ZVector&
ZVectorSetArray::operator()( const int& i, const int& j )
{
	return _v[ _i[i] + j ];
}

inline const
ZVector& ZVectorSetArray::operator()( const int& i, const int& j ) const
{
	return _v[ _i[i] + j ];
}

inline ZVector&
ZVectorSetArray::start( const int& i )
{
	return _v[ _i[i] ];
}

inline const ZVector&
ZVectorSetArray::start( const int& i ) const
{
	return _v[ _i[i] ];
}

inline ZVector&
ZVectorSetArray::end( const int& i )
{
	//return _v[ _i[i-1] - 1 ]; // it will fail for the last set
	return _v[ _i[i] + _n[i] - 1 ];
}

inline const ZVector&
ZVectorSetArray::end( const int& i ) const
{
	//return _v[ _i[i-1] - 1 ]; // it will fail for the last set
	return _v[ _i[i] + _n[i] - 1 ];
}

inline const ZIntArray&
ZVectorSetArray::n() const
{
	return _n;
}

inline const ZIntArray&
ZVectorSetArray::i() const
{
	return _i;
}

inline const ZVectorArray&
ZVectorSetArray::v() const
{
	return _v;
}

ostream&
operator<<( ostream& os, const ZVectorSetArray& object );

typedef ZVectorSetArray ZPointSetArray;

ZELOS_NAMESPACE_END

#endif

