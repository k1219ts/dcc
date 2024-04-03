//----------------//
// ZIntSetArray.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.03.08                               //
//-------------------------------------------------------//

#ifndef _ZIntSetArray_h_
#define _ZIntSetArray_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZIntSetArray
{
	private:

		ZIntArray _n;		// # of elements           (to be saved)
		ZIntArray _i;		// start index of each set (not to be saved)
		ZIntArray _v;		// data                    (to be saved)

	public:

		ZIntSetArray();
		ZIntSetArray( const ZIntSetArray& source );
		ZIntSetArray( const ZIntArray& numElements );
		ZIntSetArray( const char* filePathName );

		void reset();

		ZIntSetArray& operator=( const ZIntSetArray& source );

		void set( const ZIntArray& numElements );

		void from( const ZIntSetArray& other, const ZCharArray& mask );

		void fill( int valueForAll );
		void zeroize();

		int numSets() const;
		int numTotalElements() const;
		int count( const int& i ) const;

		int& operator[]( const int& i );
		const int& operator[]( const int& i ) const;

		int& operator()( const int& i, const int& j );
		const int& operator()( const int& i, const int& j ) const;

		int& start( const int& i );
		const int& start( const int& i ) const;

		int& end( const int& i );
		const int& end( const int& i ) const;

		void getStartElements( ZIntArray& elements ) const;
		void getEndElements( ZIntArray& elements ) const;

		void append( const ZIntSetArray& other );

		void exchangeData( ZIntSetArray& other );

		const ZIntArray& n() const;
		const ZIntArray& i() const;
		const ZIntArray& v() const;

		void write( ofstream& fout ) const;
		void read( ifstream& fin );

		bool save( const char* filePathName ) const;
		bool load( const char* filePathName );

		double usedMemorySize( ZDataUnit::DataUnit dataUnit=ZDataUnit::zBytes ) const;

		void print() const;

	private:

		void _init();
		void _allocate();
};

inline int
ZIntSetArray::numSets() const
{
	return (int)_n.size();
}

inline int
ZIntSetArray::numTotalElements() const
{
	return _v.length();
}

inline int
ZIntSetArray::count( const int& i ) const
{
	return _n[i];
}

inline int&
ZIntSetArray::operator[]( const int& i )
{
	return _v[i];
}

inline const
int& ZIntSetArray::operator[]( const int& i ) const
{
	return _v[i];
}

inline int&
ZIntSetArray::operator()( const int& i, const int& j )
{
	return _v[ _i[i] + j ];
}

inline const
int& ZIntSetArray::operator()( const int& i, const int& j ) const
{
	return _v[ _i[i] + j ];
}

inline int&
ZIntSetArray::start( const int& i )
{
	return _v[ _i[i] ];
}

inline const int&
ZIntSetArray::start( const int& i ) const
{
	return _v[ _i[i] ];
}

inline int&
ZIntSetArray::end( const int& i )
{
	//return _v[ _i[i-1] - 1 ]; // it will fail for the last set
	return _v[ _i[i] + _n[i] - 1 ];
}

inline const int&
ZIntSetArray::end( const int& i ) const
{
	//return _v[ _i[i-1] - 1 ]; // it will fail for the last set
	return _v[ _i[i] + _n[i] - 1 ];
}

inline const ZIntArray&
ZIntSetArray::n() const
{
	return _n;
}

inline const ZIntArray&
ZIntSetArray::i() const
{
	return _i;
}

inline const ZIntArray&
ZIntSetArray::v() const
{
	return _v;
}

ostream&
operator<<( ostream& os, const ZIntSetArray& object );

ZELOS_NAMESPACE_END

#endif

