//------------------//
// ZFloatSetArray.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.03.08                               //
//-------------------------------------------------------//

#ifndef _ZFloatSetArray_h_
#define _ZFloatSetArray_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZFloatSetArray
{
	private:

		ZIntArray   _n;		// # of elements           (to be saved)
		ZIntArray   _i;		// start index of each set (not to be saved)
		ZFloatArray _v;		// data                    (to be saved)

	public:

		ZFloatSetArray();
		ZFloatSetArray( const ZFloatSetArray& source );
		ZFloatSetArray( const ZIntArray& numElements );
		ZFloatSetArray( const char* filePathName );

		void reset();

		ZFloatSetArray& operator=( const ZFloatSetArray& source );

		void set( const ZIntArray& numElements );

		void from( const ZFloatSetArray& other, const ZCharArray& mask );

		void fill( float valueForAll );
		void zeroize();

		int numSets() const;
		int numTotalElements() const;
		int count( const int& i ) const;

		float& operator[]( const int& i );
		const float& operator[]( const int& i ) const;

		float& operator()( const int& i, const int& j );
		const float& operator()( const int& i, const int& j ) const;

		float& start( const int& i );
		const float& start( const int& i ) const;

		float& end( const int& i );
		const float& end( const int& i ) const;

		void getStartElements( ZFloatArray& elements ) const;
		void getEndElements( ZFloatArray& elements ) const;

		void append( const ZFloatSetArray& other );

		void exchangeData( ZFloatSetArray& other );

		const ZIntArray& n() const;
		const ZIntArray& i() const;
		const ZFloatArray& v() const;

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
ZFloatSetArray::numSets() const
{
	return (int)_n.size();
}

inline int
ZFloatSetArray::numTotalElements() const
{
	return _v.length();
}

inline int
ZFloatSetArray::count( const int& i ) const
{
	return _n[i];
}

inline float&
ZFloatSetArray::operator[]( const int& i )
{
	return _v[i];
}

inline const
float& ZFloatSetArray::operator[]( const int& i ) const
{
	return _v[i];
}

inline float&
ZFloatSetArray::operator()( const int& i, const int& j )
{
	return _v[ _i[i] + j ];
}

inline const
float& ZFloatSetArray::operator()( const int& i, const int& j ) const
{
	return _v[ _i[i] + j ];
}

inline float&
ZFloatSetArray::start( const int& i )
{
	return _v[ _i[i] ];
}

inline const float&
ZFloatSetArray::start( const int& i ) const
{
	return _v[ _i[i] ];
}

inline float&
ZFloatSetArray::end( const int& i )
{
	//return _v[ _i[i-1] - 1 ]; // it will fail for the last set
	return _v[ _i[i] + _n[i] - 1 ];
}

inline const float&
ZFloatSetArray::end( const int& i ) const
{
	//return _v[ _i[i-1] - 1 ]; // it will fail for the last set
	return _v[ _i[i] + _n[i] - 1 ];
}

inline const ZIntArray&
ZFloatSetArray::n() const
{
	return _n;
}

inline const ZIntArray&
ZFloatSetArray::i() const
{
	return _i;
}

inline const ZFloatArray&
ZFloatSetArray::v() const
{
	return _v;
}

ostream&
operator<<( ostream& os, const ZFloatSetArray& object );

ZELOS_NAMESPACE_END

#endif

