//-------------------//
// ZDoubleSetArray.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.03.08                               //
//-------------------------------------------------------//

#ifndef _ZDoubleSetArray_h_
#define _ZDoubleSetArray_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZDoubleSetArray
{
	private:

		ZIntArray    _n;	// # of elements           (to be saved)
		ZIntArray    _i;	// start index of each set (not to be saved)
		ZDoubleArray _v;	// data                    (to be saved)

	public:

		ZDoubleSetArray();
		ZDoubleSetArray( const ZDoubleSetArray& source );
		ZDoubleSetArray( const ZIntArray& numElements );
		ZDoubleSetArray( const char* filePathName );

		void reset();

		ZDoubleSetArray& operator=( const ZDoubleSetArray& source );

		void set( const ZIntArray& numElements );

		void from( const ZDoubleSetArray& other, const ZCharArray& mask );

		void fill( double valueForAll );
		void zeroize();

		int numSets() const;
		int numTotalElements() const;
		int count( const int& i ) const;

		double& operator[]( const int& i );
		const double& operator[]( const int& i ) const;

		double& operator()( const int& i, const int& j );
		const double& operator()( const int& i, const int& j ) const;

		double& start( const int& i );
		const double& start( const int& i ) const;

		double& end( const int& i );
		const double& end( const int& i ) const;

		void getStartElements( ZDoubleArray& elements ) const;
		void getEndElements( ZDoubleArray& elements ) const;

		void append( const ZDoubleSetArray& other );

		void exchangeData( ZDoubleSetArray& other );

		const ZIntArray& n() const;
		const ZIntArray& i() const;
		const ZDoubleArray& v() const;

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
ZDoubleSetArray::numSets() const
{
	return (int)_n.size();
}

inline int
ZDoubleSetArray::numTotalElements() const
{
	return _v.length();
}

inline int
ZDoubleSetArray::count( const int& i ) const
{
	return _n[i];
}

inline double&
ZDoubleSetArray::operator[]( const int& i )
{
	return _v[i];
}

inline const
double& ZDoubleSetArray::operator[]( const int& i ) const
{
	return _v[i];
}

inline double&
ZDoubleSetArray::operator()( const int& i, const int& j )
{
	return _v[ _i[i] + j ];
}

inline const
double& ZDoubleSetArray::operator()( const int& i, const int& j ) const
{
	return _v[ _i[i] + j ];
}

inline double&
ZDoubleSetArray::start( const int& i )
{
	return _v[ _i[i] ];
}

inline const double&
ZDoubleSetArray::start( const int& i ) const
{
	return _v[ _i[i] ];
}

inline double&
ZDoubleSetArray::end( const int& i )
{
	//return _v[ _i[i-1] - 1 ]; // it will fail for the last set
	return _v[ _i[i] + _n[i] - 1 ];
}

inline const double&
ZDoubleSetArray::end( const int& i ) const
{
	//return _v[ _i[i-1] - 1 ]; // it will fail for the last set
	return _v[ _i[i] + _n[i] - 1 ];
}

inline const ZIntArray&
ZDoubleSetArray::n() const
{
	return _n;
}

inline const ZIntArray&
ZDoubleSetArray::i() const
{
	return _i;
}

inline const ZDoubleArray&
ZDoubleSetArray::v() const
{
	return _v;
}

ostream&
operator<<( ostream& os, const ZDoubleSetArray& object );

ZELOS_NAMESPACE_END

#endif

