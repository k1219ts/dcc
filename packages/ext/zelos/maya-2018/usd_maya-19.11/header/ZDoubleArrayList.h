//--------------------//
// ZDoubleArrayList.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.07.05                               //
//-------------------------------------------------------//

#ifndef _ZDoubleArrayList_h_
#define _ZDoubleArrayList_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZDoubleArrayList : public vector<ZDoubleArray>
{
	private:

		typedef vector<ZDoubleArray> parent;

	public:

		ZDoubleArrayList();
		ZDoubleArrayList( const ZDoubleArrayList& source );
		ZDoubleArrayList( int m, int n=0 );
		ZDoubleArrayList( const char* filePathName );

		void reset();

		const double& operator()( const int& i, const int& j ) const;
		double& operator()( const int& i, const int& j );

		ZDoubleArrayList& operator=( const ZDoubleArrayList& other );

		void fill( double valueForAll );

		void setSize( int m, int n=0 );
		void setLength( int i, int n );

		int length() const;
		int count( const int& i ) const;
		int totalCount() const;
		int maxCount() const;

		void append( const ZDoubleArray& array );

		void write( ofstream& fout ) const;
		void read( ifstream& fin );

		bool save( const char* filePathName ) const;
		bool load( const char* filePathName );
};

inline const double&
ZDoubleArrayList::operator()( const int& i, const int& j ) const
{
	return parent::operator[](i)[j];
}

inline double&
ZDoubleArrayList::operator()( const int& i, const int& j )
{
	return parent::operator[](i)[j];
}

ostream&
operator<<( ostream& os, const ZDoubleArrayList& object );

ZELOS_NAMESPACE_END

#endif

