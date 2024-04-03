//-----------------//
// ZIntArrayList.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.07.04                               //
//-------------------------------------------------------//

#ifndef _ZIntArrayList_h_
#define _ZIntArrayList_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZIntArrayList : public vector<ZIntArray>
{
	private:

		typedef vector<ZIntArray> parent;

	public:

		ZIntArrayList();
		ZIntArrayList( const ZIntArrayList& source );
		ZIntArrayList( int m, int n=0 );
		ZIntArrayList( const char* filePathName );

		void reset();

		const int& operator()( const int& i, const int& j ) const;
		int& operator()( const int& i, const int& j );

		ZIntArrayList& operator=( const ZIntArrayList& other );

		void fill( int valueForAll );

		void setSize( int m, int n=0 );
		void setLength( int i, int n );

		int length() const;
		int count( const int& i ) const;
		int totalCount() const;
		int maxCount() const;

		void append( const ZIntArray& array );

		void write( ofstream& fout ) const;
		void read( ifstream& fin );

		bool save( const char* filePathName ) const;
		bool load( const char* filePathName );
};

inline const int&
ZIntArrayList::operator()( const int& i, const int& j ) const
{
	return parent::at(i)[j];
}

inline int&
ZIntArrayList::operator()( const int& i, const int& j )
{
	return parent::at(i)[j];
}

ostream&
operator<<( ostream& os, const ZIntArrayList& object );

ZELOS_NAMESPACE_END

#endif

