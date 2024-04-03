//-------------------//
// ZFloatArrayList.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.07.05                               //
//-------------------------------------------------------//

#ifndef _ZFloatArrayList_h_
#define _ZFloatArrayList_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZFloatArrayList : public vector<ZFloatArray>
{
	private:

		typedef vector<ZFloatArray> parent;

	public:

		ZFloatArrayList();
		ZFloatArrayList( const ZFloatArrayList& source );
		ZFloatArrayList( int m, int n=0 );
		ZFloatArrayList( const char* filePathName );

		void reset();

		const float& operator()( const int& i, const int& j ) const;
		float& operator()( const int& i, const int& j );

		ZFloatArrayList& operator=( const ZFloatArrayList& other );

		void fill( float valueForAll );

		void setSize( int m, int n=0 );
		void setLength( int i, int n );

		int length() const;
		int count( const int& i ) const;
		int totalCount() const;
		int maxCount() const;

		void append( const ZFloatArray& array );

		void write( ofstream& fout ) const;
		void read( ifstream& fin );

		bool save( const char* filePathName ) const;
		bool load( const char* filePathName );
};

inline const float&
ZFloatArrayList::operator()( const int& i, const int& j ) const
{
	return parent::operator[](i)[j];
}

inline float&
ZFloatArrayList::operator()( const int& i, const int& j )
{
	return parent::operator[](i)[j];
}

ostream&
operator<<( ostream& os, const ZFloatArrayList& object );

ZELOS_NAMESPACE_END

#endif

