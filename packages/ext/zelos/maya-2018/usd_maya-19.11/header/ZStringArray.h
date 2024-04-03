//----------------//
// ZStringArray.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.09.28                               //
//-------------------------------------------------------//

#ifndef _ZStringArray_h_
#define _ZStringArray_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZStringArray : public ZArray<ZString>
{
	private:

		typedef ZArray<ZString> parent;

	public:

		ZStringArray();
		ZStringArray( const ZStringArray& a );
		ZStringArray( int initialLength );
		ZStringArray( int initialLength, const ZString& valueForAll );

		void setLength( int length );

		ZString& append( const char* str );
		ZString& append( const ZString& str );
		void append( const ZStringArray& a );

		ZString combine( const ZString& separator=ZString("") ) const;

		void write( ofstream& fout ) const;
		void read( ifstream& fin );

		bool save( const char* filePathName ) const;
		bool load( const char* filePathName );
};

inline void
ZStringArray::setLength( int length )
{
	ZArray<ZString>::setLength( length, false );
}

ostream&
operator<<( ostream& os, const ZStringArray& object );

ZELOS_NAMESPACE_END

#endif

