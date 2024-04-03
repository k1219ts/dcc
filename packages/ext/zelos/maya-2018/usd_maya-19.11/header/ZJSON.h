//---------//
// ZJSON.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.09.06                               //
//-------------------------------------------------------//

#ifndef _ZJSON_h_
#define _ZJSON_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZJSON
{
	public:

		std::map<ZString,ZString> data; // key-value pair

	public:

		ZJSON();

		void clear();

		void append( const char* key, int value );
		void append( const char* key, float value );
		void append( const char* key, double value );
		void append( const char* key, const ZString& value );

		ZString operator[]( const char* key ) const;

		ZString get( const char* key ) const;

		bool get( const char* key, int& value ) const;
		bool get( const char* key, float& value ) const;
		bool get( const char* key, ZString& value ) const;

		bool save( const char* filePathName ) const;
		bool load( const char* filePathName );
};

ostream&
operator<<( ostream& os, const ZJSON& object );

ZELOS_NAMESPACE_END

#endif
 
