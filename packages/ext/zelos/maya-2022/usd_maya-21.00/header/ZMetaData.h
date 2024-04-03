//-------------//
// ZMetaData.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2017.04.16                               //
//-------------------------------------------------------//

#ifndef _ZMetaData_h_
#define _ZMetaData_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZMetaData
{
	private:

		std::map<ZString,int>     intData;
		std::map<ZString,float>   floatData;
		std::map<ZString,ZVector> vectorData;
		std::map<ZString,ZString> stringData;

	public:

		ZMetaData();

		void clear();

		void append( const char* name, const float&   value );
		void append( const char* name, const double&  value );
		void append( const char* name, const int&	  value );
		void append( const char* name, const ZString& value );
		void append( const char* name, const ZVector& value );

		bool get( const char* name, float&   value );
		bool get( const char* name, double&  value );
		bool get( const char* name, int&     value );
		bool get( const char* name, ZString& value );
		bool get( const char* name, ZVector& value );

		ZString json();
		ZString string();

		void loadString( const ZString& str );
		void loadJson( const ZString& json );
};

ZELOS_NAMESPACE_END

#endif

