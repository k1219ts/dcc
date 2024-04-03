//----------------//
// ZStringUtils.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZStringUtils_h_
#define _ZStringUtils_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

// Check if a given character is big or small.
inline bool
ZUpperChar( char c )
{
	if( c < 'A' ) { return false; }
	if( c > 'Z' ) { return false; }
	return true;
}

// Check if a given character is big or small.
inline bool
ZLowerChar( char c )
{
	if( c < 'a' ) { return false; }
	if( c > 'z' ) { return false; }
	return true;
}

// Check if a given character is numeric.
inline bool
ZNumericChar( char c )
{
	if( c < '0' ) { return false; }
	if( c > '9' ) { return false; }
	return true;
}

// Check if a given character is a space character.
inline bool
ZSpaceChar( char c )
{
	switch ( c )
	{
		case ' ':
		case '\t':
		case '\n':
		case '\r':
			return true;
		default:
			return false;
	}
}

// Check if a given character is a control character.
inline bool
ZControlChar( char c )
{
	if( c < 32  ) { return true; }
	if( c > 126 ) { return true; }
	return false;
}

inline char
ZLowerToUpper( char c )
{
	return ( ((c>='a')&&(c<='z')) ? (c-32) : (c) ); // 32 = 'a' - 'A'
}

inline char
ZUpperToLower( char c )
{
	return ( ((c>='A')&&(c<='Z')) ? (c+32) : (c) ); // 32 = 'a' - 'A'
}

inline int
ZStringToInt( const ZString& str )
{
	return atoi( str.c_str() );
}

inline float
ZStringToFloat( const ZString& str )
{
	return atof( str.c_str() );
}

inline double
ZToDouble( const ZString& str )
{
	char* endPtr;
	return strtod( str.c_str(), &endPtr );
}

inline int 
ZFindStringArrayIndex( const ZStringArray &strings, const char* toFind )
{
	for (int i = 0; i < strings.length(); ++i)
	{
		if (strings[i] == toFind)
			return i;
	}
	return -1;
}

template <typename T>
ZString
ZNumberToString( T n )
{
	ostringstream oss;
	oss << n;
	return oss.str();
}

//template <class T>
//ZString
//operator+( const ZString& str, T n )
//{
//	ostringstream oss;
//	oss << n;
//	return ( str + oss.str() );
//}

ZString Trim( const ZString& str );

void ZTokenize( const ZString& str, const ZString& delimiter, ZStringArray& tokens );
void ZTokenize( const ZString& str, const ZString& delimiter, ZIntArray& tokens );

ZString ZFileExtension( const ZString& filePathName );
ZString ZFileFrameNumber( float currentTime );

bool ZGetList( const ZString& string, ZIntArray& list );

ZELOS_NAMESPACE_END

#endif

