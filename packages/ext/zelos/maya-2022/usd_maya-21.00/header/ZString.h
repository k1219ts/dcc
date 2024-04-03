//-----------//
// ZString.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.11.22                               //
//-------------------------------------------------------//

#ifndef _ZString_h_
#define _ZString_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

/// @brief String.
/**
	This class manipulates strings.
	It is inherited from "STL string" class.
	Therefore, all functions of "STL string" such as size(), length(), assign(), etc. are also available.
*/
class ZString : public std::string
{
	private:

		typedef std::string parent;

	public:

		ZString();
		ZString( const char* str );
		ZString( const string& str );
		ZString( const ZString& str );

		ZString( char c );
		ZString( unsigned char c );
		ZString( char c, int length );
		ZString( unsigned char c, int length );

		ZString( bool v );
		ZString( short v );
		ZString( unsigned short v );
		ZString( int v );
		ZString( unsigned int v );
		ZString( long int v );
		ZString( unsigned long int v );
		ZString( float v );
		ZString( double v );
		ZString( long double v );

		ZString& fromTextFile( const char* filePathName );

		ZString& operator=( const char* str );
		ZString& operator=( const string& str );
		ZString& operator=( const ZString& str );

		ZString& operator=( char c );
		ZString& operator=( unsigned char c );
		ZString& operator=( bool v );
		ZString& operator=( short v );
		ZString& operator=( unsigned short v );
		ZString& operator=( int v );
		ZString& operator=( unsigned int v );
		ZString& operator=( long int v );
		ZString& operator=( unsigned long int v );
		ZString& operator=( float v );
		ZString& operator=( double v );
		ZString& operator=( long double v );

		bool operator==( const char* other ) const;
		bool operator==( const ZString& other ) const;
		bool operator!=( const char* other ) const;
		bool operator!=( const ZString& other ) const;

		template <class S>
		ZString operator+( const S& x ) const;

		template <class S>
		ZString& operator+=( const S& x );

		void set( const char* str, ... );

		int length() const;

		int firstIndexOf( char c ) const;
		int lastIndexOf( char c ) const;

		ZString subString( int start, int end ) const;

		bool isDigit() const;
		bool isAlpha() const;
		bool isAlnum() const;
		bool isLower() const;
		bool isUpper() const;

		ZString& lowerize();
		ZString& upperize();

		ZString toLower() const;
		ZString toUpper() const;

		ZString& capitalize();
		ZString& swapCase();

		const char* asChar() const;

		char at0() const;

		int asInt() const;
		float asFloat() const;
		double asDouble() const;

		// Why "fromChar" instead of "from"?
		// Warning 314: 'from' is a python keyword, renaming to '_from'
		void replace( char fromChar, char toChar );
		void replace( const char* fromStr, const char* toStr );

		void reverse();

		static ZString commify( int number );

		int split( const ZString& delimiter, vector<ZString>& tokens ) const;
		void removeSpace();

        int count( char c ) const;

		static ZString makePadding( int number, int padding );

		void write( ofstream& fout, bool writeStringLength=true ) const;
		void read( ifstream& fin, bool readStringLength=true );

		bool save( const char* filePathName ) const;
		bool load( const char* filePathName );
};

template <class S>
ZString
ZString::operator+( const S& x ) const
{
	ostringstream oss;
	oss << x;

	string tmp( *this );
	tmp += oss.str();

	return tmp;
}

template <class S>
ZString&
ZString::operator+=( const S& x )
{
	ostringstream oss;
	oss << x;

	string tmp( *this );
	tmp += oss.str();

	this->operator=( tmp );

	return (*this);
}

ostream&
operator<<( ostream& os, const ZString& object );

ZELOS_NAMESPACE_END

#endif

