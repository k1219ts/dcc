#ifndef _BS_String_h_
#define _BS_String_h_

#include <BeyondScreen.h>

BS_NAMESPACE_BEGIN

class String : public string
{
	private:

		typedef string parent;

    public:

        String()
        {
            // nothing to do
        }

        String( const String& str )
        : string( str )
        {
            // nothing to do
        }

        String( const char* str )
        : string( str )
        {
            // nothing to do
        }

        String( const string& str )
        : string( str )
        {
            // nothing to do
        }

        void clear()
        {
            parent::clear();
        }

        int length() const
        {
            return (int)parent::length();
        }

        String& operator=( const String& str )
        {
            parent::assign( str );
            return (*this);
        }

        String& operator=( const char* str )
        {
            if( !str ) { String::clear(); }
            parent::assign( str );
            return (*this);
        }

        String& operator=( const string& str )
        {
            parent::assign( str );
            return (*this);
        }

        bool operator==( const String& str ) const
        {
            const int l = String::length();
            if( l != str.length() ) { return false; }

            for( int i=0; i<l; ++i )
            {
                if( parent::at(i) != str[i] ) { return false; }
            }

            return true;
        }

        bool operator==( const char* str ) const
        {
            if( !str ) { return false; }
            String tmpStr( str );
            return this->operator==( tmpStr );
        }

        bool operator!=( const String& str ) const
        {
            const int l = String::length();
            if( l != str.length() ) { return true; }

            for( int i=0; i<l; ++i )
            {
                if( parent::at(i) != str[i] ) { return true; }
            }

            return false;
        }

        bool operator!=( const char* str ) const
        {
            if( !str ) { return false; }
            String tmpStr( str );
            return this->operator!=( tmpStr );
        }

        template <class S>
        String operator+( const S& x ) const
        {
            ostringstream oss;
            oss << x;

            string tmp( *this );
            tmp += oss.str();

            return tmp;
        }

        template <class S>
        String& operator+=( const S& x )
        {
            ostringstream oss;
            oss << x;

            string tmp( *this );
            tmp += oss.str();

            this->operator=( tmp );

            return (*this);
        }

        const char* asChar() const
        {
            return parent::c_str();
        }

        int asInt() const
        {
            return atoi( parent::c_str() );
        }

        float asFloat() const
        {
            return atof( parent::c_str() );
        }

        double asDouble() const
        {
            return strtod( parent::c_str(), NULL );
        }

        String makePadding( int number, int padding )
        {
            ostringstream oss;
            oss << setfill('0') << setw(padding) << number;
            return oss.str();
        }

        void write( ofstream& fout ) const
        {
            const int n = length();
            fout.write( (char*)&n, sizeof(int) );

            if( n )
            {
                fout.write( (char*)&parent::at(0), n*sizeof(char) );
            }
        }

        void read( ifstream& fin )
        {
            String::clear();

            int n = 0;
            fin.read( (char*)&n, sizeof(int) );

            parent::resize( n );

            if( n )
            {
                fin.read( (char*)&parent::at(0), n*sizeof(char) );
            }
        }
};

BS_NAMESPACE_END

#endif

