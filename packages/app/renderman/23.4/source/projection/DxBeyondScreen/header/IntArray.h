#ifndef _BS_IntArray_h_
#define _BS_IntArray_h_

#include <BeyondScreen.h>

BS_NAMESPACE_BEGIN

class IntArray : public vector<int>
{
    private:

        typedef vector<int> parent;

    public:

        IntArray()
        {
            // nothing to do
        }

        IntArray( int initialLength )
        {
            IntArray::setLength( initialLength );
        }

        void clear()
        {
            parent::clear();
        }

        int length() const
        {
            return (int)parent::size();
        }

        void setLength( int n )
        {
            if( n <= 0 ) { IntArray::clear(); return; }
            if( IntArray::length() == n ) { return; }
            parent::resize( n );
        }

        void fill( const int& valueForAll )
        {
            if( parent::empty() ) { return; }
            std::fill( parent::begin(), parent::end(), valueForAll );
        }

        IntArray& operator=( const IntArray& a )
        {
            if( a.length() == 0 ) { IntArray::clear(); }
            else { parent::assign( a.begin(), a.end() ); }
            return (*this);
        }

        bool operator==( const IntArray& a ) const
        {
            const int n = IntArray::length();

            for( int i=0; i<n; ++i )
            {
                if( parent::at(i) != a[i] ) { return false; }
            }

            return true;
        }

        bool operator!=( const IntArray& a ) const
        {
            const int n = IntArray::length();

            for( int i=0; i<n; ++i )
            {
                if( parent::at(i) != a[i] ) { return true; }
            }

            return false;
        }

        int& first()
        {
            return parent::front();
        }

        const int& first() const
        {
            return parent::front();
        }

        int& last()
        {
            return parent::back();
        }

        const int& last() const
        {
            return parent::back();
        }

        void append( const int& value )
        {
            parent::push_back( value );
        }

        void write( ofstream& fout ) const
        {
            const int n = IntArray::length();

            fout.write( (char*)&n, sizeof(int) );

            if( n > 0 )
            {
                fout.write( (char*)&parent::at(0), n*sizeof(int) );
            }
        }

        void read( ifstream& fin )
        {
            IntArray::clear();

            int n = 0;
            fin.read( (char*)&n, sizeof(int) );

            if( n > 0 )
            {
                IntArray::setLength( n );
                fin.read( (char*)&parent::at(0), n*sizeof(int) );
            }
        }

        bool save( const char* filePathName ) const
        {
            ofstream fout( filePathName, ios::out|ios::binary|ios::trunc );

            if( fout.fail() || !fout.is_open() )
            {
                cout << "Error@IntArray::save(): Failed to save file: " << filePathName << endl;
                return false;
            }

            IntArray::write( fout );

            fout.close();

            return true;
        }

        bool load( const char* filePathName )
        {
            IntArray::clear();

            ifstream fin( filePathName, ios::in|ios::binary );

            if( fin.fail() )
            {
                cout << "Error@ZArray::load(): Failed to load file." << endl;
                return false;
            }

            IntArray::read( fin );

            fin.close();

            return true;
        }
};

BS_NAMESPACE_END

#endif

