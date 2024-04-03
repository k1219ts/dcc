#ifndef _BS_UIntArray_h_
#define _BS_UIntArray_h_

#include <BeyondScreen.h>

BS_NAMESPACE_BEGIN

class UIntArray : public vector<unsigned int>
{
    private:

        typedef vector<unsigned int> parent;

    public:

        UIntArray()
        {
            // nothing to do
        }

        UIntArray( int initialLength )
        {
            UIntArray::setLength( initialLength );
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
            if( n <= 0 ) { UIntArray::clear(); return; }
            if( UIntArray::length() == n ) { return; }
            parent::resize( n );
        }

        void fill( const unsigned int& valueForAll )
        {
            if( parent::empty() ) { return; }
            std::fill( parent::begin(), parent::end(), valueForAll );
        }

        UIntArray& operator=( const UIntArray& a )
        {
            if( a.length() == 0 ) { UIntArray::clear(); }
            else { parent::assign( a.begin(), a.end() ); }
            return (*this);
        }

        bool operator==( const UIntArray& a ) const
        {
            const int n = UIntArray::length();

            for( int i=0; i<n; ++i )
            {
                if( parent::at(i) != a[i] ) { return false; }
            }

            return true;
        }

        bool operator!=( const UIntArray& a ) const
        {
            const int n = UIntArray::length();

            for( int i=0; i<n; ++i )
            {
                if( parent::at(i) != a[i] ) { return true; }
            }

            return false;
        }

        unsigned int& first()
        {
            return parent::front();
        }

        const unsigned int& first() const
        {
            return parent::front();
        }

        unsigned int& last()
        {
            return parent::back();
        }

        const unsigned int& last() const
        {
            return parent::back();
        }

        void append( const unsigned int& value )
        {
            parent::push_back( value );
        }

        void write( ofstream& fout ) const
        {
            const int n = UIntArray::length();

            fout.write( (char*)&n, sizeof(int) );

            if( n > 0 )
            {
                fout.write( (char*)&parent::at(0), n*sizeof(unsigned int) );
            }
        }

        void read( ifstream& fin )
        {
            UIntArray::clear();

            int n = 0;
            fin.read( (char*)&n, sizeof(int) );

            if( n > 0 )
            {
                UIntArray::setLength( n );
                fin.read( (char*)&parent::at(0), n*sizeof(unsigned int) );
            }
        }

        bool save( const char* filePathName ) const
        {
            ofstream fout( filePathName, ios::out|ios::binary|ios::trunc );

            if( fout.fail() || !fout.is_open() )
            {
                cout << "Error@UIntArray::save(): Failed to save file: " << filePathName << endl;
                return false;
            }

            UIntArray::write( fout );

            fout.close();

            return true;
        }

        bool load( const char* filePathName )
        {
            UIntArray::clear();

            ifstream fin( filePathName, ios::in|ios::binary );

            if( fin.fail() )
            {
                cout << "Error@ZArray::load(): Failed to load file." << endl;
                return false;
            }

            UIntArray::read( fin );

            fin.close();

            return true;
        }
};

BS_NAMESPACE_END

#endif

