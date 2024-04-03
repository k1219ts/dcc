#ifndef _BS_DoubleArray_h_
#define _BS_DoubleArray_h_

#include <BeyondScreen.h>

BS_NAMESPACE_BEGIN

class DoubleArray : public vector<double>
{
    private:

        typedef vector<double> parent;

    public:

        DoubleArray()
        {
            // nothing to do
        }

        DoubleArray( int initialLength )
        {
            DoubleArray::setLength( initialLength );
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
            if( n <= 0 ) { DoubleArray::clear(); return; }
            if( DoubleArray::length() == n ) { return; }
            parent::resize( n );
        }

        void fill( const double& valueForAll )
        {
            if( parent::empty() ) { return; }
            std::fill( parent::begin(), parent::end(), valueForAll );
        }

        DoubleArray& operator=( const DoubleArray& a )
        {
            parent::assign( a.begin(), a.end() );
            return (*this);
        }

        bool operator==( const DoubleArray& a ) const
        {
            const int n = DoubleArray::length();
            for( int i=0; i<n; ++i ) { if( parent::at(i) != a[i] ) { return false; } }
            return true;
        }

        bool operator!=( const DoubleArray& a ) const
        {
            const int n = DoubleArray::length();
            for( int i=0; i<n; ++i ) { if( parent::at(i) != a[i] ) { return true; } }
            return false;
        }

        void append( const double& value )
        {
            parent::push_back( value );
        }

        double& first()
        {
            return parent::front();
        }

        const double& first() const
        {
            return parent::front();
        }

        double& last()
        {
            return parent::back();
        }

        const double& last() const
        {
            return parent::back();
        }

        void write( ofstream& fout ) const
        {
            const int n = DoubleArray::length();
            fout.write( (char*)&n, sizeof(double) );

            if( n > 0 )
            {
                fout.write( (char*)&parent::at(0), n*sizeof(double) );
            }
        }

        void read( ifstream& fin )
        {
            DoubleArray::clear();

            int n = 0;
            fin.read( (char*)&n, sizeof(double) );

            if( n > 0 )
            {
                DoubleArray::setLength( n );
                fin.read( (char*)&parent::at(0), n*sizeof(double) );
            }
        }

        bool save( const char* filePathName ) const
        {
            ofstream fout( filePathName, ios::out|ios::binary|ios::trunc );

            if( fout.fail() || !fout.is_open() )
            {
                cout << "Error@DoubleArray::save(): Failed to save file: " << filePathName << endl;
                return false;
            }

            DoubleArray::write( fout );

            fout.close();

            return true;
        }

        bool load( const char* filePathName )
        {
            DoubleArray::clear();

            ifstream fin( filePathName, ios::in|ios::binary );

            if( fin.fail() )
            {
                cout << "Error@ZArray::load(): Failed to load file." << endl;
                return false;
            }

            DoubleArray::read( fin );

            fin.close();

            return true;
        }
};

BS_NAMESPACE_END

#endif

