#ifndef _BS_VectorArray_h_
#define _BS_VectorArray_h_

#include <BeyondScreen.h>

BS_NAMESPACE_BEGIN

class VectorArray : public vector<Vector>
{
    private:

        typedef vector<Vector> parent;

    public:

        VectorArray()
        {
            // nothing to do
        }

        VectorArray( int initialLength )
        {
            VectorArray::setLength( initialLength );
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
            if( n <= 0 ) { VectorArray::clear(); return; }
            if( VectorArray::length() == n ) { return; }
            parent::resize( n );
        }

        VectorArray& operator=( const VectorArray& a )
        {
            if( a.length() == 0 ) { VectorArray::clear(); }
            else { parent::assign( a.begin(), a.end() ); }
            return (*this);
        }

        Vector& first()
        {
            return parent::front();
        }

        const Vector& first() const
        {
            return parent::front();
        }

        Vector& last()
        {
            return parent::back();
        }

        const Vector& last() const
        {
            return parent::back();
        }

        void append( const Vector& v )
        {
            parent::push_back( v );
        }

        Vector center() const
        {
            Vector c;

            const int n = VectorArray::length();
            if( n == 0 ) { return c; }

            const double denom = 1.0 / double(n);

            for( int i=0; i<n; ++i )
            {
                c += parent::at(i) * denom;
            }

            return c;
        }

        BoundingBox boundingBox() const
        {
            BoundingBox bBox;

            const int n = VectorArray::length();
            if( n == 0 ) { return bBox; }

            for( int i=0; i<n; ++i )
            {
                bBox.expand( parent::at(i) );
            }

            return bBox;
        }

        void write( ofstream& fout ) const
        {
            const int n = VectorArray::length();

            fout.write( (char*)&n, sizeof(int) );

            if( n > 0 )
            {
                fout.write( (char*)&parent::at(0).x, n*sizeof(Vector) );
            }
        }

        void read( ifstream& fin )
        {
            VectorArray::clear();

            int n = 0;
            fin.read( (char*)&n, sizeof(int) );

            if( n > 0 )
            {
                VectorArray::setLength( n );
                fin.read( (char*)&parent::at(0).x, n*sizeof(Vector) );
            }
        }

        bool save( const char* filePathName ) const
        {
            ofstream fout( filePathName, ios::out|ios::binary|ios::trunc );

            if( fout.fail() || !fout.is_open() )
            {
                cout << "Error@VectorArray::save(): Failed to save file: " << filePathName << endl;
                return false;
            }

            VectorArray::write( fout );

            fout.close();

            return true;
        }

        bool load( const char* filePathName )
        {
            VectorArray::clear();

            ifstream fin( filePathName, ios::in|ios::binary );

            if( fin.fail() )
            {
                cout << "Error@VectorArray::load(): Failed to load file." << endl;
                return false;
            }

            VectorArray::read( fin );

            fin.close();

            return true;
        }
};

BS_NAMESPACE_END

#endif

