#ifndef _BS_MatrixArray_h_
#define _BS_MatrixArray_h_

#include <BeyondScreen.h>

BS_NAMESPACE_BEGIN

class MatrixArray : public vector<Matrix>
{
    private:

        typedef vector<Matrix> parent;

    public:

        MatrixArray()
        {
            // nothing to do
        }

        MatrixArray( int initialLength )
        {
            MatrixArray::setLength( initialLength );
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
            if( n <= 0 ) { MatrixArray::clear(); return; }
            if( MatrixArray::length() == n ) { return; }
            parent::resize( n );
        }

        MatrixArray& operator=( const MatrixArray& a )
        {
            if( a.length() == 0 ) { MatrixArray::clear(); }
            else { parent::assign( a.begin(), a.end() ); }
            return (*this);
        }

        Matrix& first()
        {
            return parent::front();
        }

        const Matrix& first() const
        {
            return parent::front();
        }

        Matrix& last()
        {
            return parent::back();
        }

        const Matrix& last() const
        {
            return parent::back();
        }

        void append( const Matrix& v )
        {
            parent::push_back( v );
        }

        void write( ofstream& fout ) const
        {
            const int n = MatrixArray::length();

            fout.write( (char*)&n, sizeof(int) );

            if( n > 0 )
            {
                fout.write( (char*)&parent::at(0)(0,0), n*sizeof(Matrix) );
            }
        }

        void read( ifstream& fin )
        {
            MatrixArray::clear();

            int n = 0;
            fin.read( (char*)&n, sizeof(int) );

            if( n > 0 )
            {
                MatrixArray::setLength( n );
                fin.read( (char*)&parent::at(0)(0,0), n*sizeof(Matrix) );
            }
        }

        bool save( const char* filePathName ) const
        {
            ofstream fout( filePathName, ios::out|ios::binary|ios::trunc );

            if( fout.fail() || !fout.is_open() )
            {
                cout << "Error@MatrixArray::save(): Failed to save file: " << filePathName << endl;
                return false;
            }

            MatrixArray::write( fout );

            fout.close();

            return true;
        }

        bool load( const char* filePathName )
        {
            MatrixArray::clear();

            ifstream fin( filePathName, ios::in|ios::binary );

            if( fin.fail() )
            {
                cout << "Error@MatrixArray::load(): Failed to load file." << endl;
                return false;
            }

            MatrixArray::read( fin );

            fin.close();

            return true;
        }
};

BS_NAMESPACE_END

#endif

