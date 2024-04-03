#ifndef _BS_Tuple_h_
#define _BS_Tuple_h_

#include <BeyondScreen.h>

BS_NAMESPACE_BEGIN

template <int N, typename T>
class Tuple
{
    public:

        T data[N]; // data

    public:

        Tuple()
        {
            memset( data, 0, N*sizeof(T) );
        }

        Tuple( const T& s )
        {
            Tuple::fill( s );
        }

        Tuple( const Tuple& v )
        {
            memcpy( data, v.data, N*sizeof(T) );
        }

        void zeroize()
        {
            memset( data, 0, sizeof(T)*N );
        }

        Tuple& fill( const T& s )
        {
            for( int i=0; i<N; ++i ) { data[i] = s; }
            return (*this);
        }

        T& operator[]( const int& i )
        {
            return (*(data+i)); // = data[i];
        }

        const T& operator[]( const int& i ) const
        {
            return (*(data+i)); // = data[i];
        }

        Tuple& operator=( const Tuple& v )
        {
            memcpy( data, v.data, N*sizeof(T) );
            return (*this);
        }

        void write( ofstream& fout ) const
        {
            fout.write( (char*)&data, sizeof(T)*N );
        }

        void read( ifstream& fin )
        {
            fin.read( (char*)&data, sizeof(T)*N );
        }
};

template <int N, typename T>
inline ostream& operator<<( ostream& os, const Tuple<N,T>& v )
{
	os << "( " << v.data[0];
	for( int i=1; i<N; ++i ) { os << ", " << v.data[i]; } os<<" )";
	return os;
}

typedef Tuple<2,int>    Int2;
typedef Tuple<3,int>    Int3;
typedef Tuple<4,int>    Int4;

typedef Tuple<2,float>  Float2;
typedef Tuple<3,float>  Float3;
typedef Tuple<4,float>  Float4;

typedef Tuple<2,double> Double2;
typedef Tuple<3,double> Double3;
typedef Tuple<4,double> Double4;

BS_NAMESPACE_END

#endif

