#ifndef _BS_Matrix_h_
#define _BS_Matrix_h_

#include <BeyondScreen.h>

BS_NAMESPACE_BEGIN

class Matrix
{
    public:

        union
        {
            struct
            {
                double _00, _01, _02, _03;
                double _10, _11, _12, _13;
                double _20, _21, _22, _23;
                double _30, _31, _32, _33;
            };

            double data[4][4];
        };

    public:

        Matrix()
        {
            _00=1.0; _01=0.0; _02=0.0; _03=0.0;
            _10=0.0; _11=1.0; _12=0.0; _13=0.0;
            _20=0.0; _21=0.0; _22=1.0; _23=0.0;
            _30=0.0; _31=0.0; _32=0.0; _33=1.0;
        }

        Matrix( const Matrix& m )
        {
            memcpy( (char*)data, (char*)m.data, 16*sizeof(double) );
        }

        Matrix
        (
            const double& m00, const double& m01, const double& m02, const double& m03,
            const double& m10, const double& m11, const double& m12, const double& m13,
            const double& m20, const double& m21, const double& m22, const double& m23,
            const double& m30, const double& m31, const double& m32, const double& m33
        )
        {
            _00=m00; _01=m01; _02=m02; _03=m03;
            _10=m10; _11=m11; _12=m12; _13=m13;
            _20=m20; _21=m21; _22=m22; _23=m23;
            _30=m30; _31=m31; _32=m32; _33=m33;
        }

        void zeroize()
        {
            memset( (char*)data, 0, 16*sizeof(double) );
        }

        Matrix& operator=( const Matrix& m )
        {
            memcpy( (char*)data, (char*)m.data, 16*sizeof(double) );
            return (*this);
        }

		double& operator()( const int& i, const int& j )
        {
            return data[i][j];
        }

		const double& operator()( const int& i, const int& j ) const
        {
            return data[i][j];
        }

        Matrix& transpose()
        {
            Swap(_01,_10); Swap(_02,_20); Swap(_03,_30);
            Swap(_12,_21); Swap(_13,_31);
            Swap(_23,_32);
            return (*this);
        }

        Vector transform( const Vector& v, bool asVector ) const
        {
            const double& x = v.x;
            const double& y = v.y;
            const double& z = v.z;

            Vector tmp( _00*x+_01*y+_02*z, _10*x+_11*y+_12*z, _20*x+_21*y+_22*z );

            if( asVector ) { return tmp; } // no consideration for translation

            tmp.x += _03;
            tmp.y += _13;
            tmp.z += _23;

            return tmp;
        }

        void write( ofstream& fout ) const
        {
            fout.write( (char*)data, 16*sizeof(double) );
        }

        void read( ifstream& fin ) const
        {
            fin.read( (char*)data, 16*sizeof(double) );
        }
};

inline ostream& operator<<( ostream& os, const Matrix& m )
{
    std::string ret;
    std::string indent;

    const int indentation = 0;
    indent.append( indentation+1, ' ' );

    ret.append( "[" );

    for( int i=0; i<4; ++i )
    {
        ret.append( "[" );

        for( int j=0; j<4; ++j )
        {
            if( j ) { ret.append(", "); }
            ret.append( std::to_string( m(i,j) ) );
        }

        ret.append("]");

        if( i< 4-1 )
        {
            ret.append( ",\n" );
            ret.append( indent );
        }
    }

    ret.append( "]" );

    os << ret;

    return os;
}

BS_NAMESPACE_END

#endif

