#ifndef _BS_Vector_h_
#define _BS_Vector_h_

#include <BeyondScreen.h>

BS_NAMESPACE_BEGIN

class Vector
{
    public:

        double x, y, z;

    public:

        Vector()
        : x(0.0), y(0.0), z(0.0)
        {}

        Vector( const Vector& v )
        : x(v.x), y(v.y), z(v.z)
        {}

        Vector( const float* v )
        : x(v[0]), y(v[1]), z(v[2])
        {}

        Vector( const double* v )
        : x(v[0]), y(v[1]), z(v[2])
        {}

        Vector( const double& px, const double& py, const double& pz )
        : x(px), y(py), z(pz)
        {}

        void set( const double& px, const double& py, const double& pz )
        {
            x=px; y=py; z=pz;
        }

        void zeroize()
        {
            x = y = z = 0.0;
        }

        Vector& operator=( const Vector& v )
        {
            x=v.x; y=v.y; z=v.z;
            return (*this);
        }

        Vector& operator+=( const Vector& v )
        {
            x+=v.x; y+=v.y; z+=v.z;
            return (*this);
        }

        Vector& operator-=( const Vector& v )
        {
            x-=v.x; y-=v.y; z-=v.z;
            return (*this);
        }

        Vector& operator*=( const double& s )
        {
            x*=s; y*=s; z*=s;
            return (*this);
        }

        Vector& operator/=( const double& s )
        {
            x/=s; y/=s; z/=s;
            return (*this);
        }

        Vector operator+( const Vector& v ) const
        {
            return Vector( x+v.x, y+v.y, z+v.z );
        }

        Vector operator-( const Vector& v ) const
        {
            return Vector( x-v.x, y-v.y, z-v.z );
        }

        Vector operator*( const double& s ) const
        {
            return Vector( x*s, y*s, z*s );
        }

        Vector operator/( const double& s ) const
        {
            return Vector( x/s, y/s, z/s );
        }

        Vector operator-() const
        {
            return Vector( -x, -y, -z );
        }

        bool equal( const Vector& v, const double epsilon=1e-10 ) const
        {
            if( !SAME( x, v.x ) ) { return false; }
            if( !SAME( y, v.y ) ) { return false; }
            if( !SAME( z, v.z ) ) { return false; }
            return true;
        }

        double distanceTo( const Vector& p ) const
        {
            return sqrt( SQR(x-p.x) + SQR(y-p.y) + SQR(z-p.z) );
        }

        double squaredDistanceTo( const Vector& p ) const
        {
            return ( SQR(x-p.x) + SQR(y-p.y) + SQR(z-p.z) );
        }

        double magnitude() const
        {
            return sqrt( SQR(x) + SQR(y) + SQR(z) );
        }

        Vector direction() const
        {
            const double L = Vector::magnitude();
            return Vector( x/L, y/L, z/L );
        }

        Vector& normalize()
        {
            const double L = Vector::magnitude();
            x/=L; y/=L; z/=L;
            return (*this);
        }

        double dot( const Vector& v ) const
        {
            return ( (x*v.x) + (y*v.y) + (z*v.z) );
        }

        double operator*( const Vector& v ) const
        {
            return ( (x*v.x) + (y*v.y) + (z*v.z) );
        }

        Vector cross( const Vector& v ) const
        {
			return Vector( (y*v.z)-(z*v.y), (z*v.x)-(x*v.z), (x*v.y)-(y*v.x) );
        }

        Vector operator^( const Vector& v ) const
        {
			return Vector( (y*v.z)-(z*v.y), (z*v.x)-(x*v.z), (x*v.y)-(y*v.x) );
        }

        void write( ofstream& fout ) const
        {
            fout.write( (char*)&x, sizeof(double)*3 );
        }

        void read( ifstream& fin ) const
        {
            fin.read( (char*)&x, sizeof(double)*3 );
        }
};

inline Vector operator*( const double s, const Vector& v )
{
    return Vector( s*v.x, s*v.y, s*v.z );
}

inline ostream& operator<<( ostream& os, const Vector& v )
{
	os << "( " << v.x << ", " << v.y << ", " << v.z << " ) ";
	return os;
}

BS_NAMESPACE_END

#endif

