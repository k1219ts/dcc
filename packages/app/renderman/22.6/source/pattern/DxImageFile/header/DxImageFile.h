#ifndef DxImageFile_h
#define DxImageFile_h

#include "RixShadingUtils.h"
#include <cstdio>
#include <cstring>
#include <map>
#include "PxrTextureAtlas.h"
#include <OpenImageIO/imageio.h>
OIIO_NAMESPACE_USING


///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Utility Funcions
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline bool FileExist( const char* filePathName )
{
    struct stat buffer;
    const int exist = stat( filePathName, &buffer );
    return ( (exist==0) ? true : false );
}

template <typename T>
inline T Clamp( const T& x, const T& low, const T& high )
{
    if( x < low  ) { return low;  }
    if( x > high ) { return high; }

    return x;
}

template <typename T>
inline T Lerp( const T& a, const T& b, const float t )
{
	if( t < 0.0f ) { return a; }
	if( t > 1.0f ) { return b; }

	return ( a*(1.0f-t) + b*t );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Vector2 Class
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class Vector2
{
    public:

        union
        {
            struct { T x, y; }; // position, vector
            struct { T u, v; }; // texture coordinates
            struct { T s, t; }; // texture coordinates
            struct { T i, j; }; // index
            T values[2];
        };

    public:

        Vector2()
        : x(0), y(0)
        {}

        Vector2( const Vector2& v )
        : x(v.x), y(v.y)
        {}

        template <typename S>
        Vector2( S v0, S v1 )
        : x(v0), y(v1)
        {}

        void zeroize()
        {
            x = y = 0;
        }

        Vector2& operator=( const Vector2& v )
        {
            x=v.x; y=v.y;
            return (*this);
        }

        Vector2& operator+=( const Vector2& v )
        {
            x+=v.x; y+=v.y;
            return (*this);
        }

        Vector2& operator-=( const Vector2& v )
        {
            x-=v.x; y-=v.y;
            return (*this);
        }

        template <typename S>
        Vector2& operator*=( S s )
        {
            x*=s; y*=s;
            return (*this);
        }

        template <typename S>
        Vector2& operator/=( S s )
        {
            x/=s; y/=s;
            return (*this);
        }

        Vector2 operator+( const Vector2& v ) const
        {
            return Vector2( x+v.x, y+v.y );
        }

        Vector2 operator-( const Vector2& v ) const
        {
            return Vector2( x-v.x, y-v.y );
        }

        template <typename S>
        Vector2 operator*( S s ) const
        {
            return Vector2( x*s, y*s );
        }

        template <typename S>
        Vector2 operator/( S s ) const
        {
            return Vector2( x/s, y/s );
        }

        Vector2 operator-() const
        {
            return Vector2( -x, -y );
        }

        T distanceTo( const Vector2& p ) const
        {
            return (T)sqrt( squaredDistanceTo(p) );
        }

        T squaredDistanceTo( const Vector2& p ) const
        {
            return ( (x-p.x)*(x-p.x) + (y-p.y)*(y-p.y) );
        }
};

typedef Vector2<size_t> Idx2;
typedef Vector2<int>    Vec2i;
typedef Vector2<float>  Vec2f;
typedef Vector2<double> Vec2d;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Vector3 Class
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class Vector3
{
    public:

        union
        {
            struct { T x, y, z; }; // position, vector
            struct { T r, g, b; }; // color
            struct { T u, v, w; }; // texture coordinates
            struct { T i, j, k; }; // index
            T values[3];
        };

    public:

        Vector3()
        : x(0), y(0), z(0)
        {}

        Vector3( const Vector3& v )
        : x(v.x), y(v.y), z(v.z)
        {}

        template <typename S>
        Vector3( S v0, S v1, S v2 )
        : x(v0), y(v1), z(v2)
        {}

        void zeroize()
        {
            x = y = z = 0;
        }

        Vector3& operator=( const Vector3& v )
        {
            x=v.x; y=v.y; z=v.z;
            return (*this);
        }

        Vector3& operator+=( const Vector3& v )
        {
            x+=v.x; y+=v.y; z+=v.z;
            return (*this);
        }

        Vector3& operator-=( const Vector3& v )
        {
            x-=v.x; y-=v.y; z-=v.z;
            return (*this);
        }

        template <typename S>
        Vector3& operator*=( S s )
        {
            x*=s; y*=s; z*=s;
            return (*this);
        }

        template <typename S>
        Vector3& operator/=( S s )
        {
            x/=s; y/=s; z/=s;
            return (*this);
        }

        Vector3 operator+( const Vector3& v ) const
        {
            return Vector3( x+v.x, y+v.y, z+v.z );
        }

        Vector3 operator-( const Vector3& v ) const
        {
            return Vector3( x-v.x, y-v.y, z-v.z );
        }

        template <typename S>
        Vector3 operator*( S s ) const
        {
            return Vector3( x*s, y*s, z*s );
        }

        template <typename S>
        Vector3 operator/( S s ) const
        {
            return Vector3( x/s, y/s, z/s );
        }

        Vector3 operator-() const
        {
            return Vector3( -x, -y, -z );
        }
};

typedef Vector3<size_t> Idx3;
typedef Vector3<int>    Vec3i;
typedef Vector3<float>  Vec3f;
typedef Vector3<double> Vec3d;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Image Class
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Image
{
    private:

        ImageSpec spec;
        ImageInput* img = nullptr;
        float* data = nullptr;

    public:

        Image()
        {}

        ~Image()
        {
            reset();
        }

        void reset()
        {
            if( img ) { delete img; }
            if( data ) { delete[] data; }
        }

        int width() const
        {
            return spec.width;
        }

        int height() const
        {
            return spec.height;
        }

        int channels() const
        {
            return spec.nchannels;
        }

        int index( int i, int j ) const
        {
			return ( spec.nchannels * ( i + j * spec.width ) );
        }

        Vec3f pixelValue( int i, int j ) const
        {
			int idx = index(i,j);

			float& r = data[  idx];
			float& g = data[++idx];
			float& b = data[++idx];

            return Vec3f( r, g, b );
        }

        RtColorRGB getPixel(int i, int j) const
        {
            int idx = index(i, j);

			float& r = data[  idx];
			float& g = data[++idx];
			float& b = data[++idx];

            return RtColorRGB( r, g, b );
        }

        Vec3f color( float s, float t ) const
        {
            const int& w = spec.width;
            const int& h = spec.height;

            s = Clamp( s, 0.f, 1.f );
            t = Clamp( t, 0.f, 1.f );

            const int i = Clamp( int( ( s - 0.5f ) * float(w) ), 0, w-1 );
            const int j = Clamp( int( ( t - 0.5f ) * float(h) ), 0, h-1 );

            const int& i0 = i;
            const int& j0 = j;

            const int i1 = Clamp( i+1, 0, w-1 );
            const int j1 = Clamp( j+1, 0, h-1 );

            const float fs = s*w - i;
            const float ft = t*h - j;

            const Vec3f c0 = Lerp( pixelValue(i0,j0), pixelValue(i1,j0), fs );
            const Vec3f c1 = Lerp( pixelValue(i0,j1), pixelValue(i1,j1), fs );

            return Lerp( c0, c1, ft );
        }

        bool load( const char* filePathName )
        {
            reset();

            if( FileExist( filePathName ) == false ) { return false; }

            img = ImageInput::create( filePathName );
            img->open( filePathName, spec );
            data = new float[ spec.width * spec.height * spec.nchannels ];
            img->read_image( TypeDesc::FLOAT, data );
            img->close();

            return true;
        }
};


////////////////////////////////////////////////////////////////////////////////
// Utility Functions
////////////////////////////////////////////////////////////////////////////////

template <class T>
inline T* Malloc( RixShadingContext::Allocator& pool, RtInt n )
{
   return pool.AllocForPattern<T>( n );
}

inline RtInt NumOutputs( RixSCParamInfo const* paramTable )
{
    RtInt count = 0;
    while( paramTable[count++].access == k_RixSCOutput ) {}
    return count;
}


inline RtColorRGB linRec709ToLinAP1(const RtColorRGB c)
{
    const static RtMatrix4x4 rec709toACEScg(0.610277f ,  0.0688436f , 0.0241673f, 0.0f,
                                            0.345424f ,  0.934974f  , 0.121814f , 0.0f,
                                            0.0443001f, -0.00381805f, 0.854019f , 0.0f,
                                            0.0f,        0.0f,        0.0f,       1.0f);

    // convert rec709 primaries to ACES AP1
    RtVector3 dir = rec709toACEScg.vTransform(RtVector3(c.r, c.g, c.b));

    return RtColorRGB(dir.x, dir.y, dir.z);
}


////////////////////////////////////////////////////////////////////////////////
// Image Read
////////////////////////////////////////////////////////////////////////////////
class DxReadTexture
{
public:
    DxReadTexture(
        const Image* image,
        PxrLinearizeMode const linmode = k_linearizeDisabled
    )
        : m_img(image),
          m_linearize(linmode)
    {}

    template<typename T> int
    Texture(
        int const nPoints, RtFloat2 const* st, T* resultRGB, RtFloat* resultA,
        int* rf=NULL
    )
    {
        for (int i=0; i<nPoints; ++i)
        {
            const float s = st[i].x;
            const float t = st[i].y;

            int s_id = (int)s;
            int t_id = (int)t;

            const float ss = s - s_id;
            const float tt = t - t_id;

            const int i_index = Clamp(int(ss * m_img->width()), 0, m_img->width()-1);
            const int j_index = Clamp(int(tt * m_img->height()), 0, m_img->height()-1);

            if (resultRGB)
            {
                RtColorRGB const img = m_img->getPixel(i_index, j_index);
                resultRGB[i] = img;
            }

            if (resultA)
            {
                resultA[i] = 1;
            }
        }

        RixTexture::TxProperties txProps;
        PxrLinearizeSRGB(m_linearize, txProps, nPoints, 3, (float*)resultRGB, rf);

    }

private:
    const Image*        m_img;
    PxrLinearizeMode    m_linearize;
};


#endif
