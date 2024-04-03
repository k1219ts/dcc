#ifndef _BS_Image_h_
#define _BS_Image_h_

#include <BeyondScreen.h>

BS_NAMESPACE_BEGIN

typedef Imf::Rgba Pixel;
// size(Imf::Rgba) = 8 (8 = 2 bytes x 4 channels)

class Image
{
    private:

        int _width       = 0; // the image width
        int _height      = 0; // the image height
        int _numChannels = 0; // the # of channels

        std::string _compression  = "none";

        Pixel* _pixels = nullptr; // the pixel data

    public:

        Image()
        {
            // nothing to do
        }

        ~Image()
        {
            Image::clear();
        }

        void clear()
        {
            _width = _height = _numChannels = 0;

            _compression = "none";

            Image::release();
        }

        bool create( const int& width, const int& height )
        {
            if( ( width < 1 ) || ( height < 1 ) )
            {
                cout << "Error@Iamge::create(): Invalid dimension." << endl;
                return false;
            }

            if( ( _width != width ) || ( _height != height ) )
            {
                Image::release();
            }

            _width  = width;
            _height = height;

            Image::allocate();

            return true;
        }

        bool save( const char* filePathName ) const
        {
            const String ext = FileExtension( filePathName );

            if( ext == "exr" ) { return Image::saveEXR  ( filePathName ); }
            if( ext == "jpg" ) { return Image::saveJPG  ( filePathName ); }

            cout << "Error@Image::save(): Not supported file format." << endl;

            return false;
        }

        bool load( const char* filePathName )
        {
            if( DoesFileExist( filePathName ) == false )
            {
                cout << "Error@Iamge::load(): Invalid file path & name." << endl;
                return false;
            }

            const String ext = FileExtension( filePathName );

            if( ext == "exr" ) { return Image::loadEXR  ( filePathName ); }
            if( ext == "jpg" ) { return Image::loadJPG  ( filePathName ); }

            cout << "Error@Image::load(): Not supported file format." << endl;

            return false;
        }

        int width() const
        {
            return _width;
        }

        int height() const
        {
            return _height;
        }

        int numPixels() const
        {
            return ( _width * _height );
        }

        const char* compression() const
        {
            return _compression.c_str();
        }

        int index( const int& i, const int& j ) const
        {
            return ( i + _width * j );
        }

        Pixel& operator[]( const int& i )
        {
            return _pixels[i];
        }

        const Pixel& operator[]( const int& i ) const
        {
            return _pixels[i];
        }

        Pixel& operator()( const int& i, const int& j )
        {
            return _pixels[ i + _width * j ];
        }

        const Pixel& operator()( const int& i, const int& j ) const
        {
            return _pixels[ i + _width * j ];
        }

        Pixel* pointer() const
        {
            return _pixels;
        }

    public:

        // s, t: 0.0 ~ 1.0
        Pixel& closestPixel( const double& s, const double& t )
        {
            const int i = Clamp( (int)((s+0.5)*_width ), 0, _width -1 );
            const int j = Clamp( (int)((t+0.5)*_height), 0, _height-1 );

            return _pixels[ i + _width * j ];
        }

        // s, t: 0.0 ~ 1.0
        const Pixel& closestPixel( const double& s, const double& t ) const
        {
            const int i = Clamp( (int)((s+0.5)*_width ), 0, _width -1 );
            const int j = Clamp( (int)((t+0.5)*_height), 0, _height-1 );

            return _pixels[ i + _width * j ];
        }

    private:

        void allocate()
        {
            Image::release();

            const int n = Image::numPixels();

            _pixels = new Pixel[n];

            memset( (char*)_pixels, 0, sizeof(Pixel)*n );
        }

        void release()
        {
            if( _pixels )
            {
                delete[] _pixels;
            }

            _pixels = nullptr;
        }

        bool saveEXR( const char* filePathName ) const
        {
            const int& w = _width;
            const int& h = _height;

            const String filePathNameStr( filePathName ); // for safety
            Imf::RgbaOutputFile file( filePathNameStr.c_str(), w, h, Imf::WRITE_RGBA );

            file.setFrameBuffer( _pixels, 1, w );

            file.writePixels( h );

            return true;
        }

        bool loadEXR( const char* filePathName )
        {
            const String filePathNameStr( filePathName ); // for safety
            Imf::RgbaInputFile file( filePathNameStr.c_str() );

            const Imath::Box2i dw = file.dataWindow();

            const int& w = _width  = dw.max.x - dw.min.x + 1;
            const int& h = _height = dw.max.y - dw.min.y + 1;

            Image::allocate();

            file.setFrameBuffer( _pixels - dw.min.x - dw.min.y * w, 1, w );
            file.readPixels( dw.min.y, dw.max.y );

            switch( file.compression() )
            {
                default:
                case Imf::NO_COMPRESSION:    { _compression = "NONE";  break; }
                case Imf::RLE_COMPRESSION:   { _compression = "RLD";   break; }
                case Imf::ZIPS_COMPRESSION:  { _compression = "ZIPS";  break; }
                case Imf::ZIP_COMPRESSION:   { _compression = "ZIP";   break; }
                case Imf::PIZ_COMPRESSION:   { _compression = "PIZ";   break; }
                case Imf::PXR24_COMPRESSION: { _compression = "PXR24"; break; }
                case Imf::B44_COMPRESSION:   { _compression = "B44";   break; }
                case Imf::B44A_COMPRESSION:  { _compression = "B44A";  break; }
                case Imf::DWAA_COMPRESSION:  { _compression = "DWAA";  break; }
                case Imf::DWAB_COMPRESSION:  { _compression = "DWAB";  break; }
            }

            return true;
        }

        bool saveJPG( const char* filePathName ) const
        {
            FILE* fp = fopen( filePathName, "wb" );

            if( !fp )
            {
                cout << "Error@saveJPG(): Failed to open " << filePathName << endl;
                return false;
            }

            jpeg_compress_struct cinfo;
            jpeg_error_mgr jerr;

            JSAMPLE* samples = new JSAMPLE[_width*3];

            cinfo.err = jpeg_std_error( &jerr );

            jpeg_create_compress( &cinfo );
            jpeg_stdio_dest( &cinfo, fp );

            cinfo.image_width      = _width;
            cinfo.image_height     = _height;
            cinfo.input_components = 3;
            cinfo.in_color_space   = JCS_RGB;

            jpeg_set_defaults( &cinfo );

            const int quality = 95;
            jpeg_set_quality( &cinfo, quality, (boolean)true );

            jpeg_start_compress( &cinfo, (boolean)true );

            JSAMPROW row_pointer = samples;

            for( int j=0; j<_height; ++j )
            {
                for( int i=0; i<_width; ++i )
                {
                    const Pixel& p = _pixels[ Image::index( i, j ) ];

                    samples[3*i  ] = (JSAMPLE)( p.r * 255.f );
                    samples[3*i+1] = (JSAMPLE)( p.g * 255.f );
                    samples[3*i+2] = (JSAMPLE)( p.b * 255.f );
                }

                jpeg_write_scanlines( &cinfo, &row_pointer, 1 );
            }

            jpeg_finish_compress( &cinfo );
            jpeg_destroy_compress( &cinfo );

            delete[] samples;

            fclose( fp );

            return true;
        }

        bool loadJPG( const char* filePathName )
        {
            FILE* fp = fopen( filePathName, "rb" );

            if( !fp )
            {
                cout << "Error@loadJPG(): Failed to open " << filePathName << endl;
                return false;
            }

            jpeg_decompress_struct cinfo;
            jpeg_error_mgr jerr;

            cinfo.err = jpeg_std_error( &jerr );

            jpeg_create_decompress( &cinfo );
            jpeg_stdio_src( &cinfo, fp );

            jpeg_read_header( &cinfo, (boolean)true );
            jpeg_start_decompress( &cinfo );

            _width       = cinfo.output_width;
            _height      = cinfo.output_height;
            _numChannels = 3;
            _compression = "NONE";

            Image::allocate();

            JSAMPLE* samples = new JSAMPLE[_width*_numChannels];

            for( int j=0; j<_height; ++j )
            {
                jpeg_read_scanlines( &cinfo, &samples, 1 );

                for( int i=0; i<_width; ++i )
                {
                    Pixel& p = _pixels[ Image::index( i, j ) ];

                    p.r = (half)( samples[3*i  ] / 255.f );
                    p.g = (half)( samples[3*i+1] / 255.f );
                    p.b = (half)( samples[3*i+2] / 255.f );
                    p.a = (half)( 1.f );
                }
            }

            delete[] samples;

            jpeg_finish_decompress( &cinfo );
            jpeg_destroy_decompress( &cinfo );

            fclose( fp );

            return true;
        }
};

BS_NAMESPACE_END

#endif

