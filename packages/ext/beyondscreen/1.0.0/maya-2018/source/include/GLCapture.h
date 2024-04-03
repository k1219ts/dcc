#ifndef _BS_GLCapture_h_
#define _BS_GLCapture_h_

#include <BeyondScreen.h>

BS_NAMESPACE_BEGIN

class GLCapture
{
    private:

        GLsizei _width      = 0; // image width
        GLsizei _height     = 0; // image height

        GLuint  _fboId      = 0; // framebuffer object id
        GLuint  _colorTexId = 0; // color texture buffer object id
        GLuint  _depthRboId = 0; // depth render buffer object id

        half*   _pixels = nullptr;
        Image   _img;

    public:

        GLCapture()
        {
            // nothing to do
        }

        virtual ~GLCapture()
        {
            GLCapture::clear();
        }

        void clear()
        {
            _width = _height = 0;

            if( _fboId      ) { glDeleteFramebuffers ( 1, &_fboId      ); }
            if( _colorTexId ) { glDeleteRenderbuffers( 1, &_depthRboId ); }
            if( _depthRboId ) { glDeleteTextures     ( 1, &_colorTexId ); }

            _fboId = _colorTexId = _depthRboId = 0;

            if( _pixels ) { delete[] _pixels; _pixels = nullptr; }
        }

        bool initialize( GLsizei width, GLsizei height )
        {
            if( width * height == 0 )
            {
                cout << "Error@GLCapture::initialize(): Invalid image resolution." << endl;
                GLCapture::clear();
                return false;
            }

            bool toReInitialize = false;
            {
                if( _width  != width  ) { toReInitialize = true; }
                if( _height != height ) { toReInitialize = true; }
                if( _fboId      == 0  ) { toReInitialize = true; }
                if( _colorTexId == 0  ) { toReInitialize = true; }
                if( _depthRboId == 0  ) { toReInitialize = true; }
            }

            if( toReInitialize )
            {
                GLCapture::clear();

                glGenFramebuffers( 1, &_fboId );
                glBindFramebuffer( GL_FRAMEBUFFER, _fboId );

                glGenTextures( 1, &_colorTexId );
                glBindTexture( GL_TEXTURE_2D, _colorTexId );
                glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_HALF_FLOAT, NULL );
                glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );  
                glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );  
                glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
                glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );  
                glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _colorTexId, 0 );
                glBindTexture( GL_TEXTURE_2D, 0 );

                glGenRenderbuffers( 1, &_depthRboId );
                glBindRenderbuffer( GL_RENDERBUFFER, _depthRboId );
                glRenderbufferStorage( GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height );
                glFramebufferRenderbuffer( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, _depthRboId );
                glBindRenderbuffer( GL_RENDERBUFFER, 0 );

                GLenum status = glCheckFramebufferStatus( GL_FRAMEBUFFER );

                if( status != GL_FRAMEBUFFER_COMPLETE )
                {
                    if( status == GL_FRAMEBUFFER_COMPLETE )
                    {
                        cout << "Error@GLCapture::initialize(): Failed to create framebuffer." << endl;
                        GLCapture::clear();
                        return false;
                    }
                }

                _pixels = new half[width*height*4];

                _img.create( width, height );
            }

            _width  = width;
            _height = height;

            glBindFramebuffer( GL_FRAMEBUFFER, 0 );

            return true;
        }

        void begin()
        {
            glBindFramebuffer( GL_FRAMEBUFFER, _fboId );
        }

        void end()
        {
            glReadPixels( 0, 0, _width, _height, GL_RGBA, GL_HALF_FLOAT, _pixels );

            for( GLsizei j=0; j<_height; ++j )
            for( GLsizei i=0; i<_width;  ++i )
            {
                Pixel& p = _img(i,j);

                GLsizei index = 4*(i+(_height-j+1)*_width);

                p.r = _pixels[index++];
                p.g = _pixels[index++];
                p.b = _pixels[index++];
                p.a = _pixels[index++];
            }

            glBindFramebuffer( GL_FRAMEBUFFER, 0 );
        }

        GLsizei width() const
        {
            return _width;
        }

        GLsizei height() const
        {
            return _height;
        }

        GLenum format() const
        {
            return GL_RGBA;
        }

        GLenum type() const
        {
            return GL_HALF_FLOAT;
        }

        GLuint fboId() const
        {
            return _fboId;
        }

        GLuint colorTexId() const
        {
            return _colorTexId;
        }

        GLuint depthRboId() const
        {
            return _depthRboId;
        }

        bool save( const char* filePathName ) const
        {
            return _img.save( filePathName );
        }

        const Image& getResult() const
        {
            return _img;
        }
};

BS_NAMESPACE_END

#endif

