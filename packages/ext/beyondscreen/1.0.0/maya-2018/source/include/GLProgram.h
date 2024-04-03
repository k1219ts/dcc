#ifndef _BS_GLProgram_h_
#define _BS_GLProgram_h_

#include <BeyondScreen.h>

BS_NAMESPACE_BEGIN

class GLProgram
{
    private:

        GLuint _id = 0; // program ID
        vector<GLuint> _shaderId; // shader IDs

    public:

        GLProgram()
        {
            // nothing to do
        }

        virtual ~GLProgram()
        {
            GLProgram::reset();
        }

        void reset()
        {
            for( size_t i=0; i<_shaderId.size(); ++i )
            {
                glDetachShader( _id, _shaderId[i] );
                glDeleteShader( _shaderId[i] );
            }

            _shaderId.clear();

            if( _id > 0 )
            {
                glDeleteProgram( _id );
                glUseProgram( 0 );
            }

            _id = 0;
        }

        bool addShader( const GLenum type, const char* source )
        {
            if( _id > 0 )
            {
                cout << "Error@GLProgram::addShader(): Too late." << endl;
                return false;
            }

            bool isValidType = false;
            if( type == GL_VERTEX_SHADER          ) { isValidType = true; }
            if( type == GL_FRAGMENT_SHADER        ) { isValidType = true; }
            if( type == GL_GEOMETRY_SHADER        ) { isValidType = true; }
            if( type == GL_TESS_CONTROL_SHADER    ) { isValidType = true; }
            if( type == GL_TESS_EVALUATION_SHADER ) { isValidType = true; }

            if( isValidType == false )
            {
                cout << "Error@GLProgram::addShader(): Invalid shader type." << endl;
                return false;
            }

            const GLuint shaderId = glCreateShader( type );

            glShaderSource( shaderId, 1, &source, 0 );

            glCompileShader( shaderId );
            {
                GLint status = GL_TRUE;
                glGetShaderiv( shaderId, GL_COMPILE_STATUS, &status );
                
                if( status != GL_TRUE )
                {
                    GLint logLength = 0;
                    glGetShaderiv( shaderId, GL_INFO_LOG_LENGTH, &logLength );

                    std::vector<GLchar> log( logLength );
                    glGetShaderInfoLog( shaderId, logLength, &logLength, &log[0] );

                    cout << "Error@GLProgram::addShader(): " << &log[0] << endl;

                    glDetachShader( _id, shaderId );
                    glDeleteShader( shaderId );

                    return false;
                }
            }

            _shaderId.push_back( shaderId );

            return true;
        }

        bool link()
        {
            if( _id > 0 ) { return true; }

            _id = glCreateProgram();

            if( _id == 0 )
            {
                cout << "Error@GLProgram::link(): Failed to create GLSL program." << endl;
                return false;
            }

            for( size_t i=0; i<_shaderId.size(); ++i )
            {
                glAttachShader( _id, _shaderId[i] );
            }

            glLinkProgram( _id );
            {
                GLint status = 0;
                glGetProgramiv( _id, GL_LINK_STATUS, &status );

                if( status != GL_TRUE )
                {
                    GLint logLength = 0;
                    glGetProgramiv( _id, GL_INFO_LOG_LENGTH, &logLength );

                    std::vector<GLchar> log( logLength+1 );
                    glGetProgramInfoLog( _id, logLength, &logLength, &(log[0]) );

                    std::string logMsg = (const std::string&)( &(log[0]) );
                    cout << "Error@GLProgram::link(): " << logMsg << endl;

                    GLProgram::reset();

                    return false;
                }
            }

            // Disable the programmable processors so that the fixed functionality will be used.
            glUseProgram( 0 );

            return true;
        }

        GLuint id() const
        {
            return _id;
        }

        void enable() const
        {
            glUseProgram( _id );
        }

        void disable() const
        {
            glUseProgram( 0 );
        }

        GLuint attributeLocation( const char* name )
        {
            return glGetAttribLocation( _id, name );
        }

        GLuint uniformLocation( const char* name )
        {
            return glGetUniformLocation( _id, name );
        }

        bool bindModelViewMatrix( const char* name, bool transpose=false ) const
        {
            const GLint location = glGetUniformLocation( _id, name );
            if( location < 0 ) { return false; }

            float matrix[16];
            glGetFloatv( GL_MODELVIEW_MATRIX, matrix );

            glUniformMatrix4fv( location, 1, ( transpose ? GL_TRUE : GL_FALSE ), matrix );
            return true;
        }

        bool bindProjectionMatrix( const char* name, bool transpose=false ) const
        {
            const GLint location = glGetUniformLocation( _id, name );
            if( location < 0 ) { return false; }

            float matrix[16];
            glGetFloatv( GL_PROJECTION_MATRIX, matrix );

            glUniformMatrix4fv( location, 1, ( transpose ? GL_TRUE : GL_FALSE ), matrix );
            return true;
        }

        bool bind( const char* name, const int& v ) const
        {
            const GLint location = glGetUniformLocation( _id, name );
            if( location < 0 ) { return false; }

            glUniform1i( location, v );
            return true;
        }

        bool bind( const char* name, const float& v ) const
        {
            const GLint location = glGetUniformLocation( _id, name );
            if( location < 0 ) { return false; }

            glUniform1f( location, v );
            return true;
        }

        bool bind( const char* name, const double& v ) const
        {
            const GLint location = glGetUniformLocation( _id, name );
            if( location < 0 ) { return false; }

            glUniform1f( location, (float)v );
            return true;
        }

        bool bind( const char* name, const Vector& v ) const
        {
            const GLint location = glGetUniformLocation( _id, name );
            if( location < 0 ) { return false; }

            glUniform3f( location, (float)v.x, (float)v.y, (float)v.z );
            return true;
        }

		bool bindTexture( const char* name, GLuint textureId, GLenum type ) const
        {
            GLint location = glGetUniformLocation( _id, name );
            if( location < 0 ) { return false; }

            glActiveTexture( GL_TEXTURE0 + (textureId-1) );
            glBindTexture( type, textureId );
            glUniform1i( location, (textureId-1) );

            return true;
        }
};

BS_NAMESPACE_END

#endif

