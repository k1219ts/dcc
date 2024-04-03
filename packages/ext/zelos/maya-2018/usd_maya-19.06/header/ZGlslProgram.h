//----------------//
// ZGlslProgram.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZGlslProgram_h_
#define _ZGlslProgram_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZGlslProgram
{
	private:

		GLuint      _id;
		GLenum      _texTarget;

		ZGlslShader _vs;	// vertex   shader
		ZGlslShader _fs;	// fragment shader
		ZGlslShader _gs;	// geometry shader

		map<string, GLuint> _attributeList;
		map<string, GLuint> _uniformLocationList;

	public:

		ZGlslProgram();
		//ZGlslProgram( const char* vertexShaderSrc, const char* fragmentShaderSrc );

		~ZGlslProgram();

		void reset();

		void enable() const;
		void disable();

		bool load( const char* vsFilePathName, const char* fsFilePathName );
		bool setShaders( const char* vertexShaderSrc, const char* fragmentShaderSrc );

		void setUniform1i( const GLchar* name, GLint x );
		void setUniform1f( const GLchar* name, GLfloat x );
		void setUniform2f( const GLchar* name, GLfloat x, GLfloat y );
		void setUniform3f( const char* name, float x, float y, float z );
		void setUniform4f( const char* name, float x, float y, float z, float w );
		void setUniformfv( const GLchar* name, GLfloat* v, int elementSize, int count=1 );
		void setUniformMatrix4fv( const GLchar* name, GLfloat* m, bool transpose );
		void bindTexture( const char* name, GLuint texId, GLenum textureTarget, GLint unit );

		GLuint id() const { return _id; }

		// add attribute & uniform
		void addAttribute( const string& attribute );
		void addUniform( const string& uniform );
		// An indexer that returns the location of the attribute/uniform
		GLuint operator[](const string& attribute);
		GLuint operator()(const string& uniform);
		// bind
		void bindAttribLocation( GLuint location, const char* name );
		void bindFragDataLocation( GLuint location, const char* name );
		// print
		void printActiveUniforms();
		void printActiveAttribs();
		

	private:

		void _init();
};

ZELOS_NAMESPACE_END

#endif

