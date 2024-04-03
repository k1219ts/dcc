//---------------//
// ZGlslShader.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZGlslShader_h_
#define _ZGlslShader_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZGlslShader 
{
	private:

		GLuint _id;
		GLenum _type;

	public:

		ZGlslShader();
		ZGlslShader( const char* filePathName, GLenum shaderType );

		~ZGlslShader();

		void reset();

		bool loadFromFile( const char* filePathName, GLenum shaderType );

		GLuint id() const { return _id; }
		GLenum type() const { return _type; }

	private:

		void _init();
};

ZELOS_NAMESPACE_END

#endif

