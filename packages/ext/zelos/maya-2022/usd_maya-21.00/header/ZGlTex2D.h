//------------//
// ZGlTex2D.h //
//-------------------------------------------------------//
// author: Junghyun Cho @ Seoul National Univ.           //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2016.12.07                               //
//-------------------------------------------------------//

#ifndef _ZGlTex2D_h_
#define _ZGlTex2D_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZGlTex2D
{
	private:

		GLuint  _id;

		GLsizei _width;
		GLsizei _height;
		GLsizei _depth;

	public: 

		ZGlTex2D();
		ZGlTex2D
		(
			GLsizei w, GLsizei h,
			GLenum internalFormat=GL_RGBA8, GLenum format=GL_RGBA, GLenum type=GL_UNSIGNED_BYTE,
			GLenum filter=GL_LINEAR, GLenum clamp=GL_CLAMP_TO_EDGE,
			const GLvoid* data=NULL
		);

		~ZGlTex2D();

		void reset();

		void create
		(
			GLsizei w, GLsizei h,
			GLenum internalFormat=GL_RGBA8, GLenum format=GL_RGBA, GLenum type=GL_UNSIGNED_BYTE,
			GLenum filter=GL_LINEAR, GLenum clamp=GL_CLAMP_TO_EDGE,
			const GLvoid* data=NULL
		);

		static void enable();
		static void disable();

		void bind() const;
		static void unbind();

		void setFilter( GLenum magFilter=GL_LINEAR, GLenum minFilter=GL_LINEAR );
		void setClamp( GLenum sClamp=GL_CLAMP_TO_EDGE, GLenum tClamp=GL_CLAMP_TO_EDGE );

		GLuint id() const { return _id; }

		GLsizei width()  const { return _width;  }
		GLsizei height() const { return _height; }
};

ZELOS_NAMESPACE_END

#endif

