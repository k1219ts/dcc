//------------//
// ZGlTex3D.h //
//-------------------------------------------------------//
// author: Junghyun Cho @ Seoul National Univ.           //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZGlTex3D_h_
#define _ZGlTex3D_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZGlTex3D
{
	private:

		GLuint  _id;

		GLsizei _width;
		GLsizei _height;
		GLsizei _depth;

	public: 

		ZGlTex3D();
		ZGlTex3D( GLsizei w, GLsizei h, GLsizei d,
                GLenum GLsizeiernalFormat=GL_RGBA8, GLenum format=GL_RGBA, GLenum type=GL_UNSIGNED_BYTE,
                GLenum filter=GL_LINEAR, GLenum clamp=GL_CLAMP_TO_EDGE,
                const GLvoid* data=NULL );

		~ZGlTex3D();

		void reset();

		void create
		(
			GLsizei w, GLsizei h, GLsizei d,
            GLenum internalFormat=GL_RGBA8, GLenum format=GL_RGBA, GLenum type=GL_UNSIGNED_BYTE,
            GLenum filter=GL_LINEAR, GLenum clamp=GL_CLAMP_TO_EDGE,
            const GLvoid* data=NULL
		);

		static void enable();
		static void disable();

		void bind() const;
		static void unbind();

		void setFilter( GLenum magFilter, GLenum minFilter );
		void setClamp( GLenum sClamp, GLenum tClamp, GLenum rClamp );

		GLuint id() const { return _id; }

		GLsizei width()  const { return _width;  }
		GLsizei height() const { return _height; }
		GLsizei depth()  const { return _depth;  }
};

ZELOS_NAMESPACE_END

#endif

