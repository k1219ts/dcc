//----------//
// ZGlRbo.h //
//-------------------------------------------------------//
// author: Junghyun Cho @ Seoul National Univ.           //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZGlRbo_h_
#define _ZGlRbo_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

/// @brief Render-buffer Object
class ZGlRbo
{
	private:

		GLuint  _id;

		GLsizei _width;
		GLsizei _height;

	public:

		ZGlRbo();
		ZGlRbo( GLsizei w, GLsizei h, GLenum GLsizeiernalFormat );

		~ZGlRbo();

		void reset();

		void create( GLsizei w, GLsizei h, GLenum GLsizeiernalFormat );

		void bind() const;
		static void unbind();

		GLuint id() const { return _id; }

		GLsizei getWidth()  const { return _width;  }
		GLsizei getHeight() const { return _height; }
};

ZELOS_NAMESPACE_END

#endif

