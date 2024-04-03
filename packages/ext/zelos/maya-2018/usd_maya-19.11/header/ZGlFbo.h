//----------//
// ZGlFbo.h //
//-------------------------------------------------------//
// author: Junghyun Cho @ Seoul National Univ.           //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZGlFbo_h_
#define _ZGlFbo_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

/// @brief Frame-buffer Object
class ZGlFbo 
{
	private:

		bool           _depthOnly;

		GLuint         _id;
		mutable GLuint _id0; // previous FBO ID

	public:

		ZGlFbo();

		~ZGlFbo();

		void reset();

		void create();
		void create( const ZGlTex2D& tex, bool isDepth );
		void create( const ZGlTex2D& color, const ZGlTex2D& depth );
		void create( const ZGlTex2D& color, const ZGlRbo& depth );

		void bind() const;
		static void unbind();

		void attach( const ZGlTex2D& tex, GLenum attachment=GL_COLOR_ATTACHMENT0 ) const;
		void attach( const ZGlRbo& rbo, GLenum attachment=GL_DEPTH_ATTACHMENT ) const;
		void detach( GLenum attachment ) const;
		void detachAll() const;

		// Call this function right after createWith() or attach() executed.
		bool checkStatus() const;

		static int getMaxColorAttachments();
		GLenum attachedType( GLenum attachment ) const;
		GLuint attachedId( GLenum attachment ) const;
		GLuint id() const { return _id; }

	private:

		void _guardedBind() const;
		void _guardedUnbind() const;
};

ZELOS_NAMESPACE_END

#endif

