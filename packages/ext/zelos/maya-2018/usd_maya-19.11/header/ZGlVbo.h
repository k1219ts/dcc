//----------//
// ZGlVbo.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZGlVbo_h_
#define _ZGlVbo_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

/// @brief Vertex Buffer Object
class ZGlVbo
{
	public:

		GLuint _id;

	public:

		ZGlVbo();
		~ZGlVbo();

		GLuint id() const { return _id; }

		static void unbind();
};

ostream&
operator<<( ostream& os, const ZGlVbo& object );

ZELOS_NAMESPACE_END

#endif

