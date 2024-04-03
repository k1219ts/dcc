//-------------//
// ZFMMState.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZFMMState_h_
#define _ZFMMState_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZFMMState
{
	public:

		enum FMMState
		{
			zNone      = 0,
			zFar       = 1,
			zInterface = 2,
			zUpdated   = 3,
			zTrial     = 4
		};

	public:

		ZFMMState() {}
};

inline ostream&
operator<<( ostream& os, const ZFMMState& object )
{
	os << "<ZFMMState>" << endl;
	os << endl;
	return os;
}

ZELOS_NAMESPACE_END

#endif

