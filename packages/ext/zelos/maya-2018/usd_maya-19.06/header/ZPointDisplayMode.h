//---------------------//
// ZPointDisplayMode.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZPointDisplayMode_h_
#define _ZPointDisplayMode_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZPointDisplayMode
{
	public:

		enum PointDisplayMode
		{
			zNone   = 0, ///< none
			zPoint  = 1, ///< point
			zSphere = 2  ///< sphere
		};

	public:

		ZPointDisplayMode() {}
};

inline ostream&
operator<<( ostream& os, const ZPointDisplayMode& object )
{
	os << "<ZPointDisplayMode>" << endl;
	os << endl;
	return os;
}

ZELOS_NAMESPACE_END

#endif

