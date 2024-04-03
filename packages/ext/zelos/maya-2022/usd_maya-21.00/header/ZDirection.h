//--------------//
// ZDirection.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.03.30                               //
//-------------------------------------------------------//

#ifndef _ZDirection_h_
#define _ZDirection_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZDirection
{
	public:

		enum Direction
		{
			xPositive = 0,
			yPositive = 1,
			zPositive = 2,
			xNegative = 3,
			yNegative = 4,
			zNegative = 5
		};

	public:

		ZDirection() {}

		static ZString name( ZDirection::Direction direction )
		{
			switch( direction )
			{
				case ZDirection::xPositive: { return ZString("xPositive"); }
				case ZDirection::yPositive: { return ZString("yPositive"); }
				case ZDirection::zPositive: { return ZString("zPositive"); }
				case ZDirection::xNegative: { return ZString("xNegative"); }
				case ZDirection::yNegative: { return ZString("yNegative"); }
				case ZDirection::zNegative: { return ZString("zNegative"); }
				default:                    { return ZString("none");      }
			}
		}
};

inline ostream&
operator<<( ostream& os, const ZDirection& object )
{
	os << "<ZDirection>" << endl;
	os << endl;
	return os;
}

ZELOS_NAMESPACE_END

#endif

