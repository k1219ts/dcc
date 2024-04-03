//-------------//
// ZDataUnit.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.09.26                               //
//-------------------------------------------------------//

#ifndef _ZDataUnit_h_
#define _ZDataUnit_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZDataUnit
{
	public:

		enum DataUnit
		{
			zNone      = 0,
			zBytes     = 1,
			zKilobytes = 2,
			zMegabytes = 3,
			zGigabytes = 4
		};

	public:

		ZDataUnit() {}

		static ZString name( ZDataUnit::DataUnit dataUnit )
		{
			switch( dataUnit )
			{
				default:
				case ZDataUnit::zNone:      { return ZString("none");      }
				case ZDataUnit::zBytes:     { return ZString("bytes");     }
				case ZDataUnit::zKilobytes: { return ZString("kilobytes"); }
				case ZDataUnit::zMegabytes: { return ZString("megabytes"); }
				case ZDataUnit::zGigabytes: { return ZString("gigabytes"); }
			}
		}
};

inline ostream&
operator<<( ostream& os, const ZDataUnit& object )
{
	os << "<ZDataUnit>" << endl;
	os << endl;
	return os;
}

ZELOS_NAMESPACE_END

#endif

