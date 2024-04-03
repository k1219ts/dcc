//---------------//
// ZColorSpace.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.09.26                               //
//-------------------------------------------------------//

#ifndef _ZColorSpace_h_
#define _ZColorSpace_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZColorSpace
{
	public:

		enum ColorSpace
		{
			zNone = 0,
			zRGB  = 1,
			zHSV  = 2,
			zXYZ  = 3
		};

	public:

		ZColorSpace() {}

		static ZString name( ZColorSpace::ColorSpace colorSpace )
		{
			switch( colorSpace )
			{
				case ZColorSpace::zRGB: { return ZString( "RGB"  ); }
				case ZColorSpace::zHSV: { return ZString( "HSV"  ); }
				case ZColorSpace::zXYZ: { return ZString( "XYZ"  ); }
				default:                { return ZString( "NONE" ); }
			}
		}
};

inline ostream&
operator<<( ostream& os, const ZColorSpace& object )
{
	os << "<ZColorSpace>" << endl;
	os << endl;
	return os;
}

ZELOS_NAMESPACE_END

#endif

