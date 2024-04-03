//----------------//
// ZImageFormat.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.09.26                               //
//-------------------------------------------------------//

#ifndef _ZImageFormat_h_
#define _ZImageFormat_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZImageFormat
{
	public:

		enum ImageFormat
		{
			zNone   = 0,  ///< none
			zInt8   = 1,  ///< 8-bit  integer (char)
			zUInt8  = 2,  ///< 8-bit  integer (unsigned char)
			zInt16  = 3,  ///< 16-bit integer (short)
			zUInt16 = 4,  ///< 16-bit integer (unsigned short)
			zInt32  = 5,  ///< 32-bit integer (int)
			zUInt32 = 6,  ///< 32-bit integer (unsigned int)
			zInt64  = 7,  ///< 64-bit integer (long)
			zUInt64 = 8,  ///< 64-bit integer (unsigned long)
			zHalf   = 9,  ///< 16-bit floating (half)
			zFloat  = 10, ///< 32-bit floating (float)
			zDouble = 11  ///< 64-bit floating (double)
		};

	public:

		ZImageFormat() {}

		static ZString name( ZImageFormat::ImageFormat imageFormat )
		{
			switch( imageFormat )
			{
				default:
				case ZImageFormat::zNone:   { return ZString("none");   }
				case ZImageFormat::zInt8:   { return ZString("int8");   }
				case ZImageFormat::zUInt8:  { return ZString("uint8");  }
				case ZImageFormat::zInt16:  { return ZString("int16");  }
				case ZImageFormat::zUInt16: { return ZString("uint16"); }
				case ZImageFormat::zInt32:  { return ZString("int32");  }
				case ZImageFormat::zUInt32: { return ZString("uint32"); }
				case ZImageFormat::zUInt64: { return ZString("uint64"); }
				case ZImageFormat::zInt64:  { return ZString("int64");  }
				case ZImageFormat::zHalf:   { return ZString("half");   }
				case ZImageFormat::zFloat:  { return ZString("float");  }
				case ZImageFormat::zDouble: { return ZString("double"); }
			}
		}

		static int size( ZImageFormat::ImageFormat imageFormat )
		{
			switch( imageFormat )
			{
				default:
				case ZImageFormat::zNone:   { return 0; }
				case ZImageFormat::zInt8:   { return sizeof(char);           }
				case ZImageFormat::zUInt8:  { return sizeof(unsigned char);  }
				case ZImageFormat::zInt16:  { return sizeof(short);          }
				case ZImageFormat::zUInt16: { return sizeof(unsigned short); }
				case ZImageFormat::zInt32:  { return sizeof(int);            }
				case ZImageFormat::zUInt32: { return sizeof(unsigned int);   }
				case ZImageFormat::zInt64:  { return sizeof(long);           }
				case ZImageFormat::zUInt64: { return sizeof(unsigned long);  }
				case ZImageFormat::zHalf:   { return sizeof(float)/2;        }
				case ZImageFormat::zFloat:  { return sizeof(float);          }
				case ZImageFormat::zDouble: { return sizeof(double);         }
			}
		}
};

inline ostream&
operator<<( ostream& os, const ZImageFormat& object )
{
	os << "<ZImageFormat>" << endl;
	os << endl;
	return os;
}

ZELOS_NAMESPACE_END

#endif

