//-------------//
// ZDataType.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.06.09                               //
//-------------------------------------------------------//

#ifndef _ZDataType_h_
#define _ZDataType_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZDataType
{
	public:

		enum DataType
		{
			zNone       =  0,
			zChar       =  1,
			zInt        =  2,
			zInt2       =  3,
			zInt3       =  4,
			zInt4       =  5,
			zInt8       =  6,
			zInt16      =  7,
			zFloat      =  8,
			zFloat2     =  9,
			zFloat3     = 10,
			zFloat4     = 11,
			zFloat8     = 12,
			zFloat16    = 13,
			zDouble     = 14,
			zDouble2    = 15,
			zDouble3    = 16,
			zDouble4    = 17,
			zDouble8    = 18,
			zDouble16   = 19,
			zPoint      = 20,
			zVector     = 21,
			zQuaternion = 22,
			zMatrix     = 23,
			zColor      = 24
		};

	public:

		ZDataType() {}

		static ZString name( ZDataType::DataType dataType )
		{
			switch( dataType )
			{
				case ZDataType::zChar:       { return ZString("char");       }
				case ZDataType::zInt:        { return ZString("int");        }
				case ZDataType::zInt2:       { return ZString("int2");       }
				case ZDataType::zInt3:       { return ZString("int3");       }
				case ZDataType::zInt4:       { return ZString("int4");       }
				case ZDataType::zInt8:       { return ZString("int8");       }
				case ZDataType::zInt16:      { return ZString("int16");      }
				case ZDataType::zFloat:      { return ZString("float");      }
				case ZDataType::zFloat2:     { return ZString("float2");     }
				case ZDataType::zFloat3:     { return ZString("float3");     }
				case ZDataType::zFloat4:     { return ZString("float4");     }
				case ZDataType::zFloat8:     { return ZString("float8");     }
				case ZDataType::zFloat16:    { return ZString("float16");    }
				case ZDataType::zDouble:     { return ZString("double");     }
				case ZDataType::zDouble2:    { return ZString("double2");    }
				case ZDataType::zDouble3:    { return ZString("double3");    }
				case ZDataType::zDouble4:    { return ZString("double4");    }
				case ZDataType::zDouble8:    { return ZString("double8");    }
				case ZDataType::zDouble16:   { return ZString("double16");   }
				case ZDataType::zPoint:      { return ZString("point");      }
				case ZDataType::zVector:     { return ZString("vector");     }
				case ZDataType::zQuaternion: { return ZString("quaternion"); }
				case ZDataType::zMatrix:     { return ZString("matrix");     }
				case ZDataType::zColor:      { return ZString("color");      }
				default:                     { return ZString("none");       }
			}
		}

		static int bytes( ZDataType::DataType dataType )
		{
			switch( dataType )
			{
				case ZDataType::zChar:       { return sizeof(char);        }
				case ZDataType::zInt:        { return sizeof(int);         }
				case ZDataType::zInt2:       { return sizeof(int)*2;       }
				case ZDataType::zInt3:       { return sizeof(int)*3;       }
				case ZDataType::zInt4:       { return sizeof(int)*4;       }
				case ZDataType::zInt8:       { return sizeof(int)*8;       }
				case ZDataType::zInt16:      { return sizeof(int)*16;      }
				case ZDataType::zFloat:      { return sizeof(float);       }
				case ZDataType::zFloat2:     { return sizeof(float)*2;     }
				case ZDataType::zFloat3:     { return sizeof(float)*3;     }
				case ZDataType::zFloat4:     { return sizeof(float)*4;     }
				case ZDataType::zFloat8:     { return sizeof(float)*8;     }
				case ZDataType::zFloat16:    { return sizeof(float)*16;    }
				case ZDataType::zDouble:     { return sizeof(double);      }
				case ZDataType::zDouble2:    { return sizeof(double)*2;    }
				case ZDataType::zDouble3:    { return sizeof(double)*3;    }
				case ZDataType::zDouble4:    { return sizeof(double)*4;    }
				case ZDataType::zDouble8:    { return sizeof(double)*8;    }
				case ZDataType::zDouble16:   { return sizeof(double)*16;   }
				case ZDataType::zPoint:      { return sizeof(float)*3;     }
				case ZDataType::zVector:     { return sizeof(float)*3;     }
				case ZDataType::zQuaternion: { return sizeof(float)*4;     }
				case ZDataType::zMatrix:     { return sizeof(float)*16;    }
				case ZDataType::zColor:      { return sizeof(float)*4;     }
				default:                     { return 0;                   }
			}
		}
};

inline ostream&
operator<<( ostream& os, const ZDataType& object )
{
	os << "<ZDataType>" << endl;
	os << endl;
	return os;
}

ZELOS_NAMESPACE_END

#endif

