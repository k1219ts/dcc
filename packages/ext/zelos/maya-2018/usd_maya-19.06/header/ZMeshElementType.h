//--------------------//
// ZMeshElementType.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.09.26                               //
//-------------------------------------------------------//

#ifndef _ZMeshElementType_h_
#define _ZMeshElementType_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZMeshElementType
{
	public:

		enum MeshElementType
		{
			zNone  = 0, ///< none
			zPoint = 1, ///< point
			zLine  = 2, ///< line
			zFace  = 3, ///< face
			zTet   = 4, ///< tetrahedron
			zCube  = 5  ///< cube
		};

	public:

		ZMeshElementType() {}

		static ZString name( ZMeshElementType::MeshElementType type )
		{
			switch( type )
			{
				default:
				case ZMeshElementType::zNone : { return ZString("none");        }
				case ZMeshElementType::zPoint: { return ZString("point");       }
				case ZMeshElementType::zLine : { return ZString("line");        }
				case ZMeshElementType::zFace : { return ZString("face");        }
				case ZMeshElementType::zTet  : { return ZString("tetrahedron"); }
				case ZMeshElementType::zCube : { return ZString("cube");        }
			}
		}
};

inline
ostream& operator<<( ostream& os, const ZMeshElementType& object )
{
	os << "<ZMeshElementType>" << endl;
	os << endl;
	return os;
}

ZELOS_NAMESPACE_END

#endif

