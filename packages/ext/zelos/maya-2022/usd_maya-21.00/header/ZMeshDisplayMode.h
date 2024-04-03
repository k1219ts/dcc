//--------------------//
// ZMeshDisplayMode.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZMeshDisplayMode_h_
#define _ZMeshDisplayMode_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZMeshDisplayMode
{
	public:

		enum MeshDisplayMode
		{
			zNone        = 0, ///< none
			zPoints      = 1, ///< vertex points
			zWireframe   = 2, ///< wire frame
			zSurface     = 3, ///< shaded surface
			zWireSurface = 4  ///< wire frame on shaded surface
		};

	public:

		ZMeshDisplayMode() {}
};

inline ostream&
operator<<( ostream& os, const ZMeshDisplayMode& object )
{
	os << "<ZMeshDisplayMode>" << endl;
	os << endl;
	return os;
}

ZELOS_NAMESPACE_END

#endif

