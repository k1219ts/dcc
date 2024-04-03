//--------------//
// ZTriMeshIO.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZTriMeshIO_h_
#define _ZTriMeshIO_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

bool Load_from_obj( ZTriMesh& mesh, const char* filePathName );

ZELOS_NAMESPACE_END

#endif

