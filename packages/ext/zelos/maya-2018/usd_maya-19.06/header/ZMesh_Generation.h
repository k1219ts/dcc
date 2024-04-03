//--------------------//
// ZMesh_Generation.h //
//-------------------------------------------------------//
// author: Taeyong Kim @ nVidia                          //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2017.03.30                               //
//-------------------------------------------------------//

#ifndef _ZMesh_Generation_h_
#define _ZMesh_Generation_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

bool MakePlane( ZMesh& mesh, float width=1, float height=1, int subdivisionWidth=10, int subdivisionHeight=10, ZDirection::Direction axis=ZDirection::yPositive, bool createUVs=true );

bool MakeSphere( ZMesh& mesh, float radius=1, int subdivisionAxis=10, int subdivisionHeight=10, ZDirection::Direction axis=ZDirection::xPositive, bool createUVs=true );

ZELOS_NAMESPACE_END

#endif

