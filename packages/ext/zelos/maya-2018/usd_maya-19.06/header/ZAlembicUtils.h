//-----------------//
// ZAlembicUtils.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.05.25                               //
//-------------------------------------------------------//

#ifndef _ZAlembicUtils_h_
#define _ZAlembicUtils_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

bool GetMetaData( const Alembic::Abc::MetaData& md, ZStringArray& keys, ZStringArray& values );

void GetAllDescendantObjects( const ZAlembicObject& progenitorObject, ZAlembicObjectArray& list, bool leavesOnly=false );

void GetAllDescendantProperties( const ZAlembicProperty& progenitorProperty, ZAlembicPropertyArray& list );

void GetAllParentObjects( const ZAlembicObject& childObject, ZAlembicObjectArray& list );

void GetWorldMatrix( const ZAlembicObject& shapeObject, ZMatrix& worldMatrix, int frame=0 );

void ZDrawPolyMesh( const ZPointArray& vPos, const ZIntArray& vCount, const ZIntArray& vConnects );

ZELOS_NAMESPACE_END

#endif

