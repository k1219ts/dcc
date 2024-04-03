//---------------//
// ZArrayUtils.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.02.27                               //
//-------------------------------------------------------//

#ifndef _ZArrayUtils_h_
#define _ZArrayUtils_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

// ZFloatArray <- ZDoubleArray
void Copy( ZFloatArray& to, const ZDoubleArray& from );

// ZDoubleArray <- ZFloatArray
void Copy( ZDoubleArray& to, const ZFloatArray& from );

// ZIntArray <- ZIntSetArray
void Copy( ZIntArray& to, const ZIntSetArray& from );

void AssignGroups( ZIntArray& groupIds, const ZFloatArray& likelihoods, int randomSeed=0 );

void ReverseConnections( const ZIntArray& vCounts, ZIntArray& vConnects );

void ComputeVertexNormals( const ZPointArray& vertexPositions, const ZIntArray& vCounts, const ZIntArray& vConnects, ZVectorArray& vertexNormals );

ZELOS_NAMESPACE_END

#endif

