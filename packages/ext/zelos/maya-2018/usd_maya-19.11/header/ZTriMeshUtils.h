//-----------------//
// ZTriMeshUtils.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.11.12                               //
//-------------------------------------------------------//

#ifndef _ZTriMeshUtils_h_
#define _ZTriMeshUtils_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

// Calculate attributes from barycentric coordinates.
void GetPositions( const ZTriMesh& mesh, const ZIntArray& triIndices, const ZFloat3Array& baryCoords, ZPointArray& positions, bool useOpenMP=true );

void GetUVs( const ZTriMesh& mesh, const ZIntArray& triIndices, const ZFloat3Array& baryCoords, ZPointArray& uvs, bool useOpenMP=true );

// Calculate the local axes of the points on the mesh using the predefined vertex axes.
// @param[in] mesh The given mesh.
// @param[in] triIndices The triangle indices of the points.
// @param[in] baryCoords The barycentric coordinates of the points.
// @param[in] vAxes The precalculated local axes defined on the mesh vertices.
// @param[out] axes The local axes of the given points.
void GetLocalAxes( const ZTriMesh& mesh, const ZIntArray& triIndices, const ZFloat3Array& baryCoords, const ZAxisArray& vAxes, ZAxisArray& axes, bool useOpenMP=true );

// Calculate the local axes of the points on the mesh.
// @param[in] mesh The given mesh.
// @param[in] triIndices The triangle indices of the points.
// @param[in] baryCoords The barycentric coordinates of the points.
// @param[out] axes The local axes of the given points.
void GetLocalAxes( const ZTriMesh& mesh, const ZIntArray& triIndices, const ZFloat3Array& baryCoords, ZAxisArray& axes, bool useOpenMP=true );

void GetNormalizedPositions( const ZTriMesh& mesh, const ZIntArray& triIndices, const ZFloat3Array& baryCoords, ZPointArray& normalizedPositions, bool useOpenMP=true );

void GetVertexDisplacement( const ZTriMesh& currMesh, const ZTriMesh& prevMesh, ZVectorArray& vVel, bool useOpenMP=true );

void GetTangentSpace( const ZTriMesh& mesh, ZAxisArray& axis, bool atVertex, bool useOpenMP=true );

void GetPointsFromUVs( const ZTriMesh& mesh, const ZPointArray& uvs, ZIntArray& triIndices, ZFloat3Array& baryCoords, bool useOpenMP=true );

ZELOS_NAMESPACE_END

#endif

