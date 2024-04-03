//--------------------//
// ZLevelSet2DUtils.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZLevelSet2DUtils_h_
#define _ZLevelSet2DUtils_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

void Voxelize( const ZTriMesh& mesh, ZScalarField2D& lvs, bool useOpenMP=true );
void Voxelize( const ZTriMesh& mesh, const ZVectorArray& vVel, ZScalarField2D& lvs, ZVectorField2D& vel, bool useOpenMP=true );

ZELOS_NAMESPACE_END

#endif

