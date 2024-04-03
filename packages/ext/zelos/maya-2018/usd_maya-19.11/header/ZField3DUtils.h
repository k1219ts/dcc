//-----------------//
// ZField3DUtils.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZField3DUtils_h_
#define _ZField3DUtils_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

bool Gradient( ZVectorField3D& v, const ZScalarField3D& s, bool useOpenMP=true );
bool Divergence( ZScalarField3D& s, const ZVectorField3D& v, bool useOpenMP=true );

ZELOS_NAMESPACE_END

#endif

