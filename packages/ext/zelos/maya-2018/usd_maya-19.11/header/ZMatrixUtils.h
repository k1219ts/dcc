//----------------//
// ZMatrixUtils.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZMatrixUtils_h_
#define _ZMatrixUtils_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

// for SpeedTree's combined leaves
ZMatrix ShapeMatchingMatrix( const ZPointArray& source, const ZPointArray& target );
ZMatrix ShapeMatchingMatrix( const ZPointArray& source, const ZPointArray& target, const ZPoint& sourcePivotPosition );

ZELOS_NAMESPACE_END

#endif

