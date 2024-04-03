//------------------//
// ZImageMapUtils.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZImageMapUtils_h_
#define _ZImageMapUtils_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

bool CalculateNormalMap( ZImageMap& nrmMap, const ZImageMap& hgtMap, float strength=1.f, bool useOpenMP=true );

ZELOS_NAMESPACE_END

#endif

