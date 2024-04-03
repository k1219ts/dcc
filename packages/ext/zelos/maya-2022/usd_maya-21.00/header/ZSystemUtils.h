//----------------//
// ZSystemUtils.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZSystemUtils_h_
#define _ZSystemUtils_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

bool ZGetLocalIPAddress( ZString& ipAddress );
void ZGetSystemInfo( ZString& systemInfo );

ZELOS_NAMESPACE_END

#endif

