#ifndef USDKATANA_READSKELBIND_H
#define USDKATANA_READSKELBIND_H

#include <pxr/pxr.h>
#include <usdKatana/attrMap.h>
#include <usdKatana/usdInPrivateData.h>

PXR_NAMESPACE_OPEN_SCOPE

void
UsdKatanaReadSkelBinding(
    const UsdSkelRoot &skelRoot,
    const PxrUsdKatanaUsdInPrivateData &data,
    PxrUsdKatanaAttrMap &geomsAttrMap,
    PxrUsdKatanaAttrMap &inputAttrMap
);

PXR_NAMESPACE_CLOSE_SCOPE

#endif
