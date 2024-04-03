#ifndef PXRUSDKATANA_SKINNINGUSDSKEL_H
#define PXRUSDKATANA_SKINNINGUSDSKEL_H

#include <pxr/pxr.h>

#include <usdKatana/attrMap.h>
#include <usdKatana/usdInPrivateData.h>

PXR_NAMESPACE_OPEN_SCOPE

void DxReadSkelRootGeom(
    FnKat::GeolibCookInterface& interface,
    const UsdPrim& rootPrim,
    const PxrUsdKatanaUsdInPrivateData& data,
    PxrUsdKatanaAttrMap& inputAttrMap
);

PXR_NAMESPACE_CLOSE_SCOPE

#endif
