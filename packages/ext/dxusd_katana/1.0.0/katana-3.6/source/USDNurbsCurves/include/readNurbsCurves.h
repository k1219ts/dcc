#ifndef USDKATANA_READNURBS_H
#define USDKATANA_READNURBS_H

#include <pxr/pxr.h>

#include <usdKatana/usdInPrivateData.h>

PXR_NAMESPACE_OPEN_SCOPE

void readUSDNurbsCurves(
    FnKat::GeolibCookInterface& interface,
    FnKat::GroupAttribute opArgs,
    const PxrUsdKatanaUsdInPrivateData& privateData
);

PXR_NAMESPACE_CLOSE_SCOPE

#endif
