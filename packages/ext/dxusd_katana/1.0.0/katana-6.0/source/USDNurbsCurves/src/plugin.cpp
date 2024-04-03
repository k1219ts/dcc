#include <usdKatana/attrMap.h>
#include <usdKatana/usdInPluginRegistry.h>

#include <pxr/usd/usdGeom/nurbsCurves.h>

#include "readNurbsCurves.h"

PXR_NAMESPACE_USING_DIRECTIVE

USDKATANA_USDIN_PLUGIN_DECLARE(USDNurbsToCurvesOp)
DEFINE_GEOLIBOP_PLUGIN(USDNurbsToCurvesOp)
USDKATANA_USDIN_PLUGIN_DEFINE(
    USDNurbsToCurvesOp, privateData, opArgs, interface
) {
    readUSDNurbsCurves(interface, opArgs, privateData);
}

void registerPlugins() {
    USD_OP_REGISTER_PLUGIN(USDNurbsToCurvesOp, "USDNurbsToCurves", 0, 1);
    UsdKatanaUsdInPluginRegistry::RegisterUsdType<UsdGeomNurbsCurves>("USDNurbsToCurves");
}
