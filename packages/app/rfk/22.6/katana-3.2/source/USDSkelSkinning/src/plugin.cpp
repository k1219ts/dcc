#include <usdKatana/attrMap.h>
#include <usdKatana/readPrim.h>
#include <usdKatana/readXformable.h>
#include <usdKatana/usdInPluginRegistry.h>

#include <pxr/usd/usdSkel/root.h>

#include "skinningUsdSkel.h"

PXR_NAMESPACE_USING_DIRECTIVE

PXRUSDKATANA_USDIN_PLUGIN_DECLARE(UsdSkelSkinningOp)

DEFINE_GEOLIBOP_PLUGIN(UsdSkelSkinningOp)

PXRUSDKATANA_USDIN_PLUGIN_DEFINE(UsdSkelSkinningOp, privateData, opArgs, interface)
{
    const auto prim = privateData.GetUsdPrim();

    PxrUsdKatanaAttrMap inputAttrMap;
    inputAttrMap.set("outputLocationPath", FnKat::StringAttribute(interface.getOutputLocationPath()));
    inputAttrMap.set("opArgs", opArgs);

    interface.setAttr("__UsdIn.skipAllChildren", FnKat::IntAttribute(1));

    PxrUsdKatanaAttrMap attrs;
    PxrUsdKatanaReadPrim(prim, privateData, attrs);
    attrs.toInterface(interface);

    FnKat::GroupAttribute xform;
    if (PxrUsdKatanaReadXformable(UsdSkelRoot(prim), privateData, xform)) {
        interface.setAttr("xform", xform);
    }

    DxReadSkelRootGeom(interface, prim, privateData, inputAttrMap);
}

void registerPlugins() {
    USD_OP_REGISTER_PLUGIN(UsdSkelSkinningOp, "UsdSkelSkinning", 0, 1);
    PxrUsdKatanaUsdInPluginRegistry::RegisterUsdType<UsdSkelRoot>("UsdSkelSkinning");
}
