#include <pxr/pxr.h>
#include <pxr/usd/usdSkel/root.h>

#include <usdKatana/usdInPluginRegistry.h>
#include <usdKatana/attrMap.h>
#include <usdKatana/readPrim.h>
#include <usdKatana/readXformable.h>

#include <FnAttribute/FnDataBuilder.h>
#include <FnGeolibServices/FnBuiltInOpArgsUtil.h>
#include <FnGeolib/util/Path.h>
#include <FnLogging/FnLogging.h>

#include <readSkelRoot.h>

PXR_NAMESPACE_USING_DIRECTIVE

PXRUSDKATANA_USDIN_PLUGIN_DECLARE(UsdInCore_SkelBindingOp)

DEFINE_GEOLIBOP_PLUGIN(UsdInCore_SkelBindingOp)

PXRUSDKATANA_USDIN_PLUGIN_DEFINE(UsdInCore_SkelBindingOp, privateData, opArgs, interface)
{
    UsdSkelRoot skelRoot = UsdSkelRoot(privateData.GetUsdPrim());

    PxrUsdKatanaAttrMap inputAttrMap;
    inputAttrMap.set("outputLocationPath", FnKat::StringAttribute(interface.getOutputLocationPath()));
    inputAttrMap.set("opArgs", opArgs);

    PxrUsdKatanaAttrMap geomsAttrMap;
    UsdKatanaReadSkelBinding(
        skelRoot, privateData, geomsAttrMap, inputAttrMap
    );
    inputAttrMap.toInterface(interface);

    interface.setAttr("__UsdIn.skipAllChildren", FnKat::IntAttribute(1));

    FnKat::GroupAttribute geomsSSCAttrs = geomsAttrMap.build();
    if (not geomsSSCAttrs.isValid())
    {
        return;
    }

    FnKat::GroupAttribute xform;
    if (PxrUsdKatanaReadXformable(skelRoot, privateData, xform)) {
        interface.setAttr("xform", xform);
    }

    PxrUsdKatanaUsdInArgsRefPtr usdInArgs = privateData.GetUsdInArgs();

    interface.execOp(
        "UsdIn.BuildIntermediate",
        FnKat::GroupBuilder()
            .update(opArgs)
            .set("staticScene", geomsSSCAttrs)
            .build(),
        new PxrUsdKatanaUsdInPrivateData(usdInArgs->GetRootPrim(), usdInArgs, &privateData),
        PxrUsdKatanaUsdInPrivateData::Delete
    );
}

void registerPlugins() {
    USD_OP_REGISTER_PLUGIN(UsdInCore_SkelBindingOp, "UsdInCore_SkelBindingOp", 0, 1);
    PxrUsdKatanaUsdInPluginRegistry::RegisterUsdType<UsdSkelRoot>("UsdInCore_SkelBindingOp");
}
