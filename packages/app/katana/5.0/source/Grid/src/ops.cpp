#include <FnAttribute/FnAttribute.h>
#include <FnAttribute/FnGroupBuilder.h>

#include <FnPluginSystem/FnPlugin.h>
#include <FnGeolib/op/FnGeolibOp.h>
#include <FnGeolib/util/Path.h>

#include <pystring/pystring.h>

#include <FnGeolibServices/FnGeolibCookInterfaceUtilsService.h>

namespace {

class GridOp : public Foundry::Katana::GeolibOp
{
public:

    static void setup(Foundry::Katana::GeolibSetupInterface &interface)
    {
        interface.setThreading(Foundry::Katana::GeolibSetupInterface::ThreadModeConcurrent);
    }

    static void cook(Foundry::Katana::GeolibCookInterface &interface)
    {
        if( interface.atRoot() )
        {
            interface.stopChildTraversal();
        }

        FnAttribute::GroupBuilder gb;
        gb.set("width", FnAttribute::IntAttribute(1));
        interface.setAttr("attrTest", gb.build());
    }

};

DEFINE_GEOLIBOP_PLUGIN(GridOp)

}

void registerPlugins()
{
    REGISTER_PLUGIN(GridOp, "Grid", 0, 1);
}
