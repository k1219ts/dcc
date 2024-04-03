add_library(DxUsdFeather MODULE
    src/Data.cpp
    src/NurbsCurve.cpp
    src/Utils.cpp
    src/Deformer.cpp
    src/DxUsdFeatherOp.cpp)

set_target_properties(DxUsdFeather PROPERTIES PREFIX "")

target_include_directories(DxUsdFeather
    PRIVATE
    include)

target_link_libraries(DxUsdFeather
    PRIVATE
    Katana::FnAttribute
    Katana::FnGeolibOpPlugin
    Katana::FnGeolibServices
    OpenEXR::OpenEXR)

install(FILES DxUsdFeatherNode.py DESTINATION Plugins)
install(TARGETS DxUsdFeather DESTINATION Ops)
