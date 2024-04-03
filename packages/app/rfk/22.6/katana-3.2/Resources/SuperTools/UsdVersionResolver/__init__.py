import Katana
import v2 as UsdVersionResolver

if UsdVersionResolver:
    PluginRegistry = [
        ("SuperTool", 2, "UsdVersionResolver", (UsdVersionResolver.UsdVersionResolverNode, UsdVersionResolver.GetEditor)),
    ]
