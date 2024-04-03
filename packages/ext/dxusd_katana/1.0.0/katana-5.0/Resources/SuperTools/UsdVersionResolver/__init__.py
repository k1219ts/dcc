import Katana
from . import v4 as UsdVersionResolver

if UsdVersionResolver:
    PluginRegistry = [
        ("SuperTool", 2, "UsdVersionResolver", (UsdVersionResolver.UsdVersionResolverNode, UsdVersionResolver.GetEditor)),
    ]
