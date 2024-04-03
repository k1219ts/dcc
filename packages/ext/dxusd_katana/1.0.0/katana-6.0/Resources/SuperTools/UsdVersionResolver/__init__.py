import Katana
from . import v5 as UsdVersionResolver

if UsdVersionResolver:
    PluginRegistry = [
        ("SuperTool", 2, "UsdVersionResolver", (UsdVersionResolver.UsdVersionResolverNode, UsdVersionResolver.GetEditor)),
    ]
