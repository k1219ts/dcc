"""
NAME: Macro Update
ICON: icon.png
DROP_TYPES:
SCOPE:

Selected Macro Update

"""

from Katana import NodegraphAPI

import MacroUpdate.macroUpdate as macroUpdate

selectedNodes = NodegraphAPI.GetAllSelectedNodes()
for n in selectedNodes:
    macroUpdate.NodeUpdate(n).doIt()
