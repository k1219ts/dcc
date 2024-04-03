"""
NAME: DxOutputDefine -> DxOutputDefine2
ICON:
DROP_TYPES:
SCOPE:

Selected DxOutputDefine to DxOutputDefine2

"""

from Katana import NodegraphAPI

import MacroUpdate.macroUpdate as macroUpdate

selectedNodes = NodegraphAPI.GetAllSelectedNodes()
for n in selectedNodes:
    nameParam = n.getParameter('user.macroName')
    verParam  = n.getParameter('user.version')
    if nameParam and verParam:
        name = nameParam.getValue(0)
        if name == 'DxOutputDefine2':
            nameParam.setValue('DxOutputDefine', 0)
            verParam.setValue(0.1, 0)

        macroUpdate.NodeUpdate(n).doIt()
