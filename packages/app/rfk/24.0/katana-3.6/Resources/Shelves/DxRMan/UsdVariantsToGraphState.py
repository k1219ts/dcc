"""
NAME: UsdVariantSet -> GraphStateVariable
ICON: /backstage/share/icons/pxr_usd.png
DROP_TYPES:
SCOPE:

USD VariantSetName to GraphState Variable

"""

from Katana import NodegraphAPI, Nodes3DAPI
import VariantMenu.SetVariants as SetVariants


selectedNodes = NodegraphAPI.GetAllSelectedNodes()
if selectedNodes:
    node = selectedNodes[0]
    if node.getType() == 'PxrUsdInVariantSelect' or node.getType() == 'UsdInVariantSelect':
        SetVariants.doIt(node, create=True)
    elif node.getType() == 'Group':
        for n in node.getChildren():
            nt = n.getType()
            if nt == 'PxrUsdInVariantSelect' or nt == "UsdInVariantSelect":
                SetVariants.doIt(node, create=True)
