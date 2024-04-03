"""
NAME: Import AssetTemplate
ICON: /backstage/share/icons/assetTemplate.png
DROP_TYPES:
SCOPE:

Import AssetTemplate Scene

"""

from Katana import NodegraphAPI, Nodes3DAPI
import VariantMenu.SetVariants as SetVariants
from Katana import KatanaFile


KatanaFile.Import('/assetlib/3D/katana/Asset_Katana3.6_rman23.4_dxusd2.0_template.katana')


def AddGlobalGraphStateVariable(name, options):
    varGrp = NodegraphAPI.GetRootNode().getParameter('variables')

    varParam = varGrp.createChildGroup(name)
    varParam.createChildNumber('enable', 1)
    varParam.createChildString('value', options[0])
    optionParam = varParam.createChildStringArray('options',len(options))
    for optionParam, optionValue in zip(optionParam.getChildren(), options):
        optionParam.setValue(optionValue, 0)
    return varParam.getName()



selectAssetOptions =['Show','MtK','Library','Path']
CamOptions =['katana','MtK','abc','USD','Turntable_Auto','Turntable_Free']
assetlibOptions =['']
branchlibOptions =['']
assetShowOptions =['']
branchOptions =['']

AddGlobalGraphStateVariable('selectAsset', selectAssetOptions)
AddGlobalGraphStateVariable('Cam', CamOptions)
AddGlobalGraphStateVariable('assetlib', assetlibOptions)
AddGlobalGraphStateVariable('elementlib', branchlibOptions)
AddGlobalGraphStateVariable('Asset', assetShowOptions)
AddGlobalGraphStateVariable('Branch', branchOptions)

selectedNodes = NodegraphAPI.GetAllSelectedNodes()
if selectedNodes:
    node = selectedNodes[0]
    if node.getType() == 'PxrUsdInVariantSelect':
        SetVariants.doIt(node, create=True)
    elif node.getType() == 'Group':
        for n in node.getChildren():
            if n.getType() == 'PxrUsdInVariantSelect':
                SetVariants.doIt(node, create=True)
