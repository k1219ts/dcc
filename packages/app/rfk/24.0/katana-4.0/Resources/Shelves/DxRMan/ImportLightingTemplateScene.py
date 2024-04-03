"""
NAME: Import Lighting Template(lgt)
ICON: /backstage/share/icons/katana/light.png
KEYBOARD_SHORTCUT: 
SCOPE:
Import_Lighting_Template
katana - 3.6v2 // rfk - 23.4
"""



from Katana import NodegraphAPI, Nodes3DAPI
import VariantMenu.SetVariants as SetVariants
from Katana import KatanaFile

KatanaFile.Import('/stdrepo/LNR/04_RESOURCE/02_KATANA/Shelf/LNR_Shelf/Lighting_Template_v01.katana')

def AddGlobalGraphStateVariable(name, options):
    varGrp = NodegraphAPI.GetRootNode().getParameter('variables')

    varParam = varGrp.createChildGroup(name)
    varParam.createChildNumber('enable', 1)
    varParam.createChildString('value', options[0])
    optionParam = varParam.createChildStringArray('options',len(options))
    for optionParam, optionValue in zip(optionParam.getChildren(), options):
        optionParam.setValue(optionValue, 0)
    return varParam.getName()

selectShotVariant =['']

AddGlobalGraphStateVariable('shot', selectShotVariant)



selectedNodes = NodegraphAPI.GetAllSelectedNodes()
if selectedNodes:
    node = selectedNodes[0]
    if i.getType() == 'UsdInVariantSelect':
        SetVariants.doIt(node, create=True)
    elif node.getType() == 'Group':
        for n in node.getChildren():
            nt = n.getType()
            if nt == "UsdInVariantSelect":
                SetVariants.doIt(node, create=True)


