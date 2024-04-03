"""
NAME: Import Lighting Truntable Template
ICON: /backstage/share/icons/katana/light.png
KEYBOARD_SHORTCUT: 
SCOPE:
Import_Lighting_Truntable_Template
katana - 6.0 // rfk - 25.2
"""



from Katana import NodegraphAPI, Nodes3DAPI
import VariantMenu.SetVariants as SetVariants
from Katana import KatanaFile

KatanaFile.Import('/stdrepo/LNR/00_MEMBER/taeseob/lookdevTemplate/katana/lookdevTemplate-6.0.katana')

def AddGlobalGraphStateVariable(name, options):
    varGrp = NodegraphAPI.GetRootNode().getParameter('variables')

    varParam = varGrp.createChildGroup(name)
    varParam.createChildNumber('enable', 1)
    varParam.createChildString('value', options[0])
    optionParam = varParam.createChildStringArray('options',len(options))
    for optionParam, optionValue in zip(optionParam.getChildren(), options):
        optionParam.setValue(optionValue, 0)
    return varParam.getName()




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


