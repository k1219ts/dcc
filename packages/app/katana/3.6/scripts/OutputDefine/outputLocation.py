import os
import NodegraphAPI

import dfk

def locationSetup(node):
    # clear ui param
    for i in ['i0', 'i1', 'i2']:
        node.getParameter('user.Location.shotsetup.%s' % i).setValue('', 0)

    # get show dir
    allNodes = dfk.getAllConnectedNodes([node])
    usdNodes = dfk.getNodesType(allNodes, 'UsdIn')
    source   = None
    if usdNodes:
        usdfile = usdNodes[0].getParameter('fileName').getValue(0)
        source  = usdfile.split('/')
    else:
        katfile = NodegraphAPI.NodegraphGlobals.GetProjectFile()
        if katfile:
            source = katfile.split('/')
    if not source:
        return

    # node variants
    varNodes = dfk.getNodesType(allNodes, 'UsdInVariantSelect')
    nodeVariants = list()
    for n in varNodes:
        nodeVariants.append(n.getParameter('args.variantSetName.value').getValue(0))

    if 'show' in source:
        name = source[source.index('show') + 1]
        showDir = '/show/' + name
        node.getParameter('user.Location.shotsetup.i0').setValue(showDir, 0)

    variableGroup = NodegraphAPI.GetRootNode().getParameter('variables')
    if variableGroup:
        variables = list()
        for var in variableGroup.getChildren():
            variables.append(var.getName())

        # asset
        if 'asset' in variables and 'asset' in nodeVariants:
            assetName = variableGroup.getChild('asset.value').getValue(0)
            node.getParameter('user.Location.shotsetup.i1').setValue(assetName, 0)
            node.getParameter('user.Location.ImageName').setValue(assetName, 0)

        # shot, seq
        if 'shot' in variables and 'shot' in nodeVariants:
            shotName = variableGroup.getChild('shot.value').getValue(0)
            seqName  = shotName.split('_')[0]
            node.getParameter('user.Location.shotsetup.i1').setValue(seqName, 0)
            node.getParameter('user.Location.shotsetup.i2').setValue(shotName, 0)
            node.getParameter('user.Location.ImageName').setValue(shotName, 0)
