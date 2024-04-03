import os
import NodegraphAPI

import pathConfig

def locationSetup(node):
    # clear ui param
    for i in ['i0', 'i1', 'i2']:
        node.getParameter('user.Location.shotsetup.%s' % i).setValue('', 0)

    katfile = NodegraphAPI.NodegraphGlobals.GetProjectFile()
    if not katfile:
        return

    katdir = os.path.dirname(katfile)
    source = katfile.split('/')

    if 'show' in source:
        name = source[source.index('show') + 1]
        showDir, showName = pathConfig.GetProjectPath(show=name)
        node.getParameter('user.Location.shotsetup.i0').setValue(showDir, 0)

    variableGroup = NodegraphAPI.GetRootNode().getParameter('variables')
    if variableGroup:
        # asset variant
        if variableGroup.getChild('assetVariant'):
            variant = variableGroup.getChild('assetVariant')
            assetName = variant.getChild('value').getValue(0)
            #
            node.getParameter('user.Location.shotsetup.i1').setValue(assetName, 0)
            node.getParameter('user.Location.ImageName').setValue(assetName, 0)
        # shot variant
        elif variableGroup.getChild('shotVariant'):
            variant = variableGroup.getChild('shotVariant')
            shotName = variant.getChild('value').getValue(0)
            seqName  = shotName.split('_')[0]
            #
            node.getParameter('user.Location.shotsetup.i1').setValue(seqName, 0)
            node.getParameter('user.Location.shotsetup.i2').setValue(shotName, 0)
            node.getParameter('user.Location.ImageName').setValue(shotName, 0)

