import maya.cmds as cmds


def AgentTextureAttributes():
    objects = cmds.ls('Miarmy_Contents', dag=True, type='surfaceShape', ni=True, l=True)
    for shape in objects:
        if cmds.attributeQuery('rman__riattr__user_txAssetName', n=shape, ex=True):
            assetName = cmds.getAttr('%s.rman__riattr__user_txAssetName' % shape)
            layerName = cmds.getAttr('%s.rman__riattr__user_txLayerName' % shape)
            setVal = 'txAssetName %s txLayerName %s' % (assetName, layerName)

            trans = cmds.listRelatives(shape, p=True, f=True)[0]
            if not cmds.attributeQuery('McdRMAttr', n=trans, ex=True):
                cmds.addAttr(trans, ln='McdRMAttr', dt='string')
            cmds.setAttr('%s.McdRMAttr' % trans, setVal, type='string')
