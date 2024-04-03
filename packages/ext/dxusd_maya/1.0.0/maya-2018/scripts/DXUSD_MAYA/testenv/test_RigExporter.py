#coding:utf-8
from __future__ import print_function

from pymel.all import *
import maya.cmds as cmds
import os

try:
    cmds.loadPlugin('pxrUsd')
except:
    # raise RuntimeError("not found pxrUsd plugin")
    assert False, "not found pxrUsd plugin"

if not cmds.pluginInfo('pxrUsd', q=True, l=True):
    assert False, "Failed Load PxrUsd Plugin"

# -------------------------------------------------------------------------------
# Rig Asset
def rigAssetExport(filename):
    import DXUSD_MAYA.Rig as Rig
    import DXUSD_MAYA.Exporters as exp
    import DXUSD.Vars as var

    # Requires load backstageMenu plugin
    pluginName = 'DXUSD_Maya'
    try:
        cmds.loadPlugin(pluginName)
    except:
        assert False, "not found %s plugin" % pluginName

    if not cmds.pluginInfo(pluginName, q=True, l=True):
        assert False, "Failed Load %s Plugin" % pluginName

    if not os.path.exists(filename):
        assert False, "Not exists test scene"

    cmds.file(filename, o=True, f=True)

    # # TEST
    # rigNode = cmds.ls(type='dxRig')[0]
    #
    # variantData = dict()    # {ctrlattr: [(name, index), ...]}
    # chnode = cmds.listConnections('%s.variant' % rigNode)
    # if chnode:
    #     chnode   = chnode[0]
    #     inputs   = cmds.getAttr('%s.input' % chnode, mi=True)
    #     ctrlattr = cmds.listConnections('%s.selector' % chnode, p=True)
    #     if inputs and ctrlattr:
    #         ctrlattr   = ctrlattr[0]
    #         selections = list()
    #         for i in inputs:
    #             out = cmds.getAttr('%s.input[%d]' % (chnode, i))
    #             selections.append(out)
    #         variants = list(set(selections))
    #         if len(variants) > 1:
    #             variantData[ctrlattr] = list()
    #             for v in variants:
    #                 variantData[ctrlattr].append((v, selections.index(v)))
    # print('> variants :', variantData)
    #
    # if variantData:
    #     for ctrlattr, variants in variantData.items():
    #         for n, i in variants:
    #             cmds.setAttr(ctrlattr, i)
    #             arg = exp.ARigAssetExporter()
    #             arg.scene = filename
    #             arg.node  = cmds.ls(type='dxRig')[0]
    #             arg.variant = n
    #             # if arg.Treat():
    #             #     print(arg)
    #             Rig.RigAssetExport(arg)
    # else:
    #     arg = exp.ARigAssetExporter()
    #     arg.scene = filename
    #     arg.node  = cmds.ls(type='dxRig')[0]
    #     # if arg.Treat():
    #     #     print(arg)
    #     Rig.RigAssetExport(arg)

    # FINAL
    Rig.assetExport()


#-------------------------------------------------------------------------------
# Rig Shot
def rigShotExport(filename, rigNode):
    import DXUSD_MAYA.Exporters as exp
    import DXUSD_MAYA.Rig as Rig
    import DXUSD_MAYA.MUtils as mutl

    if not os.path.exists(filename):
        assert False, "Not exists test scene"

    cmds.file(filename, o=True, f=True)

    # # TEST
    # arg = exp.ARigShotExporter()
    # arg.scene = filename
    # arg.node  = rigNode
    # arg.frameRange = mutl.GetFrameRange()
    # arg.step = 1.0
    #
    # arg.overwrite = False
    # arg.isRigUpdate = True
    # Rig.RigShotGeomExport(arg)
    #
    # if not os.path.exists(arg.D.TASKNV):
    #     assert False, "Don't export geom data"
    #
    # arg.overwrite = True
    # Rig.RigShotCompositor(arg)

    # FINAL
    Rig.shotExport(node=rigNode, show='pipe', version='v001', process='both')


#-------------------------------------------------------------------------------
#
# DOIT
#
#-------------------------------------------------------------------------------
# filename = '/show/pipe/template/Sample_Shot/CLF_0050/asset/fox/rig/fox_rig_v003.mb'
# filename = '/show/pipe/template/Sample_Shot/CLF_0050/asset/fox/rig/fox_rig_v004.mb'
# filename = '/show/pipe/template/Sample_Model/upRobot/rig/upRobot_rig_v06.mb'
# rigAssetExport(filename)


filename = '/show/koz/works/ANI/OTS/OTS_0380/pub/scenes/OTS_0380_ani_v003.mb'
node = 'armorNero:armorNero_rig_GRP'
rigShotExport(filename, node)
