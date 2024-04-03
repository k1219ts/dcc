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

#-------------------------------------------------------------------------------
# RIG
def rigClipExport():
    import DXUSD_MAYA.Exporters as exp
    import DXUSD_MAYA.Clip as Clip

    # fox
    # sceneFileName = '/show/pipe/template/fox/clip_sample/fox_clip_v001_walk.mb'
    # rootNode = 'walk:fox_rig_GRP'
    sceneFileName = '/show/pipe/template/Shot_Sample/CLF_0050/clip/fox_clip_v001_run.mb'
    rootNode = 'run:fox_rig_GRP'

    # Requires load backstageMenu plugin
    pluginName = 'DXUSD_Maya'
    try:
        cmds.loadPlugin(pluginName)
    except:
        assert False, "not found %s plugin" % pluginName

    if not cmds.pluginInfo(pluginName, q=True, l=True):
        assert False, "Failed Load %s Plugin" % pluginName

    if not os.path.exists(sceneFileName):
        assert False, "Not exists test scene"

    cmds.file(sceneFileName, o=True, f=True)

    # # TEST
    # arg = exp.ARigClipExporter()
    # arg.scene = sceneFileName
    # arg.node  = rootNode
    # # arg.timeScales= [0.8, 1.0, 1.5]
    # # arg.loopRange = (1001, 5000)
    # # arg.Treat()
    # # print(arg)
    # Clip.RigClipExport(arg)

    # FINAL
    Clip.rigExport(node=rootNode, step=0.5)


#-------------------------------------------------------------------------------
# GROOM
def groomClipExport():
    import DXUSD_MAYA.Exporters as exp
    import DXUSD_MAYA.Clip as Clip
    import DXUSD_MAYA.Groom as Groom
    import DXUSD_MAYA.MUtils as mutl
    import dxBlockUtils
    import DXUSD_MAYA.Tweakers as mtwk

    # fox
    sceneFileName = '/show/pipe/_3d/asset/fox/groom/scenes/fox_hair_v004.mb'
    rootNode   = 'fox_ZN_GRP'
    # inputCache = '/show/pipe/_3d/asset/fox/clip/walk/v001/base/fox_rig.usd'
    # inputCache = '/show/pipe/_3d/asset/fox/clip/run/v003/base/fox_rig.usd'
    # for MTK
    inputCache = '/show/pipe/works/CSP/sanghun/temp/MTK_TEST/asset/fox/clip/run/v000/base/fox_rig.usd'

    # Requires load plugin
    try:
        cmds.loadPlugin('ZENNForMaya')
    except:
        assert False, 'not found ZENNForMaya plugin'
    if not cmds.pluginInfo('ZENNForMaya', q=True, l=True):
        assert False, 'Failed Load ZENNForMaya Plugin'

    if not os.path.exists(sceneFileName):
        assert False, "Not exists test scene"

    cmds.file(sceneFileName, o=True, f=True)

    # initialize ZN_Import
    zdeforms = cmds.sets('ZN_ExportSet', q=True)
    Groom.ZN_Initialize(zdeforms)

    # Merge Cache
    dxBlockUtils.CacheMerge.UsdMerge(inputCache, [rootNode]).doIt()

    # Walking pre-frame
    frameRange = mutl.GetFrameRange()
    startTime  = int(cmds.currentTime(q=True))
    for i in range(startTime, frameRange[0]):
        cmds.currentTime(i)

    # # TEST
    # bodyMeshLayerData = Groom.GetZennBodyMeshMap(zdeforms, root=rootNode)
    # if not bodyMeshLayerData:
    #     assert False, 'not found export ZN_Deform'
    #
    # export_grooms = []
    # for shape, znodes in bodyMeshLayerData.items():
    #     export_grooms += znodes
    #
    # arg = exp.AGroomClipExporter()
    # arg.scene = sceneFileName
    # arg.node  = rootNode
    # arg.inputcache = inputCache
    # arg.bodyMeshLayerData = bodyMeshLayerData
    # arg.groom_nodes = export_grooms
    # # arg.Treat()
    # # print(arg)
    # Clip.GroomClipExport(arg)

    # FINAL
    Clip.groomExport(node=rootNode)


# DOIT
# rigClipExport()
groomClipExport()
