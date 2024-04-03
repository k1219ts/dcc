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

try:
    cmds.loadPlugin('ZENNForMaya')
except:
    assert False, "not found ZENNForMaya plugin"

#-------------------------------------------------------------------------------
#
# ASSET
#
#-------------------------------------------------------------------------------
def groomAssetExport(filename):
    import DXUSD_MAYA.Exporters as exp
    import DXUSD_MAYA.Groom as Groom

    if not cmds.pluginInfo('ZENNForMaya', q=True, l=True):
        assert False, "Failed Load ZENNForMaya Plugin"

    if not os.path.exists(filename):
        assert False, "Not exists test scene"

    cmds.file(filename, o=True, f=True)

    rootNode = cmds.ls(type='dxBlock')[0]

    # # TEST
    # if cmds.nodeType(rootNode) != 'dxBlock':
    #     assert False, 'node type error!'
    #
    # if not cmds.objExists('ZN_ExportSet'):
    #     assert False, "not exists ZN_ExportSet"
    # zdeforms = cmds.sets('ZN_ExportSet', q=True)
    #
    # bodyMeshLayerData = grm.GetZennBodyMeshMap(zdeforms, root=rootNode)
    # if not bodyMeshLayerData:
    #     assert False, 'not found export ZN_Deform'
    #
    # export_grooms = []
    # for shape, znodes in bodyMeshLayerData.items():
    #     export_grooms += znodes
    #
    # arg = exp.AGroomAssetExporter()
    # arg.scene = sceneFileName
    # arg.node  = rootNode
    # arg.bodyMeshLayerData = bodyMeshLayerData
    # arg.groom_nodes = export_grooms
    # if arg.Treat():
    #     print(arg)
    # # Groom.GroomAssetExport(arg)

    # FINAL
    Groom.assetExport(node=rootNode)



#-------------------------------------------------------------------------------
#
# SHOT
#
#-------------------------------------------------------------------------------
def GroomShotCacheMergeExport(inputCache):
    import DXUSD_MAYA.Exporters as exp
    import DXUSD_MAYA.Groom as grm
    import DXUSD_MAYA.MUtils as mutl

    # CLF_0050 : fox
    sceneFileName = ''
    groomFile  = ''
    # inputCache = '/show/pipe/_3d/shot/CLF/CLF_0050/ani/fox/v010/fox_ani.usd'
    # inputCache = '/show/pipe/_3d/shot/CLF/CLF_0050/sim/fox/v007/fox.usd'
    # groomFile  = '/show/pipe/_3d/asset/fox/groom/scenes/fox_hair_v003.mb'

    # Requires load ZENNForMaya plugin
    try:
        cmds.loadPlugin('ZENNForMaya')
    except:
        assert False, "not found ZENNForMaya plugin"

    if not cmds.pluginInfo('ZENNForMaya', q=True, l=True):
        assert False, "Failed Load ZENNForMaya Plugin"

    try:
        cmds.loadPlugin('DXUSD_Maya')
    except:
        assert False, "not found DXUSD_Maya plugin"

    if not cmds.pluginInfo('DXUSD_Maya', q=True, l=True):
        assert False, "Failed Load DXUSD_Maya Plugin"

    # # TEST
    # arg = exp.AGroomShotExporter()
    # arg.inputcache = inputCache
    # arg.scene= sceneFileName
    # arg.step = 1.0
    # # arg.overwrite = True
    # # arg.ovr_ver = 'v101'
    #
    # if arg.Treat():
    #     print('>>> GroomShotGeomExport')
    #     print(arg)
    # grm.GroomShotGeomExport(arg)
    #
    # arg.overwrite = True
    # if arg.Treat():
    #     print('>>> GroomShotCompositor')
    #     print(arg)
    # grm.GroomShotCompositor(arg)

    # FINAL
    grm.shotCacheMergeExport(inputCache, fr=[1001, 1005], process='both')


#-------------------------------------------------------------------------------
#
# DOIT
#
#-------------------------------------------------------------------------------
# filename = '/show/pipe/template/Sample_Shot/CLF_0050/asset/fox/groom/fox_hair_v004.mb'
# # filename = '/show/pipe/works/CSP/yeojin/project/maya/scenes/e9dog_groom_v005.mb'
# groomAssetExport(filename)

inputCache = '/show/pipe/_3d/shot/CLF/CLF_0050/ani/fox1/v001/fox1_ani.usd'
GroomShotCacheMergeExport(inputCache)
