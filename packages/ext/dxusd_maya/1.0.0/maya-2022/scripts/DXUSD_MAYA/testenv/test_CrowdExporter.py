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
#
# if not cmds.pluginInfo('pxrUsd', q=True, l=True):
#     assert False, "Failed Load PxrUsd Plugin"

# Requires load miarmy plugin
# try:
#     cmds.loadPlugin('MiarmyProForMaya2018')
# except:
#     assert False, "not found MiarmyProForMaya2018 plugin"
#
# if not cmds.pluginInfo('MiarmyProForMaya2018', q=True, l=True):
#     assert False, "Failed Load MiarmyProForMaya2018 Plugin"

# Requires load golaem plugin
# try:
#     cmds.loadPlugin('glmCrowd')
# except:
#     assert False, "not found GolaemCrowd plugin"

#-------------------------------------------------------------------------------
# miarmy asset
def MiarmyAgentSceneExport():
    import DXUSD_MAYA.Exporters as exp
    import DXUSD_MAYA.Crowd as crd

    sceneFile = '/show/pipe/template/Miarmy/OA/crdGhost/crdGhost_ref.ma'
    rootNode  = 'OriginalAgent_crdGhost'

    if not os.path.exists(sceneFile):
        assert False, "Not exists test scene"

    cmds.file(sceneFile, o=True, f=True)

    # TEST
    arg = exp.AMiarmyAssetExporter()
    arg.scene = sceneFile
    arg.node  = rootNode
    # if arg.Treat():
    #     print(arg)
    crd.MiarmyAssetExport(arg)


#-------------------------------------------------------------------------------
# miarmy shot
def MiarmyShotSceneExport():
    import DXUSD_MAYA.Exporters as exp
    import DXUSD_MAYA.Crowd as crd

    sceneFile = '/show/pipe/template/Miarmy/LIM_0589/LIM_0589_crowd_v002_sample.mb'

    if not os.path.exists(sceneFile):
        assert False, 'Not exists test scene'

    cmds.file(sceneFile, o=True, f=True)

    # TEST
    # arg = exp.AMiarmyShotExporter()
    # arg.scene = sceneFile
    # if arg.Treat():
    #     print(arg)

    # FINAL
    crd.shotExport_miarmy(version='v020', fr=[1001, 1010], process='both')


#-------------------------------------------------------------------------------
# golaem asset export
def GolaemAssetExport():
    import DXUSD_MAYA.Crowd as crd

    sceneFile = '/show/pipe/works/CRD/asset/asset/characters/cow.mb'
    if not os.path.exists(sceneFile):
        assert False, 'Not exists test scene'

    cmds.file(sceneFile, o=True, f=True)
    crd.assetExport_golaem()

#-------------------------------------------------------------------------------
# golaem shot export
def GolaemShotExport():
    # from pxr import Sdf

    sceneFile = '/show/ncx/works/CRD/shot/CTC/CTC_0030/pub/scenes/CTC_0030_crowd_v004.mb'
    # sceneFile = '/show/pipe/works/CRD/shot/PS44B/PS44B_0600/pub/scenes/PS44B_0600_crowd_v001.mb'
    if not os.path.exists(sceneFile):
        assert False, 'Not exists test scene'

    # tmpfile = sceneFile.replace('.mb', '.usd')
    # layer = Sdf.Layer.FindOrOpen(tmpfile)
    # if not layer:
    #     layer = Sdf.Layer.CreateNew(tmpfile, args={'format': 'usdc'})

    # try:
    #     cmds.loadPlugin('glmCrowd')
    # except:
    #     assert False, "not found GolaemCrowd plugin"
    import DXUSD_MAYA.Crowd as crd
    cmds.file(sceneFile, o=True, f=True)
    crd.shotExport_golaem(version='v004', process='both')

# DoIt
GolaemShotExport()
