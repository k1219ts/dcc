#coding:utf-8
from __future__ import print_function

from pymel.all import *
import maya.cmds as cmds
import os

try:
    cmds.loadPlugin('pxrUsd')
except:
    assert False, "not found pxrUsd plugin"

if not cmds.pluginInfo('pxrUsd', q=True, l=True):
    assert False, "Failed Load PxrUsd Plugin"

try:
    cmds.loadPlugin('DXUSD_Maya')
except:
    assert False, "not found DXUSD_Maya"

# customdir = '/WORK_DATA/temp/MTK_TEST'
customdir = '/show/pipe/works/CSP/sanghun/temp/MTK_TEST'

#-------------------------------------------------------------------------------
def modelAssetExport(filename):
    import DXUSD_MAYA.Exporters as exp
    import DXUSD_MAYA.Model as Model

    if not os.path.exists(filename):
        assert False, 'Not exists scene -> %s' % filename

    cmds.file(filename, o=True, f=True)

    # nodes = cmds.ls(['*_model_GRP', '*_model_*_GRP'])
    nodes = cmds.ls(['*_model_GRP'])

    # TEST
    arg = exp.AModelExporter()
    arg.scene = filename
    arg.nodes = nodes
    arg.customdir = customdir
    # if arg.Treat():
    #     print(arg)
    Model.ModelExport(arg)

    # # FINAL
    # Model.mtkExport(nodes=nodes, customdir=customdir)


#-------------------------------------------------------------------------------
def groomAssetExport(filename):
    import DXUSD_MAYA.Exporters as exp
    import DXUSD_MAYA.Groom as Groom

    try:
        cmds.loadPlugin('ZENNForMaya')
    except:
        assert False, "not found ZENNForMaya plugin"

    if not cmds.pluginInfo('ZENNForMaya', q=True, l=True):
        assert False, 'Failed Load ZENNForMaya Plugin'

    if not os.path.exists(filename):
        assert False, 'Not exists test scene'

    cmds.file(filename, o=True, f=True)
    rootNode = cmds.ls(type='dxBlock')[0]

    # # TEST
    # if not cmds.objExists('ZN_ExportSet'):
    #     assert False, 'Not exists ZN_ExportSet'
    # zdeforms = cmds.sets('ZN_ExportSet', q=True)
    #
    # bodyMeshLayerData = Groom.GetZennBodyMeshMap(zdeforms, root=rootNode)
    # if not bodyMeshLayerData:
    #     assert False, 'Not found export ZN_Deform'
    #
    # export_grooms = []
    # for shape, znodes in bodyMeshLayerData.items():
    #     export_grooms += znodes
    #
    # arg = exp.AGroomAssetExporter()
    # arg.scene = filename
    # arg.node  = rootNode
    # arg.bodyMeshLayerData = bodyMeshLayerData
    # arg.groom_nodes = export_grooms
    # arg.customdir = customdir
    # if arg.Treat():
    #     print(arg)

    # FINAL
    Groom.mtkExport(node=rootNode, customdir=customdir)


#-------------------------------------------------------------------------------
def rigClipExport(filename):
    import DXUSD_MAYA.Exporters as exp
    import DXUSD_MAYA.Clip as Clip

    if not os.path.exists(filename):
        assert False, 'Not exists test scene'

    cmds.file(filename, o=True, f=True)

    rootNode = cmds.ls(type='dxRig')[0]

    # # TEST
    # arg = exp.ARigClipExporter()
    # arg.scene = filename
    # arg.node  = rootNode
    # arg.customdir = customdir
    # if arg.Treat():
    #     print(arg)

    # FINAL
    Clip.mtkRigExport(node=rootNode, customdir=customdir)


#-------------------------------------------------------------------------------
#
#   DOIT
#
#-------------------------------------------------------------------------------
filename = '/show/pipe/template/Sample_Model/asdalCityTown/houseA/asdalCityTown_houseA_model_v001.mb'
# filename = '/show/pipe/template/asdalCityTown/layout/asdalCityTown_model_v003_layout.mb'
modelAssetExport(filename)

# filename = '/show/pipe/template/Shot_Sample/CLF_0050/asset/fox/groom/fox_hair_v004.mb'
# groomAssetExport(filename)

# filename = '/show/pipe/template/Shot_Sample/CLF_0050/clip/fox_clip_v001_run.mb'
# rigClipExport(filename)
