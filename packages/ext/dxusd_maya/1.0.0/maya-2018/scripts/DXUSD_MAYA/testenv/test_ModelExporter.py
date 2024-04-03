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

#-------------------------------------------------------------------------------
def modelAssetExport(filename):
    import DXUSD_MAYA.Exporters as exp
    import DXUSD_MAYA.Model as Model

    if not os.path.exists(filename):
        assert False, 'Not exists scene -> %s' % filename

    cmds.file(filename, o=True, f=True)

    nodes = cmds.ls(['*_model_GRP', '*_model_*_GRP'])
    # nodes = ['elevatorPath_pathDown_model_GRP']

    # # TEST
    # arg = exp.AModelExporter(show='pipe')
    # arg.scene = filename
    # arg.nodes = nodes
    # # arg.ovr_ver = 'v001'
    # arg.ovr_shot = 'CLF_0040'
    # # if arg.Treat():
    # #     print(arg)
    # Model.ModelExport(arg)

    # FINAL
    Model.assetExport(nodes=nodes, show='je', version='v001')


def modelShotAssetExport(filename, shotName):
    import DXUSD_MAYA.Exporters as exp
    import DXUSD_MAYA.Model as Model

    if not os.path.exists(filename):
        assert False, 'Not exists scene -> %s' % filename

    cmds.file(filename, o=True, f=True)

    nodes = cmds.ls(['*_model_GRP', '*_model_*_GRP'])

    # FINAL
    Model.assetExport(nodes=nodes, show='pipe', shot=shotName)


#-------------------------------------------------------------------------------
#
# DOIT
#
#-------------------------------------------------------------------------------
# filename = '/show/pipe/template/Sample_Model/asdalCityTown/houseE/asdalCityTown_houseE_model_v001.mb'
filename = '/show/je/works/AST/sinkedCity/scenes/sinkedCity_model_v001_w008_PUB.mb'
# filename = '/show/pipe/template/Sample_Model/asdalCityTown/layout/asdalCityTown_model_v002_primPath.mb'
# filename = '/show/pipe/template/Sample_Model/asdalCityTown/layout/asdalCityTown_model_v004_totalZip.mb'
# filename = '/show/cdh1/works/AST/elevatorPath/scenes/elevatorPath_v01_w06.mb'
modelAssetExport(filename)

# filename = '/show/cdh1/works/AST/guardA/scenes/guardA_model_v10_w02.mb'
# modelShotAssetExport(filename, 'ELV_0010')
