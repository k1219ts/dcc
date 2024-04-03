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

import DXUSD_MAYA.Exporters as exp
import maya.cmds as cmds
import DXUSD_MAYA.Layout as Layout
import DXUSD_MAYA.Model as Model
import DXUSD_MAYA.MUtils as mutl

# Requires load DXUSD_Maya plugin
try:
    cmds.loadPlugin('DXUSD_Maya')
except:
    assert False, "not found DXUSD_Maya plugin"

if not cmds.pluginInfo('pxrUsd', q=True, l=True):
    assert False, "Failed Load PxrUsd Plugin"

try:
    cmds.loadPlugin('DXUSD_Maya')
except:
    assert False, "not found DXUSD_Maya"

#-------------------------------------------------------------------------------
def assetExport():
    path ='/show/pipe/template/Layout/city_model_v004.mb'

    if not os.path.exists(path):
        assert False, "Not exists test scene"
    # current scene filename
    # sceneFile = cmds.file(q=True, sn=True)

    sceneFile = cmds.file(path, o=True, f=True)
    overwrite = True
    nodes = ['city_factoryD_model_GRP']
    arg = exp.AModelExporter()
    arg.scene = sceneFile
    arg.nodes = nodes
    arg.overwrite = overwrite
    Model.ModelExport(arg)

def layoutShotExport(filename, nodes=None):
    print('--------------------------')
    print('    Shot Set -- asset')
    print('--------------------------')

    if not os.path.exists(filename):
        assert False, "Not exists test scene"

    sceneFile = cmds.file(filename, o=True, f=True)

    # nodes = ['houseA3','houseA4']
    # nodes = ['PointInstancer_layout_withClip']
    # nodes = ['ModelExport_layout_house', 'ModelExport_layout_simModel', 'ModelExport_layout_tree']
    # nodes = ['trees', 'trees1' ,'trees2']
    # # TEST
    # # overwrite =True
    # # fr=[1001,1010]
    # step = 1.0
    # arg = exp.ALayoutExporter()
    # arg.scene = sceneFile
    # # arg.overwrite = overwrite
    # arg.nodes = nodes
    # arg.frameRange = mutl.GetFrameRange()
    # # arg.frameRange = fr
    # arg.frameSample = mutl.GetFrameSample(step)
    # # if arg.Treat():
    # #     print(arg)
    # Layout.LayoutGeomExport(arg)

    # FINAL
    Layout.shotExport(nodes=nodes, process='both')


# assetExport()

filename = '/show/pipe/template/Layout/PKL_0350_ani_v004_test.mb'
nodes = ['buildA_carSet', 'buildA_parkingLot']
# nodes = ['carSet1', 'parkingLot1']
# nodes = ['carSet']
layoutShotExport(filename, nodes)
