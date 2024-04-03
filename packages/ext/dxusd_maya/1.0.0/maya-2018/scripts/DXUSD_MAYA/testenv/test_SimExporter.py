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
# Simulation Shot
def simShotExport():
    import DXUSD_MAYA.Exporters as exp
    import DXUSD_MAYA.Sim as Sim
    import DXUSD_MAYA.MUtils as mutl

    # Requires load miarmy plugin
    try:
        cmds.loadPlugin('DXUSD_Maya')
    except:
        assert False, "not found DXUSD_Maya plugin"

    if not cmds.pluginInfo('DXUSD_Maya', q=True, l=True):
        assert False, "Failed Load DXUSD_Maya Plugin"

    sceneFileName = '/show/pipe/template/Shot_Sample/CLF_0050/sim/CLF_0050_cloth_v002_sample.mb'

    if not os.path.exists(sceneFileName):
        assert False, "Not exists test scene"

    cmds.file(sceneFileName, o=True, f=True)

    # # TEST
    # arg = exp.ASimExporter()
    # arg.scene = sceneFileName
    # arg.node  = 'fox_rig_GRP'
    # arg.frameRange = mutl.GetFrameRange()
    # # arg.autofr = True
    # arg.step = 1.0
    #
    # # Auto Versioning
    # arg.overwrite = False
    # Sim.SimGeomExport(arg)
    #
    # if not os.path.exists(arg.D.TASKNV):
    #     assert False, "Don't export geom data"
    #
    # arg.overwrite = True
    # Sim.SimCompositor(arg)

    # FINAL
    Sim.shotExport(node='fox_rig_GRP', process='both')


simShotExport()
