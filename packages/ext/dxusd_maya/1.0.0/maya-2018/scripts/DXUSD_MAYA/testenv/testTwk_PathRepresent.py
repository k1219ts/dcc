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

# import DXUSD.Vars as var
# import DXUSD.Utils as utl
import DXUSD.Message as msg

import DXUSD_MAYA.Sim as Sim
import DXUSD_MAYA.MUtils as mutl

import DXUSD_MAYA.Tweakers as mtwk

#-------------------------------------------------------------------------------
# Sim PathRepresent
try:
    cmds.loadPlugin('DXUSD_Maya')
except:
    assert False, "not found DXUSD_Maya plugin"

if not cmds.pluginInfo('DXUSD_Maya', q=True, l=True):
    assert False, "Failed Load DXUSD_Maya Plugin"

sceneFileName = '/show/pipe/works/CSP/daeseok.chae/maya/scenes/S52_3040_cloth_v001.mb'
if not os.path.exists(sceneFileName):
    assert False, "Not exists test scene"

cmds.file(sceneFileName, o=True, f=True)

arg = mtwk.APathRepresent()
arg.scene = sceneFileName
arg.node  = 'upRobot'
arg.refNode  = 'upRobot'
arg.frameRange = mutl.GetFrameRange()
# arg.node = 'upRobot1'
arg.geomfiles = [
    '/show/slc/_3d/shot/TEST/TEST_0040/sim/e9dog1/v004/e9dog.high_geom.usd'
]
if arg.Treat():
    print(arg)
pathRep = mtwk.PathRepresent(arg)
pathRep.DoIt()
# GAA = mtwk.GeomAgentAttrs(arg)
# GAA.DoIt()
