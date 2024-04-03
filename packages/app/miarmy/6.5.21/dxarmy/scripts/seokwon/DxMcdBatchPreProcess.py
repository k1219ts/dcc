# encoding=utf-8
# !/usr/bin/env python

import maya.cmds as cmds
import maya.mel as mel
from McdGeneral import *
import os

def DxMcdBatchPreProcess():
    # place agent out
    cmd = "McdPlacementCmd -am 3 -ign 0;"
    mel.eval(cmd)
    McdAfterPlaceFunction()

    # turn off mesh drive and agent cache
    allGlb = cmds.ls(type="McdGlobal")
    if McdIsBlank(allGlb):
        raise Exception("No found McdGlobal Node.")
        return
    for i in range(len(allGlb)):
        cmds.setAttr(allGlb[i] + ".enableMeshDrv", 0)
        cmds.setAttr(allGlb[i] + ".selectionCallback", 1)

        #    # make agent cache
        #    McdRenderBegin(1, 1)
