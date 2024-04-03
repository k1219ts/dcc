# encoding:utf-8
# !/usr/bin/env python

import os, sys
import json
import maya.cmds as cmds
import maya.mel as mel
import McdActionFunctionssw
import McdSaveActionF

def expAct(path, fileName, oaPath):
    cmds.file(path, force=True, open=True)
    animPlug = '/usr/autodesk/maya2017/bin/plug-ins/animImportExport.so'
    if cmds.pluginInfo(animPlug, q=True, l=True) == False:
        cmds.loadPlugin(animPlug)
    name = str(path.split(os.sep)[-1].split(".")[0])
    minT = cmds.playbackOptions(q=True, min=True)
    maxT = cmds.playbackOptions(q=True, max=True)
    loadJ = open("/dexter/Cache_DATA/CRD/Asset/JNT_Info/" + fileName + ".json").read()
    data = [str(e) for e in json.loads(loadJ)]
    if cmds.ls(type="dxRig"):
        if str(cmds.ls(type="dxRig")[0]).count(":") == 1:
            nsChar = str(cmds.ls(type="dxRig")[0]).split(":")[0]
            jntList = [nsChar + ":" + str(n) for n in data]
        else:
            nsChar = 0
    else:
        cmds.error("dxRig")
        return
    if nsChar:
        cmds.select(jntList)
        cmds.bakeResults(jntList, sm=True, t=(minT, maxT), sb=1.0)
    else:
        cmds.select(data)
        cmds.bakeResults(data, sm=True, t=(minT, maxT), sb=1.0)
    npath = os.sep.join(path.split(os.sep)[:-2]) + "/Anim/"
    if not os.path.exists(npath):
        os.mkdir(npath)
    animName = npath + name + ".anim"
    cmds.file(animName, pr=1, typ="animExport", force=1, options="precision=8;intValue=17;nodeNames=1;verboseUnits=0;whichRange=1;range=0:10;options=keys;hierarchy=none;controlPoints=0;shapes=1;helpPictures=0;useChannelBox=0;copyKeyCmd=-animation objects -option keys -hierarchy none -controlPoints 0 -shape 1 ", es=1)

    cmds.file(oaPath, force=True, open=True)
    cmds.select(data)
    cmds.currentTime(1)
    cmds.file(animName, i=True)
    cmds.playbackOptions(minTime=1, maxTime=maxT)
    apath = os.sep.join(path.split(os.sep)[:-2]) + "/Action/"
    if not os.path.exists(apath):
        os.mkdir(apath)
    McdActionFunctionssw.crtAct(name)
    fcdAction = str(cmds.ls(sl=True)[0])
    cmds.setAttr(fcdAction + ".matchName", True)
    cmds.setAttr(fcdAction + ".cycleFilter", 0.1)
    cmds.setAttr(fcdAction + ".txState", 1)
    cmds.setAttr(fcdAction + ".tyState", 0)
    cmds.setAttr(fcdAction + ".tzState", 1)
    cmds.setAttr(fcdAction + ".rxState", 0)
    cmds.setAttr(fcdAction + ".ryState", 0)
    cmds.setAttr(fcdAction + ".rzState", 0)
    mel.eval("McdSetAgentDataCmd;")
    actname = apath + name + ".ma"
    McdSaveActionF.McdSaveAction(actname)

if __name__ == '__main__':
    from pymel.all import *
    #import maya.standalone as mst
    #mst.initialize(name="python")
    path = sys.argv[1]
    fileName = sys.argv[2]
    oaPath = sys.argv[3]
    expAct(path, fileName, oaPath)
    os._exit(0)