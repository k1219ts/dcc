import os
import maya.cmds as cmds
import re
import sys

def getGuideElementsList():
    guideList = list()
    guideType = ["nurbsSurface", "mesh"]
    if cmds.ls("fxGuide"):
        for i in cmds.listRelatives("fxGuide", f=True, ad=True, type=guideType):
            meshNode = str(i).split("|")[-1]
            trNode = str(cmds.listRelatives(meshNode, f=True, p=True)[0])
            if not trNode in guideList:
                guideList.append(trNode)
            else:
                if trNode[-1].isdigit():
                    newNumb = int(trNode[-1]) + 1
                    newName = trNode.replace(trNode[-1], str(newNumb)).split("|")[-1]
                    cmds.rename(trNode, newName)
                    guideList.append(newName)
                else:
                    newName = (str(i) + "1").split("|")[-1]
                    cmds.rename(trNode, newName)
                    guideList.append(newName)
    if len(cmds.ls("*:*_rig_GRP", type="dxComponent")) != 0:
        for qw in cmds.ls("*:*_rig_GRP", type="dxComponent"):
            guideList.append(str(qw))
    return guideList

def timelineCheck():
    minT = cmds.playbackOptions(q=1, minTime=1)
    maxT = cmds.playbackOptions(q=1, maxTime=1)
    return minT, maxT

def exportAlc(fPath):
    # Export Cache
    cmds.file(fPath, force=True, open=True)
    dirName = str(os.path.splitext(os.path.basename(fPath))[0])
    exPath = os.sep.join(str(fPath).split(os.sep)[:-2]) + "/data/geoCache/"
    gdList = getGuideElementsList()
    minT, maxT = timelineCheck()
    if not os.path.exists(exPath):
        os.mkdir(exPath)
    exCacheFile = exPath + dirName + ".abc"
    bs = ""
    for i in gdList:
        bs += " -rt " + str(i)
    camCmd = "-fr %f %f -wuvs -ws -wv -ef -df ogawa %s -f %s" % (minT, maxT, bs, exCacheFile)
    cmds.AbcExport(v=1, j=camCmd)

if __name__ == '__main__':
    fPath = sys.argv[1:]
    print fPath
    from pymel.all import *
    for i in fPath:
        exportAlc(i)