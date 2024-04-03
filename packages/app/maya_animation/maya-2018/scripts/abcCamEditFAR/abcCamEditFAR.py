import os
import sys
import maya.cmds as cmds
from pymel.all import *

filmAspectRatio =  1.896
FitResolutionGate = "overscan"

abcCam = sys.argv[1]
prefix = sys.argv[2]
abcCamSplit = os.path.splitext(abcCam)
newAbcCam = abcCamSplit[0] + prefix + abcCamSplit[1]

cmds.AbcImport(abcCam, mode="import", fitTimeRange=True)

sFrame = cmds.playbackOptions(q=1, min=1)
eFrame = cmds.playbackOptions(q=1, max=1)

jobString =  "-frameRange {startFrame} {endFrame}".format(startFrame=sFrame, endFrame=eFrame)
jobString += " -dataFormat ogawa -root {camname} -file {newAbcCam}"

camList = cmds.listCameras(p=1)
for cam in camList:
    fitFilm = cmds.camera(cam, q=1, ff=True)
    if fitFilm == FitResolutionGate:
        if not cmds.camera(cam, q=1, ar=1) == filmAspectRatio:
            cmds.camera(cam, e=1, ar=filmAspectRatio)
            jobString = jobString.format(camname=cam, newAbcCam=newAbcCam)
            cmds.AbcExport( j=jobString)
        else:
            print cam, " no problem"
    else:
        print cam, " no problem"
