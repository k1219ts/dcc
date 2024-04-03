import maya.cmds as cmds
import subprocess
import os

mayalocation = "/usr/autodesk/maya2016.5/bin/mayapy"
scriptlocation = "/netapp/backstage/pub/apps/maya/2016.5/team/animation/linux/scripts/abcCamEditFAR/abcCamEditFAR.py"
#scriptlocation = "/home/gyeongheon.jeong/maya_dev/2016.5/team/animation/linux/scripts/abcCamEditFAR/abcCamEditFAR.py"
prefix = "_nuke"

def doIt():
    confirm = cmds.fileDialog2(cap="Select Camera .Abc file", fm=1, okc="select", ff="*.abc")
    if confirm:
        for i in confirm:
            abcCam = i
            p = subprocess.Popen([mayalocation, scriptlocation, abcCam, prefix])
            p.wait()