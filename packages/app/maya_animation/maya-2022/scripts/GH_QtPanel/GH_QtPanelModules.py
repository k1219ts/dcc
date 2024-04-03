# -*- coding:utf-8 -*-
__author__ = 'gyeongheon.jeong'

import maya.cmds as cmds
import maya.mel as mm
import os
import string
import time
import shutil
import subprocess

def getEnvironment():
    _env = os.environ.copy()
    libraries = list()
    for i in _env['LD_LIBRARY_PATH'].split(':'):
        if i.find('autodesk') == -1:
            libraries.append( i )
    _env['LD_LIBRARY_PATH'] = string.join( libraries, ':' )
    return _env

def updateData():
    cmds.setAttr("defaultRenderGlobals.imageFormat", 7) # IFF

    aPlayBackSliderPython = mm.eval('$tmpVar=$gPlayBackSlider')
    timeRange = cmds.timeControl(aPlayBackSliderPython, q = 1, rng = 1)
    timeRange_ = timeRange.split(":")

    _StartTime = float( timeRange_[0].strip("\"") )
    _endTime = float( timeRange_[1].strip("\"") )

    if int(_endTime - _StartTime) == 1:
        _in = cmds.playbackOptions(q=1, min=True)
        _out = cmds.playbackOptions(q=1, max=True)
    else:
        _in = float( timeRange_[0].strip("\"") )
        _out = float( timeRange_[1].strip("\"") ) - 1.0

    return _in, _out

def playBlast(offScreenValue, w_v, h_v):
    _in, _out = updateData()

    cmds.playblast(p=100, startTime = _in, endTime = _out, format="iff", quality=70, offScreen = offScreenValue, wh=( w_v, h_v ), cc=True)

def runFFmpeg(fileName, ROOT, w_v, h_v, offScreenValue):
    _in, _out = updateData()

    codecName = "H.264 LT"

    t = int(time.mktime(time.gmtime()))

    prefix = os.path.splitext(os.path.basename(fileName))[0]
    blastTempDir = os.path.join( os.path.dirname(fileName), prefix+'_'+str(t) )

    if not os.path.isdir(blastTempDir):
        os.makedirs(blastTempDir)

    cmds.playblast(filename=os.path.join(blastTempDir, prefix),
                   startTime=_in, endTime=_out, forceOverwrite=True,
                   format="image", compression='jpg',
                   viewer=False, showOrnaments=True, offScreen=offScreenValue,
                   framePadding=4, percent=100, widthHeight=(w_v, h_v))
    cmd = '%s/script/imgToMov.sh %d %d %d "%s" %s %s %s "movie"' % (ROOT, _in, _out, 24, codecName, blastTempDir, fileName, prefix)
    p = subprocess.Popen(cmd, env=getEnvironment(), shell=True)
    p.wait()

#    if self.checkBox_remove_sequence.isChecked():
#      shutil.rmtree(blastTempDir)

    shutil.rmtree(blastTempDir)

def showTypes( panelName, type = str(), val = bool()):
    if type == "gpucache":
        cmds.modelEditor(panelName, e = 1, pluginObjects = ( 'gpuCacheDisplayFilter', val))
    else:
        eval("cmds.modelEditor(panelName, e = 1, %s = %d )" % (type, val))