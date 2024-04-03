#encoding=utf-8
#!/usr/bin/env python
#-------------------------------------------------------------------------------
#
#   DEXTER STUDiOS
#
#   R&D TD          : taehyung.lee
#   CG Supervisor	: seonku.kim
#
#-------------------------------------------------------------------------------

import os, sys
from openexr_py import OpenEXR

from pymel.all import *
import argparse
import maya.cmds as cmds
import maya.mel as mel
import math
import maya.api.OpenMaya as OpenMaya
import maya.api.OpenMayaAnim as OpenMayaAnim
from datetime import timedelta, datetime, date, time

# import maya.standalone
# maya.standalone.initialize()

def getTime():
    now = datetime.now()
    current_time = now.strftime('%m/%d %H:%M:%S')
    result = '[EXR Header Insert] {0} >>'.format(current_time)
    return result

def getCameraSpeed(mbfile, start, end, step, nodename):
    cmds.file(mbfile, force=True, open=True)

    selection = OpenMaya.MSelectionList()
    selection.add(nodename)

    mobj = selection.getDependNode(0)
    mfn = OpenMaya.MFnDependencyNode(mobj)

    mtxAttr = mfn.attribute('worldMatrix')
    mtxPlug = OpenMaya.MPlug(mobj, mtxAttr)
    mtxPlug = mtxPlug.elementByLogicalIndex(0)

    returnDic = {}

    for i in range(int(start), int(end)+1):
        frameCtx = OpenMaya.MDGContext(OpenMaya.MTime(i, OpenMaya.MTime.uiUnit()))
        mtxObj = mtxPlug.asMObject(frameCtx)
        mtxData = OpenMaya.MFnMatrixData(mtxObj)
        mtxValue = mtxData.matrix()

        currX = mtxValue[12]
        currY = mtxValue[13]
        currZ = mtxValue[14]

        prevFrameCtx = OpenMaya.MDGContext(OpenMaya.MTime(i - 1, OpenMaya.MTime.uiUnit()))
        prevMtxObj = mtxPlug.asMObject(prevFrameCtx)
        prevMtxData = OpenMaya.MFnMatrixData(prevMtxObj)
        prevMtxValue = prevMtxData.matrix()

        prevX = prevMtxValue[12]
        prevY = prevMtxValue[13]
        prevZ = prevMtxValue[14]

        cmPerFrame = math.sqrt(pow(currX - prevX, 2) + pow(currY - prevY, 2) + pow(currZ - prevZ, 2))
        returnDic[i] = cmPerFrame

    return returnDic

def getCameraSpeed_backup(mbfile, start, end, step, nodename):
    cmds.file(mbfile, force=True, open=True)
    #nodename = 'xiqiaoshan_cam'
    nodename = cmds.listRelatives(nodename, p=True, f=True)[0] #get transform node
    returnDic = {}

    for i in range(int(start), int(end) + 1):
        prevX = cmds.getAttr('%s.tx' % nodename, time=i - 1)
        prevY = cmds.getAttr('%s.ty' % nodename, time=i - 1)
        prevZ = cmds.getAttr('%s.tz' % nodename, time=i - 1)

        currX = cmds.getAttr('%s.tx' % nodename, time=i)
        currY = cmds.getAttr('%s.ty' % nodename, time=i)
        currZ = cmds.getAttr('%s.tz' % nodename, time=i)
        melStr = 'abs(mag (<<%f,%f,%f>>)-mag(<<%f,%f,%f>>))' % (currX, currY, currZ, prevX, prevY, prevZ)

        #cmPerFrame = mel.eval(melStr)
        cmPerFrame = math.sqrt(pow(currX-prevX,2) + pow(currY-prevY,2) + pow(currZ-prevZ,2))
        returnDic[i] = cmPerFrame
        """
        cmPerSec = cmPerFrame * 24
        cmPerMin = cmPerSec * 60
        cmPerHour = cmPerMin * 60

        mPerHour = cmPerHour / 100.0
        kmPerHour = mPerHour / 1000.0

        print 'mPerHour', mPerHour

        print 'frame', i
        print 'cmPerFrame', cmPerFrame
        print 'cmPerSec', cmPerSec
        print 'cmPerMin', cmPerMin
        print 'cmPerHour', cmPerHour
        print 'mPerHour', mPerHour
        print 'kmPerHour', kmPerHour
        print ""
        """
    return returnDic


def insertExrHeader(frameDic, dirPath, imgName, start, end, pad):
    for i in sorted(frameDic.keys()):
        basename = os.path.join(dirPath, imgName)
        absFileName = '.'.join([basename, str(i).zfill(int(pad)), 'exr'])

        #ff = OpenEXR.InputFile('/home/taehyung.lee/Documents/test/rsync_test/metatest/ADB_0002_lgt_v01_w01_rs_Bg.1040.exr')
        try:
            exrObj = OpenEXR.InputFile(absFileName)
            exrHeader = exrObj.header()

            exrHeader['speed'] = frameDic[i]
            channels = exrHeader['channels'].keys()

            newChannels = dict(zip(channels, exrObj.channels(channels)))
            out = OpenEXR.OutputFile(absFileName, exrHeader)

            out.writePixels(newChannels)
        except Exception, e:
            print '%s %s' % (getTime(), str(e))
        print '%s %s, speed, %s' % (getTime(), absFileName, str(frameDic[i]))


def main(arg):
    speedDic = getCameraSpeed(arg.mb, arg.start, arg.end, arg.step, arg.cam)
    insertExrHeader(speedDic, arg.rd, arg.im, arg.start, arg.end, arg.pad)
    print '-'*100
    print '%s %s' % (getTime(), speedDic)
    print '-' * 100


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument(
        '-s', '--start', default='1',
        help='start frame'
    )
    parser.add_argument(
        '-e', '--end', default='1',
        help='end frame'
    )
    parser.add_argument(
        '-step', '--step', default='1',
        help='step'
    )
    parser.add_argument(
        '-pad', '--pad', default='4',
        help='padding'
    )
    parser.add_argument(
        '-cam', '--cam', default='',
        help='camera'
    )
    parser.add_argument(
        '-rd', '--rd', default='',
        help='Directory in which to store image file'
    )
    parser.add_argument(
        '-im', '--im', default='',
        help='Image file output name'
    )
    parser.add_argument(
        '-mb', '--mb', default='',
        help='maya file'
    )

    args = parser.parse_args()
    up = main(args)
    # maya.standalone.uninitialize()

    exit(0)
    # command example
    # ./insert_exr_header.sh
    # -s 1001 -e 1005
    # -cam xiqiaoshan_cam
    # -rd /home/taehyung.lee/Documents/link_test/rsync_test/metatest/test
    # -im test
    # -mb /home/taehyung.lee/FLYprv_150_v01_w03.mb
