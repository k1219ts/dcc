# DD_maya.script.name:	shotDB_common.py
# DD_maya.script.version:	v1.2
# DD_maya.script.comment:	common publish library 
# written by Daehwan Jang(daehwanj@gmail.com)
# changed by Dongho Cha(izerax.ch@gmail.com) 
# DD_maya.script.history 
#
#	1.0		daehwan.jang	first release 
#	1.5		dongho.cha 	changed and correction renewImagePlane(), convertPolyImagePlane()
# 	1.6		dongho.cha		correction renewImagePlane() for depth animation

import os
import glob
import math

from PySide2 import QtWidgets
import shiboken2

import maya.cmds as mc
import maya.mel as mm
import maya.OpenMayaUI as omu


def get_maya_window():
    ptr = omu.MQtUtil.mainWindow()
    return shiboken2.wrapInstance(long(ptr), QtWidgets.QWidget)


def get_dir_list(dir):
    result = []
    all_list = []
    try:
        all_list = os.listdir(dir)
    except:
        result.append("No Dirs")
        return result

    if all_list:
        for i in all_list:
            if os.path.isdir(os.path.join(dir, i)) and i[0] != ".":
                result.append(i)

    result.sort()
    return result


def unlockAttr(plugName):
    lockedPlug = mc.connectionInfo(plugName, getLockedAncestor=True)
    if lockedPlug != "":
        mc.setAttr(lockedPlug, lock=False)


def reorder_list(list, inx):
    try:
        id = list.index(inx)
        value = list[id]
        list.pop(id)
        list.insert(0, value)
    except:
        pass
    # print "reorder_list(): There is no %s in list.\n"%inx

    return list


def make_dir(dir):
    if not os.path.isdir(dir):
        try:
            os.makedirs(dir)
            # print "make_dir done."
            return True
        except:
            import traceback
            var = traceback.format_exc()
            print var
            return False
    else:
        # print "dir exists."
        return True


def get_file_list(path, filename="", ext="*.*"):
    all_list = []
    result = []

    try:
        # os.listdir(path)
        all_list = glob.glob(os.path.join(path, ext))

        if all_list:
            for i in all_list:
                result.append(os.path.basename(i))
            result = [x for x in result if filename in x]
            if not result:
                result = ["No File1"]
        else:
            result = ["No File2"]
    except:
        result = ["No Dir"]
        return result

    result.sort()
    return result


def get_final_ver(path="", filename="", ext=""):
    file_list = get_file_list(path, filename, ext)
    file_list.sort()

    if file_list[0].startswith("No"):
        return 0
    else:
        final_file = os.path.splitext(file_list[-1])  # result: ('ELX_0100_main_matchmove_v11', '.mb')
        final_file = final_file[0]  # 'ELX_0100_main_matchmove_v11'
        final_ver = final_file.split('_')[-1]  # result: v11
        final_ver = int(final_ver[1:])  # result: 11
        return final_ver


def checkImagePlanePath():
    ipList = mc.ls(type="imagePlane")  # result: ['perspShape2->imagePlaneShape1', 'imagePlaneShape2', ...]
    for i in ipList:
        ip_path = mc.getAttr(i + ".imageName")
        if "/matchmove/pub/image" in ip_path:
            pass
        else:
            print "%s is in matchmove publish path." % i
            return False
    return True


def renewImagePlane():
    ipList = mc.ls(type="imagePlane")  # result: ['perspShape2->imagePlaneShape1', 'imagePlaneShape2', ...]

    ipAttrs = ['displayMode', 'type', 'textureFilter', 'imageName',
               'offsetX', 'offsetY', 'useFrameExtension', 'frameOffset', 'frameCache',
               'fit', 'displayOnlyIfCurrent', 'depth', 'frameExtension',
               'coverageX', 'coverageY', 'coverageOriginX', 'coverageOriginY',
               'imageCenterX', 'imageCenterY', 'imageCenterZ',
               'width', 'height', 'maintainRatio']

    renewIpList = []
    for i in ipList:
        mc.select(i)
        mm.eval('autoUpdateAttrEd;')
        attrResult = dict()
        for a in ipAttrs:
            gv = mc.getAttr('%s.%s' % (i, a))
            gt = mc.getAttr('%s.%s' % (i, a), type=True)
            attrResult[a] = {'value': gv, 'type': gt}

        camShape = mc.listConnections(i, type='camera', shapes=True, source=False)[0]

        newtr, newshape = mc.imagePlane(camera=camShape)
        for a in attrResult:
            obj = attrResult[a]
            if obj['type'] == 'string':
                mc.setAttr('%s.%s' % (newshape, a), obj['value'], type='string')
            else:
                mc.setAttr('%s.%s' % (newshape, a), obj['value'])
        if attrResult['useFrameExtension']['value']:
            mc.expression(n='frame_ext_expression', s='%s.frameExtension=frame;' % newshape)

        if attrResult['displayOnlyIfCurrent']['value']:
            try:
                mm.eval(
                    'source AEimagePlaneTemplate.mel; optionMenu -edit -enable true AELookThroughCameraMenu; AEchangeLookThroughCamera %s;' % newshape)
            except:
                pass

        try:
            mc.copyKey(i + ".depth")
            mc.pasteKey(newshape + ".depth")  # edit by ani
        except:
            pass

        pa = mc.listRelatives(i, fullPath=True, parent=True)[0]
        mc.delete(pa)

        renewIpList.append(newshape)

    mc.select(cl=True)
    return renewIpList


def renewImagePlane_backup():
    ipList = mc.ls(type="imagePlane")  # result: ['perspShape2->imagePlaneShape1', 'imagePlaneShape2', ...]

    renewIpList = []
    for i in ipList:
        ipImage = mc.getAttr(i + ".imageName")
        ipOffsetX = mc.getAttr(i + ".offsetX")
        ipOffsetY = mc.getAttr(i + ".offsetY")
        ipUseSeq = mc.getAttr(i + ".useFrameExtension")
        ipFit = mc.getAttr(i + ".fit")
        ipdispOnly = mc.getAttr(i + ".displayOnlyIfCurrent")
        ipDepth = mc.getAttr(i + ".depth")
        ipNum = mc.getAttr(i + ".frameExtension")

        if ipUseSeq: ipFrameOffset = mc.getAttr(i + ".frameOffset")

        camShape = mc.listConnections(i, type="camera", shapes=True)[0]  # result: ['perspShape2']

        newIp = mc.createNode("imagePlane")
        mm.eval('cameraImagePlaneUpdate (\"%s\", \"%s\")' % (camShape, newIp))
        mc.setAttr(newIp + ".imageName", ipImage, type="string")
        # mc.setAttr(newIp+".frameExtension", ipNum)
        mc.setAttr(newIp + ".offsetX", ipOffsetX)
        mc.setAttr(newIp + ".offsetY", ipOffsetY)
        if ipUseSeq:
            mc.setAttr(newIp + ".useFrameExtension", 1)
            mc.expression(n="frame_ext_expression", s=newIp + ".frameExtension=frame;")
            mc.setAttr(newIp + ".frameOffset", ipFrameOffset)
        mc.setAttr(newIp + ".fit", ipFit)
        mc.setAttr(newIp + ".displayOnlyIfCurrent", ipdispOnly)
        mc.setAttr(newIp + ".depth", ipDepth)

        pa = mc.listRelatives(i, fullPath=True, parent=True)[0]
        mc.delete(pa)

        renewIpList.append(newIp)

    return renewIpList


# print "renewImagePlane Ok."

def convertPolyImagePlane(renewIpList):
    newIpList = []
    for i in renewIpList:
        camShape = mc.listConnections(i, type="camera", shapes=True)[0]
        camTransform = mc.listRelatives(camShape, parent=True)[0]
        hfa = mc.camera(camShape, query=True, hfa=True)
        vfa = mc.camera(camShape, query=True, vfa=True)
        fov = math.radians(mc.camera(camShape, query=True, hfv=True))
        aspect = vfa / hfa;

        # temporary statement
        tmp_coverageX = mc.getAttr(i + ".coverageX")
        # print "imageplane coverageX", tmp_coverageX
        if tmp_coverageX == 2420 or tmp_coverageX == 1210 or tmp_coverageX == 3168 or tmp_coverageX == 1584:
            overscan = 1.1
        else:
            overscan = 1.0

        poly = mc.polyPlane(w=1.0, h=aspect, sx=1, sy=1, ax=(0, -1, 0), cuv=1, ch=1)
        roo = mc.getAttr(camTransform + ".rotateOrder")
        # mc.setAttr(poly[0]+".rotateOrder", roo)
        noNamespaceCam = camTransform.split(":")[-1]
        newPlane = mc.rename(poly[0], "layoutImagePlane_" + noNamespaceCam + "_temp")
        mc.polyFlipUV(newPlane)
        mc.parent(newPlane, camTransform)

        xosp = mc.getAttr(camTransform + ".tx")
        ypos = mc.getAttr(camTransform + ".ty")
        zpos = mc.getAttr(camTransform + ".tz")
        depth = mc.getAttr(i + ".depth") * -1.0

        mc.setAttr(newPlane + ".tx", 0.0)
        mc.setAttr(newPlane + ".ty", 0.0)
        mc.setAttr(newPlane + ".tz", depth)

        mc.setAttr(newPlane + ".rx", 90.0)
        mc.setAttr(newPlane + ".ry", 0.0)
        mc.setAttr(newPlane + ".rz", 180.0)
        mc.setAttr(newPlane + ".sx", (2 * (depth) * (math.tan(fov / 2.0))) / overscan)
        mc.setAttr(newPlane + ".sz", (2 * (depth) * (math.tan(fov / 2.0))) / overscan)

        newPlane_new = mc.duplicate(name="layoutImagePlane_" + noNamespaceCam)[0]

        mc.parent(newPlane_new, world=True)

        Sconst = mc.scaleConstraint(newPlane, newPlane_new, maintainOffset=False)

        const = mc.parentConstraint(newPlane, newPlane_new, maintainOffset=False)

        mc.expression(
            s="float $fov = `camera -q -hfv " + camTransform + "`;\r\n\r\n" + newPlane + ".scaleX = " + newPlane + ".scaleY = " + newPlane + ".scaleZ = 2*(" + i + ".depth)*(tand($fov/2.0));\r\n\r\n" + newPlane + ".translateZ = -1*" + i + ".depth;",
            o=newPlane, ae=1, uc="all")

        # mc.expression( s = "float $fov = `camera -q -hfv " + camTransform + "`;\r\n\r\n" + newPlane + ".scaleX = " + newPlane + ".scaleY = " + newPlane + ".scaleZ = 2*(" + i  + ".depth)*(tand($fov/2.0));",  o = newPlane, ae = 1, uc = "all")

        # mc.parent(newPlane, world=True)

        # mc.bakeResults(newPlane, t=(mc.playbackOptions(q=True, ast= True), mc.playbackOptions(q=True, aet=True)), pok=True, sm=True, at=["tx", "ty", "tz", "rx", "ry", "rz", "sx", "sy", "sz"])

        mc.bakeResults(newPlane_new, t=(mc.playbackOptions(q=True, ast=True), mc.playbackOptions(q=True, aet=True)),
                       pok=True, sm=True, at=["tx", "ty", "tz", "rx", "ry", "rz", "sx", "sy", "sz"])

        mc.filterCurve()

        mc.delete(Sconst, const, newPlane)

        newIpList.append(newPlane_new)
    return newIpList
