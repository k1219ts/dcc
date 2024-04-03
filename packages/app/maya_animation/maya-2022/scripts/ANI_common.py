__author__ = 'gyeongheon.jeong'

import maya.cmds as cmds
import maya.OpenMaya as OpenMaya
import maya.OpenMayaUI as OpenMayaUI
import maya.OpenMayaAnim as OpenMayaAnim
import os
import site
import json
import re
import math
import string
import logging

logger = logging.getLogger(__name__)

site.addsitedir('/netapp/backstage/pub/lib/tactic')
from tactic_client_lib import TacticServerStub

DXNODES = ['dxRig', 'dxComponent']

def getShowShot(filename):
    pathSplit = filename.split(os.sep)
    if 'show' in pathSplit:
        showIndex = pathSplit.index('show') + 1
        showName = pathSplit[showIndex]
    else:
        showName = None
    if 'shot' in pathSplit:
        shotIndex = pathSplit.index('shot') + 2
        shotName = pathSplit[shotIndex]
    else:
        shotName = None
    return [showName, shotName]

def getNameSpace(node):
    nameSpace = string.join(node.split(":")[:-1], ":")
    logger.debug(u'Get namespace of : {0} : < {1} >'.format(node, nameSpace))
    return nameSpace

# delete unknown plugins
def cleanUnknownPlugins():
    oldPlugins = cmds.unknownPlugin(q=True, list=True)
    for plugin in oldPlugins:
        cmds.unknownPlugin(plugin, remove=True)

def getListDirs(path, isFile=False, fileType=None):
    finalList = []
    if not os.path.exists(path):
        return finalList

    fileList = os.listdir(path)
    for F in fileList:
        if not isFile:
            if os.path.isdir(path + os.sep + F):
                finalList.append(F)
        else:
            fileName =path + os.sep + F
            if os.path.isfile(fileName):
                if fileType:
                    if fileName.endswith(fileType):
                        finalList.append(F)
                else:
                    finalList.append(F)
    return finalList

def getRootNode(object, type):
    """Find parent dxNode of selected object
    """
    if type == 'dxNode' and ( cmds.objectType(object) in DXNODES ):
            return object
    node = object
    while True:
        parentNode = cmds.listRelatives(node, ap=1)
        if type == 'dxNode' and not parentNode:
            node = None
            break
        if type == 'dxNode' and ( cmds.objectType(parentNode[0]) in DXNODES ):
            node = parentNode[0]
            break
        node = parentNode[0]
    return node

def getShotnameFromFile(scenePath):
    pass

# generate wheel expression
def wheelExpresstion(body_Translate, wheel_rotationOrder, wheel_Radius, side):
    if side == "left":
        expresstionString = "%s = ( %s/(2 * 3.14 * %f) ) * 360;" % (wheel_rotationOrder, body_Translate, wheel_Radius)
        cmds.expression(s=expresstionString, o="", ae=1, uc="all")
    elif side == "right":
        expresstionString = "%s = ( %s/(2 * 3.14 * %f) ) * -360;" % (wheel_rotationOrder, body_Translate, wheel_Radius)
        cmds.expression(s=expresstionString, o="", ae=1, uc="all")

# generate locator and match t,r to selected obejct
def locToObj(selectedObj, t=1, r=1):
    Translation = cmds.xform(selectedObj, q=1, t=1, ws=1)
    Rotation = cmds.xform(selectedObj, q=1, ro=1, ws=1)
    loc = cmds.spaceLocator(n=selectedObj + "_LOC")[0]
    if t: cmds.xform(loc, t=Translation)
    if r: cmds.xform(loc, ro=Rotation)

def changeNameSpace(oldNameSpace, newNameSpace):
    cmds.namespace(set=':')
    cmds.namespace(add=newNameSpace)
    cmds.namespace(mv=(oldNameSpace, newNameSpace))
    cmds.namespace(rm=oldNameSpace)

def distance2Objects(objectA, objectB):
    Atranslate = cmds.xform(objectA, q=1, t=1, ws=1)
    Btranslate = cmds.xform(objectB, q=1, t=1, ws=1)

    tx = Btranslate[0] - Atranslate[0]
    ty = Btranslate[1] - Atranslate[1]
    tz = Btranslate[2] - Atranslate[2]

    dT_AtoB = math.sqrt((tx ** 2 + ty ** 2) + (tz ** 2))

    return dT_AtoB


def fishTail_Expression(controlers):
    controlers = cmds.ls(sl=1)

    controler = cmds.spaceLocator(n="controler#")[0]
    """
    for attr in cmds.listAttr(controler, k=True):
        cmds.setAttr(controler + "." + attr, k = False)
    """
    cmds.addAttr(controler, ln="Speed", at="double", dv=1)
    cmds.setAttr(controler + ".Speed", e=True, k=True)
    cmds.addAttr(controler, ln="Offset", at="double", dv=1)
    cmds.setAttr(controler + ".Offset", e=True, k=True)
    cmds.addAttr(controler, ln="Amplitude", at="double", dv=1)
    cmds.setAttr(controler + ".Amplitude", e=True, k=True)

    rotAttr = "rz"
    exString = ""
    var = 0
    varOffset = 0.5

    for selObj in controlers:
        exString += "%s.%s = sin((frame * %s.Speed) + %s.Offset + %f) * %s.Amplitude;\n" % (
        selObj, rotAttr, controler, controler, var, controler)
        var += varOffset

    cmds.expression(n="GH_fishTail_ex_#", s=exString, o=controlers[0], ae=1, uc="all")


def exportRetime2Nuke(ip, filename):
    min = int(float(cmds.playbackOptions(q=1, min=1)))
    max = int(float(cmds.playbackOptions(q=1, max=1)))

    f = open(filename + ".nk", "w")

    if not f.closed:
        # hide all objects.
        for panName in cmds.getPanel(all=True):
            if 'modelPanel' in panName:
                cmds.isolateSelect(panName, state=1)
        for imageplane in ip:
            frameoffset = cmds.getAttr(str(imageplane) + ".frameOffset") * -1
            if frameoffset:
                f.write("TimeOffset {\n")
                f.write(" time_offset {}\n".format(str(frameoffset)))
                f.write(' time ""\n')
                f.write(' name {}_timeOffset\n'.format(imageplane))
                f.write('}\n\n')
            f.write("TimeWarp {\n")
            f.write(" lookup {{curve x%i" % min)
            for frame in range(min, max + 1):
                #cmds.currentTime(frame, e=1)
                retimedFrame = cmds.getAttr(str(imageplane) + ".frameExtension",
                                            time=frame)
                f.write(" %d" % retimedFrame)
            f.write("}}\n")
            f.write(" filter none\n")
            f.write(" name %s_timeWarp\n" % imageplane)
            f.write("}\n\n")

        f.close()

        # show all objects.
        for panName in cmds.getPanel(all=True):
            if 'modelPanel' in panName:
                cmds.isolateSelect(panName, state=0)


#### from pipeline import tool

def tacticPlayback(show=None, shot=None):
    if show and shot:
        # get code name
        server = TacticServerStub(login='test.creator',
                                  password='eprtmxj',
                                  server='10.0.0.51',
                                  project='dexter_studios')
        expr = "@SOBJECT(sthpw/project['name', '%s'])" % show
        info = server.eval(expr)
        if not info:
            return
        code = info[0].get('code')

        # get shot info
        server = TacticServerStub(login='test.creator',
                                  password='eprtmxj',
                                  server='10.0.0.51',
                                  project=code)
        expr = "@SOBJECT(%s/shot['code', '%s'])" % (code, shot)
        info = server.eval(expr)
        if info:
            if 'tc_frame_start' in info[0].keys():
                cmds.playbackOptions(ast=info[0]['tc_frame_start'])
            else:
                cmds.playbackOptions(ast=info[0]['frame_in'])

            cmds.playbackOptions(min=info[0]['frame_in'])
            cmds.playbackOptions(max=info[0]['frame_out'])

            if 'tc_frame_end' in info[0].keys():
                cmds.playbackOptions(aet=info[0]['tc_frame_end'])
            else:
                cmds.playbackOptions(aet=info[0]['frame_out'])
            return True


def fileTextureNameReplace(oldName, newName):
    txtureList = cmds.ls(type="file")

    for i in txtureList:
        imgName = cmds.getAttr(i + ".fileTextureName")

        if oldName in imgName:
            imgName = imgName.replace(oldName, newName)
            cmds.setAttr(i + ".fileTextureName", imgName, type="string")


def createTimeWarp(selObjects=list()):
    minTime = cmds.playbackOptions(q=1, min=1)
    maxTime = cmds.playbackOptions(q=1, max=1)

    TW_timeCrvNode = cmds.createNode("animCurveTT", n="jta_timewarpCRV")
    cmds.setKeyframe(TW_timeCrvNode, t=minTime, v=minTime, itt="spline", ott="spline")
    cmds.setKeyframe(TW_timeCrvNode, t=maxTime, v=maxTime, itt="spline", ott="spline")

    TW_conditionNode = cmds.createNode("condition", n="jta_EnableWarp")

    cmds.connectAttr("time1.outTime", TW_conditionNode + ".colorIfFalseR")
    cmds.connectAttr(TW_timeCrvNode + ".output", TW_conditionNode + ".colorIfTrueR")

    for i in selObjects:
        objectAnimCrv = cmds.listConnections(i, type="animCurve")
        for a in objectAnimCrv:
            cmds.connectAttr(TW_conditionNode + ".outColorR", a + ".input")

def writeSeqData(ls_sequencer = list() ):
    seqData = dict()
    ls_seqShot = list()

    for ls_sequencer_i in ls_sequencer:
        seqData[ls_sequencer_i] = dict()
        ls_seqShot = cmds.listConnections('%s.shots' % ls_sequencer_i)

        if ls_seqShot:
            for ls_seqShot_i in ls_seqShot:
                seqData[ls_sequencer_i][ls_seqShot_i] = dict()
                ori_startFrame = cmds.getAttr(ls_seqShot_i + ".startFrame")
                ori_endFrame = cmds.getAttr(ls_seqShot_i + ".endFrame")
                seqShot_startFrame = cmds.getAttr(ls_seqShot_i + ".sequenceStartFrame")
                seqShot_endFrame = cmds.getAttr(ls_seqShot_i + ".sequenceEndFrame")
                seqShot_scale = cmds.getAttr(ls_seqShot_i + ".scale")

                seqData[ls_sequencer_i][ls_seqShot_i]["startframe"] = ori_startFrame
                seqData[ls_sequencer_i][ls_seqShot_i]["endframe"] = ori_endFrame
                seqData[ls_sequencer_i][ls_seqShot_i]["sequenceStartFrame"] = seqShot_startFrame
                seqData[ls_sequencer_i][ls_seqShot_i]["sequenceEndFrame"] = seqShot_endFrame
                seqData[ls_sequencer_i][ls_seqShot_i]["scale"] = seqShot_scale
        else:
            return 0
    return seqData

def WriteKeysFromDic(seqData):

    keyDic = dict()
    seqOriStartFrameList = list()
    seqOriEndFrameList = list()
    seqStartFrameList = list()
    seqEndFrameList = list()
    seqScaleList = list()

    for i in seqData.values():
        for a in i.values():
            seqOriStartFrameList.append(a["startframe"])
            seqOriEndFrameList.append(a["endframe"])
            seqStartFrameList.append(a["sequenceStartFrame"])
            seqEndFrameList.append(a["sequenceEndFrame"])
            seqScaleList.append(a["scale"])

    seqOriStartFrameList.sort()
    seqOriEndFrameList.sort()
    seqStartFrameList.sort()
    seqEndFrameList.sort()
    seqScaleList.sort()

    seqStartFrameList.append(seqEndFrameList[-1])
    seqOriStartFrameList.append(seqOriEndFrameList[-1])

    keyDic["retime_Time"] = seqStartFrameList
    keyDic["retime_Value"] = seqOriStartFrameList
    keyDic["retime_Scale"] = 1/seqScaleList[-1]

    return keyDic

def ExportSeqRetime(sim=True, sceneName = None, path=None):
    sequenceScale = 1

    if not path and not sceneName:
        datapath, sceneName = getScenePath(path=True, scene=True)
        sceneBaseName = os.path.splitext(sceneName)[0]
    elif not path and sceneName:
        datapath = getScenePath(path=True)
        sceneBaseName = sceneName
    elif not sceneName and path:
        datapath = path
        sceneName = getScenePath(scene=True)
        sceneBaseName = os.path.splitext(sceneName)[0]
    else:
        datapath = path
        sceneBaseName = sceneName

    # /show/kfyg/shot/POS/OPNpos_0010/ani/dev/data/retime
    jsonPath = os.sep.join( [ os.path.split(datapath)[0],
                              "data",
                              "retime" ] )

    # /show/kfyg/shot/POS/OPNpos_0010/ani/dev/data/retime/"sceneName"_retime.json
    jsonFilePath = os.path.join( jsonPath, sceneBaseName + "_retime.json" )

    addnum = 1
    newPath = jsonFilePath

    if not os.path.exists(jsonPath):
        os.makedirs(jsonPath)

    while os.path.isfile(newPath):
        newPath = os.path.splitext(jsonFilePath)[0]   # /show/kfyg/shot/POS/OPNpos_0010/ani/dev/data/retime/"sceneName"
        newPath += "_rev_%02d" % addnum           # _rev_01
        newPath += os.path.splitext(jsonFilePath)[1]  # .json
        addnum += 1

    os.system("mv %s %s" %(jsonFilePath, newPath))

    ls_seq = cmds.ls(type="sequencer")

    if sim:
        if ls_seq:
            SeqDic = writeSeqData(ls_seq)
            if SeqDic:
                SeqDicSorted = WriteKeysFromDic(SeqDic)
                f = open(jsonFilePath, 'w')
                json.dump(SeqDicSorted, f, indent=4)
                f.close()
                sequenceScale = SeqDicSorted["retime_Scale"]

        return float(sequenceScale)

    else: # export alembic to render, FX ...
        # apply timewarp
        if os.path.exists(jsonFilePath):
            timewarpNode = SetTimewarpKey( jsonFilePath )
        else:
            timewarpNode = None
        return float(sequenceScale), timewarpNode # sequenceScale = 1

def getScenePath(path=False, scene=False):
    filePath = cmds.file(q=1, sn=1)
    dirPath = os.path.dirname(filePath)
    sceneName = os.path.basename(filePath)

    if path and scene:
        return dirPath, sceneName
    elif scene:
        return sceneName
    elif path:
        return dirPath

def SetTimewarpKey(KeyDataJson):
    timewarpNode = None

    if not cmds.objExists("seq_timewarp"):
        timewarpNode = cmds.createNode("animCurveTT", n = "seq_timewarp")

    # keyData = json.loads(open(jsonFileName, 'r').read())
    with open(KeyDataJson, 'r') as f:
        keyData = json.loads(f.read())

    for i in range(len(keyData['retime_Time']))[:-1]:
        cmds.setKeyframe(timewarpNode, v=keyData["retime_Value"][i],
                         time=keyData["retime_Time"][i])

    cmds.setKeyframe(timewarpNode, v=keyData["retime_Value"][-1] + 1,
                     time=keyData["retime_Time"][-1] + 1)

    timewarpNodeKeyList = cmds.keyframe(timewarpNode, q=1)
    cmds.keyTangent(timewarpNode, itt="linear", ott="linear")
    cmds.keyTangent(timewarpNode, e=1, t=(timewarpNodeKeyList[0], timewarpNodeKeyList[0]), itt="spline", ott="spline")
    cmds.keyTangent(timewarpNode, e=1, t=(timewarpNodeKeyList[-1], timewarpNodeKeyList[-1]), itt="spline", ott="spline")

    cmds.selectKey(timewarpNode, add=True, k=True)
    cmds.setInfinity(pri="linear", poi="linear")

    cmds.connectAttr("%s.apply" % timewarpNode, "time1.timewarpIn.timewarpIn_Raw")
    cmds.setAttr("time1.enableTimewarp", 1)
    cmds.playbackOptions(e=1, max=keyData["retime_Time"][-1])

    return timewarpNode
    #os.remove( KeyDataJson )


def renameRef(ref=dict()):
    for sel in ref:
        RN_node = cmds.referenceQuery(sel, referenceNode=True)
        refFileFullName = cmds.referenceQuery(RN_node, filename=1)
        oldNameSpace = cmds.file(refFileFullName, q=True, namespace=True)
        cmds.lockNode(RN_node, l=False)

        refFileName = cmds.referenceQuery(sel, filename=1, shortName=1)
        newNameSpaceTemp = refFileName.split("_")[0]

        result = cmds.promptDialog(title='Rename Reference',
                                   message='Enter Name:',
                                   text=oldNameSpace,
                                   button=['OK', 'Cancel'],
                                   defaultButton='OK',
                                   cancelButton='Cancel',
                                   dismissString='Cancel')

        if result == 'OK':
            newNameSpace = cmds.promptDialog(query=True, text=True)
        else:
            return

        nameSpaceNumber = re.findall(r'\d+', newNameSpace)

        if nameSpaceNumber:
            nameSpaceCount = int( nameSpaceNumber[0] )
        else:
            nameSpaceCount = 1

        while cmds.namespace(exists=newNameSpace):
            newNameSpace = newNameSpaceTemp + str( nameSpaceCount )
            nameSpaceCount += 1

        reNameRN = newNameSpace + 'RN'

        rnRename = cmds.rename(RN_node, reNameRN)
        cmds.lockNode(rnRename, l=True)
        cmds.file(refFileFullName, e=1, namespace=newNameSpace)

# API custom playBlast
def playBlastByJta(path, start, end, frame):
    view = OpenMayaUI.M3dView.active3dView()
    image = OpenMaya.MImage()
    mtime = OpenMaya.MTime()

    _time = start
    step = 1
    defaultStart = 0

    mtime.setValue(_time)
    #OpenMaya.MGlobal.viewFrame(mtime)
    #OpenMayaAnim.MAnimControl.setCurrentTime(mtime)
    #view.readColorBuffer(image, True)
    #image.writeToFile(path + ".{:04d}.jpg".format(defaultStart), 'jpg')
    #outPath = path + ".####.jpg"

    while _time < end:
        #maya.cmds.currentTime(_time)
        mtime.setValue(_time)
        #OpenMaya.MGlobal.viewFrame(mtime)
        OpenMayaAnim.MAnimControl.setCurrentTime(mtime)
        view.readColorBuffer(image, True)
        image.writeToFile(path + ".{:04d}.jpg".format(defaultStart), 'jpg')
        _time += step
        defaultStart += 1
    outPath = path + ".####.jpg"

    return outPath

def getTimeWarpScale():
    animCurve = "timewarp"

    keyValue = cmds.keyframe(animCurve, q=1, vc=1)
    keyTime = cmds.keyframe(animCurve, q=1)

    timeScaleAvg = 1
    index = 0

    for i in range(len(keyValue)):
        timeScale = keyValue[i] / keyTime[i]
        if timeScaleAvg > timeScale:
            timeScaleAvg = timeScale
            print "{0} / {1} = {2}".format(keyValue[i], keyTime[i], timeScaleAvg)

    cmds.confirmDialog(message=str(timeScaleAvg))


def filterCurves():
    def resample_keys(kv, thresh):
        start = float(min(kv.keys()))
        end = float(max(kv.keys()))
        startv = float(kv[start])
        endv = float(kv[end])
        total_error = 0
        offender = -1
        outlier = -1

        for k, v in kv.items():
            offset = (k - start) / (end - start)
            sample = (offset * endv) + ((1 - offset) * startv)
            delta = abs(v - sample)
            total_error += delta
            if delta > outlier:
                outlier = delta
                offender = k
        if total_error < thresh or len(kv.keys()) == 2:
            return [{start: startv, end: endv}]
        else:
            s1 = {kk: vv for kk, vv in kv.items() if kk <= offender}
            s2 = {kk: vv for kk, vv in kv.items() if kk >= offender}
            return resample_keys(s1, thresh) + resample_keys(s2, thresh)

    def rejoin_keys(kvs):
        result = {}
        for item in kvs:
            result.update(item)
        return result

    def decimate(keys, tolerance):
        return rejoin_keys(resample_keys(keys, tolerance))

    animCrv = cmds.keyframe(q=True, n=True)
    keyDic = dict()

    for each in animCrv:
        time = cmds.keyframe(each, q=True, tc=True)
        value = cmds.keyframe(each, q=True, vc=True)
        keyDic[each] = dict()
        for i in range(len(time)):
            keyDic[each][time[i]] = value[i]

        keyDic[each] = decimate(keyDic[each], 3)
        cmds.cutKey(each, t=(time[0] + 1, time[-1] - 1), clear=True)
        for i in keyDic[each].keys():
            cmds.setKeyframe(each, t=(int(i), int(i)), v=keyDic[each][i])


"""
Offset Key From dexcmd.aniCommon
"""
def selectedOffsetKey(start, end):
    for i in cmds.ls(sl=True):
        objs = cmds.ls(i, dag=True, ni=True)
        connections = cmds.listConnections(objs, type='animCurve')
        if connections:
            for i in connections:
                ln = cmds.listConnections(i, p=True)
                src = ln[0].split('.')
                offsetKeyAttr(src[0], src[-1], 1, start, end)


def offsetKeyAttr(obj, attr, offset, start=False, end=False):
    ln = '%s.%s' % (obj, attr)
    frames = cmds.keyframe(ln, q=True, a=True)
    if end:
        # end frame
        end_value = cmds.getAttr(ln, t=frames[-1])
        tmp_value = cmds.getAttr(ln, t=frames[-1] - offset)
        set_value = end_value - tmp_value
        cmds.setKeyframe(ln, itt='spline', ott='spline', t=frames[-1] + offset,
                         at=attr, v=end_value + set_value)
        print "end"
    if start:
        # start frame
        start_value = cmds.getAttr(ln, t=frames[0])
        tmp_value = cmds.getAttr(ln, t=frames[0] + offset)
        set_value = tmp_value - start_value
        cmds.setKeyframe(ln, itt='spline', ott='spline', t=frames[0] - offset,
                         at=attr, v=start_value - set_value)
"""
===============================================================================
"""

def temp():
    import maya.cmds as cmds

    obj = "pCube1"

    delay = 3
    count = 5

    melStr = 'float ${newObj}Atx = `getAttr -t ($CFrame-{delay}) {obj}.translateX`;'
    melStr += 'float ${newObj}Aty = `getAttr -t ($CFrame-{delay}) {obj}.translateY`;'
    melStr += 'float ${newObj}Atz = `getAttr -t ($CFrame-{delay}) {obj}.translateZ`;'

    melStr += 'float ${newObj}Arx = `getAttr -t ($CFrame-{delay}) {obj}.rotateX`;'
    melStr += 'float ${newObj}Ary = `getAttr -t ($CFrame-{delay}) {obj}.rotateY`;'
    melStr += 'float ${newObj}Arz = `getAttr -t ($CFrame-{delay}) {obj}.rotateZ`;'

    melStr += 'float ${newObj}Otx = `getAttr -t $CFrame {obj}.translateX`;'
    melStr += 'float ${newObj}Oty = `getAttr -t $CFrame {obj}.translateY`;'
    melStr += 'float ${newObj}Otz = `getAttr -t $CFrame {obj}.translateZ`;'

    melStr += 'float ${newObj}Orx = `getAttr -t $CFrame {obj}.rotateX`;'
    melStr += 'float ${newObj}Ory = `getAttr -t $CFrame {obj}.rotateY`;'
    melStr += 'float ${newObj}Orz = `getAttr -t $CFrame {obj}.rotateZ`;'

    melStr += """
    if (${newObj}Atx!=${newObj}Otx && ${newObj}Aty!=${newObj}Oty && ${newObj}Atz!=${newObj}Otz){{
    	{newObj}.translateX = ${newObj}Atx;
    	{newObj}.translateY = ${newObj}Aty;
    	{newObj}.translateZ = ${newObj}Atz;
    }};
    if (${newObj}Arx!=${newObj}Orx && ${newObj}Ary!=${newObj}Ory && ${newObj}Arz!=${newObj}Orz){{
    	{newObj}.rotateX = ${newObj}Arx;
    	{newObj}.rotateY = ${newObj}Ary;
    	{newObj}.rotateZ = ${newObj}Arz;
    }};
    """
    newMelStr = 'float $CFrame = `currentTime -q`;'

    for i in range(count):
        newObj = cmds.polyCube()
        print newObj
        newMelStr += melStr.format(delay=delay + i * delay, obj=obj, newObj=newObj[0])

    cmds.expression(s=newMelStr, o="", ae=1, uc="all")
