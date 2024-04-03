'''
USD Rig Export

[ Command ]
RigShotExport(node='node name', showDir='output show dir', shot='shot name', fr=('start', 'end')).doIt()

'''
import os
import re
import string

from pxr import Sdf, Usd, UsdGeom, Gf, Vt
import maya.cmds as cmds
import maya.mel as mel

import dxsMsg
import MsgSender
import Arguments
import PathUtils
import dxsMayaUtils
import dxsUsdUtils
import Attributes
import GeomMain
import EditPrim
import dxsXformUtils
import SessionUtils
import PackageUtils
import Texture
import Material
import ClipUtils

CON_MAP = {
    'dexter': {
        'nodes': ['place_CON', 'place_NUL', 'direction_NUL', 'direction_CON', 'move_NUL', 'move_CON'],
        'attrs': ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
    }
}
# InitScaleAttributes = ['initScale', 'globalScale'] -> remove 'globalScale' 2019.01.10 request by taehoon.kim
InitScaleAttributes = ['initScale']


def SelectAnimLayer():
    rootLayer = cmds.animLayer(q=True, root=True)
    if not rootLayer:
        return
    layers = cmds.animLayer(rootLayer, q=True, children=True)
    if layers:
        layers.insert(0, rootLayer)
        for i in layers[:-1]:
            cmds.animLayer(i, e=True, selected=False)
            cmds.animLayer(i, e=True, preferred=False)
        cmds.animLayer(layers[-1], e=True, selected=True)
        cmds.animLayer(layers[-1], e=True, preferred=True)


def GetRigNodes(selected='|*'):
    nodes = cmds.ls(selected, type='dxRig', l=True, r=True)
    if not nodes:
        dxsMsg.Print('warning', "[not found 'dxRig']")
        return

    exportNodes = list()
    for n in nodes:
        if cmds.getAttr('%s.action' % n) and dxsMayaUtils.GetViz(n):
            exportNodes.append(n)
    if not exportNodes:
        dxsMsg.Print('warning', "[not found export 'dxRig']")
        return

    return exportNodes


def GetRigObjects(node, gtype):
    '''
    Args:
        node  (str): dxRig node
        gtype (str): geometry type
    '''
    attrMap = {'high': 'renderMeshes', 'mid': 'midMeshes', 'low': 'lowMeshes', 'sim': 'simMeshes'}
    objects = cmds.getAttr('%s.%s' % (node, attrMap[gtype]))
    nsLayer, nodeName = dxsMayaUtils.GetNamespaceInfo(node)
    result  = list()
    for i in objects:
        name = nsLayer + ':' if nsLayer else ''
        name += i
        fullPath = cmds.ls(name, l=True)
        # if fullPath and dxsMayaUtils.GetViz(fullPath[0]):   # add GetViz - 2019.03.15
        #     result += fullPath
        if fullPath:    # GetViz not necessary
            result += fullPath
    return result

def GetRigCons(node, cons):
    nsName, nodeName = dxsMayaUtils.GetNamespaceInfo(node)
    result = list()
    for c in cons:
        name = c if not nsName else nsName + ':' + c
        if cmds.objExists(name):
            result.append(name)
    if len(cons) == len(result):
        return result

def GetRigControlers(node):
    nsName, nodeName = dxsMayaUtils.GetNamespaceInfo(node)
    result = list()
    for c in cmds.getAttr('%s.controlers' % node):
        name = c if not nsName else nsName + ':' + c
        if cmds.objExists(name):
            result.append(name)
    return result

def GetRigAssetDir(node, showDir=None):
    assetName = cmds.getAttr('%s.assetName' % node)
    rigDir    = None
    if cmds.referenceQuery(node, isNodeReferenced=True):
        rigMayaFile = cmds.referenceQuery(node, filename=True, withoutCopyNumber=True)
        baseName    = os.path.splitext(os.path.basename(rigMayaFile))[0]
        # outdir check
        dirRule = '{SHOW}' + '/asset/{ASSET}/rig/usd/{NAME}'.format(ASSET=assetName, NAME=baseName)
        rigDir  = dirRule.format(SHOW=showDir)
        if not os.path.exists(rigDir):
            showDir, showName = PathUtils.GetRootPath(rigMayaFile)
            rigDir = dirRule.format(SHOW=showDir)
    else:
        if showDir:
            rigRoot = '{SHOW}/asset/{ASSET}/rig/usd'.format(SHOW=showDir, ASSET=assetName)
            if os.path.exists(rigRoot):
                subDirs = list()
                for i in os.listdir(rigRoot):
                    if os.path.isdir(os.path.join(rigRoot, i)):
                        subDirs.append(i)
                subDirs.sort()
                rigDir = os.path.join(rigRoot, subDirs[-1])
    return assetName, rigDir


def DelPurposeAttribute(objects):
    for o in objects:
        if cmds.attributeQuery('USD_ATTR_purpose', n=o, ex=True):
            cmds.deleteAttr('%s.USD_ATTR_purpose' % o)
def SetPurposeAttribute(setObjects, excludeObjects, purpose):
    for node in setObjects:
        if not node in excludeObjects:
            if not cmds.attributeQuery('USD_ATTR_purpose', n=node, ex=True):
                cmds.addAttr(node, ln='USD_ATTR_purpose', nn='purpose', dt='string')
            cmds.setAttr('%s.USD_ATTR_purpose' % node, purpose, type='string')
def SetRigPurposeAttribute(renderObjects, proxyObjects):
    DelPurposeAttribute(list(set(renderObjects + proxyObjects)))
    if not proxyObjects:
        return
    intersectObjects = list(set(proxyObjects).intersection(set(renderObjects)))
    SetPurposeAttribute(renderObjects, intersectObjects, 'render')
    SetPurposeAttribute(proxyObjects,  intersectObjects, 'proxy')



class RigNodeInspect:
    '''
    Args:
        fr (tuple) : base input is (start, end)
    '''
    def __init__(self, node, fr=(None, None), autofr=False):
        self.node  = node
        self.fr    = fr
        self.autofr= autofr
        self.ctrls = GetRigControlers(node)

        # Member Variables
        self.Objects = self.initObjects()
        self.extraRignodes= None
        self.extraObjects = None
        self.exportRange  = self.fr
        self.restFrame    = None
        self.rigType = cmds.getAttr("%s.rigType" % self.node)
        self.rigBake = cmds.getAttr("%s.rigBake" % self.node)

        self.rigRootCon = cmds.getAttr('%s.rootCon' % self.node)

        self.meshCheck()
        self.keyFrameCheck()
        self.ctrlsCheck()

    def initObjects(self):
        result = dict()
        for gt in ['high', 'mid', 'low', 'sim']:
            result[gt] = list()
        return result


    def meshCheck(self):
        '''
        Main Rig mesh & Extra Rig mesh and controlers update
        '''
        for gt in ['high', 'mid', 'low', 'sim']:
            self.Objects[gt] = GetRigObjects(self.node, gt)
        # extra rig
        rigNodes = list()
        for shape in cmds.ls(self.Objects['high'], dag=True, type='pxrUsdProxyShape'):
            transNode = cmds.listRelatives(shape, p=True, f=True)[0]
            if cmds.attributeQuery('rigNode', n=transNode, ex=True):
                fullPath = cmds.ls(cmds.getAttr('%s.rigNode' % transNode), l=True)
                if fullPath:
                    if dxsMayaUtils.GetViz(fullPath[0]):
                        rigNodes.append(fullPath[0])
            elif cmds.attributeQuery('refNode', n=transNode, ex=True):
                refNodes = cmds.referenceQuery(cmds.getAttr('%s.refNode' % transNode), n=True)
                if cmds.nodeType(refNodes[0]) == 'dxRig':
                    cmds.addAttr(transNode, ln='rigNode', dt='string')
                    cmds.setAttr('%s.rigNode' % transNode, refNodes[0], type='string')
                    fullPath = cmds.ls(refNodes[0], l=True)
                    if dxsMayaUtils.GetViz(fullPath[0]):
                        rigNodes.append(fullPath[0])

        if not rigNodes:
            return
        self.extraRignodes= rigNodes
        self.extraObjects = self.initObjects()
        for r in rigNodes:
            for gtype in ['high', 'mid', 'low', 'sim']:
                self.extraObjects[gtype] += GetRigObjects(r, gtype)
            self.ctrls += GetRigControlers(r)

    def keyFrameCheck(self):
        self.exportRange = (self.fr[0] - 1, self.fr[1] + 1)
        allFrames = list()
        for ctrl in self.ctrls:
            for n in cmds.listHistory(ctrl, lf=False):
                if cmds.nodeType(n).find('animCurve') > -1:
                    frames = cmds.keyframe(n, q=True)
                    if frames:
                        allFrames += frames
        allFrames.sort()
        if self.autofr and allFrames:
            restFrame = self.fr[0] - 51
            if allFrames[0] <= restFrame:
                self.restFrame  = restFrame
                self.exportRange= (restFrame, self.exportRange[1])
        if not allFrames and self.rigType != 4:
            self.exportRange = (self.fr[0], self.fr[0])

    def ctrlsCheck(self):
        if self.extraRignodes:
            worldCons = list()
            for node in self.extraRignodes:
                ns, name = dxsMayaUtils.GetNamespaceInfo(node)
                for c in ['move_CON', 'direction_CON', 'place_CON']:
                    obj = c if not ns else ns + ':' + c
                    worldCons.append(obj)
            self.ctrls = list(set(self.ctrls) - set(worldCons))


    def bake(self):
        '''
        KeyBake and +,-1 offset frame key control
        '''
        if self.exportRange[0] == self.exportRange[1]:
            return

        if not self.rigBake:
            return

        bakeOnOverrideLayer = False
        animLayer = cmds.animLayer(q=True, root=True)
        if animLayer:
            if cmds.animLayer(animLayer, q=True, children=True):
                bakeOnOverrideLayer = True

        # retime process
        bakeRange = [self.exportRange[0], self.exportRange[1]]
        orgCurrentTime = cmds.currentTime(q = True)
        isRetime = False
        if cmds.getAttr("time1.enableTimewarp"):
            isRetime = True
            cmds.currentTime(self.fr[1])
            bakeRange[1] = int(cmds.getAttr("time1.outTime")) + 1
            cmds.setAttr("time1.enableTimewarp", False)

        cmds.bakeResults(self.ctrls, simulation=True, t=tuple(bakeRange), bol=bakeOnOverrideLayer, dic=False, pok=False)
        SelectAnimLayer()

        # retime undo
        if isRetime:
            cmds.setAttr("time1.enableTimewarp", True)
            cmds.currentTime(orgCurrentTime)

        # KeyOffset value re-set
        for c in self.ctrls:
            RigNodeInspect.SetKeyoffset(c)

        # Restpos re-init
        if self.restFrame:
            SetInitializeControlers(self.node, self.restFrame, None)

        # Constant Key check
        _valid = 0
        for c in self.ctrls:
            tangents = cmds.keyTangent(c, q=True, ia=True, oa=True)
            if tangents:
                tangents = list(set(tangents))
                if len(tangents) > 1:
                    _valid += 1
        if _valid == 0 and self.rigType != 4:
            self.exportRange = (self.fr[0], self.fr[0])


    @staticmethod
    def SetKeyoffset(node):
        if cmds.listAttr(node, k=True):
            for ln in cmds.listAttr(node, k=True):
                typeln = cmds.getAttr(node + '.' + ln, type=True)
                if re.findall(r'\d+', typeln):
                    continue

                plug = node + '.' + ln
                frames = cmds.keyframe(plug, q=True, a=True)
                if not frames:
                    continue
                tangents = cmds.keyTangent(plug, q=True, ia=True, oa=True)
                tangents = list(set(tangents))
                if len(tangents) == 1:
                    continue

                # Start Frame
                ref_value  = cmds.getAttr(plug, t=frames[2])
                start_value= cmds.getAttr(plug, t=frames[1])
                set_value  = cmds.getAttr(plug, t=frames[0])
                if start_value == set_value:
                    print '# Debug : start offset value ->', plug, set_value, start_value
                    offsetVal = ref_value - start_value
                    set_value = start_value - offsetVal
                    cmds.setKeyframe(plug, itt='spline', ott='spline', t=frames[0], v=set_value)

                # End Frame
                ref_value = cmds.getAttr(plug, t=frames[-3])
                end_value = cmds.getAttr(plug, t=frames[-2])
                set_value = cmds.getAttr(plug, t=frames[-1])
                if end_value == set_value:
                    print '# Debug : end offset value ->', plug, end_value, set_value
                    offsetVal = end_value - ref_value
                    set_value = end_value + offsetVal
                    cmds.setKeyframe(plug, itt='spline', ott='spline', t=frames[-1], v=set_value)



def SetInitializeControlers(node, initFrame, firstFrame):
    '''
    Args:
        node (str): dxRig node
        initFrame  (int): initialize frame, restpos frame
        firstFrame (int): export start frame
    '''
    nsName, nodeName = dxsMayaUtils.GetNamespaceInfo(node)
    data = cmds.getAttr('%s.controlersData' % node)
    if not data:
        print 'not found controlers initialize data.'
        return
    cmds.currentTime(initFrame)

    worldCons = list()
    for i in ['move_CON', 'direction_CON', 'place_CON']:
        obj = i if not nsName else nsName + ':' + i
        worldCons.append(obj)

    constraintedNodes = list()
    data = eval(data)
    for i in data:
        # print i, data[i]['value'], data[i]['type']
        nodeName, attrName = i.split('.')
        nodeName = nodeName if not nsName else nsName + ':' + nodeName
        if not nodeName in worldCons:
            ln = nodeName + '.' + attrName
            if cmds.objExists(nodeName) and cmds.getAttr(ln, k=True) and data[i]['type'] != 'string':
                # check constrain
                connects = cmds.listConnections(nodeName + '.parentInverseMatrix[0]', type='constraint', s=False, d=True)
                if connects:
                    constraintedNodes.append(nodeName)
                try:
                    cmds.setAttr(ln, data[i]['value'])
                    cmds.setKeyframe(ln)
                except:
                    pass

    if constraintedNodes:
        for n in constraintedNodes:
            attrs = cmds.listAttr(n, st='blend*')
            if attrs:
                for a in attrs:
                    ln = n + '.' + a
                    cmds.setAttr(ln, 0)
                    cmds.setKeyframe(ln)


class MuteCtrl:
    '''
    Attribute mute control
    '''
    def __init__(self, nodes, attrs):
        self.m_nodes = nodes
        self.m_attrs = attrs
        self.m_data = dict()
        self.initScaleAttr = None

    def getVal(self):
        for node in self.m_nodes:
            self.m_data[node] = dict()
            for ln in self.m_attrs:
                if not cmds.getAttr('%s.%s' % (node, ln), l=True):
                    self.m_data[node][ln] = cmds.getAttr('%s.%s' % (node, ln))
            # initScale
            for at in InitScaleAttributes:
                if cmds.attributeQuery(at, n=node, ex=True):
                    self.initScaleAttr = at
                    self.m_data[node][at] = cmds.getAttr('%s.%s' % (node, at))

    # Mute Control
    def setMute(self):
        if not self.m_data:
            return
        # scale attributes
        scaleAttrs = ['sx', 'sy', 'sz']
        if self.initScaleAttr:
            scaleAttrs.append(self.initScaleAttr)
        for node in self.m_data:
            for ln in self.m_data[node]:
                if ln in scaleAttrs:
                    try:
                        cmds.setAttr('%s.%s' % (node, ln), 1)
                    except:
                        pass
                else:
                    try:
                        cmds.setAttr('%s.%s' % (node, ln), 0)
                    except:
                        pass
                cmds.mute('%s.%s' % (node, ln), d=False, f=True)

    def setUnMute(self):
        if not self.m_data:
            return
        for node in self.m_data:
            for ln in self.m_data[node]:
                try:
                    cmds.setAttr('%s.%s' % (node, ln), self.m_data[node][ln])
                except:
                    pass
                cmds.mute('%s.%s' % (node, ln), d=True, f=True)


def RigRootConExport(filename, nodeName, fr, step, fps=24.0):
    outFile = filename # '/show/cdh_pub/shot/TEST/TEST_0120/ani/alien/_v099/alien_rig_GRP.root_con.usd'
    stage = SessionUtils.MakeInitialStage(outFile, clear=True, fr=fr, fps=fps, usdformat='usda')
    dprim = stage.DefinePrim('/root', 'Xform')
    stage.SetDefaultPrim(dprim)

    matrixs, frames = dxsXformUtils.Get4x4MatrixByXformCmd(nodeName, fr[0], fr[1], step=step)
    xformGeom = UsdGeom.Xform(dprim)
    for i in xrange(len(frames)):
        xformGeom.MakeMatrixXform().Set(Gf.Matrix4d(*matrixs[i]), Usd.TimeCode(frames[i]))

    stage.GetRootLayer().Save()

class RigXformExport(Arguments.CommonArgs):
    '''
    Rig World xform export

    Args:
        outDir (str):
        node   (str):
        fr   (tuple): (start, end)
        step (float):
        fps  (float):

    Returns:
        (str) : outfilename,
        (list): con list,
        (list): con attr list
    '''
    def __init__(self, outDir, node, **kwargs):
        self.outDir = outDir
        self.node   = node
        nsName, nodeName = dxsMayaUtils.GetNamespaceInfo(node)
        self.nsLayer = nsName.replace(':', '_')
        self.nodeName= nodeName

        Arguments.CommonArgs.__init__(self, **kwargs)

        self.mayafile = dxsMayaUtils.GetMayaFilename()
        if self.mayafile:
            self.comment = 'Generated with %s' % self.mayafile

    def doIt(self):
        attrList = CON_MAP['dexter']['attrs']
        conList  = CON_MAP['dexter']['nodes']
        rigConList = GetRigCons(self.node, conList)
        if not rigConList or cmds.getAttr("%s.rigType" % self.node) == 4:
            return None, None, None
        xformFile = '{DIR}/{NAME}.xform.usd'.format(DIR=self.outDir, NAME=self.nodeName)
        self.makeXformStage(xformFile, rigConList)
        return xformFile, rigConList, attrList

    def makeXformStage(self, outFile, nodes):
        stage = SessionUtils.MakeInitialStage(outFile, clear=True, comment=self.comment, fr=self.fr, fps=self.fps, usdformat=self.usdformat)
        dprim = stage.DefinePrim('/root', 'Xform')
        stage.SetDefaultPrim(dprim)

        matrixs, frames = dxsXformUtils.Get4x4MatrixByXformCmd(nodes[-1], self.fr[0], self.fr[1], step=self.step)
        xformGeom = UsdGeom.Xform(dprim)
        for i in xrange(len(frames)):
            xformGeom.MakeMatrixXform().Set(Gf.Matrix4d(*matrixs[i]), Usd.TimeCode(frames[i]))

        # initScale
        for at in InitScaleAttributes:
            if cmds.attributeQuery(at, n=nodes[0], ex=True):
                scaleOp = xformGeom.AddScaleOp()
                for i in xrange(len(frames)):
                    initScale = cmds.getAttr('%s.%s' % (nodes[0], at), time = frames[i])
                    scaleOp.Set(Gf.Vec3f(initScale, initScale, initScale), Usd.TimeCode(frames[i]))

        stage.GetRootLayer().Save()




class RigGeomExport:
    '''
    If using rigAssetDir, separate geom file by frameRange is default.
    Args:
        node (str): dxRig
        rigAssetDir (str): attribute file is referencing by this path.

    Returns:
        (dict) {
            'high': filename, 'mid': filename, 'low': filename
        }
        if not using mid meshes, only export 'high'
    '''
    def __init__(self, geomDir, node, fr=(None, None), step=1.0, fps=24.0, rigAssetDir=None, rigInspector=None):
        self.geomDir = geomDir
        self.node    = node
        self.nsLayer, self.nodeName = dxsMayaUtils.GetNamespaceInfo(node)

        self.fr  = fr
        self.step= step
        self.fps = fps
        self.rigAssetDir  = rigAssetDir
        self.rigInspector = rigInspector

        self.Objects = dict()
        if self.rigInspector:
            self.Objects = self.rigInspector.Objects
        else:
            for gt in ['high', 'mid', 'low', 'sim']:
                self.Objects[gt] = GetRigObjects(self.node, gt)

    @staticmethod
    def GetOverrideCommand(dir):    # rigAssetDir
        splitPath = dir.split('/')
        if "assetlib" in splitPath:
            showDir = "/".join(splitPath[:splitPath.index('assetlib') + 2])
            assetName = splitPath[splitPath.index('asset') + 1]
        elif "show" in splitPath:
            showDir   = '/'.join(splitPath[:splitPath.index('show') + 2])
            assetName = splitPath[splitPath.index('asset') + 1]
        else:
            MsgSender.sendMsg("[USD OverrideCommand] : file directory is not show or assetlib \n >> File Path: %s\n" % dir)
            return None

        # old project ended, use under line
        # ovrcmd = {'exportUVs': False}
        ovrcmd = {'exportUVs': True}
        overdata  = PathUtils.GetConfig(showDir, 'overrideCommand.json')
        if overdata and overdata.has_key(assetName):
            return overdata[assetName]
        else:
            return ovrcmd


    def doIt(self):
        result = dict()

        objects = list()
        for gt in self.Objects:
            objects += self.Objects[gt]
        dxsMayaUtils.UpdateTextureAttributes(cmds.ls(list(set(objects)), dag=True, type='surfaceShape'), asset=cmds.getAttr('%s.assetName' % self.node))

        # high
        if self.Objects['high']:
            geomFile = self.exportGeom('high', 'low')
            result['high'] = geomFile
            # Extra Rigs
            if self.rigInspector:
                if self.rigInspector.extraRignodes and self.rigInspector.extraObjects:
                    self.extraRigGeomExport(geomFile)
        # for LOD
        if self.Objects['mid']:
            result['mid'] = self.exportGeom('mid', 'low')
            if self.Objects['low']:
                result['low'] = self.exportGeom('low', None)
        return result


    def computeSeparateFrames(self, geomFile, objects):
        '''
        Return:
            (list) [(1001, 1050), (1051, 1100), ...]
        '''
        baseName = os.path.basename(geomFile)
        refSize  = os.path.getsize(os.path.join(self.rigAssetDir, baseName))
        refSize  = refSize / (1000.0 * 1000.0 * 1000.0) # GB

        numframes = self.fr[1] - self.fr[0] + 1
        deformed  = cmds.ls(objects, dag=True, s=True, io=True)

        totalSize = refSize * (float(len(deformed)) / float(len(objects))) * numframes
        limitSize = 10.0 # GB

        if totalSize > limitSize:
            framesPerFile = int(numframes / (totalSize / limitSize))
            frames = list()
            for frame in range(self.fr[0], self.fr[1] + 1, framesPerFile):
                endFrame = frame + (framesPerFile - 1)
                if endFrame >= self.fr[1]:
                    endFrame = self.fr[1]
                frames.append((frame, endFrame))
            dxsMsg.Print('info', "[RigGeomExport.computeSeparateFrames] estimate total size : %.2fGB, per file size : %.2fGB, frames: %s" % (totalSize, refSize * framesPerFile, frames))
            return frames

    def exportGeom(self, render, proxy):
        geomFile = '{DIR}/{NAME}.{TYPE}_geom.usd'.format(DIR=self.geomDir, NAME=self.nodeName, TYPE=render)

        renderMeshes = self.Objects[render]
        proxyMeshes  = list()
        if proxy:
            proxyMeshes = self.Objects[proxy]
        SetRigPurposeAttribute(renderMeshes, proxyMeshes)

        exportObjects = renderMeshes + proxyMeshes
        if render == 'high':
            exportObjects += self.Objects['sim']

        # 20190723 daeseok.chae : if place_CON has LOD_type attr, export when selected variant
        pConName = "place_CON" if not self.nsLayer else self.nsLayer + ':' + "place_CON"
        if cmds.objExists(pConName) and cmds.attributeQuery("LOD_type", n=pConName, ex=True):
            lodTypeList = cmds.attributeQuery("LOD_type", n=pConName, listEnum=True)
            if lodTypeList:
                lodList = lodTypeList[0].split(":")
                attrs = cmds.listConnections('%s.LOD_type' % pConName, p=True, s=True, d=False, type='animCurve')
                if attrs:
                    cmds.disconnectAttr(attrs[0], '%s.LOD_type' % pConName)
                    print "# DEBUG : disconnect LOD_TYPE"
                cmds.setAttr("%s.LOD_type" % pConName, lodList.index(render))
                print "# Debug : Selected LOD : %s" % render

        isSeq = False
        if (self.fr[0] != None and self.fr[1] != None) and (self.fr[0] != self.fr[1]) and self.rigAssetDir:
            isSeq = self.computeSeparateFrames(geomFile, exportObjects)

        if self.rigAssetDir:    # shot export
            geomClass = GeomMain.Export(geomFile, exportObjects, fr=self.fr, step=self.step, isSeq=isSeq, userAttr=True, nsLayer=self.nsLayer, rigNode=self.node)
            geomClass.excludeMeshAttributes = ['txVersion', 'txBasePath', 'txLayerName', 'txmultiUV', 'txAssetName'] # never using - txAssetName
            # geomClass.overrideCommand = {'exportUVs': False}
            geomClass.overrideCommand = RigGeomExport.GetOverrideCommand(self.rigAssetDir)
            geomClass.doIt()
            self.setAttrInherit(geomFile)
        else:   # asset export
            geomClass = GeomMain.Export(geomFile, exportObjects, userAttr=True, mtlAttr=True, nsLayer=self.nsLayer, rigNode=self.node)
            geomClass.excludeMeshAttributes = ['txVarNum', 'txVersion', 'txAssetName'] # never using - txAssetName
            geomClass.doIt()
            Attributes.ExtractAttr(geomFile, st=True).doIt()
        return geomFile


    def extraRigGeomExport(self, baseGeomFile):
        geomFile = '{DIR}/extra_rig_GRP.high_geom.usd'.format(DIR=os.path.dirname(baseGeomFile))
        SetRigPurposeAttribute(self.rigInspector.extraObjects['high'], self.rigInspector.extraObjects['low'])
        objects = self.rigInspector.extraObjects['high'] + self.rigInspector.extraObjects['low']
        geomClass = GeomMain.Export(geomFile, objects, fr=self.fr, step=self.step, userAttr=True, nsLayer=None, isAbc=False)
        geomClass.excludeMeshAttributes = ['txVersion', 'txBasePath', 'txLayerName', 'txmultiUV']
        geomClass.doIt()

        # Edit Proc - remove namespace
        stage = Usd.Stage.Open(geomFile, load=Usd.Stage.LoadNone)
        edit  = Sdf.BatchNamespaceEdit()
        for n in self.rigInspector.extraRignodes:
            ns, name = dxsMayaUtils.GetNamespaceInfo(n)
            rootPath = n.replace('|', '/').replace(':', '_')
            rootPrim = stage.GetPrimAtPath(rootPath)
            treeIter = iter(Usd.PrimRange.PreAndPostVisit(rootPrim))
            for p in treeIter:
                if treeIter.IsPostVisit() and p != rootPrim:
                    primPath = p.GetPath()
                    primName = primPath.name
                    newName  = primName.replace(ns + '_', '', 1)
                    newPath  = primPath.GetParentPath().AppendChild(newName)
                    edit.Add(primPath.pathString, newPath.pathString)
        stage.GetRootLayer().Apply(edit)
        stage.GetRootLayer().Save()

        # Update baseGeomFile
        stage = Usd.Stage.Open(baseGeomFile, load=Usd.Stage.LoadNone)
        for n in self.rigInspector.extraRignodes:
            rigPath = n.replace('|', '/').replace(':', '_')
            targetNode = cmds.getAttr('%s.targetNode' % n)
            pathStr    = targetNode.replace('|', '/')
            if self.nsLayer:
                pathStr= targetNode.replace('|%s:' % self.nsLayer, '/')
            prim = stage.GetPrimAtPath(pathStr)
            if not prim:
                continue
            # Remove Properties
            if prim.HasProperty('visibility'):
                prim.RemoveProperty('visibility')
            for attr in prim.GetPropertiesInNamespace('xformOp'):
                prim.RemoveProperty(attr.GetName())
            if prim.HasProperty('xformOpOrder'):
                prim.RemoveProperty('xformOpOrder')
            # Set Reference
            prim.GetReferences().ClearReferences()
            prim.GetReferences().AddReference(
                Sdf.Reference(PathUtils.GetRelPath(baseGeomFile, geomFile), Sdf.Path(rigPath))
            )
        stage.GetRootLayer().Save()

    def setAttrInherit(self, geomFile):
        attrFile = '{DIR}/{BASENAME}'.format(DIR=self.rigAssetDir, BASENAME=os.path.basename(geomFile).replace('_geom.usd', '_attr.usd'))
        attrLayer = Sdf.Layer.FindOrOpen(attrFile)
        if not attrLayer:
            print '# Error : not found rigAsset Attribute File.'
            return
        stage = Usd.Stage.Open(geomFile, load=Usd.Stage.LoadNone)
        dprim = stage.GetDefaultPrim()
        outLayer = stage.GetRootLayer()
        SessionUtils.InsertSubLayer(outLayer, PathUtils.GetRelPath(geomFile, attrFile))

        attrPath = attrLayer.rootPrims[0].path
        dprim.GetInherits().AddInherit(attrPath)

        outLayer.Save()



class RigAssetExport(Arguments.AssetArgs):
    def __init__(self, node=None, **kwargs):
        dxsMayaUtils.PluginSetup(['backstageMenu', 'pxrUsd'])

        Arguments.AssetArgs.__init__(self, **kwargs)
        if not self.outDir and self.assetDir:
            self.outDir = self.assetDir + '/rig'

        self.node   = ''
        exportNodes = GetRigNodes(selected=node)
        if exportNodes:
            self.node = exportNodes[0]

        self.mayafile = dxsMayaUtils.GetMayaFilename()
        self.comment  = 'Generated with %s' % self.mayafile

    def doIt(self):
        assert self.mayafile, '[EXPORT] - Save current scene.'
        if not self.node:
            return

        self.sceneName = os.path.splitext(os.path.basename(self.mayafile))[0]
        self.outDir += '/usd/%s' % self.sceneName

        fileData = RigGeomExport(self.outDir, self.node).doIt()

        masterFile = self.makeGeomPackage(fileData)
        self.makeRigPackage(masterFile)


    def makeGeomPackage(self, geomFileData):
        nodeName = self.node.split('|')[-1].split(':')[-1]
        masterFile = '{DIR}/{NAME}.usd'.format(DIR=self.outDir, NAME=nodeName)

        SdfPath = '/' + nodeName
        isLod = True if len(geomFileData.keys()) > 1 else False
        for gtype in geomFileData:
            if isLod:
                SdfPath += '{lodVariant=%s}' % gtype
            SessionUtils.MakeReferenceStage(masterFile, [(geomFileData[gtype], None)], SdfPath=SdfPath, comment=self.comment)
        if isLod:
            dxsUsdUtils.LodVariantDefaultSelection(masterFile)

        # Set Materials
        Material.Main(masterFile)

        masterPayloadFile = masterFile.replace('.usd', '.payload.usd')
        SessionUtils.MakeReferenceStage(masterPayloadFile, [(masterFile, None)], SdfPath='/%s{rigVersion=%s}' % (nodeName, self.sceneName), comment=self.comment)
        return masterPayloadFile

    def makeRigPackage(self, sourceFile):
        taskFile = '{DIR}/rig/rig.usd'.format(DIR=self.assetDir)
        SessionUtils.MakeSubLayerStage(taskFile, [sourceFile])
        taskPayloadFile = taskFile.replace('.usd', '.payload.usd')
        SessionUtils.MakeReferenceStage(taskPayloadFile, [(taskFile, None)], SdfPath='/%s{taskVariant=rig}' % self.assetName, clear=True)

        if self.showDir:
            PackageUtils.AssetPackage(self.showDir, self.assetName, taskPayloadFile)




class RigShotExport(Arguments.ShotArgs):
    def __init__(self, node=None, rigUpdate=False, **kwargs):
        dxsMayaUtils.PluginSetup(['backstageMenu', 'pxrUsd'])

        # Member Variables
        self.assetName  = None
        self.rigAssetDir= None
        self.autofr = False
        self.expfr  = None
        self.refFile= None

        Arguments.ShotArgs.__init__(self, **kwargs)
        if not self.outDir and self.shotDir:
            self.outDir = self.shotDir + '/ani'

        self.fps= dxsMayaUtils.GetFPS()
        # Frame Range
        if not self.fr[0] or not self.fr[1]:
            self.fr = dxsMayaUtils.GetFrameRange()
            self.autofr = True

        self.mayafile = dxsMayaUtils.GetMayaFilename()
        self.comment = 'Generated with %s' % self.mayafile

        self.node = ''
        exportNodes = GetRigNodes(selected=node)
        if exportNodes:
            self.node = exportNodes[0]

        if self.node:
            self.refFile = RigShotExport.RigReferenceRepresent(self.node)
            if rigUpdate:
                self.refFile = RigShotExport.RigLatestVersionRepresent(self.node)

            self.nsLayer, self.nodeName = dxsMayaUtils.GetNamespaceInfo(self.node)
            self.outDir += '/' + self.nsLayer
            self.computeVersion()


    @staticmethod
    def RigReferenceRepresent(node):
        '''
        only low -> high rig represent
        '''
        if not cmds.referenceQuery(node, inr=True):
            return
        rfFile = cmds.referenceQuery(node, f=True, wcn=True)
        if '_low' in rfFile:
            newFile = rfFile.replace('_low', '')
            if os.path.exists(newFile):
                dxsMsg.Print('info', "[RigReferenceRepresent] %s -> %s" % (os.path.basename(rfFile), os.path.basename(newFile)))
                rfNode = cmds.referenceQuery(node, rfn=True)
                cmds.file(newFile, loadReference=rfNode)
                return newFile

    @staticmethod
    def RigLatestVersionRepresent(node):
        if not cmds.referenceQuery(node, inr=True):
            return
        rfFile = cmds.referenceQuery(node, f=True, wcn=True)
        rfDir  = os.path.dirname(rfFile)
        cvFile = os.path.basename(rfFile)
        rigFiles = list()
        # excludePurpose = ['_low', '_sim', '_mid', '_mocap']
        for fn in sorted(os.listdir(rfDir)):
            if '.mb' in fn and not fn.startswith('.'):
                if fn.split('_')[-2] == 'rig':
                # check = 0
                # for purpose in excludePurpose:
                #     if not purpose in fn:
                #         check += 1
                # if check == len(excludePurpose):
                    rigFiles.append(fn)

        rigFiles.sort(reverse=True)

        if rigFiles[0] == cvFile:
            return

        dxsMsg.Print('info', "[RigLatestVersionRepresent] %s -> %s" % (cvFile, rigFiles[0]))

        newFile = os.path.join(rfDir, rigFiles[0])
        rfNode  = cmds.referenceQuery(node, rfn=True)
        cmds.file(newFile, loadReference=rfNode)
        return newFile


    def preCompute(self):
        RigInspector = RigNodeInspect(self.node, self.fr, self.autofr)
        RigInspector.bake()
        self.expfr = RigInspector.exportRange
        return RigInspector


    def doIt(self):
        if not self.mayafile:
            dxsMsg.Print('warning', "[Must have to save current scene]")
            return
        if not self.node:
            return

        self.assetName, self.rigAssetDir = GetRigAssetDir(self.node, showDir=self.showDir)
        if self.rigAssetDir and os.path.exists(self.rigAssetDir):
            dxsMsg.Print('info', "[RigShotExport.rigAssetDir] %s - %s" % (self.assetName, self.rigAssetDir))
        else:
            MsgSender.sendMsg("[USD RigAsset] : First, you have to export RigAsset\n >> Rig File : %s\n >> RigAssetDir : %s" % (self.refFile, self.rigAssetDir), self.showName, self.shotName, self.user)

        RigInspector = self.preCompute()

        geomDir = '{DIR}/{VER}'.format(DIR=self.outDir, VER=self.version)
        xformFile, rigConList, attrList = RigXformExport(geomDir, self.node, fr=self.expfr, step=self.step, fps=self.fps, usdformat='usdc').doIt()


        if RigInspector.rigRootCon:
            filename = '{DIR}/{NAME}.root_con.usd'.format(DIR=geomDir, NAME=self.nodeName)
            rootConName = '%s:%s' % (self.nsLayer, RigInspector.rigRootCon)
            RigRootConExport(filename, rootConName, self.expfr, self.step, self.fps)

        muteClass = None
        if rigConList and attrList and xformFile:
            muteClass = MuteCtrl(rigConList, attrList)
            muteClass.getVal()
            muteClass.setMute()

        fileData = RigGeomExport(geomDir, self.node, fr=self.expfr, step=self.step, fps=self.fps, rigAssetDir=self.rigAssetDir, rigInspector=RigInspector).doIt()

        if muteClass:
            muteClass.setUnMute()

        masterFile = self.makeGeomPackage(xformFile, fileData)
        self.makePackage(masterFile)
        return masterFile


    def makeGeomPackage(self, xformFile, geomFileData):
        masterFile = '{DIR}/{VER}/{NAME}.usd'.format(DIR=self.outDir, VER=self.version, NAME=self.nsLayer)

        # WorldXform
        if xformFile:
            SessionUtils.MakeReferenceStage(masterFile, [(None, None)], SdfPath='/%s{WorldXform=off}' % self.nsLayer, fr=self.expfr, fps=self.fps)
            SessionUtils.MakeReferenceStage(masterFile, [(xformFile, None)], SdfPath='/%s{WorldXform=on}' % self.nsLayer, fr=self.expfr, fps=self.fps)

        customLayerData = {
            'start': int(self.fr[0]), 'end': int(self.fr[1]),
            'asset': self.assetName,
            'rigAssetDir': self.rigAssetDir
        }
        customPrimData = {
            'rig': os.path.basename(self.rigAssetDir),
            'ani': os.path.basename(self.mayafile)
        }

        SdfPath = '/' + self.nsLayer
        isLod = True if len(geomFileData.keys()) > 1 else False
        for gtype in geomFileData:
            if isLod:
                SdfPath += '{lodVariant=%s}' % gtype
            SessionUtils.MakeReferenceStage(
                masterFile, [(geomFileData[gtype], None)], SdfPath=SdfPath, fr=self.expfr, fps=self.fps,
                comment=self.comment, customLayerData=customLayerData, customPrimData=customPrimData
            )
        if isLod:
            dxsUsdUtils.LodVariantDefaultSelection(masterFile)

        # Load Materials
        target = '{DIR}/collection.usd'.format(DIR=self.rigAssetDir)
        Material.CompositeCollection(masterFile, target)

        masterPayloadFile = masterFile.replace('.usd', '.payload.usd')
        SessionUtils.MakeReferenceStage(masterPayloadFile, [(masterFile, None)], SdfPath='/%s{aniVersion=%s}' % (self.nsLayer, self.version), fr=self.fr, fps=self.fps, comment=self.comment)
        return masterPayloadFile


    def makePackage(self, sourceFile):
        layerFile = '{DIR}/{NAME}.usd'.format(DIR=self.outDir, NAME=self.nsLayer)
        SessionUtils.MakeSubLayerStage(layerFile, [sourceFile])

        layerPayloadFile = layerFile.replace('.usd', '.payload.usd')
        SessionUtils.MakeReferenceStage(layerPayloadFile, [(layerFile, None)], SdfPath='/rig/%s' % self.nsLayer, Kind='assembly', clear=True)

        aniFile = '{DIR}/ani/ani.usd'.format(DIR=self.shotDir)
        SessionUtils.MakeSubLayerStage(aniFile, [layerPayloadFile])

        aniPayloadFile = aniFile.replace('.usd', '.payload.usd')
        SessionUtils.MakeReferenceStage(aniPayloadFile, [(aniFile, None)], SdfPath='/shot/rig', Name=self.shotName, Kind='assembly', clear=True)

        PackageUtils.ShotPackage(self.showDir, self.seqName, self.shotName, aniPayloadFile, fr=self.fr, fps=self.fps)
        self.overrideVersion()

    def overrideVersion(self):
        shotFile = '{DIR}/shot/{SEQ}/{SHOT}/{SHOT}.usd'.format(DIR=self.showDir, SEQ=self.seqName, SHOT=self.shotName)
        PackageUtils.VersionSelect(shotFile, '/shot/rig/' + self.nsLayer, 'aniVersion', self.version)

        shotLgtFile = shotFile.replace('.usd', '.lgt.usd')
        if os.path.exists(shotLgtFile):
            PackageUtils.VersionSelect(shotLgtFile, '/shot/rig/' + self.nsLayer, 'aniVersion', self.version)



class RigClipExport(Arguments.AssetArgs):
    '''
    dxRig Clip Export
    Args:
        node (str): export dxRig node name
    '''
    def __init__(self, node=None, fr=(0, 0), step=1.0, loopScales=[1.0], loopRange=(0, 0), overWrite=True, **kwargs):
        dxsMayaUtils.PluginSetup(['backstageMenu', 'pxrUsd'])
        self.overWrite = overWrite

        Arguments.AssetArgs.__init__(self, **kwargs)
        if not self.outDir and self.assetDir:
            self.outDir = '{DIR}/clip'.format(DIR=self.assetDir)
        self.computeVersion()

        self.fps = dxsMayaUtils.GetFPS()
        # Frame Range
        self.fr  = fr
        self.step= step
        if not self.fr[0] or not self.fr[1]:
            self.fr = dxsMayaUtils.GetFrameRange()

        self.loopScales = loopScales
        self.loopRange  = loopRange

        self.mayafile = dxsMayaUtils.GetMayaFilename()
        if self.mayafile:
            self.comment = 'Generated with %s' % self.mayafile

        # Member variables
        self.node = None
        self.clipLayer = None

        self.node = ''
        exportNodes = GetRigNodes(selected=node)
        if exportNodes:
            self.node = exportNodes[0]

    def doIt(self):
        if not self.node:
            return

        self.assetName, self.rigAssetDir = GetRigAssetDir(self.node, showDir=self.showDir)
        self.rigCollection = '{DIR}/collection.usd'.format(DIR=self.rigAssetDir)

        ns, nn = dxsMayaUtils.GetNamespaceInfo(self.node)
        self.clipLayer = 'default' if not ns else ns
        self.clipLayer+= '_clip'
        self.nodeName  = nn

        geomDir  = '{DIR}/{VER}/{LAYER}'.format(DIR=self.outDir, VER=self.version, LAYER=self.clipLayer)
        fileData = RigGeomExport(geomDir, self.node, fr=(self.fr[0]-1, self.fr[1]+1), step=self.step, fps=self.fps, rigAssetDir=self.rigAssetDir).doIt()

        geomMasterFile = self.makeGeomPackage(fileData)

        loopClips = None
        if self.loopScales:
            loopClips = ClipUtils.LoopClip(geomMasterFile, scales=self.loopScales, fr=self.loopRange).doIt()
        if not loopClips:
            return

        taskPayloadFile = self.makeLoopClipPackage(loopClips)
        if self.showDir and self.assetDir:
            PackageUtils.AssetPackage(self.showDir, self.assetName, taskPayloadFile)


    def makeGeomPackage(self, geomFileData):
        # Add Attributes
        # for gtype in geomFileData:
        #     geomFile = geomFileData[gtype]
        #     RigShotExport.SetInherit(self.rigAssetDir, geomFile)

        masterFile = '{DIR}/{VER}/{LAYER}/{LAYER}.usd'.format(DIR=self.outDir, VER=self.version, LAYER=self.clipLayer)
        SdfPath = '/' + self.assetName
        isLod = True if len(geomFileData.keys()) > 1 else False
        for gtype in geomFileData:
            if isLod:
                SdfPath += '{lodVariant=%s}' % gtype
            SessionUtils.MakeReferenceStage(
                masterFile, [(geomFileData[gtype], None)], SdfPath=SdfPath, fr=self.fr, fps=self.fps, comment=self.comment
            )
        if isLod:
            dxsUsdUtils.LodVariantDefaultSelection(masterFile)

        # Load Materials
        Material.CompositeCollection(masterFile, self.rigCollection)

        return masterFile


    def makeLoopClipPackage(self, sourceFiles):
        # Load Materials
        for f in sourceFiles:
            Material.CompositeCollection(f, self.rigCollection)
            Material.CompositeCollection(f.replace('.payload.', '.preview.'), self.rigCollection)

        loopClip = '{DIR}/{VER}/loopClip.usd'.format(DIR=self.outDir, VER=self.version)
        loopLayer= self.clipLayer.replace('clip', 'loop')
        SessionUtils.MakeSubLayerStage(loopClip, sourceFiles, SdfPath='/%s{loopVariant=%s1_0}' % (self.assetName, loopLayer))
        loopClipPayload = loopClip.replace('.usd', '.payload.usd')
        SessionUtils.MakeReferenceStage(loopClipPayload, [(loopClip, None)], SdfPath='/%s{clipVersion=%s}' % (self.assetName, self.version))

        taskFile = '{DIR}/clip.usd'.format(DIR=self.outDir)
        SessionUtils.MakeSubLayerStage(taskFile, [loopClipPayload])
        taskPayloadFile = taskFile.replace('.usd', '.payload.usd')
        SessionUtils.MakeReferenceStage(taskPayloadFile, [(taskFile, None)], SdfPath='/%s{taskVariant=clip}' % self.assetName, clear=True)
        return taskPayloadFile
