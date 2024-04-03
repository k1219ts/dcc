## -*- coding: utf-8 -*-
'''
USD Camera export

Script Command
> dxsUsd.CameraShotExport(...).doIt()

Outputs
shot / $SEQ / $SHOT / cam / cam.payload.usd
                          / cam.usd
                          / $VER / camera.payload.usd
                                 / camera.usd
                                 / $NODE.geom.usd
                                 / $NODE_$imagePlane.imp.usd
                                 / $NODE_$imagePlane.imp.abc
'''
import os
import string
import math
import time
import getpass
import json

from pxr import Usd, UsdGeom, Sdf, UsdShade, Gf, Vt
import maya.cmds as cmds
import maya.mel as mel

import MsgSender
import dxsMsg
import Arguments
import dxsUsdUtils
import dxsMayaUtils
import dxsXformUtils
import SessionUtils
import PackageUtils

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


def GetCameraNodes(selected=None):
    '''
    Returns:
        (list) : camera transform list
    '''
    _specify = True
    if not selected:
        selected = cmds.ls(type='dxCamera')
        _specify = False
    nodes = cmds.ls(selected, dag=True, type='camera', l=True)
    result = list()
    hasRenderable = 0
    for n in nodes:
        if cmds.getAttr('%s.renderable' % n):
            hasRenderable += 1
        result += cmds.listRelatives(n, p=True, f=True)

    if not _specify:
        if not hasRenderable:
            return list()
    return result


# TODO:
class PanZoom:
    def __init__(self, node, frames):
        self.node  = node
        self.frames= frames

        keyData = dict()
        additiveData = dict()
        shakeData    = dict()

        shapeSourceH = cmds.listConnections(node + '.horizontalPan', s=True, d=False)
        shapeSourceV = cmds.listConnections(node + '.verticalPan', s=True, d=False)
        if shapeSourceH:
            if cmds.nodeType(shapeSourceH[0]) == 'animBlendNodeAdditive':
                shakeDataH = self.getKeyData(shapeSourceH[0], 'inputB')
                shakeDataV = self.getKeyData(shapeSourceV[0], 'inputB')
                panDataH = self.getKeyData(shapeSourceH[0], 'inputA')
                panDataV = self.getKeyData(shapeSourceV[0], 'inputA')
                zomData = self.getKeyData(node, 'zom')
                additiveData = {'hpn': panDataH, 'vpn': panDataV, 'zom': zomData}
                shakeData    = {'horizontal': shakeDataH, 'vertical': shakeDataV}

        if additiveData:
            keyData = additiveData
        else:
            keyData['hpn'] = self.getKeyData(node, 'hpn')
            keyData['vpn'] = self.getKeyData(node, 'vpn')
            keyData['zom'] = self.getKeyData(node, 'zom')

        self.keyData = keyData
        self.shakeData = shakeData

    def exportPanZoomData(self, filename, user):
        if self.keyData:
            body = dict()
            body["_Header"] = {"created" : time.asctime(),
                               "author" : getpass.getuser(),
                               "context" : cmds.file(q=True, sn=True)}
            body["2DPanZoom"] = self.keyData

            with open(filename, "w") as f:
                json.dump(body, f, indent = 4)

    def exportNukeCamShake(self, filename):
        hfa = cmds.getAttr('%s.hfa' % self.node)

        # Nuke Camera
        nukeNode = 'Camera {\n'
        nukeNode += '  inputs 0\n'
        # horizontalPan
        if self.shakeData.has_key('horizontal'):
            tx = '{curve'
            for index, f in enumerate(self.shakeData['horizontal']['frames']):
                # attr = panzoomShakeAttr.get('attr', '.hpn')
                value = self.shakeData['horizontal']['value'][index]
                tx += ' x%d' % f
                tx += ' %.8f' % (value / hfa * 2)
            tx += '}'
        else:
            value = cmds.getAttr(self.node + '.hpn')
            tx = '%.8f' % (value / hfa * 2)
        # verticalPan
        if self.shakeData.has_key('vertical'):
            ty = '{curve'
            for index, f in enumerate(self.shakeData['vertical']['frames']):
                # attr = panzoomShakeAttr.get('attr', '.vpn')
                value = self.shakeData['vertical']['value'][index]
                ty += ' x%d' % f
                ty += ' %.8f' % (value / hfa * 2)
            ty += '}'
        else:
            value = cmds.getAttr(self.node + '.vpn')
            ty = '%.8f' % (value / hfa * 2)
        nukeNode += '  win_translate {%s %s}\n' % (tx, ty)
        # zoom
        sc = '%.8f' % cmds.getAttr('%s.zom' % self.node)

        nukeNode += '  win_scale {%s %s}\n' % (sc, sc)
        nukeNode += '  name {0}_Shake\n'.format(self.node)
        nukeNode += '}\n'

        f = open(filename, "w")
        f.write(nukeNode)
        f.close()

    def exportNuke2DPanZoom(self, filename, user):
        # cameraName = cmds.listRelatives(self.node, p=True)[0]
        cameraName = self.node
        hfa = cmds.getAttr('%s.hfa' % self.node)

        # Nuke Camera
        nukeNode = 'Camera {\n'
        nukeNode += '  inputs 0\n'

        # horizontalPan
        connections = cmds.listConnections('%s.hpn' % self.node, type='animCurve')
        if connections:
            tx = '{curve'
            for f in self.frames:
                value = cmds.getAttr('%s.hpn' % self.node, t=f)
                tx += ' x%d' % f
                tx += ' %.8f' % (value / hfa * 2)
            tx += '}'
        else:
            value = cmds.getAttr('%s.hpn' % self.node)
            tx = '%.8f' % (value / hfa * 2)

        # verticalPan
        connections = cmds.listConnections('%s.vpn' % self.node, type='animCurve')
        if connections:
            ty = '{curve'
            for f in self.frames:
                value = cmds.getAttr('%s.vpn' % self.node, t=f)
                ty += ' x%d' % f
                ty += ' %.8f' % (value / hfa * 2)
            ty += '}'
        else:
            value = cmds.getAttr('%s.vpn' % self.node)
            ty = '%.8f' % (value / hfa * 2)

        nukeNode += '  win_translate {%s %s}\n' % (tx, ty)

        # zoom
        connections = cmds.listConnections('%s.zom' % self.node, type='animCurve')
        if connections:
            sc = '{curve'
            for f in self.frames:
                value = cmds.getAttr('%s.zom' % self.node, t=f)
                sc += ' x%d' % f
                sc += ' %.8f' % value
            sc += '}'
        else:
            sc = '%.8f' % cmds.getAttr('%s.zom' % self.node)

        nukeNode += '  win_scale {%s %s}\n' % (sc, sc)

        nukeNode += '  name %s_2DPanZoom\n' % cameraName
        nukeNode += '}\n'

        f = open(filename, "w")
        f.write(nukeNode)
        f.close()

    def getKeyData(self, node, attr):
        data = {'frames': list(), 'value': list(), 'infinity': ['constant', 'constant']}
        # time wrap
        timeWrap = cmds.listConnections('time1', d=False, s=True)
        for f in self.frames:
            wrapframe = f
            if timeWrap:
                wrapframe = cmds.getAttr('%s.output' % timeWrap[0], time=f)
            value = cmds.getAttr(node + '.' + attr, t=wrapframe)
            data['frames'].append(f)
            data['value'].append(value)
        return data



def CreatePolyImagePlane(imagePlane):
    camShape = cmds.listRelatives(imagePlane, p=True, f=True)[0]
    camTrans = cmds.listRelatives(camShape, p=True, f=True)[0]

    camName   = camTrans.split('|')[-1].split(':')[-1]
    planeName = imagePlane.split('->')[-1].split('|')[-1].split(':')[-1]

    hfa = cmds.camera(camShape, q=True, hfa=True)
    vfa = cmds.camera(camShape, q=True, vfa=True)
    fov = math.radians(cmds.camera(camShape, q=True, hfv=True))
    aspect = vfa / hfa

    overscan = 1.0
    # temporary statement
    tmp_coverageX = cmds.getAttr('%s.coverageX' % imagePlane)
    if tmp_coverageX == 2420 or tmp_coverageX == 1210 or tmp_coverageX == 3168 or tmp_coverageX == 1584:
        overscan = 1.1

    pname = '%s_polyImagePlane_%s_tmp' % (camName, planeName)
    tmpPlane = cmds.polyPlane(name=pname, w=1.0, h=aspect, sx=1, sy=1, ax=(0, -1, 0), cuv=1, ch=1)[0]
    cmds.polyFlipUV(tmpPlane, ft=0)
    cmds.polyFlipUV(tmpPlane, ft=1)

    # for i in tmpPlaneList:
    tmpPlane = cmds.parent(tmpPlane, camTrans)[0]

    expressionString = "float $hfa = `camera -q -hfa {camera}`;\n"
    expressionString += "float $vfa = `camera -q -vfa {camera}`;\n"
    expressionString += "float $fov = `camera -q -hfv {camera}`;\n"
    expressionString += "float $bbminx = {imageplane}.boundingBoxMinX;\n"
    expressionString += "float $bbmaxx = {imageplane}.boundingBoxMaxX;\n"
    expressionString += "float $bbminy = {imageplane}.boundingBoxMinY;\n"
    expressionString += "float $bbmaxy = {imageplane}.boundingBoxMaxY;\n\n"
    expressionString += "{polyplane}.scaleX = {polyplane}.scaleY = {polyplane}.scaleZ = 2*({imageplane}.depth)*(tand($fov/2.0));\n\n"
    # (offset 비율) * (imageplane boundingbox 크기)
    expressionString += "{polyplane}.translateX = ({imageplane}.offsetX / $hfa) * ($bbmaxx - $bbminx);\n"
    expressionString += "{polyplane}.translateY = ({imageplane}.offsetY / $vfa) * ($bbmaxy - $bbminy);\n"
    expressionString += "{polyplane}.translateZ = -1*{imageplane}.depth;\n"
    expressionString += "{polyplane}.rotateZ = 180 + -1*{imageplane}.rotate;\n"

    newPlaneList = list()

    depth = cmds.getAttr('%s.depth' % imagePlane) * -1.0
    cmds.setAttr('%s.rx' % tmpPlane, -90.0)
    cmds.setAttr('%s.ry' % tmpPlane, 0.0)

    imagePlaneShape = cmds.listRelatives(imagePlane, s=True, p=False)[0]
    exprString = expressionString.format(
        camera=camTrans,
        imageplane=imagePlaneShape,
        polyplane=tmpPlane
    )
    cmds.expression(s=exprString, o=tmpPlane, ae=True, uc='all')
    cmds.refresh()
    newPlane = cmds.duplicate(tmpPlane, name=tmpPlane.replace('_tmp', ''))[0]
    cmds.parent(newPlane, world=True)
    return camName + '_' + planeName, newPlane, tmpPlane

class SetAttribute:
    def __init__(self, Attr, node, attrname, frames, postMult=1):
        self.Attr = Attr
        # time wrap
        timeWrap = cmds.listConnections('time1', d=False, s=True)

        if type(attrname).__name__ == 'list':
            _anim = 0
            objs  = list()
            for at in attrname:
                obj = node + '.' + at
                objs.append(obj)
                if not self.isConstant(obj):
                    _anim += 1
            if _anim > 0:
                for i in xrange(len(frames)):
                    f = frames[i]
                    val = list()
                    for o in objs:
                        wrapframe = f
                        if timeWrap:
                            wrapframe = cmds.getAttr('%s.output' % timeWrap[0], time=f)
                        val.append(cmds.getAttr(o, time=wrapframe) * postMult)
                    self.setValue(val, time=Usd.TimeCode(f))
            else:
                val = list()
                for o in objs:
                    val.append(cmds.getAttr(o) * postMult)
                self.setValue(val)
        else:
            obj = node + '.' + attrname
            # if self.isConstant(obj):
            #     self.setValue(cmds.getAttr(obj) * postMult)
            # else:
            for i in xrange(len(frames)):
                f = frames[i]
                wrapframe = f
                if timeWrap:
                    wrapframe = cmds.getAttr('%s.output' % timeWrap[0], time=f)
                self.setValue(cmds.getAttr(obj, time=wrapframe) * postMult, time=Usd.TimeCode(f))


    def isConstant(self, obj):
        animCurve = cmds.listConnections(obj, type='animCurve')
        expression= cmds.listConnections(obj, type='expression')
        if animCurve or expression:
            return False
        else:
            return True

    def setValue(self, value, time=Usd.TimeCode.Default()):
        if type(value).__name__ == 'list':
            if len(value) == 2:
                self.Attr.Set(Gf.Vec2f(*value), time=time)
            elif len(value) == 3:
                self.Attr.Set(Gf.Vec3f(*value), time=time)
            else:
                assert False, '# msg : error'
        else:
            self.Attr.Set(value, time=time)

# Compute FilmAperture, FilmOffset
class SetFilmAperture:
    def __init__(self, geom, node, frames):
        self.geom  = geom
        self.node  = node
        self.frames= frames
        self.times = list()

        timeWrap = cmds.listConnections('time1', d=False, s=True)
        for i in xrange(len(frames)):
            f = frames[i]
            wrapframe = f
            if timeWrap:
                wrapframe = cmds.getAttr('%s.output' % timeWrap[0], time=f)
            self.times.append(wrapframe)

        # 2D Pan/Zoom
        self.isPanZoom = False
        self.ZoomVals  = list()
        self.hpnVals = list()
        self.vpnVals = list()
        if cmds.getAttr('%s.renderPanZoom' % node) and cmds.getAttr('%s.panZoomEnabled' % node):
            self.isPanZoom = True
            for t in self.times:
                self.ZoomVals.append(cmds.getAttr('%s.zoom' % node, time=t))

                shapeSourceH = cmds.listConnections(node + '.horizontalPan', s=True, d=False)
                if shapeSourceH:
                    if cmds.nodeType(shapeSourceH[0]) == 'animBlendNodeAdditive':
                        self.hpnVals.append(cmds.getAttr('%s.inputA' % shapeSourceH[0], time=t))

                shapeSourceV = cmds.listConnections(node + '.verticalPan', s=True, d=False)
                if shapeSourceV:
                    if cmds.nodeType(shapeSourceV[0]) == 'animBlendNodeAdditive':
                        self.vpnVals.append(cmds.getAttr('%s.inputA' % shapeSourceV[0], time=t))

        self.postMult = 2.54 * 10

        self.horizontalFilmAperture()
        self.horizontalFilmOffset()
        self.verticalFilmAperture()
        self.verticalFilmOffset()

    def horizontalFilmAperture(self):
        attr = self.geom.GetHorizontalApertureAttr()
        for i in xrange(len(self.frames)):
            hfa = cmds.getAttr('%s.horizontalFilmAperture' % self.node, time=self.times[i])
            if self.isPanZoom:
                postMult = self.postMult * self.ZoomVals[i]
            else:
                postMult = self.postMult * (1 / cmds.getAttr('%s.postScale' % self.node, time=self.times[i]))
            attr.Set(hfa * postMult, time=Usd.TimeCode(self.frames[i]))

    def horizontalFilmOffset(self):
        attr = self.geom.GetHorizontalApertureOffsetAttr()
        for i in xrange(len(self.frames)):
            ln = '%s.horizontalFilmOffset' % self.node
            if self.isPanZoom:
                ln = '%s.horizontalPan' % self.node
            hfo = cmds.getAttr(ln, time=self.times[i])

            if self.hpnVals:
                hfo = self.hpnVals[i]

            attr.Set(hfo * self.postMult, time=Usd.TimeCode(self.frames[i]))

    def verticalFilmAperture(self):
        attr = self.geom.GetVerticalApertureAttr()
        for i in xrange(len(self.frames)):
            vfa = cmds.getAttr('%s.verticalFilmAperture' % self.node, time=self.times[i])
            if self.isPanZoom:
                postMult = self.postMult * self.ZoomVals[i]
            else:
                postMult = self.postMult * (1 / cmds.getAttr('%s.postScale' % self.node, time=self.times[i]))
            attr.Set(vfa * postMult, time=Usd.TimeCode(self.frames[i]))

    def verticalFilmOffset(self):
        attr = self.geom.GetVerticalApertureOffsetAttr()
        for i in xrange(len(self.frames)):
            ln = '%s.verticalFilmOffset' % self.node
            if self.isPanZoom:
                ln = '%s.verticalPan' % self.node
            vfo = cmds.getAttr(ln, time=self.times[i])

            if self.vpnVals:
                vfo = self.vpnVals[i]
            attr.Set(vfo * self.postMult, time=Usd.TimeCode(self.frames[i]))




class CameraGeomExport(Arguments.CommonArgs):
    '''
    UsdGeomCamera and AlembicCamera Export

    Args:
        node (str): camera transform name
    '''
    def __init__(self, filename, node, **kwargs):
        self.filename = filename
        self.node     = node
        Arguments.CommonArgs.__init__(self, **kwargs)

        self.show = ""
        self.shot = ""
        self.user = ""
        if kwargs.has_key("show"):
            self.show = kwargs["show"]

        if kwargs.has_key("shot"):
            self.shot = kwargs["shot"]

        if kwargs.has_key("user"):
            self.user = kwargs["user"]

    def doIt(self):
        # 2019.04.24 camera export when unlock and bake simulation.
        self.preCompute()

        self.usdExport()
        self.abcExport()


    def preCompute(self):
        # first lock check keyable(in Channel Box)
        shape = cmds.listRelatives(self.node, c=True, type='camera', fullPath=True)[0]

        lockAttr = cmds.listAttr(self.node, locked=True)
        if lockAttr:
            for attr in lockAttr:
                self.unlockAttr("%s.%s" % (self.node, attr))

        lockAttr = cmds.listAttr(shape, locked=True)
        if lockAttr:
            for attr in lockAttr:
                self.unlockAttr("%s.%s" % (shape, attr))

        bakeOnOverrideLayer = False
        animLayer = cmds.animLayer(q=True, root=True)
        if animLayer:
            if cmds.animLayer(animLayer, q=True, children=True):
                bakeOnOverrideLayer = True

        # retime bake
        if self.fr[0] != self.fr[1]:
            bakeRange = [self.fr[0], self.fr[1]]
            orgCurrentTime = cmds.currentTime(q=True)
            isRetime = False
            if cmds.getAttr("time1.enableTimewarp"):
                isRetime = True
                cmds.currentTime(self.fr[1])
                bakeRange[1] = int(cmds.getAttr("time1.outTime")) + 1
                cmds.setAttr("time1.enableTimewarp", False)

            # second bake simulation start - 1 ~ end + 1
            cmds.bakeResults(self.node, simulation=True, t=tuple(bakeRange), bol=bakeOnOverrideLayer, dic=False, pok=False, shape=True)

            SelectAnimLayer()

            # retime undo
            if isRetime:
                cmds.setAttr("time1.enableTimewarp", True)
                cmds.currentTime(orgCurrentTime)

    def unlockAttr(self, plugName):
        lockedPlug = cmds.connectionInfo(plugName, gla=True)
        if lockedPlug:
            cmds.setAttr(lockedPlug, lock=False)
            return self.unlockAttr(plugName)
        else:
            return None

    def usdExport(self):
        name = self.node.split('|')[-1].split(':')[-1]
        stage = SessionUtils.MakeInitialStage(self.filename, clear=True, usdformat=self.usdformat, fr=self.fr, fps=self.fps, comment=self.comment)
        prim  = stage.DefinePrim('/' + name, 'Camera')
        stage.SetDefaultPrim(prim)
        dxsUsdUtils.SetModelAPI(prim, kind='camera', name=name)

        frames = self.getFrames()

        geom = UsdGeom.Camera(prim)
        # Near, Far Clip Plane
        clipAttr = geom.GetClippingRangeAttr()
        SetAttribute(clipAttr, self.node, ['nearClipPlane', 'farClipPlane'], frames)
        # Focal Length
        focalAttr = geom.GetFocalLengthAttr()
        SetAttribute(focalAttr, self.node, 'focalLength', frames)
        # Depth of Field - Focus Distance
        fdistAttr = geom.GetFocusDistanceAttr()
        SetAttribute(fdistAttr, self.node, 'focusDistance', frames)
        # Depth of Field - F Stop
        fstopAttr = geom.GetFStopAttr()
        SetAttribute(fstopAttr, self.node, 'fStop', frames)
        # Camera Aperture
        SetFilmAperture(geom, self.node, frames)
        # # Camera Aperture - Horizontal
        # hfAttr = geom.GetHorizontalApertureAttr()
        # SetAttribute(hfAttr, self.node, 'horizontalFilmAperture', frames, postMult=2.54*10)
        # # Camera Aperture - Vertical
        # vfAttr = geom.GetVerticalApertureAttr()
        # SetAttribute(vfAttr, self.node, 'verticalFilmAperture', frames, postMult=2.54*10)
        # # Horizontal Offset
        # hoffsetAttr = geom.GetHorizontalApertureOffsetAttr()
        # SetAttribute(hoffsetAttr, self.node, 'horizontalFilmOffset', frames, postMult=2.54*10)
        # # Vertical Offset
        # voffsetAttr = geom.GetVerticalApertureOffsetAttr()
        # SetAttribute(voffsetAttr, self.node, 'verticalFilmOffset', frames, postMult=2.54*10)
        # Projection
        geom.GetProjectionAttr().Set(UsdGeom.Tokens.perspective)
        # FilmAperture - inch
        userAttr = prim.CreateAttribute('userProperties:Camera:FilmAperture', Sdf.ValueTypeNames.Float2)
        SetAttribute(userAttr, self.node, ['horizontalFilmAperture', 'verticalFilmAperture'], frames)

        # # Overscan
        # preScale = cmds.getAttr('%s.preScale' % self.node)
        # if preScale != 1:
        #     userAttr = prim.CreateAttribute('userProperties:Camera:preScale', Sdf.ValueTypeNames.Float)
        #     SetAttribute(userAttr, self.node, 'preScale', frames)
        #
        # # 2D Pan/Zoom
        if cmds.getAttr('%s.renderable' % self.node) and cmds.getAttr("%s.panZoomEnabled" % self.node):
            if not cmds.getAttr("%s.renderPanZoom" % self.node):
                MsgSender.sendMsg("# msg : %s - renderPanZoom value error." % self.node, self.show, self.shot, self.user)
        #     # Pan
        #     userAttr = prim.CreateAttribute("userProperties:Camera:PanZoom:Pan", Sdf.ValueTypeNames.Float2)
        #     SetAttribute(userAttr, self.node, ["horizontalPan", "verticalPan"], frames)
        #     # Zoom
        #     userAttr = prim.CreateAttribute("userProperties:Camera:PanZoom:Zoom", Sdf.ValueTypeNames.Float)
        #     SetAttribute(userAttr, self.node, "zoom", frames)
        #     # Pan/Zoom Export
            expPanzoom = PanZoom(self.node, frames)
            expPanzoom.exportNukeCamShake(self.filename.replace(".usd", "_shake.nk"))
            # expPanzoom.exportPanZoomData(self.filename.replace(".usd", ".panzoom"), self.user)
            # expPanzoom.exportNuke2DPanZoom(self.filename.replace(".usd", ".nk"), self.user)

        matrixs, frames = dxsXformUtils.Get4x4MatrixByXformCmd(self.node, self.fr[0], self.fr[1], step=self.step)
        xformGeom = UsdGeom.Xform(geom)
        for i in xrange(len(frames)):
            xformGeom.MakeMatrixXform().Set(Gf.Matrix4d(*matrixs[i]), Usd.TimeCode(frames[i]))

        stage.GetRootLayer().Save()
        return self.filename

    def getFrames(self):
        result = list()
        for f in range(self.fr[0], self.fr[1]+1):
            for s in dxsMayaUtils.GetFrameSample(self.step):
                result.append(f + s)
        return result


    def abcExport(self):
        abcfile = self.filename.replace('.usd', '.abc')
        step = self.step if self.step != 0.0 else 1.0
        opts = '-ef -df ogawa -ws -sn'
        opts+= ' -fr %s %s' % (self.fr[0], self.fr[1])
        opts+= ' -step %s' % step
        opts+= ' -rt %s' % self.node
        opts+= ' -file %s' % abcfile
        cmds.AbcExport(j=opts, v=True)



class ImagePlaneExport(Arguments.CommonArgs):
    '''
    Args:
        filename (str):
        node (str): imagePlane transform node
    '''
    def __init__(self, filename, node, **kwargs):
        self.filename = filename
        self.node     = node
        Arguments.CommonArgs.__init__(self, **kwargs)

    def doIt(self):
        # create poly imageplane
        planeName, polymesh, tmpmesh = CreatePolyImagePlane(self.node)
        self.planeName = planeName
        self.polyPlane = polymesh

        matrixs, frames = dxsXformUtils.Get4x4MatrixByXformCmd(tmpmesh, self.fr[0], self.fr[1], self.step)
        dxsXformUtils.Set4x4Matrix(polymesh, matrixs, frames)
        cmds.select(polymesh)
        cmds.filterCurve()

        self.geomExport()

        cmds.delete(polymesh)
        cmds.delete(tmpmesh)


    def geomExport(self):
        dxsMayaUtils.UsdExport(self.filename, [self.polyPlane], FR=self.fr, FS=dxsMayaUtils.GetFrameSample(self.step))
        dxsMayaUtils.AbcExport(self.filename.replace('.usd', '.abc'), [self.polyPlane], FR=self.fr, FS=self.step)

        # rename poly imagePlane
        outLayer = Sdf.Layer.FindOrOpen(self.filename)
        stage = Usd.Stage.Open(outLayer)
        dprim = stage.GetDefaultPrim()
        edit = Sdf.BatchNamespaceEdit()
        edit.Add(dprim.GetPath().pathString, '/' + self.planeName)
        outLayer.Apply(edit)
        dprim = stage.GetPrimAtPath('/' + self.planeName)
        stage.SetDefaultPrim(dprim)
        # Material
        self.setMaterial(stage, dprim)
        outLayer.Save()

    def setMaterial(self, stage, dprim):
        imgMtlFile = '/assetlib/3D/material/usd/imageplane/Material.usd'
        scopePrim = stage.DefinePrim(dprim.GetPath().AppendChild('Materials'), 'Scope')
        scopePrim.SetPayload(Sdf.Payload(imgMtlFile))

        mtlPrim = stage.OverridePrim(dprim.GetPath().AppendPath('Materials/PreviewImagePlane'))
        imgPrim = stage.OverridePrim(dprim.GetPath().AppendPath('Materials/PreviewImagePlane/PlateImage'))
        imgShade= UsdShade.Shader(imgPrim)

        filename = cmds.getAttr('%s.imageName' % self.node)
        splitName= filename.split('.')

        if splitName[-1] == 'png':
            constPrim = stage.OverridePrim(dprim.GetPath().AppendPath('Materials/PreviewImagePlane/Constant'))
            constShade= UsdShade.Shader(constPrim)
            opacityAttr = constShade.GetInput('displayOpacity')
            UsdShade.ConnectableAPI.ConnectToSource(opacityAttr, imgShade.GetOutput('rgba'))

        fileAttr = imgShade.GetInput('file')
        if len(splitName) == 2:
            fileAttr.Set(filename)
        elif len(splitName) == 3:
            if cmds.getAttr('%s.useFrameExtension' % self.node):
                for f in range(self.fr[0], self.fr[1]+1):
                    imageNumber = cmds.getAttr('%s.frameExtension' % self.node, time=f)
                    fileAttr.Set('%s.%04d.%s' % (splitName[0], int(imageNumber), splitName[-1]), f)
            else:
                fileAttr.Set(filename)

        UsdShade.Material(mtlPrim).Bind(dprim)


class DummyGeoExport(Arguments.CommonArgs):
    def __init__(self, filename, node, **kwargs):
        self.filename = filename
        self.node = node
        Arguments.CommonArgs.__init__(self, **kwargs)

    def doIt(self):
        # usd export
        dxsMayaUtils.UsdExport(self.filename, [self.node], FR=self.fr, FS=dxsMayaUtils.GetFrameSample(self.step))
        # Alembic export
        abcfile = self.filename.replace('.usd', '.abc')
        dxsMayaUtils.AbcExport(abcfile, [self.node], FR=self.fr, FS=self.step)



#-------------------------------------------------------------------------------
#
#   ASSET
#
#-------------------------------------------------------------------------------
class CameraAssetExport(Arguments.AssetArgs):
    def __init__(self, node=None, **kwargs):
        dxsMayaUtils.PluginSetup(['backstageMenu', 'pxrUsd', 'AbcExport'])
        Arguments.AssetArgs.__init__(self, **kwargs)
        if not self.outDir and self.assetDir:
            self.outDir = self.assetDir + '/cam'
        self.computeVersion()

        self.fps = dxsMayaUtils.GetFPS()
        # Frame Range
        if not self.fr[0] or not self.fr[1]:
            ctime = int(cmds.currentTime(q=True))
            self.fr = (ctime, ctime)
        self.expfr = self.fr
        if self.fr[0] != self.fr[1]:
            self.expfr = (self.fr[0] - 1,  self.fr[1] + 1)

        self.mayafile = dxsMayaUtils.GetMayaFilename()
        if self.mayafile:
            self.comment = 'Generated with %s' % self.mayafile

        self.node = GetCameraNodes(node)

    def doIt(self):
        if not self.node:
            dxsMsg.Print('warning', "[CameraAssetExport] -> Not found camera.")
            return
        self.node = self.node[0]

        name = self.node.split('|')[-1].split(':')[-1]
        # Camera Geom
        geomFile = '{DIR}/{VER}/{NAME}.geom.usd'.format(DIR=self.outDir, VER=self.version, NAME=name)
        CameraGeomExport(geomFile, self.node, fr=self.expfr, step=self.step, fps=self.fps, comment=self.comment, usdformat='usdc').doIt()
        # TODO: ImagePlane
        masterFile = self.makeGeomPackage(name, geomFile)
        self.makePackage(masterFile)


    def makeGeomPackage(self, name, geomFile):
        masterFile = '{DIR}/{VER}/cameras.usd'.format(DIR=self.outDir, VER=self.version)
        SessionUtils.MakeReferenceStage(masterFile, [(geomFile, None)], SdfPath='/cameras/main_cam{cameraVariant=%s}' % name, pType='Camera', comment=self.comment)

        masterPayloadFile = masterFile.replace('.usd', '.payload.usd')
        SessionUtils.MakeReferenceStage(masterPayloadFile, [(masterFile, None)], SdfPath='/cameras{camVersion=%s}' % self.version, comment=self.comment)
        return masterPayloadFile

    def makePackage(self, sourceFile):
        taskFile = '{DIR}/cam.usd'.format(DIR=self.outDir)
        SessionUtils.MakeSubLayerStage(taskFile, [sourceFile])
        taskPayloadFile = taskFile.replace('.usd', '.payload.usd')
        SessionUtils.MakeReferenceStage(taskPayloadFile, [(taskFile, None)], SdfPath='/%s/cameras' % self.assetName)
        if self.showDir:
            PackageUtils.AssetPackage(self.showDir, self.assetName, taskPayloadFile)




#-------------------------------------------------------------------------------
#
#   SHOT
#
#-------------------------------------------------------------------------------
class CameraShotExport(Arguments.ShotArgs):
    def __init__(self, node=None, **kwargs):
        dxsMayaUtils.PluginSetup(['backstageMenu', 'pxrUsd', 'AbcExport'])

        Arguments.ShotArgs.__init__(self, **kwargs)
        if not self.outDir and self.shotDir:
            self.outDir = self.shotDir + '/cam'
        self.computeVersion()

        self.fps = dxsMayaUtils.GetFPS()
        # Frame Range
        if not self.fr[0] or not self.fr[1]:
            self.fr = dxsMayaUtils.GetFrameRange()
        self.expfr = (self.fr[0] - 1, self.fr[1] + 1)

        self.mayafile = dxsMayaUtils.GetMayaFilename()
        if self.mayafile:
            self.comment = 'Generated with %s' % self.mayafile

        self.node = GetCameraNodes(node)
        self.originNode = None


    def doIt(self):
        if not self.node:
            dxsMsg.Print('warning', "[CameraShotExport] -> Not found camera.")
            return
        self.node = self.node[0]
        # self.originNode = cmds.listRelatives(self.node, p=True)[0]
        originNode = cmds.listRelatives(self.node, p=True)
        if originNode:
            self.originNode = originNode[0]
        shape = cmds.ls(self.node, dag=True, type='camera', l=True)[0]

        name = self.node.split('|')[-1].split(':')[-1]
        # Camera Geom
        geomFile = '{DIR}/{VER}/{NAME}.geom.usd'.format(DIR=self.outDir, VER=self.version, NAME=name)
        CameraGeomExport(geomFile, self.node, fr=self.expfr, step=self.step, fps=self.fps, comment=self.comment, usdformat='usdc',
                         show = self.showName, shot=self.shotName, user=self.user).doIt()

        extraFiles = list()
        #-----------------------------------------------------------------------
        # ImagePlane
        imagePlanes = cmds.listConnections(shape, type='imagePlane', d=False)
        if imagePlanes:
            for plane in list(set(imagePlanes)):
                imgPath = cmds.getAttr('%s.imageName' % plane)
                if '/show/' in imgPath and '/shot/' in imgPath:
                    planeName = name + '_' + plane.split('->')[-1].split('|')[-1].split(':')[-1]
                    planeFile = '{DIR}/{VER}/{NAME}.imp.usd'.format(DIR=self.outDir, VER=self.version, NAME=planeName)
                    ImagePlaneExport(planeFile, plane, fr=self.expfr, step=self.step, fps=self.fps).doIt()
                    extraFiles.append(planeFile)

        if self.originNode:
            #-----------------------------------------------------------------------
            # cam_geo
            originName = self.originNode.split('|')[-1].split(':')[-1]
            camGeo = [i for i in cmds.listRelatives(self.originNode, f=True) if i.endswith('cam_geo')]
            if camGeo:
                camGeoName = camGeo[0].split('|')[-1].split(':')[-1]
                camGeoFile = '{DIR}/{VER}/{NAME}.cam_geom.usd'.format(DIR=self.outDir, VER=self.version, NAME=originName)
                DummyGeoExport(camGeoFile, camGeo[0], fr=self.expfr, step=self.step, fps=self.fps).doIt()
                extraFiles.append(camGeoFile)
            #-----------------------------------------------------------------------
            # cam_loc
            camLoc = [i for i in cmds.listRelatives(self.originNode, f=True) if i.endswith('cam_loc')]
            if camLoc:
                camLocName = camLoc[0].split('|')[-1].split(':')[-1]
                camLocFile = '{DIR}/{VER}/{NAME}.cam_loc.abc'.format(DIR=self.outDir, VER=self.version, NAME=originName)
                dxsMayaUtils.AbcExport(camLocFile, camLoc, FR=self.expfr, FS=self.step)

            assetGeoList = [i for i in cmds.listRelatives(self.originNode, f=1) if (i.endswith('_geo') and not (i.endswith('cam_geo')))]
            if assetGeoList:
                for assetGeo in assetGeoList:
                    assetGeoName = assetGeo.split('|')[-1].split(":")[-1]
                    assetGeomFile = '{DIR}/{VER}/{NAME}.cam_geom.usd'.format(DIR=self.outDir, VER=self.version,
                                                                             NAME=assetGeoName)
                    DummyGeoExport(assetGeomFile, assetGeo, fr=self.expfr, step=self.step, fps=self.fps).doIt()
                    extraFiles.append(assetGeomFile)

            assetLocList = [i for i in cmds.listRelatives(self.originNode, f=1) if (i.endswith('_loc') and not (i.endswith('cam_loc')))]
            if assetLocList:
                for assetLoc in assetLocList:
                    assetLocName = assetLoc.split('|')[-1].split(":")[-1]
                    assetLocFile = '{DIR}/{VER}/{NAME}.cam_loc.abc'.format(DIR=self.outDir, VER=self.version, NAME=assetLocName)
                    dxsMayaUtils.AbcExport(assetLocFile, [assetLoc], FR=self.expfr, FS=self.step)

        masterFile = self.makeGeomPackage(geomFile, extraFiles)
        self.makePackage(masterFile)

    def makeGeomPackage(self, geomFile, extraFiles):
        name = self.node.split('|')[-1].split(':')[-1]
        camName = name
        if cmds.getAttr('%s.renderable' % self.node):
            camName = 'main_cam'
            if name.find('left') > -1:
                camName += '_left'
            elif name.find('right') > -1:
                camName += '_right'

        customPrimData = {'scene': os.path.basename(self.mayafile)}
        masterFile = '{DIR}/{VER}/camera.usd'.format(DIR=self.outDir, VER=self.version)
        SessionUtils.MakeReferenceStage(masterFile, [(geomFile, None)], SdfPath='/cameras/' + camName, Kind='assembly', pType='Camera', customPrimData=customPrimData, comment=self.comment)

        if extraFiles:
            for f in extraFiles:
                SessionUtils.MakeReferenceStage(masterFile, [(f, None)], SdfPath='/cameras/extra', addChild=True, composite='reference')
            stage = Usd.Stage.Open(masterFile, load=Usd.Stage.LoadNone)
            prim  = stage.GetPrimAtPath('/cameras/extra')
            riattr= UsdGeom.PrimvarsAPI(prim).CreatePrimvar('ri:attributes:visibility:camera', Sdf.ValueTypeNames.Int)
            riattr.Set(0)
            riattr= UsdGeom.PrimvarsAPI(prim).CreatePrimvar('ri:attributes:visibility:indirect', Sdf.ValueTypeNames.Int)
            riattr.Set(0)
            riattr= UsdGeom.PrimvarsAPI(prim).CreatePrimvar('ri:attributes:visibility:transmission', Sdf.ValueTypeNames.Int)
            riattr.Set(0)
            stage.GetRootLayer().Save()

        masterPayloadFile = masterFile.replace('.usd', '.payload.usd')
        SessionUtils.MakeReferenceStage(masterPayloadFile, [(masterFile, None)], SdfPath='/cameras{camVersion=%s}' % self.version, fr=self.fr, fps=self.fps, comment=self.comment)
        return masterPayloadFile

    def makePackage(self, sourceFile):
        taskFile = '{DIR}/cam.usd'.format(DIR=self.outDir)
        SessionUtils.MakeSubLayerStage(taskFile, [sourceFile])

        taskPayloadFile = taskFile.replace('.usd', '.payload.usd')
        SessionUtils.MakeReferenceStage(taskPayloadFile, [(taskFile, None)], SdfPath='/shot/cameras', Name=self.shotName, Kind='assembly', clear=True)

        PackageUtils.ShotPackage(self.showDir, self.seqName, self.shotName, taskPayloadFile, fr=self.fr, fps=self.fps)
        self.overrideVersion()

    def overrideVersion(self):
        shotFile = '{DIR}/shot/{SEQ}/{SHOT}/{SHOT}.usd'.format(DIR=self.showDir, SEQ=self.seqName, SHOT=self.shotName)
        PackageUtils.VersionSelect(shotFile, '/shot/cameras', 'camVersion', self.version)

        shotLgtFile = shotFile.replace('.usd', '.lgt.usd')
        if os.path.exists(shotLgtFile):
            PackageUtils.VersionSelect(shotLgtFile, '/shot/cameras', 'camVersion', self.version)
