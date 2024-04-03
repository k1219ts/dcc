#coding:utf-8
from __future__ import print_function

from pxr import Gf, Sdf, Usd, UsdGeom, Vt

import maya.api.OpenMaya as OpenMaya
import maya.cmds as cmds

import DXUSD.Vars as var
import os

from DXUSD.Tweakers.Tweaker import Tweaker, ATweaker

import DXUSD_MAYA.Message as msg
import DXUSD_MAYA.MUtils as mutl
import DXUSD_MAYA.Utils as utl


class AModifyMayaNurbsClearWidths(ATweaker):
    def __init__(self, **kwargs):
        self.geomfiles = []
        # initialize
        ATweaker.__init__(self, **kwargs)

    def Treat(self):
        if not self.geomfiles:
            return var.IGNORE
        return var.SUCCESS

class ModifyMayaNurbsClearWidths(Tweaker):
    ARGCLASS = AModifyMayaNurbsClearWidths

    def DoIt(self):
        print('#### ModifyMayaNurbsClearWidths ####')

        for f in self.arg.geomfiles:
            filePath = f.replace('.usd', '.topology.usd')
            if os.path.exists(filePath):
                f = filePath
            msg.debug('\t >>', f)
            self.Treat(f)

        return var.SUCCESS

    def Treat(self, filename):
        dstlyr = utl.AsLayer(filename)
        if not dstlyr:
            return

        with utl.OpenStage(dstlyr) as stage:
            dprim = stage.GetDefaultPrim()
            for p in iter(Usd.PrimRange.AllPrims(dprim)):
                if p.GetTypeName() == 'NurbsCurves':
                    primPath = p.GetPath()

                    primSpec = dstlyr.GetPrimAtPath(primPath)
                    try:
                        attrSpec = primSpec.properties.get('widths')
                        if attrSpec:
                            primSpec.RemoveProperty(attrSpec)
                    except:
                        pass

        dstlyr.Save()
        del dstlyr




class AModifyMayaNurbsAddWidths(ATweaker):
    def __init__(self, **kwargs):
        self.geomfiles = []
        self.namespace = ''
        # initialize
        ATweaker.__init__(self, **kwargs)
        print('>>>>>self.geomfiles:',self.geomfiles)

    def Treat(self):
        if not self.geomfiles:
            return var.IGNORE
        return var.SUCCESS


class ModifyMayaNurbsAddWidths(Tweaker):
    ARGCLASS = AModifyMayaNurbsAddWidths

    def computeWidth(self, mpath, prim):
        shape = cmds.ls(mpath, dag=True, type='nurbsCurve')
        if not shape:
            return
        shape = shape[0]
        rootWidth = 1
        tipWidth  = 1

        attrs = cmds.listAttr(shape, st='*Width')
        if 'lineWidth' in attrs:
            val = cmds.getAttr('%s.lineWidth' % shape)
            rootWidth = val
            tipWidth  = val
        for ln in attrs:
            if ln.endswith('BaseWidth'):
                rootWidth = cmds.getAttr('%s.%s' % (shape, ln))
            if ln.endswith('TipWidth'):
                tipWidth  = cmds.getAttr('%s.%s' % (shape, ln))

        # width
        if rootWidth == tipWidth:
            widths = [rootWidth]
        else:
            mobj = mutl.GetMObject(shape, dag=False)
            curveFn = OpenMaya.MFnNurbsCurve(mobj)
            curveFn.updateCurve()
            curveLength = curveFn.length()

            widths = list()
            for i in range(curveFn.numSpans+1):
                lengthPos  = curveFn.findLengthFromParam(i)
                lengthRate = lengthPos / curveLength
                width = rootWidth + (tipWidth - rootWidth) * lengthRate
                widths.append(width)

            widths.insert(1, widths[0] + (widths[1] - widths[0]) * 0.5)
            widths.insert(-1, widths[-2] + (widths[-1] - widths[-2]) * 0.5)

        widthAttr = UsdGeom.NurbsCurves(prim).GetWidthsAttr()
        widthAttr.Set(widths)
        if len(widths) > 1:
            widthAttr.SetMetadata('interpolation', 'vertex')


    def DoIt(self):
        print('#### ModifyMayaNurbsAddWidths ####')
        for f in self.arg.geomfiles:
            self.Treat(f)

        return var.SUCCESS

    def Treat(self, filename):
        dstlyr = utl.AsLayer(filename)
        if not dstlyr:
            return

        with utl.OpenStage(dstlyr) as stage:
            dprim = stage.GetDefaultPrim()
            for p in iter(Usd.PrimRange.AllPrims(dprim)):
                if p.GetTypeName() == 'NurbsCurves':
                    primPath = p.GetPath().pathString

                    mayaPath = primPath.replace('/', '|')
                    if self.arg.namespace:
                        mayaPath = mayaPath.replace('|', '|%s:' % self.arg.namespace)
                    if cmds.objExists(mayaPath):
                        self.computeWidth(mayaPath, p)

        dstlyr.Save()
        del dstlyr



class AModifyMayaNurbsAddAttrs(ATweaker):
    def __init__(self, **kwargs):
        self.geomfiles = []
        self.namespace = ''
        # initialize
        ATweaker.__init__(self, **kwargs)

    def Treat(self):
        if not self.geomfiles:
            return var.IGNORE
        return var.SUCCESS


class ModifyMayaNurbsAddAttrs(Tweaker):
    ARGCLASS = AModifyMayaNurbsAddWidths

    def DoIt(self):
        print('#### ModifyMayaNurbsAddAttrs ####')
        for f in self.arg.geomfiles:
            if 'sim_geom' in f:
                self.Treat(f)

        return var.SUCCESS

    def Treat(self, filename):

        attrConfigFile = '/stdrepo/RIG/_config/rigAttr.json'
        data = {}

        if not os.path.exists(attrConfigFile):
            return

        if os.path.isfile(attrConfigFile):
            with open(attrConfigFile, 'r') as f:
                data = eval(f.read())
                if data.get('shapeAttr'):
                    print('data[shapeAttr]:', data['shapeAttr'])
                    #ATTRs = data['shapeAttr']
                else:
                    return

        dstlyr = utl.AsLayer(filename)
        if not dstlyr:
            return

        with utl.OpenStage(dstlyr) as stage:
            dprim = stage.GetDefaultPrim()
            for p in iter(Usd.PrimRange.AllPrims(dprim)):
                if p.GetTypeName() == 'NurbsCurves':
                    stack = p.GetPrimStack()
                    spec = stack[0]
                    primPath = p.GetPath().pathString
                    mayaPath = primPath.replace('/', '|')
                    if self.arg.namespace:
                        mayaPath = mayaPath.replace('|', '|%s:' % self.arg.namespace)

                    if cmds.objExists(mayaPath):
                        shape = cmds.ls(mayaPath, dag=True, type='nurbsCurve')
                        if not shape:
                            return
                        shape = shape[0]

                        baselist = data['shapeAttr']['base']
                        vectorlist = data['shapeAttr']['vector']

                        if baselist:
                            self.AttrSet(shape, dstlyr, spec, baselist)
                        if vectorlist:
                            self.AttrSet(shape, dstlyr, spec, vectorlist, type="vector")

        dstlyr.Save()
        del dstlyr


    def AttrSet(self, shape, dstlyr, spec, ATTRs, type=''):
        IsAttr = False
        for ATTR in ATTRs:
            for attr in cmds.listAttr(shape):
                if ATTR in attr:
                    IsAttr = True
                    break
            else:
                IsAttr = False

            if IsAttr == True:
                for frame in range(self.arg.frameRange[0] - 1, self.arg.frameRange[1] + 1):
                    for sample in mutl.GetFrameSample(self.arg.step):
                        frameSample = frame + sample
                        cmds.currentTime(frameSample)
                        if type == 'vector':
                            valueX = cmds.getAttr('%s.%sX' % (shape, ATTR))
                            valueY = cmds.getAttr('%s.%sY' % (shape, ATTR))
                            valueZ = cmds.getAttr('%s.%sZ' % (shape, ATTR))
                            value = (valueX, valueY, valueZ)
                            print(type, value)
                            attrSpec = utl.GetAttributeSpec(spec, 'primvars:%s' % ATTR, None,
                                                            Sdf.ValueTypeNames.Float3Array)
                            dstlyr.SetTimeSample(attrSpec.path, frameSample, Vt.Vec3fArray([value]))
                        else:
                            value = cmds.getAttr('%s.%s' % (shape, ATTR))
                            print(type, value)
                            attrSpec = utl.GetAttributeSpec(spec, 'primvars:%s' % ATTR, None, Sdf.ValueTypeNames.FloatArray)
                            dstlyr.SetTimeSample(attrSpec.path, frameSample, Vt.FloatArray([value]))
