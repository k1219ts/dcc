
import os, re

from Katana import FnGeolibServices, NodegraphAPI, Nodes3DAPI
import UI4
from fnpxr import Sdf, Usd, UsdGeom, Gf, Vt
VTN = Sdf.ValueTypeNames

import DXRulebook.Interface as rb

import DXUSD_KAT.Utils as utl
import DXUSD_KAT.Vars as var
import DXUSD_KAT.Compositor as cmp


class CreateLight:
    def __init__(self, evalNode, locations, output):
        self.rootProd  = Nodes3DAPI.GetGeometryProducer(node=evalNode)
        self.locations = locations
        self.output = output

        self.lightList = list()


    def getVec3fArray(self, value):
        result = list()
        for i in range(int(len(value) / 3)):
            result.append(Gf.Vec3f(*value[i*3:i*3+3]))
        return result

    def writeAttr(self, spec, attr, name, typ):
        if typ == VTN.Float3 or typ == VTN.Color3f or typ == VTN.Vector3f:
            value = Gf.Vec3f(*attr.getData())
        elif typ == VTN.FloatArray:
            value = Vt.FloatArray(attr.getData())
        elif typ == VTN.Color3fArray:
            value = Vt.Vec3fArray(self.getVec3fArray(attr.getData()))
        else:
            value = attr.getValue()
            # value re-define
            if 'spline:interpolation' in name:
                if value == 'catmull-rom':
                    value = 'catmullRom'
            if name == 'ri:combineMode':
                if value == 'mult':
                    value = 'multiply'
            if name == 'barnMode':
                if value == 1:
                    value = 'analytic'
                else:
                    value = 'physical'
            if name == 'preBarnEffect':
                if value == 1:
                    value = 'cone'
                elif value == 2:
                    value = 'noLight'
                else:
                    value = 'noEffect'
            if name == 'cookieMode':
                if value == 1:
                    value = 'analytic'
                else:
                    value = 'physical'
            if name == 'texture:wrapMode':
                if value == 1:
                    value = 'clamp'
                elif value == 2:
                    value = 'repeat'
                else:
                    value = 'off'

        if name.startswith('colorRamp') or name.startswith('falloffRamp'):
            variability = Sdf.VariabilityUniform
        else:
            variability = Sdf.VariabilityVarying

        utl.GetAttributeSpec(spec, name, value, typ, variability=variability)

    def getXform(self, parent):
        fnAttr = self.prod.getFnAttribute('xform')
        if not fnAttr:
            return
        valAttr, null = FnGeolibServices.XFormUtil.CalcTransformMatrixAtTime(fnAttr, 1)
        if valAttr:
            val = valAttr.getNearestSample(0)
            utl.GetAttributeSpec(parent, 'xformOp:transform', Gf.Matrix4d(*val), VTN.Matrix4d)
            utl.GetAttributeSpec(parent, 'xformOpOrder', ['xformOp:transform'], VTN.TokenArray)

    def prmanParams(self, parent):
        # visibility camera
        attr = self.prod.getAttribute('prmanStatements.attributes.visibility.camera')
        if attr:
            utl.GetAttributeSpec(parent, 'primvars:ri:attributes:visibility:camera', attr.getValue(), VTN.Int)


    def lightParams(self, parent, lightShader):     # light shader name
        params = self.prod.getAttribute('material.prmanLightParams')
        if not params:
            return

        _defined = dict()
        _defined.update(var._Light)
        _defined.update(var._ShapingAPI)
        _defined.update(var._ShadowAPI)
        _defined.update(var._RiLightAPI)
        if lightShader == 'PxrDomeLight' or lightShader == 'PxrRectLight':
            _defined.update(var._RiTextureAPI)
        if lightShader == 'PxrEnvDayLight':
            _defined.update(var._PxrEnvDayLight)
        if lightShader == 'PxrAovLight':
            _defined.update(var._PxrAovLight)

        for name, attr in params.childList():
            if name in _defined:
                n, t = _defined[name]
                self.writeAttr(parent, attr, n, t)
            else:
                print('>>> lightParams not pre-defined :', name)

    def iterLight(self, parent, location):
        self.prod  = self.rootProd.getProducerByPath(location)
        shaderAttr = self.prod.getAttribute('material.prmanLightShader')
        if not shaderAttr:
            return

        shader = shaderAttr.getValue()
        name   = location.split('/')[-1]

        if shader not in var.PXRTLUX:
            print('>>> not support : %s' % shader)
            return

        print('>>> export light\t:', location)

        spec = utl.GetPrimSpec(parent.layer, parent.path.AppendChild(name), type=var.PXRTLUX[shader])
        spec.SetInfo('kind', 'subcomponent')
        # utl.CreateLightSchemas(spec)
        utl.SetApiSchemas(spec, ['ShapingAPI', 'ShadowAPI'])
        self.lightList.append(spec.path)

        # xform
        self.getXform(spec)

        # light params
        self.lightParams(spec, shader)

        # prmanStatements
        self.prmanParams(spec)

        # light filter
        prod = self.rootProd.getProducerByPath(location)
        for p in prod.iterChildren():
            if p.getType() == 'light filter':
                self.iterLightFilter(spec, p.getFullName())


    def lightfilterParams(self, parent, filterShader):  # lightfilter shader name
        params = self.prod.getAttribute('material.prmanLightfilterParams')
        if not params:
            return

        _defined = dict()
        _defined.update(var._LightFilterAPI)
        try:
            eval('_defined.update(var._%s)' % filterShader)
        except:
            print('>>> Vars define error :', filterShader)
            pass

        for name, attr in params.childList():
            if name in _defined:
                n, t = _defined[name]
                self.writeAttr(parent, attr, n, t)
            else:
                print('>>> lightfilterParams not pre-defined :', name)

    def iterLightFilter(self, parent, location):
        self.prod  = self.rootProd.getProducerByPath(location)

        shaderAttr = self.prod.getAttribute('material.prmanLightfilterShader')
        if not shaderAttr:
            return

        print('>>> export lightfilter\t:', location)

        shader = shaderAttr.getValue()
        name   = location.split('/')[-1]

        spec = utl.GetPrimSpec(parent.layer, parent.path.AppendChild(name), type=shader)
        utl.SetApiSchemas(spec, ['RiSplineAPI', 'RiLightFilterAPI'])
        self.lightList.append(spec.path)

        # filters
        utl.GetRelationshipSpec(parent, 'filters', spec.path)

        # xform
        self.getXform(spec)

        # lightfilter params
        self.lightfilterParams(spec, shader)



    def doIt(self):
        outlyr = utl.AsLayer(self.output, create=True, clear=True)
        customLayerData = outlyr.customLayerData

        # current project filename
        katfile = NodegraphAPI.NodegraphGlobals.GetProjectFile()
        customData = {'sceneFile': katfile}
        customLayerData.update(customData)
        outlyr.customLayerData = customLayerData

        spec = utl.GetPrimSpec(outlyr, '/Lights', type='Scope')
        outlyr.defaultPrim = 'Lights'

        for loc in self.locations:
            self.iterLight(spec, loc)

        if self.lightList:
            utl.GetRelationshipSpec(spec, 'lightList', self.lightList)
            utl.GetAttributeSpec(spec, 'lightList:cacheBehavior', 'consumeAndContinue', VTN.Token)

        with utl.OpenStage(outlyr) as stage:
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

        outlyr.Save()
        del outlyr


def doIt(dir):
    viewNode = NodegraphAPI.GetViewNode()

    root = Nodes3DAPI.GetGeometryProducer(node=viewNode)

    sg = Nodes3DAPI.ScenegraphManager.getActiveScenegraph()
    selected = sg.getSelectedLocations()
    if not selected:
        selected.append('/root/world/lgt')

    lightPaths = list()
    def walkLocation(location):
        prod = root.getProducerByPath(location)
        if prod.getType() == 'light':
            if not prod.getAttribute('info.light.mute'):
                if not location in lightPaths:
                    lightPaths.append(location)
        for p in prod.iterChildren():
            walkLocation(p.getFullName())

    for loc in selected:
        walkLocation(loc)

    if not lightPaths:
        assert False, '[ERROR] - not found lights.'
    # print lightPaths

    output = os.path.join(dir, 'light.usd')
    # output = '/WORK_DATA/temp/material/light.usd'
    CreateLight(viewNode, lightPaths, output).doIt()

def exportDialog():
    dir = UI4.Util.AssetId.BrowseForAsset('', 'USD Export Light (select directory)', True, {'acceptDir': True})
    if dir:
        doIt(dir)
