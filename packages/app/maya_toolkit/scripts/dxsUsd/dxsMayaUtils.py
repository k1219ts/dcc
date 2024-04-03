import os
import json
import re
import string
import random

import maya.api.OpenMaya as OpenMaya
import maya.cmds as cmds

import dxsMsg


def PluginSetup(plugins):
    for p in plugins:
        if not cmds.pluginInfo(p, q=True, l=True):
            cmds.loadPlugin(p)


def GetMayaFilename():
    filename = cmds.file(q=True, sn=True)
    if filename:
        splitext = os.path.splitext(filename)
        new = splitext[0].split('--')[0] + splitext[-1]
        new = new.replace('/CacheOut_Submitter', '')
        return new


def GetViz(node):   # node is full path string
    viz = True
    source = node.split('|')
    for i in range(1, len(source)):
        path = string.join(source[:i+1], '|')
        if cmds.listConnections('%s.visibility' % path):
            vals = cmds.keyframe(path, at='visibility', q=True, vc=True)
            if not 1.0 in vals:
                return False
        else:
            viz = cmds.getAttr('%s.visibility' % path)
            if not viz:
                return viz
        connects = cmds.listConnections('%s.drawOverride' % path, type='displayLayer')
        if connects:
            for c in connects:
                viz = cmds.getAttr('%s.visibility' % c)
                if not viz:
                    return viz
    return viz



def GetNodeInfo(node):
    name = node.split('|')[-1].split(':')[-1]

    if '_low' in name:
        nodeName = name.replace('_low', '')
        geomType = 'low'
    elif '_mid' in name:
        nodeName = name.replace('_mid', '')
        geomType = 'mid'
    elif '_sim' in name:
        nodeName = name.replace('_sim', '')
        geomType = 'sim'
    else:
        nodeName = name
        geomType = 'high'

    return nodeName, geomType


def GetNamespaceInfo(node):
    src = node.split('|')[-1].split(':')
    if len(src) > 1:
        ns_name = string.join(src[:-1], ':')
    else:
        ns_name = None
    nodename = src[-1]
    return ns_name, nodename


def GetMObject(name, dag=True):
    sels = OpenMaya.MGlobal.getSelectionListByName(name)
    if dag:
        return sels.getDagPath(0)
    else:
        return sels.getDependNode(0)


def SetAttr(shape, attrName, Value, Type):
    if not cmds.attributeQuery(attrName, n=shape, ex=True):
        if Type == 'string':
            cmds.addAttr(shape, ln=attrName, dt='string')
        else:
            cmds.addAttr(shape, ln=attrName, at=Type)
    cmds.setAttr(shape + '.' + attrName, Value, type=Type)


def GetFrameRange():
    start = int(cmds.playbackOptions(q=True, min=True))
    end   = int(cmds.playbackOptions(q=True, max=True))
    return start, end

def GetFrameSample(step):
    samples = list()
    if step == 0:
        samples.append(0.0)
    else:
        for i in range(0, 100, int(step * 100)):
            samples.append(round(i * 0.01, 2))
    return samples

TIME_MAP = {
    'game': 15.0, 'film': 24.0, 'pal': 25.0, 'ntsc': 30.0, 'show': 48.0,
    'palf': 50.0, 'ntscf': 60.0
}
TIME_MAP_INV = {
    15.0: 'game', 24.0: 'film', 25.0: 'pal', 30.0: 'ntsc', 48.0: 'show',
    50.0: 'palf', 60.0: 'ntscf'
}
def GetFPS():
    timeUnit = cmds.currentUnit(q=True, t=True)
    return TIME_MAP[timeUnit]

def SetFPS(fps):
    cmds.currentUnit(t=TIME_MAP_INV[fps])

#-------------------------------------------------------------------------------
#
#   Attribute
#
#-------------------------------------------------------------------------------
_SubdivSchemeMap = {0: 'catmullClark', 1: 'loop', 100: 'none'}
_RfmSubdivScheme = 'rman__torattr___subdivScheme'
_UsdSubdivScheme = 'USD_subdivisionScheme'

class UsdGeomAttributes:
    '''
    USD Attribute setup. ( not using Alembic chaser )
    Args:
        nodes (list):
        user (bool): Ri User Attributes
        mtl (bool): MaterialSet. if not 'ObjectSet', compute by shapename
        subdiv (bool):
    '''
    def __init__(self, nodes, user=False, mtl=False, subdiv=False):
        self.nodes  = nodes
        self.userAttr= user
        self.mtlAttr = mtl
        self.subdivAttr = subdiv

        # User Attributes
        self.excludeMeshAttributes = ['txVarNum', 'txVersion']
        self.excludeReferenceAttributes = ['txBasePath', 'txLayerName', 'txmultiUV', 'txVersion', 'modelVersion']

        self.mtlShapeProc = True
        if cmds.objExists('MaterialSet'):
            self.mtlShapeProc = False

        self.addedAttributes = ['USD_UserExportedAttributesJson']


    def Clear(self):
        for shape in cmds.ls(self.nodes, l=True):
            usdAttrs = cmds.listAttr(shape, st=self.addedAttributes)
            if usdAttrs:
                for ln in usdAttrs:
                    cmds.deleteAttr('%s.%s' % (shape, ln))

    def Set(self):
        # Add MaterialSet attribute by 'MaterialSet'
        if self.mtlAttr and not self.mtlShapeProc:
            self.AddMaterialAttributeByObjectSet()

        for shape in cmds.ls(self.nodes, l=True):
            self._usdShapeAttrs = dict()

            # Ri User Attribute
            if self.userAttr:
                # self.riUserAttributes(shape)
                self.AddUserAttributes(shape)

            # Material
            if self.mtlAttr:
                if self.mtlShapeProc:
                    self.AddMaterialAttributeByShape(shape)
                self.materialsetAttribute(shape)

            # Subdivision
            if self.subdivAttr:
                self.subdivAttributes(shape)

            # Geometry rman primvars
            self.rmanPrimvars(shape)

            # Dexter scale manifold primvars
            self.scalePrimvars(shape)

            # Geometry bora primvars
            self.boraPrimvars(shape)

            # UserExportedAttributesJson
            if self._usdShapeAttrs:
                self.reduceUsdShapeAttrs()
                if not cmds.attributeQuery('USD_UserExportedAttributesJson', n=shape, ex=True):
                    cmds.addAttr(shape, ln='USD_UserExportedAttributesJson', dt='string')
                cmds.setAttr('%s.USD_UserExportedAttributesJson' % shape, json.dumps(self._usdShapeAttrs), type='string')


    def reduceUsdShapeAttrs(self):
        delAttrs = list()
        for ln in self._usdShapeAttrs.keys():
            if 'rman__riattr__user_' in ln:
                reln = ln.replace('rman__riattr__user_', '')
                if self._usdShapeAttrs.has_key(reln):
                    delAttrs.append(ln)
        for ln in delAttrs:
            self._usdShapeAttrs.pop(ln)
        delAttr = 'rman__riattr__user_txAssetName'
        if self._usdShapeAttrs.has_key('txBasePath') and self._usdShapeAttrs.has_key(delAttr):
            self._usdShapeAttrs.pop(delAttr)


    def AddUserAttributes(self, shape):
        excludeMap = {'mesh': self.excludeMeshAttributes, 'reference': self.excludeReferenceAttributes}
        shapeType = cmds.nodeType(shape)
        shapeState= 'mesh'
        if shapeType == 'pxrUsdProxyShape':
            shapeState = 'reference'

        userAttrs = cmds.listAttr(shape, ud=True, st=['rman__riattr__user_*', 'tx*', 'modelVersion'])
        if not userAttrs:
            return

        data = dict()
        for ln in userAttrs:
            attrName = ln.replace('rman__riattr__user_', '')
            attrType = 'primvar'
            if shapeState == 'reference':
                attrType = 'usdRi'
            if attrName in excludeMap[shapeState]:
                attrName = None
            if attrName:
                data[ln] = {'usdAttrName': attrName, 'usdAttrType': attrType}
        if data:
            self._usdShapeAttrs.update(data)


    def AddMaterialAttributeByObjectSet(self):
        if not cmds.objExists('MaterialSet'):
            return
        for m in cmds.sets('MaterialSet', q=True):
            source = cmds.sets(m, q=True)
            memberShapes = cmds.ls(source, dag=True, type='surfaceShape', ni=True)
            for shape in memberShapes:
                if not cmds.attributeQuery('MaterialSet', n=shape, ex=True):
                    cmds.addAttr(shape, ln='MaterialSet', dt='string')
                cmds.setAttr('%s.MaterialSet' % shape, m, type='string')

    def AddMaterialAttributeByShape(self, shape):
        if not cmds.attributeQuery('MaterialSet', n=shape, ex=True):
            nameparse = re.compile(r'_M(.*?)_').findall(shape.split('|')[-1])
            if nameparse:
                cmds.addAttr(shape, ln='MaterialSet', dt='string')
                cmds.setAttr('%s.MaterialSet' % shape, nameparse[0], type='string')

    def materialsetAttribute(self, shape):
        if cmds.attributeQuery('MaterialSet', n=shape, ex=True):
            data = {
                'MaterialSet': {
                    'usdAttrName': 'userProperties:MaterialSet', 'usdAttrType': 'userProperties'
                }
            }
            self._usdShapeAttrs.update(data)


    def subdivAttributes(self, shape):
        if cmds.attributeQuery(_RfmSubdivScheme, n=shape, ex=True):
            scheme = cmds.getAttr('%s.%s' % (shape, _RfmSubdivScheme))
            if scheme != 100:
                if not cmds.attributeQuery(_UsdSubdivScheme, n=shape, ex=True):
                    cmds.addAttr(shape, ln=_UsdSubdivScheme, dt='string')
                    self.addedAttributes.append(_UsdSubdivScheme)
                cmds.setAttr('%s.%s' % (shape, _UsdSubdivScheme), _SubdivSchemeMap[scheme], type='string')


    def rmanPrimvars(self, shape):
        data = dict()
        userAttrs = cmds.listAttr(shape, ud=True, st=['rman*'])
        if userAttrs:
            for ln in userAttrs:
                if ln.find('rmanafF') > -1 and ln.find('_AbcGeomScope') == -1:
                    usdAttrName = ln.replace('rmanafF', '')
                    data[ln] = {'usdAttrName': usdAttrName, 'usdAttrType': 'primvar', 'interpolation': 'faceVarying'}
                if ln.find('rmanP') > -1 and ln.find('_AbcGeomScope') == -1:
                    usdAttrName = ln.replace('rmanP', '')
                    data[ln] = {'usdAttrName': usdAttrName, 'usdAttrType': 'primvar', 'interpolation': 'vertex'}
        if data:
            self._usdShapeAttrs.update(data)

    # charles : export bora primvars
    def boraPrimvars(self, shape):
        data = dict()
        userAttrs = cmds.listAttr(shape, ud=True, st=['bora_*'])
        if userAttrs:
            for ln in userAttrs:
                if '_mask' in ln:
                    usdAttrName = 'bora_mask'
                    data[ln] = {'usdAttrName': usdAttrName, 'usdAttrType': 'primvar', 'interpolation': 'vertex', 'doubleAttributeAsFloatAttribute':True}
        if data:
            self._usdShapeAttrs.update(data)

    # daeseok : export scale* primvars
    def scalePrimvars(self, shape):
        data = dict()
        userAttrs = cmds.listAttr(shape, ud=True, st=['scale*'])

        if userAttrs:
            for ln in userAttrs:
                data[ln] = {'usdAttrName': ln, 'usdAttrType': 'primvar'}

        if data:
            self._usdShapeAttrs.update(data)

_txAssetNameAttr = 'rman__riattr__user_txAssetName'
_txBasePathAttr  = 'rman__riattr__user_txBasePath'
def UpdateTextureAttributes(objects, asset=None, element=None):
    for shape in objects:
        if not cmds.attributeQuery('txBasePath', n=shape, ex=True):
            if cmds.attributeQuery('txLayerName', n=shape, ex=True) or cmds.attributeQuery('rman__riattr__user_txLayerName', n=shape, ex=True):
                if cmds.attributeQuery(_txAssetNameAttr, n=shape, ex=True):
                    getVal = cmds.getAttr(shape + '.' + _txAssetNameAttr)
                    newVal = 'asset/' + getVal.split('/')[-1] + '/texture'
                    SetAttr(shape, _txBasePathAttr, newVal, 'string')
                    cmds.deleteAttr(shape + '.' + _txAssetNameAttr)
                if not cmds.attributeQuery(_txBasePathAttr, n=shape, ex=True) and asset:
                    txpath = 'asset/%s/texture' % asset
                    if element:
                        txpath = 'asset/%s/element/%s/texture' % (asset, element)
                    SetAttr(shape, _txBasePathAttr, txpath, 'string')


def AddModelVersionAttribute(objects, version='v001'):
    for shape in objects:
        if not cmds.attributeQuery('modelVersion', n=shape, ex=True):
            cmds.addAttr(shape, ln='modelVersion', dt='string')
        cmds.setAttr('%s.modelVersion' % shape, version, type='string')


def CoreKeyDump(node, attr):
    connections = cmds.listConnections('%s.%s' % (node, attr), type='animCurve', s=True, d=False)
    if connections:
        result = dict()
        result['frame'] = cmds.keyframe(node, at=attr, q=True)
        result['value'] = cmds.keyframe(node, at=attr, q=True, vc=True)
        result['angle'] = cmds.keyTangent(node, at=attr, q=True, ia=True, oa=True)
        if cmds.keyTangent(node, at=attr, q=True, wt=True)[0]:
            result['weight'] = cmds.keyTangent(node, at=attr, q=True, iw=True, ow=True)
        result['infinity'] = cmds.setInfinity(node, at=attr, q=True, pri=True, poi=True)
        return result
    else:
        gv = cmds.getAttr('%s.%s' % (node, attr))
        gt = cmds.getAttr('%s.%s' % (node, attr), type=True)
        return {'value': gv, 'type': gt}

def AttributesKeyDump(node, attrs):
    result = dict()
    for ln in attrs:
        result[ln] = CoreKeyDump(node, ln)
    return result



#-------------------------------------------------------------------------------
#
#   ANIMATION
#
#-------------------------------------------------------------------------------
class KeyOffset:
    def __init__(self, node, offset=1):
        connections = cmds.listConnections(node, type='animCurve')
        if not connections:
            return
        for c in connections:
            ln  = cmds.listConnections(c, p=True)
            src = ln[0].split('.')
            self.attributeKeyOffset(src[0], src[-1], offset)

    def attributeKeyOffset(self, node, attr, offset):
        ln = node + '.' + attr
        frames = cmds.keyframe(ln, q=True, a=True)
        if frames:
            end_value = cmds.getAttr(ln, t=frames[-1])
            tmp_value = cmds.getAttr(ln, t=frames[-1]-offset)
            set_value = end_value - tmp_value
            cmds.setKeyframe(ln, itt='spline', ott='spline', t=frames[-1]+offset, at=attr, v=end_value+set_value)

            start_value = cmds.getAttr(ln, t=frames[0])
            tmp_value   = cmds.getAttr(ln, t=frames[0]+offset)
            set_value   = tmp_value - start_value
            cmds.setKeyframe(ln, itt='spline', ott='spline', t=frames[0]-offset, at=attr, v=start_value-set_value)


class ConnectBlendShape:
    def __init__(self, destination=None, source=None, sourceroot=None):
        self.source = source
        self.sourceroot  = sourceroot
        self.destination = destination
        self.blendPlug = None
        self.getSource()

    def getSource(self):
        if self.source:
            return

        sourceMeshes = list()
        for s in cmds.ls(self.destination.split('|')[-1], l=True):
            if self.sourceroot and s.startswith("|" + self.sourceroot + '|'):
                sourceMeshes.append(s)
            else:
                if s.find(self.destination) == -1:
                    sourceMeshes.append(s)
        sourceMeshes = cmds.ls(sourceMeshes, dag=True, type='surfaceShape', ni=True, l=True)
        if len(sourceMeshes) == 1:
            self.source = sourceMeshes[0]
        else:
            for s in sourceMeshes:
                if s.find('|render') > -1:
                    self.source = s

    def doIt(self):
        if not self.source or not self.destination:
            # assert False, "# msg : not plug blendShape"
            dxsMsg.Print('warning', 'Connection Fail : destination[%s] <- source[%s]' % (self.destination, self.source))
            return
        cmds.blendShape(self.source, self.destination, w = [(0, 1), (1, 0)])
        dxsMsg.Print('info', 'Connection : %s <- %s' % (self.destination, self.source))



#-------------------------------------------------------------------------------
#
#   MAYA COMMAND
#
#-------------------------------------------------------------------------------
def UsdExport(filename, nodes, FR=[None, None], FS=0.0, **kwargs):
    '''
    Args:
        filename (str) :
        nodes    (list):
        FR (tuple) : frameRange (start(int), end(int)). default is None, don't write timeSample.
        FS (list)  : frameSample [0.0, 0.5]
    '''
    cmds.select(nodes)

    opts = {
        'file': filename,
        'append': False,
        'exportColorSets': False,
        'defaultMeshScheme': 'none',
        'exportDisplayColor': False,
        'eulerFilter': True,
        'exportInstances': True,
        'exportRefsAsInstanceable': True,
        'exportReferenceObjects': True,
        'kind': 'component',
        'shadingMode': 'none',
        'exportSkels': 'none',
        'exportSkin': 'none',
        'selection': True,
        'stripNamespaces': False,
        'exportUVs': True,
        'exportVisibility': True,
        'verbose': True
    }

    if FR[0] != None and FR[1] != None:
        opts['frameRange'] = FR
        opts['frameSample']= FS

    # Override Options
    if kwargs:
        opts.update(kwargs)

    cmds.usdExport(**opts)
    cmds.select(cl=True)
    return filename

def UsdImport(filename, **kwargs):
    opts = {
        'file': filename,
        'assemblyRep': "",
        'shadingMode': "none",
#        "variant" : ("lodVariant", "high")
    }

    # Override Options
    if kwargs:
        opts.update(kwargs)

    node = cmds.usdImport(**opts)
    return node

def UsdProxyImport(filename):
    from pxr import Usd
    stage = Usd.Stage.Open(filename, load=Usd.Stage.LoadNone)
    dprim = stage.GetDefaultPrim()
    defaultName = dprim.GetName()

    node = cmds.createNode('pxrUsdProxyShape', name='%sShape' % defaultName)
    # cmds.connectAttr('time1.outTime', '%s.time' % node)
    cmds.setAttr('%s.filePath' % node, filename, type='string')
    return cmds.listRelatives(node, p=True)[0]

def UsdAssemblyImport(filename):
    from pxr import Usd
    stage = Usd.Stage.Open(filename, load=Usd.Stage.LoadNone)
    dprim = stage.GetDefaultPrim()
    defaultName = dprim.GetName()

    node = cmds.assembly(name=defaultName, type='pxrUsdReferenceAssembly')
    # cmds.connectAttr('time1.outTime', '%s.time' % node)
    cmds.setAttr('%s.filePath' % node, filename, type='string')
    cmds.assembly(node, edit=True, active='Collapsed')
    return node


def AbcExport(filename, nodes, FR=[1, 1], FS=1.0):
    '''
    Args:
        filename (str) : alembic filename
        nodes    (list): export node list
        FR (tuple) : frameRange (start(double), end(double))
        FS (double): step
    Returns:
        filename
    '''
    currentFrame = cmds.currentTime(q=True)
    if FR[0] == None and FR[1] == None:
        fr = (currentFrame, currentFrame)
    else:
        fr = FR

    opts  = '-uv -wv -wuvs -ef -df ogawa -ws'
    opts += ' -a MaterialSet'
    opts += ' -atp rman'
    opts += ' -fr %s %s' % (fr[0], fr[1])
    opts += ' -step %s' % FS
    for n in nodes:
        opts += ' -rt %s' % n
    opts += ' -file %s' % filename
    cmds.AbcExport(j=opts, v=True)
    print "# Export alembic file '%s'" % filename
    return filename


#-------------------------------------------------------------------------------
#
#   Randomize Offset
#
#-------------------------------------------------------------------------------
def RandomizeOffsetByDxTimeOffset(nodes, minOffset=0.0, maxOffset=0.0, step=1.0):
    # setup offset type
    offsetStepList = [minOffset]
    while (True):
        value = offsetStepList[-1] + step
        if value > maxOffset:
            break
        offsetStepList.append(value)

    for node in nodes:
        if cmds.nodeType(node) == "pxrUsdReferenceAssembly":
            timeOffsetNode = cmds.listConnections("%s.time" % node, s=True, d=False)
            if not timeOffsetNode:
                timeOffsetNode = cmds.createNode("dxTimeOffset")
                cmds.connectAttr("time1.outTime", "%s.time" % timeOffsetNode)
                cmds.connectAttr("%s.outTime" % timeOffsetNode, "%s.time" % node)
            else:
                timeOffsetNode = timeOffsetNode[0]
            cmds.setAttr("%s.offset" % timeOffsetNode, offsetStepList[random.randint(0, len(offsetStepList) - 1)])

def ConnectTimeOffset(selected=None):
    if not selected:
        selected = cmds.ls(sl=True, type='pxrUsdReferenceAssembly')
        if not selected:
            selected = cmds.ls(sl=True, type='pxrUsdProxyShape')
    selected = cmds.ls(selected)
    for node in selected:
        name = node.split(':')[-1].split('|')[-1]
        connected = cmds.listConnections('%s.time' % node, s=True, d=False)
        if connected:
            ctype = cmds.nodeType(connected[0])
            if ctype != 'dxTimeOffset':
                offsetNode = cmds.createNode('dxTimeOffset', n='%s_TimeOffset' % name)
                cmds.connectAttr('%s.outTime' % connected[0], '%s.time' % offsetNode, f=True)
                cmds.connectAttr('%s.outTime' % offsetNode, '%s.time' % node, f=True)
        else:
            offsetNode = cmds.createNode('dxTimeOffset', n='%s_TimeOffset' % name)
            cmds.connectAttr('time1.outTime', '%s.time' % offsetNode, f=True)
            cmds.connectAttr('%s.outTime' % offsetNode, '%s.time' % node, f=True)


def ReloadReferenceAssembly():
    from pxr import UsdMaya
    selNodes = cmds.ls(sl = True, type = "pxrUsdReferenceAssembly", l = True)
    if not selNodes:
        selNodes = cmds.ls(type = "pxrUsdReferenceAssembly", l = True)

    for node in selNodes:
        UsdMaya.ReloadStage(node)
