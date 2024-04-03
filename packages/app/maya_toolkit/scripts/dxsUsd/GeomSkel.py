import os, sys
import string
import re
import random
import math
import time

from pxr import Usd, UsdGeom, UsdSkel, UsdUtils, Sdf, Kind, Gf, Vt
import maya.api.OpenMaya as OpenMaya
import maya.api.OpenMayaAnim as OpenMayaAnim
import maya.cmds as cmds

import PathUtils
import dxsMayaUtils
import dxsUsdUtils
import ClipUtils
import SessionUtils
import PackageUtils
import Attributes


class GetJoints:
    def __init__(self, node):
        self.node = cmds.ls(node, l=True)[0]
        self.allJoints = cmds.ls(self.node, dag=True, type='joint', l=True)

        allJointsPath = list()
        for j in self.allJoints:
            allJointsPath.append(self.getJointPath(j))

        splitStr = allJointsPath[0].split('_')
        prefix = splitStr[0] + '_'
        suffix = '_' + '_'.join(splitStr[2:])
        self.allJointsPath = list()
        self.allJointsName = list()
        for j in allJointsPath:
            n = j.replace(prefix, '').replace(suffix, '')
            self.allJointsPath.append(n)
            self.allJointsName.append(n.split('/')[-1])


    def getJointPath(self, joint):
        jpath = joint.replace(self.node, '').replace('|', '/')[1:]
        return jpath

    def getOrientList(self):
        orients = list()
        for j in self.allJoints:
            jorient = cmds.getAttr('%s.jointOrient' % j)[0]
            quat = OpenMaya.MEulerRotation(math.radians(jorient[0]), math.radians(jorient[1]), math.radians(jorient[2]), 0).asQuaternion()
            orients.append(quat)
        return orients


def GetClusterMap():
    result = dict()
    for c in cmds.ls(type='skinCluster'):
        geoms = cmds.skinCluster(c, q=True, g=True)
        if geoms:
            for g in geoms:
                result[cmds.ls(g, l=True)[0]] = c
    return result


class AgentSkelSetup:
    '''
    Args
        geomfile (str): geometry filename
        node (str): 'OriginalAgent_*' node
    '''
    def __init__(self, geomfile, node):
        self.geomfile = geomfile
        self.node = cmds.ls(node, l=True)[0]

        self.outfile = geomfile.replace('.geom.', '.skel.')
        self.JT = GetJoints(self.node)
        self.ClusterMap = GetClusterMap()   # {shapename(fullpath): clustername, ...}

    def doIt(self):
        self.EditGeom()

        # create *.skel.usd
        stage = SessionUtils.MakeInitialStage(self.outfile)
        rootName = self.node.split('|')[-1]
        skelRoot = UsdSkel.Root.Define(stage, '/' + rootName)
        skelPrim = skelRoot.GetPrim()
        stage.SetDefaultPrim(skelPrim)

        geoPrim = stage.DefinePrim(skelPrim.GetPath().AppendChild('Geometry'), 'Xform')
        geoPrim.SetPayload(Sdf.Payload('./' + os.path.basename(self.geomfile)))
        # default txVarNum
        dxsUsdUtils.CreateConstPrimvar(geoPrim, 'txVarNum', 1, Sdf.ValueTypeNames.Int)

        for p in geoPrim.GetAllChildren():
            if p.GetTypeName() == 'Xform':
                self.variantActiveSet(geoPrim, p)

        self.makeSkeleton(stage, skelPrim)

        stage.GetRootLayer().Save()
        del stage
        return self.outfile


    def makeSkeleton(self, stage, parent): # parent prim
        rig = stage.DefinePrim(parent.GetPath().AppendChild('Rig'), 'Scope')
        Usd.ModelAPI(rig).SetKind(Kind.Tokens.component)
        skel = UsdSkel.Skeleton.Define(stage, rig.GetPath().AppendChild('Skel'))
        bindTransform, restTransform = self.getJointTransforms()
        skel.CreateBindTransformsAttr().Set(Vt.Matrix4dArray(bindTransform))
        skel.CreateRestTransformsAttr().Set(Vt.Matrix4dArray(restTransform))
        skel.CreateJointsAttr().Set(self.JT.allJointsPath)
        skel.CreateJointNamesAttr().Set(self.JT.allJointsName)

        # Skel purpose : guide
        UsdGeom.Scope(skel.GetPrim()).CreatePurposeAttr(UsdGeom.Tokens.guide)

        # BindingAPI
        binding = UsdSkel.BindingAPI.Apply(parent)
        binding.CreateSkeletonRel().SetTargets([Sdf.Path('Rig/Skel')])

    def getJointTransforms(self):
        bind = list(); rest = list()
        for j in self.JT.allJoints:
            wsMtx = cmds.xform(j, q=True, ws=True, m=True)  # BindTransform
            bind.append(Gf.Matrix4d(*wsMtx))
            osMtx = cmds.xform(j, q=True, os=True, m=True)  # RestTransform
            rest.append(Gf.Matrix4d(*osMtx))
        return bind, rest

    def variantActiveSet(self, parent, target):
        '''
        Args
            parent (Usd.Prim): 'Geometry' prim
            target (Usd.Prim): random-root prim
        '''
        name = 'random_' + target.GetName()
        vset = parent.GetVariantSets().AddVariantSet(name)
        for c in target.GetAllChildren():
            value = c.GetName()
            vset.AddVariant(value)
            vset.SetVariantSelection(value)
            with vset.GetVariantEditContext():
                c.SetActive(True)


    def EditGeom(self):
        attrFile = self.geomfile.replace('.geom.usd', '.attr.usd')
        Attributes.ExtractAttr(self.geomfile, attrFile=attrFile).doIt()

        stage = Usd.Stage.Open(self.geomfile)
        dprim = stage.GetDefaultPrim()

        oid = 0
        for p in iter(Usd.PrimRange.AllPrims(dprim)):
            if p.GetTypeName() == 'Mesh':
                # remove extent
                p.RemoveProperty('extent')
                # Set Kind
                Usd.ModelAPI(p).SetKind(Kind.Tokens.component)
                # Add object_id
                dxsUsdUtils.AddPrimvar(UsdGeom.Mesh(p), 'object_id', Sdf.ValueTypeNames.Int, UsdGeom.Tokens.constant, oid)
                # Add Skel Attributes
                self.setMeshSkelAttributes(p)
                oid += 1

        # set active for geometry randomization
        for p in dprim.GetAllChildren():
            if p.GetTypeName() == 'Xform':
                for c in p.GetAllChildren():
                    c.SetActive(False)

        stage.GetRootLayer().Save()

    def setMeshSkelAttributes(self, prim):
        primPathStr = prim.GetPath().pathString

        shapename = cmds.ls(primPathStr.split('/')[-1], dag=True, type='surfaceShape', ni=True, l=True)[0]
        if not self.ClusterMap.has_key(shapename):
            return
        cluster = self.ClusterMap[shapename]

        skinClusterFn = OpenMayaAnim.MFnSkinCluster(dxsMayaUtils.GetMObject(cluster, False))
        influenceDagList = skinClusterFn.influenceObjects()

        influenceIndices = list()
        for j in influenceDagList:
            jp = j.fullPathName()
            index = self.JT.allJoints.index(jp)
            influenceIndices.append(index)

        shapeObject = dxsMayaUtils.GetMObject(shapename)
        meshFn      = OpenMaya.MFnMesh(shapeObject)
        numVertices = meshFn.numVertices
        singleIdComp= OpenMaya.MFnSingleIndexedComponent()
        vertexComp  = singleIdComp.create(OpenMaya.MFn.kMeshVertComponent)

        weightData  = skinClusterFn.getWeights(shapeObject, vertexComp)
        weights     = weightData[0]
        numInfluences = int(weightData[1])
        maxInfluences = numInfluences

        JointIndices = [0]   * maxInfluences * numVertices
        JointWeights = [0.0] * maxInfluences * numVertices

        for vtx in range(numVertices):
            inputOffset = vtx * numInfluences
            outputOffset= vtx * maxInfluences
            for i in range(numInfluences):
                weight = weights[inputOffset + i]
                if not Gf.IsClose(weight, 0.0, 1e-8):
                    JointIndices[outputOffset] = influenceIndices[i]
                    JointWeights[outputOffset] = weight
                    outputOffset += 1

        binding = UsdSkel.BindingAPI.Apply(prim)
        binding.CreateGeomBindTransformAttr().Set(Gf.Matrix4d())
        binding.CreateJointIndicesPrimvar(constant=False, elementSize=numInfluences).Set(Vt.IntArray(JointIndices))
        binding.CreateJointWeightsPrimvar(constant=False, elementSize=numInfluences).Set(Vt.FloatArray(JointWeights))



def GetRandomPrims(filename, geomPath):
    '''
    -- 2019.07.24 - will be delete, does not need new agent asset
    Args:
        filename (str):
        geomPath (str): /OriginalAgent_$AGTYPE/Geometry_$AGTYPE
    Returns:
        [
            [Sdf.Path(xxx), Sdf.Path(xxx), ...],
            [Sdf.Path(xxx), ...]
        ]
    '''
    result = list()
    stage = Usd.Stage.Open(filename)
    rootPrim = stage.GetPrimAtPath(geomPath)
    if not rootPrim:
        return result
    for p in rootPrim.GetChildren():
        if p.GetTypeName() == 'Xform':
            paths = list()
            for sp in p.GetChildren():
                tmp = sp.GetPath().pathString.split('/')
                paths.append(Sdf.Path(string.join(tmp[2:], '/')))
            result.append(paths)
    return result

def GetRandomizeData(agentAssetList, agentAssetFiles):
    result = list()
    for i in range(len(agentAssetList)):
        name = agentAssetList[i].replace('OriginalAgent_', '')
        geomPath = '/OriginalAgent_{NAME}/Geometry_{NAME}'.format(NAME=name)
        result.append(GetRandomPrims(agentAssetFiles[i], geomPath))
    return result

def GetAgentGroupData():
    groups = dict()
    if cmds.pluginInfo('MiarmyForDexter', q=True, l=True):
        for n in cmds.ls(type='AgentGroup'):
            name = n.split(':')[-1].split('|')[-1]
            groups[name] = cmds.getAttr('%s.agentIds' % n)
    return groups

def GetAgentParentPath(agentId, agentGroups):
    '''
    Args:
        agentGroups (dict) : result of GetAgentGroupData
    '''
    primPath = '/Miarmy'
    if agentGroups:
        gname = None
        for n in agentGroups:
            if agentId in agentGroups[n]:
                gname = n
        if gname:
            primPath += '/' + gname
    return primPath


class ExportSkelAnimation:
    '''
    Args:
        allAgents (list): all agents in scene
        assetList (list): OriginalAgent name list
        assetFiles (list): agent type filename list
        jointsList (list):
        jointsOrientList (list):
        randomizeData (list):
        hideList (list): hide agent list
        fetchJointData (list):
    '''
    def __init__(self, outfilename, **kwargs):
        self.outfile = outfilename
        self.allAgents = list()
        self.assetList = list()
        self.assetFiles= list()
        self.assetJointsList = list()
        self.assetJointsOrientList = list()
        self.fetchJointData = list()
        self.randomizeData  = list()
        self.hideList = dict()
        self.usdformat = 'usdc'

        if kwargs.has_key('allAgents'):
            self.allAgents = kwargs['allAgents']
        if kwargs.has_key('assetList'):
            self.assetList = kwargs['assetList']
        if kwargs.has_key('assetFiles'):
            self.assetFiles= kwargs['assetFiles']
        if kwargs.has_key('jointsList'):
            self.assetJointsList = kwargs['jointsList']
        if kwargs.has_key('jointsOrientList'):
            self.assetJointsOrientList = kwargs['jointsOrientList']
        if kwargs.has_key('fetchJointData'):
            self.fetchJointData = kwargs['fetchJointData']
        if kwargs.has_key('randomizeData'):
            self.randomizeData = kwargs['randomizeData']
        if kwargs.has_key('hideList'):
            self.hideList = kwargs['hideList']

        self.currentFrame = cmds.currentTime(q=True)

    def initPath(self):
        self.agentRelPath = list()
        for f in self.assetFiles:
            self.agentRelPath.append(PathUtils.GetRelPath(self.outfile, f))

    def initAgentGroup(self, parent):
        self.agentGroups = GetAgentGroupData()
        if not self.agentGroups:
            return

        for name in self.agentGroups:
            self.stage.DefinePrim(parent.GetPath().AppendChild(name), 'Xform')


    def doIt(self):
        startTime = time.time()

        self.initPath()
        self.stage = SessionUtils.MakeInitialStage(self.outfile, usdformat=self.usdformat, clear=True)
        self.stage.SetStartTimeCode(self.currentFrame)
        self.stage.SetEndTimeCode(self.currentFrame)

        # customLayerData
        customLayerData = {}
        customLayerData['agentAssetFiles'] = string.join(self.assetFiles, ',')
        self.stage.GetRootLayer().customLayerData = customLayerData

        dprim = self.stage.DefinePrim('/Miarmy', 'Xform')
        self.stage.SetDefaultPrim(dprim)
        Usd.ModelAPI(dprim).SetKind(Kind.Tokens.assembly)
        # Custom Attributes
        self.makeCustomAttributes(dprim)

        # AgentGroup - query and create prim
        self.initAgentGroup(dprim)

        self.dataIndex = 0
        for i in xrange(len(self.allAgents)):
            self.dataIndex += 2
            aid = cmds.getAttr('%s.agentId' % self.allAgents[i])
            tid = cmds.getAttr('%s.tid' % self.allAgents[i])    # tempTypeId
            prim= self.stage.GetPrimAtPath(GetAgentParentPath(aid, self.agentGroups))
            self.makeSkelRoot(prim, aid, tid)

        self.stage.Load()

        # Texture Randomize & Hide
        self.AgentPostProcess()

        self.stage.GetRootLayer().Save()

        endTime = time.time()
        sys.stderr.write(' -Skel : %.3f sec' % (endTime - startTime))


    def makeCustomAttributes(self, parent):
        agentTypeAttr = parent.CreateAttribute('userProperties:Crowd:agentTypes', Sdf.ValueTypeNames.StringArray)
        typeList = list()
        for ag in self.assetList:
            typeList.append(ag.replace('OriginalAgent_', ''))
        agentTypeAttr.Set(Vt.StringArray(typeList))


    def makeSkelRoot(self, parent, agentId, typeId):
        currentJoints = self.assetJointsList[typeId]
        currentJointsOrient = self.assetJointsOrientList[typeId]

        skelRoot = UsdSkel.Root.Define(self.stage, parent.GetPath().AppendChild('Agent%d' % agentId))
        skelPrim = skelRoot.GetPrim()
        Usd.ModelAPI(skelPrim).SetKind(Kind.Tokens.assembly)
        # Reference Agent Asset
        # skelPrim.GetReferences().AddReference(Sdf.Reference(self.agentRelPath[typeId]))
        skelPrim.SetPayload(Sdf.Payload(self.agentRelPath[typeId]))

        skelanim = UsdSkel.Animation.Define(self.stage, skelPrim.GetPath().AppendChild('SkelAnim'))
        skelanim.CreateJointsAttr().Set(self.assetJointsList[typeId])

        T = list(); R = list(); S = list()
        # joint count iter
        for i in range(len(self.assetJointsList[typeId])):
            trans = self.fetchJointData[self.dataIndex:self.dataIndex+3]
            T.append(Gf.Vec3f(*trans))
            self.dataIndex += 3

            rotate = self.fetchJointData[self.dataIndex:self.dataIndex+3]
            orient = OpenMaya.MEulerRotation(math.radians(rotate[0]), math.radians(rotate[1]), math.radians(rotate[2]), 0).asQuaternion()
            setval = orient * currentJointsOrient[i]
            R.append(Gf.Quath(setval.w, setval.x, setval.y, setval.z).Normalize())
            self.dataIndex += 3

            scale = self.fetchJointData[self.dataIndex:self.dataIndex+3]
            if i == 0:
                S.append(Gf.Vec3f(*scale))
            else:
                S.append(Gf.Vec3f(1, 1, 1))
            self.dataIndex += 3

            nbData = self.fetchJointData[self.dataIndex]
            self.dataIndex += 1 + int(nbData)

        # -- 2019.08.13 - add agent translate
        skelRoot.AddTranslateOp().Set(T[0], Usd.TimeCode(self.currentFrame))
        # init root transform
        T[0] = Gf.Vec3f(0, 0, 0)

        skelanim.GetTranslationsAttr().Set(Vt.Vec3fArray(T), Usd.TimeCode(self.currentFrame))
        skelanim.GetRotationsAttr().Set(Vt.QuatfArray(R), Usd.TimeCode(self.currentFrame))
        skelanim.GetScalesAttr().Set(Vt.Vec3fArray(S), Usd.TimeCode(self.currentFrame))

        # -- 2019.07.24 - this apply, after skel files flatten merge -- (important)
        # binding = UsdSkel.BindingAPI.Apply(skelPrim)
        # binding.CreateAnimationSourceRel().SetTargets([skelanim.GetPrim().GetPath()])

        # custom primvar
        dxsUsdUtils.AddPrimvar(skelRoot, 'agentIndex', Sdf.ValueTypeNames.Int, UsdGeom.Tokens.constant, agentId)
        dxsUsdUtils.AddPrimvar(skelRoot, 'agentTypeIndex', Sdf.ValueTypeNames.Int, UsdGeom.Tokens.constant, typeId)


    @staticmethod
    def SetTextureRandomize(parent, agentId):
        random.seed(agentId * 1000)
        val = random.randint(1, 9)
        # SkelRoot
        dxsUsdUtils.CreateConstPrimvar(parent, 'txVarNum', val, Sdf.ValueTypeNames.Int)
        # Geometry
        gprim = parent.GetChild('Geometry')
        if gprim:
            dxsUsdUtils.CreateConstPrimvar(gprim, 'txVarNum', val, Sdf.ValueTypeNames.Int)

        # Child Geometry
        for p in iter(Usd.PrimRange(parent)):
            if p.GetTypeName() == 'Mesh':
                attr = p.GetAttribute('primvars:texid')
                if attr:
                    texid = attr.Get()
                    random.seed(agentId * 1000 + texid)
                    val = random.randint(1, 9)
                    dxsUsdUtils.CreateConstPrimvar(p, 'txVarNum', val, Sdf.ValueTypeNames.Int)

    def AgentPostProcess(self):
        '''
        Texture Randomize & Set Hide
        '''
        dprim = self.stage.GetDefaultPrim()
        for i in xrange(len(self.allAgents)):
            aid = cmds.getAttr('%s.agentId' % self.allAgents[i])
            prim= self.stage.GetPrimAtPath(dprim.GetPath().AppendChild('Agent%d' % aid))
            ExportSkelAnimation.SetTextureRandomize(prim, aid)
            if self.hideList and self.hideList[self.allAgents[i]] == 1:
                prim.SetActive(False)



def ExportBakeGeom(skelfile):
    geomfile = skelfile.replace('.skel', '.geom')
    if os.path.exists(geomfile):
        os.remove(geomfile)

    startTime = time.time()
    sys.stderr.write(' -Skel File :%s\n' % skelfile)

    stage = Usd.Stage.Open(skelfile)
    UsdSkel.BakeSkinning(stage.Traverse())
    stage.GetRootLayer().Export(geomfile)

    endTime = time.time()
    sys.stderr.write(' -SkelBake : %.3f sec' % (endTime - startTime))
    return geomfile
