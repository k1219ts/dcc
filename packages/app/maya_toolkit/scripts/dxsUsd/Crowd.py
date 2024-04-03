'''
USD Miarmy Crowd Export

Script Command
> Crowd.AgentExport(showDir=$showDir, asset=$assetName).doIt()
> Crowd.AgentExport(node=$nodeName, showDir=$showDir, asset=$assetName).doIt()
'''

import os, sys
import re
import random
import glob
import string
import json
import shutil

from pxr import Sdf, Usd, UsdGeom, UsdSkel, UsdShade, UsdUtils, Kind
import maya.cmds as cmds
import maya.mel as mel
import maya.api.OpenMaya as OpenMaya

import MsgSender
import dxsMsg
import Arguments
import PathUtils
import dxsMayaUtils
import dxsUsdUtils
import SessionUtils
import PackageUtils
import ClipUtils
import Attributes
import GeomMain
import GeomSkel
import EditPrim
import Texture
import Material
import Miarmy


#-------------------------------------------------------------------------------
#
#   ASSET
#
#-------------------------------------------------------------------------------
class AgentExport(Arguments.AssetArgs):
    '''
    '''
    def __init__(self, node=None, **kwargs):
        dxsMayaUtils.PluginSetup(['backstageMenu', 'pxrUsd'])
        Arguments.AssetArgs.__init__(self, **kwargs)

        self.node, self.agentType = AgentExport.GetNode(assetName=self.assetName, selected=node)
        assert self.node, '# msg : not found agent node.'
        if not self.outDir and self.assetDir:
            self.outDir = self.assetDir + '/agent/%s' % self.agentType
        self.computeVersion()

        self.mayafile = dxsMayaUtils.GetMayaFilename()
        if self.mayafile:
            self.comment = 'Generated with %s' % self.mayafile

        # for Developer debug.
        self.debugCreate = False

    @staticmethod
    def GetNode(assetName='', selected=None):
        node = None; agtype = None
        if selected:
            if selected.find('OriginalAgent_') > -1 and selected.find(assetName) > -1:
                node = cmds.ls(selected, l=True)[0]
        else:
            node = cmds.ls('OriginalAgent_%s*' % assetName, l=True)[0]
        if node:
            agtype = node.split('|')[-1].replace('OriginalAgent_', '')
        return node, agtype

    @staticmethod
    def SetTextureRandomize(objects):
        shadingEngines = list()
        for shape in objects:
            sg = cmds.listConnections(shape, type='shadingEngine')
            if not sg[0] in shadingEngines:
                shadingEngines.append(sg[0])
        for shape in objects:
            sg = cmds.listConnections(shape, type='shadingEngine')
            index = shadingEngines.index(sg[0])
            attrName = 'rman__riattr__user_texid'
            if not cmds.attributeQuery(attrName, n=shape, ex=True):
                cmds.addAttr(shape, ln=attrName, at='long')
            cmds.setAttr('%s.%s' % (shape, attrName), index)


    def doIt(self):
        assert self.node, '# [ERROR] Crowd.AgentExport - not found agent.'

        name = self.node.split('|')[-1]
        geomFile = '{DIR}/{VER}/{NAME}.geom.usd'.format(DIR=self.outDir, VER=self.version, NAME=name)
        self.geomExport(geomFile)
        skelFile = GeomSkel.AgentSkelSetup(geomFile, self.node).doIt()

        geomMasterFile = self.makeGeomPackage(skelFile)
        taskPayloadFile= self.makeAgentPackage(geomMasterFile)
        if self.showDir:
            PackageUtils.AssetPackage(self.showDir, self.assetName, taskPayloadFile)


    def makeGeomPackage(self, geomFile):
        rootName = self.node.split('|')[-1]
        SdfPath  = '/' + rootName

        masterFile = '{DIR}/{VER}/{NAME}.usd'.format(DIR=self.outDir, VER=self.version, NAME=self.agentType)
        SessionUtils.MakeReferenceStage(masterFile, [(geomFile, None)], SdfPath=SdfPath, Type='SkelRoot', comment=self.comment, clear=True)
        # Set Materials
        Material.Main(masterFile)
        AgentExport.OverridePrvMtl(masterFile)

        masterPayloadFile = masterFile.replace('.usd', '.payload.usd')
        SdfPath += '{agentVersion=%s}' % self.version
        SessionUtils.MakeReferenceStage(masterPayloadFile, [(masterFile, None)], SdfPath=SdfPath, Type='SkelRoot', comment=self.comment, clear=True)
        return masterPayloadFile

    def makeAgentPackage(self, sourceFile):
        rootName = self.node.split('|')[-1]
        typeFile = '{DIR}/{NAME}.usd'.format(DIR=self.outDir, NAME=self.agentType)
        SdfPath = '/%s{agentVersion=%s}' % (rootName, self.version)
        SessionUtils.MakeSubLayerStage(typeFile, [sourceFile], SdfPath=SdfPath)
        typePayloadFile = typeFile.replace('.usd', '.payload.usd')
        SdfPath = '/%s{agentVariant=%s}' % (self.assetName, self.agentType)
        SessionUtils.MakeReferenceStage(typePayloadFile, [(typeFile, None)], SdfPath=SdfPath, clear=True)

        taskFile = '{DIR}/agent/agent.usd'.format(DIR=self.assetDir)
        SessionUtils.MakeSubLayerStage(taskFile, [typePayloadFile])
        taskPayloadFile = taskFile.replace('.usd', '.payload.usd')
        SdfPath = '/%s{taskVariant=agent}' % self.assetName
        SessionUtils.MakeReferenceStage(taskPayloadFile, [(taskFile, None)], SdfPath=SdfPath, clear=True)
        return taskPayloadFile


    def geomExport(self, filename): # filename: *.geom.usd
        node = self.node.replace('OriginalAgent_', 'Geometry_')
        if not cmds.objExists(node):
            assert False, '# msg : not found agent geometry group.'

        objects = cmds.ls(node, dag=True, type='surfaceShape', ni=True)
        dxsMayaUtils.UpdateTextureAttributes(objects, asset=self.assetName)
        AgentExport.SetTextureRandomize(objects)

        # subdivision
        for shape in objects:
            if cmds.attributeQuery('USD_ATTR_subdivisionScheme', n=shape, ex=True):
                getVal = cmds.getAttr('%s.USD_ATTR_subdivisionScheme' % shape)
                if getVal == 'none':
                    cmds.setAttr('%s.USD_ATTR_subdivisionScheme' % shape, 'catmullClark', type='string')

        geomClass = GeomMain.Export(filename, node, userAttr=True, mtlAttr=True, subdivAttr=True, isAbc=False)
        geomClass.overrideCommand = {'defaultMeshScheme': 'catmullClark'}
        geomClass.doIt()


    @staticmethod
    def OverridePrvMtl(filename):
        stage = Usd.Stage.Open(filename)
        dprim = stage.GetDefaultPrim()

        vset = dxsUsdUtils.VariantSelection(dprim, 'preview', 'on')
        with vset.GetVariantEditContext():
            OverrideAgent.proc_diffC(stage, dprim, 1)

        stage.GetRootLayer().Save()
        del stage



#-------------------------------------------------------------------------------
#
#   SHOT
#
#-------------------------------------------------------------------------------
class CrowdShotExport(Arguments.ShotArgs):
    '''
    Args
        selectionOnly (bool): selected agent export
        sceneFile (str) : if batch mode, use this option
    '''
    def __init__(self, selectionOnly=False, expfr=(0, 0), task='', sceneFile='', **kwargs):
        dxsMayaUtils.PluginSetup(['backstageMenu', 'pxrUsd'])

        Arguments.ShotArgs.__init__(self, **kwargs)
        if not self.outDir and self.shotDir:
            self.outDir = self.shotDir + '/crowd'

        # Member variables
        self.selectionOnly = selectionOnly
        self.expfr = expfr
        self.task  = task
        self.sceneFile = sceneFile

        # Initialize Miarmy
        if sceneFile:
            self.preCompute()

        self.fps = dxsMayaUtils.GetFPS()
        # Frame Range
        if not self.fr[0] or not self.fr[1]:
            self.fr = dxsMayaUtils.GetFrameRange()
        if not self.expfr[0] and not self.expfr[1]:
            self.expfr = (self.fr[0] - 1, self.fr[1] + 1)

        self.cacheFolder, self.cacheName, self.cachefr = Miarmy.GetCacheInfo()

        if self.expfr[0] < self.cachefr[0]:
            self.expfr = (self.cachefr[0], self.expfr[1])

        self.mayafile = dxsMayaUtils.GetMayaFilename()
        self.comment = 'Generated with %s' % self.mayafile

        self.computeVersion()

    def preCompute(self):
        if not Miarmy.InitializeMiarmy(self.sceneFile):
            MsgSender.sendMsg("[USD Crowd] : Miarmy initialize failed.", self.showName, self.shotName, self.user)

    def doIt(self):
        if not self.mayafile:
            dxsMsg.Print('warning', "[Must have to save current scene]")

        outDir= os.path.join(self.outDir, self.version)
        MSP   = Miarmy.SceneParse(self.outDir)
        if MSP.m_error:
            MsgSender.sendMsg("[USD AgentAssetFile] : not found\n\t->%s" % string.join(MSP.m_error, ', '), self.showName, self.shotName, self.user)

        if self.task != 'payload':
            #-------------------------------------------------------------------
            # dispatched in parallel

            # MeshDrive Cache Check
            if MSP.m_simData and MSP.m_simData['indexMap']:
                if not Miarmy.CheckMeshDriveCache():
                    MsgSender.sendMsg("[USD Crowd] : This scene is included simulation meshes. Not found 'MeshDrive Cache'", self.showName, self.shotName, self.user)

            # pre-frame setup
            brainNode   = mel.eval('McdSimpleCommand -execute 3;')
            solverFrame = int(cmds.getAttr('%s.startTime' % brainNode))
            solverFrame-= 1
            for f in range(solverFrame, self.expfr[0]):
                cmds.currentTime(f)

            self.MakeSkelGeom(outDir, self.expfr, MSP.m_agentData)

            # Simulation Mesh
            if MSP.m_simData and MSP.m_simData['indexMap']:
                self.ExportSimulationMesh(outDir, MSP.m_simData)

        if self.task != 'geom':
            #-------------------------------------------------------------------
            # single task process

            # write MSP
            mspFile = '{DIR}/MSP.json'.format(DIR=outDir)
            with open(mspFile, 'w') as wfile:
                data = {'simData': MSP.m_simData, 'agentGroups': GeomSkel.GetAgentGroupData()}
                json.dump(data, wfile, indent=4)

            PackageSkelAnimation(outDir, fr=self.fr, expfr=self.expfr, fps=self.fps, comment=self.comment).doIt()


    def MakeSkelGeom(self, outDir, frameRange, argsData):
        selAgents = cmds.ls(sl=True, dag=True, type='McdAgent') if self.selectionOnly else list()

        for f in range(frameRange[0], frameRange[1] + 1):
            cmds.currentTime(f)
            fetchJointData = list()
            if self.selectionOnly:
                for n in selAgents:
                    cmds.select(n)
                    fetchJointData.extend(mel.eval('McdAgentMatchCmd -mm 4'))
                argsData['allAgents'] = selAgents
            else:
                fetchJointData = mel.eval('McdAgentMatchCmd -mm 3')
            argsData['fetchJointData'] = fetchJointData

            skelfile = os.path.join(outDir, 'crowd.skel.%04d.usd' % f)
            sys.stderr.write('%04d\t' % f)
            GeomSkel.ExportSkelAnimation(skelfile, **argsData).doIt()
            sys.stderr.write('\n')


    def ExportSimulationMesh(self, outDir, simData):
        if not self.sceneFile:
            Miarmy.MiarmyMeshDriveExport()
        if not Miarmy.InitMeshDrive():
            return

        # AgentGroup
        agentGroups = GeomSkel.GetAgentGroupData()

        objects = self.getSimulationMeshes(simData)
        dxsMayaUtils.UpdateTextureAttributes(objects)
        UsdAttr = dxsMayaUtils.UsdGeomAttributes(objects, user=True, mtl=True, subdiv=True)
        UsdAttr.Set()

        for f in range(self.expfr[0], self.expfr[1]+1):
            cmds.currentTime(f)
            meshfile = os.path.join(outDir, 'crowd.mesh.%04d.usd' % f)
            dxsMayaUtils.UsdExport(meshfile, objects, FR=(f, f))

        UsdAttr.Clear()

    def getSimulationMeshes(self, simData):
        meshes = list()
        for i in simData['assetData']:
            for n in simData['assetData'][i]:
                for shape in cmds.ls('MDG_%s*' % n, l=True):
                    if shape.find('MDGGRPMASTER') > -1:
                        meshes.append(shape)
        return meshes



class UsdSkelBakeOnly:
    '''
    Args
        inputPath (str): skel file or version dir
    '''
    def __init__(self, inputPath):
        self.rootDir   = inputPath

        self.skelfiles = list()
        if os.path.isdir(inputPath):
            self.skelfiles = self.getSkelFiles(inputPath)
        else:
            self.rootDir = os.path.dirname(inputPath)
            self.skelfiles.append(inputPath)

    def getSkelFiles(self, dirPath):
        fileRule = '{DIR}/crowd.skel.[0-9]*.usd'.format(DIR=dirPath)
        return glob.glob(fileRule)

    def doIt(self):
        geomFiles = list()
        for sf in self.skelfiles:
            sys.stderr.write('> %s\n' % sf)
            geomFile = GeomSkel.ExportBakeGeom(sf)
            geomFiles.append(geomFile)
            sys.stderr.write('\n')

        meshFiles = glob.glob(self.rootDir + '/crowd.mesh.[0-9]*.usd')
        if meshFiles:
            RepresentSimMesh(geomFiles)

        topologyFile = self.rootDir + '/crowd.geom.topology.usd'
        topologyLayer= Sdf.Layer.FindOrOpen(topologyFile)
        if topologyLayer:
            topologyLayer.Clear()
            UsdUtils.StitchClipsTopology(topologyLayer, [geomFiles[0]])




#-------------------------------------------------------------------------------
#
#   Skel Animation Package.
#
#-------------------------------------------------------------------------------
def ComputeMergeFrames(refFile, fr):
    refSize = os.path.getsize(refFile)
    refSize = refSize / (1000.0 * 1000.0 * 1000.0) # GB

    numframes = fr[1] - fr[0] + 1
    totalSize = refSize * numframes
    limitSize = 10.0    # default 10.0 GB

    if refSize < (limitSize * 0.5):
        dxsMsg.Print('info', "[Crowd.ComputeMergeFrames]")
        dxsMsg.Print('info', "\t-> sample size : %.3fGB" % refSize)
        dxsMsg.Print('info', "\t-> estimate total size : %.3fGB" % totalSize)

        framesPerFile = int(numframes / (totalSize / limitSize))
        frames = list()
        for frame in range(fr[0], fr[1] + 1, framesPerFile):
            endframe = frame + (framesPerFile - 1)
            if endframe >= fr[1]:
                endframe = fr[1]
            frames.append((frame, endframe))
        dxsMsg.Print('info', "\t-> frames : %s" % frames)
        return frames

class PackageSkelAnimation:
    '''
    '''
    def __init__(self, outDir, fr=(1, 1), expfr=None, fps=24.0, comment=None):
        self.outDir= outDir
        self.fr    = fr
        self.expfr = expfr
        if not expfr:
            self.expfr = (self.fr[0] - 1, self.fr[1] + 1)
        self.fps = fps
        self.digit = len(str(int(self.expfr[1])))
        self.comment  = comment
        self.showName = None
        self.shotDir  = None
        self.seqName  = None
        self.shotName = None
        self.version  = None
        self.pathParse()

    def pathParse(self):
        splitPath = self.outDir.split('/')
        if 'show' in splitPath:
            index = splitPath.index('show')
            self.showDir = string.join(splitPath[:index+2], '/')
            self.showName= splitPath[index+1]
        if 'shot' in splitPath:
            index = splitPath.index('shot')
            self.seqName = splitPath[index+1]
            self.shotName= splitPath[index+2]
        self.version = splitPath[-1]


    def doIt(self):
        skelRule = self.outDir + '/crowd.skel.*.usd'
        skelFiles= ClipUtils.GetPerFrameFiles(skelRule, self.expfr)

        # remove reference for flatten merge, after that add reference
        referenceMap, deActiveList = PackageSkelAnimation.GetSkelRootStatus(skelFiles[0])

        # sample skin bake
        # mergeFrames = ComputeMergeFrames(skelFiles[0], self.expfr)
        mergeFrames = list()
        if mergeFrames:
            for sf in skelFiles:
                PackageSkelAnimation.RemoveSkelRootCompositionArcs(sf)
        else:
            for sf in skelFiles:
                PackageSkelAnimation.SetSkelRootStatus(sf, acts=deActiveList)

        ClipUtils.MergeCoalesceFiles(skelRule, self.expfr, mergeFrames=mergeFrames, modify=False, clipSet='skelClip').doIt()
        if mergeFrames:
            # add reference, setActive, etc...
            for sf in glob.glob(self.outDir + '/crowd.skel.*.usd'):
                PackageSkelAnimation.SetSkelRootStatus(sf, referenceMap, deActiveList)

        # simulation mesh
        RepresentSimMesh(self.outDir, self.expfr).doIt()

        self.MakePackage()


    @staticmethod
    def RemoveSkelRootCompositionArcs(filename):
        stage = Usd.Stage.Open(filename, load=Usd.Stage.LoadNone)
        for p in iter(Usd.PrimRange.AllPrims(stage.GetDefaultPrim())):
            if p.GetTypeName() == 'SkelRoot':
                # p.GetReferences().ClearReferences()
                p.GetPayloads().ClearPayloads()
        stage.GetRootLayer().Save()

    @staticmethod
    def GetSkelRootStatus(filename):
        '''
        Query prepend references , active
        Returns:
            (dict) : referenceMap {'string path': Sdf.Reference, ...}
            (list) : deactive string path list
        '''
        referenceMap = dict()
        deActiveList = list()

        stage = Usd.Stage.Open(filename, load=Usd.Stage.LoadNone)
        for p in iter(Usd.PrimRange.AllPrims(stage.GetDefaultPrim())):
            if p.GetTypeName() == 'SkelRoot':
                pathStr = p.GetPath().pathString
                # refs = p.GetMetadata('references')
                # if refs:
                #     referenceMap[pathStr] = refs.prependedItems[0]
                pyl = p.GetMetadata('payload')
                if pyl:
                    referenceMap[pathStr] = pyl.explicitItems[0]
            if not p.IsActive():
                deActiveList.append(p.GetPath().pathString)

        return referenceMap, deActiveList

    @staticmethod
    def SetSkelRootStatus(filename, refs=dict(), acts=list()):
        '''
        Add reference, setActive, CreateAnimationSourceRel, ComputeExtent
        Args
            refs (dict): reference dict data {pathString: Sdf.Reference, ...}
            acts (list): de-active list data
        '''
        stage = Usd.Stage.Open(filename)
        fstart= int(stage.GetStartTimeCode())
        fend  = int(stage.GetEndTimeCode())

        for pstr in refs:
            prim = stage.GetPrimAtPath(pstr)
            if prim:
                # prim.GetReferences().AddReference(refs[pstr])
                prim.SetPayload(refs[pstr])
        for pstr in acts:
            prim = stage.OverridePrim(pstr)
            if prim:
                prim.SetActive(False)

        for p in iter(Usd.PrimRange.AllPrims(stage.GetDefaultPrim())):
            if p.GetTypeName() == 'SkelRoot':
                binding = UsdSkel.BindingAPI.Apply(p)
                binding.CreateAnimationSourceRel().SetTargets([Sdf.Path('SkelAnim')])

                # Extent
                for f in range(fstart, fend+1):
                    geom = UsdSkel.Root(p)
                    extent = UsdGeom.Boundable.ComputeExtentFromPlugins(geom, Usd.TimeCode(f))
                    if extent:
                        geom.GetExtentAttr().Set(extent, Usd.TimeCode(f))

                # Geometry Randomize VariantSet
                gprim = p.GetChild('Geometry')
                if gprim:
                    agentId = p.GetAttribute('primvars:agentIndex').Get()
                    random.seed(agentId - 1)
                    for name in gprim.GetVariantSets().GetNames():
                        if name.startswith('random_'):
                            vset  = gprim.GetVariantSets().GetVariantSet(name)
                            values= vset.GetVariantNames()
                            index = random.randint(0, len(values) - 1)
                            vset.SetVariantSelection(values[index])

        stage.GetRootLayer().Save()
        del stage


    def MakePackage(self):
        skelFile = '{DIR}/crowd.skel.usd'.format(DIR=self.outDir)
        OverrideAgent(skelFile)

        source = list()
        source.append((skelFile, None))
        meshFile = '{DIR}/crowd.mesh.usd'.format(DIR=self.outDir)
        if os.path.exists(meshFile):
            source.append((meshFile, None))

        # crow master
        crowdfile = '{DIR}/crowd.usd'.format(DIR=self.outDir)
        SessionUtils.MakeReferenceStage(crowdfile, source, SdfPath='/Miarmy', fr=self.fr, fps=self.fps, Kind='assembly', composite='reference', comment=self.comment, clear=True)

        crowdpayload = '{DIR}/crowd.payload.usd'.format(DIR=self.outDir)
        SessionUtils.MakeReferenceStage(crowdpayload, [(crowdfile, None)], SdfPath='/Miarmy{crowdVersion=%s}' % self.version, fr=self.fr, fps=self.fps, Kind='assembly', comment=self.comment, clear=True)

        if not self.showDir or not self.shotName:
            return

        outDir = '{SHOW}/shot/{SEQ}/{SHOT}/crowd'.format(SHOW=self.showDir, SEQ=self.seqName, SHOT=self.shotName)
        # collect version geom
        collect_file = '{DIR}/crowd.usd'.format(DIR=outDir)
        SessionUtils.MakeSubLayerStage(collect_file, [crowdpayload])

        collect_master = '{DIR}/crowd.payload.usd'.format(DIR=outDir)
        SessionUtils.MakeReferenceStage(collect_master, [(collect_file, None)], SdfPath='/shot/crowd', Name=self.shotName)

        PackageUtils.ShotPackage(self.showDir, self.seqName, self.shotName, collect_master, fr=self.fr, fps=self.fps)

        self.overrideVersion()

    def overrideVersion(self):
        shotusd = '{SHOW}/shot/{SEQ}/{SHOT}/{SHOT}.usd'.format(SHOW=self.showDir, SEQ=self.seqName, SHOT=self.shotName)
        PackageUtils.VersionSelect(shotusd, '/shot/crowd', 'crowdVersion', self.version)

        shotLgtFile = shotusd.replace('.usd', '.lgt.usd')
        if os.path.exists(shotLgtFile):
            PackageUtils.VersionSelect(shotLgtFile, '/shot/crowd', 'crowdVersion', self.version)




#-------------------------------------------------------------------------------
class RepresentSimMesh:
    '''
    Simulation mesh represent
        add collection for preview and prman material
    '''
    def __init__(self, rootDir, fr):
        self.rootDir = rootDir
        self.fr = fr
        # read MSP.json
        mspFile = os.path.join(rootDir, 'MSP.json')
        self.MSP= json.load(open(mspFile))

    def doIt(self):
        meshRule = self.rootDir + '/crowd.mesh.*.usd'
        meshFiles= ClipUtils.GetPerFrameFiles(meshRule, self.fr)
        if meshFiles:
            dstroots = list()
            self.agentAssetMap = self.GetAgentAssetMap()
            for f in meshFiles:
                dstroots = self.proc(f)
            mergeFrames = ComputeMergeFrames(meshFiles[0], self.fr)
            ClipUtils.MergeCoalesceFiles(meshRule, self.fr, mergeFrames=mergeFrames, modify=False, clipSet='meshClip').doIt()

            self.postSetup(dstroots)


    def GetAgentAssetMap(self):
        result = dict()
        rootLayer = Sdf.Layer.FindOrOpen(self.rootDir + '/crowd.skel.usd')
        if not rootLayer:
            return

        stage = Usd.Stage.Open(rootLayer)

        simData = self.MSP['simData']
        for i in simData['indexMap']:
            aid = simData['indexMap'][i]['aid']
            dstpath = GeomSkel.GetAgentParentPath(aid, self.MSP['agentGroups'])
            dstpath+= '/Agent' + str(aid)
            prim = stage.GetPrimAtPath(dstpath)
            if prim:
                refs = prim.GetMetadata('references')
                assetPath = refs.prependedItems[0].assetPath
                absPath   = assetPath
                if not assetPath.startswith('/'):
                    absPath = os.path.abspath(os.path.join(self.rootDir, assetPath))
                result[prim.GetPath().pathString] = absPath
        return result

    def proc(self, filename):   # meshfile
        stage = Usd.Stage.Open(filename, load=Usd.Stage.LoadNone)

        dstroots = list()
        edit  = Sdf.BatchNamespaceEdit()
        simData = self.MSP['simData']
        for i in simData['indexMap']:
            aid = simData['indexMap'][i]['aid']
            tid = simData['indexMap'][i]['tid']

            meshpath = '/MDGGRPMASTER/MDGGrp_%s' % i
            mdgprim  = stage.GetPrimAtPath(meshpath)
            if mdgprim:
                children = mdgprim.GetAllChildren()
                for mesh in simData['assetData'][str(tid)]:
                    for prim in children:
                        name   = prim.GetName()
                        srcpath= prim.GetPath()
                        if name.find('MDG_' + mesh) > -1:
                            dstpath = GeomSkel.GetAgentParentPath(aid, self.MSP['agentGroups'])
                            dstpath+= '/Agent' + str(aid) + '_sim'
                            self.GetPrim(stage, dstpath)
                            agprim = stage.GetPrimAtPath(dstpath)
                            dxsUsdUtils.CreateConstPrimvar(agprim, 'agentIndex', int(aid), Sdf.ValueTypeNames.Int)
                            dxsUsdUtils.CreateConstPrimvar(agprim, 'agentTypeIndex', int(tid), Sdf.ValueTypeNames.Int)
                            dstroots.append(dstpath)
                            dstpath = Sdf.Path(dstpath).AppendChild(mesh)

                            edit.Add(srcpath, dstpath)

        edit.Add(Sdf.Path('/MDGGRPMASTER'), Sdf.Path.emptyPath)

        rootLayer = stage.GetRootLayer()
        rootLayer.Apply(edit)

        dprim = stage.GetPrimAtPath('/Miarmy')
        if dprim:
            stage.SetDefaultPrim(dprim)

        tmpfile = filename.replace('.usd', '_tmp.usd')
        rootLayer.Export(tmpfile, args={'format': 'usdc'})
        shutil.copy(tmpfile, filename)
        os.remove(tmpfile)

        return dstroots


    def GetPrim(self, stage, primPath):
        prim = stage.GetPrimAtPath(primPath)
        if not prim:
            for sp in Sdf.Path(primPath).GetPrefixes():
                if not stage.GetPrimAtPath(sp):
                    stage.DefinePrim(sp, 'Xform')


    def postSetup(self, agents): # agents is agent root path(str)
        meshFiles = glob.glob(self.rootDir + '/crowd.mesh.*.usd')
        for f in meshFiles:
            stage = Usd.Stage.Open(f)
            self.editAttribute(stage, agents)
            stage.GetRootLayer().Save()

    def editAttribute(self, stage, agents):
        for agp in agents:
            assetPath = self.agentAssetMap[agp.replace('_sim', '')]
            prim = stage.GetPrimAtPath(agp)

            aidAttr = prim.GetAttribute('primvars:agentIndex')
            if aidAttr:
                aid = aidAttr.Get()
                random.seed(aid * 1000)
                val = random.randint(1, 9)
                dxsUsdUtils.CreateConstPrimvar(prim, 'txVarNum', val, Sdf.ValueTypeNames.Int)

            for p in iter(Usd.PrimRange.AllPrims(prim)):
                Usd.ModelAPI(p).SetKind(Kind.Tokens.component)
                names = p.GetAuthoredPropertyNames()
                if 'primvars:txBasePath' in names and 'primvars:txLayerName' in names:
                    txBasePath = p.GetAttribute('primvars:txBasePath').Get()
                    splitPath  = txBasePath.split('/')
                    rootName   = splitPath[splitPath.index('texture') - 1]
                    txLayerName= p.GetAttribute('primvars:txLayerName').Get()

                    # Create Class Prim
                    cprimpath = Sdf.Path('/_{NAME}_{LAYER}_txAttr'.format(NAME=rootName, LAYER=txLayerName))
                    if not stage.GetPrimAtPath(cprimpath):
                        txAttrFile = '{PATH}/tex/tex.attr.usd'.format(PATH=txBasePath)
                        if not txAttrFile.startswith('/'):
                            splitPath = assetPath.split('/')
                            showDir   = '/'.join(splitPath[:splitPath.index('show') + 2])
                            txAttrFile= showDir + '/' + txAttrFile
                        cprim = stage.CreateClassPrim(cprimpath)
                        cprim.SetPayload(Sdf.Payload(PathUtils.GetRelPath(self.rootDir, txAttrFile), Sdf.Path('/' + txLayerName)))
                        # modelVersion
                        modelVersionAttr = p.GetAttribute('primvars:modelVersion')
                        if modelVersionAttr:
                            vset = dxsUsdUtils.VariantSelection(cprim, 'modelVersion', modelVersionAttr.Get())

                    # Remove Attribute
                    p.RemoveProperty('primvars:txBasePath')
                    p.RemoveProperty('primvars:txLayerName')

                    # Inherit Class Prim
                    p.GetInherits().AddInherit(cprimpath)




class OverrideAgent:
    def __init__(self, filename):
        self.filename = filename

        self.doIt()

    def doIt(self):
        rootLayer = Sdf.Layer.FindOrOpen(self.filename)
        if not rootLayer:
            return

        stage = Usd.Stage.Open(rootLayer)

        for p in iter(Usd.PrimRange.AllPrims(stage.GetPseudoRoot())):
            if 'Agent' in p.GetName():
                # preview material
                attr = p.GetAttribute('primvars:txVarNum')
                if attr:
                    OverrideAgent.proc_diffC(stage, p, attr.Get())

        stage.GetRootLayer().Save()


    @staticmethod
    def GetTxPath(prim):
        stack = prim.GetPrimStack()
        prvDir= None
        for s in stack:
            f = s.layer.identifier
            if os.path.basename(f) == 'prv_mtl.usd':
                prvDir = os.path.dirname(f)
        return prvDir

    @staticmethod
    def proc_diffC(stage, prim, txVarNum):
        stageFile = stage.GetRootLayer().identifier
        prvPath = prim.GetPath().AppendPath('Materials/preview')
        prvPrim = stage.GetPrimAtPath(prvPath)
        if prvPrim:
            for p in iter(Usd.PrimRange.AllPrims(prvPrim)):
                if p.GetName() == 'diffC_Tex':
                    sdfpath = p.GetPath()
                    overprim= stage.OverridePrim(sdfpath)
                    txPath  = OverrideAgent.GetTxPath(overprim)

                    shader  = UsdShade.Shader(overprim)
                    fileAttr= shader.GetInput('file')
                    if txPath and fileAttr:
                        value = fileAttr.Get().path
                        baseName = value.split('/')[-1]

                        udim = False
                        if len(baseName.split('.')) > 2:
                            udim = True

                        splitPath = baseName.split('.')[0].split('_')
                        if re.match('\d+', splitPath[-1]):
                            splitPath[-1] = str(txVarNum)
                        else:
                            splitPath.append(str(txVarNum))

                        name = '_'.join(splitPath)
                        if udim:
                            name += '.<UDIM>'
                        name += '.jpg'
                        fullname = os.path.join(txPath, name)
                        fileAttr.Set(PathUtils.GetRelPath(stageFile, fullname))
