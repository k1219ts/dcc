'''
USD Model Export

Script Command
> dxsUsd.ModelExport(node, showDir=$showDir, asset=$assetName).doIt()
> dxsUsd.ModelExport(node, isPurpose=True, ...).doIt()
> dxsUsd.ModelExport(node, isPurpose=True, isLod=True, ...).doIt()
> dxsUsd.ModelExport(node, isElement=True, isPurpose=True, isLod=True, version=$version).doIt()

Script Command
> dxsUsd.ModelClipExport(node, isPurpose=False, loopScales=[0.5, 1.0, 1.5], loopRange=[1001, 2000], overWrite=False, ...).doIt()

'''
import os

from pxr import Sdf, Usd, UsdGeom, Kind
import maya.cmds as cmds

import Arguments
import PathUtils
import dxsUsdUtils
import dxsMayaUtils
import EditPrim
import Attributes
import SessionUtils
import PackageUtils
import GeomCurves
import GeomMain
import ClipUtils
import Texture
import Material


class ModelExport(Arguments.AssetArgs):
    '''
    Model Asset Export
    Args:
        node (str): export root node name
        isElement (bool): asset element export
        isPrupose (bool):
        isLod (bool): lodVariant
        isVersion (bool): set modelVersion
    '''
    def __init__(self, node=None, isElement=False, isPurpose=False, isLod=False, overWrite=True, **kwargs):
        dxsMayaUtils.PluginSetup(['backstageMenu', 'pxrUsd', 'AbcExport'])
        self.isElement = isElement
        self.isPurpose = isPurpose
        self.isLod     = isLod
        self.overWrite = overWrite

        self.node = None
        self.elementName = None

        Arguments.AssetArgs.__init__(self, **kwargs)
        if not self.outDir and self.assetDir:
            if self.isElement:
                self.elementName = self.getElementName(node)
                self.outDir = '{DIR}/element/{NAME}/model'.format(DIR=self.assetDir, NAME=self.elementName)
            else:
                self.outDir = '{DIR}/model'.format(DIR=self.assetDir)
        self.computeVersion()

        self.node = ModelExport.GetNode(assetName=self.assetName, selected=node)
        self.mayafile = dxsMayaUtils.GetMayaFilename()
        if self.mayafile:
            self.comment = 'Generated with %s' % self.mayafile

    def getElementName(self, node):
        name = node.split('_model')[0]
        name = name.replace('%s_' % self.assetName, '', 1)
        return str(name)

    @staticmethod
    def GetNode(assetName='', selected=None):
        if selected:
            if selected.find(assetName) > -1:
                return cmds.ls(selected, l=True)[0]
        else:
            findName = assetName
            if not assetName:
                findName = '*'
            return cmds.ls('|%s_*model*' % findName, l=True)[0]


    def doIt(self):
        if not self.node:
            cmds.confirmDialog(
                title='Error!', message='# ModelExport : not found model',
                icon='warning', button=['OK']
            )
            return

        ns, nn = dxsMayaUtils.GetNamespaceInfo(self.node)
        nodeName, geomType = dxsMayaUtils.GetNodeInfo(self.node)
        geomFile = '{DIR}/{VER}/{NAME}.{TYPE}_geom.usd'.format(DIR=self.outDir, VER=self.version, NAME=nodeName, TYPE=geomType)
        _doit = True
        if os.path.exists(geomFile):
            _doit = self.overWrite
        if _doit:
            dxsMayaUtils.UpdateTextureAttributes(cmds.ls(self.node, dag=True, type='surfaceShape'), asset=self.assetName, element=self.elementName)
            geomClass = GeomMain.Export(geomFile, self.node, userAttr=True, mtlAttr=True, subdivAttr=True, nsLayer=ns, modelVersion=self.version)
            geomClass.excludeMeshAttributes = ['txVarNum', 'txVersion', 'modelVersion']
            geomClass.doIt()
            Attributes.ExtractAttr(geomFile).doIt()

        # Default Prim Name
        self.defaultName = self.assetName if not self.elementName else self.elementName

        geomMasterFile= self.makeGeomPackage(geomFile, geomType)

        # Packaging
        if self.elementName:
            taskPayloadFile = self.makeElementPackage(geomMasterFile)
        else:
            taskPayloadFile = self.makeModelPackage(geomMasterFile)

        if self.showDir and self.assetDir:
            PackageUtils.AssetPackage(self.showDir, self.assetName, taskPayloadFile)


    def makeModelPackage(self, sourceFile):
        taskFile = '{DIR}/model.usd'.format(DIR=self.outDir)
        SessionUtils.MakeSubLayerStage(taskFile, [sourceFile])
        taskPayloadFile = taskFile.replace('.usd', '.payload.usd')
        SessionUtils.MakeReferenceStage(taskPayloadFile, [(taskFile, None)], SdfPath='/%s{taskVariant=model}' % self.assetName, clear=True)
        return taskPayloadFile

    def makeElementPackage(self, sourceFile):
        modelFile = '{DIR}/model.usd'.format(DIR=self.outDir)
        SessionUtils.MakeSubLayerStage(modelFile, [sourceFile])

        elmFile = '{DIR}/element/{NAME}/{NAME}.usd'.format(DIR=self.assetDir, NAME=self.elementName)
        SessionUtils.MakeReferenceStage(elmFile, [(modelFile, None)], SdfPath='/'+self.elementName, composite='reference')
        elmPayloadFile = elmFile.replace('.usd', '.payload.usd')
        SessionUtils.MakeReferenceStage(elmPayloadFile, [(elmFile, None)], SdfPath='/%s{elementVariant=%s}' % (self.assetName, self.elementName), clear=True)

        taskFile = '{DIR}/element/element.usd'.format(DIR=self.assetDir)
        SessionUtils.MakeSubLayerStage(taskFile, [elmPayloadFile])
        taskPayloadFile = taskFile.replace('.usd', '.payload.usd')
        SessionUtils.MakeReferenceStage(taskPayloadFile, [(taskFile, None)], SdfPath='/%s{taskVariant=element}' % self.assetName, clear=True)
        return taskPayloadFile


    def makeGeomPackage(self, geomFile, geomType):
        geomDir = os.path.dirname(geomFile)
        masterFile = '{DIR}/{NAME}.usd'.format(DIR=geomDir, NAME=self.defaultName)
        purpose = None
        if self.isPurpose:
            purpose = 'proxy' if geomType == 'low' else 'render'
        sourceFiles = [(geomFile, purpose)]
        SdfPath = '/' + self.defaultName
        if self.isLod:
            SdfPath += '{lodVariant=%s}' % geomType
        # SessionUtils.MakeReferenceStage(masterFile, sourceFiles, clearVariantSet=True, SdfPath=SdfPath, Name=self.defaultName)
        SessionUtils.MakeReferenceStage(masterFile, sourceFiles, SdfPath=SdfPath, Name=self.defaultName)

        if self.isPurpose and self.isLod:
            dxsUsdUtils.PerfectionLodVariantPackage(masterFile)
        if self.isLod:
            dxsUsdUtils.LodVariantDefaultSelection(masterFile)

        # Set Materials
        Material.Main(masterFile)

        masterPayloadFile = masterFile.replace('.usd', '.payload.usd')
        SdfPath = '/%s{modelVersion=%s}' % (self.defaultName, self.version)
        SessionUtils.MakeReferenceStage(masterPayloadFile, [(masterFile, None)], SdfPath=SdfPath, clear=True)
        return masterPayloadFile



class ModelClipExport(Arguments.AssetArgs):
    def __init__(self, node=None, fr=(0, 0), step=1.0, loopScales=[1.0], loopRange=(0, 0),
                    isPurpose=False, isLod=False, isSeq=False, overWrite=True, **kwargs):
        dxsMayaUtils.PluginSetup(['backstageMenu', 'pxrUsd', 'AbcExport'])
        self.isPurpose = isPurpose
        self.isLod     = isLod
        self.isSeq     = isSeq
        self.overWrite = overWrite

        self.node = node

        Arguments.AssetArgs.__init__(self, **kwargs)
        if not self.outDir and self.assetDir:
            self.outDir = '{DIR}/clip'.format(DIR=self.assetDir)
        self.computeVersion()

        self.fr  = fr
        self.step= step
        # frameRange
        if not self.fr[0] or not self.fr[1]:
            self.fr = dxsMayaUtils.GetFrameRange()
        self.fps = dxsMayaUtils.GetFPS()

        self.loopScales = loopScales
        self.loopRange  = loopRange

        self.mayafile = dxsMayaUtils.GetMayaFilename()
        if self.mayafile:
            self.comment = 'Generated with %s' % self.mayafile


    def doIt(self):
        if not self.node:
            cmds.confirmDialog(
                title='Error!', message='# ModelClipExport : not found model.',
                icon='warning', button=['OK']
            )
            return
        # Init Texture
        # if self.version != 'v000':
        #     Texture.TextureExport(overWrite=False, showDir=self.showDir, asset=self.assetName, version='v001').doIt()
        # last model version
        modelVersion = PathUtils.GetVersion('{DIR}/model'.format(DIR=self.assetDir), overWrite=False)

        ns, nn = dxsMayaUtils.GetNamespaceInfo(self.node)
        clipLayer = 'default' if not ns else ns
        clipLayer+= '_clip'
        nodeName, geomType = dxsMayaUtils.GetNodeInfo(self.node)
        geomFile = '{DIR}/{VER}/{LAYER}/{NAME}.{TYPE}_geom.usd'.format(DIR=self.outDir, VER=self.version, LAYER=clipLayer, NAME=nodeName, TYPE=geomType)
        # Extract Attribute
        extractAttr = False if self.isSeq else True
        # Geometry OverWrite
        _doit = True
        if os.path.exists(geomFile):
            _doit = self.overWrite
        if _doit:
            dxsMayaUtils.UpdateTextureAttributes(cmds.ls(self.node, dag=True, type='surfaceShape'), asset=self.assetName)
            GeomMain.Export(
                geomFile, self.node, fr=(self.fr[0]-1, self.fr[1]+1), step=self.step, isSeq=self.isSeq,
                userAttr=True, mtlAttr=True, subdivAttr=True, nsLayer=ns, modelVersion=modelVersion
            ).doIt()
            Attributes.ExtractAttr(geomFile).doIt()
            # attrFile = geomFile.replace('_geom.usd', '_attr.usd')
            # EditPrim.AddTextureVersion(attrFile, nsName=ns).doIt()

        # Default Prim Name
        self.rootName = self.assetName
        loopClips = None

        geomMasterFile = self.makeGeomPackage(geomFile, geomType)
        if self.loopScales:
            loopClips = ClipUtils.LoopClip(geomMasterFile, scales=self.loopScales, fr=self.loopRange).doIt()
        if not loopClips:
            return

        taskPayloadFile = self.makeLoopClipPackage(loopClips, clipLayer)
        if self.showDir and self.assetDir:
            PackageUtils.AssetPackage(self.showDir, self.assetName, taskPayloadFile)


    def makeGeomPackage(self, geomFile, geomType):
        geomDir = os.path.dirname(geomFile)
        masterFile = '{DIR}/{NAME}.usd'.format(DIR=geomDir, NAME=self.rootName)
        purpose = None
        if self.isPurpose:
            purpose = 'proxy' if geomType == 'low' else 'render'
        sourceFiles = [(geomFile, purpose)]
        SdfPath = '/' + self.rootName
        if self.isLod:
            SdfPath += '{lodVariant=%s}' % geomType
        SessionUtils.MakeReferenceStage(masterFile, sourceFiles, SdfPath=SdfPath, Name=self.rootName, fr=self.fr, fps=self.fps)

        if self.isPurpose and self.isLod:
            dxsUsdUtils.PerfectionLodVariantPackage(masterFile)
        if self.isLod:
            dxsUsdUtils.LodVariantDefaultSelection(masterFile)
        return masterFile


    def makeLoopClipPackage(self, sourceFiles, clipLayer):
        loopClip = '{DIR}/{VER}/loopClip.usd'.format(DIR=self.outDir, VER=self.version)
        loopLayer= clipLayer.replace('clip', 'loop')
        SessionUtils.MakeSubLayerStage(loopClip, sourceFiles, SdfPath='/%s{loopVariant=%s1_0}' % (self.rootName, loopLayer))
        loopClipPayload = loopClip.replace('.usd', '.payload.usd')
        SessionUtils.MakeReferenceStage(loopClipPayload, [(loopClip, None)], SdfPath='/%s{clipVersion=%s}' % (self.rootName, self.version))

        taskFile = '{DIR}/clip.usd'.format(DIR=self.outDir)
        SessionUtils.MakeSubLayerStage(taskFile, [loopClipPayload])
        taskPayloadFile = taskFile.replace('.usd', '.payload.usd')
        SessionUtils.MakeReferenceStage(taskPayloadFile, [(taskFile, None)], SdfPath='/%s{taskVariant=clip}' % self.assetName, clear=True)
        return taskPayloadFile
