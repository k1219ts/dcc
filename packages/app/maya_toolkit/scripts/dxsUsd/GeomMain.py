import os
import re
import pprint

from pxr import Usd, UsdGeom, Sdf, Vt, Gf

import maya.cmds as cmds

import SessionUtils
import ClipUtils
import GeomCurves
import Attributes
import EditPrim
import dxsMayaUtils
import PathUtils


class Export:
    '''
    PipeLine Main Geometry Export.
        Set Attribute -> usdExport -> Change Curves -> Composite re-present
        -> remove namespace -> extract attributes
    Args:
        filename (str):
        nodes   (list):
        fr   (tuple): (start(int), end(int)) export frame range
        step (float):
        isSeq (list or bool): sperate big size geom file. [(start, end), ...]
        userAttr (bool):
        mtlAttr  (bool):
        subdivAttr  (bool):
        nsLayer (str): remove maya namespace name
        modelVersion (str):
        isAbc  (bool): export alembic on, off
        rigNode (str): dxRig. if Rig export.
    '''
    def __init__(self, filename, nodes, fr=[None, None], step=1.0, isSeq=False,
                       userAttr=False, mtlAttr=False, subdivAttr=True,
                       nsLayer=None, modelVersion=None, isAbc=False, rigNode=None):
        self.filename = filename
        self.nodes    = cmds.ls(nodes, l=True)
        self.frameRange = fr
        self.frameSample= dxsMayaUtils.GetFrameSample(step)
        self.step  = step

        self.isSeq = isSeq
        self.userAttr = userAttr
        self.mtlAttr  = mtlAttr
        self.subdivAttr = subdivAttr
        self.nsLayer = nsLayer
        self.modelVersion = modelVersion
        self.isAbc   = isAbc
        self.rigNode = rigNode

        # User Attributes
        self.excludeMeshAttributes = ['txVarNum', 'txVersion']
        self.excludeReferenceAttributes = ['txBasePath', 'txLayerName', 'txmultiUV', 'txVersion', 'modelVersion']

        self.overrideCommand = dict()


    def doIt(self):
        objects = cmds.ls(self.nodes, dag=True, type='surfaceShape', ni=True)
        UsdAttr = dxsMayaUtils.UsdGeomAttributes(objects, user=self.userAttr, mtl=self.mtlAttr, subdiv=self.subdivAttr)
        UsdAttr.excludeMeshAttributes = self.excludeMeshAttributes
        UsdAttr.excludeReferenceAttributes = self.excludeReferenceAttributes
        UsdAttr.Set()

        if (self.frameRange != None and self.frameRange != None) and (self.frameRange[0] != self.frameRange[1]) and self.isSeq:
            self.SequenceExport()
        else:
            dxsMayaUtils.UsdExport(self.filename, self.nodes, FR=self.frameRange, FS=self.frameSample, **self.overrideCommand)
            # NurbsCurves To BasisCurves
            GeomCurves.NurbsToBasis(self.filename, self.nsLayer)
            # remove namespace
            EditPrim.Edit(self.filename, ns=self.nsLayer, rigNode=self.rigNode).doIt()

        UsdAttr.Clear()

        if self.isAbc:
            self.alembicExport()

        # composite archive represent
        self.CompositeRepresent()


    def SequenceExport(self):
        frameFiles = list()
        for start, end in self.isSeq:
            fn = self.filename.replace('.usd', '.%04d.usd' % start)
            dxsMayaUtils.UsdExport(fn, self.nodes, FR=(start, end), FS=self.frameSample, **self.overrideCommand)
            # NurbsCurves To BasisCurves
            GeomCurves.NurbsToBasis(fn, self.nsLayer)
            # remove namespace
            EditPrim.Edit(fn, ns=self.nsLayer, rigNode=self.rigNode).doIt()
            frameFiles.append(fn)
        clipFile = ClipUtils.StitchFiles(frameFiles).doIt()
        os.rename(clipFile, self.filename)


    def CompositeRepresent(self):
        stage = Usd.Stage.Open(self.filename, load=Usd.Stage.LoadNone)
        dprim = stage.GetDefaultPrim()

        # edit reference - pxrUsdReferenceAssembly
        targetNodes = list()
        for n in cmds.ls(self.nodes, dag=True, type='pxrUsdReferenceAssembly', l=True):
            targetNodes.append(n)
        self.editReference(stage, targetNodes)

        # make reference - xBlock or pxrUsdProxyShape
        targetNodes = list()
        for shape in cmds.ls(self.nodes, dag=True, type='pxrUsdProxyShape', l=True):
            transform = cmds.listRelatives(shape, p=True, f=True)[0]
            if cmds.nodeType(transform) != 'pxrUsdReferenceAssembly':
                targetNodes.append(transform)
        for xblock in cmds.ls(self.nodes, dag=True, type='xBlock', l=True):
            btype = cmds.getAttr('%s.type' % xblock)
            action= cmds.getAttr('%s.action' % xblock)
            if btype == 1 and action == 2:  # type: mode, action: reference
                targetNodes.append(xblock)
        targetNodes = list(set(targetNodes))
        if targetNodes:
            self.makeReference(stage, targetNodes)

        stage.GetRootLayer().Save()


    def editReference(self, stage, nodes):
        '''
        Edit Reference for pxrUsdReferenceAssembly.
        '''
        outLayer = stage.GetRootLayer()
        FileCountMap = dict()
        NodeMap = dict()    # {node: {'prim': xxx, 'asset': xxx}}
        for n in nodes:
            pathStr = n.replace('|', '/')
            if self.nsLayer:
                pathStr = n.replace('|%s:' % self.nsLayer, '/')
            prim = stage.GetPrimAtPath(pathStr)
            if not prim:
                continue

            refs = prim.GetMetadata('references')
            if refs:
                assetPath = refs.prependedItems[0].assetPath
                if not FileCountMap.has_key(assetPath):
                    FileCountMap[assetPath] = 0
                FileCountMap[assetPath] += 1
                NodeMap[n] = {'prim': prim, 'asset': assetPath}

        for n in NodeMap:
            prim      = NodeMap[n]['prim']
            assetPath = NodeMap[n]['asset']

            # visibility
            VisibilitySetKey(n, prim)

            # Set instanceable - exported default value is True
            if assetPath.find('_set') > -1 or FileCountMap[assetPath] == 1:
                prim.SetInstanceable(False)

            # Time
            timeNode = cmds.listConnections(n, s=True, d=False, type='time')
            timeOffsetNode = cmds.listConnections(n, s=True, d=False, type='dxTimeOffset')
            if timeNode or timeOffsetNode:
                prim.SetInstanceable(False)
            if timeOffsetNode:
                offset = cmds.getAttr('%s.offset' % timeOffsetNode[0])
                if offset:
                    prim.GetReferences().ClearReferences()
                    ref = Sdf.Reference(assetPath, layerOffset=Sdf.LayerOffset(offset * -1))
                    prim.GetReferences().AddReference(ref)

            # replace render.usd -> will be delete
            # new = assetPath.replace('.usd', '.render.usd')
            # if os.path.exists(new):
            #     prim.GetReferences().ClearReferences()
            #     prim.GetReferences().AddReference(Sdf.Reference(PathUtils.GetRelPath(self.filename, new)))
            #     self.lodVariantClearSelection(outLayer, prim)


    def XmakeReference(self, stage, nodes):
        '''
        Create Reference for xBlock or pxrUsdProxyShape.
            - Not support : time
        '''
        FileCountMap = dict()
        NodeAssetMap = dict()   # {node:(assetPath, primPath, excludePrimPaths)}
        for n in nodes:
            if cmds.nodeType(n) == 'xBlock':
                assetPath = cmds.getAttr('%s.importFile' % n)
            else:
                proxyShape= cmds.ls(n, dag=True, type='pxrUsdProxyShape', l=True)[0]
                assetPath = cmds.getAttr('%s.filePath' % proxyShape)
            new = assetPath.replace('.usd', '.render.usd')
            if os.path.exists(new):
                assetPath = new
            NodeAssetMap[n] = assetPath
            if not FileCountMap.has_key(assetPath):
                FileCountMap[assetPath] = 0
            FileCountMap[assetPath] += 1

        edit = Sdf.BatchNamespaceEdit()
        for n in nodes:
            pathStr = n.replace('|', '/')
            if self.nsLayer:
                pathStr = n.replace('|%s:' % self.nsLayer, '/')
            prim = stage.GetPrimAtPath(pathStr)
            if not prim:
                continue

            # remove children
            self.removeChildren(prim, edit)

            # Set instanceable - exported default value is True
            assetPath = NodeAssetMap[n]
            if assetPath.find('_set') > -1 or FileCountMap[assetPath] == 1:
                prim.SetInstanceable(False)

            # Composite Archive
            relpath = PathUtils.GetRelPath(self.filename, assetPath)
            ref = Sdf.Reference(relpath)
            prim.GetReferences().AddReference(ref)

        outLayer = stage.GetRootLayer()
        outLayer.Apply(edit)
        stage.Load()

    def makeReference(self, stage, nodes):
        '''
        Create Reference for xBlock or pxrUsdProxyShape.
            - Not support : time
        '''
        FileCountMap = dict()
        NodeAssetMap = dict()   # {node: (assetPath, primPath, excludePrimPath)}
        for n in nodes:
            primPath = None; excludePrimPaths = None
            if cmds.nodeType(n) == 'xBlock':
                assetPath = cmds.getAttr('%s.importFile' % n)
            else:
                proxyShape= cmds.ls(n, dag=True, type='pxrUsdProxyShape', l=True)[0]
                assetPath = cmds.getAttr('%s.filePath' % proxyShape)
                primPath  = cmds.getAttr('%s.primPath' % proxyShape)
                excludePrimPaths = cmds.getAttr('%s.excludePrimPaths' % proxyShape)
            new = assetPath.replace('.usd', '.render.usd')
            if os.path.exists(new):
                assetPath = new
            NodeAssetMap[n] = (assetPath, primPath, excludePrimPaths)
            if not FileCountMap.has_key(assetPath):
                FileCountMap[assetPath] = 0
            FileCountMap[assetPath] += 1

        edit = Sdf.BatchNamespaceEdit()
        for n in nodes:
            pathStr = n.replace('|', '/')
            if self.nsLayer:
                pathStr = n.replace('|%s:' % self.nsLayer, '/')
            prim = stage.GetPrimAtPath(pathStr)
            if not prim:
                continue

            # remove children
            self.removeChildren(prim, edit)

            assetPath, primPath, excludePrimPaths = NodeAssetMap[n]
            if assetPath.find('_set') > -1 or primPath or excludePrimPaths or FileCountMap[assetPath] == 1:
                prim.SetInstanceable(False)

            relpath = PathUtils.GetRelPath(self.filename, assetPath)
            ref = Sdf.Reference(relpath)
            if primPath:
                ref = Sdf.Reference(relpath, Sdf.Path(primPath))
            prim.GetReferences().AddReference(ref)

            if excludePrimPaths:
                for pstr in excludePrimPaths.split(','):
                    # prel = '/'.join(pstr.split('/')[2:])
                    prel = pstr.strip().replace(primPath, '', 1)
                    overprim = stage.OverridePrim(prim.GetPath().AppendPath(prel[1:]))
                    overprim.SetActive(False)

        outLayer = stage.GetRootLayer()
        outLayer.Apply(edit)
        stage.Load()


    def removeChildren(self, parent, editor):
        if parent.GetChildren():
            treeIter = iter(Usd.PrimRange.AllPrims(parent))
            treeIter.next()
            for p in treeIter:
                print '>>', p
                editor.Add(p.GetPath().pathString, Sdf.Path.emptyPath)

    def lodVariantClearSelection(self, rootLayer, prim):
        primSpec = rootLayer.GetPrimAtPath(prim.GetPath())
        for n, v in primSpec.variantSelections.items():
            if n == 'lodVariant':
                prim.GetVariantSets().GetVariantSet(n).ClearVariantSelection()


    def alembicExport(self):
        version = re.compile(r'\/v(\d+)?\/').findall(self.filename)
        if version:
            baseName = os.path.basename(self.filename)
            baseDir  = self.filename.split('/v%s/' % version[-1])[0]
            abcFile = os.path.join(baseDir, self.getAlembicName(baseName, version[-1]))
            dxsMayaUtils.AbcExport(abcFile, self.nodes, self.frameRange, self.step)

    def getAlembicName(self, baseName, version):
        geomType = 'high'
        if 'mid_geom' in baseName:
            geomType = 'mid'
        if 'low_geom' in baseName:
            geomType = 'low'
        baseName = baseName.split('_model')[0]
        if self.nsLayer:
            baseName += '_' + self.nsLayer
        baseName+= '_model'
        if geomType != 'high':
            baseName += '_' + geomType
        baseName += '_v' + version + '.abc'
        return baseName


    def setVisibilityKey(self, node, prim):
        if not cmds.listConnections('%s.visibility' % node, type='animCurve'):
            return
        frames = cmds.keyframe(node, q=True)
        values = cmds.keyframe(node, q=True, vc=True)

        visAttr = prim.GetAttribute('visibility')


class VisibilitySetKey:
    def __init__(self, node, prim):
        animCurve = cmds.listConnections('%s.visibility' % node, s=True, d=False, type='animCurve')
        if not animCurve:
            return
        frameOffset= 0
        inputCurve = cmds.listConnections('%s.input' % animCurve[0], p=True, s=True, d=False)
        if inputCurve:
            frameOffset = cmds.currentTime(q=True) - cmds.getAttr(inputCurve[0])

        self.viztoken = {0.0: 'invisible', 1.0: 'inherited'}

        frames = cmds.keyframe(node, at='visibility', q=True)
        values = cmds.keyframe(node, at='visibility', q=True, vc=True)

        self.visAttr = prim.GetAttribute('visibility')
        for i in xrange(len(values)):
            frame = frames[i] + frameOffset
            self.visAttr.Set(self.viztoken[values[i]], Usd.TimeCode(frame))
            # if i == 0:
            #     if values[i] == 0.0:
            #         self.visAttr.Set(self.viztoken[values[i]], Usd.TimeCode(frame))
            # else:
            #     self.setKey(values[i], frame)

    def setKey(self, value, frame):
        # inverse value
        opvalue = {0.0: 1.0, 1.0: 0.0}
        # before
        self.visAttr.Set(self.viztoken[opvalue[value]], Usd.TimeCode(frame-1))
        # current
        self.visAttr.Set(self.viztoken[value], Usd.TimeCode(frame))
