'''
USD xBlock Export
'''
import os, string, shutil

from pxr import Sdf, Usd, UsdGeom
import maya.cmds as cmds

import MsgSender
import dxsMsg
import Arguments
import PathUtils
import dxsMayaUtils
import dxsUsdUtils
import GeomMain
import SessionUtils
import PackageUtils


def GetSimNodes(selected='|*'):
    '''
    Return
        (list) [(node, ref-node, namespace), (...)]
    '''
    nodes = cmds.ls(selected.split(':')[-1], type='xBlock', l=True, r=True)
    if not nodes:
        dxsMsg.Print('warning', "[not found 'xBlock' for simulation]")
        return

    exportNodes = list()
    for n in nodes:
        if cmds.getAttr('%s.action' % n) == 1 and cmds.getAttr('%s.type' % n) == 3 and dxsMayaUtils.GetViz(n):
            refnode = n
            refconnected = cmds.listConnections('%s.referencedXBlock' % n, s=True, d=False)
            if refconnected:
                refnode = refconnected[0]
            nsLayer = cmds.getAttr('%s.nsLayer' % refnode)
            if nsLayer:
                exportNodes.append((n, refnode, nsLayer))
            else:
                dxsMsg.Print('warning', "[Sim.GetSimNodes] : nsLayer setup error - '%s'" % n)
    if not exportNodes:
        dxsMsg.Print('warning', "[not found export simulation 'xBlock']")
        return
    return exportNodes



class EditGeom:
    def __init__(self, filename, node):
        self.filename = filename
        self.node     = node    # fullpath

        # Member Variables
        self.rootPath = ''
        self.attrFile = ''

        self.getNodeInfo()

    def getNodeInfo(self):
        refNode = self.node
        refconnected = cmds.listConnections('%s.referencedXBlock' % self.node, s=True, d=False)
        if refconnected:
            refNode = refconnected[0]

        self.rootPath = cmds.getAttr('%s.rootPrimPath' % refNode)
        baseGeomFile  = cmds.getAttr('%s.importFile' % refNode)
        if baseGeomFile:
            attrFile = baseGeomFile.split('.')[0] + '.high_attr.usd'
            if os.path.exists(attrFile):
                self.attrFile = attrFile


    def doIt(self):
        self.outLayer = Sdf.Layer.FindOrOpen(self.filename)
        if not self.outLayer:
            dxsMsg.Print('warning', "[Sim.EditGeom] not found file -> %s" % self.filename)
            return
        self.stage = Usd.Stage.Open(self.outLayer, load=Usd.Stage.LoadNone)
        self.dprim = self.stage.GetDefaultPrim()

        self.pathRepresent()

        # set defaultPrim
        rootPath  = self.outLayer.rootPrims[0].path
        self.dprim= self.stage.GetPrimAtPath(rootPath)
        self.stage.SetDefaultPrim(self.dprim)

        # remove root xform
        for attr in self.dprim.GetPropertiesInNamespace('xformOp'):
            self.dprim.RemoveProperty(attr.GetName())
        self.dprim.RemoveProperty('xformOpOrder')

        # Inherit Attribute
        if self.attrFile:
            self.SetInheritAttr()

        # Purpose Attribute
        self.purposeSetup()

        # Save
        tmpfile = self.filename.replace('.usd', '_tmp.usd')
        self.outLayer.Export(tmpfile, args={'format': 'usdc'})
        os.remove(self.filename)
        os.rename(tmpfile, self.filename)


    def pathRepresent(self):
        edit = Sdf.BatchNamespaceEdit()
        for shape in cmds.ls(self.node, dag=True, s=True, ni=True):
            trnode = cmds.listRelatives(shape, p=True, f=True)[0]
            srcpath= trnode.replace(':', '_').replace('|', '/')
            dstpath= cmds.getAttr('%s.primPath' % trnode)
            if srcpath != dstpath:
                self.InitPrim(dstpath)
                edit.Add(srcpath, dstpath)
                # copy xform
                self.XformCopy(srcpath, dstpath)
        self.outLayer.Apply(edit)

        dprimPath = cmds.ls(self.node, l=True)[0].replace(':', '_').replace('|', '/')
        if dprimPath != self.rootPath:
            edit = Sdf.BatchNamespaceEdit()
            edit.Add('/' + dprimPath.split('/')[1], Sdf.Path.emptyPath)
            self.outLayer.Apply(edit)


    def InitPrim(self, dst):
        prefixes = Sdf.Path(dst).GetPrefixes()
        for i in range(len(prefixes) - 1):
            prim = self.stage.GetPrimAtPath(prefixes[i])
            if not prim:
                self.stage.DefinePrim(prefixes[i], 'Xform')

    def XformCopy(self, src, dst):
        dstPath  = Sdf.Path(dst)
        prefixes = Sdf.Path(src).GetPrefixes()
        prefixes.reverse()
        for sp in prefixes[1:-1]:
            prim   = self.stage.GetPrimAtPath(sp)
            dstPath= dstPath.GetParentPath()
            if sp != dstPath:
                xformOpAttrs = prim.GetAuthoredPropertiesInNamespace('xformOp')
                if xformOpAttrs:
                    for attr in xformOpAttrs:
                        attrPath = attr.GetPath()
                        attrName = attr.GetName()
                        Sdf.CopySpec(self.outLayer, attrPath, self.outLayer, Sdf.Path(dstPath.pathString + '.' + attrName))
                    Sdf.CopySpec(self.outLayer, Sdf.Path(sp.pathString + '.xformOpOrder'), self.outLayer, Sdf.Path(dstPath.pathString + '.xformOpOrder'))
                else:
                    dstprim = self.stage.GetPrimAtPath(dstPath)
                    UsdGeom.Xform(dstprim).CreateXformOpOrderAttr().Block()

    def SetInheritAttr(self):
        rootLayer = Sdf.Layer.FindOrOpen(self.attrFile)
        rootPath  = rootLayer.rootPrims[0].path
        SessionUtils.InsertSubLayer(self.outLayer, PathUtils.GetRelPath(self.filename, self.attrFile))
        self.dprim.GetInherits().AddInherit(rootPath)

    def purposeSetup(self):
        nsName, nodeName = dxsMayaUtils.GetNamespaceInfo(self.node)
        for n in cmds.ls(self.node, dag=True, type='transform', l=True):
            if cmds.attributeQuery('USD_ATTR_purpose', n=n, ex=True):
                val = cmds.getAttr('%s.USD_ATTR_purpose' % n)
                if val == 'default':
                    if nsName:
                        primPath = n.replace('|' + nsName + ':', '/')
                    else:
                        primPath = n.replace('|', '/')
                    prim = self.stage.GetPrimAtPath(primPath)
                    if prim:
                        UsdGeom.Scope(prim).CreatePurposeAttr(UsdGeom.Tokens.default_)



class SimExport(Arguments.ShotArgs):
    def __init__(self, node=None, **kwargs):
        dxsMayaUtils.PluginSetup(['backstageMenu', 'pxrUsd'])

        # Member Variables
        self.expfr = None

        Arguments.ShotArgs.__init__(self, **kwargs)
        if not self.outDir and self.shotDir:
            self.outDir = self.shotDir + '/sim'

        self.node    = ''
        self.refNode = ''
        self.nsLayer = ''
        ndata = GetSimNodes(selected=node)
        if ndata:
            self.node, self.refNode, self.nsLayer = ndata[0]

        self.outDir += '/' + self.nsLayer
        self.computeVersion()

        self.mayafile = dxsMayaUtils.GetMayaFilename()
        self.comment  = 'Generated with %s' % self.mayafile

    def computeFrameRange(self):
        self.fps= dxsMayaUtils.GetFPS()
        autofr  = False
        if not self.fr[0] or not self.fr[1]:
            self.fr = dxsMayaUtils.GetFrameRange()
            autofr = True

        self.expfr = self.fr
        if self.fr[0] != self.fr[1]:
            if autofr:
                ast = int(cmds.playbackOptions(q=True, ast=True))
                if self.fr[0] - ast == 51:
                    self.expfr = (ast, self.fr[1] + 1)
                else:
                    self.expfr = (self.fr[0] - 1, self.fr[1] + 1)
            else:
                self.expfr = (self.fr[0] - 1, self.fr[1] + 1)


    def doIt(self):
        if not self.mayafile:
            dxsMsg.Print('warning', "[Must have to save current scene]")
            return
        if not self.node:
            return

        self.computeFrameRange()

        nsName, nodeName = dxsMayaUtils.GetNamespaceInfo(self.node)
        geomFile = '{DIR}/{VER}/{NAME}.high_geom.usd'.format(DIR=self.outDir, VER=self.version, NAME=nodeName)
        GeomMain.Export(geomFile, [self.node], fr=self.expfr, step=self.step).doIt()    # not remove namespace
        EditGeom(geomFile, self.node).doIt()    # remove namespace and object path represent by primPath attribute

        masterPayloadFile = self.makeGeomPackage(geomFile)
        self.makePackage(masterPayloadFile)
        return masterPayloadFile


    def makeGeomPackage(self, sourceFile):
        # dependency
        tasks = list(); versions = list()
        inputCache = cmds.getAttr('%s.mergeFile' % self.refNode)
        splitPath  = os.path.dirname(inputCache).split('/')
        tasks.append(splitPath[splitPath.index(self.shotName)+1])
        versions.append(splitPath[-1])

        customLayerData = {
            'start': int(self.fr[0]), 'end': int(self.fr[1]),
            'simInputCacheFile': inputCache
        }
        customPrimData = {
            'sim': os.path.basename(self.mayafile),
            'simInputCache': string.join(splitPath[splitPath.index(self.shotName)+1:], '/')
        }
        masterFile = '{DIR}/{VER}/{NAME}.usd'.format(DIR=self.outDir, VER=self.version, NAME=self.nsLayer)
        SdfPath = '/' + self.nsLayer
        SessionUtils.MakeReferenceStage(
            masterFile, [(sourceFile, None)], SdfPath=SdfPath, comment=self.comment,
            customLayerData=customLayerData, customPrimData=customPrimData
        )

        masterPayloadFile = masterFile.replace('.usd', '.payload.usd')
        stage = SessionUtils.MakeInitialStage(masterPayloadFile, clear=True, fr=self.fr, fps=self.fps, comment=self.comment)
        dprim = stage.DefinePrim('/' + self.nsLayer, 'Xform')
        dxsUsdUtils.SetModelAPI(dprim, kind='component', name=self.nsLayer)
        stage.SetDefaultPrim(dprim)

        tasks.append('sim')
        versions.append(self.version)

        def AddVariantSet(editcontext, prim, name, value):
            if editcontext:
                editcontext.__enter__()
            if name == 'simVersion':
                dxsUsdUtils.VariantSelection(prim, name, 'v000')
            vset = dxsUsdUtils.VariantSelection(prim, name, value)
            return vset.GetVariantEditContext()

        ectx = None
        for i in range(len(tasks)):
            task = tasks[i]
            ver  = versions[i]
            ectx = AddVariantSet(ectx, dprim, '%sVersion' % task, ver)
        ectx.__enter__()
        payload = Sdf.Payload(PathUtils.GetRelPath(masterPayloadFile, masterFile))
        dxsUsdUtils.SetPayload(dprim, payload)
        ectx.__exit__(None, None, None)
        stage.GetRootLayer().Save()
        return masterPayloadFile


    def makePackage(self, sourceFile):
        layerFile = '{DIR}/{NAME}.usd'.format(DIR=self.outDir, NAME=self.nsLayer)
        SessionUtils.MakeSubLayerStage(layerFile, [sourceFile])

        layerPayloadFile = layerFile.replace('.usd', '.payload.usd')
        SessionUtils.MakeReferenceStage(layerPayloadFile, [(layerFile, None)], SdfPath='/rig/%s' % self.nsLayer, Kind='assembly', clear=True)

        simFile = '{DIR}/sim/sim.usd'.format(DIR=self.shotDir)
        SessionUtils.MakeSubLayerStage(simFile, [layerPayloadFile])

        simPayloadFile = simFile.replace('.usd', '.payload.usd')
        SessionUtils.MakeReferenceStage(simPayloadFile, [(simFile, None)], SdfPath='/shot/rig', Name=self.shotName, Kind='assembly', clear=True)

        PackageUtils.ShotPackage(self.showDir, self.seqName, self.shotName, simPayloadFile, fr=self.fr, fps=self.fps)
        self.overrideVersion()

    def overrideVersion(self):
        shotFile = '{DIR}/shot/{SEQ}/{SHOT}/{SHOT}.usd'.format(DIR=self.showDir, SEQ=self.seqName, SHOT=self.shotName)
        PackageUtils.VersionSelect(shotFile, '/shot/rig/' + self.nsLayer, 'simVersion', self.version)

        shotLgtFile = shotFile.replace('.usd', '.lgt.usd')
        if os.path.exists(shotLgtFile):
            PackageUtils.VersionSelect(shotLgtFile, '/shot/rig/' + self.nsLayer, 'simVersion', self.version)
