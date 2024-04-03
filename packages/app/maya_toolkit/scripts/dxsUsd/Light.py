'''
USD Light Instance Export

Outputs
asset / $ASSET / light / light.paylod.usd
                       / light.usd
                       / $VER / $ASSET.payload.usd
                              / $ASSET.usd
                              / $NODE.geom.usd
'''

import EnvSet
import Arguments
import dxsMayaUtils
import SessionUtils
import dxsUsdUtils
import dxsXformUtils
import PackageUtils

import maya.cmds as cmds
from pxr import UsdGeom, Vt, Sdf

class LightInstanceExport(Arguments.AssetArgs):
    '''
    lighting to pointInstancer setup
    '''
    def __init__(self, node=None, **kwargs):
        dxsMayaUtils.PluginSetup(['backstageMenu', 'pxrUsd'])

        self.node = node
        Arguments.AssetArgs.__init__(self, **kwargs)

        if not self.outDir and self.assetDir:
            self.outDir = '{DIR}/lighting'.format(DIR=self.assetDir)
        self.computeVersion()

        if cmds.nodeType(self.node) != "xBlock":
            print "# msg: node type isn't xBlockNode!"
            return

        if cmds.getAttr("%s.type" % self.node) != 5:
            print "# msg: check export type"
            return

        self.mayafile = dxsMayaUtils.GetMayaFilename()
        if self.mayafile:
            self.comment = 'Generated with %s' % self.mayafile

    def initTransform(self, node):
        mtx = cmds.xform(node, q=True, m=True, ws=True)
        cmds.setAttr('%s.translate' % node, 0, 0, 0, type='double3')
        cmds.setAttr('%s.rotate' % node, 0, 0, 0, type='double3')
        cmds.setAttr('%s.scale' % node, 1, 1, 1, type='double3')
        return mtx

    def doIt(self):
        if not self.node:
            return
        # self.node = self.node
        if not dxsMayaUtils.GetViz(self.node):
            print '# msg : viz off'
            return

        name = self.node.split('|')[-1].split(':')[-1]
        geomFile = '{DIR}/{VER}/{NAME}.lgt_geom.usd'.format(DIR=self.outDir, VER=self.version, NAME=name)

        geomMtx = self.initTransform(self.node)
        Data = EnvSet.PointInstancerBlock()

        EnvSet.GetDxAssembly(self.node, Data).doIt()
        if self.geomExport(Data, geomFile, self.comment):
            self.geomFile = geomFile

        if geomMtx:
            cmds.xform(self.node, m=geomMtx, ws=True)

        if not self.geomFile:
            return

        geomMasterFile = self.makeGeomPackage(geomFile)

        # Packaging
        taskPayloadFile = self.makeLightPackage(geomMasterFile)

        if self.showDir and self.assetDir:
            PackageUtils.AssetPackage(self.showDir, self.assetName, taskPayloadFile)

    def geomExport(self, data, geomFile, comment):
        self.data = data
        self.filename = geomFile

        self.comment = comment

        if not self.data.protoTypes:
            print '# msg : not found data'
            return False

        name  = self.node.split('|')[-1].split(':')[-1]
        stage = SessionUtils.MakeInitialStage(self.filename, usdformat='usdc', clear=True, comment=self.comment)
        dprim = stage.DefinePrim('/' + name, 'Xform')
        stage.SetDefaultPrim(dprim)
        dxsUsdUtils.SetModelAPI(dprim, name=name, kind='component')

        if self.data.ids:
            self.makePointInstancer(stage, dprim)

        stage.GetRootLayer().Save()
        print "# Export usd file '%s'" % self.filename
        return True

    def makePointInstancer(self, stage, parent):
        istPrim = stage.DefinePrim(parent.GetPath().AppendChild('scatter'), 'PointInstancer')
        istGeom = UsdGeom.PointInstancer(istPrim)
        istGeom.CreateIdsAttr(self.data.ids)
        istGeom.CreatePositionsAttr(Vt.Vec3fArray(self.data.positions))
        istGeom.CreateScalesAttr(Vt.Vec3fArray(self.data.scales))
        istGeom.CreateOrientationsAttr(Vt.QuathArray(self.data.orients))
        istGeom.CreateProtoIndicesAttr(Vt.IntArray(self.data.protoIndices))
        dxsUsdUtils.AddPrimvar(istGeom, "lightColor", Sdf.ValueTypeNames.Float3Array, UsdGeom.Tokens.uniform, self.data.tintColor)

        protorel = self.makePrototypes(stage, istPrim)
        istGeom.GetPrototypesRel().SetTargets(protorel)

    def makePrototypes(self, stage, parent):
        protoprim = stage.DefinePrim(parent.GetPath().AppendChild('Prototypes'))
        protorel  = list()
        for proto in self.data.protoTypes:
            prim = stage.DefinePrim(protoprim.GetPath().AppendChild(proto['name']))
            if proto.has_key('reference'):
                prim.GetReferences().AddReference(proto['reference'])
                # Add protoXform
                if proto.has_key('mtx'):
                    dxsXformUtils.AddXformOp(prim, matrix=proto['mtx'])
                # variant selection
                if proto.has_key('vsets'):
                    for name in proto['vsets']:
                        vset = prim.GetVariantSets().GetVariantSet(name)
                        vset.SetVariantSelection(proto['vsets'][name])
            protorel.append(prim.GetPath())
        return protorel

    def makeGeomPackage(self, geomFile):
        masterFile = '{DIR}/{VER}/{NAME}.usd'.format(DIR=self.outDir, VER=self.version, NAME=self.assetName)
        SessionUtils.MakeReferenceStage(masterFile, [(geomFile, None)], SdfPath='/'+self.assetName, composite='reference', addChild=True, comment=self.comment)

        masterPayloadFile = masterFile.replace('.usd', '.payload.usd')
        SdfPath = '/%s{lightVersion=%s}' % (self.assetName, self.version)
        SessionUtils.MakeReferenceStage(masterPayloadFile, [(masterFile, None)], SdfPath=SdfPath, clear=True, comment=self.comment)
        return masterPayloadFile

    def makeLightPackage(self, sourceFile):
        taskFile = '{DIR}/light.usd'.format(DIR=self.outDir)
        SessionUtils.MakeSubLayerStage(taskFile, [sourceFile])
        taskPayloadFile = taskFile.replace('.usd', '.payload.usd')
        SdfPath = '/%s{taskVariant=light}' % self.assetName
        SessionUtils.MakeReferenceStage(taskPayloadFile, [(taskFile, None)], SdfPath=SdfPath, clear=True)
        return taskPayloadFile
