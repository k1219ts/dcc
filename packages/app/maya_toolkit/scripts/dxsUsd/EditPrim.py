import os
import string

from pxr import Sdf, Usd, UsdGeom
import maya.cmds as cmds

import PathUtils
import Texture
import dxsMsg


class Edit:
    def __init__(self, filename, ns=None, rigNode=None):
        self.filename  = filename
        self.nameSpace = ns
        self.rigNode   = rigNode

    def doIt(self):
        self.outLayer = Sdf.Layer.FindOrOpen(self.filename)
        if not self.outLayer:
            print '# Error : not found file -> %s' % self.filename
            return

        self.stage = Usd.Stage.Open(self.outLayer, load=Usd.Stage.LoadNone)
        self.defaultPrim = self.stage.GetDefaultPrim()

        # self.clearNullXform() # edit 2019.06.14 - pxrUsdProxyShape expor error by this.

        if self.rigNode:
            self.rootRepresent()

        if self.nameSpace:
            self.removeNamespace()

        self.outLayer.Save()


    def clearNullXform(self):
        edit = Sdf.BatchNamespaceEdit()
        for p in iter(Usd.PrimRange.AllPrims(self.defaultPrim)):
            if p.GetTypeName() == 'Xform':
                chd = p.GetAllChildren()
                if not chd:
                    edit.Add(p.GetPath().pathString, Sdf.Path.emptyPath)
        self.outLayer.Apply(edit)


    def removeNamespace(self):
        nameSpace = self.nameSpace.replace(':', '_')    # if using multi namespace. tiger:tiger:objectname
        edit = Sdf.BatchNamespaceEdit()

        treeIter = iter(Usd.PrimRange.PreAndPostVisit(self.defaultPrim))
        for p in treeIter:
            if treeIter.IsPostVisit():
                primPath = p.GetPath()
                primName = primPath.name
                if not nameSpace in primName:
                    continue
                newName  = primName.replace('%s_' % nameSpace, '', 1)
                newPath  = primPath.GetParentPath().AppendChild(newName)

                edit.Add(primPath.pathString, newPath.pathString)

        self.outLayer.Apply(edit)

        rootPath = self.outLayer.rootPrims[0].path
        self.defaultPrim = self.stage.GetPrimAtPath(rootPath)
        self.stage.SetDefaultPrim(self.defaultPrim)


    def rootRepresent(self):
        nodePath = cmds.ls(self.rigNode, l=True)[0]
        nodePath = nodePath.replace(':', '_').replace('|', '/')
        pathSplit= nodePath.split('/')
        if len(pathSplit) < 3:
            return

        newPath = '/' + pathSplit[-1]
        edit = Sdf.BatchNamespaceEdit()
        edit.Add(nodePath, newPath)
        edit.Add('/' + pathSplit[1], Sdf.Path.emptyPath)

        self.outLayer.Apply(edit)

        self.defaultPrim = self.stage.GetPrimAtPath(newPath)
        self.stage.SetDefaultPrim(self.defaultPrim)


# class AddTextureVersion:
#     '''
#     Create Texture Version Class
#         _$AssetName_$modelVersion_txVer
#     Args:
#         inputFile (str): attribute file
#         version (str): model version
#         nsName  (str): namespace name
#     '''
#     def __init__(self, inputFile, version=None, nsName=None):
#         self.inputFile = inputFile
#         self.version = version
#         self.nsName  = nsName
#         # member variable
#         self.verClassList = list()
#         self._msg = list()
#
#         self.pathParse()
#
#         # for Developer debug.
#         self.debugCreate = False
#
#     def pathParse(self):
#         if '/assetlib' in self.inputFile:
#             self.showDir = '/assetlib/3D'
#         else:
#             if '/show/' in self.inputFile:
#                 splitPath = self.inputFile.split('/')
#                 showIndex = splitPath.index('show')
#                 self.showDir = string.join(splitPath[:showIndex+2], '/')
#
#
#     def doIt(self):
#         outLayer= Sdf.Layer.FindOrOpen(self.inputFile)
#         if not outLayer:
#             return
#         stage   = Usd.Stage.Open(outLayer, load=Usd.Stage.LoadNone)
#         treeIter= iter(Usd.PrimRange.AllPrims(stage.GetPseudoRoot()))
#         treeIter.next()
#         for p in treeIter:
#             txPath = p.GetAttribute('primvars:txBasePath').Get()
#             if txPath:
#                 name = p.GetPath().name
#                 if self.nsName:
#                     name = self.nsName + ':' + name
#                 shape= cmds.ls(name, dag=True, s=True, ni=True)
#                 if shape:
#                     version = None
#                     if cmds.attributeQuery('modelVersion', n=shape[0], ex=True):
#                         version = cmds.getAttr('%s.modelVersion' % shape[0])
#                     else:
#                         if self.version:
#                             version = self.version
#                     if version:
#                         className = self.makeVersionClass(stage, txPath, version)
#                         p.GetInherits().AddInherit(className)
#                     else:
#                         msg = "AddTextureVersion : not found 'modelVersion' attribute -> %s" % shape[0]
#                         self._msg.append(msg)
#         if self._msg:
#             msg = list(set(self._msg))
#             msg.sort()
#             dxsMsg.Print('warning', msg)
#         stage.GetRootLayer().Save()


    def makeVersionClass(self, stage, txBasePath, version):
        assetName = txBasePath.split('/')[-2]
        className = '/_{NAME}_{VER}_txVer'.format(NAME=assetName, VER=version)
        if className in self.verClassList:
            return className
        prim = stage.CreateClassPrim(className)
        vset = prim.GetVariantSets().GetVariantSet('modelVersion')
        vset.SetVariantSelection(version)
        texFile = os.path.join(self.showDir, txBasePath, 'texture.usd')
        if not os.path.exists(texFile):
            if self.debugCreate:
                Texture.TextureExport(overWrite=False, showDir=self.showDir, asset=assetName, version=version).doIt()
            msg = "[EditPrim.AddTextureVersion.makeVersionClass] - not found file -> '%s'" % texFile
            self._msg.append(msg)
        prim.SetPayload(Sdf.Payload(PathUtils.GetRelPath(self.inputFile, texFile)))
        self.verClassList.append(className)
        return className
