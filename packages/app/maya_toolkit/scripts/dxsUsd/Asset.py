import os
import string
import time

import maya.api.OpenMaya as OpenMaya
import maya.cmds as cmds

import dxsMayaUtils
import SessionUtils
import GeomMain
import EnvSet
import Model
import Sim
import Zenn
import XGen


def GetTempAssetPath():
    current = cmds.file(q=True, sn=True)
    if current:
        current = os.path.dirname(current)
    else:
        current = cmds.workspace(q=True, rd=True)
    rootDir = os.path.join(current, 'tmp')
    return str(rootDir)


def TempAssetExport(tmpDir=None, showDir=None, clear=True):
    if not tmpDir:
        tmpDir = GetTempAssetPath()
    outFiles = list()
    cmds.waitCursor(state=True)
    selected  = cmds.ls(sl=True)
    startTime = time.time()

    #---------------------------------------------------------------------------
    # model
    outFiles += Temp_Model(tmpDir, selected)

    #---------------------------------------------------------------------------
    # xBlock
    outFiles += Temp_xBlock(tmpDir, selected)

    #---------------------------------------------------------------------------
    # zenn
    outFiles += Temp_Zenn(tmpDir, selected)
    # xgen
    outFiles += Temp_XGen(tmpDir, selected)

    #---------------------------------------------------------------------------
    # EnvSet
    outFiles += Temp_EnvSet(tmpDir, selected, showDir)

    # ---------------------------------------------------------------------------
    # Yetti Feather
    outFiles += Temp_YettiFeather(tmpDir, selected, showDir)

    #---------------------------------------------------------------------------
    OpenMaya.MGlobal.displayInfo('# Result : TempAssetExport finished!')
    endTime = time.time()
    print '# compute time : %.3f sec' % (endTime - startTime)

    cmds.waitCursor(state=False)
    if not outFiles:
        return
    tmpFile= os.path.join(tmpDir, 'cache.usd')
    source = list()
    for f in list(set(outFiles)):
        source.append((f, None))
    SessionUtils.MakeReferenceStage(tmpFile, source, SdfPath='/asset', composite='reference', clear=clear)
    return tmpFile


#-------------------------------------------------------------------------------
# Model
#-------------------------------------------------------------------------------
def Temp_Model(tmpDir, selected):
    outFiles = list()

    if selected:
        cmds.select(selected)
        nodes = cmds.ls('|*_model_*GRP', sl=True, r=True)
    else:
        nodes = cmds.ls('|*_model_*GRP', r=True)
    exportNodes = list()
    for n in nodes:
        if n.find('_low') == -1 and n.find('_mid') == -1:
            exportNodes.append(n)
    for n in exportNodes:
        if dxsMayaUtils.GetViz(cmds.ls(n, l=True)[0]):
            assetName = n.split('_model')[0]
            outDir = os.path.join(tmpDir, assetName)
            modelClass = Model.ModelExport(node=n, outDir=outDir, asset=assetName, version='v000')
            modelClass.doIt()
            outFiles.append(os.path.join(outDir, 'model.usd'))
    return outFiles

#-------------------------------------------------------------------------------
# xBlock
#-------------------------------------------------------------------------------
def Temp_xBlock(tmpDir, selected):
    outFiles = list()

    if selected:
        nodes = cmds.ls(selected, type='xBlock', l=True)
    else:
        nodes = cmds.ls(type='xBlock', l=True)

    ctime  = int(cmds.currentTime(q=True))
    for n in nodes:
        if dxsMayaUtils.GetViz(cmds.ls(n, l=True)[0]):
            nsName, nodeName = dxsMayaUtils.GetNamespaceInfo(n)
            assetName = nodeName.split('_rig_GRP')[0]
            action    = cmds.getAttr('%s.action' % n)
            if action == 1:
                geomFile = '{DIR}/{NAME}/v000/{NODE}.high_geom.usd'.format(DIR=tmpDir, NAME=assetName, NODE=nodeName)
                GeomMain.Export(geomFile, [n], fr=(ctime, ctime), isAbc=False).doIt()
                Sim.EditGeom(geomFile, n).doIt()
                outFiles.append(geomFile)
            elif action == 2:
                importFile = cmds.getAttr('%s.importFile' % n)
                outFiles.append(importFile)
    return outFiles

#-------------------------------------------------------------------------------
# Zenn
#-------------------------------------------------------------------------------
def Temp_Zenn(tmpDir, selected):
    if not cmds.objExists('ZN_ExportSet'):
        return list()
    globalNode = cmds.ls(type='ZN_Global')
    rootNode   = cmds.listRelatives(globalNode[0], p=True)[0]
    assetName  = rootNode.split('|')[-1].split(':')[-1].split('_ZN')[0]

    exportNodes= list()
    setNodes   = cmds.sets('ZN_ExportSet', q=True)
    if selected:
        nodes = cmds.ls(selected, type=['ZN_Deform', 'ZN_FeatherInstance'])
        for n in nodes:
            if n in setNodes:
                exportNodes.append(n)
    else:
        exportNodes = setNodes
    if not exportNodes:
        return list()

    geomDir = '{DIR}/{NAME}/v000'.format(DIR=tmpDir, NAME=assetName)
    ctime   = int(cmds.currentTime(q=True))
    outFile = Zenn.ZennGeomExport(geomDir, nodes=exportNodes, fr=(ctime, ctime)).doIt()
    return [outFile]

#-------------------------------------------------------------------------------
# XGen
#-------------------------------------------------------------------------------
def Temp_XGen(tmpDir, selected):
    if not cmds.objExists('XG_ExportSet'):
        return list()
    dxsMayaUtils.PluginSetup(['DXS_XGenPlugin'])
    globalNode = cmds.ls(type='xgmSplineDescription')
    rootNode   = cmds.listRelatives(globalNode[0], p=True)[0]
    assetName  = rootNode.split('|')[-1].split(':')[-1].split('_')[0]

    exportNodes= list()
    setNodes   = cmds.sets('XG_ExportSet', q=True)
    if selected:
        nodes = cmds.ls(selected, type=['xgmSplineDescription'])
        for n in nodes:
            if n in setNodes:
                exportNodes.append(n)
    else:
        exportNodes = setNodes
    if not exportNodes:
        return list()

    geomDir = '{DIR}/{NAME}/v000'.format(DIR=tmpDir, NAME=assetName)
    ctime   = int(cmds.currentTime(q=True))
    outFile = XGen.XGenGeomExport(geomDir, nodes=exportNodes, fr=(ctime, ctime)).doIt()
    return [outFile]

#-------------------------------------------------------------------------------
# EnvSet
#-------------------------------------------------------------------------------
def Temp_EnvSet(tmpDir, selected, showDir = ""):
    outFiles = list()

    if selected:
        nodes = list()
        for n in cmds.ls(selected, l=True, r=True):
            if n.find('_set') > -1:
                nodes.append(n)
    else:
        nodes = cmds.ls('*_set*', tr=True, l=True, r=True)

    for n in EnvSet.GetSetNodes(nodes):
        if dxsMayaUtils.GetViz(n) and n.find('_set') > -1:
            name = n.split('|')[-1].split(':')[-1]
            assetName= name.split('_set')[0] + '_set'
            outDir   = os.path.join(tmpDir, assetName)
            SetClass = EnvSet.SetAssetExport(node=n, outDir=outDir, asset=assetName, version='v000')
            SetClass.doIt()
            if showDir and SetClass.geomFile:
                texOverrideFile = '{DIR}/asset/{ASSET}/texture/texture.override.usd'.format(DIR=showDir, ASSET=assetName)
                if os.path.exists(texOverrideFile):
                    EnvSet.SetAssetExport.AddOverrideInherit(SetClass.geomFile, texOverrideFile)
            outFiles.append(os.path.join(outDir, 'model.usd'))
        else:
            cmds.error('# asset name convention error!')
    return outFiles

#-------------------------------------------------------------------------------
# Temp Yetti Feather
#-------------------------------------------------------------------------------
def Temp_YettiFeather(tmpDir, selected, showDir = ""):
    if not cmds.pluginInfo("pgYettiMaya", q=True, l=True):
        return list()

    outFiles = list()
    nodes = list()
    if selected:
        pass
    else:
        pass

    for n in nodes:
        filename = os.path.join(tmpDir) # tmpDir Path in dxsMTK + $assetName/model/v000/files
        outFiles.append(dxsMayaUtils.UsdExport(filename, n))# Export Cmd

    return outFiles


# def LibExport(version='v001', overWrite=False, type='model'):
#     for n in cmds.ls(sl=True):
#         assetName = n.split('|')[-1].split(':')[-1].split('_model')[0]
#         nodeName, geomType = dxsMayaUtils.GetNodeInfo(n)
#         print assetName, nodeName, geomType
#         DelTxAttr(n)
#         if type == 'model':
#             print '>> [ Model ]'
#             Model.AssetExport(n, isPurpose=True, overWrite=overWrite, showDir='/assetlib/3D', asset=assetName, version=version).doIt()
#
#         if type == 'clip':
#             print '>> [ Clip ]'
#             Model.ClipExport(n, isPurpose=True, overWrite=overWrite, loopScales=[0.5, 1.0, 1.5], loopRange=(1001, 2000), showDir='/assetlib/3D', asset=assetName, version=version).doIt()
#
#         Texture.TextureExport(showDir='/assetlib/3D', asset=assetName, version='v001').doIt()
#
# def Export(version='v001'):
#     current = cmds.file(q=True, sn=True)
#     if not current:
#         current = cmds.workspace(q=True, rd=True)
#     splitPath = current.split('/')
#     showIndex = splitPath.index('show')
#     showDir = string.join(splitPath[:showIndex+2], '/')
#
#     for n in cmds.ls(sl=True):
#         assetName = n.split('|')[-1].split(':')[-1].split('_model')[0]
#         Model.AssetExport(n, isPurpose=True, showDir=showDir, asset=assetName, version=version).doIt()
#
#         Texture.TextureExport(showDir=showDir, asset=assetName, version='v001').doIt()


_txAssetName = 'rman__riattr__user_txAssetName'
_txLayerName = 'rman__riattr__user_txLayerName'
_txBasePath  = 'rman__riattr__user_txBasePath'

def DelTxAttr(node):
    for s in cmds.ls(node, dag=True, type='surfaceShape', ni=True):
        if cmds.attributeQuery(_txBasePath, n=s, ex=True):
            cmds.deleteAttr('%s.%s' % (s, _txBasePath))
        if cmds.attributeQuery(_txAssetName, n=s, ex=True):
            cmds.deleteAttr('%s.%s' % (s, _txAssetName))


def PrintLayerNames():
    layerNames = list()
    for i in cmds.ls(sl=True, dag=True, type='surfaceShape', ni=True, l=True):
        if cmds.attributeQuery(_txLayerName, n=i, ex=True):
            getVal = cmds.getAttr('%s.%s' % (i, _txLayerName))
            layerNames.append(getVal)
    layerNames = list(set(layerNames))
    layerNames.sort()
    print layerNames


def ClearUserAttributes():
    for s in cmds.ls(sl=True, dag=True, s=True, ni=True):
        attrs = cmds.listAttr(s, ud=True)
        if attrs:
            for ln in attrs:
                cmds.deleteAttr('%s.%s' % (s, ln))


def CopyAttributes():
    sel = cmds.ls(sl=True, dag=True, s=True, ni=True)
    if len(sel) != 2:
        assert False, '# msg : selection error!'
    userAttrs = cmds.listAttr(sel[0], ud=True)
    if userAttrs:
        for ln in userAttrs:
            getVal = cmds.getAttr('%s.%s' % (sel[0], ln))
            getTyp = cmds.getAttr('%s.%s' % (sel[0], ln), typ=True)
            if not cmds.attributeQuery(ln, n=sel[1], ex=True):
                if getTyp == 'string':
                    cmds.addAttr(sel[1], ln=ln, dt=getTyp)
                else:
                    cmds.addAttr(sel[1], ln=ln, at=getTyp)
            if getTyp == 'string':
                cmds.setAttr('%s.%s' % (sel[1], ln), getVal, type=getTyp)
            else:
                cmds.setAttr('%s.%s' % (sel[1], ln), getVal)
