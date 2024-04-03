# encoding=utf-8
# !/usr/bin/env python

# -------------------------------------------------------------------------------
#
#   Dexter RenderMan TD sanghun.kim
#   Dexter CGSupervisor daeseok.chae
#
#	2020.08.18	$3
# -------------------------------------------------------------------------------

import os
import string
import getpass
import datetime
import json

import maya.cmds as cmds
import maya.mel as mel

import dxpublish.mayaCallBack as mayaCallBack
import dxpublish.insertDB as insertDB

import dxConfig
DB_IP = dxConfig.getConf('DB_IP')

import DXRulebook.Interface as rb

def GetViz(node):
    '''
    :node - full path string
    '''
    viz = True
    source = node.split('|')
    for i in range(1, len(source)):
        path = string.join(source[:i+1], '|')
        if cmds.listConnections('%s.visibility' % path):
            vals = cmds.keyframe(path, at='visibility', q=True, vc=True)
            if vals and not 1.0 in vals:
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

def checkRigReferenced(current_file):
    src = current_file.split('/')
    if not 'show' in src:
        return

    isUpdate = False
    versionUpRigNodeList = []
    nodes = cmds.ls(type = 'dxRig')
    for rigNode in nodes:
        if not cmds.referenceQuery(rigNode, inr=True):
            continue

        referenceFile = cmds.referenceQuery(rigNode, filename=True)
        referenceFile = referenceFile.split("{")[0]
        rigPubDir = os.path.dirname(referenceFile)
        curRigVerFile = os.path.basename(referenceFile)
        fileList = []
        excludePurpose = ['_low', '_sim', '_mid', '_mocap']
        for rigVerFile in sorted(os.listdir(rigPubDir)):
            if ".mb" in rigVerFile and not rigVerFile.startswith("."):
                check = 0
                for purpose in excludePurpose:
                    if not purpose in rigVerFile:
                        check += 1
                if check == len(excludePurpose):
                    fileList.append(rigVerFile)

        fileList.sort(reverse=True)

        if fileList[0] == curRigVerFile:
            continue

        isUpdate = True
        versionUpRigNodeList.append(rigNode.split(":")[-1])

    if isUpdate:
        versionUpRigNodeList = list(set(versionUpRigNodeList))

        text = ""
        for i in versionUpRigNodeList:
            text += "%s\n" % i

        text += u'\n해당 리그가 업데이트 되었습니다.\n버전을 확인해주세요'
        cmds.confirmDialog(title='Warning', message=text, button=['Ok'], icon = "Warning")


def getRigReferenced():
    referList = []
    for i in cmds.ls(type='dxRig'):
        if cmds.referenceQuery(i, inr = True):
            referenceFile = cmds.referenceQuery(i, filename=True)
            referenceFile = referenceFile.split("{")[0]
        else:
            referenceFile = ""
        referList.append(
            [i, cmds.getAttr('%s.action' % i), GetViz(cmds.ls(i, l=True)[0]), referenceFile]
        )
    return referList


# def getLayout():
#     layout = []
#     # pxrUsdReferenceAssembly
#     for n in cmds.ls(type='pxrUsdReferenceAssembly'):
#         if n.find('_set') > -1:
#             layout.append([n, 0, GetViz(cmds.ls(n, l=True)[0])])
#     # dxBlock
#     for n in cmds.ls(type='dxBlock'):
#         if n.find('_set') > -1 and (cmds.getAttr('%s.type' % n) == 2 or cmds.getAttr('%s.type' % n) == 1):
#             layout.append([n, cmds.getAttr('%s.action' % n), GetViz(cmds.ls(n, l=True)[0])])
#     return layout
def getLayout():
    layout = []
    # pxrUsdReferenceAssembly
    for n in cmds.ls('|*', type='pxrUsdReferenceAssembly'):
        layout.append([n, 1, GetViz(cmds.ls(n, l=True)[0]), 'extra'])
    # pxrUsdProxyShape
    for n in cmds.ls('|*|*', type='pxrUsdProxyShape'):
        trans = cmds.listRelatives(n, p=True)[0]
        layout.append([trans, 1, GetViz(cmds.ls(trans, l=True)[0]), 'extra'])
    # dxBlock
    for n in cmds.ls('|*', type='dxBlock'):
        if cmds.getAttr('%s.type' % n) == 1 or cmds.getAttr('%s.type' % n) == 2:
            layout.append([n, cmds.getAttr('%s.action' % n), GetViz(cmds.ls(n, l=True)[0]), 'dxBlock'])
    return layout


def getCameras():
    excludeCamera = list()
    cams = []
    isRenderable = 0
    # dxCamera
    for node in cmds.ls(type="dxCamera"):
        cams.append([node, cmds.getAttr('%s.action' % node), GetViz(cmds.ls(node, l=True)[0])])
        for cShape in cmds.ls(node, dag=True, type='camera', l=True):
            if cmds.getAttr('%s.renderable' % cShape) == True:
                isRenderable += 1
            excludeCamera.append(cShape)

    # excludeRule = ['|front', '|top', '|persp', '|back', '|side', '|left', '|right', '|bottom']
    # for camera in cmds.ls(type='camera', l=True):
    #     _ifever = 0
    #     for rule in excludeRule:
    #         if camera.startswith(rule):
    #             _ifever += 1
    #     if _ifever == 0 and not camera in excludeCamera:
    #         cams.append([camera, 1, GetViz(cmds.ls(camera, l=True)[0])])

    if isRenderable == 0:
        return list()
    return cams


def getCrowdInfo():
    result = []
    if cmds.pluginInfo('MiarmyProForMaya2018', q=True, l=True) and cmds.ls(type = "McdGlobal"):
        result.append(["crowd", 1, 1])
    elif cmds.pluginInfo('glmCrowd', q=True, l=True) and cmds.ls(type='SimulationCacheProxy'):
        for n in cmds.ls(type='SimulationCacheProxy'):
            enable = cmds.getAttr('%s.enable' % n)
            crowdFields = str(cmds.getAttr('%s.crowdFields' % n)).split(';')
            if isinstance(crowdFields, list):
                for cf in crowdFields:
                    result.append(['%s:%s' % (n, cf), 1, enable, 'golaem'])
            else:
                result.append(['%s:%s' % (n, crowdFields), 1, enable, 'golaem'])
    return result


def getSimulation():
    simNodes = []
    for n in cmds.ls(type='dxBlock'):
        if cmds.getAttr("%s.type" % n) == 3: # and cmds.getAttr("%s.action" % n) == 1:
            nsLayer = cmds.getAttr("%s.nsLayer" % n)
            nodeName = n
            if not ":" in n:
                nodeName = "%s:%s" % (nsLayer, n)
            simNodes.append(
                [nodeName, cmds.getAttr("%s.action" % n), GetViz(cmds.ls(n, l=True)[0])]
            )
    return simNodes

def getZennNodes():
    zennNodes = []
    for set in cmds.ls("ZN_ExportSet", r=True):
        for node in cmds.sets(set, q=True):
            zennNodes.append([node, 1, 1])
    return zennNodes

def clearGarbage():
    # Clear generate script userSetup.py, vaccine.py
    for i in cmds.ls(type='script'):
        if '_gene' in  i:
            cmds.delete(i)
            print('removeGarbage:', i)

    # delete script 'userSetup.py', 'vaccine.py', 'vaccine.pyc'
    homePath = os.path.expanduser('~/maya/scripts')
    if os.path.isdir(homePath):
        for i in os.listdir(homePath):
            if i in ['userSetup.py', 'vaccine.py', 'vaccine.pyc']:
                os.remove(os.path.join(homePath, i))
                print('removeGarbageFile:', i)

    # Clear 'Error: line 1: Cannot find procedure "DCF_updateViewportList".'
    EVIL_METHOD_NAMES = ['DCF_updateViewportList', 'CgAbBlastPanelOptChangeCallback']
    capitalEvilMethodNames = [name.upper() for name in EVIL_METHOD_NAMES]
    modelPanelLabel = mel.eval('localizedPanelLabel("ModelPanel")')
    processedPanelNames = []
    panelName = cmds.sceneUIReplacement(getNextPanel=('modelPanel', modelPanelLabel))
    while panelName and panelName not in processedPanelNames:
        editorChangedValue = cmds.modelEditor(panelName, query=True, editorChanged=True)
        parts = editorChangedValue.split(';')
        newParts = []
        changed = False
        for part in parts:
            for evilMethodName in capitalEvilMethodNames:
                if evilMethodName in part.upper():
                    changed = True
                    break
            else:
                newParts.append(part)
        if changed:
            cmds.modelEditor(panelName, edit=True, editorChanged=';'.join(newParts))
        processedPanelNames.append(panelName)
        panelName = cmds.sceneUIReplacement(getNextPanel=('modelPanel', modelPanelLabel))

# -------------------------------------------------------------------------------
#
#	OPEN
#
# -------------------------------------------------------------------------------
def openCallback(*args):
    current_file = cmds.file(q=True, sn=True)
    # NEW
    try:
        print('NEW')
        attachdic = {}
        attachdic['geoCache'] = getRigReferenced()
        attachdic['layout'] = getLayout()
        attachdic['camera'] = getCameras()
        attachdic["sim"] = getSimulation()
        attachdic["zenn"] = getZennNodes()
        attachdic["crowd"] = getCrowdInfo()
        result = insertDB.recordWork('maya', 'open', current_file, attachdic )
        checkRigReferenced(current_file)
    except Exception as e:
        print(e.message)

    source = current_file.split('/')
    if not 'show' in source:
        return

    for i in cmds.ls(rn=True, assemblies=True):
        if i.startswith('Mesh'):
            # need to know file path, nodeName
            # filepath
            refFilePath = cmds.referenceQuery(i, filename=True)
            # nodeName
            refNodeName = cmds.referenceQuery(i, rfn=True)
            cmds.file(refFilePath, loadReferenceDepth="asPrefs", loadReference=refNodeName)

    # scene clear
    clearGarbage()

    print('## Debug : open callback -> %s' % current_file)


# -------------------------------------------------------------------------------
#
#	SAVE
#
# -------------------------------------------------------------------------------
def saveCallback(*args):
    current_file = cmds.file(q=True, sn=True)

    # NEW
    try:
        attachdic = {}
        attachdic['geoCache'] = getRigReferenced()
        attachdic['layout'] = getLayout()
        attachdic['camera'] = getCameras()
        attachdic["sim"] = getSimulation()
        attachdic["zenn"] = getZennNodes()
        attachdic["crowd"] = getCrowdInfo()

        filename = os.path.basename(current_file)

        coder = rb.Coder()
        print(filename)
        rbRet = coder.F.MAYA.Decode(filename, 'BASE')
        print(rbRet)
        if rbRet.get('task'):
            attachdic["task"] = rbRet['task']
            writeSaveCallback(current_file, attachdic)
            result = insertDB.recordWork('maya', 'save', current_file, attachdic)
    except Exception as e:
        print(e.message)

    source = current_file.split('/')
    if not 'show' in source:
        return
    inclass = mayaCallBack.InsertFile(current_file)
    inclass.doUpdate()

    # ANI TACTIC LIBRARY UMASK CHANGE -rwxr--r-- -> -rwxrw-rw-
    if current_file.startswith('/tactic/library/ani'):
        try:
            os.chmod(current_file, 0766)
        except:
            pass

    # NEW
    try:
        checkRigReferenced(current_file)
    except Exception as e:
        print(e.message)


def writeSaveCallback(filePath, dataDic):
    dataDic['artist'] = getpass.getuser()
    dataDic['time'] = datetime.datetime.now().isoformat()
    dataDic['file'] = filePath
    fr = (cmds.playbackOptions(q=True, min = True), cmds.playbackOptions(q=True, max = True))
    dataDic['frameRange'] = fr
    dataDic['mayaVersion'] = os.environ['REZ_MAYA_VERSION']
    dataDic['rezResolve'] = []
    dataDic['rezRequest'] = []
    if os.environ.get('REZ_USED_RESOLVE'):
        for package in os.environ['REZ_USED_RESOLVE'].split():
            if 'centos' not in package:
                dataDic['rezResolve'].append(package)
    if os.environ.get('REZ_USED_REQUEST'):
        for package in os.environ['REZ_USED_REQUEST'].split():
            if 'centos' not in package:
                dataDic['rezRequest'].append(package)

    with open(filePath.replace(".mb", ".json"), "w") as f:
        json.dump(dataDic, f, indent=4)
