# encoding=utf-8
# !/usr/bin/env python

import maya.cmds as cmds
import maya.mel as mm

def delConstrains(parent):
    parent.textBrwsr.append("# Delete Constraints \n")
    StartTime = cmds.playbackOptions(q=1, min=True) - 1
    EndTime = cmds.playbackOptions(q=1, max=True) + 1

    consListA = cmds.ls(type='parentConstraint')
    consListB = cmds.ls(type='pointConstraint')
    consListC = cmds.ls(type='orientConstraint')
    consListD = cmds.ls(type='aimConstraint')
    consListE = cmds.ls(type='scaleConstraint')

    AllConstrainList = consListA + consListB + consListC + consListD + consListE

    for tempPrCons in AllConstrainList:
        if cmds.referenceQuery(tempPrCons, isNodeReferenced=True) == 0:
            ConnetedObj = cmds.listRelatives(tempPrCons, parent=True, type='transform')[0]
            cmds.bakeResults(ConnetedObj, simulation=False, dic=False, t=(StartTime, EndTime))
            mm.eval("performEulerFilter graphEditor1FromOutliner")
            cmds.keyTangent(ConnetedObj, itt='linear', ott='linear', animation='objects')
            cmds.delete(tempPrCons)
            parent.textBrwsr.append(" - Delete : {0}".format(tempPrCons))

    parent.textBrwsr.append("----" * 30 + "\n")

# 디스플레이 레이어 삭제

def delDispLYR(parent):
    parent.textBrwsr.append("# Delete Display Layer {0}\n".format("----"*30))
    dispLayers = cmds.ls(type='displayLayer')

    for dislyr in dispLayers:
        if dislyr != 'defaultLayer':
            if cmds.referenceQuery(dislyr, isNodeReferenced=True) == 0:
                cmds.delete(dislyr)

    parent.textBrwsr.append("----" * 30 + "\n")

# 사용하지 않는 노드 삭제
def deleteUnusedNodeCmd(parent):
    parent.textBrwsr.append("# Delete Unused Node \n")
    mm.eval('hyperShadePanelMenuCommand("hyperShadePanel1", "deleteUnusedNodes");')

    # delete turtle Node
    turtleNode = ['TurtleBakeLayerManager', 'TurtleDefaultBakeLayer', 'TurtleRenderOptions', 'TurtleUIOptions']
    rmanList = cmds.ls("rman*")
    rendermanList = cmds.ls("renderMan*")

    if rmanList or rendermanList:
        for rman in rmanList:
            cmds.delete(rman)
            parent.textBrwsr.append("- Delete Unused : {0}".format(rman))
        for renderman in rendermanList:
            cmds.delete(renderman)
            parent.textBrwsr.append("- Delete Unused : {0}".format(renderman))

    for i in turtleNode:
        try:
            cmds.lockNode(i, lock=False)
            cmds.delete(i)
            parent.textBrwsr.append("- Delete Unused : {0}".format(i))
        except:
            pass
    parent.textBrwsr.append("----" * 30 + "\n")

def deleteUnknownNode(parent):
    # unknownNode = cmds.ls(type = ("unknown", "unknownDag", "unknownTransform"))
    parent.textBrwsr.append('# Deleting Unknown Nodes \n')
    unknownNode = cmds.ls(type=("unknown"))

    for i in unknownNode:
        lockState = cmds.lockNode(i, q=True)[0]

        if lockState:
            cmds.lockNode(i, l=False)
            parent.textBrwsr.append("- Unlock Node : {0}".format( i ))

    try:
        cmds.delete(unknownNode)
        for node in unknownNode:
            parent.textBrwsr.append("- Delete Unknown Node : {0}".format( node))
    except:
        pass
    parent.textBrwsr.append("----" * 30 + "\n")

# AnimLayer 체크, merge

def AnimLYRmerge(parent):
    parent.textBrwsr.append('# Merging Anim Layer \n')
    animLayers = cmds.ls(type='animLayer')
    if animLayers != ['BaseAnimation'] and animLayers != []:
        LYRstr = ""

        for al in animLayers:
            LYRstr = LYRstr + '"%s"' % al
            if al != animLayers[-1]:
                LYRstr = LYRstr + ","

        mm.eval('string $layers[]={%s}; layerEditorMergeAnimLayer( $layers, 0 )' % LYRstr)
    parent.textBrwsr.append("----" * 30 + "\n")

def DeleteBarPlane(parent):
    parent.textBrwsr.append('# Delete BAR imagePlane \n')
    imgpln = cmds.ls(type="imagePlane")

    for i in imgpln:
        if i.find("BAR_plane") != -1:
            i_trans = cmds.listRelatives(i, p=1)
            cmds.delete(i_trans)
            parent.textBrwsr.append(' - Delete : {0}'.format(i_trans))

    parent.textBrwsr.append("----" * 30 + "\n")

def cleanUnknownPlugins(parent):
    parent.textBrwsr.append('# Delete Unknown Plugins \n')
    oldPlugins = cmds.unknownPlugin(q=True, list=True)
    if oldPlugins:
        for plugin in oldPlugins:
            cmds.unknownPlugin(plugin, remove=True)
            parent.textBrwsr.append(' - Delete : {0}'.format(plugin))
    parent.textBrwsr.append("----"*30 + "\n")


def checkDuplicateNames(parent):
    parent.textBrwsr.append('# Check Duplicated Names \n')
    cams = cmds.listCameras()
    meshs = cmds.ls(g=True, sn=True)
    meshNameDic = {}
    camNameDic = {}
    dupedNames = []

    parent.textBrwsr.append("  meshs : ")
    for i in meshs:
        meshTransform = cmds.listRelatives(i, p=True)[0]

        if meshTransform.find(":") != -1:
            ls_meshTransformSplit = meshTransform.split(":")
            meshNameDic[meshTransform] = ls_meshTransformSplit[-1]
        else:
            meshNameDic[meshTransform] = meshTransform

        meshList = cmds.ls(meshNameDic[meshTransform])

        if len(meshList) >= 2:
            if meshTransform not in dupedNames:
                dupedNames.append(meshTransform)
                parent.textBrwsr.append("   - {}".format(meshTransform))

    parent.textBrwsr.append("\n  cameras : ")
    for i in cams:
        if i.find(":") != -1 and i.find("|") != -1:
            ls_camsSplit = i.split(":")
            try:
                camName = ls_camsSplit[-1].split("|")
            except:
                camName = ls_camsSplit[-1]
            camNameDic[i] = camName
            camList = cmds.ls(camNameDic[i])
        elif i.find(":") != -1:
            ls_camsSplit = i.split(":")
            camNameDic[i] = ls_camsSplit[-1]
            camList = cmds.ls(camNameDic[i])
            print i, camList
        elif i.find("|") != -1:
            camName = i.split("|")[-1]
            camNameDic[i] = camName
            camList = cmds.ls(camNameDic[i])
        else:
            camNameDic[i] = i
            camList = cmds.ls(camNameDic[i])

        if len(camList) >= 2:
            if i not in dupedNames:
                dupedNames.append(i)
                parent.textBrwsr.append("   - {}".format(i))
    print dupedNames

    parent.textBrwsr.append("----" * 30 + "\n")

def cleanupSequencer(parent):
    if parent:
        parent.textBrwsr.append("# Delete Sequencer \n")
    seqList = cmds.ls(type='sequencer')
    deleteList = []

    for i in seqList:
        connections = cmds.listConnections(i)
        if not connections:
            cmds.delete(i)
        elif connections and len(connections) < 2:
            cmds.delete(i)
            deleteList.append(i)
            if parent:
                parent.textBrwsr.append("\n  > {0}".format(i))
        else:
            deleteNode = True
            for connection in connections:
                nodeType = cmds.nodeType(connection)
                if nodeType == 'shot':
                    deleteNode = False
            if deleteNode:
                cmds.delete(i)
                deleteList.append(i)
                if parent:
                    parent.textBrwsr.append("\n  > {0}".format(i))
    if parent:
        parent.textBrwsr.append("----" * 30 + "\n")
    return deleteList