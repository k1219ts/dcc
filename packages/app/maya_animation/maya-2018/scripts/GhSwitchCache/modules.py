# -*- coding: utf-8 -*-

import os
import re
import logging
import json
import string
import maya.cmds as cmds
import sgAnimation
import sgAlembic
import sgCommon
import sgUI
import dxRigUI as drg

logger = logging.getLogger(__name__)

#DXNODES = ['dxRig', 'dxComponent', 'dxCamera', 'dxAssembly']
DXNODES = ['dxRig', 'dxComponent']
GCD_RIGS = ['/show/gcd1/asset/char/normalSpider/rig/pub/scenes/normalSpider_rig_v05_low.mb']

def gcdRigList():
    return GCD_RIGS

def getNameSpace(node):
    nameSpace = string.join(node.split(":")[:-1], ":")
    logger.debug(u'getNameSpace : {0} : < {1} >'.format(node, nameSpace))
    return nameSpace

def getAssetName(file):
    fileName = file.split(os.sep)[-1]
    assetName = fileName.split("_")[0]
    return assetName

def writeJson(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
        f.close()


def readJson(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def getBackstageDataPath():
    scenePath = os.path.dirname( cmds.file(q=True, sn=True) )
    dataPath = os.sep.join(scenePath.split(os.sep)[:-1] + ["data"])
    return dataPath


def getRootNode(object, type):
    """Find parent dxNode of selected object
    """
    if type == 'dxNode' and ( cmds.objectType(object) in DXNODES ):
            return object
    elif type == 'world_CON' and ( object.find("world_CON") != -1 ):
            return object

    node = object
    while True:
        parentNode = cmds.listRelatives(node, ap=1)
        if type == 'world_CON' and not parentNode:
            return node
        if type == 'dxNode' and not parentNode:
            node = None
            break
        if type == 'dxNode' and ( cmds.objectType(parentNode[0]) in DXNODES ):
            node = parentNode[0]
            break
        node = parentNode[0]
    return node


def getWorldConData(object):
    """Get world controler's key data
    
    :param object: dxRig or dxComponent node name
    :return: 
    """
    dxNode = getRootNode(object, type='dxNode')
    type = cmds.objectType(dxNode)
    if type == 'dxRig':
        pass
    elif type == 'dxComponent':
        worldCon = getRootNode(object, type='world_CON')
#        objectChilds = cmds.listRelatives(object, c=True)
        attr_worldCon = cmds.listAttr(worldCon, k=True)
        attr_worldCon.remove('visibility')
        attr_worldCon.remove('scaleX')
        attr_worldCon.remove('scaleY')
        attr_worldCon.remove('scaleZ')
#        attr_worldCon.remove('offset')
#        attr_worldCon.remove('speed')
        keyData = sgCommon.attributesKeyDump(node=worldCon, attrs=attr_worldCon)
        return keyData


def getCacheOptions(node):
    rootWorldCon = getRootNode(node, type='world_CON')
    options = dict()
    options['offset'] = float( cmds.getAttr(rootWorldCon + '.offset') )
    options['speed'] = float( cmds.getAttr(rootWorldCon + '.speed') )
    options['cacheFile'] = cmds.getAttr( node + '.abcFileName')
    return options


def setLibraryKey(node, dxRigNode, nameSpace):
    """
    
    :param node:dxComponent Node
    :return: 
    """
    if not cmds.pluginInfo('atomImportExport', q=True, l=True):
        cmds.loadPlugin('atomImportExport')

    option = getCacheOptions(node)

    if option['cacheFile'].find('_low') != -1:
        atomFileName = option['cacheFile'].replace('_low.abc', '.atom')
    else:
        atomFileName = option['cacheFile'].replace('.abc', '.atom')

    logger.debug(u'Atom file : {0}'.format(atomFileName))
    atomBaseName = os.path.basename(atomFileName)
    atomNameSpace = getNameSpace(atomBaseName)

    f = open(atomFileName, 'r')
    atomFile = f.read()
    atomFileSplit = atomFile.split(";")
    srcStart = 1.0
    srcEnd = 1.0
    for i in atomFileSplit:
        if i.find('startTime') != -1:
            srcStart = float(i.split(' ')[1])
        if i.find('endTime') != -1:
            srcEnd = float(i.split(' ')[1]) + 1
    dstStart = float(option['offset'])# + srcStart
    dstEnd = float(option['offset']) + (srcEnd - srcStart) / float(option['speed'])# + srcStart

    atomOption  = ";;targetTime=1;srcTime={sourceStart}:{sourceEnd};"
    atomOption += "dstTime={destStart}:{destEnd};"
    atomOption += "option=scaleReplace;match=string;;selected=selectedOnly;"
    atomOption += "search={character};replace={nameSpace};prefix=;suffix=;mapFile=;"

    atomOption = atomOption.format(sourceStart=srcStart,
                                   sourceEnd=srcEnd,
                                   destStart=dstStart,
                                   destEnd=dstEnd,
                                   character=atomNameSpace,
                                   nameSpace=nameSpace)

    #                                   character=re.sub('[^a-zA-Z]', '', nameSpace), # 'abc123' -> 'abc'
    print atomOption
    drg.selectAttributeObjects("{dxRignode}.controlers".format(dxRignode=dxRigNode))
    worldCon = [string.join([nameSpace, "place_CON"], ":"),
                string.join([nameSpace, "move_CON"], ":"),
                string.join([nameSpace, "direction_CON"], ":")]
    cmds.select(worldCon, d=True)
    cmds.file(atomFileName,
              i=True,
              type="atomImport",
              ra=True,
              namespace=nameSpace,
              options=atomOption)


def addCacheOptionTodxRigAttr(dxRigNode, options):
    cmds.addAttr(dxRigNode, ln='cacheName', dt='string')
    cmds.addAttr(dxRigNode, ln='cacheSpeed', at='float')
    cmds.addAttr(dxRigNode, ln='cacheOffset', at='float')
    cmds.setAttr(dxRigNode + '.cacheName', options['cacheFile'], type='string')
    cmds.setAttr(dxRigNode + '.cacheSpeed', float(options['speed']))
    cmds.setAttr(dxRigNode + '.cacheOffset', float(options['offset']))


def firstSwitch(node, namespace, options):
    """Switch library cache to rig

    :param node:dxComponent Node 
    :param options: 
    """
    logger.debug(u'Switch Cache < {0} >'.format(node))
    rigFile = options['rigfile']

    if not namespace:
        _namespace = getAssetName(rigFile)
    else:
        num = 1000
        p = re.compile('(\D+)(\d+)')
        re_num = p.search( namespace )
        if re_num:
            num = int(re_num.group(2)) + 1000
            _namespace = re_num.group(1) + str(num)
        else:
            _namespace = namespace + str(num)
        logger.debug(u'Namespace changed into < {0} >'.format(_namespace))

    wrdKeyData = getWorldConData(node)
    wrdNodeName = getRootNode(node, type='world_CON')
    logger.debug(u'Load Reference < {0} >, Namespace < {1} >'.format(rigFile, _namespace))
    dxRigNode = sgAnimation.loadReference(file=rigFile,
                                          namespace=_namespace,
                                          type='dxRig')
    logger.debug(u'Referenced dxRig Node < {0} >'.format(dxRigNode))
    cacheOptions = getCacheOptions(node)
    addCacheOptionTodxRigAttr(dxRigNode, cacheOptions)
    refNameSpace = getNameSpace(dxRigNode)
    placeCon = string.join([refNameSpace, "place_CON"], ":")
    sgCommon.attributesKeyLoad(placeCon, wrdKeyData)
    setLibraryKey(node, dxRigNode, refNameSpace)
    cmds.delete(wrdNodeName)
    return dxRigNode


def matchRigToGround(nameSpace, ground):
    cmds.setAttr(nameSpace + ":place_CON.recognizeGround", 1)
    groudShape = cmds.listRelatives(ground, s=True)[0]
    for i in cmds.ls("{0}:*_CPM*".format(nameSpace)):
        cmds.connectAttr('{}.worldMatrix'.format(ground),
                         '{}.inputMatrix'.format(i), f=True)
        cmds.connectAttr('{}.outMesh'.format(groudShape),
                         '{}.inMesh'.format(i), f=True)


def exportCache(selection, options):
    """Export gpu cache
    
    :param selection: from selection or all dxRig nodes. (boolean)
    :param options: export options. (dictionary)
    """
    if selection:
        nodes = options['nodes']
        #nodes = cmds.ls(sl=True, type=['dxRig', 'dxComponent'])
    else:
        nodes = cmds.ls(type='dxRig')
    filePath = options['filepath']
    start = options['startFrame']
    end = options['endFrame']
    step = options['step']
    fjust = False

    if not os.path.exists(filePath):
        os.mkdir(filePath)

    logger.debug(u"export cache of : {0}".format(nodes))

    abcClass = sgAlembic.CacheExport(FilePath=filePath,
                                     Nodes=nodes,
                                     Start=start,
                                     End=end,
                                     Step=step,
                                     Just=bool(fjust))
    abcClass.doIt()

    # write rig <-> cache link data file
    linkData = dict()
    for node in nodes:
        linkData[node+".abc"] = cmds.referenceQuery(node, referenceNode=True)
        linkDataFile = os.sep.join([filePath, node + ".link"])
        writeJson(linkData, linkDataFile)

def switchCache(selection, options):
    """Switch gpu cache

    :param selection: from selection or all dxRig nodes. (boolean)
    :param options: switch options. (dictionary)
    """
    if selection:
        nodes = options['nodes']
    else:
        nodes = cmds.ls(type='dxRig')

    for node in nodes:
        nodeType = cmds.nodeType(node)
        cacheName = node + '.abc'
        cachePath = os.sep.join([options['crowdCachePath'], cacheName])
        linkFilePath = os.sep.join([options['crowdCachePath'], node + '.link'])
        if nodeType == 'dxRig':
            if not os.path.exists(cachePath):
                return
            logger.debug("node : {0}".format(node))
            logger.debug("to Alembic : {0}\n".format(cacheName))
            refNodeName = cmds.referenceQuery(node, referenceNode=True)
            cmds.file(unloadReference=refNodeName)
            sgCompClass = sgUI.ComponentImport(Files=[cachePath])
            sgCompClass.doIt()
        else:
            logger.debug("node : {0}".format(node))
            logger.debug("to Rigging : {0}\n".format(options['rigfile']))

            crowdCachePath = options['crowdCachePath']
            if os.path.exists(crowdCachePath) and cacheName in os.listdir(crowdCachePath):
                linkData = readJson(linkFilePath)
                wrdNodeName = getRootNode(node, type='world_CON')
                refNodeName = linkData[cacheName]
                cmds.delete(wrdNodeName)
                cmds.file(loadReference=refNodeName)
            else:
                firstSwitch(node=node, namespace=None, options=options)



