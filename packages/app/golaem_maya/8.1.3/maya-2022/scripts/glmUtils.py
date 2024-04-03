#coding:utf-8
import os
import maya.mel as mel
import maya.cmds as cmds
import DXUSD_MAYA.MUtils as mutl

import glm.crowdUtils as crdUtl

if not cmds.pluginInfo('glmCrowd', q=True, l=True):
    cmds.loadPlugin('glmCrowd')

def bakeCacheProxy():
    rangeOp = ''

    nodes = cmds.ls(sl=True)
    if nodes:
        if cmds.ls(nodes[0], dag=True, type='SimulationCacheProxy', ni=True):
            rangeOp = cmds.confirmDialog(title='GolaemUtils', message='%s\nStart Cache Bake?!' % ', '.join(nodes),
                                         button=['CacheRange', 'FrameRange', 'FrameRange+10', 'Cancel'], defaultButton='CacheRange', cancelButton='No')

            if rangeOp in ['Cancel', 'dismiss']:
                cmds.confirmDialog(title='GolaemUtils', message='Cancel Bake!')
                return
    else:
        cmds.confirmDialog(title='GolaemUtils', message='Select cacheProxy Node!')
        return

    for node in nodes:
        cacheProxy = cmds.ls(node, dag=True, type='SimulationCacheProxy', ni=True)[0]
        renderProxy = cmds.ls(node, dag=True, type='CrowdRenderProxy', ni=True)[0]
        cacheDir = str(cmds.getAttr('%s.inputCacheDir' % cacheProxy))
        cacheName = str(cmds.getAttr('%s.inputCacheName' % cacheProxy))

        if 'CacheRange' == rangeOp:
            frameRange = [int(cmds.getAttr('%s.isf' % cacheProxy)),
                          int(cmds.getAttr('%s.ief' % cacheProxy))]
        elif 'FrameRange' == rangeOp:
            frameRange = mutl.GetFrameRange()
        elif 'FrameRange+10' == rangeOp:
            range = mutl.GetFrameRange()
            frameRange = [range[0] - 10, range[1] + 10]

        print '[%s] frameRange:' % rangeOp, frameRange

        if '_bake' not in cacheName or '_bake' not in cacheDir:
            bCacheName = cacheName + '_bake'
            bCacheDir = cacheDir + '_bake'

            cmds.setAttr('crowdManagerShape.bkStartFrame', frameRange[0])
            cmds.setAttr('crowdManagerShape.bkEndFrame', frameRange[1])
            cmds.setAttr('crowdManagerShape.expBakeCacheName', bCacheName, type='string')
            cmds.setAttr('crowdManagerShape.expBakeCacheLayoutName', bCacheName, type='string')
            cmds.setAttr('crowdManagerShape.expBakeCacheOutDir', bCacheDir, type='string')

            cmds.glmCrowdSimulationBaker(sF=frameRange[0], eF=frameRange[1], rpn=renderProxy, eid='*',
                                         cen=True, can=bCacheName, layoutName=bCacheName, cod=bCacheDir)

            caches = cmds.ls(type='SimulationCacheProxy', ni=True)
            for cache in caches:
                caName = cmds.getAttr('%s.ica' % cache)
                if bCacheName in caName:
                    cmds.disconnectAttr('crowdManagerShape.bkStartFrame', '%s.inputStartFrame' % cache)
                    cmds.disconnectAttr('crowdManagerShape.bkEndFrame', '%s.inputEndFrame' % cache)
                    cmds.disconnectAttr('crowdManagerShape.expBakeCacheName', '%s.inputCacheName' % cache)
                    cmds.disconnectAttr('crowdManagerShape.expBakeCacheOutDir', '%s.inputCacheDir' % cache)
        else:
            cmds.confirmDialog(title='GolaemUtils', message='%s - This node is already Baked!\nCheck please.' % node)

    cmds.confirmDialog(title='GolaemUtils', message='Bake cacheProxy Complate!')

def exportCharacter(rootBone, geoGroup, filename):  # filename is maya scene
    ext = os.path.basename(filename).split('.')[-1]
    outputChar    = filename.replace('.%s' % ext, '.gcha')
    outputCharGeo = filename.replace('.%s' % ext, '.gcg')

    cmLocator = cmds.ls(typ='CharacterMakerLocator')
    if not cmLocator:
        if os.path.exists(outputChar):
            cmds.confirmDialog(title='GolaemUtils', message='File Exists!!\n%s' % outputChar)
            outputChar = cmds.fileDialog2()
            if not outputChar:
                cmds.confirmDialog(title='GolaemUtils', message='Cancel!!')
                return None
            outputCharGeo = outputChar.replace('.%s' % ext, '.gcg')

        cmds.glmCharacterMaker(script=True, file='',
                               addSkeleton=(rootBone, 1, 1, 2, 0.05, 0.05),
                               addGeometry=('CharacterNode', 'Character', geoGroup),
                               outputFile=outputChar)

        # create default rendering type
        cmds.glmCharacterMaker(script=True, file=outputChar,
                               addRenderingType='Default',
                               outputFile=outputChar)

        # init weights
        targets = []
        for i in cmds.ls(geoGroup, dag=True, type='transform'):
            if not cmds.listRelatives(i, children=True, type='surfaceShape'):
                targets.append(i)

        for i in targets:
            cmds.glmCharacterMaker(script=True, file=outputChar,
                                   editProperties=('AssetGroupNode', i, 'weighted=0'),
                                   outputFile=outputChar)
    else:
        # gcha save as to worksPath
        outputChar = cmds.getAttr(cmLocator[0] + '.currentFile')
        if os.path.exists(outputChar):
            cmds.confirmDialog(title='GolaemUtils', message='File Exists!!\n%s' % outputChar)
            outputChar = cmds.fileDialog2()
            if not outputChar:
                cmds.confirmDialog(title='GolaemUtils', message='Cancel!!')
                return None
            outputCharGeo = outputChar.replace('.%s' % ext, '.gcg')

        cmds.glmCharacterMaker(script=True, file=outputChar, outputFile=outputChar)

    # export GCG
    cmds.glmExportCharacterGeometry(characterFile=outputChar,
                                    outputFileGCG=outputCharGeo)

    # # Link the two together
    characterMeshes = cmds.glmCharacterFileTool(characterFile=outputChar,
                                                getAttr='meshAssets')
    characterBbox = crdUtl.computeCharacterBBox(rootBone, characterMeshes, 0.2)
    geometryParam = 'geometry=' + outputCharGeo + ',' + str(characterBbox[0]) + \
                    ',' + str(characterBbox[1]) + str(',0,0,10000') # bounding box / LOD min distance / LOD max distance
    cmds.glmCharacterMaker(script=True, file=outputChar,
                           editProperties=('CharacterNode', 'Character', geometryParam),
                           outputFile=outputChar)

    return outputChar

def openGchaFile():
    scenes = cmds.file(q=True, sn=True)
    gchaFile = scenes.replace('.mb', '.gcha')
    if os.path.exists(gchaFile):
        cmds.glmCharacterMaker(file=gchaFile)
    else:
        cmds.confirmDialog(title='GolaemUtils', message='Please check gcha file path!!\n%s' % gchaFile)

def initWeights():
    targets = []
    geoGroup = cmds.ls('geometry_GRP', type='transform')

    for i in cmds.ls(geoGroup, dag=True, type='transform'):
        if not cmds.listRelatives(i, children=True, type='surfaceShape'):
            targets.append(i)

    cmLocator = cmds.ls(typ='CharacterMakerLocator')
    outputChar = ''
    if cmLocator:
        outputChar = cmds.getAttr(cmLocator[0] + '.currentFile')
    if not outputChar:
        scenes = cmds.file(q=True, sn=True)
        outputChar = scenes.replace('.mb', '.gcha')

    if os.path.exists(outputChar):
        for i in targets:
            cmds.glmCharacterMaker(script=True, file=outputChar,
                                   editProperties=('AssetGroupNode', i, 'weighted=0'),
                                   outputFile=outputChar)

        cmds.confirmDialog(title='GolaemUtils', message='Weighted=0\nset Complate!\n\nPlease gcha file reOpen!!\n%s' % outputChar)
    else:
        cmds.confirmDialog(title='GolaemUtils', message='Please check gchafile!!\n%s' % outputChar)

def initWorkspace(rootBone=None, geoGroup=None):
    sceneFile = cmds.file(q=True, sn=True)
    if not rootBone:
        rootBone = cmds.ls(type='joint', dag=True)
    if not geoGroup:
        geoGroup = cmds.ls('geometry_GRP', type='transform')

    # disable Joint Segment Scale Compensate
    jnt = cmds.ls(type='joint', dag=True)
    for i in jnt:
        cmds.setAttr('%s.segmentScaleCompensate' % i, 0)

    if rootBone and geoGroup:
        gchaFile = exportCharacter(rootBone[0], geoGroup[0], sceneFile)
        if gchaFile:
            cmds.glmCharacterMaker(file=gchaFile)
    else:
        cmds.confirmDialog(title='GolaemUtils', message='not found Nodes!!')

def replacePopTool():
    popTools = cmds.ls(type='PopulationToolLocator', dag=True, sl=True)
    if not popTools:
        popTools = cmds.ls(type='PopulationToolLocator')
    else:
        for pt in popTools:
            ptShape = cmds.getAttr('%s.particleSystemName' % pt)
            cmds.glmEmitParticles(pt=pt, rpl=ptShape)
        else:
            cmds.confirmDialog(title='GolaemUtils',
                               message='%s\npopTool replace Complate!!' % popTools)

def Get3List(data, index):
    return [data[index*3], data[index*3+1], data[index*3+2]]

def Get4List(data, index):
    return [data[index*4], data[index*4+1], data[index*4+2], data[index*4+3]]
