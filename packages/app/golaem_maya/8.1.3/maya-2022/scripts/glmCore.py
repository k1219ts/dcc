#coding:utf-8
from __future__ import print_function
import os, json
import maya.cmds as cmds

import glm.devkit as glmapi

from glm.simCacheLib import simCacheLib
import glmUtils


class ReadSimCache:
    def __init__(self, cacheNode, gscb=None):
        # Member Variables
        self.cacheDir     = None
        self.cacheNode    = cacheNode
        self.cacheName    = None
        self.fieldNames   = None
        self.charFiles    = None
        self.sTerrainFile = ''
        self.dTerrainFile = ''
        self.layoutFiles  = list()
        self.killEntities = list()
        self.removeMesh   = dict()
        self.cacheRange   = [0, 0]
        self.assetFiles   = list()
        self.assetJoints  = list()
        self.agentSrc     = list()

        if gscb: self.gscb_proc(gscb)   # cached file process
        else:    self.node_proc()       # node process

        if not (self.cacheDir and self.cacheName and self.fieldNames and self.charFiles):
            assert False, 'Not found SimulationCache Information.'

        # init Golaem
        glmapi.initGolaem()
        glmapi.initSimulationCacheFactory()
        glmapi.loadSimulationCacheFactoryCharacters(';'.join(self.charFiles))

        if self.sTerrainFile or self.dTerrainFile:
            glmapi.loadSimulationCacheFactoryTerrain(self.sTerrainFile,
                                                     self.dTerrainFile)

        if self.layoutFiles:
            # kill entities node query
            for gscl in self.layoutFiles:
                f = open(gscl, "r")
                infos = json.load(f)
                f.close()

                for i in infos['nodes']:
                    if i['name'] == 'Kill' and i['active'] != 0:
                        connection = self.listNodeConnections(infos, i['ID'])
                        self.checkKillEntities(infos, connection)
                    elif i['name'] == 'AddRemoveMeshAssets' and i['active'] != 0:
                        for attr in i['attributes']:
                            if 'meshIndicesToRemove' in attr['name']:
                                infos['tmp_value'] = attr['values'][0]
                        connection = self.listNodeConnections(infos, i['ID'])
                        self.AddRemoveMeshAssets(infos, connection)

    def gscb_proc(self, filename):
        lib = simCacheLib.SimCacheLib()
        lib.readLibFile(filename)

        item = lib.items[0]

        self.cacheDir   = str(item.cacheDir)
        self.cacheName  = str(item.cacheName)
        self.fieldNames = []
        for n in item.crowdFields:
            self.fieldNames.append(str(n))
        self.charFiles = str(item.characterFiles).split(';')
        self.cacheRange = [int(item.startFrame), int(item.endFrame)]
        self.sTerrainFile = str(item.sourceTerrain)
        self.dTerrainFile = str(item.destTerrain)
        self.layoutFiles = str(item.layoutFile).split(';')

    def node_proc(self):
        nodes = cmds.ls(self.cacheNode, type='SimulationCacheProxy')
        if nodes:
            cacheProxy = nodes[0]
        else:
            return

        self.cacheDir    = str(cmds.getAttr('%s.inputCacheDir' % cacheProxy))
        self.cacheName   = str(cmds.getAttr('%s.inputCacheName' % cacheProxy))
        self.fieldNames  = str(cmds.getAttr('%s.crowdFields' % cacheProxy)).split(';')
        self.charFiles   = str(cmds.getAttr('%s.characterFiles' % cacheProxy)).split(';')
        self.cacheRange  = [cmds.getAttr('%s.isf' % cacheProxy),
                            cmds.getAttr('%s.ief' % cacheProxy)]

        if cmds.getAttr('%s.sourceTerrainFile' % cacheProxy):
            self.sTerrainFile = str(cmds.getAttr('%s.sourceTerrainFile' % cacheProxy))
        if cmds.getAttr('%s.destinationTerrainFile' % cacheProxy):
            self.dTerrainFile = str(cmds.getAttr('%s.destinationTerrainFile' % cacheProxy))

        if cmds.getAttr('%s.enableLayout' % cacheProxy):
            try:
                paths = cmds.getAttr('%s.layoutFiles[*].path' % cacheProxy)
                if isinstance(paths, list):
                    for path in paths:
                        if not os.path.isfile(path):
                            assert False, 'Error open layout data.'
                        self.layoutFiles.append(str(path))
                else:
                    if not os.path.isfile(paths):
                        assert False, 'Error open layout data.'
                    self.layoutFiles.append(str(paths))
            except:
                pass

    def doIt(self, fieldName):
        if self.layoutFiles:
            glmapi.loadSimulationCacheFactoryLayoutFiles(';'.join(self.layoutFiles))

        self.cacheIndex = glmapi.getCachedSimulationIndex(self.cacheDir, self.cacheName, fieldName)

        self.simData = glmapi.getFinalSimulationData(self.cacheIndex)
        if not self.simData:
            assert False, 'Error open simulation data.'

        # simulation static data
        self.entityCount  = self.simData._entityCount
        self.entityIds    = glmapi.int64Array_frompointer(self.simData._entityIds)
        self.entityScales = glmapi.floatArray_frompointer(self.simData._scales)
        self.charIds      = glmapi.intArray_frompointer(self.simData._characterIdx)

        # character data
        self.charsBoneIds = list()
        self.charsParentBoneIds = list()
        for f in self.charFiles:
            boneNames = glmapi.getBoneNames(f).split(';')
            boneIds = glmapi.intArray_frompointer(glmapi.getSortedBones(f))
            pBoneIds = glmapi.intArray_frompointer(glmapi.getParentBones(f))

            tmpB = []
            tmpPB = []
            for i in range(0, len(boneNames)):
                tmpB.append(boneIds[i])
                tmpPB.append(pBoneIds[i])

            self.charsBoneIds.append(tmpB)
            self.charsParentBoneIds.append(tmpPB)

    def getFrame(self, frame):
        frameData = glmapi.getFinalFrameData(self.cacheIndex, frame)
        if not frameData:
            # assert False, 'Error open simulation frame data.'
            return False
        self.bonePositions    = glmapi.floatArray_frompointer(glmapi.getBonePositions(frameData))
        self.boneOrientations = glmapi.floatArray_frompointer(glmapi.getBoneOrientations(frameData))
        return True

    def getEntity(self, index):     # entity index
        boneCount  = glmapi.getEntityBoneCount(index, self.simData)
        boneOffset = glmapi.getEntityBoneOffsetIndex(index, self.simData)

        cid = self.charIds[index]

        # pos, orient set by bone index
        bonePositions = [None] * boneCount
        boneOrientations = [None] * boneCount
        for i in range(0, boneCount):
            bid = self.charsBoneIds[cid][i]
            bonePositions[bid] = glmUtils.Get3List(self.bonePositions, boneOffset + i)
            boneOrientations[bid] = glmUtils.Get4List(self.boneOrientations, boneOffset + i)

        return boneCount, bonePositions, boneOrientations

    def getPBone(self, cid, bcnt):
        pid = glmapi.intArray_frompointer(glmapi.getParentBones(self.charFiles[cid]))
        return pid[bcnt]

    def getNodeById(self, infos, nodeId):
        for n in infos['nodes']:
            if n['ID'] == nodeId:
                return n

    def listNodeConnections(self, infos, nodeId):
        nodes = list()
        for i in infos['connections']:
            if nodeId == i[1]:
                nodes.append(i[0])
        return nodes

    def getInstance(self, inst):
        nodeId = None
        idx = None

        if '(' in inst[0]:
            for i in inst:
                if 'o(' in i:
                    nodeId = int(i.replace('o(', ''))
                else:
                    idx = int(i.replace(')', ''))
        return nodeId, idx

    def checkKillEntities(self, infos, nodes):
        for n in nodes:
            node = self.getNodeById(infos, n)
            if node['name'] == 'Selector' and node['active'] != 0:
                et = node['entities'].split(',')
                # per = int(node['percent'])
                nodeId, idx = self.getInstance(et)
                if nodeId:
                    connection = self.listNodeConnections(infos, nodeId)
                    self.checkKillEntities(infos, connection)
                elif isinstance(et, list):
                    # perIdx = int((len(et) * per) * 0.01)
                    # for e in et[:perIdx]:
                    for e in et:
                        self.killEntities.append(int(e))
                else:
                    self.killEntities.append(int(et))
            else:
                connection = self.listNodeConnections(infos, n)
                self.checkKillEntities(infos, connection)

    def AddRemoveMeshAssets(self, infos, nodes):
        for n in nodes:
            node = self.getNodeById(infos, n)
            if node['name'] == 'Selector' and node['active'] != 0:
                et = node['entities'].split(',')
                nodeId, idx = self.getInstance(et)
                if nodeId:
                    connection = self.listNodeConnections(infos, nodeId)
                    self.AddRemoveMeshAssets(infos, connection)
                elif isinstance(et, list):
                    for e in et:
                        self.removeMesh[int(e)] = infos['tmp_value']
                else:
                    self.removeMesh[int(et)] = infos['tmp_value']
            else:
                connection = self.listNodeConnections(infos, n)
                self.AddRemoveMeshAssets(infos, connection)

    def getRemoveMeshs(self, idx):
        cid = self.charIds[idx]
        entityId  = self.entityIds[idx]
        if self.removeMesh.has_key(entityId):
            removeMeshName = []
            meshes = cmds.glmCharacterFileTool(characterFile=self.charFiles[cid], getAttr='meshAssets')

            for i in self.removeMesh[entityId]:
                removeMeshName.append(meshes[i])
            return removeMeshName
        else:
            return None
