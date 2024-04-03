import os
import json
import string
import logging
from alembic.Abc import *
import maya.cmds as cmds
import aniCommon

logger = logging.getLogger(__name__)

DXNODES = ['dxRig', 'dxComponent']


def writeJson(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
        f.close()


def readJson(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def getBackstageDataPath():
    scenePath = os.path.dirname(cmds.file(q=True, sn=True))
    dataPath = os.sep.join(scenePath.split(os.sep)[:-1] + ["data"])
    return dataPath


def getRootNode(object, type):
    """Find parent dxNode of selected object
    """
    if type == 'dxNode' and (cmds.objectType(object) in DXNODES):
        return object
    elif type == 'world_CON' and (object.find("world_CON") != -1):
        return object

    node = object
    while True:
        parentNode = cmds.listRelatives(node, ap=1)
        if type == 'world_CON' and not parentNode:
            return node
        if type == 'dxNode' and not parentNode:
            node = None
            break
        if type == 'dxNode' and (cmds.objectType(parentNode[0]) in DXNODES):
            node = parentNode[0]
            break
        node = parentNode[0]
    return node


class switchCache():
    def __init__(self, options):
        self._options = options
        if not cmds.pluginInfo('AbcExport', q=True, l=True):
            cmds.loadPlugin('AbcExport')
        if not cmds.pluginInfo('gpuCache', q=True, l=True):
            cmds.loadPlugin('gpuCache')

    @property
    def options(self):
        return self._options

    def abcJobCommand(self, filename):
        opts = '-uv -ws -wv -wuvs -ef -a ObjectSet -a ObjectName -atp rman -df ogawa'
        cmd = '-fr {start} {end} -s 1 -file {fileName} {options}'
        cmd = cmd.format(start=self.options['startFrame'],
                         end=self.options['endFrame'],
                         fileName=filename,
                         options=opts)
        return cmd

    def getRefnode(self, selection):
        node = cmds.getAttr(selection + ".referenceNode")
        return node

    def getAlembicMeshes(self, abcFile):
        iarch = IArchive(abcFile)
        root = iarch.getTop()

        meshes = list()
        for mesh in root.children:
            meshes.append(mesh.getName())

        return meshes

    def getLowMeshes(self, node, meshtype='lowMeshes'):
        nameSpace = aniCommon.getNameSpace(node)
        meshes = list()

        for mesh in cmds.getAttr(node + '.{meshType}'.format(meshType=meshtype)):
            fullName = string.join([nameSpace, mesh], ":")
            if cmds.objExists(fullName):
                meshes.append(fullName)

        return meshes

    def addToJob(self, node, filename):
        cmd = self.abcJobCommand(filename)
        lowMeshes = self.getLowMeshes(node, self.options['meshType'])

        if not lowMeshes:
            return

        for mesh in lowMeshes:
            cmd += ' -root {0}'.format(mesh)

        return cmd

    def importAlembic(self, node, cachefile):
        refNodeName = cmds.referenceQuery(node, referenceNode=True)
        cmds.file(unloadReference=refNodeName)
        dxCompNode = cmds.createNode('dxComponent', n=node)

        if self.options['toGpuCache']:
            cacheNode = cmds.createNode('gpuCache')
            cacheTransform = cmds.listRelatives(cacheNode, p=True)[0]
            cmds.rename(cacheTransform, node + '_GPU')
            cmds.setAttr(cacheNode + '.cacheFileName', cachefile, type='string')
            cmds.parent(cacheNode, dxCompNode)
        else:
            cmds.AbcImport(cachefile, mode='import')
            importedMeshes = self.getAlembicMeshes(cachefile)
            cmds.parent(importedMeshes, dxCompNode)

        cmds.addAttr(dxCompNode, ln='referenceNode', dt='string')
        cmds.setAttr(dxCompNode + '.referenceNode', e=True, keyable=False)
        cmds.setAttr(dxCompNode + '.referenceNode', refNodeName, type='string')


    def create(self, nodes):
        jobCmd = list()
        for node in nodes:
            cacheName = node + '.abc'
            cachePath = os.sep.join([self.options['filepath'], cacheName])

            if cmds.nodeType(node) == 'dxRig':
                if self.options['exportMaterials'] and self.options['update']:
                    meshes = self.getLowMeshes(node, meshtype=self.options['meshType'])
                    gpuCacheKwrgs = {'startTime': float(self.options['startFrame']),
                                     'endTime': float(self.options['endFrame']),
                                     'optimize': True,
                                     'optimizationThreshold': 40000,
                                     'writeMaterials':True,
                                     'dataFormat': "ogawa",
                                     'directory':self.options['filepath'],
                                     'fileName': node,
                                     'saveMultipleFiles': False}
                    cmds.gpuCache(meshes, **gpuCacheKwrgs)
                else:
                    if not os.path.exists(cachePath) or self.options['update']:
                        cmd = self.addToJob(node, cachePath)
                        jobCmd.append(cmd)

        if not self.options['exportMaterials']:
            cmds.AbcExport(v=True, j=jobCmd)

    def switch(self):
        nodes = self.options['nodes']

        if not os.path.exists(self.options['filepath']):
            os.makedirs(self.options['filepath'])

        self.create(nodes)

        for node in nodes:
            nodeType = cmds.nodeType(node)
            cacheName = node + '.abc'
            cachePath = os.sep.join([self.options['filepath'], cacheName])

            if nodeType == 'dxRig':
                logger.debug("node : {0}".format(node))
                logger.debug("to Alembic : {0}\n".format(cacheName))
                self.importAlembic(node, cachePath)
            else:
                logger.debug("node : {0}".format(node))
                logger.debug("to Rigging\n")

                switchCachePath = self.options['filepath']

                if os.path.exists(switchCachePath) and cacheName in os.listdir(switchCachePath):
                    rootNodeName = getRootNode(node, type='dxNode')
                    refNodeName = self.getRefnode(node)
                    cmds.delete(rootNodeName)
                    cmds.file(loadReference=refNodeName)
