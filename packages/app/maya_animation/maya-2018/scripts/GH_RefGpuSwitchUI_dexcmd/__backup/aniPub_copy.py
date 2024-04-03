# encoding=utf-8
#!/usr/bin/env python

#-------------------------------------------------------------------------------
#
#   RenderMan TD
#
#       Sanghun Kim, rman.td@gmail.com
#
#	for Animation Publish
#
#	2016.01.21 $4
#-------------------------------------------------------------------------------

import os, sys
import time
import string
import subprocess
import json
import shutil

import dexcmd.dexCommon as dexCommon
import dexcmd.batchCommon as batchCommon

import maya.cmds as cmds
import maya.mel as mel


#-------------------------------------------------------------------------------
def offsetKey(objectName, offset):
    objs = cmds.ls(objectName, dag=True, ni=True)
    connections = cmds.listConnections(objs, type='animCurve')
    if connections:
        for i in connections:
            ln = cmds.listConnections(i, p=True)
            src = ln[0].split('.')
            offsetKeyAttr(src[0], src[-1], offset)


def offsetKeyAttr(obj, attr, offset):
    ln = '%s.%s' % (obj, attr)
    frames = cmds.keyframe(ln, q=True, a=True)
    # end frame
    end_value = cmds.getAttr(ln, t=frames[-1])
    tmp_value = cmds.getAttr(ln, t=frames[-1] - offset)
    set_value = end_value - tmp_value
    cmds.setKeyframe(ln, itt='spline', ott='spline', t=frames[-1] + offset,
                     at=attr, v=end_value + set_value)
    # start frame
    start_value = cmds.getAttr(ln, t=frames[0])
    tmp_value = cmds.getAttr(ln, t=frames[0] + offset)
    set_value = tmp_value - start_value
    cmds.setKeyframe(ln, itt='spline', ott='spline', t=frames[0] - offset,
                     at=attr, v=start_value - set_value)


def keyBake(Objects=list(), Start=1, End=1):
    constraintList = cmds.listRelatives(Objects, type='constraint')

    if Start == End:
        if constraintList:
            cmds.delete(constraintList)
        return

    cmds.bakeResults(
        Objects, t=(Start, End),
        #			simulation=True,
        sampleBy=1, dic=True, pok=True, sac=False, ral=False,
        bol=False, mr=True, controlPoints=False, shape=False)

    if constraintList:
        cmds.delete(constraintList)


def getExportObjects():
    result = list()
    rigAttrs = ['charRig', 'propRig', 'etcRig']
    for i in cmds.ls('|*', recursive=True):
        for a in rigAttrs:
            if cmds.attributeQuery(a, n=i, ex=True) and cmds.getAttr('%s.visibility' % i):
                result.append(i)
    return result


# get object by group attribute : renderMesh, simMesh, lowMesh
def getObjectsByAttribute(objName, attrName):
    result = list()
    if cmds.attributeQuery(attrName, n=objName, ex=True):
        ns_name, node_name = dexCommon.getNameSpace(objName)
        for s in cmds.getAttr('%s.%s' % (objName, attrName)):
            check = True
            node = s
            if ns_name:
                node = '%s:%s' % ( ns_name, s )
            if cmds.objExists(node):
                name = getExportObjectName(node)
                if name:
                    result.append(name)
            else:
                name = getExportObjectName(s)
                if name:
                    result.append(name)
    return result


def getExportObjectName(objName):
    nodetypes = cmds.nodeType(objName, i=True)
    if 'transform' in nodetypes:
        return objName
    elif 'shape' in nodetypes:
        trans = cmds.listRelatives(objName, parent=True)
        #		if cmds.ls( objName, io=True ):
        #			target = cmds.ls( trans[0], dag=True, type='surfaceShape', ni=True )
        #			cmds.rename( objName, '%sOrig' % objName )
        #			cmds.rename( target[0], objName )
        return trans[0]


#-------------------------------------------------------------------------------




#-------------------------------------------------------------------------------
# World Locator Export
class WorldLocatorExport:
    def __init__(self, Path=None, Root=list(), Start=None, End=None, Step=1, CleanUp=1):
        # members
        self.worldFiles = list()
        self.worldLocator = list()
        self.Path = Path
        self.Selection = Root
        self.Start = Start
        self.End = End
        self.Step = Step
        self.CleanUp = CleanUp

    def doIt(self):
        if not self.Selection:
            self.Selection = getExportObjects()
        if not self.Selection:
            mel.eval('print "# Error : Not found objects\\n"')
            return

        self.getFrameRange()
        self.createWorld()
        self.abcExport()
        if self.worldLocator and self.CleanUp == 1:
            cmds.delete(self.worldLocator)

    def getFrameRange(self):
        if not self.Start:
            self.Start = int(cmds.playbackOptions(q=True, min=True))
        if not self.End:
            self.End = int(cmds.playbackOptions(q=True, max=True))
        if self.Start != self.End:
            self.Start -= 1
            self.End += 1

    def createWorld(self):
        ifBake = 0
        for i in self.Selection:
            cmds.namespace(set=':')
            if cmds.attributeQuery('charRig', n=i, ex=True):
                ifBake = 1
                src = i.split(':')
                target = 'move_CON'
                if len(src) > 1:
                    if not cmds.namespace(exists=':%s' % src[0]):
                        cmds.namespace(add=src[0])
                    cmds.namespace(set=src[0])
                    target = '%s:move_CON' % src[0]
                # create locator
                loc = cmds.spaceLocator(n='%s_world' % src[-1])
                self.worldLocator += loc
                self.createConstraint(target, loc[0])
        cmds.namespace(set=':')
        # bake
        if ifBake:
            keyBake(self.worldLocator, self.Start, self.End)

    def createConstraint(self, target, source):
        cmds.pointConstraint(target, source)
        cmds.orientConstraint(target, source)

    #cmds.scaleConstraint( target, source )

    def abcExport(self):
        if not self.Path:
            return
        if not os.path.exists(self.Path):
            os.makedirs(self.Path)

        opt = '-uv -wv -wuvs -ef -a ObjectSet -a ObjectName -atp rman -df ogawa'
        opt += ' -fr %s %s -s %s' % ( self.Start, self.End, self.Step )
        command = ''
        for i in self.worldLocator:
            abcfile = os.path.join(self.Path, '%s.abc' % i)
            self.worldFiles.append(abcfile)
            command += ' -j "%s -rt %s -file %s"' % ( opt, i, abcfile )

        mel.eval('AbcExport -v%s' % command)

        # debug
        for f in self.worldFiles:
            mel.eval('print "# Result : AbcExport Path <%s>\\n"' % f)


def import_abcWorldLocator(filename):
    options = ' -d -m import -ftr -crt'
    basename = os.path.basename(filename).split('.abc')[0]
    if cmds.objExists(basename):
        options += ' -ct "%s"' % basename
    else:
        options += ' -ct "/"'
    mel.eval('AbcImport %s "%s"' % (options, filename))
    mel.eval('print "# Result : AbcImport File <%s>\\n"' % filename)


#------------------------------------------------------------------------------
# alembic export
class dxAbcExport:
    def __init__(self, Path=None,
                 Root=list(),
                 Start=None, End=None, Just=False, Step=1,
                 World=True):
        # members
        self.fileLogs = {'render': list(),
                         'sim': list(),
                         'low': list(),
                         'etc': list(),
                         'world': list()}
        self.m_meshTypes = ['render', 'sim', 'low']
        self.worldLocator = list()

        self.Path = Path
        self.Selection = Root
        self.Start = Start
        self.End = End
        self.Just = Just
        self.Step = Step
        self.World = World

        self.renderStart = self.Start
        self.renderEnd = self.End

        self.options = '-uv -wv -wuvs -ef -a ObjectSet -a ObjectName -atp rman -df ogawa'

        if not Path:
            return

        # plugin setup
        if not cmds.pluginInfo('AbcExport', q=True, l=True):
            cmds.loadPlugin('AbcExport')
        cmds.autoKeyframe(state=False)

    def doIt(self):
        startTime = time.time()

        self.getObjectGroups()
        if not self.Selection:
            mel.eval('print "# Error : Not found export objects\\n"')
            return
        mel.eval('print "# Debug : export nodes -> %s\\n"' % string.join(self.Selection, ' '))

        self.getFrameRange()

        if self.World:
            self.createWorldLocator()

        self.abcExport()
        if self.worldLocator:
            cmds.delete(self.worldLocator)

        endTime = time.time()
        mel.eval('print "# Result : %.2f sec\\n"' % (endTime - startTime))

        # debug
        for i in self.fileLogs.keys():
            if self.fileLogs[i]:
                for f in self.fileLogs[i]:
                    mel.eval('print "# Result : AbcExport <%s>\\n"' % f)


    #----------------------------------------
    def getObjectGroups(self):
        if not self.Selection:
            self.Selection = cmds.ls(sl=True)
        rigAttrs = ['charRig', 'propRig', 'etcRig']
        if not self.Selection:
            for i in cmds.ls('|*', recursive=True):
                if cmds.attributeQuery('abcState', n=i, ex=True):
                    # abc imported
                    if cmds.getAttr('%s.abcState' % i):
                        if cmds.nodeType(cmds.ls(i, dag=True, s=True)[0]) == 'locator':
                            self.Selection.append(cmds.listRelatives(i, type='transform')[0])
                        else:
                            self.Selection.append(i)
                    else:
                        if cmds.nodeType(cmds.ls(i, dag=True, s=True)[0]) == 'locator':
                            self.fileLogs['world'].append(cmds.getAttr('%s.abcFile' % i))
                            meshGrp = cmds.listRelatives(i, type='transform')[0]
                            self.fileLogs['render'].append(cmds.getAttr('%s.abcFile' % meshGrp))
                else:
                    # rig animation
                    for a in rigAttrs:
                        if cmds.attributeQuery(a, n=i, ex=True):
                            animConnections = cmds.listConnections(i, p=True, type='animCurve')
                            if animConnections or cmds.getAttr('%s.visibility' % i):
                                self.Selection.append(i)

    # get export objects by group
    def getObjects(self, groupName):
        result = {'render': list(), 'sim': list(), 'low': list(), 'etc': list()}

        # etc mesh
        if cmds.attributeQuery('etcRig', n=groupName, ex=True):
            result['etc'] = cmds.listRelatives(groupName, c=True)

        # render mesh
        if 'render' in self.m_meshTypes:
            result['render'] = getObjectsByAttribute(groupName, 'renderMesh')
            if not result['render']:
                for s in cmds.ls(groupName, dag=True, type='surfaceShape', ni=True):
                    if s.find('dummy') == -1:
                        trans = cmds.listRelatives(s, parent=True, fullPath=True)
                        result['render'].append(trans[0])

        # simulation mesh
        if 'sim' in self.m_meshTypes:
            result['sim'] = getObjectsByAttribute(groupName, 'simMesh')

        # low mesh
        if 'low' in self.m_meshTypes:
            result['low'] = getObjectsByAttribute(groupName, 'lowMesh')

        return result

    # frame range
    def getFrameRange(self):
        if not self.Start:
            # min
            self.Start = int(cmds.playbackOptions(q=True, min=True))
        if not self.End:
            # max
            self.End = int(cmds.playbackOptions(q=True, max=True))
        self.renderStart = self.Start
        self.renderEnd = self.End
        if self.Start != self.End:
            self.Start -= 1
            self.End += 1
            if not self.Just:
                self.findBindPosKey()

    def findBindPosKey(self):
        for i in self.Selection:
            if cmds.attributeQuery('charRig', n=i, ex=True) and cmds.attributeQuery('allControls', n=i, ex=True):
                controls = cmds.getAttr('%s.allControls' % i)
                if 'root_CON' in controls:
                    node = 'root_CON'
                    ns_name, node_name = dexCommon.getNameSpace(i)
                    if ns_name:
                        node = '%s:root_CON' % ns_name
                    keyattrs = cmds.listAttr(node, k=True)
                    for a in keyattrs:
                        keys = cmds.keyframe('%s.%s' % (node, a), q=True)
                        if keys:
                            if int(keys[0]) < self.Start:
                                self.Start = int(keys[0])


    #----------------------------------------
    # world locator - for charRig
    def createWorldLocator(self):
        ifBake = 0
        for i in self.Selection:
            cmds.namespace(set=':')
            if cmds.attributeQuery('charRig', n=i, ex=True):
                ifBake = 1
                target = 'move_CON'
                ns_name, node_name = dexCommon.getNameSpace(i)
                if ns_name:
                    if not cmds.namespace(exists=':%s' % ns_name):
                        cmds.namespace(add=ns_name)
                    cmds.namespace(set=ns_name)
                    target = '%s:move_CON' % ns_name
                # create locator
                loc = cmds.spaceLocator(n='%s_world' % node_name)
                self.worldLocator += loc
                self.createConstraint(target, loc[0])
                # visibility key copy
                self.keyCopy(i, loc[0])
        cmds.namespace(set=':')
        # bake
        if ifBake:
            keyBake(self.worldLocator, self.Start, self.End)

    def createConstraint(self, target, source):
        cmds.pointConstraint(target, source)
        cmds.orientConstraint(target, source)

    # visibility key copy
    def keyCopy(self, source, target):
        aniplug = cmds.listConnections('%s.visibility' % source, p=True, type='animCurve')
        if aniplug:
            cmds.copyKey(source, attribute='visibility')
            cmds.pasteKey(target, attribute='visibility')


    #----------------------------------------
    # abcExport Job command
    def jobCommand(self, groupName):
        job = ''
        meshes = self.getObjects(groupName)
        for t in meshes.keys():
            objs = meshes[t]
            if objs:
                # low mesh visibility control
                tmp_opt = self.options
                if t == 'low':
                    tmp_opt = self.options.replace('-wv', '')

                job += ' -j "%s' % tmp_opt
                for i in objs:
                    job += ' -rt %s' % i
                if t == 'low' or t == 'sim':
                    abcfile = os.path.join(self.Path, '%s_%s.abc' % (groupName, t))
                else:
                    abcfile = os.path.join(self.Path, '%s.abc' % groupName)
                self.fileLogs[t].append(abcfile)
                job += ' -file %s"' % abcfile
        return job

    # abcExport
    def abcExport(self):
        if not os.path.exists(self.Path):
            os.makedirs(self.Path)

        self.options += ' -fr %s %s' % ( self.Start, self.End )
        if self.Start != self.End:
            self.options += ' -s %s' % self.Step

        command = ''

        if self.World:
            self.getConValue()

        # abc imported world locator
        abcimportedLocators = list()

        # mesh
        for i in self.Selection:
            if cmds.attributeQuery('abcState', n=i, ex=True):
                parentNode = cmds.listRelatives(i, p=True)
                if parentNode:
                    if cmds.nodeType(cmds.ls(parentNode[0], dag=True, s=True)[0]) == 'locator':
                        abcimportedLocators.append(parentNode[0])
            job = self.jobCommand(i)
            if job:
                command += job

        # world
        for i in self.worldLocator:
            abcfile = os.path.join(self.Path, '%s.abc' % i)
            self.fileLogs['world'].append(abcfile)
            command += ' -j "%s -rt %s -file %s"' % ( self.options, i, abcfile )

        mel.eval('AbcExport -v%s' % command)

        if self.World:
            self.setConValue()

        if abcimportedLocators:
            for i in abcimportedLocators:
                if cmds.attributeQuery('abcFile', n=i, ex=True):
                    origfile = cmds.getAttr('%s.abcFile' % i)
                    copyfile = os.path.join(self.Path, os.path.basename(origfile))
                    self.fileLogs['world'].append(copyfile)
                    shutil.copy(origfile, self.Path)


    #----------------------------------------
    # get con value & set mute - for charRig
    def getConValue(self):
        self.ConValue = dict()
        conList = ['move_CON', 'direction_CON', 'place_CON']
        # get value
        for c in conList:
            for i in self.Selection:
                if cmds.attributeQuery('charRig', n=i, ex=True):
                    con = c
                    ns_name, node_name = dexCommon.getNameSpace(i)
                    if ns_name:
                        con = '%s:%s' % ( ns_name, c )
                    self.ConValue[con] = dict()
                    self.ConValue[con]['t'] = cmds.getAttr('%s.t' % con)[0]
                    self.ConValue[con]['r'] = cmds.getAttr('%s.r' % con)[0]
        # set mute
        for c in self.ConValue:
            for i in self.ConValue[c]:
                cmds.setAttr('%s.%s' % (c, i), 0, 0, 0)
                cmds.mute('%s.%s' % (c, i), d=False, f=True)

    # set con value & set un-mute
    def setConValue(self):
        for c in self.ConValue:
            for i in self.ConValue[c]:
                gv = self.ConValue[c][i]
                cmds.setAttr('%s.%s' % (c, i), gv[0], gv[1], gv[2])
                cmds.mute('%s.%s' % (c, i), d=True, f=True)


#------------------------------------------------------------------------------
# find gpu node
def getGpuNode(gpuNode):
    node = gpuNode
    try:
        cmds.unloadPlugin('gpuCache')
    except:
        node = 'gpuCache'
    try:
        cmds.unloadPlugin('ZAlembicCache')
    except:
        node = 'ZAlembicCache'
    cmds.loadPlugin(node)
    return node


#------------------------------------------------------------------------------
# import alembic
#	Mode : meshmode, gpumode
#	GpuNodeType : gpuCache, ZAlembicCache
#	MeshType : Render, Simulation, Low, All
#
# description
#	add attributes : abcImport, renderMesh
#	add attributes : abcFile, abcState, renderMesh
#
class dxAbcImport:
    def __init__(self, Path=None,
                 Mode='meshmode',
                 GpuNodeType='gpuCache', MeshType='Render',
                 World=True,
                 Root=list(),
                 FitTime=True,
                 DrawingBBox=False):
        if not Path:
            return
        # plugin setup
        if not cmds.pluginInfo('AbcImport', q=True, l=True):
            cmds.loadPlugin('AbcImport')

        self.Path = Path
        self.Mode = Mode
        self.GpuNodeType = GpuNodeType
        self.MeshType = MeshType
        self.World = World
        self.Root = Root
        self.FitTime = FitTime
        self.DrawingBBox = DrawingBBox

        self.importMeshFiles = list()
        self.importWorldFiles = list()

    def doIt(self):
        startTime = time.time()

        self.getFiles()
        if self.Mode == 'meshmode':
            self.import_mesh()
        elif self.Mode == 'gpumode':
            self.import_gpu()
        if self.World:
            self.import_world()

        cmds.select(cl=True)

        if self.importMeshFiles and self.FitTime:
            logfile = batchCommon.anipubLogFile(self.importMeshFiles[0])
            if logfile and os.path.exists(logfile):
                batchCommon.anipubLogRead(logfile)

        endTime = time.time()
        mel.eval('print "# Result : %.2f sec\\n"' % (endTime - startTime))

    def getFiles(self):
        source = list()
        source_files = list()
        if type(self.Path).__name__ == 'list':
            for i in self.Path:
                if i.find('.abc') > -1:
                    source_files.append(i)
        else:
            for i in os.listdir(self.Path):
                if i.find('.abc') > -1:
                    source_files.append(os.path.join(self.Path, i))

        for i in source_files:
            name = i.replace('_sim.abc', '')
            name = name.replace('_low.abc', '')
            name = name.replace('_world.abc', '')
            name = name.replace('.abc', '')
            source.append(name)

        for i in list(set(source)):
            if self.MeshType == 'Render' or self.MeshType == 'All':
                if os.path.exists(i + '.abc'):
                    self.importMeshFiles.append(i + '.abc')
            if self.MeshType == 'Simulation' or self.MeshType == 'All':
                if os.path.exists(i + '_sim.abc'):
                    self.importMeshFiles.append(i + '_sim.abc')
            if self.MeshType == 'Low' or self.MeshType == 'All':
                if os.path.exists(i + '_low.abc'):
                    self.importMeshFiles.append(i + '_low.abc')
            if self.World:
                if os.path.exists(i + '_world.abc'):
                    self.importWorldFiles.append(i + '_world.abc')

    def addAbcImportedAttribute(self, objName, fileName):
        if not cmds.attributeQuery('abcFile', n=objName, ex=True):
            cmds.addAttr(objName, ln='abcFile', dt='string')
        cmds.setAttr('%s.abcFile' % objName, lock=False)
        cmds.setAttr('%s.abcFile' % objName, fileName, type='string', lock=True)
        if not cmds.attributeQuery('abcState', n=objName, ex=True):
            cmds.addAttr(objName, ln='abcState', at='enum', en='import:export')
        cmds.setAttr('%s.abcState' % objName, 0)

    def getNodename(self, name):
        result = name
        if cmds.objExists(name):
            result = name
        else:
            tmp = name.split(':')[-1]
            if cmds.objExists(tmp):
                result = tmp
        return result

    def import_mesh(self):
        createdGroups = dict()
        for i in self.importMeshFiles:
            options = '-d -m import'
            if self.FitTime:
                options += ' -ftr'
            cmds.namespace(set=':')

            ns_name = None;
            node_name = None
            basename = os.path.splitext(os.path.basename(i))[0]
            basename = basename.replace('_sim', '')
            basename = basename.replace('_low', '')
            src = basename.split(':')
            if cmds.objExists(src[-1]):
                options += ' -ct "%s"' % src[-1]
            else:
                ns_name, node_name = dexCommon.getNameSpace(basename)
                if ns_name:
                    if not cmds.namespace(exists=':%s' % ns_name):
                        cmds.namespace(add=ns_name)
                    cmds.namespace(set=ns_name)
                if cmds.objExists(basename):
                    options += ' -crt -ct "%s"' % basename
                else:
                    name = src[-1]
                    if self.Root:
                        name = self.Root[self.importMeshFiles.index(i)]
                    root = cmds.group(n=name, em=True)
                    createdGroups[root] = i
                    options += ' -rpr "%s"' % root

            mel.eval('AbcImport %s "%s"' % (options, i))

            # debug
            mel.eval('print "# Result : AbcImport <%s>\\n"' % i)

        # re-set namespace
        cmds.namespace(set=':')

        # set attributes
        for i in createdGroups.keys():
            self.addAbcImportedAttribute(i, createdGroups[i])
            shapes = cmds.ls(i, dag=True, type='surfaceShape', ni=True)
            cmds.addAttr(i, ln='renderMesh', dt='stringArray', h=True)
            cmds.setAttr('%s.renderMesh' % i, *( [len(shapes)] + shapes ), type='stringArray')

    def import_gpu(self):
        gpuNode = getGpuNode(self.GpuNodeType)

        for i in self.importMeshFiles:
            basename = os.path.splitext(os.path.basename(i))[0]
            basename = basename.replace('_sim', '')
            basename = basename.replace('_low', '')
            if cmds.objExists(basename) and cmds.nodeType(basename) == gpuNode:
                self.gpuShape = cmds.ls(basename, dag=True, type=gpuNode)[0]
                cmds.setAttr('%s.cacheFileName' % self.gpuShape, i, type='string')
                # debug
                mel.eval('print "# Result : %s import set file <%s>\\n"' % (gpuNode, i))
            else:
                self.gpuShape = cmds.createNode(gpuNode, n='%sShape' % basename)
                cmds.setAttr('%s.visibleInReflections' % self.gpuShape, 1)
                cmds.setAttr('%s.visibleInRefractions' % self.gpuShape, 1)
                if self.DrawingBBox:
                    cmds.setAttr('%s.overrideEnabled' % self.gpuShape, 1)
                    cmds.setAttr('%s.overrideLevelOfDetail' % self.gpuShape, 1)
                cmds.setAttr('%s.cacheFileName' % self.gpuShape, i, type='string')

                self.addGpuRenderAttributes(self.gpuShape)
                # debug
                mel.eval('print "# Result : %s import <%s>\\n"' % (gpuNode, i))

        # fit time range
        if self.FitTime:
            timeRange = eval('cmds.%s( "%s", q=True, animTimeRange=True )' % (gpuNode, self.gpuShape))
            cmds.playbackOptions(minTime=timeRange[0])
            cmds.playbackOptions(maxTime=timeRange[1])
            cmds.playbackOptions(animationStartTime=timeRange[0])
            cmds.playbackOptions(animationEndTime=timeRange[1])

    def addGpuRenderAttributes(self, shape):
        if not cmds.attributeQuery('rman__torattr___preShapeScript', n=shape, ex=True):
            cmds.addAttr(shape, ln='rman__torattr___preShapeScript', dt='string')
        cmds.setAttr('%s.rman__torattr___preShapeScript' % shape, 'dxarc', type='string')
        if not cmds.attributeQuery('dt', n=shape, ex=True):
            cmds.addAttr(shape, ln='dt', nn='Velocity Delta Time', at='double')
        if not cmds.attributeQuery('subdiv', n=shape, ex=True):
            cmds.addAttr(shape, ln='subdiv', nn='Attribute Subdiv', dv=1, at='bool')
        if not cmds.attributeQuery('objectid', n=shape, ex=True):
            cmds.addAttr(shape, ln='objectid', nn='Object ID', dv=1, at='bool')
        if not cmds.attributeQuery('groupid', n=shape, ex=True):
            cmds.addAttr(shape, ln='groupid', nn='Group ID', dv=0, at='bool')
        # renderfile set
        cfn = cmds.getAttr('%s.cacheFileName' % shape)
        if cfn.find('_low') > -1:
            rfn = cfn.replace('_low', '')
            if os.path.exists(rfn):
                cmds.addAttr(shape, ln='renderFile', dt='string')
                cmds.setAttr('%s.renderFile' % shape, rfn, type='string')

    def import_world(self):
        for i in self.importWorldFiles:
            options = '-d -m import -crt'
            basename = os.path.splitext(os.path.basename(i))[0]
            baseNode = self.getNodename(basename)
            if cmds.objExists(baseNode):
                options += ' -ct "%s"' % baseNode
            else:
                options += ' -ct "/"'

            mel.eval('AbcImport %s "%s"' % (options, i))

            baseNode = self.getNodename(basename)
            # set attribute
            if cmds.objExists(baseNode):
                self.addAbcImportedAttribute(baseNode, i)

            targetNode = self.getNodename(basename.split('_world')[0])
            if cmds.objExists(targetNode):
                children = cmds.listRelatives(baseNode, type='transform')
                if children:
                    if not targetNode in children:
                        self.setParent(targetNode, baseNode)
                else:
                    self.setParent(targetNode, baseNode)

            # debug
            mel.eval('print "# Result : AbcImport <%s>\\n"' % i)

    def setParent(self, target, source):
        cmds.parent(target, source)
        cmds.setAttr('%s.t' % target, 0, 0, 0)
        cmds.setAttr('%s.r' % target, 0, 0, 0)
        cmds.setAttr('%s.s' % target, 1, 1, 1)



#------------------------------------------------------------------------------
def abcMeshOrder(filename):
    command = '/netapp/backstage/pub/lib/extern/bin/abcecho %s' % filename
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         shell=True)
    result, errors = p.communicate()
    if not result:
        print errors
        return

    objects = list()
    lines = result.split('\n')
    for i in lines:
        if i.find('Object name') > -1:
            name = i.split('=')[-1].split('/')[-1]
            objects.append(name)
    shapes = list()
    for i in range(len(objects) / 2):
        shapes.append(objects[i * 2 + 1])
    return shapes


def alembicCoreImport(filename):
    basename = os.path.basename(filename).split('.abc')[0]
    abcNode = cmds.createNode('AlembicNode', name='%s_AlembicNode' % basename)
    cmds.setAttr('%s.abc_File' % abcNode, filename, type='string')
    timeNode = cmds.ls(type='time')[0]
    cmds.connectAttr('%s.outTime' % timeNode, '%s.time' % abcNode, f=True)

    abcObjects = abcMeshOrder(filename)

    for i in cmds.ls(type='surfaceShape', ni=True):
        if i in abcObjects:
            index = abcObjects.index(i)
            cmds.connectAttr('%s.outPolyMesh[%s]' % (abcNode, index), '%s.inMesh' % i, f=True)

    startFrame = cmds.getAttr('%s.startFrame' % abcNode)
    endFrame = cmds.getAttr('%s.endFrame' % abcNode)
    #print startFrame, endFrame
    mel.eval('print "# Result : AbcCoreImport <%s>\\n"' % filename)

