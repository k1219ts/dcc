#coding:utf-8
from __future__ import print_function
import os
import re
from pxr import Sdf, Gf

import maya.cmds as cmds

from DXUSD.Structures import Arguments
import DXUSD.Vars as var

import DXUSD_MAYA.Message as msg
import DXUSD_MAYA.Exporters as exp
import DXUSD_MAYA.MUtils as mutl
import DXUSD_MAYA.Utils as utl


def GetSimNodes(selected='|*'):
    '''
    Return
        (list) [(node, ref-node, namespace), (...)]
    '''
    nodes = cmds.ls(selected.split(':')[-1], type='dxBlock', l=True, r=True)
    if not nodes:
        msg.Warning("not found 'dxBlock' for simulation")
        return

    exportNodes = list()
    for n in nodes:
        if cmds.getAttr('%s.action' % n) == 1 and cmds.getAttr('%s.type' % n) == 3 and mutl.GetViz(n):
            refnode = n
            refconnected = cmds.listConnections('%s.referencedXBlock' % n, s=True, d=False)
            if refconnected:
                refnode = refconnected[0]
            nsLayer = cmds.getAttr('%s.nsLayer' % refnode)
            if nsLayer:
                exportNodes.append((n, refnode, nsLayer))
            else:
                msg.Warning("[Sim.GetSimNodes] : nsLayer setup error - '%s'" % n)
    if not exportNodes:
        msg.Warning("not found export simulation 'xBlock'")
        return
    return exportNodes


class SimGeomExport(exp.SimBase):
    def preProcess(self):
        if not cmds.objExists(self.arg.node):
            msg.errorQuit('not found rig node [%s]' % self.arg.node)

        if not self.arg.scene:
            msg.errorQuit('Must have to save current scene')

        if self.arg.autofr:
            self.arg.frameRange = mutl.GetFrameRange()

        self.exportFrameRange = self.arg.frameRange
        if self.arg.frameRange[0] != self.arg.frameRange[1]:
            if self.arg.autofr:
                ast = int(cmds.playbackOptions(q=True, ast=True))
                if self.arg.frameRange[0] - ast == 51:
                    self.exportFrameRange = (ast, self.arg.frameRange[1] + 1)
                else:
                    self.exportFrameRange = (self.arg.frameRange[0] - 1, self.arg.frameRange[1] + 1)
            else:
                self.exportFrameRange = (self.arg.frameRange[0] - 1, self.arg.frameRange[1] + 1)

        dxBlockData = GetSimNodes(self.arg.node)
        if dxBlockData:
            self.node, self.refNode, self.nsLayer = dxBlockData[0]

        self.arg.refNode = self.refNode.split('|')[-1]

        self.mergeCache = cmds.getAttr('%s.mergeFile' % self.refNode)
        self.inputRigFile = cmds.getAttr('%s.importFile' % self.refNode)

        self.customLayerData = {
            'sceneFile': self.arg.scene,
            'start': self.arg.frameRange[0],
            'end': self.arg.frameRange[1],
            'step': self.arg.step,
            'rigFile': self.inputRigFile,
            'inputCache': self.mergeCache,
        }

    def computeSeparateFrames(self, geomFile, objects, lod):
        '''

        :param geomFile: rig high Usd File
        :param objects: rig Shape Nodes
        :return: (list) [(1001, 1050), (1051, 1100), ...]
        '''

        arg = Arguments()
        arg.D.SetDecode(os.path.dirname(geomFile))
        arg.desc = '_'.join(arg.nslyr.split('_')[:-1]) + '_GRP'
        arg.lod = lod
        refFile = os.path.join(arg.D.TASKN, arg.F.GEOM)
        refSize = os.path.getsize(refFile)
        refSize = refSize / (1000.0 * 1000.0 * 1000.0) # GB

        numframes = self.arg.frameRange[1] - self.arg.frameRange[0] + 1
        # deformed = cmds.ls(objects, dag=True, s=True, io=True)
        # totalSize = refSize * (float(len(deformed)) / float(len(objects))) * numframes

        totalSize = refSize * numframes
        limitSize = 20.0 # GB

        msg.debug('refFile :', refFile)
        msg.debug("refSize :", refSize)
        msg.debug("totalSize :", totalSize)
        msg.debug("limitSize :", limitSize)

        if totalSize > limitSize:
            framesPerFile = int(numframes / (totalSize / limitSize))
            frames = list()
            for frame in range(self.arg.frameRange[0], self.arg.frameRange[1] + 1, framesPerFile):
                endFrame = frame + (framesPerFile - 1)
                if endFrame >= self.arg.frameRange[1]:
                    endFrame = self.arg.frameRange[1]
                frames.append((frame, endFrame))

            msg.message('[Rig.computeSeparateFrames] estimate total size : %.2fGB, per file size : %.2fGB, frames: %s' % (totalSize, refSize * framesPerFile, frames))
            msg.debug(frames)
            return frames

    def separate_geomExport(self, filename, objects, frames, ovr_opts):
        for start, end in frames:
            fn = filename.replace('.usd', '.%04d.usd' % start)
            msg.debug('> GEOM FILE\t:', fn)
            utl.UsdExport(fn, objects, fr=(start, end), fs=self.arg.frameSample, **ovr_opts).doIt()
            self.arg.separate_geomfiles.append(fn)

            # try:
            #     abcfile = fn.replace('.usd', '.abc')
            #     mutl.AbcExport(abcfile, objects, fr=(start, end), step=self.arg.step)
            #     self.arg.abcFiles.append(abcfile)
            # except:
            #     pass

    def Exporting(self):
        # override
        print('#### SimExport ####')
        print('> node\t\t:', self.arg.node)

        self.arg.frameSample = mutl.GetFrameSample(self.arg.step)

        # frameRange setup & dxBlock Node Setup
        self.preProcess()

        ovr_opts = {'exportUVs': False}

        geomFile = utl.SJoin(self.arg.D.TASKNV, self.arg.F.GEOM)
        frames = self.computeSeparateFrames(self.inputRigFile, [self.arg.node], var.T.HIGH)
        if frames:
            self.separate_geomExport(geomFile, [self.arg.node], frames, ovr_opts)
        else:
            msg.debug('> GEOM FILE\t:', geomFile)
            utl.UsdExport(geomFile, [self.arg.node], fr=self.exportFrameRange, fs=self.arg.frameSample,
                          customLayerData=self.customLayerData, **ovr_opts).doIt()
        self.arg.geomfiles.append(geomFile)

    def Compositing(self):
        return var.SUCCESS


class SimCompositor(exp.SimBase):
    def Exporting(self):
        return var.SUCCESS

    def Tweaking(self):
        return var.SUCCESS


def shotExport(node=None, overwrite=False, show=None, seq=None, shot=None, version=None, user='anonymous',
               fr=[0, 0], step=1.0, process='geom'):
    if not node:
        return
    # current scene filename
    sceneFile = cmds.file(q=True, sn=True)

    arg = exp.ASimExporter()
    arg.scene = sceneFile
    arg.node  = node
    arg.nslyr = cmds.getAttr('%s.nsLayer' % node)
    arg.frameRange = mutl.GetFrameRange()
    arg.autofr = True
    arg.overwrite = overwrite
    arg.step = step

    # override
    if show: arg.ovr_show = show
    if seq:  arg.ovr_seq  = seq
    if shot: arg.ovr_shot = shot
    if version: arg.nsver = version
    if fr != [0, 0]:
        arg.frameRange = fr
        arg.autofr = False

    if process == 'both':
        SimGeomExport(arg)
        arg.overwrite = True
        exporter = SimCompositor(arg)
    else:
        if process == 'geom':
            exporter = SimGeomExport(arg)
        else:
            arg.overwrite = True
            exporter = SimCompositor(arg)
    return exporter.arg.master
