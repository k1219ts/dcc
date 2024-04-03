# coding:utf-8
from __future__ import print_function
import pprint

from pymel.all import *
import maya.standalone
maya.standalone.initialize("Python")

plugins = ['backstageMenu', 'pxrUsd', 'DXUSD_Maya']
for p in plugins:
    if not cmds.pluginInfo(p, q=True, l=True):
        cmds.loadPlugin(p)

import os
import maya.cmds as cmds
import argparse

from DXUSD.Structures import Arguments
import DXUSD.Vars as var
import DXUSD_MAYA.Model as Model
import DXUSD.Utils as utl
from pxr import Usd, Sdf

import utils as cutl
import getpass


class ModelExport():
    def __init__(self, arg):
        self.modeldir = arg.orgModelDir
        self.newShow = arg.newShow
        self.newAsset = arg.newAsset
        self.newBranch = arg.newBranch
        self.hairPath = arg.hairPath

        if arg.versionExp == 'True':
            self.versionExp = True
        else:
            self.versionExp = False

        self.basepath = utl.GetBasePath(self.newAsset, self.newBranch)
        self.abname = self.newAsset
        if self.newBranch is 'None':
            self.abname = '%s_%s' %(self.newAsset, self.newBranch)

        self.mayafile = cutl.getMayaPath(self.abname)


    def doit(self):
        cmds.file(rename=self.mayafile)

        if self.versionExp == False:
            nodes = cutl.importData(self.modeldir)
            nodes = cutl.rename(nodes, self.basepath, self.abname)
            print('nodes Export:', nodes)
            Model.assetExport(nodes= nodes, show= self.newShow, version='v001')
            cmds.file(save=True, type="mayaBinary")

        else:
            arg = Arguments()
            arg.D.SetDecode(self.modeldir)
            arg.task = var.T.MODEL
            arg.taskproduct = "TASKV"
            taskdir = arg.D.TASK
            stage = Usd.Stage.Open(os.path.join(arg.D.TASK, arg.F.TASK))
            dPrim = stage.GetDefaultPrim()
            versions = dPrim.GetVariantSet("modelVer").GetVariantNames()
            for ver in versions:
                print('model version:', ver)
                modeldir = os.path.join(taskdir,ver)
                nodes = cutl.importData(modeldir)
                nodes = cutl.rename(nodes, self.basepath, self.abname)
                cmds.file(save=True, type="mayaBinary")
                print('nodes Export:', nodes)
                Model.assetExport(nodes= nodes, show= self.newShow, version=ver)
                cmds.delete(nodes)

        if os.path.exists(self.mayafile):
            os.remove(self.mayafile)

        if not self.hairPath == 'None':
            if self.versionExp == True or self.versionExp == False:
                cutl.groomExport(self.hairPath, self.newShow, self.abname, self.basepath)

parser = argparse.ArgumentParser()
parser.add_argument('--orgModelDir', dest='orgModelDir', default='', help='model directory')
parser.add_argument('--newShow', dest='newShow', default='', help='show')
parser.add_argument('--newAsset', dest='newAsset', default='', help='asset')
parser.add_argument('--newBranch', dest='newBranch', default='', help='branch')
parser.add_argument('--hairPath', dest='hairPath', default='', help='maya hair scene path ')
parser.add_argument('--versionExp', dest='versionExp', default='', help='version Export ')
arg = parser.parse_args()
ModelExport(arg).doit()
# ModelExport(arg)


cmds.quit()
cmds.quit(force=True)

    

    










