# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
#
#   Dexter Lighting TD
#
#		joonkyun.oh		jacade@naver.com
#
#   Bora Ocean Preset Process In Inventory
#
# -------------------------------------------------------------------------------

import os
import maya.cmds as cmds
import rfmShading
import maya.mel as mel


class importMayaPreset():
    def __init__(self, item):
        self.itemData = item.getItemData()
        self.fractalMin = {'force_1':0, 'force_2':0.1, 'force_3':0.2, 'force_4':0.6, 'force_5':0.8}
        self.sceneFile =  self.itemData['files']['sceneFile']
        self.xmlFile = self.itemData['files']['xmlFile']
        self.paramFile = self.itemData['files']['paramFile']
        self.simulfile =  self.itemData['files']['simulFile']
        self.force = self.itemData['name']

    def importScene(self):
        # print self.sceneFile
        # print '{0}/{1}/{1}.oceanParams'.format(self.paramPath, self.force)
        # print 'ocean_Fractal_Remap.outputMin', self.fractalMin[self.force]
        cmds.file(self.sceneFile, i=True)
        if self.itemData['category'] == 'MayaRenderman':
            rfmShading.importRlf(self.xmlFile, 'Add')
            cmds.setAttr('PresetPxrBoraOcean.inputFile', self.paramFile, type='string')
            cmds.setAttr('ocean_Fractal_Remap.outputMin', self.fractalMin[self.force])
            print 'Bora ocean preset {} impoted'.format(self.force)

    def importSimul(self):
        # print self.simulfile, os.path.split(self.paramFile)[0], self.force
        cmds.file(self.simulfile, i=True)
        setParamCmd = 'BoraOceanCmd '
        setParamCmd += '-nodeName "{}" '.format('PreviewBoraOceanShape')
        setParamCmd += '-toolName "import" '
        setParamCmd += '-filePath "{}" '.format(os.path.split(self.paramFile)[0])
        setParamCmd += '-fileName "{}"'.format(self.force)
        mel.eval(setParamCmd)