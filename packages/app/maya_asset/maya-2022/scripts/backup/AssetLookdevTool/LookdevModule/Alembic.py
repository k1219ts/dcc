import glob # return file & directory list
import json
import os
import sys
import site

import maya.cmds as cmds
import maya.mel as mel
from String import String
from MessageBox import MessageBox

MAYA_VERSION = cmds.about(version = True)

import sgUI

class Alembic():
    
    def __init__(self):
        self.alembicList = []
        self.alembicPath = ""
        self.alembicCurrentVersion = ""
        self.alembicCurrentVersionPath = ""
        self.msgDismiss = False
        self.alembicFileName = ""
        self.isLoadingAlembic = False
        self.alembicPubData = None

    # list_abc -> loadAlembicVersionList
    def loadAlembicVersionList(self, alembicPath):
        alembicListBeforeSort = []

        for alembicFile in glob.glob('{0}/*.abc'.format(alembicPath)):
            fileName = os.path.splitext(os.path.basename(alembicFile))[0]

            if "low" in fileName.split('_')[-2]:
                continue
            
            alembicListBeforeSort.append(alembicFile)
            
            alembicListBeforeSort.sort( key = os.path.getmtime )
            alembicListBeforeSort.reverse()

        alembicList = []
            
        for alembicFileSorted in alembicListBeforeSort:
            alembicFileName = os.path.basename(alembicFileSorted)
            alembicName = os.path.splitext(alembicFileName)[0]
            alembicList.append(alembicName)

        self.alembicList = alembicList
        
        self.setAlembicPath(alembicPath)
            
    # define_current_version -> setAlembicPubData 
    def loadAlembicJsonData(self):
        try:
            alembicJsonPath = self.getCurrentVersionPath() + '.json'
            with open(alembicJsonPath, 'r') as alembicJsonData:
                alembicPubData = json.load(alembicJsonData)
            
            self.alembicPubData = alembicPubData
        except IOError as e:
            MessageBox( Message='Json file does not exist',
                              Button = ["OK"])
        
    def getAlembicPubData(self):
        return self.alembicPubData
        
    def setCurrentVersionPath(self):
        self.alembicCurrentVersionPath = os.path.join(self.getAlembicPath(), self.getCurrentVersion())
        
    def getCurrentVersionPath(self):
        return self.alembicCurrentVersionPath
    
    def setAlembicPath(self, Path):
        self.alembicPath = Path
        
    def getAlembicPath(self):
        return self.alembicPath
    
    def setCurrentVersion(self, Version):
        self.alembicCurrentVersion = Version
        
    def getCurrentVersion(self):
        return self.alembicCurrentVersion
    
    # if 'AbcImport' don't has name to plugin... cmds.file('AbcImport Path') import plugin 
    def alembicImport(self):
        currentVersionPath = self.getCurrentVersionPath()
        
        alembicImportPath = currentVersionPath + '.abc'
        
        msg = MessageBox( winTitle='Alembic Import',
                          Message='How should import?',
                          Icon='question',
                          Button = ['GPU', 'Alembic'] )
        
        if msg == 'dismiss':
            self.msgDismiss = True
        else:
            self.msgDismiss = False

        ciClass = sgUI.ComponentImport( Files=[unicode(alembicImportPath)], World = 1 ) # 1 : Baked
        ciClass.m_display = 1 # 1 : "Render"
        ciClass.m_fitTime = True

        if msg == 'GPU':
            ciClass.m_mode = 1 # 1 : "gpumode"
        elif msg == 'Alembic':
            ciClass.m_mode = 0 # 0 : "meshmode"

        self.alembicFileName = ciClass.doIt()[0]

        self.isLoadingAlembic = True
#         cmds.file(alembicImportPath, i = True, mergeNamespacesOnClash = True, namespace = ':')
        
    def isAlembicFile(self, alembicName):
        return cmds.objExists(alembicName)
    
    def deleteAlembic(self, alembicFileName):
        cmds.select(alembicFileName)
        mel.eval("doDelete;")