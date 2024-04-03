import maya.cmds as cmds
import maya.mel as mel
import os
import re
import sys
import json

import Qt
import Qt.QtGui as QtGui

from String import String

currentScriptPath = os.path.abspath(__file__)
srcPath = os.path.dirname(currentScriptPath)

class HDRISet():
    
    def __init__(self):
        self.showDataDic = {}
        self.loadEnvJsonData()
    
    # load env_data from a .json file and store data into show_data_dict
    # load_show_data -> loadEnvJsonData
    def loadEnvJsonData(self):
        jsonPath = os.path.join(srcPath, String.envJsonPath)
        
        with open(jsonPath, 'r') as jsonData:
            self.showDataDic = json.load(jsonData)
            
    def getEnvData(self, showName):
        self.envPath = String.envPath
        
        if not showName in self.showDataDic[String.showInformation]:
            showName = 'default'
        
        self.envMapPath = os.path.join(self.envPath, self.showDataDic[String.showInformation][showName]['env_map'])
        if self.showDataDic[String.showInformation][showName].has_key("direct_env_map"):
            self.envMapPath = self.showDataDic[String.showInformation][showName]["direct_env_map"]
        self.envMapName = os.path.splitext(os.path.basename(self.envMapPath))[0]
    
    def getThumbGUI(self):
        imgPath = os.path.join(self.envPath, '.hdrThumbs/') + self.envMapName + '.jpg'
        envThumbPath = QtGui.QImage(imgPath)
        
        return QtGui.QPixmap(envThumbPath.scaledToWidth(95))
    
    # set hdri map setting to maya
    def setEnvLight(self, showName):
        isPxrHDirLight = cmds.objExists("{0}_pxrHdriLight".format(showName));
        
        if not isPxrHDirLight and cmds.objExists("transform1"):
            mel.eval("select -r transform1;")
            mel.eval("doDelete;")
            
        elif isPxrHDirLight:
            return
        
        self.hdriNode = cmds.shadingNode('PxrDomeLight', asLight = True, name = "{0}_pxrHdriLight".format(showName))
        cmds.setAttr('{0}.lightColorMap'.format(self.hdriNode),
                         self.envMapPath,
                         type = 'string')
        hdriRotation = self.showDataDic[String.showInformation][showName]["rotation"]
        cmds.setAttr('{0}.rotateX'.format(self.hdriNode), hdriRotation[0])
        cmds.setAttr('{0}.rotateY'.format(self.hdriNode), hdriRotation[1])
        cmds.setAttr('{0}.rotateZ'.format(self.hdriNode), hdriRotation[2])