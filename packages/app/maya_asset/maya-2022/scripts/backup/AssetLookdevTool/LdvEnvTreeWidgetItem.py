#coding=cp949
import Qt.QtCore as QtCore
import Qt.QtGui as QtGui
import Qt.QtWidgets as QtWidgets

from dxname import rulebook

import maya.cmds as cmds
import maya.mel as mel

import os

from LookdevModule.Alembic import Alembic
from LookdevModule.TextureCtrl import TextureCtrl
from LookdevModule.ShaderMng import ShaderMng
from LookdevModule.MessageBox import MessageBox

# define column num
ASSET_NAME = 1
ALEMBIC_VERSION = 2
TEXTURE_VERSION = 3
SHADER_TYPE = 4

class LdvEnvTreeWidgetItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent, filePath, attrIndex = -1):
        super(self.__class__, self).__init__(parent)

        # name convention rule book
        self.rulebook = rulebook.Coder()
        self.rulebook.load_rulebook("/netapp/backstage/pub/lib/python_lib/dxname/name_for_asset.yaml")

        print 'filePath :', filePath

        decodingRule = self.rulebook.decode(filePath, "env_data_path")

        self.rulebook.flag["ASSETNAME"] = decodingRule["ASSETNAME"]
        self.rulebook.flag["TYPE"] = decodingRule["TYPE"]
        self.rulebook.flag["PROJECT"] = decodingRule["PROJECT"]
        self.rulebook.flag["ENVSOURCE"] = decodingRule["ENVSOURCE"]
        self.rulebook.flag["DATANAME"] = decodingRule["DATANAME"]

        self.assetName = str(self.rulebook.flag["ASSETNAME"])
        self.assetType = str(self.rulebook.flag["TYPE"])
        self.showName = str(self.rulebook.flag["PROJECT"])
        self.envAssetPath = str(self.rulebook.product["env_data_path"])
        self.assetPath = str(self.rulebook.product["asset_path"])
        self.sourceName = str(self.rulebook.flag["ENVSOURCE"])
        self.dataName = str(self.rulebook.flag["DATANAME"])

        self.alembicFile = None
        self.shaderInfo = None
        self.textureInfo = None
        
        # isLoaded?
        self.isLoadFailed = False
        
        # Column Setting
        
        # 0 == ASSET_NAME
        self.assetLabel = QtWidgets.QLabel()
        self.assetLabel.setText(self.dataName)
        parent.setItemWidget(self, ASSET_NAME, self.assetLabel)
        
        # 1 == ALEMBIC_VERSION
        self.alembicVersionComboBox = QtWidgets.QComboBox()
        parent.setItemWidget(self, ALEMBIC_VERSION, self.alembicVersionComboBox)
        
        # 2 == TEXTURE_VERSION
        self.textureVersionComboBox = QtWidgets.QComboBox()
        parent.setItemWidget(self, TEXTURE_VERSION, self.textureVersionComboBox)
        
        # 3 == SHADER_TYPE
        self.shaderVersionComboBox = QtWidgets.QComboBox()
        parent.setItemWidget(self, SHADER_TYPE, self.shaderVersionComboBox)

        self.ldvNode = cmds.ls(type='dxLdvNode')[0]

        if attrIndex == -1:
            size = cmds.getAttr('%s.ldvEnvPath' % self.ldvNode, size=True)
            self.attrIndex = size
            print '%s.ldvEnvPath[%d]' % (self.ldvNode, self.attrIndex)
            print 'size :', size
            cmds.setAttr('%s.ldvEnvPath[%d]' % (self.ldvNode, self.attrIndex), filePath, type='string')
            abcVer = -1
            texVer = -1
            shaVer = -1
        else:
            self.attrIndex = attrIndex
            abcVer = cmds.getAttr('%s.ldvEnvAbcVer[%d]' % (self.ldvNode, self.attrIndex))
            texVer = cmds.getAttr('%s.ldvEnvTexVer[%d]' % (self.ldvNode, self.attrIndex))
            shaVer = cmds.getAttr('%s.ldvEnvShaderVer[%d]' % (self.ldvNode, self.attrIndex))

        if not self.loadAlembic(abcVer):
            self.isLoadFailed = True
            return

        self.loadTexture(texVer)

        self.loadShader(shaVer)
                
    def loadAlembic(self, ver):
        self.alembicFile = Alembic()
        
        print self.envAssetPath
        self.alembicFile.loadAlembicVersionList(self.envAssetPath)
        self.alembicVersionComboBox.addItems(self.alembicFile.alembicList)

        if not (ver == -1):
            self.alembicVersionComboBox.setCurrentText(ver)
        
        self.alembicFile.setCurrentVersion(self.alembicVersionComboBox.currentText())
        self.alembicFile.setCurrentVersionPath()
        
        self.alembicFile.loadAlembicJsonData()

        if ver == -1:
            self.alembicFile.alembicImport()
            cmds.setAttr('%s.ldvEnvAbcVer[%d]' % (self.ldvNode, self.attrIndex),
                         self.alembicVersionComboBox.currentText(), type='string')

        if self.alembicFile.msgDismiss:
            MessageBox(Message = "alembic Load Cancel",
                       Button = ["OK"])
            return False

        if ver == -1:
            self.alembicFileName = self.alembicFile.alembicFileName
            cmds.setAttr('%s.ldvEnvAbcName[%d]' % (self.ldvNode, self.attrIndex), self.alembicFileName, type='string')
        else:
            self.alembicFileName = cmds.getAttr('%s.ldvEnvAbcName[%d]' % (self.ldvNode, self.attrIndex))

        self.alembicVersionComboBox.currentIndexChanged.connect(self.changeAlembicImport)
        
        return True
        
    def changeAlembicImport(self):
        cmds.select(self.alembicFileName)
        mel.eval("doDelete;")

        cmds.setAttr('%s.ldvEnvAbcVer[%d]' % (self.ldvNode, self.attrIndex),
                     self.alembicVersionComboBox.currentText(),
                     type='string')
        self.alembicFile.setCurrentVersion(self.alembicVersionComboBox.currentText())
        self.alembicFile.setCurrentVersionPath()
        
        self.alembicFile.loadAlembicJsonData()
                
        self.alembicFile.alembicImport()
        
        if self.alembicFile.msgDismiss:
            MessageBox(Message = "alembic Load Cancel",
                       Button = ["OK"])
            return
        
        self.alembicFileName = self.alembicFile.alembicFileName
        cmds.setAttr('%s.ldvEnvAbcName[%d]' % (self.ldvNode, self.attrIndex), self.alembicFileName, type='string')
            
    def loadTexture(self, ver):
        # construct texture
        self.textureInfo = TextureCtrl(self.assetPath)

        # add comboBox
        self.textureInfo.loadVersionInfo()
        print "TextureVersionList :", self.textureInfo.pubTextureVersionList
        self.textureVersionComboBox.addItems(self.textureInfo.pubTextureVersionList)

        if not (ver == -1):
            self.textureVersionComboBox.setCurrentText(ver)
        else:
            cmds.setAttr('%s.ldvEnvTexVer[%d]' % (self.ldvNode, self.attrIndex), self.textureVersionComboBox.currentText(),
                         type='string')

        self.textureInfo.loadTextureChannel(version = self.textureVersionComboBox.currentText())

        # connect signal
        self.textureVersionComboBox.currentIndexChanged.connect(self.changeTextureVersion)

    def changeTextureVersion(self):
        if self.shaderInfo != None and self.textureInfo != None:
            cmds.setAttr('%s.ldvEnvTexVer[%d]' % (self.ldvNode, self.attrIndex), self.textureVersionComboBox.currentText(),
                         type='string')
            self.textureInfo.loadTextureChannel(version=self.textureVersionComboBox.currentText())

            self.shaderInfo.changeVersion(self.textureVersionComboBox.currentText())

    def loadShader(self, ver):
        self.shaderInfo = ShaderMng(self.assetName)

        self.shaderChannelAddComboBox()

        if not (ver == -1):
            self.shaderVersionComboBox.setCurrentText(ver)
        else:
            cmds.setAttr('%s.ldvEnvShaderVer[%d]' % (self.ldvNode, self.attrIndex), self.shaderVersionComboBox.currentText(),
                         type='string')

        self.shaderVersionComboBox.currentIndexChanged.connect(self.changeShaderFile)

        shaderPath = ""
        if not self.shaderVersionComboBox.currentText() == "Default":
            shaderPath = '{0}/{1}'.format(self.shaderDirPath, self.shaderVersionComboBox.currentText())

        self.shaderInfo.loadShader(alembicJsonData=self.alembicFile.getAlembicPubData(),
                                   textureChannelList=self.textureInfo.texChannelList,
                                   textureVersion=self.textureVersionComboBox.currentText(),
                                   shaderVersion=self.shaderVersionComboBox.currentText(),
                                   shaderPath=shaderPath)

    def shaderChannelAddComboBox(self):
        self.shaderVersionComboBox.clear()
        self.shaderVersionComboBox.addItem("Default")
        self.shaderDirPath = '/show/{0}/asset/shaders/{1}'.format(str(self.rulebook.flag["PROJECT"]),
                                                                  str(self.rulebook.flag["ASSETNAME"]))

        print "shaderDirPath :", self.shaderDirPath

        if not os.path.isdir(self.shaderDirPath):
            os.mkdir(self.shaderDirPath)

        for dir in os.listdir(self.shaderDirPath):
            self.shaderVersionComboBox.addItem(dir)

    def changeShaderFile(self, index):
        if not len(self.shaderInfo.shaderList) == 0:
            '''
            만약 이미 load한적이 있으면 (쉐이더를 바꿀경우)
            Shader를 지워줍니다.
            '''
            # try:
            self.shaderInfo.deleteAllShader()
            # except:
            #     self.shaderInfo.shaderList.clear()

        shaderPath = ""
        cmds.setAttr('%s.ldvEnvShaderVer[%d]' % (self.ldvNode, self.attrIndex), self.shaderVersionComboBox.currentText(),
                     type='string')
        if not self.shaderVersionComboBox.currentText() == "Default":
            shaderPath = '{0}/{1}'.format(self.shaderDirPath, self.shaderVersionComboBox.currentText())

        self.shaderInfo.loadShader(alembicJsonData=self.alembicFile.getAlembicPubData(),
                                   textureChannelList=self.textureInfo.texChannelList,
                                   textureVersion=self.textureVersionComboBox.currentText(),
                                   shaderVersion=self.shaderVersionComboBox.currentText(),
                                   shaderPath = shaderPath)
    
    def deleteItemInfo(self):
        self.alembicFile.deleteAlembic(self.alembicFileName)
        self.shaderInfo.deleteAllShader()

        size = cmds.getAttr('%s.ldvEnvPath' % self.ldvNode, size=True)
        for index in range(self.attrIndex, size - 1):
            cmds.setAttr('%s.ldvEnvPath[%d]' % (self.ldvNode, index),
                         cmds.getAttr('%s.ldvEnvPath[%d]' % (self.ldvNode, index + 1)), type='string')

            cmds.setAttr('%s.ldvEnvAbcVer[%d]' % (self.ldvNode, index),
                         cmds.getAttr('%s.ldvEnvAbcVer[%d]' % (self.ldvNode, index + 1)), type='string')

            cmds.setAttr('%s.ldvEnvTexVer[%d]' % (self.ldvNode, index),
                         cmds.getAttr('%s.ldvEnvTexVer[%d]' % (self.ldvNode, index + 1)), type='string')

            cmds.setAttr('%s.ldvEnvShaderVer[%d]' % (self.ldvNode, index),
                         cmds.getAttr('%s.ldvEnvShaderVer[%d]' % (self.ldvNode, index + 1)), type='string')

            cmds.setAttr('%s.ldvEnvAbcName[%d]' % (self.ldvNode, index),
                         cmds.getAttr('%s.ldvEnvAbcName[%d]' % (self.ldvNode, index + 1)), type='string')

        print (self.ldvNode, size - 1)
        mel.eval('AEremoveMultiElement %s.ldvEnvPath[%d]' % (self.ldvNode, size - 1))
        mel.eval('AEremoveMultiElement %s.ldvEnvAbcVer[%d]' % (self.ldvNode, size - 1))
        mel.eval('AEremoveMultiElement %s.ldvEnvTexVer[%d]' % (self.ldvNode, size - 1))
        mel.eval('AEremoveMultiElement %s.ldvEnvShaderVer[%d]' % (self.ldvNode, size - 1))
        mel.eval('AEremoveMultiElement %s.ldvEnvAbcName[%d]' % (self.ldvNode, size - 1))
        
    def getAssetName(self):
        return self.sourceName

    def getAssetPath(self):
        return self.assetPath