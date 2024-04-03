# coding=cp949
import os
import json
from String import String
from MessageBox import MessageBox
import glob

import re

import maya.cmds as cmds

currentScriptPath = os.path.abspath(__file__)
srcPath = os.path.dirname(currentScriptPath)

class ShaderMng():
    def __init__(self, assetName):
        '''
        Chapter1 : Load Shader
        Chapter2 : Change Shader
        Chapter3 : Delete Shader
        '''
        self.assetName = assetName
        self.shaderList = list()

    def loadShader(self, alembicJsonData, textureChannelList, textureVersion, shaderVersion = "Default", shaderPath = ""):
        if shaderVersion == "Default": # 만약 쉐이더 펍한게 아니라면
            '''
            Action1 : LoadJsonData
            Action2 : GetMaterialInfoForAlembic
            Action3 : Import Shader / Channel
            Action4 : Connection Shader Setting
            '''
            self.loadShaderJsonData()
            self.getMaterialInfoForAlembic(alembicJsonData)
            self.importBasicShaderFile(self.assetName)
            self.connectBasicChannel(self.assetName, textureChannelList, textureVersion)
        else:
            '''
            Action1 : Import Shader
            '''
            file = glob.glob(os.path.join(shaderPath, "rfm", '*.ma'))

            self.importShaderFile(file[0])

    def loadShaderJsonData(self):
        '''
        기본적으로 해 줄 쉐이더의 Json정보를 불러옵니다.
        :return: 
        '''
        jsonPath = os.path.join(srcPath, String.shaderJsonPath)

        with open(jsonPath, 'r') as jsonData:
            self.basicShaderJson = json.load(jsonData)

    def getMaterialInfoForAlembic(self, alembicJsonData):
        '''
        Alembic
        :param alembicJsonData: Alembic Pub할때 나오는 Json데이터를 넘겨줍니다. 
        :return: 
        '''
        self.materialNameList = []

        if alembicJsonData == None:
            return

        for layer in alembicJsonData["DisplayLayer"]:
            for layerMember in alembicJsonData["DisplayLayer"][layer]["members"]:
                if not "_PLY" in layerMember:
                    MessageBox(Message='{0} not have "_PLY"'.format(layerMember),
                               Button=["OK"])
                    continue
                splitText = layerMember.split('_')
                length = len(splitText)
                self.materialNameList.append(splitText[length - 2])

        self.materialNameList = list(set(self.materialNameList))

    def importBasicShaderFile(self, assetName):
        shaderPath = String.shaderPath

        if len(self.shaderList) != 0:
            self.shaderList.clear()

        for materialName in self.materialNameList:  # Matrerial Read List
            if cmds.objExists(self.shaderName(assetName, materialName)) == False:  # Overlap Check
                if materialName in os.listdir(shaderPath):  # Shader Check
                    for listdir in os.listdir(shaderPath):
                        if listdir in materialName:  # if Exist Material name
                            shaderLatestFile = sorted(glob.glob(os.path.join(shaderPath, listdir, "*.ma")))[-1]
                            cmds.file(shaderLatestFile, i=True, mnc=1)

                            rename = cmds.rename(self.shaderName("assetName", listdir), self.shaderName(assetName, materialName))

                            objAov = cmds.shadingNode("DxObjectAOV", asTexture=True,
                                                      name=self.shaderName(assetName, materialName) + '_AOV')
                            cmds.connectAttr('%s.resultAOV' % objAov, '%s.utilityPattern[0]' % rename)

                            self.shaderList.append(rename)

                            cmds.select(rename)
                            shadingEngine = cmds.listConnections(type="shadingEngine")
                            cmds.rename(shadingEngine, rename + "_SG")
                            break
                elif 'M' == materialName[0]:
                    shadingName = cmds.shadingNode('PxrSurface', asShader=True, n = self.shaderName(assetName, materialName))

                    objAov = cmds.shadingNode("DxObjectAOV", asTexture=True, name=self.shaderName(assetName, materialName) + '_AOV')
                    cmds.connectAttr('%s.resultAOV' % objAov, '%s.utilityPattern[0]' % shadingName)

                    cmds.createNode('shadingEngine', n = self.shaderName(assetName, materialName) + '_SG')

                    cmds.connectAttr(self.shaderName(assetName, materialName) + '.outColor', self.shaderName(assetName, materialName) + '_SG.surfaceShader')
                    self.shaderList.append(shadingName)
            else:
                self.shaderList.append(self.shaderName(assetName, materialName))


        if len(self.shaderList) == 0:
            MessageBox(Message='Material does not exist',
                       Button=["OK"])

    def importShaderFile(self, shaderFilePath):
        # if len(self.shaderList) != 0:
        #     self.shaderList.clear()

        # cmds.file(shaderFilePath, i = True, mnc = True)

        # shaderFilePath = '/show/god/asset/shaders/tattooMan1/txv04/rfm/tattooMan1_txv04.ma'

        f = open(shaderFilePath, 'r')
        data = f.read()
        f.close()

        self.shaderList = []
        # clear redundant shader
        fileSG = re.findall(r'createNode .+ -n "(.*)"', data)
        for file in fileSG:
            if not 'lightlinker' in file.lower():
                self.shaderList.append(file)

        cmds.file(shaderFilePath, i=True, type='mayaAscii',
                  mergeNamespacesOnClash=False,
                  rpr=os.path.splitext(os.path.basename(shaderFilePath))[0],
                  options='v=0;p=17;f=0', pr=True)

        # debug
        cmds.warning('Import Maya : %s' % shaderFilePath)

        for index, i in enumerate(self.shaderList):
            print i, self.assetName + "_" + i
            if not self.assetName in i:
                # print "rename :", cmds.rename(i, self.assetName + "_" + i)
                self.shaderList[index] = cmds.rename(i, self.assetName + "_" + i)


    def connectBasicChannel(self, assetName, textureChannelList, textureVersion):
        shaderDxTexture = "DxTexture"
        shaderName = str(assetName)
        shaderTextureList = textureChannelList

        for tex in shaderTextureList:
            # tex have diffC, specG, specR, norm, difI
            if not tex in self.basicShaderJson[String.dxTextureCommon]:
                continue

            shaderNodeName = "{0}_{1}".format(shaderName, tex)
            if not cmds.objExists(shaderNodeName):
                texNode = cmds.shadingNode(shaderDxTexture, asTexture=True, name=shaderNodeName)
                self.shaderList.append(texNode)
                cmds.setAttr("{0}.txchannel".format(texNode), tex, type="string")
                cmds.setAttr("{0}.linearize".format(texNode), self.basicShaderJson[String.dxTextureCommon][tex]["linearize"])
                try:
                    cmds.setAttr("{0}.txmode".format(texNode), 1)
                except:
                    cmds.setAttr("{0}.mode".format(texNode), 1)  # File : 0 Attribute : 1
                cmds.setAttr("{0}.txversion".format(texNode), textureVersion, type="string")
                color = self.basicShaderJson[String.dxTextureCommon][tex]['color']
                cmds.setAttr("{0}.missingColor".format(texNode), color[0], color[1], color[2], type="double3")
            else:
                texNode = shaderNodeName

            for materialName in self.materialNameList:
                connectionList = self.basicShaderJson[String.dxTextureCommon][tex][String.connections]

                startConnection = None
                endConnection = None

                # connection work
                for connection in connectionList:
                    connectShaderNodeName = "{assetName}_{material}_{channel}_{connection}".format(assetName = shaderName, channel = tex, connection = connection, material = materialName)
                    connectShadingNode = cmds.shadingNode(connection, asTexture=True, name=connectShaderNodeName)
                    self.shaderList.append(connectShadingNode)

                    connectionChannel = "default"
                    if self.basicShaderJson[String.utility_Common][connection].has_key(tex):
                        connectionChannel = tex

                    for attr in self.basicShaderJson[String.utility_Common][connection][connectionChannel]:
                        if attr != "input" and attr != "output":
                            cmds.setAttr("{nodeName}.{attr}".format(nodeName = connectShadingNode, attr = attr), self.basicShaderJson[String.utility_Common][connection][connectionChannel][attr])
                            print connectShadingNode, attr, self.basicShaderJson[String.utility_Common][connection][connectionChannel][attr]

                    connection_input = connectShadingNode + self.basicShaderJson[String.utility_Common][connection][connectionChannel]['input']
                    connection_output = connectShadingNode + self.basicShaderJson[String.utility_Common][connection][connectionChannel]['output']

                    if startConnection == None:
                        startConnection = connection_input
                    elif not len(connectionList) == 1:
                        cmds.connectAttr(endConnection,
                                         connection_input)

                    endConnection = connection_output

                # X -> startConnection
                cmds.connectAttr(texNode + self.basicShaderJson[String.dxTextureCommon][tex]["output"],
                                 startConnection)

                # endconnection -> Y[]

                for arriveShaderType in self.basicShaderJson[String.dxTextureCommon][tex]["arrive"]:
                    for arriveProperty in self.basicShaderJson[String.dxTextureCommon][tex]["arrive"][arriveShaderType]:
                        try:
                            cmds.connectAttr(endConnection,
                                             "{0}_{1}_SHD{2}".format(assetName, materialName, arriveProperty))
                        except Exception as e:
                            print endConnection, "{0}_{1}_SHD{2}".format(assetName, materialName, arriveProperty)
                            print e.message
                            pass

                if tex == "disI":
                    ShaderNodeName = "{0}_{1}_{2}_PxrDispTransform".format(shaderName, materialName, tex)
                    cmds.setAttr("{0}.dispRemapMode".format(ShaderNodeName), 2)
                    cmds.setAttr("{0}.dispCenter".format(ShaderNodeName), 0.5)

    def shaderName(self, assetName, materialName):
        return '%s_%s_SHD' % (assetName, materialName)

    def changeVersion(self, version):
        dxTextureList = cmds.ls("%s*" % self.assetName, type = 'DxTexture')
        for i in dxTextureList:
            cmds.setAttr('%s.txversion' % i, version, type = "string")

    # def deleteShader(self, shaderName):
    #     cmds.select(shaderName)
    #     connectList = set(cmds.listConnections())
    #     cmds.delete(shaderName)
    #
    #     for connectItem in connectList:
    #         if self.assetName in connectItem:
    #             self.childRemoveShader(connectItem)
    #
    # def childRemoveShader(self, parentItem):
    #     cmds.select(parentItem)
    #     connectList = set(cmds.listConnections())
    #     cmds.delete(parentItem)
    #
    #     if connectList == None:
    #         return
    #
    #     for child in connectList:
    #         if self.assetName in child:
    #             self.childRemoveShader(child)

    def deleteAllShader(self):
        cmds.delete(self.shaderList)
        self.shaderList = []
