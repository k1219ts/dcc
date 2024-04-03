'''
'    @author    : daeseok.chae
'    @date      : 2017.02.10
'    @brief     : Asset Publish Tool MainScript
'''
import os
import sys
import subprocess
import glob
import string
import re
import json

import pymongo
from pymongo import MongoClient
from MongoDB import MongoDB
from MessageBox import MessageBox

from dxname import tag_parser

import pymodule.Qt as Qt
from pymodule.Qt import QtGui
from pymodule.Qt import QtCore

from pymodule.Qt import QtWidgets

if "PySide" in Qt.__binding__:
    import maya.cmds as cmds
    import maya.OpenMayaUI as mui
    if Qt.__qt_version__ > "5.0.0":
        import shiboken2 as shiboken
    else:
        import shiboken as shiboken

from AstRefPubUI import Ui_AstRefPub

import getpass

import dxConfig
import site
site.addsitedir(dxConfig.getConf("TRACTOR_API"))

import tractor.api.author as author

def getMayaWindow():
    try:
        ptr = mui.MQtUtil.mainWindow()
        return shiboken.wrapInstance(long(ptr), QtWidgets.QWidget)
    except:
        return None

from dxstats import inc_tool_by_user

# import maya.mel as mel

currentScriptPath = os.path.abspath(__file__)
rootpath = os.path.dirname(currentScriptPath)

category_for_type = [["animal", "creature", "male", "female", "etc"],
                     ["weapon", "technology", "fruits", "accessories", "machine", "interior", "etc"],
                     ["tree", "mountain", "grass", "rock", "building", "road", "plant", "etc"],
                     ["car", "aircraft", "vessel", "etc"]]

class AST_PUB_MAIN(QtWidgets.QWidget):
    def __init__(self, parent = getMayaWindow()):
        super(AST_PUB_MAIN, self).__init__(parent)

        self.setWindowFlags(QtCore.Qt.Window)

        self.ui = Ui_AstRefPub()
        self.ui.setupUi(self)

        self.ui.assetPathBtn.clicked.connect(self.openModelPath)
        self.ui.texturePathBtn.clicked.connect(self.openTextureDirectory)
        self.ui.mariPathBtn.clicked.connect(self.openTextureDirectory)
        self.ui.zBrushPathBtn.clicked.connect(self.openZBrushPath)
        self.ui.shaderPathBtn.clicked.connect(self.openShaderDirectory)
        self.ui.hairPathBtn.clicked.connect(self.openHairPath)
        self.ui.marvPathBtn.clicked.connect(self.openMarvPath)
        self.ui.STDPathBtn.clicked.connect(self.openSTDPath)

        self.ui.importPicBtn.clicked.connect(self.importPicture)
        self.ui.takePicBtn.clicked.connect(self.takeScreenShot)

        self.ui.resetBtn.clicked.connect(self.resetBtnClicked)
        self.ui.closeBtn.clicked.connect(self.closeBtn)
        self.ui.publishBtn.clicked.connect(self.publish)

        self.ui.typeComboBox.currentIndexChanged.connect(self.changedTypeComboBox)
        self.changedTypeComboBox(0)

        self.prjName = ""
        self.assetName = ""
        self.assetType = ""
        self.thumbnailFolderPath = "/home/{0}/Desktop/.thumbnail".format(getpass.getuser())
        self.thumbnailPath = ""

        inc_tool_by_user.run('action.AssetPubTool.open', getpass.getuser())


    def resetBtnClicked(self):
        self.prjName = ""
        self.assetName = ""
        self.assetType = ""

        self.ui.assetTitleEdit.setText('')
        self.ui.assetPathEdit.setText('')
        self.ui.texturePathEdit.setText('')
        self.ui.shaderPathEdit.setText('')
        self.ui.mariPathEdit.setText('')
        self.ui.zBrushPathEdit.setText('')
        self.ui.hairPathEdit.setText('')

        self.ui.tagsTextEdit.setPlainText('')
        self.ui.descTextEdit.setPlainText('')

        self.ui.imgLabel.setPixmap(None)
        self.ui.imgLabel.setScaledContents(True)

        self.previewFileName = ""
        if os.path.exists(self.previewPath):
            os.system('rm -rf %s' % self.previewPath)
        self.previewPath = ""

        self.thumbnailFileName = ""
        if os.path.exists(self.thumbnailPath):
            os.system('rm -rf %s' % self.thumbnailPath)
        self.thumbnailPath = ""


    def changedTypeComboBox(self, currentIndex):
        self.ui.categoryComboBox.clear()
        self.ui.categoryComboBox.addItems(category_for_type[currentIndex])

    # auto basic set
    def autoSet(self, fileName, isShader = False):
        print fileName
        splitFileName = fileName.split('/')

        try:
            self.assetName = splitFileName[5]
            self.prjName = splitFileName[2]
            if isShader == False:
                self.assetType = splitFileName[4]

            notShaderPath = self.getPrjBasePath(False)
            ShaderPath = self.getPrjBasePath(True)

            if self.ui.assetTitleEdit.text() == "":
                self.ui.assetTitleEdit.setText(self.assetName)

            tagList = tag_parser.run(fileName)
            self.addTag(tagList)

            if self.ui.assetPathEdit.text() == "":
                assetPath = "{0}/model/pub/scenes".format(notShaderPath)
                alembicListBeforeSort = []
                for alembicFile in glob.glob('{0}/*.abc'.format(assetPath)):
                    alembicListBeforeSort.append(alembicFile)

                alembicListBeforeSort.sort( key = os.path.getmtime )
                alembicListBeforeSort.reverse()

                if len(alembicListBeforeSort) > 0:
                    print "firstModel : ", alembicListBeforeSort[0]

                    self.ui.assetPathEdit.setText(alembicListBeforeSort[0])

            if self.ui.texturePathEdit.text() == "":
                assetPath = "{0}/texture/pub/tex".format(notShaderPath)
                textureVersionList = []

                for listdir in os.listdir(assetPath):
                    if os.path.isdir(os.path.join(assetPath, listdir)):
                        textureVersionList.append(listdir)

                textureVersionList.sort(reverse=True)

                if len(textureVersionList) > 0:
                    print "firstTexture :", "{0}/{1}".format(assetPath, textureVersionList[0])
                    self.ui.texturePathEdit.setText("{0}/{1}".format(assetPath, textureVersionList[0]))

            if self.ui.shaderPathEdit.text() == "":
                shaderPubList = glob.glob("{0}/txv*".format(ShaderPath))
                shaderPubList.sort()
                assetPath = shaderPubList[-1]
                self.ui.shaderPathEdit.setText(assetPath)

            if self.ui.hairPathEdit.text() == "":
                assetPath = "{0}/hair/pub/scenes".format(notShaderPath)
                alembicListBeforeSort = []
                for alembicFile in glob.glob('{0}/*.mb'.format(assetPath)):
                    alembicListBeforeSort.append(alembicFile)

                    alembicListBeforeSort.sort( key = os.path.getmtime )
                    alembicListBeforeSort.reverse()

                if len(alembicListBeforeSort) > 0:
                    self.ui.hairPathEdit.setText(alembicListBeforeSort[0])

        except Exception as e:
            print "autoSet Error :", e.message

        try:
            if self.ui.mariPathEdit.text() == "":
                assetPath = "{0}/texture/pub/mari".format(notShaderPath)
                mariVersionList = []

                mtime = lambda f: os.stat(os.path.join(assetPath, f)).st_mtime
                mariFiles = list(sorted(os.listdir(assetPath), key = mtime))

                for listdir in mariFiles:
                    if os.path.isfile(os.path.join(assetPath, listdir)):
                        mariVersionList.append(listdir)

                print mariVersionList
                mariVersionList.sort(reverse=True)

                if len(mariVersionList) > 0:
                    print "firstTexture :", "{0}/{1}".format(assetPath, mariVersionList[0])
                    self.ui.mariPathEdit.setText("{0}/{1}".format(assetPath, mariVersionList[0]))

        except Exception as e:
            print "autoSet Error :", e.message

        try:
            if self.ui.zBrushPathEdit.text() == "":
                assetPath = "{0}/model/pub/ztl".format(notShaderPath)

                if os.path.exists(assetPath):
                    if len(os.listdir(assetPath)) > 0:
                        self.ui.zBrushPathEdit.setText(assetPath)

        except Exception as e:
            print "autoSet Error :", e.message

        try:
            if self.ui.marvPathEdit.text() == "":
                assetPath = "{0}/model/pub/marv".format(notShaderPath)

                if os.path.exists(assetPath):
                    if len(os.listdir(assetPath)) > 0:
                        self.ui.marvPathEdit.setText(assetPath)

        except Exception as e:
            print "autoSet Error :", e.message

        try:
            if self.ui.STDPathEdit.text() == "":
                assetPath = "{0}/model/pub/spm".format(notShaderPath)

                if os.path.exists(assetPath):
                    if len(os.listdir(assetPath)) > 0:
                        self.ui.STDPathEdit.setText(assetPath)

        except Exception as e:
            print "autoSet Error :", e.message

    def getFilePathToName(self, filePath):
        if filePath == "":
            return ""
        splitPath = filePath.split('/')
        fileName = splitPath[len(splitPath) - 1]
        print fileName
        return fileName

    def getFilePathToList(self, basePath, filePath):
        if filePath == "":
            return []
        fileNameList = []

        dirName = os.path.basename(filePath)

        for fileName in os.listdir(filePath):
            if not fileName in ".thumb":
                fileNameList.append("{0}/{1}/{2}".format(basePath, dirName, fileName))

        print fileNameList
        return fileNameList

    def closeBtn(self):
        print "closeBtn"
        self.close()

    def publish(self):
        if self.ui.assetTitleEdit.text() == "":
            MessageBox("assetName is blank")
            return

        if self.ui.texturePathEdit.text() == "":
            MessageBox("texturePath is blank")
            return

        # if self.ui.shaderPathEdit.text() == "":
        #     MessageBox("shaderPath is blank")
        #     return

        # if self.thumbnailFileName == "":
        #     MessageBox("take picture")
        #     return

        assetName = self.assetName
        exportName = str(self.ui.assetTitleEdit.text())
        dataName = string.join(os.path.basename(str(self.ui.assetPathEdit.text())).split('_')[0:-2], '_')

        print assetName, dataName

        if assetName != dataName:
            messageBox = QtWidgets.QMessageBox()
            messageBox.setWindowTitle('assetName Check')
            messageBox.setIcon(QtWidgets.QMessageBox.Information)
            messageBox.setFont(QtGui.QFont("Cantarell", 13))
            messageBox.setText('''  === Export File Name Check ===  
    - model    : {0}_model_(VER).abc
    - texture  : {0}_(CHANNEL).(UDIM).tex
    - shader   : shaders/{1}/tx(VER)
    - mari     : {2}'''.format(dataName, assetName, str(os.path.basename(self.ui.mariPathEdit.text()))))
            messageBox.addButton("Yes", QtWidgets.QMessageBox.YesRole)
            messageBox.addButton("No", QtWidgets.QMessageBox.NoRole)

            fileNameCheckResult = messageBox.exec_()
            # result => Yes : 0 | No : 1
            if fileNameCheckResult == 1:
                # input new Name
                dialog = QtWidgets.QDialog(self)
                dialog.setGeometry(0, 0, 410, 40)
                dialog.setWindowTitle("Input data name [assetName => dataName]")

                label = QtWidgets.QLabel(dialog)
                label.setText("Data Name")
                label.setGeometry(10, 10, 90, 20)

                newNameLineEdit = QtWidgets.QLineEdit(dialog)
                newNameLineEdit.setGeometry(110, 10, 150, 20)

                okButton = QtWidgets.QPushButton(dialog)
                okButton.setText("OK")
                okButton.clicked.connect(dialog.accept)
                okButton.setGeometry(270, 10, 60, 20)

                closeButton = QtWidgets.QPushButton(dialog)
                closeButton.setText("Cancel")
                closeButton.clicked.connect(dialog.reject)
                closeButton.setGeometry(340, 10, 60, 20)

                dialogResult = dialog.exec_()
                print "DialogExec :", dialogResult
                print QtWidgets.QDialog.Accepted
                print QtWidgets.QDialog.Rejected

                if dialogResult == QtWidgets.QDialog.Rejected:
                    return

                dataName = str(newNameLineEdit.text())

        typeName = str(self.ui.typeComboBox.currentText())
        categoryName = str(self.ui.categoryComboBox.currentText())

        tagList = self.getTagsList()
        description = self.getDescription()
        if description == None:
            return

        modelPath = self.ui.assetPathEdit.text()                # /show/PRJ/asset/TYPE/ASSETNAME/model/pub/scenes/*.abc
        texturePath = self.ui.texturePathEdit.text()            # /show/PRJ/asset/TYPE/ASSETNAME/texture/pub/tex/VER
        mariPath = self.ui.mariPathEdit.text()                  # /show/PRJ/asset/TYPE/ASSETNAME/texture/pub/mari/VER
        zbrushPath = self.ui.zBrushPathEdit.text()              # /show/PRJ/asset/TYPE/ASSETNAME/model/pub/ztl
        marvPath = self.ui.marvPathEdit.text()  # /show/PRJ/asset/TYPE/ASSETNAME/model/pub/ztl
        shaderPath = self.ui.shaderPathEdit.text()              # /show/PRJ/asset/
        hairPath = self.ui.hairPathEdit.text()
        STDPath = self.ui.STDPathEdit.text()

        #======== DB insert working
        DBIP = dxConfig.getConf("DB_IP")
        dbName = "inventory"
        collName = "assets"

        client = MongoClient(DBIP)
        database = client[dbName]
        coll = database[collName]

        # nextAssetName = '%s%s' % (assetName, str(coll.find({'name': regex}).count() + 1).zfill(4))

        regex = re.compile("%s[0-9]{4}" % exportName)
        try:
            lastName = coll.find_one({'name': regex,
                                      'project':typeName,
                                      'category':categoryName}, sort=[('name', pymongo.DESCENDING)])['name']
            namePaddingNum = str(int(lastName[-4:]) + 1).zfill(4)
        except:
            namePaddingNum = '0000'

        nextAssetName = '%s%s' % (exportName, namePaddingNum)

        dbPlugin = MongoDB(dbName, collName)
        dbPlugin.setName(nextAssetName)
        dbPlugin.setProject(typeName)
        dbPlugin.setCategory(categoryName)
        dbPlugin.setDesc(description)
        dbPlugin.setTags(tagList)

        # checkDB = {}
        # checkDB["name"] = nextAssetName
        # checkDB["project"] = typeName
        # checkDB["category"] = categoryName
        # checkDB["enabled"] = True
        #
        # if dbPlugin.existDocument(checkDB):
        #     MessageBox("exists publish in inventory [%s]" % exportName)
        #     return

        # checkDB["enabled"] = False
        # if dbPlugin.existDocument(checkDB):
        #     MessageBox("current uploading in inventory [%s]" % exportName)
        #     return

        dbRecord = dbPlugin.getRecord()
        dbPlugin.insertDocument()

        nextAssetName = dbPlugin.resultID

        basePath = "/assetlib/3D/asset"
        projectPath = "{0}/{1}/{2}".format(basePath, typeName, categoryName)
        assetPath = "{0}/{1}".format(projectPath, nextAssetName)


        # tractor Setting
        job = author.Job()
        job.title = '(AssetLib) %s' % exportName
        # job.comment = ''
        # job.metadata = ''
        job.envkey = ['rfm2-21.7-maya-2017']
        job.service = 'convert'
        job.maxactive = 1
        job.tier = 'user'
        job.projects = ['user']

        serviceKey = 'convert'

        # /show/mkk3/asset/char/stag/model/pub/scenes/stag_model_v07.abc

        # directory mapping
        job.newDirMap(src='S:/', dst='/show/', zone='NFS')
        job.newDirMap(src='N:/', dst='/netapp/', zone='NFS')
        job.newDirMap(src='R:/', dst='/dexter/', zone='NFS')

        mainTask = author.Task(title='main Task')
        mainTask.serialsubtasks = 1

        scriptRoot = os.path.dirname(os.path.abspath(__file__))
        if '/home/' in scriptRoot:
            # scriptRoot = '/netapp/backstage/pub/apps/maya2/versions/2017/team/asset/linux/scripts/AssetBackupTool'
            scriptRoot = '/dexter/Cache_DATA/RND/daeseok/Inventory'

        makeDirTask = author.Task(title='make directories')
        mainTask.addChild(makeDirTask)

        tempThumPath = "/dexter/Cache_DATA/ASSET/.assetlib_thumbnail/"

        filesDic = {"thumbnail": "{0}/{1}".format(assetPath, "Thumbnail.tif"),
                    "preview": "{0}/{1}".format(assetPath, "Preview.tif")}

        if not self.thumbnailPath == "":
            print "Preview :", self.previewPath, tempThumPath
            if os.path.exists(self.previewPath):
                args = ['mv', '-vf', self.previewPath, tempThumPath]
                print args
                subprocess.Popen(args).wait()
            else:
                MessageBox("Not Find ScreenShot File\nPlease take a picture again")
                return

            print "Thumbnail :", self.thumbnailPath, tempThumPath
            if os.path.exists(self.thumbnailPath):
                args = ['mv', '-vf', self.thumbnailPath, tempThumPath]
                print args
                subprocess.Popen(args).wait()
            else:
                MessageBox("Not Find ScreenShot File\nPlease take a picture agin")
                return

            filesDic = {"thumbnail": "{0}/{1}".format(assetPath, self.thumbnailFileName),
                        "preview": "{0}/{1}".format(assetPath, self.previewFileName)}

            previewTask = author.Task(title='move previewFiles')
            command = "mv -vf {0}{1} {2}".format(tempThumPath, self.thumbnailFileName, assetPath)
            print command
            previewTask.addCommand(author.Command(argv=command, service=serviceKey))

            command = "mv -vf {0}{1} {2}".format(tempThumPath, self.previewFileName, assetPath)
            print command
            previewTask.addCommand(author.Command(argv=command, service=serviceKey))

            mainTask.addChild(previewTask)



        if not modelPath == "":
            command = "install -d -m 755 {0}".format(os.path.join(assetPath, 'model'))
            makeDirTask.addCommand(author.Command(argv=command, service=serviceKey))

            modelTask = author.Task(title='batchExportModel')

            # publish high model
            alembicTask = author.Task(title='ExportHighAbc')
            modelPubName = os.path.join(assetPath, 'model', '%s_model.abc' % dataName)
            command = ['mayapy', '%%D(%s/AstPub.py)' % scriptRoot,
                       '%%D(%s)' % modelPath,
                       '%%D(%s)' % assetPath,
                       '%%D(%s)' % modelPubName]
            print command
            alembicTask.addCommand(author.Command(argv=command, service=serviceKey, tags=['py']))
            modelTask.addChild(alembicTask)
            filesDic['model'] = modelPubName

            # publish json model
            jsonPath = modelPath.replace('.abc', '.json')
            if os.path.exists(jsonPath):
                jsonAlembicTask = author.Task(title='CopyJson')
                command = ['cp', '-rf', modelPath.replace('.abc', '.json'), modelPubName.replace('.abc', '.json')]
                print command
                jsonAlembicTask.addCommand(author.Command(argv=command, service=serviceKey))
                modelTask.addChild(jsonAlembicTask)

            mainTask.addChild(modelTask)

            modelName = modelPubName
            filesDic['model_json'] = modelName.replace('.abc', '.json')

            # if exists lowmodel, publish low model
            lowModelList = []
            for modelFileName in os.listdir(os.path.dirname(modelPath)):
                if '_low' in modelFileName:
                    lowModelList.append(modelFileName)

            if len(lowModelList) > 0:
                lowModelList.sort()
                lowModelList.reverse()

                lowAlembicTask = author.Task(title='ExportLowAbc')
                lowModelPath = os.path.join(os.path.dirname(modelPath), lowModelList[0])
                modelPubName = os.path.join(assetPath, 'model', '%s_model_low.abc' % dataName)
                command = ['mayapy', '%%D(%s/AstPub.py)' % scriptRoot,
                           '%%D(%s)' % lowModelPath,
                           '%%D(%s)' % assetPath,
                           '%%D(%s)' % modelPubName]
                print command
                lowAlembicTask.addCommand(author.Command(argv=command, service=serviceKey, tags=['py']))
                modelTask.addChild(lowAlembicTask)
                filesDic['model_low'] = modelPubName

            # if exists mid model, publish low model
            midModelList = []
            for modelFileName in os.listdir(os.path.dirname(modelPath)):
                if '_mid' in modelFileName:
                    midModelList.append(modelFileName)

            if len(midModelList) > 0:
                midModelList.sort()
                midModelList.reverse()

                midAlembicTask = author.Task(title='ExportMidAbc')
                midModelPath = os.path.join(os.path.dirname(modelPath), midModelList[0])
                modelPubName = os.path.join(assetPath, 'model', '%s_model_mid.abc' % dataName)
                command = ['mayapy', '%%D(%s/AstPub.py)' % scriptRoot,
                           '%%D(%s)' % midModelPath,
                           '%%D(%s)' % assetPath,
                           '%%D(%s)' % modelPubName]
                print command
                midAlembicTask.addCommand(author.Command(argv=command, service=serviceKey, tags=['py']))
                modelTask.addChild(midAlembicTask)
                filesDic['model_mid'] = modelPubName

        if not shaderPath == "":
            command = "install -d -m 755 {0}".format(os.path.join(assetPath, 'shader', 'rfm'))
            makeDirTask.addCommand(author.Command(argv=command, service=serviceKey))

            shaderTask = author.Task(title='batchExportShader')
            command = ['mayapy', '%%D(%s/ShaderPub.py)' % scriptRoot,
                       '%%D(%s)' % os.path.join(shaderPath, 'rfm', '%s_%s.ma' % (assetName, os.path.basename(shaderPath))),
                       '%%D(%s)' % os.path.join(assetPath, 'shader', 'rfm', '%s_%s.json' % (exportName, os.path.basename(shaderPath))),
                       '%%D(%s)' % dataName,
                       '%s' % getpass.getuser()]
            print command
            shaderTask.addCommand(author.Command(argv=command, service=serviceKey, tags=['py']))

            command = ['cp', '-rf', '%s' % os.path.join(shaderPath, 'rfm', '%s_%s.xml' % (assetName, os.path.basename(shaderPath))),
                                    '%s' % os.path.join(assetPath, 'shader', 'rfm', '%s_%s.xml' % (exportName, os.path.basename(shaderPath)))]
            print command
            shaderTask.addCommand(author.Command(argv=command, service=serviceKey))

            mainTask.addChild(shaderTask)

            shaderJsonFilePath = os.path.join(assetPath,
                                              'shader',
                                              'rfm',
                                              '%s_%s.json' % (exportName, os.path.basename(shaderPath)))

            shaderXmlFilePath = os.path.join(assetPath,
                                             'shader',
                                             'rfm',
                                             '%s_%s.xml' % (exportName, os.path.basename(shaderPath)))

            filesDic['shader_json'] = shaderJsonFilePath
            filesDic['shader_xml'] = shaderXmlFilePath

        if not texturePath == "":
            command = "install -d -m 755 {0}".format(os.path.join(assetPath, 'texture', 'tex'))
            makeDirTask.addCommand(author.Command(argv=command, service=serviceKey))

            command = "install -d -m 755 {0}".format(os.path.join(assetPath, 'texture', 'images'))
            makeDirTask.addCommand(author.Command(argv=command, service=serviceKey))

            command = "install -d -m 755 {0}".format(os.path.join(assetPath, 'texture', 'proxy'))
            makeDirTask.addCommand(author.Command(argv=command, service=serviceKey))

            # Read Layer Info
            layerNameList = []
            if not shaderPath == "":
                if os.path.exists(os.path.join(shaderPath, 'at')):
                    attributeFile = os.listdir(os.path.join(shaderPath, 'at'))
                    if len(attributeFile) > 0:
                        with open(os.path.join(shaderPath, 'at', attributeFile[0]), 'r') as f:
                            attributeData = json.load(f)
                            for shapeName in attributeData["Attributes"]:
                                try:
                                    layerName = attributeData["Attributes"][shapeName]['rman__riattr__user_txLayerName']
                                except:
                                    continue
                                if not layerName in layerNameList:
                                    layerNameList.append(layerName)
            else:
                layerNameList.append(dataName)

            print layerNameList

            textureTask = author.Task(title='batchCopyTexture')
            texFileList = []
            print texturePath
            for layerName in layerNameList:
                for texFile in os.listdir(texturePath):
                    print layerName, texFile
                    if layerName in texFile:
                        texTask = author.Task(title='image %s Copy'.format(texFile))
                        command = ['cp', '-f', '%s/%s' % (texturePath, texFile), '%s' % os.path.join(assetPath, "texture", "tex", texFile)]
                        texFileList.append(os.path.join(assetPath, "texture", "tex", texFile))
                        print command
                        texTask.addCommand(author.Command(argv=command, service=serviceKey))

                        textureTask.addChild(texTask)

            if len(texFileList):
                filesDic['tex'] = texFileList

            proxyFileList = []
            proxyPath = texturePath.replace('/tex/', '/proxy/')
            for layerName in layerNameList:
                for proxyFile in os.listdir(proxyPath):
                    print layerName, proxyFile
                    if layerName in proxyFile:
                        pxyTask = author.Task(title='image %s Copy'.format(proxyFile))
                        command = ['cp', '-f', '%s/%s' % (proxyPath, proxyFile),
                                   '%s' % os.path.join(assetPath, "texture", "proxy", proxyFile)]
                        proxyFileList.append(os.path.join(assetPath, "texture", "proxy", proxyFile))
                        print command
                        pxyTask.addCommand(author.Command(argv=command, service=serviceKey))

                        textureTask.addChild(pxyTask)

            if len(proxyFileList):
                filesDic['proxy'] = proxyFileList

            # if len(proxyFileList) + len(texFileList) > 0:
            #     mainTask.addChild(textureTask)

            copyImagePath = os.path.join(os.path.dirname(os.path.dirname(texturePath)), os.path.basename(texturePath))

            # extensionList = []
            # for i in os.listdir(copyImagePath):
            #     if os.path.isfile(os.path.join(copyImagePath, i)) and not os.path.splitext(os.path.join(copyImagePath, i))[-1] in extensionList:
            #         extensionList.append(os.path.splitext(os.path.join(copyImagePath, i))[-1])
            #         continue

            # print extensionList

            imgFileList = []
            for layerName in layerNameList:
                for imageFileName in os.listdir(copyImagePath):
                    if layerName in imageFileName:
                        imageTask = author.Task(title = 'image %s Copy'.format(imageFileName))
                        command = ['cp', '-f', '%s/%s' % (copyImagePath, imageFileName), '%s' % os.path.join(assetPath, "texture", "images", imageFileName)]
                        print command
                        imageTask.addCommand(author.Command(argv=command, service=serviceKey))
                        imgFileList.append('%s' % os.path.join(assetPath, "texture", "images", imageFileName))
                        textureTask.addChild(imageTask)

            if len(imgFileList) > 0:
                filesDic['images'] = imgFileList

            if len(proxyFileList) + len(imgFileList) + len(texFileList) > 0:
                mainTask.addChild(textureTask)

        if not mariPath == "":
            command = "install -d -m 755 {0}".format(os.path.join(assetPath, 'mari'))
            makeDirTask.addCommand(author.Command(argv=command, service=serviceKey))

            mariTask = author.Task(title='CopyMariArchive')
            archivePath = '%s' % os.path.join(assetPath, "mari", '%s.mra' % exportName)
            command = ['cp', '-rf', '%s' % mariPath, archivePath]
            print command
            mariTask.addCommand(author.Command(argv=command, service=serviceKey))

            filesDic['mari'] = archivePath

            mainTask.addChild(mariTask)

        if not zbrushPath == "":
            command = "install -d -m 755 {0}".format(os.path.join(assetPath, 'ztl'))
            makeDirTask.addCommand(author.Command(argv=command, service=serviceKey))

            zbrushTask = author.Task(title='CopyZbrush')
            archivePath = '%s' % os.path.join(assetPath, "ztl")
            command = ['cp', '-rf', '%s' % zbrushPath, archivePath]
            print command
            zbrushTask.addCommand(author.Command(argv=command, service=serviceKey))

            filesDic['zbrush'] = archivePath

            mainTask.addChild(zbrushTask)

        if not marvPath == "":
            command = "install -d -m 755 {0}".format(os.path.join(assetPath, 'marv'))
            makeDirTask.addCommand(author.Command(argv=command, service=serviceKey))

            marvTask = author.Task(title='Copy Marvelous')
            archivePath = '%s' % os.path.join(assetPath, "marv")
            command = ['cp', '-rf', '%s' % marvPath, archivePath]
            print command
            marvTask.addCommand(author.Command(argv=command, service=serviceKey))

            filesDic['marv'] = archivePath

            mainTask.addChild(marvTask)

        if not STDPath == "":
            command = "install -d -m 755 {0}".format(os.path.join(assetPath, 'spm'))
            makeDirTask.addCommand(author.Command(argv=command, service=serviceKey))

            stdTask = author.Task(title='Copy Speed Tree Data')
            archivePath = '%s' % os.path.join(assetPath, "spm")
            command = ['cp', '-rf', '%s' % marvPath, archivePath]
            print command
            stdTask.addCommand(author.Command(argv=command, service=serviceKey))

            filesDic['speedtree'] = archivePath

            mainTask.addChild(stdTask)

        if not hairPath == "":
            command = "install -d -m 755 {0}".format(os.path.join(assetPath, 'hair'))
            makeDirTask.addCommand(author.Command(argv=command, service=serviceKey))

            hairTask = author.Task(title='batchCopyHairScene')
            command = ['cp', '-rf', '%s' % hairPath, '%s' % os.path.join(assetPath, "hair", "%s_hair.mb" % exportName)]

            print command
            hairTask.addCommand(author.Command(argv=command, service=serviceKey, tags=['py']))

            fileName = os.path.basename(hairPath)
            version = os.path.splitext(fileName)[0].split('_')[-1]
            cachePath = os.path.join(hairPath.split('/scenes')[0], 'data', 'zenn', '%s_hair_%s' % (assetName, version))

            print cachePath # /show/god/asset/char/jhKnn/model/pub/scenes/jhKnn_model_v03.abc
            if os.path.exists(cachePath):
                cacheOutputPath = os.path.join(assetPath, "hair", "data", "zenn", "%s_hair" % exportName)

                command = "install -d -m 755 {0}".format(cacheOutputPath)
                hairTask.addCommand(author.Command(argv=command, service=serviceKey))

                command = ['cp', '-rf', '%s/.' % cachePath, cacheOutputPath]

                print command
                hairTask.addCommand(author.Command(argv=command, service=serviceKey, tags=['py']))

                filesDic['hair_cache'] = '%s' % os.path.join(assetPath, "hair", "data", "zenn", "%s_hair" % exportName)

            mainTask.addChild(hairTask)

            filesDic['hair'] = os.path.join(assetPath, "hair", "%s_hair.mb" % exportName)

        if self.thumbnailPath == "":
            tempSceneTask = author.Task(title = 'make lookdev temp scene')

            previewSceneFile = '%s/%s' % (assetPath, "Preview.mb")
            command = ['mayapy', '%%D(%s/AutoLookdev.py)' % scriptRoot, dbPlugin.resultID, previewSceneFile]
            # render spool

            print command
            tempSceneTask.addCommand(author.Command(argv=command, service = serviceKey, tags = ['py']))
            mainTask.addChild(tempSceneTask)

            # previewSceneFile = '/assetlib/3D/asset/prop/weapon/5a02ba13469477229ef35360/preview.mb'

            previewTask = author.Task(title = "Preview Render")
            previewTask.serialsubtasks = 1
            self.renderCmd(Parent = previewTask, previewScene = previewSceneFile)

            filesDic["preview"] = '%s/Preview.tif' % os.path.dirname(previewSceneFile)
            filesDic["thumbnail"] = '%s/Thumbnail.tif' % os.path.dirname(previewSceneFile)

            convertTask = author.Task(title = "Preview Render")

            command = ['ffmpeg', '-i', filesDic['preview'], '-s', '320x240', filesDic['thumbnail']]

            convertTask.addCommand(author.Command(argv = command, service = serviceKey))

            previewTask.addChild(convertTask)

            mainTask.addChild(previewTask)


        dbPlugin.updateDocument(dbPlugin.resultID, filesDic)

        command = "/netapp/backstage/pub/bin/inventory/enableDBRecord.py {0} {1} {2}".format(dbName, collName, dbPlugin.resultID)
        print command
        # dbEnableTask = author.Task(title="db enabled change true")
        mainTask.addCommand(author.Command(argv = command, service=serviceKey, tags = ['py']))
        # mainTask.addChild(dbEnableTask)

        job.addChild(mainTask)
        job.priority = 999

        # author.setEngineClientParam(hostname=dxConfig.getConf("TRACTOR_CACHE_IP"),
        author.setEngineClientParam(hostname="10.0.0.35",
                                    port=dxConfig.getConf("TRACTOR_PORT"),
                                    user=getpass.getuser(),
                                    debug=True)

        job.spool()

        author.closeEngineClient()
        job.asTcl()

        MessageBox("success job spool.")

        self.resetBtnClicked()

    def renderCmd(self, Parent = None, previewScene = None):

        # make Render command for rman
        command = ['Render', '-r', 'rman']

        # output file name convention
        command += ['-fnc', 2]

        # output file format
        command += ['-of', 'Tiff8']

        # ignore image plane
        command += ['-iip']

        # sRGB set
        command += ['-rgb', 1, '-gamma', 2.2]

        # attrbute sampling
        command += ['-setAttr', "Hider:incremental", 1]
        command += ['-setAttr', "Hider:minsamples", 0]
        command += ['-setAttr', "Hider:maxsamples", 64]

        # set frame
        # command += ['-s', 1]
        # command += ['-e', 1]
        # command += ['-b', 1]

        # set resolution
        command += ['-res', 960, 540]

        # set prj directory
        command += ['-proj', "%%D(%s)" % os.path.dirname(previewScene)]
        command += ['-rd', "%%D(%s)" % os.path.dirname(previewScene)]
        command += ['-im', "%%D(%s/Preview)" % os.path.dirname(previewScene)]

        # last, render scene files
        command += ["%%D(%s)" % previewScene]

        RenderTask = author.Task(title=str("Preview Render"))
        RenderTask.addCommand(
            author.Command(service='PixarRender',
                           tags=['prman', 'lt_prman'],
                           argv=command, atleast=1)
        )

        Parent.addChild(RenderTask)

    def addTag(self, tagList):
        for tag in tagList:
            if not tag in self.ui.tagsTextEdit.toPlainText():
                print tag
                self.ui.tagsTextEdit.append(tag)

    def getTagsList(self):
        tagList = []
        for tag in self.ui.tagsTextEdit.toPlainText().split('\n'):
            tagList.append(str(tag))
        print tagList
        return tagList

    def getDescription(self):
        print self.ui.descTextEdit.toPlainText()
        try:
            return self.ui.descTextEdit.toPlainText()
        except UnicodeEncodeError:
            MessageBox("Only English")
            return None

    def getPrjBasePath(self, isShader = False):
        dirPath = "/show"

        if not self.prjName == "":
            if isShader == False:
                dirPath += "/{0}/asset".format(self.prjName)
                dirPath += "/{0}/{1}".format(self.assetType, self.assetName)
            else:
                dirPath += "/{0}/asset/shaders/{1}".format(self.prjName, self.assetName)

        print dirPath
        return dirPath

    def getOpenDirectory(self, titleCaption, startDirPath):
        fileName = ""
        if "PyQt" in Qt.__binding__:
            fileName = QtWidgets.QFileDialog.getExistingDirectory(self, titleCaption, startDirPath)
        else:
            fileName = str(cmds.fileDialog2(fileMode = 3,
                                       caption = titleCaption,
                                       okCaption = "Load",
                                       startingDirectory = startDirPath)[0])

        if fileName.startswith('/netapp/dexter/show'):
            fileName = fileName.replace('/netapp/dexter/show', '/show')
        return fileName

    def getOpenFile(self, titleCaption, startDirPath, exrCaption):
        fileName = ""
        if "PyQt" in Qt.__binding__:
            fileName = QtWidgets.QFileDialog.getOpenFileName(self, titleCaption, startDirPath, exrCaption)
        else:
            print "?"
            fileName = str(cmds.fileDialog2(fileMode = 1,
                                       caption = titleCaption,
                                       okCaption = "Load",
                                       startingDirectory = startDirPath)[0])
        if fileName.startswith('/netapp/dexter/show'):
            fileName = fileName.replace('/netapp/dexter/show', '/show')
        return fileName

    def openShaderDirectory(self):
        titleCaption = "Load Shader Directory"
        dirPath = self.getPrjBasePath(True)
        print dirPath

        if self.ui.shaderPathEdit.text() != "":
            dirPath = str(self.ui.shaderPathEdit.text())

        fileName = self.getOpenDirectory(titleCaption, dirPath)

        self.autoSet(fileName, True)

        if not fileName == "":
            self.ui.shaderPathEdit.setText(fileName)

    def openTextureDirectory(self):
        titleCaption = "Load Texture Directory"
        dirPath = self.getPrjBasePath()

        if self.ui.texturePathEdit.text() != "":
            dirPath = str(self.ui.texturePathEdit.text())

        fileName = self.getOpenDirectory(titleCaption, dirPath)

        self.autoSet(fileName, False)

        if not fileName == "":
            self.ui.texturePathEdit.setText(fileName)

    def openMariPath(self):
        titleCaption = "Load Mari Archive"
        exrCaption = "Mari Archive File (*.mra)"

        dirPath = self.getPrjBasePath()

        if self.ui.mariPathEdit.text() != "":
            dirPath = str(self.ui.mariPathEdit.text())

        fileName = self.getOpenFile(titleCaption, dirPath, exrCaption)

        self.autoSet(fileName, False)

        if not fileName == "":
            self.ui.mariPathEdit.setText(fileName)

    def openZBrushPath(self):
        titleCaption = "Load Zbrush Directory"

        dirPath = self.getPrjBasePath()
        if self.ui.zBrushPathEdit.text() != "":
            dirPath = str(self.ui.zBrushPathEdit.text())

        fileName = self.getOpenDirectory(titleCaption, dirPath)

        self.autoSet(fileName, False)

        if not fileName == "":
            self.ui.zBrushPathEdit.setText(fileName)

    def openMarvPath(self):
        titleCaption = "Load Marv Directory"

        dirPath = self.getPrjBasePath()
        if self.ui.marvPathEdit.text() != "":
            dirPath = str(self.ui.marvPathEdit.text())

        fileName = self.getOpenDirectory(titleCaption, dirPath)

        self.autoSet(fileName, False)

        if not fileName == "":
            self.ui.marvPathEdit.setText(fileName)

    def openSTDPath(self):
        titleCaption = "Load Speed Tree Directory"

        dirPath = self.getPrjBasePath()
        if self.ui.STDPathEdit.text() != "":
            dirPath = str(self.ui.STDPathEdit.text())

        fileName = self.getOpenDirectory(titleCaption, dirPath)

        self.autoSet(fileName, False)

        if not fileName == "":
            self.ui.STDPathEdit.setText(fileName)

    def openModelPath(self):
        titleCaption = "Load Alembic File"
        exrCaption = "Alembic File (*.abc)"

        dirPath = self.getPrjBasePath()
        if self.ui.assetPathEdit.text() != "":
            dirPath = str(self.ui.assetPathEdit.text())

        fileName = self.getOpenFile(titleCaption, dirPath, exrCaption)

        self.autoSet(fileName, False)

        if not fileName == "":
            self.ui.assetPathEdit.setText(fileName)

    def openHairPath(self):
        titleCaption = "Load Hair File"
        exrCaption = "shader File (*.mb)"

        dirPath = self.getPrjBasePath()
        if self.ui.hairPathEdit.text() != "":
            dirPath = str(self.ui.hairPathEdit.text())

        fileName = self.getOpenFile(titleCaption, dirPath, exrCaption)

        self.autoSet(fileName, False)

        if not fileName == "":
            self.ui.hairPathEdit.setText(fileName)

    def importPicture(self):
        fileName = ""
        # if "PyQt" in Qt.__binding__:
        titleCaption = "Load Thumbnail Image"
        exrCaption = "Image File (*.png)"

        dirPath = self.getPrjBasePath()

        fileName = QtWidgets.QFileDialog.getOpenFileName(self, titleCaption, dirPath, exrCaption)
        # else:
        #     cmds.setAttr("defaultRenderGlobals.imageFormat", 32)
        #
        #     fileName = "{0}/{1}".format(self.thumbnailFolderPath, self.assetName)
        #     cmds.playblast(frame=[1], format="image", f=fileName, fp = 0, viewer=True, p=100, widthHeight=[230, 160], orn=False)
        #     fileName += '.0.png'

        print fileName
        imgPix = QtGui.QPixmap(fileName)
        imgPix.scaled(230, 160, QtCore.Qt.KeepAspectRatio)

        self.ui.imgLabel.setPixmap(imgPix)
        self.ui.imgLabel.setScaledContents(True);

        args = ['mkdir', '-p', self.thumbnailFolderPath]
        if not os.path.isdir(self.thumbnailFolderPath):
            print args
            subprocess.Popen(args).wait()

        prevImgPix = imgPix
        self.previewFileName = "{0}_preview.png".format(self.ui.assetTitleEdit.text())
        self.previewPath = "{0}/{1}".format(self.thumbnailFolderPath, self.previewFileName)
        prevImgPix.save(self.previewPath, "PNG", 100)

        thumImgPix = imgPix
        self.thumbnailFileName = "{0}_thumbnail.png".format(self.ui.assetTitleEdit.text())
        self.thumbnailPath = "{0}/{1}".format(self.thumbnailFolderPath, self.thumbnailFileName)

        thumImgPix = thumImgPix.scaled(320, 240, QtCore.Qt.KeepAspectRatio)
        thumImgPix.save(self.thumbnailPath, "PNG", 100)

    def setPixmap(self, fileName):
        print fileName
        imgPix = QtGui.QPixmap(fileName)
        imgPix.scaled(230, 160, QtCore.Qt.KeepAspectRatio)

        self.ui.imgLabel.setPixmap(imgPix)
        self.ui.imgLabel.setScaledContents(True);

        args = ['mkdir', '-p', self.thumbnailFolderPath]
        if not os.path.isdir(self.thumbnailFolderPath):
            print args
            subprocess.Popen(args).wait()

        prevImgPix = imgPix
        self.previewFileName = "{0}_preview.png".format(self.ui.assetTitleEdit.text())
        self.previewPath = "{0}/{1}".format(self.thumbnailFolderPath, self.previewFileName)
        prevImgPix.save(self.previewPath, "PNG", 100)

        thumImgPix = imgPix
        self.thumbnailFileName = "{0}_thumbnail.png".format(self.ui.assetTitleEdit.text())
        self.thumbnailPath = "{0}/{1}".format(self.thumbnailFolderPath, self.thumbnailFileName)

        thumImgPix = thumImgPix.scaled(320, 240, QtCore.Qt.KeepAspectRatio)
        thumImgPix.save(self.thumbnailPath, "PNG", 100)

    def takeScreenShot(self):
        args = ['/bin/gnome-screenshot', '-a', '-c']
        subprocess.Popen(args).wait()

        clip = QtWidgets.QApplication.clipboard()
        imgPix = clip.pixmap()
        imgPix.scaled(230, 160, QtCore.Qt.KeepAspectRatio)

        self.ui.imgLabel.setPixmap(imgPix)
        self.ui.imgLabel.setScaledContents(True)

        args = ['mkdir', '-p', self.thumbnailFolderPath]
        if not os.path.isdir(self.thumbnailFolderPath):
            print args
            subprocess.Popen(args).wait()

        prevImgPix = clip.pixmap()
        self.previewFileName = "{0}_preview.png".format(self.ui.assetTitleEdit.text())
        self.previewPath = "{0}/{1}".format(self.thumbnailFolderPath, self.previewFileName)
        prevImgPix.save(self.previewPath, "PNG", 100)

        thumImgPix = clip.pixmap()
        self.thumbnailFileName = "{0}_thumbnail.png".format(self.ui.assetTitleEdit.text())
        self.thumbnailPath = "{0}/{1}".format(self.thumbnailFolderPath, self.thumbnailFileName)

        thumImgPix = thumImgPix.scaled(320, 240, QtCore.Qt.KeepAspectRatio)
        thumImgPix.save(self.thumbnailPath, "PNG", 100)

def main():
    if "PyQt" in Qt.__binding__:
        app = QtWidgets.QApplication(sys.argv)
        astPubMain = AST_PUB_MAIN()
        astPubMain.show()
        sys.exit(app.exec_())
    elif "PySide" in Qt.__binding__:
        astPubMain = AST_PUB_MAIN()
        astPubMain.show()

if __name__ == "__main__":
    main()
