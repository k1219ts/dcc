import sys
# import Qt
# from Qt import QtGui
# from Qt import QtCore
# from Qt import QtWidgets
from PySide2 import QtWidgets, QtCore, QtGui

# if "Side" in Qt.__binding__:
import maya.cmds as cmds

from HIKImportDialogUI import Ui_Form
from MessageBox import MessageBox
from MongoDB import MongoDB

DBNAME = "inventory"
COLLNAME = "anim_tags"

class HIKImportDialog(QtWidgets.QDialog):
    def __init__(self, parent):
        self.parent = parent
        QtWidgets.QDialog.__init__(self, parent)

        self.ui = Ui_Form()
        self.ui.setupUi(self)

        mayaWindow = parent
        self.move(mayaWindow.frameGeometry().center() - self.frameGeometry().center())

        self.dbPlugin = MongoDB(DBNAME, COLLNAME)

        # Connect Signal
        self.ui.Title1ComboBox.currentIndexChanged.connect(self.currentTitle1IndexChange)
        self.ui.Title2ComboBox.currentIndexChanged.connect(self.currentTitle2IndexChange)
        self.ui.crdRadioBtn.clicked.connect(self.radioBtnState)
        self.ui.aniRadioBtn.clicked.connect(self.radioBtnState)
        self.ui.mcpRadioBtn.clicked.connect(self.radioBtnState)

        self.initializeDataSet(0, "Animation")

        #         self.ui.loadAbcFileBtn.clicked.connect(self.loadAbcFile)
        self.ui.loadAnimFileBtn.clicked.connect(self.loadAnimFile)
        self.ui.loadJsonFileBtn.clicked.connect(self.loadJsonFile)

        self.ui.okBtn.clicked.connect(self.okBtnClick)

        self.ui.cancelBtn.clicked.connect(self.closeBtnClick)

    def radioBtnState(self):
        if self.ui.crdRadioBtn.isChecked():
            self.initializeDataSet(0, "Crowd")
        elif self.ui.aniRadioBtn.isChecked():
            self.initializeDataSet(0, "Animation")
        elif self.ui.mcpRadioBtn.isChecked():
            self.initializeDataSet(0, "Mocap")

    def initializeDataSet(self, startIndex, category):
        # tag 1 tier setting
        self.tagData = self.dbPlugin.getTagData(category)
        self.tagDict = {}
        for tag1Tier in self.tagData:
            if not self.tagDict.has_key(tag1Tier["tag_tier1"]):
                self.tagDict[tag1Tier["tag_tier1"]] = {}

            if not tag1Tier.has_key("tag_tier2"):
                continue
            elif not self.tagDict[tag1Tier["tag_tier1"]].has_key(tag1Tier["tag_tier2"]):
                self.tagDict[tag1Tier["tag_tier1"]][tag1Tier["tag_tier2"]] = list()

            if not tag1Tier.has_key("tag_tier3"):
                continue
            else:
                self.tagDict[tag1Tier["tag_tier1"]][tag1Tier["tag_tier2"]].append(tag1Tier["tag_tier3"])

        self.ui.Title1ComboBox.clear()

        for key in self.tagDict.keys():
            item = QtWidgets.QTreeWidgetItem()
            item.setText(0, key)
            for key2 in self.tagDict[key].keys():
                item2 = QtWidgets.QTreeWidgetItem(item)
                item2.setText(0, key2)
                for key3 in self.tagDict[key][key2]:
                    item3 = QtWidgets.QTreeWidgetItem(item2)
                    item3.setText(0, key3)

            self.ui.Title1ComboBox.addItem(key)

        self.ui.Title1ComboBox.setCurrentIndex(startIndex)

    def currentTitle1IndexChange(self, index):
        try:
            self.ui.Title2ComboBox.clear()

            self.ui.Title2ComboBox.addItems(self.tagDict[self.ui.Title1ComboBox.currentText()].keys())
        except:
            pass

    def currentTitle2IndexChange(self, index):
        try:
            self.ui.Title3ComboBox.clear()

            self.ui.Title3ComboBox.addItems(self.tagDict[self.ui.Title1ComboBox.currentText()][self.ui.Title2ComboBox.currentText()])
        except:
            pass

    # Load Maya File Path
    def getOpenFile(self, titleCaption, startDirPath, exrCaption):
        fileName = ""
        if not "Side" in Qt.__binding__:
            fileName = QtWidgets.QFileDialog.getOpenFileName(self, titleCaption, startDirPath, exrCaption)
        else:
            fileName = cmds.fileDialog2(fileMode=1,
                                        caption=titleCaption,
                                        okCaption="Load",
                                        startingDirectory=startDirPath,
                                        fileFilter=exrCaption)
            if fileName == None:
                return None
            fileName = str(fileName[0])
        return fileName

    def loadAnimFile(self):
        titleCaption = "Load Anim File"
        exrCaption = "Anim File (*.anim)"

        dirPath = "/show"

        fileName = self.getOpenFile(titleCaption, dirPath, exrCaption)

        if self.ui.jsonPathEdit.text() == "":
            splitFileName = fileName.split('.anim')
            self.ui.jsonPathEdit.setText(splitFileName[0] + ".json")

        self.ui.animPathEdit.setText(fileName)

    def loadJsonFile(self):
        titleCaption = "Load Json File"
        exrCaption = "Json File (*.json)"

        dirPath = "/show"

        fileName = self.getOpenFile(titleCaption, dirPath, exrCaption)

        if self.ui.jsonPathEdit.text() == "":
            splitFileName = fileName.split('.json')
            self.ui.jsonPathEdit.setText(splitFileName[0] + ".json")

        self.ui.jsonPathEdit.setText(fileName)

    # ok Btn Click
    def okBtnClick(self):

        itemName = "Is {0}/{1}/{2} correct?".format(self.ui.Title1ComboBox.currentText(),
                                                    self.ui.Title2ComboBox.currentText(),
                                                    self.ui.Title3ComboBox.currentText())
        msg = MessageBox(Message = itemName,
                         Button = ["OK", "CANCEL"],
                         Icon = QtWidgets.QMessageBox.Question,
                         winTitle = "question")

        if msg == "OK":
            if self.parent.importCallDialog() == True:
                self.closeBtnClick()
            else:
                MessageBox(Message = "upload fail..\ntry check error in script editor",
                           Button = ["OK"],
                           Icon = "warning")

    # ok Btn Click
    def closeBtnClick(self):
        self.close()

    # UI getFunction
    def getRestFrame(self):
        return int(self.ui.restFrameLineEdit.text())

    def getStartFrame(self):
        return int(self.ui.startFrameLineEdit.text())