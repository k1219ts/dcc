# import Qt
# from Qt import QtGui
# from Qt import QtCore
# from Qt import QtWidgets

from PySide2 import QtWidgets, QtCore, QtGui

# if "Side" in Qt.__binding__:
import maya.cmds as cmds

from NoneHIKImportDialogUI import Ui_Form
from MessageBox import MessageBox
from MongoDB import MongoDB

DBNAME = "inventory"
COLLNAME = "anim_tags"

class NoneHIKImportDialog(QtWidgets.QDialog):
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

        self.initializeDataSet(0)

        self.ui.okBtn.clicked.connect(self.okBtnClick)

        self.ui.cancelBtn.clicked.connect(self.closeBtnClick)

    def initializeDataSet(self, startIndex):
        # tag 1 tier setting
        self.tagData = self.dbPlugin.getTagData("Animation")
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
        self.ui.Title2ComboBox.clear()

        self.ui.Title2ComboBox.addItems(self.tagDict[self.ui.Title1ComboBox.currentText()].keys())

    def currentTitle2IndexChange(self, index):
        self.ui.Title3ComboBox.clear()
        self.ui.Title3ComboBox.addItems(self.tagDict[self.ui.Title1ComboBox.currentText()][self.ui.Title2ComboBox.currentText()])

    # ok Btn Click
    def okBtnClick(self):

        itemName = "Is {0}/{1}/{2} correct?".format(self.ui.Title1ComboBox.currentText(),
                                                    self.ui.Title2ComboBox.currentText(),
                                                    self.ui.Title3ComboBox.currentText())
        msg = MessageBox(Message = itemName,
                         Button = ["OK", "CANCEL"],
                         Icon = QtWidgets.QMessageBox.Question,
                         winTitle = "question"
                   )

        if msg == "OK":
            if self.parent.importCallDialogNoHIK() == True:
                self.closeBtnClick()
            else:
                MessageBox(Message = "upload fail..\ntry check error in script editor",
                           Button = ["OK"],
                           Icon = "warning")

    # ok Btn Click
    def closeBtnClick(self):
        self.close()
