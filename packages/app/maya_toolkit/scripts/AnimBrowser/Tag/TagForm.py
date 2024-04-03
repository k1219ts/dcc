# import Qt
# from Qt import QtCore
# from Qt import QtWidgets
from PySide2 import QtWidgets, QtCore

from TagFormUI import Ui_Form

try:
    from Pipeline.MessageBox import MessageBox
    from Pipeline.MongoDB import MongoDB
except:
    from AnimBrowser.Pipeline.MessageBox import MessageBox
    from AnimBrowser.Pipeline.MongoDB import MongoDB

DBNAME = "inventory"
COLLNAME = "anim_tags"

import getpass
from dxstats import inc_tool_by_user

class MainForm(QtWidgets.QDialog):
    def __init__(self, parent):
        QtWidgets.QDialog.__init__(self, parent)

        if "Side" in Qt.__binding__:
            self.setWindowFlags(QtCore.Qt.Window)

        self.ui = Ui_Form()
        self.ui.setupUi(self)

        mayaWindow = parent
        self.move(mayaWindow.frameGeometry().center() - self.frameGeometry().center())

        self.dbPlugin = MongoDB(DBNAME, COLLNAME)

        self.initializeDataSet(1)

        self.ui.tagTier1ComboBox.currentIndexChanged.connect(self.tag1TierIndexChange)
        self.tag1TierIndexChange(0)

        self.ui.tagTier2ComboBox.currentIndexChanged.connect(self.tag2TierIndexChange)
        self.tag2TierIndexChange(0)

        self.ui.animRadioBtn.clicked.connect(lambda : self.initializeDataSet(1))
        self.ui.mocapRadioBtn.clicked.connect(lambda: self.initializeDataSet(1))
        self.ui.crowdRadioBtn.clicked.connect(lambda: self.initializeDataSet(1))

        self.ui.tier1PushBtn.clicked.connect(self.tag1TierAdd)
        self.ui.tier2PushBtn.clicked.connect(self.tag2TierAdd)
        self.ui.tier3PushBtn.clicked.connect(self.tag3TierAdd)

        self.ui.tagTreeWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.tagTreeWidget.customContextMenuRequested.connect(self.rmbClicked)

        inc_tool_by_user.run('action.AnimBrowser.tagManager', getpass.getuser())

    def initializeDataSet(self, startIndex):
        self.tagCategory = ""
        if self.ui.animRadioBtn.isChecked():
            self.tagCategory = "Animation"
        elif self.ui.mocapRadioBtn.isChecked():
            self.tagCategory = "Mocap"
        else:
            self.tagCategory = "Crowd"

        # tag 1 tier setting
        self.tagData = self.dbPlugin.getTagData(self.tagCategory)
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

        self.ui.tagTreeWidget.clear()
        self.ui.tagTier1ComboBox.clear()

        for key in sorted(self.tagDict.keys()):
            item = QtWidgets.QTreeWidgetItem()
            item.setText(0, key)
            for key2 in sorted(self.tagDict[key].keys()):
                item2 = QtWidgets.QTreeWidgetItem(item)
                item2.setText(0, key2)
                for key3 in sorted(self.tagDict[key][key2]):
                    item3 = QtWidgets.QTreeWidgetItem(item2)
                    item3.setText(0, key3)

            self.ui.tagTreeWidget.addTopLevelItem(item)
            self.ui.tagTier1ComboBox.addItem(key)

        self.ui.tagTier1ComboBox.setCurrentIndex(startIndex)

    def tag1TierIndexChange(self, index):
        self.ui.tagTier2ComboBox.clear()

        if self.ui.tagTier1ComboBox.currentText() != "" and len(self.tagDict[self.ui.tagTier1ComboBox.currentText()].keys()) > 0:
            self.ui.tagTier2ComboBox.addItems(self.tagDict[self.ui.tagTier1ComboBox.currentText()].keys())

    def tag2TierIndexChange(self, index):
        self.ui.tagTier3ComboBox.clear()

        if self.ui.tagTier1ComboBox.currentText() == "" or self.ui.tagTier2ComboBox.currentText() == "":
            return

        if len(self.tagDict[self.ui.tagTier1ComboBox.currentText()][self.ui.tagTier2ComboBox.currentText()]) > 0:
            self.ui.tagTier3ComboBox.addItems(self.tagDict[self.ui.tagTier1ComboBox.currentText()][self.ui.tagTier2ComboBox.currentText()])

    def tag1TierAdd(self):
        if self.ui.tagTier1lineEdit.text() == "":
            MessageBox("input tier1 text")
            return

        dbRecord = {"tag_tier1": self.ui.tagTier1lineEdit.text(),
                    "category": self.tagCategory}
        if not self.dbPlugin.insertTagDocument(dbRecord):
            MessageBox("overlap tier1 text")
            return

        inc_tool_by_user.run('action.AnimTagsManager.insertTag1', getpass.getuser())

        self.initializeDataSet(self.ui.tagTier1ComboBox.currentIndex())

    def tag2TierAdd(self):
        if self.ui.tagTier2lineEdit.text() == "":
            MessageBox("input tier2 text")
            return

        dbRecord = {"tag_tier1": self.ui.tagTier1ComboBox.currentText(),
                    "tag_tier2": self.ui.tagTier2lineEdit.text(),
                    "category": self.tagCategory}
        if not self.dbPlugin.insertTagDocument(dbRecord):
            MessageBox("overlap tier2 text")
            return

        inc_tool_by_user.run('action.AnimTagsManager.insertTag2', getpass.getuser())

        self.initializeDataSet(self.ui.tagTier1ComboBox.currentIndex())

    def tag3TierAdd(self):
        if self.ui.tagTier3lineEdit.text() == "":
            MessageBox("input tier3 text")
            return

        dbRecord = {"tag_tier1": self.ui.tagTier1ComboBox.currentText(),
                    "tag_tier2": self.ui.tagTier2ComboBox.currentText(),
                    "tag_tier3": self.ui.tagTier3lineEdit.text(),
                    "category": self.tagCategory}
        if not self.dbPlugin.insertTagDocument(dbRecord):
            MessageBox("overlap tier3 text")
            return

        inc_tool_by_user.run('action.AnimTagsManager.insertTag3', getpass.getuser())

        self.initializeDataSet(self.ui.tagTier1ComboBox.currentIndex())

    def rmbClicked(self, pos):
        menu = QtWidgets.QMenu(self)

        menu.addAction(u"Remove Tag", self.removeTag)

        menu.popup(self.mapToGlobal(QtCore.QPoint(pos.x() + 15, pos.y() + 85)))

    def removeTag(self):
        currentItem = self.ui.tagTreeWidget.currentItem()
        if currentItem.childCount() > 0:
            msg = MessageBox(Message = "It also clears the included tags.",
                             Button=["Ok", "Cancel"])
            if msg == "Cancel":
                return
        tierList = [currentItem.text(0)]
        self.getTierInfo(currentItem, tierList)

        document = {}
        for index in range(len(tierList)):
            document["tag_tier{0}".format(len(tierList) - index)] = tierList[index]

        self.dbPlugin.removeTagDocument(document)

        self.initializeDataSet(self.ui.tagTier1ComboBox.currentIndex())

    def getTierInfo(self, widgetItem, tierList = []):
        type(widgetItem.parent())
        if widgetItem.parent() is None:
            return

        tierList.append(widgetItem.parent().text(0))
        self.getTierInfo(widgetItem.parent(), tierList = tierList)
