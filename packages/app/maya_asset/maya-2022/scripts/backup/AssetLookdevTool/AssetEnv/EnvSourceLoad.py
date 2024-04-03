import os

import Qt
import Qt.QtGui as QtGui
import Qt.QtWidgets as QtWidgets
import Qt.QtCore as QtCore

from EnvSourceLoadUI import Ui_Form
from EnvFileTreeWidgetItem import EnvFileTreeWidgetItem

import getpass
from dxstats import inc_tool_by_user

class EnvSourceLoad(QtWidgets.QDialog):
    def __init__(self, parent = None, isZenv = False):
        self.parent = parent
        QtWidgets.QDialog.__init__(self, parent = parent)

        self.isZenv = isZenv

#         self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.Dialog)
        
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        
        showDirPath = QtCore.QDir("/show")
        
        self.ui.showComboBox.currentIndexChanged.connect(self.changedShowComboBox)
        self.ui.AssetComboBox.currentIndexChanged.connect(self.changedAssetComboBox)
        if not self.isZenv:
            self.ui.dataComoboBox.currentIndexChanged.connect(self.changedDataComboBox)
        self.ui.pushButton.clicked.connect(self.addAssetBtnClick)
        
        for dirName in showDirPath.entryInfoList(filters = QtCore.QDir.Dirs | QtCore.QDir.NoDotAndDotDot):
            self.ui.showComboBox.addItem(str(dirName.baseName()))

        self.ui.fileListTreeWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        self.ui.fileListTreeWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.fileListTreeWidget.customContextMenuRequested.connect(self.rmbContextMenu)


    def rmbContextMenu(self):
        menu = QtWidgets.QMenu(self)
        menu.addAction("== select items ==", self.selectItemsContext)
        menu.addAction("== unselect items ==", self.unselectItemsContext)
        menu.exec_(QtGui.QCursor.pos())

    def selectItemsContext(self):
        for item in self.ui.fileListTreeWidget.selectedItems():
            item.loadCheckBox.setCheckState(QtCore.Qt.Checked)

    def unselectItemsContext(self):
        for item in self.ui.fileListTreeWidget.selectedItems():
            item.loadCheckBox.setCheckState(QtCore.Qt.Unchecked)
            
    def changedShowComboBox(self, currentIndex):
        projectPath = "/show/{0}/asset/env".format(self.ui.showComboBox.currentText())
        
        if not os.path.isdir(projectPath):
            self.ui.AssetComboBox.addItem("blank env")
            return
        
        self.ui.AssetComboBox.clear()
        self.ui.AssetComboBox.addItems(os.listdir(projectPath))
        
    def changedAssetComboBox(self, currentIndex):
        if self.isZenv:
            projectPath = "/show/{0}/asset/env/{1}/model/pub/zenv/abc/".format(self.ui.showComboBox.currentText(), self.ui.AssetComboBox.currentText())

            self.ui.fileListTreeWidget.clear()

            if not os.path.isdir(projectPath):
                return

            for sourceFile in sorted(os.listdir(projectPath)):
                itemWidget = EnvFileTreeWidgetItem(parent=self.ui.fileListTreeWidget,
                                                   fileName=sourceFile)
        else:
            projectPath = "/show/{0}/asset/env/{1}/model/pub/data/abc/".format(self.ui.showComboBox.currentText(),
                                                                               self.ui.AssetComboBox.currentText())
        
            self.ui.dataComoboBox.clear()
            self.ui.dataComoboBox.addItems(os.listdir(projectPath))

    def changedDataComboBox(self, currentIndex):
        projectPath = "/show/{0}/asset/env/{1}/model/pub/data/abc/{2}".format(self.ui.showComboBox.currentText(),
                                                                           self.ui.AssetComboBox.currentText(),
                                                                              self.ui.dataComoboBox.currentText())

        self.ui.fileListTreeWidget.clear()

        if not os.path.isdir(projectPath):
            return

        for sourceFile in sorted(os.listdir(projectPath)):
            if os.path.isdir(os.path.join(projectPath, sourceFile)) and sourceFile != "backup":
                itemWidget = EnvFileTreeWidgetItem(parent=self.ui.fileListTreeWidget,
                                                   fileName=sourceFile)

            
    def addAssetBtnClick(self):
        addAssetList = []

        if self.isZenv:
            for index in range(0, self.ui.fileListTreeWidget.topLevelItemCount()):
                if self.ui.fileListTreeWidget.topLevelItem(index).getState() == True:
                    addAssetList.append(self.ui.fileListTreeWidget.topLevelItem(index).getFileName())

            self.parent.addZenvSourceFile(self.ui.showComboBox.currentText(), self.ui.AssetComboBox.currentText(), addAssetList)

            inc_tool_by_user.run('action.LookdevTool.AddZenvAsset.{0}'.format(self.ui.AssetComboBox.currentText()), getpass.getuser())
        else:
            for index in range(0, self.ui.fileListTreeWidget.topLevelItemCount()):
                if self.ui.fileListTreeWidget.topLevelItem(index).getState() == True:
                    addAssetList.append(self.ui.fileListTreeWidget.topLevelItem(index).getFileName())

            self.parent.addEnvSourceFile(self.ui.showComboBox.currentText(),
                                         self.ui.AssetComboBox.currentText(),
                                         self.ui.dataComoboBox.currentText(),
                                         addAssetList)

            inc_tool_by_user.run('action.LookdevTool.AddEnvAsset.{0}'.format(self.ui.AssetComboBox.currentText()),
                                 getpass.getuser())
        
        self.close()
        