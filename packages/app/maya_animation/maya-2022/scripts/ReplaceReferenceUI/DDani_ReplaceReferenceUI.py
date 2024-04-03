__author__ = 'gyeongheon.jeong'

import os

import maya.cmds as cmds
import maya.OpenMayaUI as mui

from ui.ui_ReferenceReplace import Ui_MainWindow
import DDani_ListReferences

# Qt Module
from PySide2 import QtWidgets
from shiboken2 import wrapInstance

TITLE = "Replace Reference v1.1"


def mayaWindow():
    ptr = mui.MQtUtil.mainWindow()
    if ptr is not None:
        return wrapInstance(long(ptr), QtWidgets.QWidget)


class DDani_replaceReferencesUI(QtWidgets.QDialog):
    def __init__(self, parent = mayaWindow()):
        QtWidgets.QDialog.__init__(self, parent)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.initGUI()
        self.connectSlot()

    def initGUI(self):
        self.ui.DDaniRp_refList.clear()
        self.ui.DDaniRp_fileList.clear()
        self.refList, refPath = DDani_ListReferences.ListReferences()
        self.ui.DDaniRp_refList.addItems(self.refList)
        showTypeList = DDani_ListReferences.listReferenceDirs("/show")
        self.ui.Show_CmbBox.addItems(showTypeList)


    def connectSlot(self):
        self.ui.DDaniRp_refList.itemSelectionChanged.connect(self.reloadPath)
        self.ui.Show_CmbBox.currentIndexChanged.connect(self.reloadAssetList)
        self.ui.Asset_CmbBox.currentIndexChanged.connect(self.reloadWorkcode)
        self.ui.Workcode_CmbBox.currentIndexChanged.connect(self.reloadFiles)
        self.ui.mbFileList_CmbBox.currentIndexChanged.connect(self.editPathText)
        self.ui.DDaniRp_DoBtn.clicked.connect(self.DoIt)

    def reloadAssetList(self):
        try:
            self.showPath = os.sep.join( [ str( "/show" ), str( self.ui.Show_CmbBox.currentText() + "/_3d/asset" ) ] )
            self.AssetList = DDani_ListReferences.listReferenceDirs(self.showPath)

            self.ui.Asset_CmbBox.clear()
            self.ui.Asset_CmbBox.addItem("-- Asset --")
            self.ui.Asset_CmbBox.addItems(self.AssetList)
        except OSError:
            pass

    def reloadWorkcode(self):
        try:
            self.AssetPath = os.sep.join( [ str( self.showPath ), str( self.ui.Asset_CmbBox.currentText() ) ] )
            self.WorkcodeList = DDani_ListReferences.listReferenceDirs(self.AssetPath)
            self.ui.Workcode_CmbBox.clear()
            self.ui.Workcode_CmbBox.addItem("-- Workcode --")
            self.ui.Workcode_CmbBox.addItems(self.WorkcodeList)
        except OSError:
            pass

    def reloadFiles(self):
        try:
            self.ui.mbFileList_CmbBox.clear()
            self.CustomAssetFilePath = os.sep.join( [str( self.AssetPath ), str( self.ui.Workcode_CmbBox.currentText() ), "scenes"] )

            self.mbFileList = DDani_ListReferences.listReferenceDirFiles(self.CustomAssetFilePath)
            self.ui.mbFileList_CmbBox.addItems(self.mbFileList)
        except OSError:
            pass

    def reloadPath(self):
        self.ui.DDaniRp_fileList.clear()
        if self.ui.DDaniRp_refList.selectedItems():
            self.RefNode = str( self.ui.DDaniRp_refList.selectedItems()[0].text() ).split(" ")[0]
            self.fileName = cmds.referenceQuery(self.RefNode, filename=True )
            self.filePath = os.sep.join( self.fileName.split(os.sep)[:-1] )
            fileList = DDani_ListReferences.listReferenceDirFiles(self.filePath)
            self.ui.DDaniRp_fileList.addItems(fileList)

            path_split_list = self.filePath.split("/")
            showName = path_split_list[ path_split_list.index("show") + 1 ]
            assetName = path_split_list[ path_split_list.index("asset") + 1 ]

            showNameIndex = self.ui.Show_CmbBox.findText( showName )
            self.ui.Show_CmbBox.setCurrentIndex( showNameIndex )
            assetNameIndex = self.ui.Asset_CmbBox.findText( assetName )
            self.ui.Asset_CmbBox.setCurrentIndex( assetNameIndex )



    def editPathText(self):
        self.NewNameSpace = str(self.ui.mbFileList_CmbBox.currentText()).split("_")[0]
        self.OldNameSpace = cmds.file(self.fileName, q = 1, namespace = 1)

        self.ui.OldNameSpace_lineEdit.setText(self.OldNameSpace)

        if str(self.ui.Workcode_CmbBox.currentText()) == "rig":
            self.ui.NewNameSpace_lineEdit.setText(self.NewNameSpace)
        else:
            self.ui.NewNameSpace_lineEdit.setText(self.OldNameSpace)

        try:
            self.ui.AssetPath_Txt.clear()
            self.ui.AssetPath_Txt.setText(self.CustomAssetFilePath)
        except AttributeError:
            pass

    def DoIt(self):
        refFilePathList = list()
        for item in self.ui.DDaniRp_refList.selectedItems():
            item_RefName = str( item.text() ).split(" ")[1]
            refFilePathList.append( self.filePath + os.sep + item_RefName )

        if self.ui.ReferencingType_Tab.currentIndex() == 0:
            newFilePath = self.filePath + os.sep + self.ui.DDaniRp_fileList.currentItem().text()

            for fileName in refFilePathList:
                DDani_ListReferences.ReplaceReferences( fileName, str( newFilePath ) )
            self.initGUI()

        elif self.ui.ReferencingType_Tab.currentIndex() == 1:
            customAssetPath = os.sep.join( [ str( self.CustomAssetFilePath), str( self.ui.mbFileList_CmbBox.currentText() ) ] )
            for fileName in refFilePathList:
                DDani_ListReferences.ReplaceReferences( fileName, str(customAssetPath))
            self.initGUI()

        OldNameSpace = str(self.ui.OldNameSpace_lineEdit.text())
        NewNameSpace = str(self.ui.NewNameSpace_lineEdit.text())

        if NewNameSpace != OldNameSpace:
            cmds.namespace( rename=(OldNameSpace, NewNameSpace) )
            cmds.namespace( rm = OldNameSpace )


def showUI():
    mainWidget = DDani_replaceReferencesUI()
    mainWidget.setWindowTitle(TITLE)
    mainWidget.show()
