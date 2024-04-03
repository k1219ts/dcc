# encoding:utf-8

import os
import logging
import maya.cmds as cmds
from PySide2 import QtGui, QtWidgets, QtCore, QtUiTools
import feModules
reload(feModules)
import aniCommon

logger = logging.getLogger(__name__)

CURRENT_FILE = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_FILE)
UIFILE = os.sep.join([CURRENT_DIR, 'ui', 'fbxExport.ui'])

FBX_VERSION = {
    '2016': 'FBX201600',
    '2014': 'FBX201400'
}


def setup_ui(ui, base_instance=None):
    for member in dir(ui):
        if not member.startswith('__') and member is not 'staticMetObject':
            setattr(base_instance, member, getattr(ui, member))


class FbxExportWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(FbxExportWidget, self).__init__(parent)
        uiFile = QtCore.QFile(UIFILE)
        uiFile.open(QtCore.QFile.ReadOnly)

        loader = QtUiTools.QUiLoader()
        ui = loader.load(uiFile)
        setup_ui(ui, self)

        self.setWindowTitle("Fbx Exporter for Motionbuilder")
        self.fbxVersion_comboBox.addItems(FBX_VERSION.keys())
        self.characterName_lineEdit.setReadOnly(True)
        self.sceneScale_lineEdit.setText("10")
        self.sceneScale_lineEdit.setEnabled(False)
        self.objectList_listWidget.addItems(feModules.FbxExport.getAssemblies())
        self.reload_Btn.setIcon(
            QtGui.QIcon(
                QtGui.QPixmap(os.path.join(CURRENT_DIR, 'ui/icons/refreshBtn.png'))
            )
        )
        self.connectSignals()

    def connectSignals(self):
        self.editScale_checkBox.toggled.connect(self.editScale)
        self.export_Btn.clicked.connect(self.export)
        self.cancel_Btn.clicked.connect(self.close)
        self.objectList_listWidget.itemSelectionChanged.connect(self.objectSelect)
        self.addObject_Btn.clicked.connect(self.addItem)
        self.removeObject_Btn.clicked.connect(self.removeItem)
        self.reload_Btn.clicked.connect(self.reloadListWidget)
        self.addCharacter_Btn.clicked.connect(self.addCharacter)
        self.browse_Btn.clicked.connect(self.browse)

    def reloadListWidget(self):
        logger.debug(u'Reload Object List')
        self.objectList_listWidget.clear()
        self.objectList_listWidget.addItems(feModules.FbxExport.getAssemblies())

    def addCharacter(self):
        selection = cmds.ls(sl=True)
        if len(selection) > 1:
            QtWidgets.QMessageBox.warning(self, '하나만 선택해 주세요.')
            return
        self.characterName_lineEdit.setText(selection[0])

    def editScale(self):
        status = self.editScale_checkBox.isChecked()
        self.sceneScale_lineEdit.setEnabled(status)

    def objectSelect(self):
        items = self.objectList_listWidget.selectedItems()
        cmds.select(clear=True)
        for item in items:
            cmds.select(item.text(), add=True)

    def getListWidgetItems(self):
        items = list()
        for index in xrange(self.objectList_listWidget.count()):
            items.append(self.objectList_listWidget.item(index))
        item_labels = [i.text() for i in items]
        return item_labels

    def addItem(self):
        selections = cmds.ls(sl=True)
        item_labels = self.getListWidgetItems()
        for sel in selections:
            if sel not in item_labels:
                self.objectList_listWidget.addItem(sel)
                logger.debug(u'< {0} > added to list'.format(sel))
            else:
                logger.debug(u'# failed. < {0}> already in the list'.format(sel))

    def removeItem(self):
        items = self.objectList_listWidget.selectedItems()
        for item in items:
            self.objectList_listWidget.takeItem(self.objectList_listWidget.row(item))
            logger.debug(u'< {0} > removed from list'.format(item.text()))

    def browse(self):
        scenePath = cmds.file(q=True, sn=True)
        startPath = str(self.output_lineEdit.text())
        fileName = QtWidgets.QFileDialog.getExistingDirectory(self,
                                                              'Select Folder',
                                                              startPath)
        if len(fileName):
            shotName = aniCommon.getShowShot(scenePath)[1]
            fileName = os.sep.join([fileName, shotName])
            self.output_lineEdit.setText(str(fileName))

    def setOptions(self, cls):
        cls.version = FBX_VERSION[str(self.fbxVersion_comboBox.currentText())]
        cls.character = str(self.characterName_lineEdit.text())
        cls.scale = float(self.sceneScale_lineEdit.text())
        cls.output = str(self.output_lineEdit.text())
        cls.objects = self.getListWidgetItems()

    def export(self):
        exportCls = feModules.FbxExport()
        self.setOptions(exportCls)

        if not os.path.exists(exportCls.output):
            os.mkdir(exportCls.output)

        isFinished = exportCls.export()

        if isFinished:
            QtWidgets.QMessageBox.information(
                self,
                'Finished.',
                'Fbx Export Finished'
            )


def showUI():
    global _win
    try:
        _win.close()
    except:
        pass
    _win = FbxExportWidget()
    _win.show()
    _win.resize(600, 500)
    return _win
