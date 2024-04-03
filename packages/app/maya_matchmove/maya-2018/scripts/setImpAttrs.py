import os
import maya.cmds as cmds

from PySide2 import QtCore, QtGui, QtWidgets

import ui_setImpAttrs
reload(ui_setImpAttrs)


class MainForm(QtWidgets.QWidget):
    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent)
        self.ui = ui_setImpAttrs.Ui_MainWindow()
        self.ui.setupUi(self)

        # get scene path
        scene = cmds.file(q=True, sn=True)
        self.sceneDir = os.path.dirname(scene)
        self.imageDir = self.sceneDir.replace('scenes', 'image')
        self.ui.imageplane_textEdit.setText(self.imageDir)

        self.ui.imageplane_find_pushButton.clicked.connect(self.openDialog_selectImpDir)
        self.ui.ok_pushButton.clicked.connect(self.doIt)

    def doIt(self):
        impPath = self.ui.imageplane_textEdit.toPlainText()

        if not impPath:
            self.messagePopup('select impPath plz')
            return

        cameras = cmds.ls(type='camera')

        for img in os.listdir(impPath):
            filename = img.split('.')[0]

            for cam in cameras:
                if filename[1:] in cam:
                    trans = cmds.listRelatives(cam)[0]
                    imp = cmds.listRelatives(trans)[0]
                    print cam, trans, imp, os.path.join(impPath, img)
                    
                    cmds.setAttr('%s.nearClipPlane' % cam, 0.1)
                    cmds.setAttr('%s.displayOnlyIfCurrent' % imp, True)
                    cmds.setAttr('%s.imageName' % imp, os.path.join(impPath, img), type="string")
                    cmds.setAttr('%s.depth' % imp, 1000)

        self.messagePopup('complate!!!')

    def openDialog_selectImpDir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Find Imageplane Directory", self.imageDir, QtWidgets.QFileDialog.ShowDirsOnly)

        if path:
            self.ui.imageplane_textEdit.setText(path)

    def messagePopup(self, msg):
        QtWidgets.QMessageBox.information(self, 'setAttr Imageplane', msg, QtWidgets.QMessageBox.Ok)
