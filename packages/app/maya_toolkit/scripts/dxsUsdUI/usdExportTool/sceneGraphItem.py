# QT
from pymodule.Qt import QtWidgets
from pymodule.Qt import QtGui
from pymodule.Qt import QtCore

import glob
import re
import os

import usdCommonSetup

NAME = 0
ACTION = 1
VISIBLE = 2
FILEPATH = 3

class sceneGraphItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent, show, seq, shot, category, nodeInfo, isComp = False, isOnlyZenn = False, exportDisable = True, isOnlyBake = False):
        QtWidgets.QTreeWidgetItem.__init__(self, parent)

        self.exportCheckBox = QtWidgets.QCheckBox()
        self.exportCheckBox.setChecked(True)
        self.exportCheckBox.clicked.connect(self.exportStatus)
        parent.treeWidget().setItemWidget(self, 0, self.exportCheckBox)

        self.versionLineEdit = QtWidgets.QLineEdit()
        self.versionLineEdit.textChanged.connect(lambda: self.overwriteVersionCheck())
        parent.treeWidget().setItemWidget(self, 1, self.versionLineEdit)

        self.availableColor = QtGui.QColor(QtCore.Qt.green)
        self.unavailableColor = QtGui.QColor(QtCore.Qt.red)
        self.disableColor = QtGui.QColor(QtCore.Qt.gray)
        self.onlyZennColor = QtGui.QColor(QtCore.Qt.blue)
        self.fontColor = QtGui.QColor(QtCore.Qt.white)
        self.labelColorSet(self.exportCheckBox, self.fontColor)

        self.isOnlyZenn = isOnlyZenn
        self.isOnlyBake = isOnlyBake

        self.pubPath = "{SHOWDIR}/shot/{SEQ}/{SHOT}/{CATEGORY}".format(
            SHOWDIR=usdCommonSetup.GetShowDir(show), SEQ=seq, SHOT=shot, CATEGORY=category
        )
        self.category = category
        if category == "ani" or category == "zenn" or category == "sim":
            nsLayer, nodeName = nodeInfo[NAME].split(":")
            self.nsLayerPath = os.path.join(self.pubPath, nsLayer)
            print self.nsLayerPath
            if os.path.exists(self.nsLayerPath):
                try:
                    # versionList = sorted(glob.glob('{0}/v*'.format(self.nsLayerPath)))
                    # lastestVersion = os.path.basename(versionList[-1])
                    res = [f for f in sorted(os.listdir(self.nsLayerPath)) if re.search(r'v[0-9]{3}', f)]
                    lastestVersion = res[-1]
                    versionCount = int(lastestVersion[1:])
                except Exception as e:
                    versionCount = 0
                    print e.message
            else:
                versionCount = 0


            if isOnlyZenn and category != "zenn":
                self.versionLineEdit.setText('v%s' % str(versionCount).zfill(3))
            else:
                self.versionLineEdit.setText('v%s' % str(versionCount + 1).zfill(3))
        elif category == "crowd":
            try:
                # versionList = sorted(glob.glob('{0}/v*'.format(self.pubPath)))
                # lastestVersion = os.path.basename(versionList[-1])
                res = [f for f in sorted(os.listdir(self.pubPath)) if re.search(r'v[0-9]{3}', f)]
                lastestVersion = res[-1]
                versionCount = int(lastestVersion[1:])
            except:
                versionCount = 0

            if self.isOnlyBake:
                self.versionLineEdit.setText('v%s' % str(versionCount).zfill(3))
            else:
                self.versionLineEdit.setText('v%s' % str(versionCount + 1).zfill(3))
        else:
            try:
                # versionList = sorted(glob.glob('{0}/v*'.format(self.pubPath)))
                # lastestVersion = os.path.basename(versionList[-1])
                res = [f for f in sorted(os.listdir(self.pubPath)) if re.search(r'v[0-9]{3}', f)]
                lastestVersion = res[-1]
                versionCount = int(lastestVersion[1:])
                print category, versionCount
            except:
                versionCount = 0

            self.versionLineEdit.setText('v%s' % str(versionCount + 1).zfill(3))

        self.exportCheckBox.setText(nodeInfo[NAME])

        if len(nodeInfo) == FILEPATH + 1:
            self.exportCheckBox.setToolTip(nodeInfo[FILEPATH])


        self.parentDisabled = True

        if exportDisable:
            if isComp:
                # if nodeInfo[ACTION] == 1
                # self.setDisable(True)
                pass
            else:
                if category == "set":
                    if nodeInfo[ACTION] == 0:
                        # reference
                        self.exportCheckBox.setStyleSheet("QCheckBox { color: skyblue; font: bold 15px; }")
                    if not nodeInfo[VISIBLE]:
                        self.setItemDisable()
                elif category == "zenn":
                    pass
                elif category == "crowd":
                    if nodeInfo[ACTION] == 0:
                        print "# skel, geom export"
                    elif nodeInfo[ACTION] == 1:
                        print "# Mesh drive export"
                else:
                    if nodeInfo[ACTION] == 0 or not nodeInfo[VISIBLE]:
                        self.setItemDisable()
        else:
            self.setItemDisable()

    def setItemDisable(self):
        self.setDisabled(True)
        # self.setDisable(True)
        self.versionLineEdit.setDisabled(True)
        self.labelColorSet(self.versionLineEdit, self.disableColor)
        self.exportCheckBox.setChecked(False)
        self.exportCheckBox.setDisabled(True)
        self.labelColorSet(self.exportCheckBox, self.disableColor)
        self.exportCheckBox.setStyleSheet("QCheckBox { color: gray; font: bold 15px; }")


    def exportStatus(self, status):
        self.versionLineEdit.setEnabled(self.exportCheckBox.isChecked())

        if self.exportCheckBox.isChecked() == True and self.parentDisabled:
            self.overwriteVersionCheck()
        else:
            self.labelColorSet(self.versionLineEdit, self.disableColor)

    def overwriteVersionCheck(self):
        if self.category == "ani" or self.category == "zenn" or self.category == "sim":
            versionPath = os.path.join(self.nsLayerPath, self.versionLineEdit.text())
        else:
            versionPath = os.path.join(self.pubPath, self.versionLineEdit.text())

        if os.path.exists(versionPath):
            if self.isOnlyZenn and self.category != "zenn":
                self.labelColorSet(self.versionLineEdit, self.onlyZennColor)
                self.overwriteVersion = False
            else:
                self.labelColorSet(self.versionLineEdit, self.unavailableColor)
                self.overwriteVersion = True
        else:
            self.labelColorSet(self.versionLineEdit, self.availableColor)
            self.overwriteVersion = False

    def labelColorSet(self, label, qcolor):
        palette = label.palette()
        palette.setColor(label.foregroundRole(), qcolor)
        label.setPalette(palette)

    def setDisable(self, status):
        self.versionLineEdit.setDisabled(status)
        self.parentDisabled = not status

        if self.exportCheckBox.isChecked() and self.parentDisabled:
            self.overwriteVersionCheck()
        else:
            self.labelColorSet(self.versionLineEdit, self.disableColor)


class categoryItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent, category, checked = True):
        QtWidgets.QTreeWidgetItem.__init__(self, parent)

        self.exportCheckBox = QtWidgets.QCheckBox()
        self.exportCheckBox.setChecked(checked)
        self.exportCheckBox.clicked.connect(self.exportStatus)
        parent.setItemWidget(self, 0, self.exportCheckBox)

        self.exportCheckBox.setText(category)

    def exportStatus(self, status):
        if self.exportCheckBox.isChecked() == True:
            for index in range(self.childCount()):
                self.child(index).setDisable(False)
        else:
            for index in range(self.childCount()):
                self.child(index).setDisable(True)
            # self.labelColorSet(self.versionLineEdit, self.disableColor)

    def labelColorSet(self, label, qcolor):
        palette = label.palette()
        palette.setColor(label.foregroundRole(), qcolor)
        label.setPalette(palette)