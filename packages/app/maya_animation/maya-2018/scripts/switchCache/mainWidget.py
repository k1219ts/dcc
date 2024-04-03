# -*- coding: utf-8 -*-
__author__ = 'gyeongheon.jeong'

import os
import subprocess
from PySide2 import QtCore, QtGui, QtWidgets
import logging
import maya.OpenMaya as OpenMaya
import maya.cmds as cmds
import modules

reload(modules)

logger = logging.getLogger(__name__)


class SwitchCacheWidget(QtWidgets.QMainWindow):
    # selChangedCallback = None

    def __init__(self, parent=None):
        super(SwitchCacheWidget, self).__init__(parent)
        self.setWindowTitle("Dexter Animation - Switch Cache")

        # main widget
        main_widget = QtWidgets.QWidget(self)
        main_layout = QtWidgets.QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        radioBtnLayout = QtWidgets.QHBoxLayout()
        verticalSpacer1 = QtWidgets.QSpacerItem(20, 30, QtWidgets.QSizePolicy.Expanding,
                                                QtWidgets.QSizePolicy.Minimum)
        emailLabel = QtWidgets.QLabel("gyeongheon.jeong@tactic.com")
        radioBtnLayout.addItem(verticalSpacer1)
        radioBtnLayout.addWidget(emailLabel)
        main_layout.addLayout(radioBtnLayout)

        # table widget
        self.nodeListTableWidget = QtWidgets.QTableWidget()
        self.nodeListTableWidget.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                               QtWidgets.QSizePolicy.Expanding)
        self.nodeListTableWidget.setColumnCount(3)
        self.nodeListTableWidget.setHorizontalHeaderLabels(['Name', 'Type', 'Cache'])
        self.nodeListTableWidgetHeader = self.nodeListTableWidget.horizontalHeader()
        # self.nodeListTableWidgetHeader.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.nodeListTableWidgetHeader.setStretchLastSection(True)
        self.nodeListTableWidgetHeader.resizeSection(0, 380)
        self.nodeListTableWidget.setColumnWidth(1, 100)
        main_layout.addWidget(self.nodeListTableWidget)

        # tab widget
        self.main_tab_widget = QtWidgets.QTabWidget()
        self.main_tab_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                           QtWidgets.QSizePolicy.Minimum)

        """Create tab"""
        create_widget = QtWidgets.QWidget()
        create_layout = QtWidgets.QVBoxLayout(create_widget)

        # start, end, step options
        options_layout = QtWidgets.QHBoxLayout()
        startframeLabel = QtWidgets.QLabel('Start : ')
        self.startFrameLineEdit = QtWidgets.QLineEdit()
        endframeLabel = QtWidgets.QLabel('End : ')
        self.endFrameLineEdit = QtWidgets.QLineEdit()
        verticalSpacer2 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Expanding,
                                                QtWidgets.QSizePolicy.Minimum)
        self.startFrameLineEdit.setFixedWidth(60)
        self.endFrameLineEdit.setFixedWidth(60)
        options_layout.addWidget(startframeLabel)
        options_layout.addWidget(self.startFrameLineEdit)
        options_layout.addWidget(endframeLabel)
        options_layout.addWidget(self.endFrameLineEdit)
        options_layout.addItem(verticalSpacer2)
        create_layout.addLayout(options_layout)
        create_layout.addWidget(self.HLine())
        # --------------------------------------------------------------------------- #

        updateCache_layout = QtWidgets.QHBoxLayout()
        self.updateCheckBox = QtWidgets.QCheckBox('Update Cache')
        self.exportMaterialsCheckBox = QtWidgets.QCheckBox('Export Materials (gpuCache export)')
        updateCache_layout.addWidget(self.updateCheckBox)
        updateCache_layout.addWidget(self.exportMaterialsCheckBox)
        updateCache_layout.addItem(verticalSpacer2)
        create_layout.addLayout(updateCache_layout)
        create_layout.addWidget(self.HLine())


        cacheType_layout = QtWidgets.QHBoxLayout()
        cacheTypeLabel = QtWidgets.QLabel('Cache Type :')
        self.gpuCacheRadioButton = QtWidgets.QRadioButton('Gpu Cache')
        self.meshRadioButton = QtWidgets.QRadioButton('Mesh')
        self.gpuCacheRadioButton.setChecked(True)

        self.radioButtonGroup = QtWidgets.QButtonGroup(self)
        self.radioButtonGroup.addButton(self.gpuCacheRadioButton)
        self.radioButtonGroup.addButton(self.meshRadioButton)

        cacheType_layout.addWidget(cacheTypeLabel)
        cacheType_layout.addWidget(self.gpuCacheRadioButton)
        cacheType_layout.addWidget(self.meshRadioButton)
        cacheType_layout.addItem(verticalSpacer2)
        create_layout.addLayout(cacheType_layout)
        create_layout.addWidget(self.HLine())

        # --------------------------------------------------------------------------- #

        meshType_layout = QtWidgets.QHBoxLayout()
        meshTypeLabel = QtWidgets.QLabel('Mesh Type :')
        self.renderMeshRadioButton = QtWidgets.QRadioButton('Render Mesh')
        self.renderMeshRadioButton.setObjectName('renderMeshes')
        self.midMeshRadioButton = QtWidgets.QRadioButton('Mid Mesh')
        self.midMeshRadioButton.setObjectName('midMeshes')
        self.lowMeshRadioButton = QtWidgets.QRadioButton('Low Mesh')
        self.lowMeshRadioButton.setObjectName('lowMeshes')
        self.lowMeshRadioButton.setChecked(True)

        self.meshTypeRadioButtonGroup = QtWidgets.QButtonGroup(self)
        self.meshTypeRadioButtonGroup.addButton(self.renderMeshRadioButton)
        self.meshTypeRadioButtonGroup.addButton(self.midMeshRadioButton)
        self.meshTypeRadioButtonGroup.addButton(self.lowMeshRadioButton)

        meshType_layout.addWidget(meshTypeLabel)
        meshType_layout.addWidget(self.renderMeshRadioButton)
        meshType_layout.addWidget(self.midMeshRadioButton)
        meshType_layout.addWidget(self.lowMeshRadioButton)
        meshType_layout.addItem(verticalSpacer2)
        create_layout.addLayout(meshType_layout)
        create_layout.addWidget(self.HLine())


        # file browser widgets
        fileBrowse_layout = QtWidgets.QHBoxLayout()
        outpathLabel = QtWidgets.QLabel('Cache Path : ')
        self.outCachePathLineEdit = QtWidgets.QLineEdit()
        self.browseFilePathButton = QtWidgets.QPushButton()
        self.browseFilePathButton.setText('...')
        self.browseFilePathButton.setFixedSize(30, 20)
        self.openDirButton = QtWidgets.QPushButton()
        self.openDirButton.setText('Open')
        self.openDirButton.setFixedSize(30,20)
        fileBrowse_layout.addWidget(outpathLabel)
        fileBrowse_layout.addWidget(self.outCachePathLineEdit)
        fileBrowse_layout.addWidget(self.browseFilePathButton)
        fileBrowse_layout.addWidget(self.openDirButton)
        create_layout.addLayout(fileBrowse_layout)
        create_layout.addWidget(self.HLine())
        # --------------------------------------------------------------------------- #

        # create cache button
        self.createCacheBtn = QtWidgets.QPushButton()
        self.createCacheBtn.setText("Switch")
        self.createCacheBtn.setMinimumHeight(50)
        self.createCacheBtn.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                          QtWidgets.QSizePolicy.Fixed)
        create_layout.addWidget(self.createCacheBtn)

        self.main_tab_widget.addTab(create_widget, "Switch")
        main_layout.addWidget(self.main_tab_widget)

        self.initUI()
        self.connectSignals()
        self.mayaSelectionChangeCallback()

    def HLine(self):
        toto = QtWidgets.QFrame()
        toto.setFrameShape(QtWidgets.QFrame.HLine)
        toto.setFrameShadow(QtWidgets.QFrame.Sunken)
        return toto

    def VLine(self):
        toto = QtWidgets.QFrame()
        toto.setFrameShape(QtWidgets.QFrame.VLine)
        toto.setFrameShadow(QtWidgets.QFrame.Sunken)
        return toto

    def initUI(self):
        minTime = int(cmds.playbackOptions(q=True, min=True))
        maxTime = int(cmds.playbackOptions(q=True, max=True))
        dataPath = self.getPath()['cache']

        self.startFrameLineEdit.setText(str(minTime))
        self.endFrameLineEdit.setText(str(maxTime))
        self.outCachePathLineEdit.setText(str(dataPath))

    def connectSignals(self):
        self.browseFilePathButton.clicked.connect(
            lambda: self.browseOutPath(self.outCachePathLineEdit, isDir=True)
        )
        self.openDirButton.clicked.connect(self.openDir)
        self.createCacheBtn.clicked.connect(self.switch)
        self.exportMaterialsCheckBox.stateChanged.connect(self.forceGpuCache)

    def closeEvent(self, event):
        OpenMaya.MMessage.removeCallback(self.selChangedCallback)
        reply = QtWidgets.QMessageBox.question(self, 'Message', 'Are you sure to quit?',
                                               QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def mayaSelectionChangeCallback(self):
        """Create Maya's selection changed callback
        """
        self.selChangedCallback = OpenMaya.MEventMessage.addEventCallback("SelectionChanged",
                                                                          self.updateSelectionList)

    def forceGpuCache(self):
        checkState = self.exportMaterialsCheckBox.isChecked()
        self.gpuCacheRadioButton.setChecked(checkState)
        for button in self.radioButtonGroup.buttons():
            button.setEnabled(not checkState)

    def openDir(self):
        path = str(self.outCachePathLineEdit.text())
        p = subprocess.Popen('/usr/bin/nautilus {0}'.format(path), shell=True)

    def getPath(self):
        dataPath = modules.getBackstageDataPath()
        cachePath = os.sep.join([dataPath, 'geoCache', 'switchCache'])

        return {'cache': cachePath, 'data':dataPath}

    def browseOutPath(self, lineEdit, isDir=True):
        startPath = modules.getBackstageDataPath()
        if isDir:
            outPath = QtWidgets.QFileDialog.getExistingDirectory(self, 'select directory', startPath)
        else:
            outPath = QtWidgets.QFileDialog.getOpenFileName(self, 'select file')[0]
        if outPath:
            lineEdit.setText(outPath)

    def updateSelectionList(self, *args, **kwargs):
        """Add selection list to table widget
        """
        selectedObject = cmds.ls(sl=True)

        rowCount = 1
        nodeList = list()
        if not selectedObject:
            self.nodeListTableWidget.setRowCount(0)
            return
        for i, item in enumerate(selectedObject):
            node = modules.getRootNode(item, type='dxNode')
            if (not node) and (len(selectedObject) == 1):
                self.nodeListTableWidget.setRowCount(0)
                return
            if (not node) or (node in nodeList):
                continue
            nodeList.append(node)
            self.nodeListTableWidget.setRowCount(rowCount)
            objectType = cmds.objectType(node)
            self.nodeListTableWidget.setItem(
                rowCount - 1, 0, QtWidgets.QTableWidgetItem(node)
            )
            self.nodeListTableWidget.setItem(
                rowCount - 1, 1, QtWidgets.QTableWidgetItem(objectType)
            )
            self.nodeListTableWidget.item(rowCount - 1, 1).setTextAlignment(
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter
            )
            rowCount += 1
        self.nodeListTableWidget.resizeRowsToContents()

        if not nodeList:
            self.nodeListTableWidget.setRowCount(0)

    def getSelectionList(self):
        """Get selection list from table widget

        :return: A list of table widget items
        """
        selection = list()
        rowCount = self.nodeListTableWidget.rowCount()
        for row in range(rowCount):
            item = str(self.nodeListTableWidget.item(row, 0).text())
            selection.append(item)
        return selection

    def getOptions(self):
        filePath = str(self.outCachePathLineEdit.text())

        options = dict()
        options['filepath'] = filePath
        options['nodes'] = self.getSelectionList()
        options['startFrame'] = int(self.startFrameLineEdit.text())
        options['endFrame'] = int(self.endFrameLineEdit.text())
        options['update'] = self.updateCheckBox.isChecked()
        options['toGpuCache'] = self.gpuCacheRadioButton.isChecked()
        options['meshType'] = str(self.meshTypeRadioButtonGroup.checkedButton().objectName())
        options['exportMaterials'] = self.exportMaterialsCheckBox.isChecked()
        return options

    def switch(self):
        """Switch cache to rig, or rig to cache
        """
        options = self.getOptions()
        switchCls = modules.switchCache(options=options)
        switchCls.switch()
