import os, sys
from Katana import UI4, NodegraphAPI
from PyQt5 import QtGui, QtCore, QtWidgets
from . import openPath


class OpenPathTab(QtWidgets.QDialog):
    def __init__(self):
        QtWidgets.QDialog.__init__(self, UI4.App.MainWindow.GetMainWindow())

        self.mainUI()

    def mainUI(self):
        # Create panel widget
        self.setWindowTitle('Open Path Tab')

        self.setMinimumWidth(400)
        self.setMinimumHeight(300)
        # self.setWindowOpacity(0.9)

        # Add Label
        labelStyle = 'font-weight: bold; font-size: 18pt;'
        self.label = QtWidgets.QLabel('Select', self)
        self.label.setStyleSheet(labelStyle)

        # Add ComboBox
        comboBoxStyle = 'font-size: 13pt;'
        self.comboBox = QtWidgets.QComboBox(self)
        self.comboBox.addItem('Texture')
        self.comboBox.addItem('Material')
        self.comboBox.addItem('Asset,Shot')
        self.comboBox.setStyleSheet(comboBoxStyle)
        # self.comboBox.resize(400, 35)

        # Add Button
        openButtonStyle = 'font-size: 11pt;'
        openButton = QtWidgets.QPushButton('Open', self)
        openButton.setStyleSheet(openButtonStyle)
        # openButton.resize(400, 35)

        # Create Layout
        hLayout = QtWidgets.QHBoxLayout()
        hLayout.addStretch(1)
        hLayout.addWidget(self.label)
        hLayout.addStretch(1)

        areawidget = QtWidgets.QWidget()
        vLayout = QtWidgets.QVBoxLayout(areawidget)
        vLayout.addStretch(2)
        vLayout.addLayout(hLayout)
        vLayout.addStretch(2)
        vLayout.addWidget(self.comboBox)
        vLayout.addStretch(2)
        vLayout.addWidget(openButton)
        vLayout.addStretch(2)

        # Create Scroll Area
        scrollarea = QtWidgets.QScrollArea()
        scrollarea.setWidget(areawidget)
        scrollarea.setWidgetResizable(True)

        # Create main Widget
        mainLayout = QtWidgets.QVBoxLayout()
        mainLayout.addWidget(scrollarea)
        self.setLayout(mainLayout)

        # Run
        openButton.clicked.connect(self.onActivated)

    def onActivated(self):
        comboBox_text = self.comboBox.currentText()
        if comboBox_text == 'Texture':
            self.openTexturePath()
        elif comboBox_text == 'Material':
            self.openMaterialPath()
        elif comboBox_text == 'Asset,Shot':
            self.openShotPath()

    # Open Texture Path
    def openTexturePath(self):
        openTexturePath = openPath.OpenTexturePath()
        openTexturePath.textureDoIt()

    # Open Material Path
    def openMaterialPath(self):
        openMaterialPath = openPath.OpenMaterialPath()
        openMaterialPath.materialDoIt()

    # Open Shot Path
    def openShotPath(self):
        openShotPath = openPath.OpenShotPath()
        openShotPath.shotDoIt()
