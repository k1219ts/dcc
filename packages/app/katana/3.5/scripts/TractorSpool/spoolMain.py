#coding=utf8

from PyQt5 import QtWidgets
from spoolMainUI import Ui_Dialog
import os

class SpoolMain(QtWidgets.QDialog):
    def __init__(self, renderNode, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.renderNode = renderNode

        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        # define
        self.configPath = "/backstage/apps/Tractor/{}_config/2.3"

        # Query Tractor Information
        self.engineIPParm = renderNode.getParameter('farmSettings.Tractor.engine')
        self.projectsParm = renderNode.getParameter('farmSettings.Tractor.projects')
        self.tagsParm = renderNode.getParameter('farmSettings.Tractor.tags')

        ipIndex = self.ui.ipCombo.findText(self.engineIPParm.getValue(0))

        if ipIndex == -1:
            ipIndex = 0

        # Set
        self.ui.ipCombo.setCurrentIndex(ipIndex)

        # Signal
        self.ui.ipCombo.currentIndexChanged.connect(self.engineIPChanged)
        self.ui.limitTagCombo.currentIndexChanged.connect(self.limitTagChanged)
        self.ui.submitBtn.clicked.connect(self.submitBtnClicked)

        self.engineIPChanged(ipIndex)

        tagIndex = self.ui.limitTagCombo.findText(self.tagsParm.getValue(0))
        if tagIndex != -1:
            self.ui.limitTagCombo.setCurrentIndex(tagIndex)

        projectIndex = self.ui.projectCombo.findText(self.projectsParm.getValue(0))
        if projectIndex != -1:
            self.ui.projectCombo.setCurrentIndex(projectIndex)
        else:
            self.ui.projectCombo.setCurrentIndex(0)


    def engineIPChanged(self, index):
        engineIP = self.ui.ipCombo.currentText()
        configPath = self.configPath.format(engineIP.split('.')[-1])
        limitsConfig = os.path.join(configPath, 'limits.config')

        self.limitTags = self.getLimitTags(limitsConfig)

        self.ui.limitTagCombo.clear()
        self.ui.limitTagCombo.addItems(self.limitTags)

    def limitTagChanged(self, index):
        self.ui.projectCombo.clear()

        if self.ui.limitTagCombo.currentText() == "":
            return

        prjList = self.limitTags[self.ui.limitTagCombo.currentText()]['Shares']
        prjList.pop('default')
        self.ui.projectCombo.addItems(prjList)

    def getLimitTags(self, limitsConfig):
        with open(limitsConfig, 'r') as f:
            data = f.read()
            originalLimits = eval(data)

            return originalLimits['Limits']

    def submitBtnClicked(self):
        self.engineIPParm.setValue(str(self.ui.ipCombo.currentText()), 0)
        self.projectsParm.setValue(str(self.ui.projectCombo.currentText()), 0)
        self.tagsParm.setValue(str(self.ui.limitTagCombo.currentText()), 0)

        self.accept()