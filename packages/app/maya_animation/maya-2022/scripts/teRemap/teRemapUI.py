
from PySide2 import QtCore, QtGui, QtWidgets
import maya.cmds as cmds
import maya.mel as mel

_Win = None

def showUI():
    global _Win
    if _Win:
        _Win.close()
    _Win = TeRemapWidget()
    _Win.show()
    _Win.resize(300, 100)
    _Win.move(2000, 500)

class TeRemapWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(TeRemapWidget, self).__init__(parent)
        self.setWindowTitle("TimeEditor Remap")
        main_layout = QtWidgets.QVBoxLayout(self)

        sourceClip_layout = QtWidgets.QHBoxLayout()
        sourceClipLabel = QtWidgets.QLabel('Select Source Clip : ')
        sourceClipLabel.setFixedWidth(150)
        sourceClipLabel.setAlignment(QtCore.Qt.AlignRight)
        self.sourceClipComboBox = QtWidgets.QComboBox()
        self.sourceClipComboBox.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sourceClip_layout.addWidget(sourceClipLabel)
        sourceClip_layout.addWidget(self.sourceClipComboBox)

        source_layout = QtWidgets.QHBoxLayout()
        sourceLabel = QtWidgets.QLabel('Select Anim Source : ')
        sourceLabel.setFixedWidth(150)
        sourceLabel.setAlignment(QtCore.Qt.AlignRight)
        self.sourceComboBox = QtWidgets.QComboBox()
        self.sourceComboBox.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        source_layout.addWidget(sourceLabel)
        source_layout.addWidget(self.sourceComboBox)

        target_layout = QtWidgets.QHBoxLayout()
        targetLabel = QtWidgets.QLabel('Select Target Namespace : ')
        targetLabel.setFixedWidth(150)
        targetLabel.setAlignment(QtCore.Qt.AlignRight)
        self.targetComboBox = QtWidgets.QComboBox()
        self.targetComboBox.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        target_layout.addWidget(targetLabel)
        target_layout.addWidget(self.targetComboBox)

        self.remapButton = QtWidgets.QPushButton("R E M A P")

        main_layout.addLayout(sourceClip_layout)
        main_layout.addLayout(source_layout)
        main_layout.addLayout(target_layout)
        main_layout.addWidget(self.remapButton)

        self.comboBoxAddItems()

        self.remapButton.clicked.connect(self.doit)

    def comboBoxAddItems(self):
        animClips = cmds.ls(type='timeEditorClip')
        animSources = cmds.ls(type='timeEditorAnimSource')
        cmds.namespace(setNamespace=":")
        namespaces = cmds.namespaceInfo(listOnlyNamespaces=True, recurse=True)
        try:
            namespaces.remove('UI')
            namespaces.remove('shared')
        except:
            pass
        self.sourceClipComboBox.addItems(animClips)
        self.sourceComboBox.addItems(animSources)
        self.targetComboBox.addItems(namespaces)

    def doit(self):
        sourceClip = str(self.sourceClipComboBox.currentText())
        animSource = str(self.sourceComboBox.currentText())
        targetNamespace = str(self.targetComboBox.currentText())
        self.remap(sourceClip, animSource, targetNamespace)

    def getClipMembers(self, clipId):
        attrs = cmds.timeEditorClip(clipId, q=True, remappedTargetAttrs=True)
        retVal = list()
        for i in range(0, len(attrs), 2):
            retVal.append(attrs[i + 1])
        return retVal

    def getPopulateAttributes(self, sources, targetNamespace):
        nodes = list()

        for s in sources:
            node = s.split(".")[0]
            if node not in nodes:
                nodes.append(node)

        nodesWithoutNS = [node.split(":")[-1] for node in nodes]

        contents = cmds.namespaceInfo(targetNamespace, listOnlyDependencyNodes=True, recurse=True)
        attributesToPopulate = str()
        if contents:
            for itemContent in contents:
                nameWithoutNS = itemContent.split(":")[-1]
                if nameWithoutNS in nodesWithoutNS:
                    itemAttributes = str()
                    for s in sources:
                        attrNode = s.split(".")[0].split(":")[-1]
                        if attrNode == nameWithoutNS:
                            itemAttributes += " -at " + itemContent + "." + s.split(".")[1]
                    attributesToPopulate += itemAttributes
        if attributesToPopulate == "":
            attributesToPopulate = " -emptySource"

        return attributesToPopulate

    def remap(self, sourceClip, animSource, targetNamespace):
        timeClipSource = animSource
        duration = cmds.getAttr(timeClipSource + ".duration")
        startFrame = cmds.getAttr(sourceClip + '.clip[0].clipStart')

        timeClipSourceAttr = cmds.timeEditorAnimSource(timeClipSource, q=True, targets=True)
        activeComp = cmds.timeEditorComposition(q=True, active=True)
        tracksNodeName = cmds.timeEditorComposition(activeComp, q=True, tracksNode=True)
        trackIndex = cmds.timeEditorTracks(tracksNodeName, e=True, addTrack=-1, trackType=0)
        sources = cmds.timeEditorAnimSource(timeClipSource, q=True, targets=True)

        attributesToPopulate = self.getPopulateAttributes(sources, targetNamespace)

        newClipName = timeClipSource + "_" + targetNamespace
        populateCmd = "timeEditorClip " + attributesToPopulate
        populateCmd += " -track \"" + tracksNodeName + ":{}".format(trackIndex)
        populateCmd += "\" -startTime " + str(startFrame)
        populateCmd += " -duration " + str(duration)
        populateCmd += " \"" + newClipName + "\"" + ";"

        newClipId = mel.eval(populateCmd)

        targetAttrs = self.getClipMembers(newClipId)

        for targetAttr in targetAttrs:
            for sourceAttr in timeClipSourceAttr:
                attrBaseName = sourceAttr.split(":")[-1]
                if attrBaseName in targetAttr:
                    cmds.timeEditorClip(e=True, remapSource=(targetAttr, sourceAttr), clipId=newClipId)

        cmds.timeEditorClip(e=True, animSource=timeClipSource, existingOnly=True, clipId=newClipId)

        for targetAttr in targetAttrs:
            for sourceAttr in timeClipSourceAttr:
                attrBaseName = sourceAttr.split(":")[-1]
                if attrBaseName in targetAttr:
                    cmds.timeEditorClip(e=True, remap=(targetAttr, sourceAttr), clipId=newClipId)

