# -*- coding: utf-8 -*-
####################################################
########## coding by RND youkyoung.kim #############
####################################################
import os, sys
# import Qt
# from Qt import QtGui
# from Qt import QtWidgets
# from Qt import QtCore

from PySide2 import QtWidgets, QtCore, QtGui

import historyAction
from spanner2_ui_saveDevel import saveDev_Ui_Form
from spanner2_ui_savePub import savePub_Ui_Form
CURRENTPATH = os.path.dirname( os.path.abspath( __file__ ) )

try:
    import maya.cmds as cmds
    import maya.mel as mel
except ImportError:
    pass

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# Maya Reference, Multi Reference, Import, Multi Import
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
class MayaActions(object):
    def __init__(self, filePath=None, fileName=None):
    # type: (object, object) -> object
        self.filePath = filePath
        self.fileName = fileName

    def getNamespace(self):
    # reference namespace
        if self.filePath.split('/')[3] == 'asset' or '/prev/asset' in self.filePath:
            namespace = self.fileName.split('_')[0]

        else:
            namespace = '_'.join(self.fileName.split('_')[0:2])
            if cmds.namespace( exists=namespace) == True:
                namespace += '_1'
        return namespace

    def referenceAct(self):
    # reference file open
        openfile = os.path.join(self.filePath, self.fileName)
        namespace = self.getNamespace()
        node = cmds.file(openfile, r=True, f=True, gl=True, namespace=namespace, lrd="all", options="v=0", iv=True)

        try:
            topNode = cmds.referenceQuery( node, nodes=True)[0]
            assetName = cmds.getAttr('%s.assetName'%topNode)
            rnNode = cmds.referenceQuery( node, referenceNode=True, topReference=True)
            cmds.lockNode(rnNode, lock=False)
            cmds.addAttr(rnNode, longName ='assetName',niceName='assetName', dataType="string" )
            cmds.setAttr('%s.assetName'%rnNode, assetName, type = 'string')
            cmds.lockNode(rnNode, lock=True)
        except: print '// Error: ref Node set attr error'

    def multiReferenceAct(self):
    # multireference file open
        openfile = os.path.join(self.filePath, self.fileName)
        namespace = self.getNamespace()
        # get number
        mi = multiImportDialog()
        mi.show()
        result = mi.exec_()
        if result == 1:
            num = mi.item
            for i in range(num):
                node = cmds.file(openfile, r=True, f=True, gl=True, namespace=namespace, lrd="all", options="v=0", iv=True)
                try:
                    topNode = cmds.referenceQuery(node, nodes=True)[0]
                    assetName = cmds.getAttr('%s.assetName' % topNode)
                    rnNode = cmds.referenceQuery(node, referenceNode=True, topReference=True)
                    cmds.lockNode(rnNode, lock=False)
                    cmds.addAttr(rnNode, longName='assetName', niceName='assetName', dataType="string")
                    cmds.setAttr('%s.assetName' % rnNode, assetName, type='string')
                    cmds.lockNode(rnNode, lock=True)
                except:
                    print '// Error: ref Node set attr error'

                # import asset
    def importAct(self):
    # import abc file
        openfile = os.path.join(self.filePath, self.fileName)
        if os.path.splitext(str(self.fileName))[-1] == '.abc':
            if not cmds.pluginInfo('AbcImport', l=True, q=True):
                cmds.loadPlugin('AbcImport')
            cmd = 'AbcImport -mode import "%s"'  % openfile
            mel.eval(cmd)
        else:
            mel.eval(
                "file -import -type \"mayaBinary\" -rdn -rpr \"clash\" -options \"v=0;p=17\"  -pr -loadReferenceDepth \"all\" \"%s\"" % (
                    openfile))

    def importNSAct(self):
    # import file
        openfile = os.path.join(self.filePath, self.fileName)
        namespace = self.getNamespace()
        mel.eval("file -import -namespace \"%s\" -ra true -options \"v=0\"  -pr -loadReferenceDepth \"all\" \"%s\"" % (
        namespace, openfile))

    def multiImportAct(self):
    # multi import file
        openfile = os.path.join(self.filePath, self.fileName)
        namespace = self.getNamespace()
        # get number
        mi = multiImportDialog()
        mi.show()
        result = mi.exec_()
        if result == 1:
            num = mi.item
            for i in range(num):
                cmds.file(openfile, i=True, ra=True, ns=(namespace), pr=True, lrd='all', op='v=0')

# import multiple asset by namespace
class multiImportDialog(QtWidgets.QDialog):
    def __init__(self):
        QtWidgets.QDialog.__init__(self)
        ma = MayaActions()
        # ui
        label = QtWidgets.QLabel("Number of Copies:")
        self.numRerference = QtWidgets.QSpinBox()
        self.numRerference.setRange(1, 1000)
        self.ok_btn = QtWidgets.QPushButton("Ok")
        self.close_btn = QtWidgets.QPushButton("Close")
        layout2 = QtWidgets.QHBoxLayout()
        layout2.addWidget(self.ok_btn)
        layout2.addWidget(self.close_btn)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.numRerference)
        layout.addLayout(layout2)
        self.setLayout(layout)
        self.setWindowTitle("Multiple Import")

        #connection
        self.ok_btn.clicked.connect(self.outNum)
        self.close_btn.clicked.connect(self.reject)

    def outNum(self):
        self.item = self.numRerference.value()
        self.accept()

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# Save DEV : version, wipversion, description
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
class SaveDevForm(QtWidgets.QDialog):
    def __init__(self, parent=None, fileName='', filePath=''):
        QtWidgets.QDialog.__init__(self)
        self.SnapShotClass = historyAction.SnapShots()
        self.CommentClass = historyAction.CommentDB()
        self.dsc = ''
        temp = cmds.file(q=True, sn=True)
        self.fileName = temp.split('/')[-1]
        self.filePath = '/'.join(temp.split('/')[0:-1])
        if self.filePath.find('/pub'):
            self.filePath = self.filePath.replace('/pub','/dev')
        self.setWindowFlags(QtCore.Qt.Window)
        self.ui = saveDev_Ui_Form()
        self.ui.setupUi(self)
        self.setWindowTitle('save Devel')
        self.connections()
        self.getDevInfo()
        self.setDescription()
        self.ui.snapshot_checkBox.setChecked(1)

    def connections(self):
        self.ui.dsc_lineEdit.textChanged.connect(self.setDescription)
        self.ui.dsc_lineEdit.editingFinished.connect(self.setDescription)
        self.ui.dsc_lineEdit.textEdited.connect(self.setDescription)
        self.ui.version_spinBox.valueChanged.connect(self.setVer)
        self.ui.wipVersion_spinBox.valueChanged.connect(self.setWip)
        self.ui.buttonBox.accepted.connect(self.saveDev)
        self.ui.buttonBox.rejected.connect(self.reject)

    def getDevInfo(self):
    # current dev info get
        self.ui.fileName_lineEdit.setText(self.fileName)
        names = self.fileName.split('.mb')[0]
        temp = names.split('_')

        if 'asset' in self.filePath:
            self.assetShot = 'asset'
            self.assetName = temp[0]
            self.assetWorkCode = temp[1]
        else:
            self.assetShot = 'prevShot'
            self.shotName = temp[0]
            self.shotWorkCode = temp[1]

        if len(temp) == 5: # description set
            self.description = temp[4]
        else:
            self.description = ''

        if self.fileName.find('_v') > -1: # version set
            self.version = int(names.split('v')[-1].split('_')[0])#version

        if self.fileName.find('_w') > -1: # wipversion set
            dirList = []
            for i in os.listdir(self.filePath):
                if '_v' + str(self.version).zfill(2) in i and i.split('.')[-1] == 'mb' \
                        and len(i.split('_')) > 3:
                    dirList.append(i)
            if dirList:
                lastWipFile = sorted(dirList)[-1].split('.')[0]
                self.wipVersion = int(lastWipFile.split('_')[3].split('w')[1])
            else:
                self.wipVersion = int(temp[3].split('w')[1])
        else:
            self.wipVersion = 0

        # add version number
        self.ui.version_spinBox.setValue(self.version)

        # add wipversion number
        self.ui.wipVersion_spinBox.setValue(self.wipVersion+1)
        self.setDevInfo()

    def setDevInfo(self):
    # next dev info set
        if not self.description == '':
            self.ui.dsc_lineEdit.setText(self.description)
            if self.assetShot == 'asset':
                self.fileName = '%s_%s_v%s_w%s_%s.mb' % (
                    self.assetName, self.assetWorkCode, str(self.upVersion).zfill(2),
                    str(self.wipVersion).zfill(2), self.description)
            if self.assetShot == 'prevShot':
                self.fileName = '%s_%s_v%s_w%s_%s.mb' % (
                    self.shotName, self.shotWorkCode, str(self.upVersion).zfill(2),
                    str(self.wipVersion).zfill(2), self.description)
        else:
            if self.assetShot == 'asset':
                self.fileName = '%s_%s_v%s_w%s.mb' % (
                    self.assetName, self.assetWorkCode, str(self.upVersion).zfill(2),
                    str(self.wipVersion).zfill(2))
            if self.assetShot == 'prevShot':
                self.fileName = '%s_%s_v%s_w%s.mb' % (
                    self.shotName, self.shotWorkCode, str(self.upVersion).zfill(2),
                    str(self.wipVersion).zfill(2))

        self.ui.fileName_lineEdit.setText(self.fileName)

    def setVer(self):
        self.upVersion = self.ui.version_spinBox.value()
        self.setDevInfo()

    def setWip(self):
        self.wipVersion = self.ui.wipVersion_spinBox.value()
        self.setDevInfo()

    def setDescription(self):
        self.description = self.ui.dsc_lineEdit.text()
        self.setDevInfo()

    def saveDev(self):
    # save dev
        self.savePath = os.path.join(self.filePath, self.fileName)
        if os.path.isfile(self.savePath):
            wd = WaringDialog(QtWidgets.QDialog)
            wd.show()
            result = wd.exec_()
            if result == 1:
                pass
            else:
                return

        # save comment to DB
        comment = self.ui.saveDevComment_textEdit.toPlainText()
        if comment != '':
            self.CommentClass.saveDBComment(self.filePath, self.fileName, comment)

        # save file
        cmds.file(rename=str(self.savePath))
        cmds.file(save=True, type='mayaBinary')

        # take snapshot
        if self.ui.snapshot_checkBox.isChecked():
            self.SnapShotClass.takeSnapShot(self.filePath, self.fileName)
        self.accept()

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# Save Pub
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
class SavePubForm(QtWidgets.QDialog):
    def __init__(self, parent=None, fileName='', filePath=''):
        QtWidgets.QDialog.__init__(self)
        self.SnapShotClass = historyAction.SnapShots()
        self.CommentClass = historyAction.CommentDB()
        self.description = ''
        temp = cmds.file(q=True, sn=True)
        self.fileName = temp.split('/')[-1]
        self.filePath = '/'.join(temp.split('/')[0:-1])
        self.setWindowFlags(QtCore.Qt.Window)
        self.ui = savePub_Ui_Form()
        self.ui.setupUi(self)

        self.setWindowTitle('save Publish')
        self.getPubInfo()
        self.setDevInfo()
        self.setDescription()
        self.ui.model_groupBox.setEnabled(0)

        # connection
        self.ui.buttonBox.rejected.connect(self.reject)
        self.ui.buttonBox.accepted.connect(self.savePub)
        self.ui.dsc_lineEdit.textChanged.connect(self.setDescription)
        self.ui.dsc_lineEdit.editingFinished.connect(self.setDescription)
        self.ui.dsc_lineEdit.textEdited.connect(self.setDescription)

    def getPubInfo(self):
    # current pub info get
        self.ui.fileName_lineEdit.setText(self.fileName)
        names = self.fileName.split('.mb')[0]
        self.currentScene = names.split('_')

        if 'asset' in self.filePath:
            self.assetShot = 'asset'
            self.assetName = self.currentScene[0]
            self.assetWorkCode = self.currentScene[1]
        else:
            self.assetShot = 'prevShot'
            self.shotName = self.currentScene[0]
            self.shotWorkCode = self.currentScene[1]

        if len(self.currentScene) == 5: # description set
            self.description = self.currentScene[4]
        else:
            self.description = ''

        # published file
        if self.fileName.find('_v') > -1:
            self.version = int(names.split('v')[-1].split('_')[0])#version

        # dev file
        self.setPubInfo()

    def setPubInfo(self):
    # next pub info set
        if self.assetShot == 'asset':
            if self.description:
                self.fileName = '%s_%s_v%s_%s.mb' % (
                    self.assetName, self.assetWorkCode, str(self.version).zfill(2), self.description)
            else:
                self.fileName = '%s_%s_v%s.mb' % (
                    self.assetName, self.assetWorkCode, str(self.version).zfill(2))

        if self.assetShot == 'prevShot':
            if self.description:
                self.fileName = '%s_%s_v%s_%s.mb' % (
                    self.shotName, self.shotWorkCode, str(self.version).zfill(2), self.description)
            else:
                self.fileName = '%s_%s_v%s.mb' % (
                    self.shotName, self.shotWorkCode, str(self.version).zfill(2))

        self.ui.fileName_lineEdit.setText(self.fileName)

        # dev pub file path
        if self.filePath.find('/pub') > -1:
            self.devPath = self.filePath.replace('/pub','/dev')
        else:
            self.devPath = self.filePath
        self.pubPath = self.filePath.replace('/dev','/pub')

    def setDevInfo(self):
    # next dev info set
        lastWipFile = ''
        path = self.devPath
        # add wipversion number
        dirList = []
        for i in os.listdir(path):
            if '_v' + str(self.version).zfill(2) in i and i.split('.')[-1] == 'mb' and i.find('_w') > -1:
                dirList.append(i)
        if dirList:
            lastWipFile = sorted(dirList)[-1].split('.')[0]
            wipVersion = int(lastWipFile.split('w')[-1].split('_')[0])#wipversion
        else:
            wipVersion = 0

        nextWipVer = int(wipVersion) + 1

        # description exists:
        if not self.description == '':
            if self.assetShot == 'asset':
                self.nextDev = '%s_%s_v%s_w%s_%s.mb' % (
                    self.assetName, self.assetWorkCode, str(self.version).zfill(2),
                    str(nextWipVer).zfill(2), self.description)
            if self.assetShot == 'prevShot':
                self.nextDev = '%s_%s_v%s_w%s_%s.mb' % (
                    self.shotName, self.shotWorkCode, str(self.version).zfill(2),
                    str(nextWipVer).zfill(2), self.description)
            self.ui.dsc_lineEdit.setText(self.description)
        else:
            if self.assetShot == 'asset':
                self.nextDev = '%s_%s_v%s_w%s.mb' % (
                    self.assetName, self.assetWorkCode, str(self.version).zfill(2),
                    str(nextWipVer).zfill(2))
            if self.assetShot == 'prevShot':
                self.nextDev = '%s_%s_v%s_w%s.mb' % (
                    self.shotName, self.shotWorkCode,
                    str(self.version).zfill(2), str(nextWipVer).zfill(2))
        self.ui.nextDev_lineEdit.setText(self.nextDev)

    def setDescription(self):
        self.description = self.ui.dsc_lineEdit.text()
        self.setPubInfo()
        self.setDevInfo()

    def savePub(self):
        # publish path
        self.saveNextDevPath = os.path.join(self.devPath, self.nextDev)
        self.savePubPath = os.path.join(self.pubPath, self.fileName)
        self.workCode = self.savePubPath.split('/')[6]
        self.taskType = self.savePubPath.split('/')[3]

        if os.path.isfile(self.savePubPath):
            # show exist error!
            wd = WaringDialog(QtWidgets.QDialog)
            wd.show()
            result = wd.exec_()
            if result == 1:
                print 'save Pub'
            else:
                return
        # save Dev : check whether to save dev file
        devChecking = self.ui.saveDev_groupBox.isChecked()
        if devChecking == True:
            cmds.file(rename=str(self.saveNextDevPath))
            cmds.file(save=True, type='mayaBinary')
        else:
            print 'save Current File'
            cmds.file(save=True, type='mayaBinary')

        if not os.path.exists(os.path.dirname(self.savePubPath)):
            os.makedirs(os.path.dirname(self.savePubPath))

        cmds.file(rename=str(self.savePubPath))
        cmds.file(save=True, type='mayaBinary')

        # take snapshot
        self.SnapShotClass.takeSnapShot(self.pubPath, self.fileName)

        # set comment
        self.comment = self.ui.savePubComment_textEdit.toPlainText()
        if self.comment != '':
            self.CommentClass.saveDBComment(self.pubPath, self.fileName, self.comment)
        self.accept()

# warnig Dialog
class WaringDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, text=''):
        QtWidgets.QDialog.__init__(self)
        # ui
        label = QtWidgets.QLabel("File exists. Are you sure to save?\n")
        if text:
            label = QtWidgets.QLabel(text)
        self.ok_btn = QtWidgets.QPushButton("Ok")
        self.close_btn = QtWidgets.QPushButton("Close")
        layout2 = QtWidgets.QHBoxLayout()
        layout2.addWidget(self.ok_btn)
        layout2.addWidget(self.close_btn)
        layout = QtWidgets.QGridLayout()
        layout.addWidget(label)
        layout.addLayout(layout2,3,0)
        self.setLayout(layout)
        self.setWindowTitle("Warning")

        #connection
        self.ok_btn.clicked.connect(self.accept)
        self.close_btn.clicked.connect(self.reject)
