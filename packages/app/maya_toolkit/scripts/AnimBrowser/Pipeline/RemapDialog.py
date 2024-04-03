import os
# import Qt
# from Qt import QtGui
# from Qt import QtCore
# from Qt import QtWidgets
from PySide2 import QtWidgets, QtCore, QtGui

# if "Side" in Qt.__binding__:
import maya.cmds as cmds

from RemapDialogUI import Ui_Form
import AnimBrowser.retargetting.bvhExporter as bvhExporter
import dxConfig
DBIP = dxConfig.getConf("DB_IP")
import Zelos
import datetime
import getpass
from pymongo import MongoClient
import json

CURRENTDIR = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
retargetPath = "/dexter/Cache_DATA/RND/daeseok/AnimBrowser_retargetting"
class RemapDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, sourcefile="", targetfile="", target_ns="" ):
        '''
        source joint to target joint remapping dialog
        :param parent: parent fot QtWidget
        :param sourcefile: retarget by source file(.bvh)
        :param targetfile: retarget by target file(.bvh)
        :param target_ns:  namespace of target ('NameSpace':NodeName)
        '''
        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowFlags(QtCore.Qt.Window)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.styleSetting()
        self.connections()

        self.sourcefile = sourcefile
        self.targetfile = targetfile
        self.arachneSourceSKel = None
        self.targetSkel = None
        self.target_ns = target_ns
        # self.jointList = []
        self.remapData = {}
        # self.remapInverse = {}
        self.presets = {}
        self.status = False
        self.ui.targetNs_lineEdit.setText(self.target_ns)
        self.loadPreset() # load preset data

        # initialize remap
        # read source bvh file to get joint list.
        # first, source setup from bvh file.
        self.arachneSourceSKel = Zelos.skeleton()
        self.arachneSourceSKel.load(self.sourcefile)
        self.recurSourceSetup(self.arachneSourceSKel.getRoot())
        # set joints to list widget
        self.targetRemapJointSetup()
        # setup namespace update
        self.updateTargetNS()

    def styleSetting(self):
        self.ui.matching_treeWidget.setRootIsDecorated(True)
        self.resize(1000,1500)
        self.ui.matching_treeWidget.header().resizeSection(0, 600)
        self.ui.matching_treeWidget.header().resizeSection(1, 30)
        self.ui.matching_treeWidget.header().resizeSection(3, 40)
        listWidget_style = '''
                       QListWidget { background: rgb(80, 80, 80); color: white; font: 12px; }
                       QListWidget::item { padding: 7 10 5 10 px; margin: 0px; border: 0 px}
                       QListWidget::item:selected{background: rgb(50, 155, 255, 150);}
                       '''
        self.ui.preset_listWidget.setStyleSheet(listWidget_style)
        radiobutton_style = """
            QRadioButton { background-color: #494949; }
        """
        self.ui.biped_radioButton.setStyleSheet(radiobutton_style)
        self.ui.quad_radioButton.setStyleSheet(radiobutton_style)

    def connections(self):
        self.ui.ok_pushButton.clicked.connect(self.okBtnClick)
        self.ui.cancel_pushButton.clicked.connect(self.closeBtnClick)
        self.ui.preset_listWidget.itemDoubleClicked.connect(self.setPreset) # click -> double Click
        self.ui.findSource_lineEdit.editingFinished.connect(self.findSource) # find source joint name in tree widget
        self.ui.findTarget_lineEdit.editingFinished.connect(self.findTarget) # find target joint name in tree widget
        self.ui.targetNs_lineEdit.editingFinished.connect(self.updateTargetNS) # update target by namespace

        self.ui.preset_listWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.preset_listWidget.customContextMenuRequested.connect(self.presetMenu)
        self.ui.biped_radioButton.clicked.connect(self.setBipedQuadPreset)
        self.ui.quad_radioButton.clicked.connect(self.setBipedQuadPreset)

    def recurSourceSetup(self, arachneJoint):
        '''
        recursive execute to remapData base setup.
        :param arachneJoint:
        :return:
        '''
        jointName = str(self.arachneSourceSKel.jointName(arachneJoint))

        if '_JNT_End' not in jointName:  # not (joint end from bvh)
            # self.remapData[jointName] = []
            # self.jointList.append(jointName)
            for i in range(self.arachneSourceSKel.childNum(arachneJoint)):
                child = self.arachneSourceSKel.childAt(arachneJoint, i)
                self.recurSourceSetup(child)

    def targetRemapJointSetup(self):
        # clear tree widget
        while self.ui.matching_treeWidget.topLevelItemCount() > 0:
            treeitem = self.ui.matching_treeWidget.topLevelItem(0)
            if treeitem:
                if treeitem.childCount() > 0:
                    treeitem.takeChildren()
            self.ui.matching_treeWidget.takeTopLevelItem(0)

        # first root setup, next recursively data parsing
        self.recurSetupTreeWidget(self.sourceSkel.getRoot(), self.ui.matching_treeWidget)

    def recurSetupTreeWidget(self, arachneJoint=None, parentItem=None):
        jointName = str(self.sourceSkel.jointName(arachneJoint))
        if '_JNT_End' not in jointName:
            item = MatchingItem(parentItem) # custom treeWidget item class

            item.jointname_lineEdit.setText(jointName) # first, source joint name setup
            if self.remapData.has_key(jointName): # already setup data, ex) preset data
                if self.remapData[jointName]: # has jointName and targetName
                    targetName = self.remapData[jointName][0]
                    if self.target_ns:
                        targetName = "%s:%s" % (self.target_ns, targetName)
                    item.conname_lineEdit.setText(targetName)
            else:
                targetName = "%s:%s" % (self.target_ns, jointName)
                if cmds.objExists(targetName):
                    item.conname_lineEdit.setText(targetName)

            # if item.conname_lineEdit.text():
            #     targetName = item.jointname_lineEdit.text()
            #     self.remapInverse[item.conname_lineEdit.text()] = targetName
            item.setExpanded(True)

            for i in range(self.sourceSkel.childNum(arachneJoint)):
                child = self.sourceSkel.childAt(arachneJoint, i)
                self.recurSetupTreeWidget(child, item)

    def updateTargetNS(self):
        treeitem = self.ui.matching_treeWidget.topLevelItem(0)
        if treeitem:
            self.updatetargetNS_recur(treeitem)

    def updatetargetNS_recur(self, item):
        if item.conname_lineEdit.text():
            ns = self.ui.targetNs_lineEdit.text()

            prevName = item.conname_lineEdit.text()
            if ":" in prevName:
                prevName = prevName.split(":")[-1]
            new = "%s:%s" % (ns, prevName)
            item.conname_lineEdit.setText(new)

        for i in range(item.childCount()):
            childItem = item.child(i)
            self.updatetargetNS_recur(childItem)

    def exportRemapJoints(self):
        root = None
        rootItem = self.ui.matching_treeWidget.topLevelItem(0)
        conname = rootItem.conname_lineEdit.text()
        rootJointName = rootItem.jointname_lineEdit.text()

        if cmds.objExists("|" + rootJointName):
            cmds.confirmDialog(title="Error!",
                               message="already %s" % rootJointName,
                               icon="warning",
                               button=["OK"])
            return

        cmds.namespace(set=':')
        if cmds.objExists(conname):
            root = cmds.duplicate(conname, n=rootJointName, parentOnly=True)[0]
            cmds.parent(root,world=1)
        else:
            return

        for i in range(rootItem.childCount()):
            childItem = rootItem.child(i)
            self.exportRemapJoints_recur(childItem)
        # cmds.namespace(set=':')

        # export #
        exporter = bvhExporter.BVHExporter()
        targetfile = os.path.join(retargetPath, getpass.getuser(), 'proxy.bvh')
        exporter.generate_joint_data(root, targetfile, 0, 0)
        return targetfile

    def exportRemapJoints_recur(self, item):
        if item.conname_lineEdit.text():
            conname = item.conname_lineEdit.text()
            jointName = item.jointname_lineEdit.text()
            joint = cmds.duplicate(conname, n=jointName, parentOnly=True)[0]
            parent = item.parent().jointname_lineEdit.text()
            if cmds.objExists(parent):
                cmds.parent(joint, parent)
            else:
                cmds.confirmDialog(title="Error!",
                                   message="parent error[ parent : %s, own : %s ]" % (parent, joint),
                                   icon="warning",
                                   button=["OK"])
                return

        for i in range(item.childCount()):
            childItem = item.child(i)
            self.exportRemapJoints_recur(childItem)

    '''
    upper refactoring code data
    '''


    def savePreset(self):
        name = self.savePresetDialog()
        if name:
            client = MongoClient(DBIP)
            db = client['animBrowser']
            coll = db['REMAP']
            content = {'name':name,
                        "user": getpass.getuser(),
                        "time": datetime.datetime.now().isoformat(),
                        'content':self.remapData }

            if coll.find({'name':name }).count() > 0:
                result = self.showConfirmDialog()
                if result == 'Yes':
                    coll.update({'name': name},
                                {'$set': content})
                else:
                    self.reject()
            else:
                coll.insert(content)

            print '# RETARGETING : DB inserted ', name

        else:
            result = self.showConfirmDialog()
            if result == 'No':
                return

        # finally
        self.exportRemapJoints()
        self.close()

    def loadPreset(self):
        self.ui.preset_listWidget.clear()
        client = MongoClient(DBIP)
        db = client['animBrowser']
        coll = db['REMAP']
        result = coll.find({})
        if result.count() > 0:
            for i in result:
                self.presets[i['name']] = i['content']
                self.ui.preset_listWidget.addItem(i['name'])
            self.ui.preset_listWidget.sortItems(QtCore.Qt.AscendingOrder)

        return result

    def setPreset(self):
        name = self.ui.preset_listWidget.currentItem().text()
        preset = self.presets[name]
        self.updateConData(preset)
        # clear

        self.setupJoints(self.remapData)

    def setBipedQuadPreset(self):
        filename = '/dexter/Cache_DATA/RND/jeongmin/AnimBrowser/retargetting/jointPreset.json'
        with open(filename) as jsonfile:
            data =  json.load(jsonfile)
            if self.ui.biped_radioButton.isChecked():
                self.remapData = data['biped']
                self.setupJoints(data['biped'])
            elif self.ui.quad_radioButton.isChecked():
                self.remapData = data['quad']
                self.setupJoints(data['quad'])

    def updateConData(self, preset):
        for i in self.remapData:
            if i in preset:
                self.remapData[i] = preset[i]

    def findSource(self):
        toFind = currentText = ""
        toFind = self.ui.findSource_lineEdit.text().lower()
        self.ui.findSource_lineEdit.clear()
        for i in range(self.ui.matching_treeWidget.topLevelItemCount()):
            self.ui.matching_treeWidget.topLevelItem(i).jointname_lineEdit.deselect()
            if toFind:
                currentText = self.ui.matching_treeWidget.topLevelItem(i).jointname_lineEdit.text().lower()
                if currentText.find(toFind) > -1:
                    self.ui.matching_treeWidget.topLevelItem(i).jointname_lineEdit.selectAll()

    def findTarget(self):
        toFind = currentText = ""
        toFind = self.ui.findTarget_lineEdit.text().lower()
        self.ui.findTarget_lineEdit.clear()
        for i in range(self.ui.matching_treeWidget.topLevelItemCount()):
            self.ui.matching_treeWidget.topLevelItem(i).conname_lineEdit.deselect()
            if toFind:
                currentText = self.ui.matching_treeWidget.topLevelItem(i).conname_lineEdit.text().lower()
                if currentText.find(toFind) > -1:
                    self.ui.matching_treeWidget.topLevelItem(i).conname_lineEdit.selectAll()

    ### PRESET RIGHT CLICK MENU ###
    def presetMenu(self, pos):
        pos = pos + (QtCore.QPoint(20,0))
        menu = QtWidgets.QMenu()
        action = menu.addAction('delete')
        action.triggered.connect(self.deletePreset)
        menu.exec_(self.focusWidget().mapToGlobal(pos))

    def deletePreset(self):
        if self.showConfirmDialog():
            presetName = self.ui.preset_listWidget.currentItem().text()
            client = MongoClient(DBIP)
            db = client['animBrowser']
            coll = db['REMAP']
            coll.remove({'name': presetName})
            self.loadPreset()

    ### DIALOGUE ###
    def okBtnClick(self):
        self.savePreset()
        self.accept()

    def closeBtnClick(self):
        self.status = False
        self.reject()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()
        if event.key() == QtCore.Qt.Key_Enter:
            self.okBtnClick()

    def closeEvent(self, event):
        self.reject()

    def showConfirmDialog(self):
        result = cmds.confirmDialog( title='Confirm',
                                                message='Are you sure?',
                                                button=['Yes','No'],
                                                defaultButton='Yes',
                                                cancelButton='No',
                                                dismissString='No' )
        return result

    def savePresetDialog(self):
        text = ""
        result = cmds.promptDialog( title='Save Preset',
		                            message='Enter Name:',
		                            button=['OK', 'Cancel'],
		                            defaultButton='OK',
		                            cancelButton='Cancel',
		                            dismissString='Cancel',
		                            text = text )

        if result == 'OK':
            text = cmds.promptDialog(query=True, text=True)
            return text


### ITEM TO CONNECT JOINT AND CONTROLLERS ###
class MatchingItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent=None ):
        QtWidgets.QTreeWidgetItem.__init__(self, parent)

        self.setChildIndicatorPolicy(QtWidgets.QTreeWidgetItem.DontShowIndicatorWhenChildless)

        self.jointname_lineEdit = NameLineEdit()
        self.jointname_lineEdit.setReadOnly(True)
        self.jointname_lineEdit.setMinimumSize(0,30)
        self.treeWidget().setItemWidget(self, 0, self.jointname_lineEdit)

        self.setText(1, "->")
        self.setTextAlignment(1, QtCore.Qt.AlignCenter)
        self.conname_lineEdit = NameLineEdit()
        self.treeWidget().setItemWidget(self, 2, self.conname_lineEdit)
        self.conname_lineEdit.setMinimumSize(0,30)

        self.conname_lineEdit.textChanged.connect(self.updateDict)
        self.conname_lineEdit.textEdited.connect(self.updateDict)
        self.conname_lineEdit.editingFinished.connect(self.updateDict)
        self.jointname_lineEdit.clicked.connect(self.selectMayaObject)
        self.conname_lineEdit.clicked.connect(self.selectMayaObject)

    def updateDict(self):
        joint_n = self.jointname_lineEdit.text()
        if ':' in joint_n:
            joint_n = joint_n.split(':')[1]
        con_n = self.conname_lineEdit.text()
        if ':' in con_n:
            con_n = con_n.split(':')[1]

        treewidget = self.treeWidget()
        widget = treewidget.parent()
        widget.remapData[joint_n] = [ con_n ]

        # CHECK JOINT NAME
        if cmds.objExists(self.conname_lineEdit.text()):
            self.conname_lineEdit.setStyleSheet('''background:rgb(20,100,20); color:white;''')
            targets = cmds.listRelatives(self.conname_lineEdit.text(), c=1, type='joint')
            if targets:
                if len(targets) == 1:
                    child = self.child(0)
                    if child:
                        child.conname_lineEdit.setText(targets[0])

        else:
            self.conname_lineEdit.setStyleSheet('''background:rgb(40,40,40); color:white;''')

    def selectMayaObject(self, text):
        toFind = text
        if cmds.objExists(toFind):
            cmds.select(toFind)

### ITEMS ###
class PresetItem(QtWidgets.QListWidgetItem):
    def __init__(self, parent=None):
        QtWidgets.QTreeWidgetItem.__init__(self, parent)

        self.jointData = {}

    def setData(self, data):
        self.jointData = data

    def getData(self):
        return self.jointData


class NameLineEdit(QtWidgets.QLineEdit):
    clicked = QtCore.Signal(str)
    def __init__(self, parent=None):
        super(NameLineEdit, self).__init__(parent)

    def focusInEvent(self, e):
        self.selectAll()
        self.clicked.emit(self.text())
