import sys, os
# import Qt
# from Qt import QtGui
# from Qt import QtCore
# from Qt import QtWidgets
from PySide2 import QtWidgets, QtCore, QtGui

# if "Side" in Qt.__binding__:
import maya.cmds as cmds

from ImportSourceDialogUI import Ui_Form
import AnimBrowser.retargetting.bvhImporter_new as bvhImporter
reload(bvhImporter)
import dxConfig
DBIP = dxConfig.getConf("DB_IP")
import Zelos
import datetime
import getpass
from pymongo import MongoClient
import time
from pprint import pprint
import json

CURRENTDIR = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
class ImportSourceDialog(QtWidgets.QWidget):
    def __init__(self, parent=None, skeleton=None, ns="", retargetfile=""):
        QtWidgets.QWidget.__init__(self, None)
        self.setWindowFlags(QtCore.Qt.Window)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.styleSetting()

        self.skeleton = skeleton
        self.bvhfilepath = ""
        self.animfilepath = ""
        self.target_ns = ns
        self.retargetfile = retargetfile
        self.jointList = []
        self.conData = {}
        self.remapData = {}
        self.presets = {}
        self.status = False
        self.connections()
        self.ui.findTargetNS_lineEdit.setText(self.target_ns)

    def styleSetting(self):
        self.ui.matching_treeWidget.setRootIsDecorated(False)
        self.ui.matching_treeWidget.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.ui.matching_treeWidget.header().resizeSection(0, 30)
        self.ui.matching_treeWidget.header().resizeSection(2, 30)
        self.ui.matching_treeWidget.header().resizeSection(4, 40)
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
        self.ui.jointToFK_radioButton.clicked.connect(self.updateSearchMethod)
        self.ui.jointToIK_radioButton.clicked.connect(self.updateSearchMethod)
        self.ui.ok_pushButton.clicked.connect(self.okBtnClick)
        self.ui.cancel_pushButton.clicked.connect(self.closeBtnClick)
        self.ui.preset_listWidget.itemClicked.connect(self.setPreset)
        self.ui.findSource_lineEdit.editingFinished.connect(self.findSource)
        self.ui.findTarget_lineEdit.editingFinished.connect(self.findTarget)
        self.ui.findTargetNS_lineEdit.editingFinished.connect(self.updateTargetNS)
        self.ui.biped_radioButton.clicked.connect(self.setConPreset)
        self.ui.quad_radioButton.clicked.connect(self.setConPreset)
        self.ui.preset_listWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.preset_listWidget.customContextMenuRequested.connect(self.presetMenu)

    def updateTargetNS(self):
        for i in range( self.ui.matching_treeWidget.topLevelItemCount() ):
            treeitem = self.ui.matching_treeWidget.topLevelItem(i)
            if treeitem:
                if treeitem.conname_lineEdit.text():
                    ns = self.ui.findTargetNS_lineEdit.text()
                    if ns:
                        prev = treeitem.conname_lineEdit.text()
                        if ':' in prev:
                            prev = prev.split(':')[-1]
                        new = ns + ":" + prev
                        treeitem.conname_lineEdit.setText(new)
                    else:
                        prev = treeitem.conname_lineEdit.text()
                        if ':' in prev:
                            prev = prev.split(':')[-1]
                        treeitem.conname_lineEdit.setText(prev)

    def init_Retargetting(self):
        self.loadPreset()
        self.ui.jointToIK_radioButton.setChecked(True)

        # read source bvh file to get joint list
        self.createDefaultSet_recur(self.skeleton.getRoot())
        ns = ""
        if self.target_ns:
            ns = self.target_ns+':'
        cmds.parent(self.rootFKNull, ns+'move_CON')
        # set joints to list widget
        self.setupJoints(self.conData)

    def init_remapRetargetting(self):
        print "init_remapRetargetting"
        root = self.skeleton.getRoot()
        rootname = str(self.skeleton.jointName(root))
        self.loadPreset()
        self.ui.jointToIK_radioButton.setChecked(True)

        # read remapped bvh file to get jont list
        self.createRemapSet_recur(root)
        self.setupJoints(self.conData)


    def updateSearchMethod(self):
        self.jointList = []
        self.conData = {}
        self.createDefaultSet_recur(self.skeleton.getRoot())
        self.setupJoints(self.conData)

    def createDefaultSet_recur(self, joint):
        jointName = str( self.skeleton.jointName(joint) )
        if '_JNT_End' not in jointName:
            conName = ""
            self.jointList.append(jointName)
            self.conData[jointName] = {}

            # find FK controllers by joint name and set key
            if self.ui.jointToFK_radioButton.isChecked():
                conName = self.findFKControllers(jointName)
                if conName:
                    self.conData[jointName]['fk'] = [conName]

            # find IK controllers by joint name and set key
            elif self.ui.jointToIK_radioButton.isChecked():
                #conNull = self.createFKControllerSet(jointName)
                #if joint == self.skeleton.getRoot():
                #    self.rootFKNull = conNull
                conName = self.findIKControllers(jointName)
                if conName:
                    self.conData[jointName]['ik'] = [conName]
                if not conName:
                    pvCon = self.findPVControllers(jointName)
                    if pvCon:
                        self.conData[jointName]['ik'] = [conName]

            for i in range(self.skeleton.childNum(joint)):
                child = self.skeleton.childAt(joint, i)
                self.createDefaultSet_recur(child)

    def createRemapSet_recur(self, joint):
        print "createRemapSet_recur"
        jointName = str(self.skeleton.jointName(joint))
        print "jointName :", jointName
        if '_JNT_End' not in jointName:
            conName = ""
            ### find remap joint name ###
            if self.remapData:
                if jointName in self.remapData:
                    if self.remapData[jointName]:
                        print jointName, " -> ", self.remapData[jointName][0]
                        jointName = self.remapData[jointName][0]

            self.jointList.append(jointName)
            if self.conData.has_key(jointName):
                print "already has key %s" % jointName

            self.conData[jointName] = {}

            # find FK controllers by joint name and set key
            if self.ui.jointToFK_radioButton.isChecked():
                conName = self.findFKControllers(jointName)
                if conName:
                    self.conData[jointName]['fk'] = [conName]

            # find IK controllers by joint name and set key
            elif self.ui.jointToIK_radioButton.isChecked():
                #conNull = self.createRemapFKControllerSet( joint, jointName)
                #if joint == self.skeleton.getRoot():
                #    self.rootFKNull = conNull
                conName = self.findIKControllers(jointName)
                if conName:
                    self.conData[jointName]['ik'] = [conName]
                if not conName:
                    pvCon = self.findPVControllers(jointName)
                    if pvCon:
                        self.conData[jointName]['ik'] = [conName]

            for i in range(self.skeleton.childNum(joint)):
                child = self.skeleton.childAt(joint, i)
                self.createRemapSet_recur(child)

    def setConPreset(self):
        filename = '/dexter/Cache_DATA/RND/jeongmin/AnimBrowser/retargetting/conPreset.json'
        with open(filename) as jsonfile:
            data =  json.load(jsonfile)
            if self.ui.biped_radioButton.isChecked():
                condata = data['biped']
            elif self.ui.quad_radioButton.isChecked():
                condata = data['quad']

            for keyjoint in self.conData.keys():
                if keyjoint in condata:
                    self.conData[keyjoint]['ik'] = condata[keyjoint]

            self.setupJoints(self.conData)


    ### IMPORT ANIMATION TO CONTROLLERS ###
    def importKeys(self):
        print ' # IMPORT SOURCE DIALOG : import anim to CON'
        root = self.skeleton.getRoot()
        rootname = str(self.skeleton.jointName(root))
        self.createRemapFKControllerSet_recur(root, rootname)
        if self.target_ns:
            ns = self.target_ns + ':'
        else:
            ns = ""
        cmds.parent(self.rootFKNull, ns + 'move_CON')

        importer = bvhImporter.BVHImporter()
        importer.isCON = True
        importer.conData = self.conData
        importer.target_ns = self.target_ns
        importer.read_bvh(self.retargetfile)
        cmds.delete(cmds.ls(['proxy*_CON','proxy*_NUL']))

    def setupJoints(self, data):
        image_non = QtGui.QPixmap( os.path.join(CURRENTDIR, 'retargetting/Sign-X01-Red.png') )
        image_exists = QtGui.QPixmap( os.path.join(CURRENTDIR, 'retargetting/Sign-Checkmark01-Green.png') )

        # clear
        while self.ui.matching_treeWidget.topLevelItemCount() > 0:
            self.ui.matching_treeWidget.takeTopLevelItem(0)

        if self.ui.quad_radioButton.isChecked() or self.ui.biped_radioButton.isChecked():
            conType = 'ik'
        else:
            if self.ui.jointToFK_radioButton.isChecked():
                conType = 'fk'
            if self.ui.jointToIK_radioButton.isChecked():
                conType = 'ik'

        print data
        sortedkeys = sorted(data.keys())
        for i in range(len( sortedkeys )):
            item = MatchingItem(self.ui.matching_treeWidget, num=i)
            item.jointname_lineEdit.setText(sortedkeys[i])
            if sortedkeys[i] in data:
                if conType in data[sortedkeys[i]]:
                    conName = data[sortedkeys[i]][conType][0]
                    if conName:
                        item.conname_lineEdit.setText(conName)
                else:
                    item.conname_lineEdit.setText("")

    def savePreset(self):
        name = self.savePresetDialog()
        if name:
            client = MongoClient(DBIP)
            db = client['animBrowser']
            coll = db['PRESET']
            content = {'name':name,
                        "user": getpass.getuser(),
                        "time": datetime.datetime.now().isoformat(),
                        'content':self.conData }

            if coll.find({'name':name }).count() > 0:
                result = self.showConfirmDialog()
                if result == 'Yes':
                    coll.update({'name': name},
                                {'$set': content})
                else:
                    return
            else:
                coll.insert(content)

            print '# IMPORT SOURCE DIALOG : db inserted successfully'

        else:
            result = self.showConfirmDialog()
            if result == 'No':
                return

        # finally
        print '# IMPORT SOURCE DIALOG : Start Retargetting'
        self.close()
        self.importKeys()
        cmds.delete(cmds.ls('proxy*',type='nurbsCurve')+cmds.ls('proxy*:*',type='joint'))

    def loadPreset(self):
        self.ui.preset_listWidget.clear()
        client = MongoClient(DBIP)
        db = client['animBrowser']
        coll = db['PRESET']
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
        self.setupJoints(self.conData)

    def updateConData(self, preset):
        if self.ui.jointToIK_radioButton.isChecked():
            conType = 'ik'

        for i in self.conData:
            if i in preset:
                if preset[i].has_key(conType):
                    self.conData[i][conType] = preset[i][conType]

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

    '''
    CREATE FK CONTROLLERS TO CONNECT FK -> IK
    '''
    def createRemapFKControllerSet_recur(self, joint, jointName):
        name = ""
        if self.target_ns:
            name = self.target_ns+':'+jointName
        else:
            #name = jointName
            if jointName in self.remapData:
                name = self.remapData[jointName][0]

        if name:
            jointName = name.split(':')[-1]
            if '_JNT_End' not in jointName:
                conName = 'proxy_' + jointName.replace('_Skin_','_FK_').replace('_JNT','_CON')
                self.conData[jointName]['proxy_fk'] = [ conName ]
                if not cmds.objExists(conName):
                    con = cmds.circle(n=conName, nr=[1,0,0], d=3, s=8, ch=0)[0]
                    conNull = cmds.group(con, n=conName.replace('_CON','_NUL'))
                    temp = cmds.parentConstraint(name, conNull, mo=0, w=1)
                    cmds.delete(temp)
                    if not joint == self.skeleton.getRoot():
                        parentJoint = self.skeleton.getParent(joint)
                        parentJoint = str(self.skeleton.jointName(parentJoint))
                        parentJoint = self.remapData[parentJoint][0]
                        pCON = 'proxy_' + parentJoint.replace('_Skin_','_FK_').replace('_JNT','_CON')
                        cmds.parent(conNull, pCON)

                    else:
                        self.rootFKNull = conNull

        for i in range(self.skeleton.childNum(joint)):
            child = self.skeleton.childAt(joint, i)
            childname = str(self.skeleton.jointName(child))
            self.createRemapFKControllerSet_recur(child, childname)

    '''
    ### SEARCH CONTROLLERS ###
    '''
    def findFKControllers(self, jointName):
        fkcon = None
        finding = cmds.parentConstraint(jointName, q=1, tl=1 )
        if finding:
            if cmds.nodeType(finding[0]) == 'joint':
                # must be blend joint, find again
                finding = cmds.parentConstraint(finding[0], q=1, tl=1 )
                finding_fk = filter(lambda x: '_FK_' in x, finding ) # fk joint
                if finding_fk:
                    fkcon = cmds.parentConstraint(finding_fk, q=1, tl=1 )[0]
            elif cmds.nodeType(finding[0]) == 'transform' and '_CON' in finding:
                fkcon = finding[0]
        return fkcon

    def findIKControllers(self, jointName):
        ikcon = None
        if self.target_ns:
            jointName = self.target_ns + ':' + jointName
        finding = cmds.parentConstraint(jointName, q=1, tl=1 )
        if finding:
            if cmds.nodeType(finding[0]) == 'joint':
                # must be blend joint, find again
                finding = cmds.parentConstraint(finding[0], q=1, tl=1 )
                if finding:
                    finding_ik = filter(lambda x: '_IK_' in x, finding ) # ik joint
                    if finding_ik:
                        effectorName = cmds.listConnections(finding_ik[0], t=1, type='ikEffector')
                        if effectorName:
                            ikhandle = cmds.listConnections(effectorName[0],type='ikHandle')
                            if ikhandle:
                                point = cmds.listConnections(ikhandle[0],type='pointConstraint', et=1)
                                if point:
                                    ikcon = cmds.pointConstraint(point[0], q=1, tl=1)[0]

            elif cmds.nodeType(finding[0]) == 'transform' and '_CON' in finding[0]:
                ikcon = finding[0]

        return ikcon

    def findPVControllers(self, jointName):
        polcon = None
        if self.target_ns:
            jointName = self.target_ns + ':' + jointName
        effectorName = cmds.listConnections(jointName, type='ikEffector')
        if effectorName:
            ikhandle = cmds.listConnections(effectorName[0],type='ikHandle')
            if ikhandle:
                polevector = cmds.listConnections(ikhandle[0],type='poleVectorConstraint', et=1)
                if polevector:
                    polcon = cmds.poleVectorConstraint(polevector[0], q=1, tl=1)[0]

        return polcon

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
            coll = db['PRESET']
            coll.remove({'name': presetName})
            self.loadPreset()

    ### DIALOGUE ###
    def okBtnClick(self):
        self.savePreset()

    def closeBtnClick(self):
        self.status = False
        self.close()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()
        if event.key() == QtCore.Qt.Key_Enter:
            self.okBtnClick()

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
    def __init__(self, parent=None, num=0 ):
        QtWidgets.QTreeWidgetItem.__init__(self, parent)

        self.setChildIndicatorPolicy(QtWidgets.QTreeWidgetItem.DontShowIndicatorWhenChildless)

        self.setText(0, str(num))
        self.setTextAlignment(0, QtCore.Qt.AlignCenter)
        self.jointname_lineEdit = NameLineEdit()
        self.treeWidget().setItemWidget(self, 1, self.jointname_lineEdit)

        self.setText(2, "->")
        self.setTextAlignment(2, QtCore.Qt.AlignCenter)
        self.conname_lineEdit = NameLineEdit()
        self.treeWidget().setItemWidget(self, 3, self.conname_lineEdit)

        self.exists = QtWidgets.QLabel()
        self.treeWidget().setItemWidget(self, 4, self.exists)

        self.conname_lineEdit.textChanged.connect(self.updateDict)
        self.conname_lineEdit.textEdited.connect(self.updateDict)
        self.jointname_lineEdit.clicked.connect(self.selectMayaObject)
        self.conname_lineEdit.clicked.connect(self.selectMayaObject)

    def updateDict(self):
        joint_n = self.jointname_lineEdit.text()
        con_n = self.conname_lineEdit.text()
        if joint_n in self.treeWidget().parent().conData:
            treewidget = self.treeWidget()
            widget = treewidget.parent()
            widget.conData[joint_n]['fk'] = [con_n]

        if self.treeWidget().parent().ui.jointToIK_radioButton.isChecked():
            if self.treeWidget().parent().conData:
                proxy_fk = 'proxy_' + joint_n.replace('_Skin_','_FK_').replace('_JNT','_CON')
                self.treeWidget().parent().conData[joint_n]['proxy_fk'] = [ proxy_fk ]
                self.treeWidget().parent().conData[joint_n]['ik'] = [ self.conname_lineEdit.text() ]

        if cmds.objExists(con_n):
            self.conname_lineEdit.setStyleSheet('''padding: 3px; background:rgb(20,100,20); color:white;''')
        else:
            self.conname_lineEdit.setStyleSheet('''padding: 3px; background:rgb(40,40,40); color:white;''')

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
