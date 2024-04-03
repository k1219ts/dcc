# coding:utf-8
import sys, os

baseDirPath = "/dexter/Cache_DATA/RND/daeseok/AnimBrowserRepository"
retargetPath = "/dexter/Cache_DATA/RND/jeongmin/AnimBrowser_retargetting"

sys.path.append("/netapp/backstage/pub/lib/zelos/py")
sys.path.append("/netapp/backstage/pub/lib/zelos/lib")

#
# MCD_MODULE = '/netapp/backstage/pub/apps/miarmy2/applications/linux/6.2.05RC/maya/scripts'
# if not MCD_MODULE in sys.path:
#     sys.path.append(MCD_MODULE)

try:
    import McdFunctionModified
except:
    pass
import animBridge.animBridges as animBridge
import studiolibrary
import mutils  # studioLibrary module
import time
import shutil

try:
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-p", "--plugins", dest="plugins",
                      help="", metavar="PLUGINS", default="None")
    parser.add_option("-r", "--root", dest="root",
                      help="", metavar="ROOT")
    parser.add_option("-n", "--name", dest="name",
                      help="", metavar="NAME")
    parser.add_option("-v", "--version", dest="version",
                      help="", metavar="VERSION")
    (options, args) = parser.parse_args()
    eval(options.plugins)
except:
    print "################### studiolibrary warning ###################"

import maya.cmds as cmds
import maya.mel as mel
import Zelos
import dxRigUI

from pymodule.Qt import QtCore
from pymodule.Qt import QtWidgets
import pymodule.Qt as Qt

from FormUI import Ui_Form

from Item.DirItem import DirItem
from Item.AniSubItem import AniSubItem

from Pipeline.HIKImportDialog import HIKImportDialog
from Pipeline.NoneHIKImportDialog import NoneHIKImportDialog
from Pipeline.MocapImportDialog import MocapImportDialog
from Pipeline.CrowdImportDialog import CrowdImportDialog
from Pipeline.MakeMov import MakeMov
from Pipeline.MessageBox import MessageBox
from Pipeline.MongoDB import MongoDB
from Pipeline.TractorPublish import TractorPublish
from Pipeline.ChangeTierDialog import ChangeTierDialog
from Pipeline.ExportSourceDialog import ExportSourceDialog
from Pipeline.ImportSourceDialog import ImportSourceDialog
from Pipeline.RemapDialog import RemapDialog
import AnimBrowser.retargetting.bvhExporter as bvhExporter
import AnimBrowser.retargetting.bvhImporter_new as bvhImporter
import Pipeline.dbConfig as dbConfig

from AniContent.ContentTab import ContentTab

from Tag.TagForm import MainForm as TagForm

import os
import getpass

from dxstats import inc_tool_by_user

DBNAME = "inventory"
COLLNAME = "anim_item"
from pprint import pprint

CURRENTDIR = os.path.dirname(os.path.abspath(__file__))


class MainForm(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)

        self.setWindowFlags(QtCore.Qt.Window)

        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.applyStyleSheet()

        self.dbPlugin = MongoDB(DBNAME, COLLNAME)
        self.tagPlugin = MongoDB(DBNAME, "anim_tags")

        self.initializeDataSet()

        # for dirName in self.dbPlugin.getGroupList():
        #     dataDirItem = DirItem(groupName = dirName,
        #                           parentWidget = self.ui.aniTreeWidget)

        self.contentTabList = {}

        # RETARGETTING #
        self.sourcefile = os.path.join(retargetPath, getpass.getuser(), "copyed_source.bvh")
        self.targetfile = os.path.join(retargetPath, getpass.getuser(), "proxy.bvh")
        self.remapData = {}

        # Connect Setting
        self.ui.aniTreeWidget.itemDoubleClicked.connect(self.aniFileDoubleClick)

        # Mouse Right Click Signal
        self.ui.aniTreeWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.aniTreeWidget.customContextMenuRequested.connect(self.rmbClicked)

        self.ui.ContentTabWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.ContentTabWidget.customContextMenuRequested.connect(self.contentTabRmbClicked)

        self.ui.ContentScaleSlider.valueChanged.connect(self.sliderValueChanged)

        self.ui.ContentTabWidget.setTabsClosable(True)
        self.ui.ContentTabWidget.tabCloseRequested.connect(self.tabCloseEvent)

        self.ui.tagSearchEdit.returnPressed.connect(self.tagSearchPressed)

        self.ui.moreBtn.clicked.connect(self.moreLoadData)

        inc_tool_by_user.run('action.AnimBrowser.open', getpass.getuser())

    def getNamespace(self):
        if cmds.ls(sl=1):
            sel = cmds.ls(sl=True)[0]
            nsChar = str(sel.rsplit(':')[0])
            return nsChar
        else:
            cmds.warning('"select Character!!"')
            return

    def applyStyleSheet(self):
        item_styles = """
        QTreeWidget::item { padding: 4 2 4 2 px; margin: 0px; border: 0 px}
        QTreeWidget::item:selected{background: rgb(67, 124, 185);}
        """
        self.setStyleSheet(item_styles)

    def initializeDataSet(self):
        # tag 1 tier setting
        self.tagData = self.tagPlugin.getTagData()
        self.tagDict = {}
        for tag1Tier in self.tagData:
            teamCategory = tag1Tier['category']
            if not self.tagDict.has_key(teamCategory):
                self.tagDict[teamCategory] = {}
            if not self.tagDict[teamCategory].has_key(tag1Tier["tag_tier1"]):
                self.tagDict[teamCategory][tag1Tier["tag_tier1"]] = {}

            if not tag1Tier.has_key("tag_tier2"):
                continue
            elif not self.tagDict[teamCategory][tag1Tier["tag_tier1"]].has_key(tag1Tier["tag_tier2"]):
                self.tagDict[teamCategory][tag1Tier["tag_tier1"]][tag1Tier["tag_tier2"]] = list()

            if not tag1Tier.has_key("tag_tier3"):
                continue
            else:
                self.tagDict[teamCategory][tag1Tier["tag_tier1"]][tag1Tier["tag_tier2"]].insert(0,
                                                                                                tag1Tier["tag_tier3"])

        self.ui.aniTreeWidget.clear()

        for key in self.tagDict.keys():
            item = DirItem(key)
            # make category
            for key2 in self.tagDict[key].keys():
                parentText = key
                item2 = AniSubItem(parentText=parentText,
                                   itemName=key2,
                                   parentWidget=item)
                # make tier1
                for key3 in self.tagDict[key][key2]:
                    parentText = key
                    parentText = os.path.join(parentText, key2)
                    item3 = AniSubItem(parentText=parentText,
                                       itemName=key3,
                                       parentWidget=item2)

                    for key4 in self.tagDict[key][key2][key3]:
                        parentText = key
                        parentText = os.path.join(parentText, key2, key3)
                        item4 = AniSubItem(parentText=parentText,
                                           itemName=key4,
                                           parentWidget=item3)

            self.ui.aniTreeWidget.addTopLevelItem(item)

    def tabCloseEvent(self, tabIndex):
        # select close tabWidget
        tabName = self.ui.ContentTabWidget.tabText(tabIndex)

        # listWidget Clear
        self.contentTabList[tabName].clear()

        # del tab
        self.contentTabList.pop(tabName, None)

        # Remove TabWidget
        self.ui.ContentTabWidget.removeTab(tabIndex)

    def rmbClicked(self, pos):
        menu = QtWidgets.QMenu(self)

        menu.addAction(u"HIK 애니메이션 업로드(&H)", self.openImportDialog)
        menu.addAction(u"No HIK 애니메이션 업로드(&N)", self.openImportNoneHIKDialog)
        menu.addAction(u"Mocap 데이터 업로드(&N)", self.openMocapUploadDialog)
        menu.addAction(u"Crowd 데이터 업로드(&N)", self.openCrowdUploadDialog)
        menu.addAction(u"Open &TagManager(&T)", self.openTagManager)

        # JM
        icon = Qt.QtGui.QIcon(os.path.join(str(CURRENTDIR), "retargetting/spider_purple.png"))
        menu.addAction(icon, u"Upload Source for Retarget", self.openExportSourceDialog)
        menu.addAction(icon, u"export BVH", self.exportBVH)

        if self.ui.aniTreeWidget.currentItem() is not None:
            try:
                itemName = self.ui.aniTreeWidget.currentItem().itemName
                if not itemName in ['Mocap', "Crowd", "Animation"]:
                    menu.addAction(u'%s 이름 변경' % self.ui.aniTreeWidget.currentItem().itemName,
                                   lambda: self.tierNameChanged(self.ui.aniTreeWidget.currentItem()))
            except:
                pass

        menu.popup(self.mapToGlobal(QtCore.QPoint(pos.x() + 15, pos.y() + 15)))

    def tierNameChanged(self, currentItem):
        print currentItem.itemFullPath

        dialog = QtWidgets.QInputDialog(self)
        dialog.setWindowTitle(u"티어 이름 변경")
        dialog.setLabelText(u"%s -> ?" % currentItem.itemName)
        msg = dialog.exec_()

        if msg == 1:
            splitTier = currentItem.itemFullPath.split('/')
            print splitTier
            if len(splitTier) > 1:
                category = splitTier[0]
                tierList = [""] * 3
                for index in range(3):
                    if len(splitTier) - 1 > index:
                        tierList[index] = splitTier[index + 1]
                    else:
                        tierList[index] = ""
                dbConfig.renameTier(len(splitTier) - 1, dialog.textValue(), category=category, tierList=tierList)
                self.initializeDataSet()

    def contentTabRmbClicked(self, pos):
        if self.ui.ContentTabWidget.currentWidget() is None or self.ui.ContentTabWidget.currentWidget().currentItem() is None:
            return
        menu = QtWidgets.QMenu(self)

        menu.addAction(u"HIK .anim 데이터 불러오기(&A)", self.applyHIKAnimFile)  # HIK
        menu.addAction(u"StudioLibrary .anim 데이터 불러오기(&R)", self.applySTDAnimFile)  # STD
        menu.addAction(u"모캡용 애니(.anim) 데이터 불러오기(&R)", self.applyMocapAnimFile)  # ANI
        menu.addAction(u"군중 액션(.ma) 데이터 불러오기(&R)", self.applyCRDAction)  # ACT

        menu.addAction(u"데이터 지우기", self.removeContent)

        menu.addAction(u"티어 변경하기", self.changeTier)
        menu.addAction(u"덮어쓰기", self.overwriteItem)
        if self.ui.ContentTabWidget.currentWidget().currentItem().contentInfo.has_key('hashTag'):
            menu.addAction(u"해쉬태그 수정하기", self.modifyHashTag)

        # JM
        icon = Qt.QtGui.QIcon(os.path.join(str(CURRENTDIR), "retargetting/spider_purple.png"))
        menu.addAction(icon, "Import BVH", self.importRetargetSourceToJoint)
        menu.addAction(icon, "preview BVH", self.createPreviewNode)
        menu.addAction(icon, "Retarget & Preview", self.importRetargetSourceToZArachne)
        menu.addAction(icon, "Retarget & Import to CON", self.importRetargetSourceToCON)

        menu.popup(self.mapToGlobal(QtCore.QPoint(pos.x() + 200, pos.y() + 15)))

    ################################################## Apply START ##################################################

    def applyHIKAnimFile(self):
        animFilePath = self.ui.ContentTabWidget.currentWidget().currentItem().getAnimFilePath()
        isHIK = self.ui.ContentTabWidget.currentWidget().currentItem().isHIK()

        if isHIK == 1:
            hikAnim = animBridge.Window()
            nsChar = self.getNamespace()
            hikAnim.makeTpose()
            hikAnim.loadHikAnim(nsChar,animFilePath)
        # elif isHIK == 0:
        #     a = mutils.Animation.fromPath(animFilePath)
        #     a.load()
        else:
            MessageBox(u"잘못된 데이터 타입입니다.\n다시 확인해주세요")
            return

        MessageBox("apply animation")

    def applySTDAnimFile(self):
        animFilePath = self.ui.ContentTabWidget.currentWidget().currentItem().getAnimFilePath()
        isHIK = self.ui.ContentTabWidget.currentWidget().currentItem().isHIK()

        # if isHIK == 1:
        #     hikAnim = animBridge.Window()
        #     hikAnim.makeTpose()
        #     hikAnim.loadHikAnim(animFilePath)
        if isHIK == 0:
            a = mutils.Animation.fromPath(animFilePath)
            a.load()
        else:
            MessageBox(u"잘못된 데이터 타입입니다.\n다시 확인해주세요")
            return

        MessageBox("apply animation")
        # inc_tool_by_user.run('action.AnimBrowser.exportAnimation', getpass.getuser())

    # Export Function
    def applyMocapAnimFile(self):
        animFilePath = self.ui.ContentTabWidget.currentWidget().currentItem().getAnimFilePath()
        isHIK = self.ui.ContentTabWidget.currentWidget().currentItem().isHIK()

        # if isHIK == 1:
        hikAnim = animBridge.Window()
        # hikAnim.makeTpose()
        nsChar = self.getNamespace()
        hikAnim.impMc(nsChar, animFilePath)
        # elif isHIK == 0:
        #     a = mutils.Animation.fromPath(animFilePath)
        #     a.load()
        # else:
        #     MessageBox(u"잘못된 데이터 타입입니다.\n다시 확인해주세요")
        #     return

        MessageBox("apply animation")

        # inc_tool_by_user.run('action.AnimBrowser.exportAnimation', getpass.getuser())

    def applyCRDAction(self):
        actionFilePath = self.ui.ContentTabWidget.currentWidget().currentItem().getActionFilePath()
        McdFunctionModified.McdLoadActions(actionFilePath)

    ################################################## Apply END ##################################################

    def overwriteItem(self):
        curItem = self.ui.ContentTabWidget.currentWidget().currentItem()
        print curItem.contentInfo['files'].keys()

        print "# Select File"
        dstSelectFile = ''
        for file in curItem.contentInfo['files'].keys():
            print os.path.basename(curItem.contentInfo['files'][file])
            dstSelectFile = curItem.contentInfo['files'][file]

        print "# if select file"
        print "# select overwrite file"
        ext = os.path.splitext(dstSelectFile)[-1]

        srcFileName = QtWidgets.QFileDialog.getOpenFileName(self, 'Select %s file' % ext, "/", '*' + ext)[0]

        print "os.system('cp -rf %s %s')" % (srcFileName, dstSelectFile)

    def modifyHashTag(self):
        dialog = QtWidgets.QInputDialog(self)
        dialog.setLabelText(u"해쉬태그 수정")
        dialog.setTextValue(' '.join(self.ui.ContentTabWidget.currentWidget().currentItem().contentInfo['hashTag']))
        msg = dialog.exec_()

        if msg == 1:
            hashTag = dialog.textValue().split(' ')
            dbItem = dbConfig.updateHashTag(self.ui.ContentTabWidget.currentWidget().currentItem().contentInfo['_id'],
                                            hashTag)
            self.ui.ContentTabWidget.currentWidget().currentItem().contentInfo = dbItem

    def changeTier(self):
        dialog = ChangeTierDialog(self, self.ui.ContentTabWidget.currentWidget().currentItem().contentInfo,
                                  self.tagDict)
        dialog.exec_()

    def removeContent(self):
        category = self.ui.ContentTabWidget.currentWidget().currentItem().contentInfo['category']
        tag1Tier = self.ui.ContentTabWidget.currentWidget().currentItem().contentInfo['tag1tier']
        tag2Tier = self.ui.ContentTabWidget.currentWidget().currentItem().contentInfo['tag2tier']
        tag3Tier = self.ui.ContentTabWidget.currentWidget().currentItem().contentInfo['tag3tier']
        fileNum = self.ui.ContentTabWidget.currentWidget().currentItem().contentInfo["fileNum"]
        msg = MessageBox(winTitle="Confirm!", Message="Do you try remove Content? [%s%s]" % (tag3Tier, fileNum),
                         Icon="question", Button=["OK", "CANCEL"])

        if msg == "OK":
            self.dbPlugin.removeDocument(tag1Tier=tag1Tier,
                                         tag2Tier=tag2Tier,
                                         tag3Tier=tag3Tier,
                                         category=category,
                                         fileNum=fileNum)

            self.ui.ContentTabWidget.currentWidget().takeItem(self.ui.ContentTabWidget.currentWidget().currentRow())

    '''
    ################
    ### RETARGET ###
    ################
    '''

    def openExportSourceDialog(self):
        self.exportSrcDialog = ExportSourceDialog(self)
        self.exportSrcDialog.exec_()

    def doRetargetting(self, node):
        if not cmds.pluginInfo("ZArachneForMaya", q=True, l=True):
            cmds.loadPlugin("ZArachneForMaya")

        sourcefile = str(self.ui.ContentTabWidget.currentWidget().currentItem().getBVHFilePath())
        targetfile = self.targetfile  # selected Node

        ### CHECK VALIDITY ###
        s = Zelos.skeleton()
        s.load(sourcefile)
        if s.numJoints() == 0:
            print 'source not valid'
            return

        target_ns = ""
        if node.split(':') > 1:
            target_ns = ':'.join(node.split(':')[0:-1])

        ### REMAP DIALOG ###
        remap = self.RemapConfirmDialog()
        if remap == 2:
            # YES ROLE
            self.remapDialog = RemapDialog(self, sourcefile=sourcefile, target_ns=target_ns)
            self.remapDialog.setAttribute(QtCore.Qt.WA_DeleteOnClose)
            self.remapDialog.show()
            self.remapDialog.init_Remap()
            result = self.remapDialog.exec_()
            if result:
                # do REMAP
                self.remapData = self.remapDialog.remapData
            else:
                return

        elif remap == 1:
            # NO ROLE
            # without REMAP
            exporter = bvhExporter.BVHExporter()
            exporter.generate_joint_data(node, targetfile, 0, 2)

        else:
            # CANCEL ROLE
            return

        # retarget
        shutil.copyfile(sourcefile, self.sourcefile)
        sourcefile = self.sourcefile
        mel.eval(""" ZArachneRetargetBVH "%s" "%s" "spider"; """ % (sourcefile, targetfile))
        # os.remove(targetfile)
        retargetfile = targetfile.replace(".bvh", "_retarget.bvh")
        if os.path.exists(retargetfile):
            return retargetfile

    # import retarget source
    def importRetargetSourceToZArachne(self):
        selection = cmds.ls(sl=True, type="joint")  # rootJoint
        if selection:
            selection = selection[0]
            retargetfile = self.doRetargetting(selection)

    def exportBVH(self):
        targetfile = os.path.join(retargetPath, getpass.getuser(), "exported.bvh")
        sel = cmds.ls(sl=True, type='joint')
        if sel: sel = sel[0]
        exporter = bvhExporter.BVHExporter()
        exporter.generate_joint_data(sel, targetfile, 0, 0)
        self.createPreviewNode(targetfile)

    # import retarget source and construct joint
    def importRetargetSourceToJoint(self):
        selection = cmds.ls(sl=True, type="joint")
        if selection:
            target_ns = ""
            selection = selection[0]
            retargetfile = self.doRetargetting(selection)
            if retargetfile:
                ns = selection.split(':')
                if os.path.exists(retargetfile):
                    importer = bvhImporter.BVHImporter()
                    importer.isCON = False
                    if ns > 1:
                        importer.target_ns = ':'.join(ns[0:-1])
                    importer.read_bvh(retargetfile)

        else:
            time1 = time.time()
            sourcefile = self.sourcefile
            importer = bvhImporter.BVHImporter()
            importer.isCON = False
            importer.read_bvh(sourcefile)
            print time.time() - time1

    # import retarget source to CONTROLLER
    def importRetargetSourceToCON(self):
        selection = cmds.ls(sl=True, type="dxRig")
        if selection:
            target_ns = ""
            selection = selection[0]
            ns = selection.split(':')
            if ns > 1:
                target_ns = ':'.join(ns[0:-1])

            dxRigUI.controlersInit("%s.controlers" % selection)
            dxRigUI.controlersInitWorld("%s.controlers" % selection)
            dxRigUI.selectAttributeObjects("%s.controlers" % selection)
            cmds.cutKey(cmds.ls(sl=1), clear=True, hi='none', cp=0, shape=1)
            cmds.select(cl=1)

            cmds.delete(cmds.ls(['proxy*_CON', 'proxy*_NUL']) + cmds.ls('proxy*:*', type='joint') + cmds.ls(
                'ZArachneCharacter*'))

            # GET SCALE AND SET TO 1
            placecon = target_ns + 'place_CON'
            if cmds.objExists(placecon):
                if 'initScale' in cmds.listAttr(placecon):
                    initscale = cmds.getAttr('%s.initScale' % placecon)
                    initscaleLock = cmds.getAttr('%s.initScale' % placecon, l=True)
                    if initscaleLock:
                        cmds.setAttr('%s.initScale' % placecon, l=False)
                    cmds.setAttr('%s.initScale' % placecon, 1)

                if 'globalScale' in cmds.listAttr(placecon):
                    globalscale = cmds.getAttr('%s.globalScale' % placecon)
                    globalscaleLock = cmds.getAttr('%s.globalScale' % placecon, l=True)
                    if globalscaleLock:
                        cmds.setAttr('%s.globalScale' % placecon, l=False)
                    cmds.setAttr('%s.globalScale' % placecon, 1)

            retargetfile = self.doRetargetting(selection)
            if retargetfile:
                targetfile = self.targetfile
                sourcefile = self.sourcefile
                skeleton = Zelos.skeleton()
                skeleton.load(targetfile)
                if self.remapData:
                    pprint(self.remapData)
                    self.importRetargetSrcDialog = ImportSourceDialog(self, skeleton, target_ns, retargetfile, )
                    self.importRetargetSrcDialog.remapData = self.remapData
                    self.importRetargetSrcDialog.show()
                    self.importRetargetSrcDialog.init_remapRetargetting()
                else:
                    self.importRetargetSrcDialog = ImportSourceDialog(self, skeleton, target_ns, retargetfile)
                    self.importRetargetSrcDialog.show()
                    self.importRetargetSrcDialog.init_Retargetting()

            # SET SCALE TO ORIGINAL
            if cmds.objExists(placecon):
                if 'initScale' in cmds.listAttr(placecon):
                    cmds.setAttr('%s.initScale' % placecon, initscale)
                    if initscaleLock:
                        cmds.setAttr('%s.initScale' % placecon, l=True)
                if 'globalScale' in cmds.listAttr(placecon):
                    cmds.setAttr('%s.globalScale' % placecon, globalscale)
                    if globalscaleLock:
                        cmds.setAttr('%s.globalScale' % placecon, l=True)

            cmds.delete(cmds.ls(['proxy*_CON', 'proxy*_NUL']) + cmds.ls('proxy*:*', type='joint'))

        else:
            cmds.error("No Root Joint Selected")

    def createPreviewNode(self, filename=""):
        if not cmds.pluginInfo("ZArachneForMaya", q=True, l=True):
            cmds.loadPlugin("ZArachneForMaya")
        if not filename:
            filename = self.ui.ContentTabWidget.currentWidget().currentItem().getBVHFilePath()
        node = cmds.createNode("ZArachneCharacter")
        cmds.setAttr("%s.filePath" % node, filename, type="string")
        cmds.currentTime(0)

    def RemapConfirmDialog(self, text='Remap Joints?'):
        dialog = QtWidgets.QMessageBox()
        dialog.setText(text)
        dialog.setWindowTitle('MESSAGE')
        dialog.addButton('Cancel', QtWidgets.QMessageBox.RejectRole)
        dialog.addButton('No Remap', QtWidgets.QMessageBox.NoRole)
        dialog.addButton('Remap', QtWidgets.QMessageBox.YesRole)
        dialog.move(QtWidgets.QDesktopWidget().availableGeometry().center())
        dialog.show()
        return dialog.exec_()

    ###########################
    # Import Dialog Function
    ###########################
    def openImportDialog(self):
        self.importDialog = HIKImportDialog(self)
        self.importDialog.exec_()

    def importCallDialog(self):
        try:
            if self.importDialog.ui.aniRadioBtn.isChecked():
                category = "Animation"
            elif self.importDialog.ui.mcpRadioBtn.isChecked():
                category = "Mocap"
            elif self.importDialog.ui.crdRadioBtn.isChecked():
                category = "Crowd"
            else:
                cmds.confirmDialog("don't select category")
                return False

            for i in range(1, 5):
                mel.eval("modelEditor - e - nurbsCurves false modelPanel{0}".format(i))
                mel.eval("modelEditor - e - joints false modelPanel{0}".format(i))

            title1Tier = self.importDialog.ui.Title1ComboBox.currentText()
            title2Tier = self.importDialog.ui.Title2ComboBox.currentText()
            title3Tier = self.importDialog.ui.Title3ComboBox.currentText()
            isTractor = self.importDialog.ui.tractorCheckBox.isChecked()

            checkDB = {}
            checkDB["category"] = category
            checkDB["tag1tier"] = title1Tier
            checkDB["tag2tier"] = title2Tier
            checkDB["tag3tier"] = title3Tier

            fileNum = self.dbPlugin.existDocument(checkDB) + 1

            # step 1 : load anim file
            if self.importDialog.ui.animPathEdit.text() == "":
                if not os.path.isdir(
                        "{0}/{1}/{2}/{3}/{4}/{5}".format(baseDirPath, category, title1Tier, title2Tier, title3Tier,
                                                         fileNum)):
                    os.makedirs(
                        "{0}/{1}/{2}/{3}/{4}/{5}".format(baseDirPath, category, title1Tier, title2Tier, title3Tier,
                                                         fileNum))

                animFilePath = "{0}/{1}/{2}/{3}/{4}/{5}/{2}_{3}_{4}.anim".format(baseDirPath, category, title1Tier,
                                                                                 title2Tier, title3Tier, fileNum)
                jsonFilePath = "{0}/{1}/{2}/{3}/{4}/{5}/{2}_{3}_{4}.json".format(baseDirPath, category, title1Tier,
                                                                                 title2Tier, title3Tier, fileNum)

                self.importDialog.ui.animPathEdit.setText(animFilePath)
                self.importDialog.ui.jsonPathEdit.setText(jsonFilePath)
                hikAnim = animBridge.Window()
                # Rest Frame Default : 950
                # Start Action Frame Default : 1001
                restFrame = int(self.importDialog.getRestFrame())
                startFrame = int(self.importDialog.getStartFrame())
                hikAnim.makeTpose()
                nsChar = self.getNamespace()
                hikAnim.saveHikAnim(exAnim=animFilePath,
                                    restFrame=restFrame,
                                    startFrame=startFrame)

            # step 2 : make preview gif
            playblastMov = MakeMov(baseDirPath, category, title1Tier, title2Tier, title3Tier, fileNum)
            movFileName = playblastMov.takePlayblast()
            playblastMov.movToGif(movFileName)

            # step 3 : publish
            self.publish(category=category,
                         tag1TierTitle=self.importDialog.ui.Title1ComboBox.currentText(),
                         tag2TierTitle=self.importDialog.ui.Title2ComboBox.currentText(),
                         tag3TierTitle=self.importDialog.ui.Title3ComboBox.currentText(),
                         animFilePath=self.importDialog.ui.animPathEdit.text(),
                         jsonFilePath=self.importDialog.ui.jsonPathEdit.text(),
                         previewFilePath=playblastMov.gifFilePath,
                         isHIK=1,
                         isTractor=isTractor,
                         movFile=movFileName)

            for i in range(1, 5):
                mel.eval("modelEditor - e - nurbsCurves true modelPanel{0}".format(i))
                mel.eval("modelEditor - e - joints true modelPanel{0}".format(i))

            inc_tool_by_user.run('action.AnimBrowser.importAnimation_HIK', getpass.getuser())
            return True

        except Exception as e:
            for i in range(1, 5):
                mel.eval("modelEditor - e - nurbsCurves true modelPanel{0}".format(i))
                mel.eval("modelEditor - e - joints true modelPanel{0}".format(i))

            cmds.warning(e.message)
            # self.importDialog.closeBtnClick()
            return False

    def openMocapUploadDialog(self):
        self.importMocapDialog = MocapImportDialog(self)

        self.importMocapDialog.exec_()

    def importCallDialogMocap(self):
        # try:
        for i in range(1, 5):
            mel.eval("modelEditor - e - nurbsCurves false modelPanel{0}".format(i))
            mel.eval("modelEditor - e - joints false modelPanel{0}".format(i))

        category = "Mocap"
        title1Tier = self.importMocapDialog.ui.Title1ComboBox.currentText()
        title2Tier = self.importMocapDialog.ui.Title2ComboBox.currentText()
        title3Tier = self.importMocapDialog.ui.Title3ComboBox.currentText()
        isTractor = self.importMocapDialog.ui.tractorCheckBox.isChecked()
        hashTagStr = self.importMocapDialog.ui.hasTagEdit.text()

        print category, title1Tier, title2Tier, title3Tier, isTractor

        checkDB = {}
        checkDB["category"] = category
        checkDB["tag1tier"] = title1Tier
        checkDB["tag2tier"] = title2Tier
        checkDB["tag3tier"] = title3Tier

        fileNum = self.dbPlugin.existDocument(checkDB) + 1

        # step 1 : load anim file
        if self.importMocapDialog.ui.animPathEdit.text() == "":
            # if not os.path.isdir("{0}/{1}/{2}/{3}/{4}/{5}".format(baseDirPath, category, title1Tier, title2Tier, title3Tier, fileNum)):
            #     os.makedirs("{0}/{1}/{2}/{3}/{4}/{5}".format(baseDirPath, category, title1Tier, title2Tier, title3Tier, fileNum))
            #
            # animFilePath = "{0}/{1}/{2}/{3}/{4}/{5}/{2}_{3}_{4}.anim".format(baseDirPath, category, title1Tier, title2Tier, title3Tier, fileNum)
            #
            # self.importMocapDialog.ui.animPathEdit.setText(animFilePath)
            MessageBox(u"anim파일을 입력해주세요")
            return True
        elif self.importMocapDialog.ui.movPathEdit.text() == "":
            MessageBox(u"mov파일을 입력해주세요")
            return True
        # hikAnim = animBridge.Window()
        #     # Rest Frame Default : 950
        #     # Start Action Frame Default : 1001
        #     restFrame = int(self.importDialog.getRestFrame())
        #     startFrame = int(self.importDialog.getStartFrame())
        #     hikAnim.makeTpose()
        #     hikAnim.saveHikAnim(exAnim = animFilePath,
        #                         restFrame = restFrame,
        #                         startFrame = startFrame)
        #
        # step 2 : make preview gif
        playblastMov = MakeMov(baseDirPath, category, title1Tier, title2Tier, title3Tier, fileNum)
        movFileName = self.importMocapDialog.ui.movPathEdit.text()
        playblastMov.movToGif(movFileName)

        # step 3 : publish
        self.publish(category=category,
                     tag1TierTitle=self.importMocapDialog.ui.Title1ComboBox.currentText(),
                     tag2TierTitle=self.importMocapDialog.ui.Title2ComboBox.currentText(),
                     tag3TierTitle=self.importMocapDialog.ui.Title3ComboBox.currentText(),
                     animFilePath=self.importMocapDialog.ui.animPathEdit.text(),
                     previewFilePath=playblastMov.gifFilePath,
                     isHIK=2,
                     hashTag=hashTagStr.split(' '),
                     isTractor=isTractor,
                     movFile=movFileName)

        for i in range(1, 5):
            mel.eval("modelEditor - e - nurbsCurves true modelPanel{0}".format(i))
            mel.eval("modelEditor - e - joints true modelPanel{0}".format(i))

        inc_tool_by_user.run('action.AnimBrowser.importCallDialogMocap', getpass.getuser())
        return True

        # except Exception as e:
        #     for i in range(1, 5):
        #         mel.eval("modelEditor - e - nurbsCurves true modelPanel{0}".format(i))
        #         mel.eval("modelEditor - e - joints true modelPanel{0}".format(i))
        #
        #     cmds.warning(e.message)
        #     # self.importDialog.closeBtnClick()
        #     return False

    def openCrowdUploadDialog(self):
        self.importCrowdDialog = CrowdImportDialog(self)

        self.importCrowdDialog.exec_()

    def importCallDialogCrowd(self):
        try:
            for i in range(1, 5):
                mel.eval("modelEditor - e - nurbsCurves false modelPanel{0}".format(i))
                mel.eval("modelEditor - e - joints false modelPanel{0}".format(i))

            category = "Crowd"
            title1Tier = self.importCrowdDialog.ui.Title1ComboBox.currentText()
            title2Tier = self.importCrowdDialog.ui.Title2ComboBox.currentText()
            title3Tier = self.importCrowdDialog.ui.Title3ComboBox.currentText()
            isTractor = self.importCrowdDialog.ui.tractorCheckBox.isChecked()
            hashTagStr = self.importCrowdDialog.ui.hashTagEdit.text()

            print category, title1Tier, title2Tier, title3Tier, isTractor

            checkDB = {}
            checkDB["category"] = category
            checkDB["tag1tier"] = title1Tier
            checkDB["tag2tier"] = title2Tier
            checkDB["tag3tier"] = title3Tier

            fileNum = self.dbPlugin.existDocument(checkDB) + 1

            # step 1 : load ma file
            if self.importCrowdDialog.ui.animPathEdit.text() == "":
                # if not os.path.isdir("{0}/{1}/{2}/{3}/{4}/{5}".format(baseDirPath, category, title1Tier, title2Tier, title3Tier, fileNum)):
                #     os.makedirs("{0}/{1}/{2}/{3}/{4}/{5}".format(baseDirPath, category, title1Tier, title2Tier, title3Tier, fileNum))
                #
                # animFilePath = "{0}/{1}/{2}/{3}/{4}/{5}/{2}_{3}_{4}.ma".format(baseDirPath, category, title1Tier, title2Tier, title3Tier, fileNum)
                # self.importCrowdDialog.ui.animPathEdit.setText(animFilePath)
                MessageBox(u"ma파일을 입력해주세요")
                return True
            elif self.importCrowdDialog.ui.movPathEdit.text() == "":
                MessageBox(u"mov파일을 입력해주세요")
                return True
            # hikAnim = animBridge.Window()
            #     # Rest Frame Default : 950
            #     # Start Action Frame Default : 1001
            #     restFrame = int(self.importDialog.getRestFrame())
            #     startFrame = int(self.importDialog.getStartFrame())
            #     hikAnim.makeTpose()
            #     hikAnim.saveHikAnim(exAnim = animFilePath,
            #                         restFrame = restFrame,
            #                         startFrame = startFrame)
            #
            # step 2 : make preview gif
            playblastMov = MakeMov(baseDirPath, category, title1Tier, title2Tier, title3Tier, fileNum)
            movFileName = self.importCrowdDialog.ui.movPathEdit.text()
            playblastMov.movToGif(movFileName)

            # step 3 : publish
            self.publish(category=category,
                         tag1TierTitle=self.importCrowdDialog.ui.Title1ComboBox.currentText(),
                         tag2TierTitle=self.importCrowdDialog.ui.Title2ComboBox.currentText(),
                         tag3TierTitle=self.importCrowdDialog.ui.Title3ComboBox.currentText(),
                         animFilePath=self.importCrowdDialog.ui.animPathEdit.text(),
                         previewFilePath=playblastMov.gifFilePath,
                         isHIK=2,
                         hashTag=hashTagStr.split(' '),
                         isTractor=isTractor,
                         movFile=movFileName)

            for i in range(1, 5):
                mel.eval("modelEditor - e - nurbsCurves true modelPanel{0}".format(i))
                mel.eval("modelEditor - e - joints true modelPanel{0}".format(i))

            inc_tool_by_user.run('action.AnimBrowser.importCallDialogCrowd', getpass.getuser())
            return True

        except Exception as e:
            for i in range(1, 5):
                mel.eval("modelEditor - e - nurbsCurves true modelPanel{0}".format(i))
                mel.eval("modelEditor - e - joints true modelPanel{0}".format(i))

            cmds.warning(e.message)
            # self.importDialog.closeBtnClick()
            return False

    def openImportNoneHIKDialog(self):
        self.noneHIKImportDialog = NoneHIKImportDialog(self)

        self.noneHIKImportDialog.exec_()

    def importCallDialogNoHIK(self):
        try:
            for i in range(1, 5):
                mel.eval("modelEditor - e - nurbsCurves false modelPanel{0}".format(i))
                mel.eval("modelEditor - e - joints false modelPanel{0}".format(i))

            category = "Animation"
            title1Tier = self.noneHIKImportDialog.ui.Title1ComboBox.currentText()
            title2Tier = self.noneHIKImportDialog.ui.Title2ComboBox.currentText()
            title3Tier = self.noneHIKImportDialog.ui.Title3ComboBox.currentText()
            isTractor = self.noneHIKImportDialog.ui.tractorCheckBox.isChecked()

            checkDB = {}
            checkDB["category"] = category
            checkDB["tag1tier"] = title1Tier
            checkDB["tag2tier"] = title2Tier
            checkDB["tag3tier"] = title3Tier

            fileNum = self.dbPlugin.existDocument(checkDB) + 1

            # step 1 : load anim file
            if not os.path.isdir(
                    "{0}/{1}/{2}/{3}/{4}/{5}".format(baseDirPath, category, title1Tier, title2Tier, title3Tier,
                                                     fileNum)):
                os.makedirs("{0}/{1}/{2}/{3}/{4}/{5}".format(baseDirPath, category, title1Tier, title2Tier, title3Tier,
                                                             fileNum))

            outputPath = "{0}/{1}/{2}/{3}/{4}/{5}/{2}_{3}_{4}.anim".format(baseDirPath, category, title1Tier,
                                                                           title2Tier, title3Tier, fileNum)

            objects = cmds.ls(sl=True)
            animModule = mutils.Animation.fromObjects(objects)
            animModule.save(outputPath)

            # step 2 : make preview gif
            playblastMov = MakeMov(baseDirPath, category, title1Tier, title2Tier, title3Tier, fileNum)
            movFileName = playblastMov.takePlayblast()
            playblastMov.movToGif(movFileName)

            # step 3 : publish
            self.publish(category=category,
                         tag1TierTitle=self.noneHIKImportDialog.ui.Title1ComboBox.currentText(),
                         tag2TierTitle=self.noneHIKImportDialog.ui.Title2ComboBox.currentText(),
                         tag3TierTitle=self.noneHIKImportDialog.ui.Title3ComboBox.currentText(),
                         animFilePath=outputPath,
                         previewFilePath=playblastMov.gifFilePath,
                         isHIK=0,
                         isTractor=isTractor,
                         movFile=movFileName)

            for i in range(1, 5):
                mel.eval("modelEditor - e - nurbsCurves true modelPanel{0}".format(i))
                mel.eval("modelEditor - e - joints true modelPanel{0}".format(i))

            inc_tool_by_user.run('action.AnimBrowser.importAnimation_NoneHIK', getpass.getuser())
            return True
        except Exception as e:
            for i in range(1, 5):
                mel.eval("modelEditor - e - nurbsCurves true modelPanel{0}".format(i))
                mel.eval("modelEditor - e - joints true modelPanel{0}".format(i))

            cmds.warning(e.message)
            # self.noneHIKImportDialog.closeBtnClick()
            return False

    def openTagManager(self):
        self.tagManagerDialog = TagForm(self)

        self.tagManagerDialog.exec_()

        self.initializeDataSet()

    def publish(self, category, tag1TierTitle, tag2TierTitle, tag3TierTitle, animFilePath, previewFilePath, isHIK=2,
                jsonFilePath="", isTractor=True, hashTag=[], movFile="", bvhFilePath=""):
        '''
        import anim & json File

        :param groupName: groupName [ top level name ]
        :param assetName: assetName [ groupName -> assetName ]
        :param animFilePath: local anim file path
        :param jsonFilePath: local anim file path (if isn't HIK ==> "") 
        :param previewFilePath: local anim file path
        :param tagList:
        :return:
        '''

        # step 1 : files fix local Path & pub Path [anim, json, preview gif]
        # document overlap check
        checkDB = {}
        checkDB["category"] = category
        checkDB["tag1tier"] = tag1TierTitle
        checkDB["tag2tier"] = tag2TierTitle
        checkDB["tag3tier"] = tag3TierTitle

        fileNum = self.dbPlugin.existDocument(checkDB) + 1
        self.dbPlugin.setFileNum(fileNum)

        # pub base directory path : /assetlib/anim_browser/groupName/assetNames
        pubBaseDirPath = "/assetlib/anim_browser/{0}/{1}/{2}/{3}/{4}".format(category, tag1TierTitle, tag2TierTitle,
                                                                             tag3TierTitle, fileNum)

        animPubPath = "{0}/{1}".format(pubBaseDirPath, os.path.basename(animFilePath))
        prevGifPubPath = "{0}/{1}".format(pubBaseDirPath, os.path.basename(previewFilePath))

        # step 2 : db document insert
        files = {"preview": prevGifPubPath}

        if category == "Crowd":
            files["ma"] = animPubPath
        else:
            files["anim"] = animPubPath

        jsonPubPath = ""
        if jsonFilePath != "":
            jsonPubPath = "{0}/{1}".format(pubBaseDirPath, os.path.basename(jsonFilePath))
            files["json"] = jsonPubPath

        bvhPubPath = ""
        if bvhFilePath != "":
            bvhPubPath = "{0}/{1}".format(pubBaseDirPath, os.path.basename(bvhFilePath))
            files["bvh"] = bvhPubPath

        movPubPath = ""
        if not movFile == "":
            movPubPath = "{0}/{1}".format(pubBaseDirPath, os.path.basename(movFile))
            files["mov"] = movPubPath

        self.dbPlugin.setCategory(category)
        self.dbPlugin.setTag1TierName(tag1TierTitle)
        self.dbPlugin.setTag2TierName(tag2TierTitle)
        self.dbPlugin.setTag3TierName(tag3TierTitle)
        self.dbPlugin.setFiles(files)
        self.dbPlugin.setIsHIK(isHIK)
        self.dbPlugin.setHashTag(hashTag)

        self.dbPlugin.updateRecord()

        # insert document
        self.dbPlugin.insertDocument()

        # step 3 : send job for tractor
        tractor = TractorPublish()
        tractor.createJob(
            jobName="{0}_{1}_{2}_{3}_{4}".format(category, tag1TierTitle, tag2TierTitle, tag3TierTitle, fileNum))
        tractor.createRootTask()

        # make command
        # cmd : make directory
        mkdirCmd = "install -d -m 755 {0}".format(pubBaseDirPath)

        # cmd : move preview file
        mvPrevFileCmd = "mv -vf {0} {1}".format(previewFilePath, prevGifPubPath)

        # cmd : move or copy anim file
        keyword = "cp"
        if baseDirPath in animFilePath:
            keyword = "mv"
        animFileCmd = "{0} -vf {1} {2}".format(keyword, animFilePath, animPubPath)

        # cmd : move json file
        jsonFileCmd = ""
        if not jsonFilePath == "":
            jsonFileCmd = "{0} -vf {1} {2}".format(keyword, jsonFilePath, jsonPubPath)

        # cmd : move bvh file
        bvhFileCmd = ""
        if not bvhFilePath == "":
            bvhFileCmd = "{0} -vf {1} {2}".format(keyword, bvhFilePath, bvhPubPath)

        movFileCmd = ""
        if not movFile == "":
            movFileCmd = "cp -vf {0} {1}".format(movFile, movPubPath)

        # cmd : turn on enable record in database
        enableDBCmd = "python /netapp/backstage/pub/bin/inventory/enableDBRecord.py {0} {1} {2}".format(DBNAME,
                                                                                                        COLLNAME,
                                                                                                        self.dbPlugin.resultID)

        print enableDBCmd

        if isTractor:
            # 3-1 make directory
            tractor.addTask(parentTask=tractor.rootTask,
                            title="directory create command",
                            command=mkdirCmd)

            # 3-2 move preview file
            tractor.addTask(parentTask=tractor.rootTask,
                            title="move thumbnail",
                            command=mvPrevFileCmd)

            # 3-3 move or copys anim file
            if not animFilePath == "":
                tractor.addTask(parentTask=tractor.rootTask,
                                title="anim file copy",
                                command=animFileCmd)

            if not bvhFilePath == "":
                tractor.addTask(parentTask=tractor.rootTask,
                                title="bvh file copy",
                                command=bvhFileCmd)

                # 3-4 move json file
            if not jsonFilePath == "":
                tractor.addTask(parentTask=tractor.rootTask,
                                title="json file copy",
                                command=jsonFileCmd)

            if not movFile == "":
                tractor.addTask(parentTask=tractor.rootTask,
                                title="mov file copy",
                                command=movFileCmd)

            # 3-5 is upload success ? True : False
            tractor.addTask(parentTask=tractor.rootTask,
                            title="db enabled change true",
                            command=enableDBCmd)

            tractor.sendJobSpool()
        else:
            # try:
            # 3-1 make directory
            self.processSystemCmd(mkdirCmd)

            # 3-2 move preview file
            self.processSystemCmd(mvPrevFileCmd)

            # 3-3 move or copys anim file
            if not animFilePath == "":
                self.processSystemCmd(animFileCmd)

            # 3-4 move json file
            if not jsonFilePath == "":
                self.processSystemCmd(jsonFileCmd)

            if not movFile == "":
                self.processSystemCmd(bvhFileCmd)

            if not movFile == "":
                self.processSystemCmd(movFileCmd)

            # 3-5 is upload success ? True : False
            os.system(enableDBCmd)

            MessageBox("upload success.")

            for tabWidget in self.contentTabList.values():
                tabWidget.refreshUI()

                # self.moreLoadData()
                # except Exception as e:
                #     cmds.warning(e.message)

    def processSystemCmd(self, executeCmd):
        baseCmd = 'echo "dexter" | su render -c "%s"' % executeCmd
        result = os.system(baseCmd)
        if not result == 0:
            raise Exception("%s Fail...[%s]" % (executeCmd, baseCmd))

    def addTab(self, itemFullPath):
        tierSplitItem = itemFullPath.split('/')
        print tierSplitItem

        if len(tierSplitItem) < 4:
            return None

        tab = ContentTab(self.dbPlugin.getContentInfo(tierSplitItem[0],
                                                      tierSplitItem[1],
                                                      tierSplitItem[2],
                                                      tierSplitItem[3]), itemFullPath)

        tab.itemDoubleClicked.connect(self.sourceItemClicked)

        scaleValue = (self.ui.ContentScaleSlider.value() * 0.01) + 1
        tab.setIconScale(scaleValue)

        if not itemFullPath in self.contentTabList:
            tab.loadContent()
            self.ui.ContentTabWidget.addTab(tab, itemFullPath)
            self.contentTabList[itemFullPath] = tab
            return tab

        return self.contentTabList[itemFullPath]

    def aniFileDoubleClick(self, item, column):
        if item.childCount() != 0:
            return

        currentTab = self.addTab(item.itemFullPath)
        if currentTab == None:
            return
        self.ui.moreBtn.setDisabled(not currentTab.isMoreLoading)
        self.ui.moreBtn.setText('MoreBtn [%s / %s]' % (currentTab.currentLoadCount, currentTab.maxLoadCount))
        self.ui.ContentTabWidget.setCurrentWidget(currentTab)

    def sliderValueChanged(self, value):
        scaleValue = (value * 0.01) + 1
        for contentTab in self.contentTabList:
            self.contentTabList[contentTab].setIconScale(scaleValue)

    def tagSearchPressed(self):
        findCategory = {"$or": []}
        appendStr = ""
        if self.ui.aniCheckBox.isChecked():
            findCategory["$or"].append({'category': "Animation"})
            appendStr += "A"
        if self.ui.mcpCheckBox.isChecked():
            findCategory["$or"].append({'category': "Mocap"})
            appendStr += "M"
        if self.ui.crdCheckBox.isChecked():
            findCategory["$or"].append({'category': "Crowd"})
            appendStr += "C"

        if appendStr == "":
            MessageBox(u"팀을 선택해주세요.")
            return

        text = self.ui.tagSearchEdit.text()

        resultItem = self.dbPlugin.findForTag(text.split(' '), category=findCategory)

        text += appendStr

        # tierSplitItem = itemFullPath.split('/')
        tab = ContentTab(resultItem)
        tab.itemDoubleClicked.connect(self.sourceItemClicked)

        scaleValue = (self.ui.ContentScaleSlider.value() * 0.01) + 1
        tab.setIconScale(scaleValue)

        if not text in self.contentTabList:
            tab.loadContent()
            self.ui.moreBtn.setDisabled(not tab.isMoreLoading)
            self.ui.moreBtn.setText('MoreBtn [%s / %s]' % (tab.currentLoadCount, tab.maxLoadCount))
            self.ui.ContentTabWidget.addTab(tab, text)
            self.contentTabList[text] = tab
            self.ui.ContentTabWidget.setCurrentWidget(tab)
        else:
            currentTab = self.contentTabList[text]
            self.ui.moreBtn.setDisabled(not currentTab.isMoreLoading)
            self.ui.moreBtn.setText('MoreBtn [%s / %s]' % (currentTab.currentLoadCount, currentTab.maxLoadCount))
            self.ui.ContentTabWidget.setCurrentWidget(currentTab)

    def moreLoadData(self):
        tabWidget = self.ui.ContentTabWidget.currentWidget()
        tabWidget.loadContent()
        self.ui.moreBtn.setDisabled(not tabWidget.isMoreLoading)
        self.ui.moreBtn.setText('MoreBtn [%s / %s]' % (tabWidget.currentLoadCount, tabWidget.maxLoadCount))

    def sourceItemClicked(self, sourceItem):
        print sourceItem
        sourceId = dbConfig.getExistRelativeInfo(sourceItem.contentInfo['_id'])

        if sourceId == None:
            sourceId = sourceItem.contentInfo['_id']

        self.ui.relativeView.refreshUI(sourceId)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()