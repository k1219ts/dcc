#coding:utf-8
'''
@ author    : daeseok.chae
@ date      : 2018.11.22
@ comment   : Animbrower에 BVH 소스를 등록시켜주는 클래스.
'''
import os
import getpass
# import Qt
from PySide2 import QtWidgets
import shutil

# if "Side" in Qt.__binding__:
import maya.cmds as cmds

from ExportSourceDialogUI import Ui_Form
from MessageBox import MessageBox

from MongoDB import MongoDB
import AnimBrowser.retargetting.bvhExporter as bvhExporter
import Zelos
from MakeMov import MakeMov

DBNAME = "inventory"
COLLNAME = "anim_tags"
CURRENTDIR = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
retargetPath = "/dexter/Cache_DATA/RND/jeongmin/AnimBrowser_retargetting"

class ExportSourceDialog(QtWidgets.QDialog):
    def __init__(self, parent):
        self.parent = parent
        QtWidgets.QDialog.__init__(self, parent)

        self.ui = Ui_Form()
        self.ui.setupUi(self)

        mayaWindow = parent
        self.move(mayaWindow.frameGeometry().center() - self.frameGeometry().center())

        self.dbPlugin = MongoDB(DBNAME, COLLNAME)

        self.bvhfilepath = ""
        self.animfilepath = ""

        # Connect Signal
        self.ui.Title1ComboBox.currentIndexChanged.connect(self.currentTitle1IndexChange)
        self.ui.Title2ComboBox.currentIndexChanged.connect(self.currentTitle2IndexChange)

        self.initializeDataSet(0, "Animation")
        self.ui.startFrameLineEdit.setText( str( cmds.playbackOptions( q=1, min=1)) )
        self.ui.endFrameLineEdit.setText( str( cmds.playbackOptions( q=1, max=1)))

        self.ui.okBtn.clicked.connect(self.okBtnClick)
        self.ui.cancelBtn.clicked.connect(self.closeBtnClick)

    def initializeDataSet(self, startIndex, category):
        # tag 1 tier setting
        self.tagData = self.dbPlugin.getTagData(category)
        self.tagDict = {}
        for tag1Tier in self.tagData:
            if not self.tagDict.has_key(tag1Tier["tag_tier1"]):
                self.tagDict[tag1Tier["tag_tier1"]] = {}

            if not tag1Tier.has_key("tag_tier2"):
                continue
            elif not self.tagDict[tag1Tier["tag_tier1"]].has_key(tag1Tier["tag_tier2"]):
                self.tagDict[tag1Tier["tag_tier1"]][tag1Tier["tag_tier2"]] = list()

            if not tag1Tier.has_key("tag_tier3"):
                continue
            else:
                self.tagDict[tag1Tier["tag_tier1"]][tag1Tier["tag_tier2"]].append(tag1Tier["tag_tier3"])

        self.ui.Title1ComboBox.clear()

        for key in self.tagDict.keys():
            item = QtWidgets.QTreeWidgetItem()
            item.setText(0, key)
            for key2 in self.tagDict[key].keys():
                item2 = QtWidgets.QTreeWidgetItem(item)
                item2.setText(0, key2)
                for key3 in self.tagDict[key][key2]:
                    item3 = QtWidgets.QTreeWidgetItem(item2)
                    item3.setText(0, key3)

            self.ui.Title1ComboBox.addItem(key)

        self.ui.Title1ComboBox.setCurrentIndex(startIndex)

    def currentTitle1IndexChange(self, index):
        try:
            self.ui.Title2ComboBox.clear()

            self.ui.Title2ComboBox.addItems(self.tagDict[self.ui.Title1ComboBox.currentText()].keys())
        except:
            pass

    def currentTitle2IndexChange(self, index):
        try:
            self.ui.Title3ComboBox.clear()

            self.ui.Title3ComboBox.addItems(self.tagDict[self.ui.Title1ComboBox.currentText()][self.ui.Title2ComboBox.currentText()])
        except:
            pass

    # Load Maya File Path
    def getOpenFile(self, titleCaption, startDirPath, exrCaption):
        fileName = ""
        if not "Side" in Qt.__binding__:
            fileName = QtWidgets.QFileDialog.getOpenFileName(self, titleCaption, startDirPath, exrCaption)
        else:
            fileName = cmds.fileDialog2(fileMode=1,
                                        caption=titleCaption,
                                        okCaption="Load",
                                        startingDirectory=startDirPath,
                                        fileFilter=exrCaption)
            if fileName == None:
                return None
            fileName = str(fileName[0])
        return fileName

    # ok Btn Click
    def okBtnClick(self):
        basePath = os.path.join( retargetPath, getpass.getuser() )
        charPath = "{0}/{1}/{2}".format(self.ui.Title1ComboBox.currentText(),
                                        self.ui.Title2ComboBox.currentText(),
                                        self.ui.Title3ComboBox.currentText())

        fileName = self.ui.Title2ComboBox.currentText()

        bvhfilepath = os.path.join(basePath, charPath, fileName+".bvh")
        animfilepath = os.path.join(basePath, charPath, fileName+".anim")


        print "# EXPORT SOURCE DIALOG : export source bvh file to: ", bvhfilepath
        print "# EXPORT SOURCE DIALOG : export source anim file to: ", animfilepath

        # export bvh C_skin_root_JNT
        selection = cmds.ls(sl=True, type="joint") #rootJoint
        rootJnt = ""
        if not selection:
            MessageBox("first, select root_JNT")
            return
        else:
            for i in selection:
                if "root_JNT" in i:
                    rootJnt = i
                    break

        if not rootJnt:
            MessageBox("select root_JNT")
            return

        '''
        # initialize scale 1
        placecon = ns + 'place_CON'
        if cmds.objExists(placecon):
            initscale = globalscale = 1
            if 'initScale' in cmds.listAttr(placecon):
                initscale = cmds.getAttr('%s.initScale'%placecon)
                initscaleLock = cmds.getAttr('%s.initScale'%placecon, l=True)
                if initscaleLock:
                    cmds.setAttr('%s.initScale'%placecon, l=False)
                cmds.setAttr('%s.initScale'%placecon, 1 )
                if initscaleLock:
                    cmds.setAttr('%s.initScale'%placecon, l=True)
            if 'globalScale' in cmds.listAttr(placecon):
                globalscale = cmds.getAttr('%s.globalScale'%placecon)
                globalscaleLock = cmds.getAttr('%s.globalScale'%placecon, l=True)
                if globalscaleLock:
                    cmds.setAttr('%s.globalScale'%placecon, l=False)
                cmds.setAttr('%s.globalScale'%placecon, 1 )
                if globalscaleLock:
                    cmds.setAttr('%s.globalScale'%placecon, l=True)
        '''

        start = int(self.getStartFrame())
        end = int(self.getEndFrame())
        exporter = bvhExporter.BVHExporter()
        exporter.generate_joint_data( selection, bvhfilepath, start, end )

        #export anim
        if os.path.exists(bvhfilepath):
            skeleton = Zelos.skeleton()
            skeleton.load(str(bvhfilepath))
            skeleton.save(str(animfilepath))
            self.close()
            print "# EXPORT SOURCE DIALOG : bvh, anim export success"
        else:
            print "# EXPORT SOURCE DIALOG : bvh, anim export failed"
            return

        self.bvhfilepath = bvhfilepath
        self.animfilepath = animfilepath
        self.exportRetargetSource()
        '''
        if cmds.objExists(placecon):
            if 'initScale' in cmds.listAttr(placecon):
                cmds.setAttr('%s.initScale'%placecon, initscale )
            if 'globalScale' in cmds.listAttr(placecon):
                cmds.setAttr('%s.globalScale'%placecon, globalscale )
        '''
        return

    def exportRetargetSource(self):
        if not cmds.pluginInfo( "ZArachneForMaya", q=True, l=True ):
            cmds.loadPlugin( "ZArachneForMaya" )

        category = "Animation"
        title1Tier = self.ui.Title1ComboBox.currentText()
        title2Tier = self.ui.Title2ComboBox.currentText()
        title3Tier = self.ui.Title3ComboBox.currentText()
        isTractor = self.ui.tractorCheckBox.isChecked()

        checkDB = {}
        checkDB["category"] = category
        checkDB["tag1tier"] = title1Tier
        checkDB["tag2tier"] = title2Tier
        checkDB["tag3tier"] = title3Tier

        fileNum = self.dbPlugin.existDocument(checkDB) + 1

        # step 2 : make preview gif
        baseDirPath = os.path.join(retargetPath, getpass.getuser() )
        playblastMov = MakeMov(baseDirPath, category, title1Tier, title2Tier, title3Tier, fileNum)
        movFileName = playblastMov.takePlayblast()
        gifFilePath = playblastMov.movToGif(movFileName)

        # step 3 : publish
        self.parent.publish(category=category,
                             tag1TierTitle = self.ui.Title1ComboBox.currentText(),
                             tag2TierTitle = self.ui.Title2ComboBox.currentText(),
                             tag3TierTitle = self.ui.Title3ComboBox.currentText(),
                             animFilePath = self.animfilepath,
                             bvhFilePath = self.bvhfilepath,
                             previewFilePath = gifFilePath,
                             isHIK = 0,
                             isTractor = isTractor,
                             movFile=movFileName)

        # clean
        if os.path.exists(baseDirPath):
            shutil.rmtree(baseDirPath)
            print 'clean up'

        return True

    # UI getFunction
    def getEndFrame(self):
        return float(self.ui.endFrameLineEdit.text())

    def getStartFrame(self):
        return float(self.ui.startFrameLineEdit.text())

    # ok Btn Click
    def closeBtnClick(self):
        self.close()
