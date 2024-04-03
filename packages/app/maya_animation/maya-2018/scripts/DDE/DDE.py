# encoding=utf-8
#!/usr/bin/env python
#-------------------------------------------------------------------------------
#   Dexter Rigging&CFX Part minjeong.kim
#	
#   2018.04.02
#-------------------------------------------------------------------------------

import maya.cmds as cmds
import maya.mel as mel
import os

import PySide2.QtGui as QtGui
import PySide2.QtCore as QtCore
import PySide2.QtWidgets as QtWidgets
import PySide2.QtUiTools as QtUiTools

import dxRigUI as drg;
import Quad_retargeting_MD as Quad_retargeting_MD; reload(Quad_retargeting_MD)
import Bip_retargeting_MD as Bip_retargeting_MD; reload(Bip_retargeting_MD)

import maya.OpenMayaUI as omui
from shiboken2 import wrapInstance

BASE_PATH = os.path.abspath('%s/../' % __file__)
UI_FILE = os.path.join(BASE_PATH, 'DDEUI.ui')
loadUI = QtUiTools.QUiLoader()

QUAD = Quad_retargeting_MD.QUADRETARGET()
BIP = Bip_retargeting_MD.BIPADRETARGRT()

def getMayaWin():
    ptr = omui.MQtUtil.mainWindow()
    return wrapInstance(int(ptr), QtWidgets.QMainWindow)


class DDEUI(QtWidgets.QDialog):

    def __init__(self,parent=getMayaWin()):
        super(DDEUI, self).__init__(parent)
        self.ui = loadUI.load(UI_FILE)
        self.ui.setWindowFlags(QtCore.Qt.Window)
        
        mel.eval('HIKCharacterControlsTool;')
        self.ui.setWindowTitle('DD-E')
        # Direct Delivery Else character
        self.startFr = str(cmds.playbackOptions(q=1, minTime=True))
        self.endFr = str(cmds.playbackOptions(q=1, maxTime=True) + 1)
        self.ui.startFrame_textEdit.setText(self.startFr)
        self.ui.endFrame_textEdit.setText(self.endFr)
        
        self.callConnections()
        self.set_handGroundCtrl()
        self.toggle_handGround()
    
    def callConnections(self):
        self.ui.sourceNS_pushButton.clicked.connect(self.selectSource)
        self.ui.targetNS_pushButton.clicked.connect(self.selectTarget)
        self.ui.link_pushButton.clicked.connect(self.hikLink)
        self.ui.timeReset_pushButton.clicked.connect(self.resetTime)
        self.ui.bake_pushButton.clicked.connect(self.keyBake)
        self.ui.reset_pushButton.clicked.connect(self.reset)
        self.ui.dde_pushButton.clicked.connect(self.helpCom)
        self.ui.biped_radioButton.clicked.connect(lambda: self.toggle_handGround())
        self.ui.quadruped_radioButton.clicked.connect(lambda: self.toggle_handGround(True))

    def set_handGroundCtrl(self):

        layout = self.ui.handBottom_horizontalLayout
        self.ground_label = QtWidgets.QLabel("handGroundCtrl")
        self.ground_value = QtWidgets.QLineEdit()
        self.ground_value.setMaximumWidth(50)
        self.ground_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.ground_slider.setRange(0,1000)
        self.ground_slider.valueChanged.connect(self.update_sliderValue)
        self.ground_value.textChanged.connect(self.update_textValue)
        self.ground_value.textChanged.connect(self.connect_HIKgroundAttr)
        self.ground_value.setText("0.5")
        font = QtGui.QFont()
        font.setPointSize(10)
        self.ground_label.setFont(font)
        layout.addWidget(self.ground_label)
        layout.addWidget(self.ground_value)
        layout.addWidget(self.ground_slider)

    def toggle_handGround(self, vis = False ):
        if vis == False:
            self.ground_label.setHidden(True)
            self.ground_value.setHidden(True)
            self.ground_slider.setHidden(True)
        else:
            self.ground_label.setVisible(True)
            self.ground_value.setVisible(True)
            self.ground_slider.setVisible(True)
    
    def update_sliderValue(self,value):
        
        self.ground_value.setText(str(value / 1000.0))
    
    def update_textValue(self, text):
        try:
            value = float(text)
            int_value = int(value * 1000.0)
            self.ground_slider.setValue(int_value)
        except: pass

    def connect_HIKgroundAttr(self, text):
        try:
            cmds.setAttr("%s:HIKproperties1.HandBottomToWrist" %(self.targetNS), float(text))
        except: pass
        
    def selectSource(self):
        rig  = cmds.ls(sl= True )[0]
        rig_typeNum = cmds.getAttr("%s.rigType"% rig)
        self.sourceNS = rig.split(":")[0]
        self.ui.source_lineEdit.setText(self.sourceNS)
        if rig_typeNum == 0:
            self.ui.biped_radioButton.setChecked(True)
            self.toggle_handGround()
            BIP.defineSource(self.sourceNS)
        elif rig_typeNum == 1:
            self.ui.quadruped_radioButton.setChecked(True)
            self.toggle_handGround(True)
            QUAD.defineSource(self.sourceNS)
        else:
            cmds.error("Check Rig Type")

    def selectTarget(self):
        rig  = cmds.ls(sl= True )[0]
        self.targetNS = rig.split(":")[0]
        self.ui.target_lineEdit.setText(self.targetNS)
        if self.ui.biped_radioButton.isChecked():
            BIP.defineTarget(self.targetNS)
        elif self.ui.quadruped_radioButton.isChecked():
            QUAD.defineTarget(self.targetNS)
        
    def hikLink(self):

        if self.ui.biped_radioButton.isChecked():
            BIP.defineSource(self.sourceNS)
            BIP.defineTarget(self.targetNS)
        elif self.ui.quadruped_radioButton.isChecked():
            QUAD.defineSource(self.sourceNS)
            QUAD.defineTarget(self.targetNS)
        
        gmainPane = mel.eval('global string $gMainPane; $temp =$gMainPane;')
        cmds.paneLayout(gmainPane, e = 1, manage = 0 )
        self.restFrame = cmds.playbackOptions(q=1, minTime=True) - 50
        cmds.currentTime(self.restFrame)
        Loc = cmds.spaceLocator(n = "%s:originLegLength_LOC" % self.targetNS)[0]
        cmds.matchTransform(Loc , "%s:C_Skin_chest_JNT" % self.targetNS)
        cmds.select(self.sourceNS+":"+"*"+"_rig_GRP", self.targetNS+":"+"*"+"_rig_GRP", r = 1)
        self.setDefaultPose()
        cmds.select(self.sourceNS+":"+"*"+"CON", r= 1)
        cmds.setKeyframe()

        self.importHIK("source")
        self.importHIK("target")
        
        self.sourceHIK = self.sourceNS + ':set_HIK'
        self.targetHIK = self.targetNS + ':set_HIK'
        if self.ui.biped_radioButton.isChecked():
            if self.ui.toFK_radioButton.isChecked() == 1:
                BIP.HIKtoFK()
                mel.eval('refreshAllCharacterLists();')
                mel.eval('mayaHIKsetCharacterInput("%s","%s");' %
                    (self.targetHIK, self.sourceHIK))
            elif self.ui.toIK_radioButton.isChecked() == 1:
                BIP.HIKtoIK()
                mel.eval('refreshAllCharacterLists();')
                mel.eval('mayaHIKsetCharacterInput("%s","%s");' %
                    (self.targetHIK, self.sourceHIK))
                BIP.IKPOVBake()
        elif self.ui.quadruped_radioButton.isChecked():
            QUAD.HIKtoIK()
            mel.eval('refreshAllCharacterLists();')
            mel.eval(
                'mayaHIKsetCharacterInput("%s","%s");' %
                (self.targetHIK, self.sourceHIK))
            mel.eval('refreshAllCharacterLists();')
            nodeName =self.sourceNS+":"+"*"+"_rig_GRP.controllers"
            cmds.currentTime(self.restFrame)
            drg.controllersInit(nodeName)
            cmds.select(self.sourceNS+":"+"*"+"CON", r= 1)
            cmds.setKeyframe()
            QUAD.set_IKrollCon()
            QUAD.set_HIKproperties()
            if self.ui.toFK_radioButton.isChecked() == 1:
                QUAD.HIKtoFK()
            else:
                pass
        cmds.hide("%s:HIKJoint_LOC" % self.sourceNS, "%s:HIKJoint_LOC" % self.targetNS )
        cmds.currentTime(self.startFr)
        cmds.paneLayout(gmainPane, e = 1, manage = 1 )
        

    def setDefaultPose(self):
        rig_list = cmds.ls(sl =True)
        for rig in rig_list:
            if self.ui.quadruped_radioButton.isChecked():
                if ":" in rig:
                    ns_name = rig.split(":")[0]
                    rootTpose = "%s:C_IK_root_NUL.tPoseValue" %ns_name
                else:
                    rootTpose = "C_IK_root_NUL.tPoseValue"
                    ns_name = None
                # if cmds.objExists(rootTpose):
                data = cmds.getAttr(rootTpose)
                if not data:
                    print (" Not found controllers initialize data")
                    return
                data = eval(data)
                for i in data:
                    nodeName = i
                    if nodeName == "place_CON.initScale":
                        pass
                    else:
                        if ns_name:
                            nodeName = '%s:%s' % (ns_name, i)
                            try:
                                if data[i]['type'] == 'string':
                                    cmds.setAttr( nodeName, data[i]['value'], type='string' )
                                else:
                                    cmds.setAttr( nodeName, data[i]['value'] )
                            except: pass
                        else:
                            try:
                                if data[i]['type'] == 'string':
                                    cmds.setAttr( nodeName, data[i]['value'], type='string' )
                                else:
                                    cmds.setAttr( nodeName, data[i]['value'] )
                            except:
                                    pass
            else:
                nodeName ="%s.controllers"%rig
                drg.controllersInit(nodeName)
 
    def importHIK(self, type):
        if type == "source":
            if self.ui.biped_radioButton.isChecked():
                BIP.getSourceHIK()
            elif self.ui.quadruped_radioButton.isChecked():
                QUAD.getSourceHIK()
        if type == "target":
            if self.ui.biped_radioButton.isChecked():
                BIP.getTargetHIK()
            elif self.ui.quadruped_radioButton.isChecked():
                QUAD.getTargetHIK()
    
    def resetTime(self):
        self.ui.startFrame_textEdit.clear()
        self.ui.endFrame_textEdit.clear()
        self.startFr = str(cmds.playbackOptions(q=1, minTime=True))
        self.endFr = str(cmds.playbackOptions(q=1, maxTime=True) + 1)
        self.ui.startFrame_textEdit.setText(self.startFr)
        self.ui.endFrame_textEdit.setText(self.endFr)

    def keyBake(self):
        startFr = self.ui.startFrame_textEdit.text()
        endFr = self.ui.endFrame_textEdit.text()
        cmds.currentTime(startFr)
        if self.ui.worldCon_checkBox.isChecked():
            if self.ui.biped_radioButton.isChecked():
                BIP.muteWorldCon()
            elif self.ui.quadruped_radioButton.isChecked():
                QUAD.muteWorldCon()
        else:
            pass
        if self.ui.biped_radioButton.isChecked():
            if self.ui.toFK_radioButton.isChecked() == 1:
                BIP.selectControler("FK")
            elif self.ui.toIK_radioButton.isChecked() == 1:
                BIP.selectControler("IK")
            cmds.select( "%s:move_CON" %self.targetNS, add=1)
            cmds.bakeResults(t=(str(startFr), str(endFr)), simulation=True, sb=1)
            BIP.deleteAll()
        elif self.ui.quadruped_radioButton.isChecked():
            if self.ui.toFK_radioButton.isChecked() == 1:
                QUAD.selectControler("FK")
            elif self.ui.toIK_radioButton.isChecked() == 1:
                QUAD.selectControler("IK")
            cmds.select( "%s:move_CON" %self.targetNS, add=1)
            cmds.bakeResults(t=(str(startFr), str(endFr)), simulation=True, sb=1)
            QUAD.deleteAll() 
        for ns in [ self.targetNS, self.sourceNS ]:
            if cmds.objExists(ns+':HIKJoint_LOC'):
                mel.eval('refreshAllCharacterLists();')
                mel.eval('deleteCharacter( "%s:set_HIK" );'%(ns)) 
                cmds.delete(ns+':HIKJoint_LOC' )
            else: pass
        nodeName =self.targetNS+":"+"*"+"_rig_GRP.controllers"
        drg.controllersInit(nodeName)
        cmds.select(self.targetNS+":"+"*"+"CON", r= 1)
        cmds.filterCurve()
        if self.ui.toFK_radioButton.isChecked() == 1:
            switch_list = [
                'R_foreLeg_switch_CON.IKFKBlend', 'L_foreLeg_switch_CON.IKFKBlend',
                'R_hindLeg_switch_CON.IKFKBlend', 'L_hindLeg_switch_CON.IKFKBlend']
            if cmds.objExists('%s:%s' %(self.targetNS,switch_list[0])):
                for a in range(len(switch_list)):
                    cmds.setAttr('%s:%s' %(self.targetNS,switch_list[a]),0)
            elif cmds.objExists('%s:%s.addFK' %(self.targetNS,self.POV_set_List[0])):
                for con in range(len(self.POV_set_List)):
                    cmds.setAttr('%s:%s.addFK' %(self.targetNS,self.POV_set_List[con]),1)

    def reset(self):
        cmds.currentTime(self.startFr)
        if self.ui.biped_radioButton.isChecked():
            BIP.deleteAll()
        elif self.ui.quadruped_radioButton.isChecked():
            QUAD.deleteAll()
        nodeName = self.targetNS +":"+"*"+"_rig_GRP.controllers"
        drg.controllersInit(nodeName)
        for ns in [ self.targetNS, self.sourceNS ]:
            if cmds.objExists(ns+':HIKJoint_LOC'):
                mel.eval('refreshAllCharacterLists();')
                mel.eval('deleteCharacter( "%s:set_HIK" );'%(ns)) 
                cmds.delete(ns+':HIKJoint_LOC' )
            else: pass
        try:
            nodeName =self.sourceNS+":"+"*"+"_rig_GRP.controllers"
            drg.controllersInit(nodeName)
            cmds.select(self.sourceNS+":"+"*"+"CON", r= 1)
            cmds.currentTime(self.restFrame)
            mel.eval("timeSliderClearKey;")
        except: pass
        nodeName = self.targetNS +":"+"*"+"_rig_GRP.controllers"
        drg.controllersInit(nodeName)
        gmainPane = mel.eval('global string $gMainPane; $temp =$gMainPane;')
        if cmds.paneLayout(gmainPane, q =1, manage = 1 ) == False:
            cmds.paneLayout(gmainPane, e = 1, manage = 1 )
        cmds.currentTime(self.startFr)
        
    def helpCom(self):
        
        ma = "▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬\nDD-E TOOL 사용전 주의 사항\n▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬\n1. 네임스페이스를 가진 레퍼런스 리그만 불러 옵니다. \n2. 리그들은 원점에서 같은 레스트 포즈를 지녀야 합니다. 사족리그는 T포즈가 저장되어 있어야 합니다. \n3. 두 리그 다 +Z축 방향을 바라보고 있어야 합니다. \n4. 타겟 리그에 애니메이션 키가 잡혀있지 않은지 확인하세요. \n▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬"
        cmds.confirmDialog(title = "help",message = (ma),b = "ok")

def openUI():
    global dde
    try:
        dde.close()
        dde.deleteLater()
    except:
        pass
    dde = DDEUI()
    dde.ui.show()

