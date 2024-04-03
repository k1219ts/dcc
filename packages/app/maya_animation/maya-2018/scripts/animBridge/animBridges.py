# encoding:utf-8
# !/usr/bin/env python

import os
import maya.cmds as cmds
import maya.mel as mel
import json
import math
import subprocess
from pymodule.Qt import QtCore, QtGui, QtWidgets, QtCompat

mel.eval("cycleCheck -e off")
currentpath = os.path.abspath(__file__)
uiFile = os.path.join(os.path.dirname(currentpath), "animBridge_moon.ui")

class undoCheck(object):
    def __enter__(self):
        cmds.undoInfo(openChunk=True)
    def __exit__(self, *exc):
        cmds.undoInfo(closeChunk=True)

def setup_ui(ui, base_instance=None):
    for member in dir(ui):
        if not member.startswith('__') and member is not 'staticMetaObject':
            setattr(base_instance, member, getattr(ui, member))

class Window(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        ui = QtCompat.load_ui(uiFile)
        setup_ui(ui, self)
        mel.eval('HIKCharacterControlsTool;')
        # attatch List
        self.connectSignal()
        self.moduleJoint_list = ['C_Skin_hip_JNT', 'C_Skin_spine1_JNT', 'C_Skin_spine2_JNT', 'C_Skin_spine3_JNT',
                                 'C_Skin_chest_JNT', 'C_Skin_neck_JNT', 'C_Skin_head_JNT', 'L_Skin_shoulder_JNT',
                                 'L_Skin_upArm_JNT', 'L_Skin_foreArm_JNT', 'L_Skin_hand_JNT', 'L_Skin_middle1_JNT',
                                 'R_Skin_shoulder_JNT', 'R_Skin_upArm_JNT', 'R_Skin_foreArm_JNT', 'R_Skin_hand_JNT',
                                 'R_Skin_middle1_JNT', 'L_Skin_leg_JNT', 'L_Skin_lowLeg_JNT', 'L_Skin_foot_JNT',
                                 'L_Skin_ball_JNT', 'R_Skin_leg_JNT', 'R_Skin_lowLeg_JNT', 'R_Skin_foot_JNT',
                                 'R_Skin_ball_JNT']
        self.HIKJoint_list = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head', 'LeftShoulder', 'LeftArm',
                              'LeftForeArm', 'LeftHand', 'LeftFingerBase', 'RightShoulder', 'RightArm', 'RightForeArm',
                              'RightHand', 'RightFingerBase', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase',
                              'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase']
        self.mocapJoint_list = ['Crw_Hips', 'Crw_Spine', 'Crw_Spine1', 'Crw_Spine2', 'Crw_Spine3', 'Crw_Neck',
                                'Crw_Head', 'Crw_LeftShoulder', 'Crw_LeftArm', 'Crw_LeftForeArm', 'Crw_LeftHand',
                                'Crw_LefthandSub1', 'Crw_RightShoulder', 'Crw_RightArm', 'Crw_RightForeArm',
                                'Crw_RightHand', 'Crw_RightHandSub1', 'Crw_LeftUpLeg', 'Crw_LeftLeg', 'Crw_LeftFoot',
                                'Crw_LeftToeBase', 'Crw_RightUpLeg', 'Crw_RightLeg', 'Crw_RightFoot',
                                'Crw_RightToeBase']
        self.FKController_list = ['root_CON', 'C_IK_lowBody_CON', 'C_IK_upBodyRot1_CON', 'C_IK_upBodyRot2_CON', 'C_IK_upBody_CON', 'C_IK_neck_CON', 'C_IK_head_CON', 'L_FK_shoulder_CON', 'L_FK_upArm_CON', 'L_FK_foreArm_CON', 'L_FK_hand_CON', 'R_FK_shoulder_CON', 'R_FK_upArm_CON', 'R_FK_foreArm_CON', 'R_FK_hand_CON', 'L_FK_leg_CON', 'L_FK_lowLeg_CON', 'L_FK_foot_CON', 'R_FK_leg_CON', 'R_FK_lowLeg_CON', 'R_FK_foot_CON']
        self.HIK_attatch_toFK_list = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'RightUpLeg', 'RightLeg', 'RightFoot']
        self.IKController_list = ['root_CON', 'C_IK_lowBody_CON', 'C_IK_upBodyRot1_CON', 'C_IK_upBodyRot2_CON', 'C_IK_upBody_CON', 'C_IK_neck_CON', 'C_IK_head_CON', 'L_FK_shoulder_CON', 'L_IK_hand_CON', 'R_FK_shoulder_CON', 'R_IK_hand_CON', 'L_IK_foot_CON', 'R_IK_foot_CON', 'L_IK_handVec_CON', 'R_IK_handVec_CON', 'L_IK_footVec_CON', 'R_IK_footVec_CON']
        self.HIK_attatch_toIK_list = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head', 'LeftShoulder', 'LeftHand', 'RightShoulder', 'RightHand', 'LeftFoot', 'RightFoot']
        self.IKPOV_list = [['LeftUpLeg', 'LeftLeg', 'LeftFoot', 'L_IK_footVec_CON'],
                           ['RightUpLeg', 'RightLeg', 'RightFoot', 'R_IK_footVec_CON'],
                           ['LeftArm', 'LeftForeArm', 'LeftHand', 'L_IK_handVec_CON'],
                           ['RightArm', 'RightForeArm', 'RightHand', 'R_IK_handVec_CON']]
        self.place_list = ['move_CON', 'direction_CON', 'place_CON']
        self.trAttr = ['translateX', 'translateY', 'translateZ', 'rotateX', 'rotateY', 'rotateZ']
        animPlug = '/usr/autodesk/maya2017/bin/plug-ins/animImportExport.so'
        if cmds.pluginInfo(animPlug, q=True, l=True) == False:
            try:
                cmds.loadPlugin(animPlug)
                cmds.pluginInfo(animPlug, edit=True, autoload=True)
            except:
                pass
        self.delConstraints = []
        self.targetNS = []

    def connectSignal(self):
#        self.exAnim.clicked.connect(self.getHIK)
        self.mocapBtn.clicked.connect(self.getMcp)
#        self.run_BTN.clicked.connect(self.assignHIK)
#        self.makeTp_BTN.clicked.connect(self.makeTpose)
        self.asMcBtn.clicked.connect(self.importMocapAnim)
#        self.htmBtn.clicked.connect(self.hTom)
#        self.offBtn.clicked.connect(self.offSetFrames)
        self.pvls.stateChanged.connect(self.chkState)
        self.itfBtn.clicked.connect(self.itfMain)
        self.tutBtn.setIcon(QtGui.QIcon(QtGui.QPixmap("/netapp/backstage/pub/apps/maya2/versions/2017/team/animation/linux/icons/system-help.png")))
        self.tutBtn.setIconSize(QtCore.QSize(23, 23))
        self.tutBtn.clicked.connect(self.tut)
#        self.inputTfr.setText("950")
#        self.inActStt.setText("1001")
        self.inputSF.setText("1001")

    def tut(self):
        exp = "/usr/bin/evince"
        fileName = "/netapp/backstage/pub/apps/maya2/versions/2017/team/animation/linux/scripts/animBridge/animBridge.pdf"
        subprocess.Popen([exp, fileName])

    def getNamespace(self):
        if cmds.ls(sl=1):
            sel = cmds.ls(sl=True)[0]
            self.nsChar = str(sel.rsplit(':')[0])
            return self.nsChar
        else:
            cmds.warning('"select Character!!"')

##### 월드컨트롤러 위치값을 저장하고 현재 위치값을 원점으로 설정 #############################################################

    def getWorldPostion(self):

        atts = ("tx","ty","tz","rx","ry","rz")
        pos = []
        nsChar = self.getNamespace()
        for att in atts:
            value = cmds.getAttr(nsChar + ":place_CON." + att)
            pos.append(value)
            cmds.setAttr(nsChar + ":place_CON." + att, 0)

        return pos            

########################################################################################

    def IKPOVBake(self, POV_set_List=None):
        self.timelineSet()
        nsChar = [self.nsChar]
        if POV_set_List == None:
            POV_set_List = ['LeftLeg', 'RightLeg', 'LeftArm', 'RightArm']
        else:
            pass
        decomposedMatrix_list = []
        plusMinusAverage_list = []
        multiplyDivide_list = []
        delCon = []
        for i in range(len(POV_set_List)):
            del decomposedMatrix_list[0:len(decomposedMatrix_list)]
            del plusMinusAverage_list[0:len(plusMinusAverage_list)]
            del multiplyDivide_list[0:len(multiplyDivide_list)]
            for x in range(4):
                decomposed_node = cmds.createNode('decomposeMatrix', n='%s:%s_%s_DCM' % (nsChar[0], POV_set_List[i], x + 1))
                decomposedMatrix_list.append(decomposed_node)
                delCon.append(decomposed_node)
            for y in range(6):
                plusMinusAverage_node = cmds.createNode('plusMinusAverage', n='%s:%s_%s_PMA' % (nsChar[0], POV_set_List[i], y + 1))
                if y <= 3:
                    cmds.setAttr('%s.operation' % plusMinusAverage_node, 2)
                plusMinusAverage_list.append(plusMinusAverage_node)
                delCon.append(plusMinusAverage_node)
            for z in range(2):
                multiplyDivide_node = cmds.createNode('multiplyDivide', n='%s:%s_%s_MPD' % (nsChar[0], POV_set_List[i], z + 1))
                cmds.setAttr('%s.input2X' % multiplyDivide_node, 2)
                cmds.setAttr('%s.input2Y' % multiplyDivide_node, 2)
                cmds.setAttr('%s.input2Z' % multiplyDivide_node, 2)
                if z <= 0:
                    cmds.setAttr('%s.operation' % multiplyDivide_node, 2)
                multiplyDivide_list.append(multiplyDivide_node)
                delCon.append(multiplyDivide_node)
            composeMatrix_node = cmds.createNode('composeMatrix', n='%s:%s_%s_CPM' % (nsChar[0], POV_set_List[i], i + 1))
            delCon.append(composeMatrix_node)
            multMatrix_node = cmds.createNode('multMatrix', n='%s:%s_%s_MMX' % (nsChar[0], POV_set_List[i], i + 1))
            delCon.append(multMatrix_node)
            for a in range(3):
                cmds.connectAttr('%s:%s.worldMatrix' % (nsChar[0], self.IKPOV_list[i][a]), '%s.inputMatrix' % decomposedMatrix_list[a])
                cmds.connectAttr('%s.outputTranslate' % decomposedMatrix_list[a], '%s.input3D[0]' % plusMinusAverage_list[a])
            cmds.connectAttr('%s.output3D' % plusMinusAverage_list[0], '%s.input3D[0]' % plusMinusAverage_list[4])
            cmds.connectAttr('%s.output3D' % plusMinusAverage_list[2], '%s.input3D[1]' % plusMinusAverage_list[4])
            cmds.connectAttr('%s.output3D' % plusMinusAverage_list[4], '%s.input1' % multiplyDivide_list[0])
            cmds.connectAttr('%s.output' % multiplyDivide_list[0], '%s.input3D[1]' % plusMinusAverage_list[3])
            cmds.connectAttr('%s.output3D' % plusMinusAverage_list[1], '%s.input3D[0]' % plusMinusAverage_list[3])
            cmds.connectAttr('%s.output3D' % plusMinusAverage_list[3], '%s.input1' % multiplyDivide_list[1])
            cmds.connectAttr('%s.output' % multiplyDivide_list[1], '%s.input3D[0]' % plusMinusAverage_list[5])
            cmds.connectAttr('%s.output' % multiplyDivide_list[0], '%s.input3D[1]' % plusMinusAverage_list[5])
            cmds.connectAttr('%s.output3D' % plusMinusAverage_list[5], '%s.inputTranslate' % composeMatrix_node)
            cmds.connectAttr('%s.outputMatrix' % composeMatrix_node, '%s.matrixIn[0]' % multMatrix_node)
            cmds.connectAttr('%s.matrixSum' % multMatrix_node, '%s.inputMatrix' % decomposedMatrix_list[3])
            cmds.connectAttr('%s.outputTranslate' % decomposedMatrix_list[3], '%s:%s.translate' % (nsChar[0], self.IKPOV_list[i][3]))
            cmds.connectAttr('%s:%s.parentInverseMatrix' % (nsChar[0], self.IKPOV_list[i][3]), '%s.matrixIn[1]' % multMatrix_node)  ## IK poleVector Bake
        cmds.currentTime(cmds.playbackOptions(q=True, minTime=True))
        cmds.select(cl=True)
        if 'RightArm' in POV_set_List:
            polVec = {self.nsChar + ":RightArm_4_DCM": self.nsChar + ":R_IK_handVec_CON", self.nsChar + ":LeftArm_4_DCM": self.nsChar + ":L_IK_handVec_CON", self.nsChar + ":RightLeg_4_DCM": self.nsChar + ":R_IK_footVec_CON", self.nsChar + ":LeftLeg_4_DCM": self.nsChar + ":L_IK_footVec_CON"}
        else:
            polVec = {}
            povNd = {"L_FK_lowLeg_JNT": {self.nsChar + ":L_FK_lowLeg_JNT_4_DCM": self.nsChar + ":L_IK_footVec_CON"}, "R_FK_lowLeg_JNT": {self.nsChar + ":R_FK_lowLeg_JNT_4_DCM": self.nsChar + ":R_IK_footVec_CON"}, "L_FK_upArm_JNT": {self.nsChar + ":L_FK_upArm_JNT_4_DCM": self.nsChar + ":L_IK_handVec_CON"}, "R_FK_upArm_JNT": {self.nsChar + ":R_FK_upArm_JNT_4_DCM": self.nsChar + ":R_IK_handVec_CON"}}
            for dn in POV_set_List:
                if povNd[dn]:
                    polVec.update(povNd[dn])
                else:
                    pass
        for n in polVec.values():
            if len(cmds.ls(str(n))) != 0:
                cmds.select(str(n), add=True)
        cmds.bakeResults(cmds.ls(sl=True), simulation=True, t=(self.minCurrent, self.maxCurrent), sampleBy=1, dic=True, pok=True, sac=False, ral=False, bol=False, mr=True, controlPoints=False, shape=True)
        for w in range(len(polVec)):
            cmds.disconnectAttr(str(polVec.keys()[w]) + ".outputTranslate", str(polVec.values()[w]) + ".translate")
            cmds.delete(str(polVec.keys()[w]))
#####
    def timelineSet(self):
        self.minCurrent = cmds.playbackOptions(q=1, min=True) - 1
        self.maxCurrent = cmds.playbackOptions(q=1, max=True) + 1
#####
    def itfMain(self):
        self.getNamespace()
        self.timelineSet()
        selCtrl = cmds.ls(sl=True)
        self.IKPOV_list = []
        povList = {"L_legBlend_CON": "L_FK_lowLeg_JNT", "R_legBlend_CON": "R_FK_lowLeg_JNT", "L_armBlend_CON": "L_FK_upArm_JNT", "R_armBlend_CON": "R_FK_upArm_JNT"}
        blendJnt = {"L_legBlend_CON": ["L_Blend_leg_JNT", "L_Blend_lowLeg_JNT", "L_Blend_foot_JNT"],
                    "R_legBlend_CON": ["R_Blend_leg_JNT", "R_Blend_lowLeg_JNT", "R_Blend_foot_JNT"],
                    "L_armBlend_CON": ["L_Blend_upArm_JNT", "L_Blend_foreArm_JNT", "L_Blend_hand_JNT"],
                    "R_armBlend_CON": ["R_Blend_upArm_JNT", "R_Blend_foreArm_JNT", "R_Blend_hand_JNT"]}
        pvSet = {"L_legBlend_CON": ['L_FK_leg_JNT', 'L_FK_lowLeg_JNT', 'L_FK_foot_JNT', 'L_IK_footVec_CON'],
                 "R_legBlend_CON": ['R_FK_leg_JNT', 'R_FK_lowLeg_JNT', 'R_FK_foot_JNT', 'R_IK_footVec_CON'],
                 "L_armBlend_CON": ['L_FK_upArm_JNT', 'L_FK_foreArm_JNT', 'L_FK_hand_JNT', 'L_IK_handVec_CON'],
                 "R_armBlend_CON": ['R_FK_upArm_JNT', 'R_FK_foreArm_JNT', 'R_FK_hand_JNT', 'R_IK_handVec_CON']}
        itfBlendList = []
        ftiBlendList = []
        povSetList = []
        for i in selCtrl:
            if blendJnt[str(i).split(":")[1]]:
                if cmds.getAttr(str(i) + ".FKIKBlend") == 1:  # if IK
                    itfBlendList += blendJnt[str(i).split(":")[1]]
                elif cmds.getAttr(str(i) + ".FKIKBlend") == 0:  # if FK
                    ftiBlendList += blendJnt[str(i).split(":")[1]]
                    self.IKPOV_list.append(pvSet[str(i).split(":")[1]])
                    povSetList.append(povList[str(i).split(":")[1]])
                else:
                    cmds.warning("Invalid blend value")
            else:
                cmds.warning("Select controllers for blend")
        if len(itfBlendList) != 0 and len(ftiBlendList) == 0:
            self.ikTofk(selCtrl, itfBlendList)
        elif len(ftiBlendList) != 0 and len(itfBlendList) == 0:
            self.fkToik(selCtrl, povSetList)
        elif len(ftiBlendList) != 0 and len(itfBlendList) != 0:
            self.ikTofk(selCtrl, itfBlendList)
            self.fkToik(selCtrl, povSetList)
        else:
            pass

    def ikTofk(self, blendCtrl, blendList):
        with undoCheck():
            fkaJnt = {"R_Blend_upArm_JNT": "R_FK_upArm_CON", "R_Blend_foreArm_JNT": "R_FK_foreArm_CON", "R_Blend_hand_JNT": "R_FK_hand_CON",
                      "L_Blend_upArm_JNT": "L_FK_upArm_CON", "L_Blend_foreArm_JNT": "L_FK_foreArm_CON", "L_Blend_hand_JNT": "L_FK_hand_CON",
                      "R_Blend_leg_JNT": "R_FK_leg_CON", "R_Blend_lowLeg_JNT": "R_FK_lowLeg_CON", "R_Blend_foot_JNT": "R_FK_foot_CON",
                      "L_Blend_leg_JNT": "L_FK_leg_CON", "L_Blend_lowLeg_JNT": "L_FK_lowLeg_CON", "L_Blend_foot_JNT": "L_FK_foot_CON"}
            ikConSet = {"L_legBlend_CON":["L_IK_foot_CON","L_IK_footVec_CON"],
                        "R_legBlend_CON":["R_IK_foot_CON","R_IK_footVec_CON"],
                        "L_armBlend_CON":["L_IK_hand_CON","L_IK_handVec_CON"],
                        "R_armBlend_CON":["R_IK_hand_CON","R_IK_handVec_CON"]}
            rtAttr = ['rotateX', 'rotateY', 'rotateZ']
            # Joint - Locators Constraint
            cmds.currentTime(self.minCurrent + 1)
            locL = []       # Locators List
            temConstList = []       # Temp Constrain List
            for i in blendList:
                worldP = cmds.spaceLocator()[0]
                locL.append(worldP)
                temCons = cmds.parentConstraint(self.nsChar + ":" + i, worldP, w=1, mo=False)[0]
                temConstList.append(temCons)
            cmds.select(cl=True)
            # Bake Locators & Delete Constraints
            for k in locL:
                cmds.select(k,add=True)
            cmds.bakeResults(cmds.ls(sl=True), simulation=True, t=(self.minCurrent + 1, self.maxCurrent -1), sampleBy=1, dic=True, pok=True, sac=False, ral=False, bol=False, mr=True, controlPoints=False, shape=True)
            cmds.delete(temConstList)
            # FK Ctrls On
            ikConList = []
            for e in blendCtrl:
                cmds.setAttr(str(e) + ".FKIKBlend", 0)
                ikConList += ikConSet[str(e).split(":")[1]]
            cmds.currentTime(self.minCurrent + 1)
            # BlendJoint's Attr - FK Ctrls Data minTime Pose Sync
            for q in range(len(blendList)):
                for e in range(3):
                    rt = cmds.getAttr(self.nsChar + ":" + blendList[q] + "." + rtAttr[e])
                    cmds.setAttr(self.nsChar + ":" + fkaJnt[blendList[q]] + "." + rtAttr[e], rt)
            # Locators - FK Ctrls Constraint
            fkList = []
            for j in range(len(blendList)):
                temCon = cmds.orientConstraint(locL[j], self.nsChar + ":" + fkaJnt[blendList[j]], w=1, mo=False)[0]
                fkList.append(fkaJnt[blendList[j]])
            cmds.select(cl=True)
            # FK Ctrls Bake
            for t in fkList:
                cmds.select(self.nsChar + ":" + t,add=True)
            cmds.bakeResults(cmds.ls(sl=True), simulation=True, t=(self.minCurrent + 1, self.maxCurrent -1), sampleBy=1, dic=True, pok=True, sac=False, ral=False, bol=False, mr=True, controlPoints=False, shape=True)
            # Delete Orient Constraints & Locators
            cmds.delete(temCon)
            cmds.delete(locL)
            cmds.select(cl=True)
            for t in fkList:
                cmds.select(self.nsChar + ":" + t,add=True)
            cmds.selectKey(k=True)
            #cmds.filterCurve(f="euler")
            cmds.select(cl=True)
            for sn in ikConList:
                cmds.select(self.nsChar + ":" + sn, add=True)
            cmds.selectKey(k=True)
            cmds.cutKey(cl=True)
            cmds.select(cl=True)

    def fkToik(self, blendCtrl, povSetList):
        with undoCheck():
            fkList = {"L_armBlend_CON": {"L_FK_hand_CON": "L_IK_hand_CON"}, "R_armBlend_CON": {"R_FK_hand_CON": "R_IK_hand_CON"},
                      "L_legBlend_CON": {"L_FK_foot_CON": "L_IK_foot_CON"}, "R_legBlend_CON": {"R_FK_foot_CON": "R_IK_foot_CON"}}
            fkConSet = {"L_legBlend_CON": ["L_FK_leg_CON", "L_FK_lowLeg_CON", "L_FK_foot_CON", "L_FK_ball_CON"],
                        "R_legBlend_CON": ["R_FK_leg_CON", "R_FK_lowLeg_CON", "R_FK_foot_CON", "R_FK_ball_CON"],
                        "L_armBlend_CON": ["L_FK_upArm_CON", "L_FK_foreArm_CON", "L_FK_hand_CON"],
                        "R_armBlend_CON": ["R_FK_upArm_CON", "R_FK_foreArm_CON", "R_FK_hand_CON"]}
            cmds.currentTime(self.minCurrent + 1)
            # Joint - Locators Constraint
            locL = []
            temConsc = []
            locD = dict()
            for i in blendCtrl:
                worldP = cmds.spaceLocator()[0]
                locL.append(worldP)
                locD[str(i).split(":")[1]] = worldP
                temConsc.append(cmds.pointConstraint(self.nsChar + ":" + fkList[str(i).split(":")[1]].keys()[0], worldP, w=1, mo=False)[0])
                temConsc.append(cmds.orientConstraint(self.nsChar + ":" + fkList[str(i).split(":")[1]].keys()[0], worldP, w=1, mo=True)[0])
            cmds.select(cl=True)
            # Bake Locators & Delete Constraints
            for k in locL:
                cmds.select(k, add=True)
            cmds.bakeResults(cmds.ls(sl=True), simulation=True, t=(self.minCurrent + 1, self.maxCurrent - 1), sampleBy=1, dic=True, pok=True, sac=False, ral=False, bol=False, mr=True, controlPoints=False, shape=True)
            cmds.delete(temConsc)
            fkConList = []
            for e in blendCtrl:
                cmds.setAttr(str(e) + ".FKIKBlend", 1)
                fkConList += fkConSet[str(e).split(":")[1]]
            stemCon = []
            for j in blendCtrl:
                stemCon.append(cmds.parentConstraint(locD[str(j).split(":")[1]], self.nsChar + ":" + fkList[str(j).split(":")[1]].values()[0], w=1, mo=True)[0])
            cmds.select(cl=True)
            for c in blendCtrl:
                cmds.select(self.nsChar + ":" + fkList[str(c).split(":")[1]].values()[0], add=True)
            cmds.bakeResults(cmds.ls(sl=True), simulation=False, t=(self.minCurrent + 1, self.maxCurrent - 1), sampleBy=1, dic=True, pok=True, sac=False, ral=False, bol=False, mr=True, controlPoints=False, shape=True)
            cmds.delete(stemCon)
            cmds.delete(locL)
            # IK Ctrls On
            cmds.currentTime(self.minCurrent + 1)
            self.IKPOVBake(POV_set_List=povSetList)
            cmds.select(cl=True)
            for r in blendCtrl:
                cmds.select(self.nsChar + ":" + fkList[str(r).split(":")[1]].values()[0], add=True)
            cmds.selectKey(k=True)
            #cmds.filterCurve(f="euler")
            cmds.select(cl=True)
            for sn in fkConList:
                cmds.select(self.nsChar + ":" + sn, add=True)
            cmds.selectKey(k=True)
            cmds.cutKey(cl=True)
            cmds.select(cl=True)

    def chkState(self):
        tx = self.pvls.checkState()
        if str(tx) == "PySide2.QtCore.Qt.CheckState.Unchecked":
            self.pvls.setText("Pole Vector Local Space  :   0")
        else:
            self.pvls.setText("Pole Vector Local Space  :   1")

    def getMcp(self):
        exAnim = cmds.fileDialog2(startingDirectory="/stdrepo/ANI/", fileMode=0, fileFilter="animExport (*.anim)", caption="Export Anim")
        if exAnim:
            pass
        else:
            return
        with undoCheck():
            self.getNamespace()
            self.timelineSet()
            if cmds.ls(type='transform').count("%s:mocap_char" % self.nsChar + "_mcp") == 1:
                cmds.delete("%s:mocap_char" % self.nsChar + "_mcp")
            cmds.file('/stdrepo/ANI/Asset/HumanIK_for_AnimBrowser/mocapJoint.ma', i=True, type='mayaAscii', ra=True, mergeNamespacesOnClash=True, namespace=self.nsChar + "_mcp", options='v=0')
            mocapJoint_constraint_list = []
            for i in range(len(self.mocapJoint_list)):
                temp_point_con = cmds.pointConstraint("%s:%s" % (self.nsChar, self.moduleJoint_list[i]), "%s:%s" % (self.nsChar + "_mcp", self.mocapJoint_list[i]), mo=0, w=1)
                mocapJoint_constraint_list.append(temp_point_con[0])
            cmds.delete(mocapJoint_constraint_list)
            for i in range(len(self.mocapJoint_list)):
                cmds.parentConstraint("%s:%s" % (self.nsChar, self.moduleJoint_list[i]), "%s:%s" % (self.nsChar + "_mcp", self.mocapJoint_list[i]), mo=1, w=1)

            cmds.select("%s:%s" % (self.nsChar + "_mcp", "Crw_Hips"), hi=True)
            cmds.bakeResults(cmds.ls(sl=True), hi=True, simulation=True, t=(int(self.minCurrent), int(self.maxCurrent)), sampleBy=1, dic=True, pok=True, sac=False, ral=False, bol=False, mr=True, controlPoints=False, shape=True)
            cmds.selectKey(k=True)
            #cmds.filterCurve(f="euler")
            cmds.select(cl=True)
            cmds.select("%s:%s" % (self.nsChar + "_mcp", "Crw_Hips"), hi=True)
            cmds.file(exAnim, force=True, options="precision=8;nodeNames=1;verboseUnits=0;whichRange=1;options=keys;hierarchy=below;controlPoints=0;shapes=1;useChannelBox=0;copyKeyCmd=-animation objects -option keys -hierarchy below -controlPoints 0 -shape 1", typ="animExport", eas=True)

    def getHIK(self):
        exAnim = cmds.fileDialog2(startingDirectory="/stdrepo/ANI/", fileMode=0, fileFilter="animExport (*.anim)", caption="Export Anim")
        if exAnim:
            pass
        else:
            return
        restF = int(self.inputTfr.text())
        startF = int(self.inActStt.text())
        nsChar = self.getNamespace()
        self.saveHikAnim(nsChar, exAnim[0], restF, startF)

    def saveHikAnim(self, nsChar, exAnim, restFrame, startFrame):
        with undoCheck():
            self.checkFr()
            self.timelineSet()
            self.saveKeyFrames(exAnim, restFrame, startFrame)
            cmds.file('/stdrepo/ANI/Asset/HumanIK_for_AnimBrowser/HIKJoint.ma', i=True, type='mayaAscii', ra=True, mergeNamespacesOnClash=True, namespace=nsChar, options='v=0')
            HIKJoint_constraint_list = []
            for i in range(len(self.HIKJoint_list)):
                temp_point_con = cmds.pointConstraint("%s:%s" % (nsChar, self.moduleJoint_list[i]), "%s:%s" % (nsChar, self.HIKJoint_list[i]), mo=0, w=1)
                HIKJoint_constraint_list.append(temp_point_con[0])
            cmds.delete(HIKJoint_constraint_list)
            self.targetNS.append(nsChar)
            HIKJoint_constraint_list = []
            for i in range(len(self.HIKJoint_list)):
                temp_point_con = cmds.parentConstraint("%s:%s" % (nsChar, self.moduleJoint_list[i]), "%s:%s" % (nsChar, self.HIKJoint_list[i]), mo=1, w=1)
                HIKJoint_constraint_list.append(temp_point_con[0])
            cmds.select("%s:%s" % (nsChar, "Hips"), hi=True)
            cmds.bakeResults(cmds.ls(sl=True), simulation=True, hi=True, t=(int(self.minCurrent), int(self.maxCurrent)), sampleBy=1, dic=True, pok=True, sac=False, ral=False, bol=False, mr=True, controlPoints=False, shape=True)
            cmds.file(exAnim, force=True, options="precision=8;nodeNames=1;verboseUnits=0;whichRange=1;options=keys;hierarchy=below;controlPoints=0;shapes=1;useChannelBox=0;copyKeyCmd=-animation objects -option keys -hierarchy below -controlPoints 0 -shape 1", type="animExport", eas=True)
            if self.delConstraints == None:
                pass
            else:
                try:
                    cmds.delete('*:HIKJoint_LOC')
                    cmds.delete('*:set_HIK')
                except:
                    pass
            self.unmuteCh(restFrame)

    def checkFr(self):
        check_list = ['L_IK_footVec_CON', 'R_IK_footVec_CON', 'L_IK_foot_CON', 'R_IK_foot_CON', 'L_IK_hand_CON', 'R_IK_hand_CON', 'root_CON', 'C_IK_head_CON', 'direction_CON', 'move_CON']
        for i in check_list:
            contN = self.nsChar + ":" + i
            for j in self.trAttr:
                if cmds.keyframe(contN, at=j, q=True) == None:
                    cmds.setKeyframe(contN, at=j)
                else:
                    pass

    def bakeIKControl(self):
        self.timelineSet()
        nsChar = self.targetNS
        cmds.select(cl=True)
        for i in range(len(self.IKController_list)):
            cmds.select('%s:%s' % (nsChar[0], self.IKController_list[i]), add=True)
        cmds.select("%s:move_CON" % nsChar[0], add=True)
        cmds.currentTime(self.minCurrent)
        cmds.bakeResults(simulation=True, t=(int(self.minCurrent), int(self.maxCurrent)), sampleBy=1, dic=True, pok=True, sac=False, ral=False, bol=False, mr=True, controlPoints=False, shape=True)
        cmds.select(cl=True)

    def assignHIK(self):
        '''
        Human IK Anim Load. Anim File path input.
        :return: None
        '''
        animDir = cmds.fileDialog2(fileMode=1, fileFilter="animExport (*.anim)", caption="Import Anim File")
        if animDir:
            pass
        else:
            return
        nsChar = self.getNamespace()
        self.loadHikAnim(nsChar, animDir[0])

    def blendCheck(self):
        '''
        IK/FK Blend Check. FK -> IK
        :return: None
        '''
        self.getNamespace()
        for i in cmds.ls(self.nsChar + ":*Blend_CON"):
            if cmds.getAttr(str(i) + ".FKIKBlend") != 1:
                cmds.setAttr(str(i) + ".FKIKBlend", 1)

    def setDft(self):
        '''
        모든 컨트롤러 값 0으로 초기화. T-Pose 함수 뒤에 사용되지 않도록 주의.
        :return: None
        '''
        cmds.currentTime(cmds.playbackOptions(q=1, min=True))
        self.restMaker()
        cmds.cutKey(self.nsChar + ":*_CON", cl=True)
        allCon = cmds.ls(self.nsChar + ":*_CON")
        for i in allCon:
            if cmds.listAttr(str(i), k=True).count("rotateX") == 1:
                cmds.setAttr(str(i) + ".rotateX", 0)
                cmds.setAttr(str(i) + ".rotateY", 0)
                cmds.setAttr(str(i) + ".rotateZ", 0)
            else:
                pass
        trDft = ["root_CON", "L_IK_foot_CON", "R_IK_foot_CON", "R_IK_hand_CON", "L_IK_hand_CON", "L_IK_footVec_CON", "R_IK_footVec_CON", "L_IK_handVec_CON", "R_IK_handVec_CON"]
        delLst = ["libr:HIKJoint_LOC", "*:set_HIK"]
        for j in trDft:
            cmds.setAttr(self.nsChar + ":" + j + ".translateX", 0)
            cmds.setAttr(self.nsChar + ":" + j + ".translateY", 0)
            cmds.setAttr(self.nsChar + ":" + j + ".translateZ", 0)
        for delf in delLst:
            if len(cmds.ls(delf)) != 0:
                cmds.delete(delf)
            else:
                pass

    def loadHikAnim(self, nsChar, animDir):
        '''
        - Human IK Anim File Load
        - Keyframes data transfer
        :param animDir: Anim File Path.
        :return: None
        '''
        with undoCheck():
            self.targetNS = []
            self.delConstraints = []
            # Name Space 설정
            self.getNamespace()
            self.blendCheck()
            mel.eval("cycleCheck -e off")
            # HIKJoint 뼈 로드해서 ANIM 넣기. (NameSpace: libr)
            cmds.select(cl=True)
            cmds.file('/stdrepo/ANI/Asset/HumanIK_for_AnimBrowser/HIKJoint.ma', i=True, type='mayaAscii', ra=True, mergeNamespacesOnClash=True, namespace='libr', options='v=0')
            cmds.select("libr:Hips")
            cmds.file(animDir, i=True, iv=True)
            mxT = cmds.keyframe("libr:Hips", q=True, lastSelected=True, timeChange=True)
            mnT = min(cmds.keyframe("libr:Hips", q=True))
            cmds.playbackOptions(minTime=mnT, maxTime=mxT[0])
            cmds.select(cl=True)

            # HIKJoint 뼈 로드해서 선택한 캐릭터에 Constrain 걸기
            if cmds.ls(type='transform').count("%s:HIKJoint_LOC" % nsChar) == 1:
                cmds.delete("%s:HIKJoint_LOC" % nsChar)
            cmds.file('/stdrepo/ANI/Asset/HumanIK_for_AnimBrowser/HIKJoint.ma', i=True, type='mayaAscii', ra=True, mergeNamespacesOnClash=True, namespace=nsChar, options='v=0')
            HIKJoint_constraint_list = []
            for i in range(len(self.HIKJoint_list)):
                temp_point_con = cmds.pointConstraint("%s:%s" % (nsChar, self.moduleJoint_list[i]),
                                                      "%s:%s" % (nsChar, self.HIKJoint_list[i]), mo=0, w=1)
                HIKJoint_constraint_list.append(temp_point_con[0])
            cmds.delete(HIKJoint_constraint_list)
            self.targetNS.append(nsChar)

            for j in range(len(self.HIK_attatch_toIK_list)):
                self.toIK_Const = cmds.parentConstraint("%s:%s" % (nsChar, self.HIK_attatch_toIK_list[j]),
                                                        "%s:%s" % (nsChar, self.IKController_list[j]), mo=1, w=1)
                self.delConstraints.append(self.toIK_Const[0])

            mel.eval('$gHIKCurrentCharacter = "%s:%s"' % (nsChar, "set_HIK"))
            mel.eval('hikToggleLockDefinition();')
            mel.eval('refreshAllCharacterLists();')
            mel.eval('mayaHIKsetCharacterInput("%s:%s","%s:%s");' % (nsChar, "set_HIK", 'libr', "set_HIK"))
            self.IKPOVBake()
            self.timelineSet()
            self.bakeIKControl()
            jPath = str(animDir).replace(".anim", ".json")
            cmds.delete(nsChar + ":*_CON", sc=True)
            if os.path.exists(jPath):
                self.neverDieMyGraph(jPath)
            else:
                pass
            if self.delConstraints == None:
                pass
            else:
                cmds.delete(self.toIK_Const)
                cmds.delete(nsChar + ":HIKJoint_LOC")
                delLst = ["libr:HIKJoint_LOC", "*:set_HIK"]
                for delf in delLst:
                    if len(cmds.ls(delf)) != 0:
                        cmds.delete(delf)
                    else:
                        pass
            self.restMaker()
            cmds.currentTime(self.minCurrent)
            self.makeTpose()
            cmds.select(cmds.ls(nsChar + ":*_CON"))
            cmds.selectKey()
            #cmds.filterCurve(f="euler")
            cmds.select(cl=True)

    def restMaker(self):
        frm = int(cmds.playbackOptions(q=1, min=True) - 1)
        conList = cmds.ls(self.nsChar + ":*_CON")
        conList.remove(self.nsChar + ":place_CON")
        conList.remove(self.nsChar + ":direction_CON")
        conList.remove(self.nsChar + ":move_CON")
        for i in conList:
            for f in self.trAttr:
                if cmds.listAttr(i, k=True).count(f) == 1:
                    cmds.setKeyframe(i, at=f, t=frm, v=0)
                else:
                    pass

    def importMocapAnim(self):
        nsChar = self.getNamespace()
        scrP = "/stdrepo/ANI/Library/Mocap_Library/02_Data/02_CharMocap"
        animDir = str(cmds.fileDialog2(fileMode=1, fileFilter="animExport (*.anim)", dir=scrP, caption="Import Anim File")[0])
        self.impMc(nsChar, animDir)

    def impMc(self, nsChar, animDir):
        '''
        - Mocap Anim File Import
        :return: None
        '''


        with undoCheck():
            self.targetNS = []
            self.delConstraints = []
            self.blendCheck()
            if animDir:
                pass
            else:
                return



            origPos = self.getWorldPostion() #월드컨트롤러의 원래 위치를 저장해놓고 베이크작업을 위해 원점으로 월드컨트롤러를 이동시킴

            self.makeTpose()
            
            #모션캡쳐 에님을 적용시킬수 있는 템플릿 조인트구조를 임포트함
            cmds.file('/stdrepo/ANI/Asset/HumanIK_for_AnimBrowser/mocapJoint.ma', i=True,
                      type='mayaAscii', ra=True, mergeNamespacesOnClash=True, namespace="mcp", options='v=0') 
            cmds.select("mcp:Crw_Hips")
            
            #animDir(anim파일)을 루트조인트(mcp:Crw_Hips)에 적용시킴
            cmds.file(animDir, i=True, applyTo="mcp:Crw_Hips") 


###################  전체 키값들을 원하는 프레임으로 옵셋  ##################################################################################


            #불러온 모션캡쳐 키데이타에서 가장 앞프레임에 있는 키의 위치 찾기
            cmds.select("mcp:Crw_Hips",r=1, hi=1)
            mcpList = cmds.ls(sl=1)
            minKeys = []
            cmds.selectKey()

            for num in range(len(mcpList)):
                keyTime = cmds.keyframe(mcpList[num], sl=True,query=True,tc=True) #선택된 모든 오브젝트의 키위치정보를 읽어옴

                if keyTime != None:
                    minKeys.append(min(keyTime))

            minTime = min(minKeys) #가장 앞에있는 키의 위치값
            startF = int(self.inputSF.text())
            offSetValue = abs(startF - minTime)
            cmds.keyframe(animation="keys", option="over", relative=1, timeChange=offSetValue)


#####################################################################################################################################




            # 불러온 anim데이타가 들어간 모션캡쳐 템플릿 조인트의 양쪽 다리 볼조인트에 키가 들어가 있으면 키가 들어가있는 볼조인트의 각 프레임의 키값을 모두 저장해둔다.
            toeDict = dict()
            toes = {"mcp:Crw_LeftToeBase": nsChar + ":L_IK_ball_CON", "mcp:Crw_RightToeBase": nsChar + ":R_IK_ball_CON"}
            atts = ["rotateX", "rotateY", "rotateZ"]

            try:
                for jnt in toes.keys():
                    toeDict[jnt] = dict()
                    for at in atts:
                        if cmds.keyframe(jnt, q=True, at=at):
                            toeDict[jnt][at] = dict()
                            for frs in cmds.keyframe(jnt, q=True):
                                toeDict[jnt][at][int(frs)] = cmds.keyframe(jnt, q=True, t=(frs, frs), vc=True)[0]
            except:
                pass
                            


     # 타임라인의 처음과 끝을 키가 시작되는 프레임과 끝나는 프레임으로 설정한다.
            mxT = cmds.keyframe("mcp:Crw_Hips", q=True, lastSelected=True, timeChange=True)
            mnT = min(cmds.keyframe("mcp:Crw_Hips", q=True))
            cmds.playbackOptions(minTime=mnT, maxTime=mxT[0])
            cmds.currentTime(mnT)





            # 휴먼 IK 파일을 불러옴
            cmds.file('/stdrepo/ANI/Asset/HumanIK_for_AnimBrowser/huType_hikJoint.ma', i=True,
                      type='mayaAscii', ra=True, mergeNamespacesOnClash=True, namespace='mcp_hik', options='v=0')


            # 모캡조인트의 위치에 휴먼IK 조인트의 위치를 맞추기 위해 컨스트레인을 사용후 다시 컨스트레인 삭제
            mocapJoint_constraint_list = []
            for i in range(len(self.mocapJoint_list)):
                temp_point_con = cmds.pointConstraint("%s:%s" % ("mcp", self.mocapJoint_list[i]),
                                                      "%s:%s" % ("mcp_hik", self.mocapJoint_list[i]), mo=0, w=1)
                mocapJoint_constraint_list.append(temp_point_con[0])  # constrain 일괄 삭제를 위한 리스트 축적
            cmds.delete(mocapJoint_constraint_list)  # Constrain 삭제


            # 모캡조인트에 휴먼IK조인트를 parent constraint 시킴
            for j in range(len(self.mocapJoint_list)):
                cmds.parentConstraint("%s:%s" % ("mcp", self.mocapJoint_list[j]),
                                      "%s:%s" % ("mcp_hik", self.mocapJoint_list[j]), mo=1, w=1)


            # 이미 불러져 있는 HIKJoint 가 있는지 확인해보고 존재하면 지우기
            if cmds.ls(type='transform').count("%s:HIKJoint_LOC" % nsChar) == 1:
                cmds.delete("%s:HIKJoint_LOC" % nsChar)

            # HIKJoint 뼈 로드해서 선택한 캐릭터의 스킨조인트 위치에 맞추기 위해 컨스트레인 사용후 다시 컨스트레인 삭제
            cmds.file('/stdrepo/ANI/Asset/HumanIK_for_AnimBrowser/HIKJoint.ma', i=True,
                      type='mayaAscii', ra=True, mergeNamespacesOnClash=True, namespace=nsChar, options='v=0')

            HIKJoint_constraint_list = []
            for u in range(len(self.HIKJoint_list)):
                temp_point_con = cmds.pointConstraint("%s:%s" % (nsChar, self.moduleJoint_list[u]),
                                                      "%s:%s" % (nsChar, self.HIKJoint_list[u]), mo=0, w=1)
                HIKJoint_constraint_list.append(temp_point_con[0])  # constrain 일괄 삭제를 위한 리스트 축적
            cmds.delete(HIKJoint_constraint_list)  # Constrain 삭제



            self.targetNS.append(nsChar)  # NameSpace 입력
            for r in range(len(self.HIK_attatch_toIK_list)):
                self.toIK_Const = cmds.parentConstraint("%s:%s" % (nsChar, self.HIK_attatch_toIK_list[r]),
                                                        "%s:%s" % (nsChar, self.IKController_list[r]), mo=1, w=1)
                self.delConstraints.append(self.toIK_Const[0])




            #####################여기서부터 다시################################################################################
            mel.eval('$gHIKCurrentCharacter = "%s:%s"' % (nsChar, "set_HIK"))
            mel.eval('hikToggleLockDefinition();')
            mel.eval('refreshAllCharacterLists();')
            mel.eval('mayaHIKsetCharacterInput("%s:%s","%s:%s");' % (nsChar, "set_HIK", 'mcp_hik', "Character1"))
            self.IKPOVBake()
            self.timelineSet()
            self.bakeIKControl()
            # Load Toes Keyframes
            try:
                for jntName in toeDict.keys():
                    for jntAtt in toeDict[jntName].keys():
                        for jntFr in toeDict[jntName][jntAtt].keys():
                            cmds.setKeyframe(toes[jntName], at=jntAtt, t=(jntFr, jntFr), v=toeDict[jntName][jntAtt][jntFr])
            except:
                pass
            ##
            for i in range(len(self.IKController_list)):
                cmds.select('%s:%s' % (nsChar, self.IKController_list[i]), add=True)
            cmds.selectKey(k=True)
            #cmds.filterCurve(f="euler")

            if self.delConstraints == None:
                pass
            else:
                cmds.delete(self.toIK_Const)
                cmds.delete(nsChar + ":HIKJoint_LOC")
                delLst = ["mcp:mocap_char", "mcp_hik:mocap_char", "*:set_HIK"]
                for delf in delLst:
                    if len(cmds.ls(delf)) != 0:
                        cmds.delete(delf)
                    else:
                        pass
            cmds.delete("mcp_hik:*")
            cmds.delete(nsChar + ":*_CON", sc=True, uac=False, hi="none", cp=False, s=True)
            cmds.currentTime(cmds.playbackOptions(q=True, min=True))

            # 이동 컨트롤러 잡고 Y축 아래로 -0.72463만큼 내려서 바닥에 발 맞추는 부분
            # 리타겟팅으로 인해 발이 지면에서 뜨게 되는데 이는 HIK Joint Set 의 root joint의 높이차 때문
            if len(cmds.ls(nsChar + ":root_CON")) != 0:
                movecon = ["root_CON", "L_IK_hand_CON", "R_IK_hand_CON", "L_IK_foot_CON", "R_IK_foot_CON",
                           "L_IK_footVec_CON", "R_IK_footVec_CON", "L_IK_handVec_CON", "R_IK_handVec_CON"]
            elif len(cmds.ls(nsChar + ":C_IK_root_CON")) != 0:
                movecon = ["C_IK_root_CON", "L_IK_foreLeg_CON", "R_IK_foreLeg_CON", "L_IK_knee_CON", "R_IK_knee_CON",
                           "L_IK_hindLeg_CON", "R_IK_hindLeg_CON", "L_IK_edbow_CON", "R_IK_edbow_CON"]
            else:
                pass
            for i in movecon:
                cmds.keyframe(nsChar + ":" + i, at="translateY", e=True, iub=True, r=True, o="over", vc=-0.72463)

            ##########################여기까지########################################################################################



            # 원래의 월드컨트롤러 위치값을 가져와서 원점에 있는 월드컨트롤러를 원래의 위치로 옮김
            atts = ("tx","ty","tz","rx","ry","rz")
            for num in range(len(atts)):
                cmds.setAttr(nsChar + ":place_CON." + atts[num], origPos[num])
            


    def offSetFrames(self):
        '''
        - Imported Anim data offset
        :return: None 
        '''
        self.getNamespace()
        inputTime = str(self.offSet.text())
        if inputTime.count("-") == 1:
            if inputTime[1:].isdigit():
                offsetFrames = int(self.offSet.text())
            else:
                cmds.error("Input Error for Offset.")
                return
        elif inputTime.count("+") == 1:
            if inputTime[1:].isdigit():
                offsetFrames = int(self.offSet.text())
            else:
                cmds.error("Input Error for Offset.")
                return
        else:
            if inputTime.isdigit():
                offsetFrames = int(self.offSet.text())
            else:
                cmds.error("Input Error for Offset.")
                return
        cmds.selectKey(cmds.ls(self.nsChar + ":*_CON"))
        cmds.keyframe(e=True, iub=True, r=True, o="over", tc=offsetFrames)

    def hTom(self):
        '''
        - Human IK Anim data assign to mocap character. Anim File path input.  
        :return: None 
        '''
        animDir = cmds.fileDialog2(fileMode=1, fileFilter="animExport (*.anim)", caption="Import Anim File")
        if animDir:
            pass
        else:
            return
        self.hTomEx(animDir[0])

    def hTomEx(self, inputPath):
        '''
        - Human IK Anim data assign to mocap Character.
        :param inputPath: 
        :return: None
        '''
        with undoCheck():
            self.timelineSet()
            if cmds.ls("Crw_Hips"):
                if cmds.keyframe("Crw_Hips", q=True) != None:
                    cmds.select("Crw_Hips", hi=True)
                    cmds.selectKey(keyframe=True)
                    cmds.currentTime(cmds.playbackOptions(q=1, min=True))
                    cmds.delete(all=True, c=True)
                else:
                    pass
            else:
                pass

            self.delConstraints = []
            # Name Space 설정
            mel.eval("cycleCheck -e off")
            # HIKJoint 뼈 로드해서 ANIM 넣기. (NameSpace: libr)
            cmds.select(cl=True)
            animDir = inputPath
            cmds.file('/stdrepo/ANI/Asset/HumanIK_for_AnimBrowser/HIKJoint.ma', i=True, type='mayaAscii', ra=True, mergeNamespacesOnClash=True, namespace='libr', options='v=0')
            cmds.select("libr:Hips")
            cmds.file(animDir, i=True, iv=True)
            mAtt = []
            atb = ["tx", "ty", "tz", "rx", "ry", "rz"]
            for s in atb:
                mAtt.append(cmds.getAttr("libr:Hips." + s))
            for w in range(len(atb)):
                cmds.setAttr("Crw_Hips." + atb[w], mAtt[w])
            mxT = cmds.keyframe("libr:Hips", q=True, lastSelected=True, timeChange=True)
            mnT = min(cmds.keyframe("libr:Hips", q=True))
            cmds.playbackOptions(minTime=mnT, maxTime=mxT[0])
            cmds.select(cl=True)
            if cmds.ls(type='transform').count("mcap:HIKJoint_LOC") == 1:
                cmds.delete("mcap:HIKJoint_LOC")
            cmds.file('/stdrepo/ANI/Asset/HumanIK_for_AnimBrowser/HIKJoint.ma', i=True, type='mayaAscii', ra=True, mergeNamespacesOnClash=True, namespace="mcap", options='v=0')
            HIKJoint_constraint_list = []
            for i in range(len(self.mocapJoint_list)):
                temp_point_con = cmds.pointConstraint(self.mocapJoint_list[i], "mcap:%s" % self.HIKJoint_list[i], mo=0, w=1)
                HIKJoint_constraint_list.append(temp_point_con[0])
            cmds.delete(HIKJoint_constraint_list)
            for j in range(len(self.mocapJoint_list)):
                self.toMo_Const = cmds.parentConstraint("mcap:%s" % self.HIKJoint_list[j], self.mocapJoint_list[j], mo=1, w=1)
                self.delConstraints.append(self.toMo_Const[0])
            mel.eval('$gHIKCurrentCharacter = "mcap:set_HIK"')
            mel.eval('hikToggleLockDefinition();')
            mel.eval('refreshAllCharacterLists();')
            mel.eval('mayaHIKsetCharacterInput("mcap:set_HIK","libr:set_HIK");')
            cmds.select("Crw_Hips", hi=True)
            cmds.bakeResults(cmds.ls(sl=True), hi=True, simulation=True, t=(int(self.minCurrent), int(self.maxCurrent)), sampleBy=1, dic=True, pok=True, sac=False, ral=False, bol=False, mr=True, controlPoints=False, shape=True)
            if self.delConstraints == None:
                pass
            else:
                cmds.delete(self.delConstraints)
                delLst = ["mcap:HIKJoint_LOC", "libr:HIKJoint_LOC", "mcap:*", "libr:*", "*:set_HIK"]
                for delf in delLst:
                    if len(cmds.ls(delf)) != 0:
                        cmds.delete(delf)
                    else:
                        pass
            cmds.select(cl=True)
            cmds.select("Crw_Hips", hi=True)
            cmds.selectKey(keyframe=True)
            ofrm = cmds.keyframe(q=True)[0] - 1
            cmds.keyframe(e=True, iub=True, r=True, o="over", tc=-ofrm)
            tlm = cmds.keyframe("Crw_Hips", q=True, lastSelected=True, timeChange=True)
            cmds.playbackOptions(minTime=1, maxTime=tlm[0])

    def neverDieMyGraph(self, jfilePath):
        '''
        - Restore keyframe data from JSON file.
        :param jfilePath: JSON file path 
        :return: None
        '''
        loadJ = open(jfilePath).read()
        jData = json.loads(loadJ)
        mel.eval("channelBoxCommand -break; ")
        for i in range(len(jData['nk'].keys())):
            nConName = self.nsChar + ":" + str(jData['nk'].keys()[i])
            conNa = nConName.split(".")[0]
            conNe = nConName.split(".")[1]
            if cmds.ls(type="transform").count(conNa) == 1:
                if cmds.attributeQuery(conNe, n=conNa, ex=True) == 1:
                    mel.eval("CBdeleteConnection %s;" % nConName)
                    cmds.setAttr(nConName, jData['nk'].values()[i])
                else:
                    pass
            else:
                pass
        contJson = jData['pl']
        fkBlJson = jData['blCon']
        fkJson = jData['bcf']
        del jData['pl']
        del jData['nk']
        del jData['refFile']
        del jData['blCon']
        del jData['bcf']
        for j in jData.keys():
            pConNam = self.nsChar + ":" + str(j)
            if cmds.ls(type="transform").count(pConNam) == 1:
                for k in jData[j].keys():
                    if cmds.attributeQuery(str(k), n=pConNam, ex=True) == 1:
                        if cmds.keyframe(pConNam, at=str(k), q=True) != None:
                            cmds.selectKey(pConNam + "." + str(k), k=True)
                            for r in jData[j][k].keys():
                                cmds.selectKey(pConNam + "." + str(k), rm=True, k=True, t=(r, r))
                            try:
                                cmds.cutKey(an="keys", cl=True)
                            except:
                                pass
                        else:
                            pass
                    else:
                        pass
            else:
                pass
        for a in contJson.keys():
            contNames = self.nsChar + ":" + str(a)
            if cmds.ls(type="transform").count(contNames) == 1:
                for d in self.trAttr:
                    if cmds.attributeQuery(d, n=contNames, ex=True) == 1:
                        if type(contJson[a][d]) == dict:
                            for s in range(len(contJson[a][d])):
                                frms = contJson[a][d].keys()[s]
                                frmvs = contJson[a][d].values()[s]
                                cmds.setKeyframe(contNames, at=d, t=(frms, frms), v=frmvs)
                        elif type(contJson[a][d]) == float:
                            klsVal = contJson[a][d]
                            cmds.setAttr(contNames + "." + d, klsVal)
                    else:
                        cmds.warning(contNames + " Controller don't have " + d + " Attribute.")
            else:
                cmds.warning("There is no " + contNames + " Controller.")
        for q in fkJson.keys():
            if fkJson[q] == 1:
                cmds.setAttr(self.nsChar + ":" + str(q) + ".FKIKBlend", 0)
                for p in fkBlJson[str(q)].keys():
                    fkCon = self.nsChar + ":" + str(p)
                    if cmds.ls(type="transform").count(fkCon) == 1:
                        for u in self.trAttr:
                            if cmds.attributeQuery(u, n=fkCon, ex=True) == 1:
                                if type(fkBlJson[q][p][u]) == dict:
                                    for n in range(len(fkBlJson[q][p][u])):
                                        fks = fkBlJson[q][p][u].keys()[n]
                                        fkvs = fkBlJson[q][p][u].values()[n]
                                        cmds.setKeyframe(fkCon, at=u, t=(fks, fks), v=fkvs)
                                elif type(fkBlJson[q][p][u]) == float:
                                    tfkv = fkBlJson[q][p][u]
                                    cmds.setAttr(fkCon + "." + u, tfkv)
                            else:
                                cmds.warning(fkCon + " Controller don't have " + u + " Attribute.")
                    else:
                        cmds.warning("There is no " + fkCon + " Controller.")
            else:
                pass
        for w in jData.keys():
            pConNm = self.nsChar + ":" + str(w)
            h = str(w)
            if cmds.ls(type="transform").count(pConNm) == 1:
                for e in jData[h].keys():
                    if cmds.attributeQuery(str(e), n=pConNm, ex=True) == 1:
                        if cmds.keyframe(pConNam, at=str(e), q=True) != None:
                            for c in range(len(jData[h][e].keys())):
                                fr = jData[h][e].keys()[c]
                                val = jData[h][e].values()
                                cmds.keyTangent(pConNm, at=str(e), t=(fr, fr), l=val[c][4], ia=val[c][0], oa=val[c][1], iw=val[c][2], ow=val[c][3])
                        else:
                            pass
                    else:
                        pass
            else:
                pass




##########################################################



    def makeTpose(self):

        with undoCheck():
            self.getNamespace()
            self.blendCheck()
            polv_List = ["L_IK_handVec_CON", "R_IK_handVec_CON", "L_IK_footVec_CON", "R_IK_footVec_CON"]
            L_uA = self.nsChar + ":" + "L_Skin_upArm_JNT"
            L_fA = self.nsChar + ":" + "L_Skin_foreArm_JNT"
            L_hNd = self.nsChar + ":" + "L_Skin_hand_JNT"
            L_hndCon = self.nsChar + ":" + "L_IK_hand_CON"
            R_uA = self.nsChar + ":" + "R_Skin_upArm_JNT"
            R_fA = self.nsChar + ":" + "R_Skin_foreArm_JNT"
            R_hNd = self.nsChar + ":" + "R_Skin_hand_JNT"
            R_hndCon = self.nsChar + ":" + "R_IK_hand_CON"
            # 어깨 컨트롤러 회전 각도 쿼리
            shrs = self.asr_check.checkState()
            if str(shrs) == "PySide2.QtCore.Qt.CheckState.Unchecked":
                pass
            else:
                up = cmds.xform(L_uA, q=True, ws=True, t=True)
                hand = cmds.xform(L_hNd, q=True, ws=True, t=True)
                length = ((hand[0] - up[0]) ** 2 + (hand[1] - up[1]) ** 2 + (hand[2] - up[2]) ** 2) ** 0.5
                rto = ((hand[0] - up[0]) ** 2 + (hand[2] - up[2]) ** 2) ** 0.5
                res = abs(rto / length)
                if res < 0.65:
                    an = 10 + 2 * (1 - res)
                elif res >= 0.65 and res < 0.85:
                    an = 10 + 2 * (1 - res)
                elif res >= 0.85 and res < 0.94:
                    an = 4 + 2 * (1 - res)
                else:
                    an = 2 + 2 * (1 - res)
                cmds.setAttr(self.nsChar + ":L_FK_shoulder_CON.rotateY", an)
                cmds.setAttr(self.nsChar + ":R_FK_shoulder_CON.rotateY", an)
            #손목 Y축 회전을 위한 각도 값 쿼리
            R_ax = cmds.xform(R_uA, q=True, t=True, ws=True)[0] - cmds.xform(R_hNd, q=True, t=True, ws=True)[0]
            R_az = cmds.xform(R_hNd, q=True, t=True, ws=True)[2] - cmds.xform(R_uA, q=True, t=True, ws=True)[2]
            R_en = (R_ax ** 2 + R_az ** 2) ** 0.5
            R_radn = math.acos(R_ax / R_en)
            R_rotY = cmds.xform(R_hndCon, q=True, ro=True, ws=True)[1] - (R_radn * (180 / math.pi))
            L_ax = cmds.xform(L_hNd, q=True, t=True, ws=True)[0] - cmds.xform(L_uA, q=True, t=True, ws=True)[0]
            L_az = cmds.xform(L_uA, q=True, t=True, ws=True)[2] - cmds.xform(L_hNd, q=True, t=True, ws=True)[2]
            L_en = (L_ax ** 2 + L_az ** 2) ** 0.5
            L_radn = math.acos(L_ax / L_en)
            L_rotY = cmds.xform(L_hndCon, q=True, ro=True, ws=True)[1] + (L_radn * (180 / math.pi))
            cmds.setAttr(R_hndCon + ".ry", R_rotY)
            cmds.setAttr(L_hndCon + ".ry", L_rotY)
            # tz 값을 어깨와 동일하게 맞추기
            hndCon_List = {L_uA:L_hndCon, R_uA:R_hndCon}
            for i in hndCon_List:
                ws_Tz = cmds.xform(i, q=True, t=True, ws=True)[2]
                con_Tz_ws = cmds.xform(hndCon_List[i], q=True, t=True, ws=True)[2]
                con_Tz = cmds.getAttr(hndCon_List[i] + ".tz")
                L_conSub_Z = con_Tz - con_Tz_ws
                if L_conSub_Z < 0:
                    cmds.setAttr(hndCon_List[i] + ".tz", ws_Tz - abs(L_conSub_Z))
                elif L_conSub_Z > 0:
                    cmds.setAttr(hndCon_List[i] + ".tz", ws_Tz + abs(L_conSub_Z))
                else:
                    cmds.setAttr(hndCon_List[i] + ".tz", ws_Tz)
            # 손목 회전을 위한 각도 값 쿼리
            R_angx = cmds.xform(R_uA, q=True, t=True, ws=True)[0] - cmds.xform(R_hNd, q=True, t=True, ws=True)[0]
            R_angy = cmds.xform(R_uA, q=True, t=True, ws=True)[1] - cmds.xform(R_hNd, q=True, t=True, ws=True)[1]
            R_en = (R_angx ** 2 + R_angy ** 2) ** 0.5
            R_rad = math.acos(R_angx / R_en)
            R_rigAngle = R_rad * (180 / math.pi)
            R_plang = cmds.xform(R_hndCon, q=True, ro=True, ws=True)[2] - R_rigAngle
            L_angx = cmds.xform(L_hNd, q=True, t=True, ws=True)[0] - cmds.xform(L_uA, q=True, t=True, ws=True)[0]
            L_angy = cmds.xform(L_uA, q=True, t=True, ws=True)[1] - cmds.xform(L_hNd, q=True, t=True, ws=True)[1]
            L_en = (L_angx ** 2 + L_angy ** 2) ** 0.5
            L_rad = math.acos(L_angx / L_en)
            L_rigAngle = L_rad * (180 / math.pi)
            L_plang = cmds.xform(L_hndCon, q=True, ro=True, ws=True)[2] + L_rigAngle
            cmds.setAttr(R_hndCon + ".rz", R_plang)
            cmds.setAttr(L_hndCon + ".rz", L_plang)
            # ty 값을 어깨와 동일하게 맞추기
            plsy = cmds.getAttr(L_hndCon + ".ty") + cmds.xform(L_uA, q=True, t=True, ws=True)[1] - cmds.xform(L_hNd, q=True, t=True, ws=True)[1]
            L_conYsub = cmds.xform(L_hNd, q=True, t=True, ws=True)[1] - cmds.xform(L_hndCon, q=True, t=True, ws=True)[1]
            if L_conYsub < 0:
                cmds.setAttr(L_hndCon + ".ty", plsy - abs(L_conYsub))
            elif L_conYsub > 0:
                cmds.setAttr(L_hndCon + ".ty", plsy + abs(L_conYsub))
            else:
                cmds.setAttr(L_hndCon + ".ty", plsy)
            Rplsy = cmds.getAttr(R_hndCon + ".ty") + cmds.xform(R_uA, q=True, t=True, ws=True)[1] - cmds.xform(R_hNd, q=True, t=True, ws=True)[1]
            R_conYsub = cmds.xform(R_hNd, q=True, t=True, ws=True)[1] - cmds.xform(R_hndCon, q=True, t=True, ws=True)[1]
            if R_conYsub < 0:
                cmds.setAttr(R_hndCon + ".ty", Rplsy - abs(R_conYsub))
            elif R_conYsub > 0:
                cmds.setAttr(R_hndCon + ".ty", Rplsy + abs(R_conYsub))
            else:
                cmds.setAttr(R_hndCon + ".ty", Rplsy)
            # 팔 길이를 계산하여 T 포즈가 되도록 손목을 x축으로 당겨주기
            stxUpArm = cmds.xform(L_fA, q=True, t=True, ws=True)[0] - cmds.xform(L_uA, q=True, t=True, ws=True)[0]
            styUpArm = cmds.xform(L_uA, q=True, t=True, ws=True)[1] - cmds.xform(L_fA, q=True, t=True, ws=True)[1]
            stzArm = cmds.xform(L_fA, q=True, t=True, ws=True)[2] - cmds.xform(L_uA, q=True, t=True, ws=True)[2]
            stxforeArm = cmds.xform(L_hNd, q=True, t=True, ws=True)[0] - cmds.xform(L_fA, q=True, t=True, ws=True)[0]
            styforeArm = cmds.xform(L_hNd, q=True, t=True, ws=True)[1] - cmds.xform(L_fA, q=True, t=True, ws=True)[1]
            dix = cmds.xform(L_hNd, q=True, t=True, ws=True)[0] - cmds.xform(L_uA, q=True, t=True, ws=True)[0]
            eix = (stxUpArm ** 2 + styUpArm ** 2 + stzArm ** 2) ** 0.5 + (styforeArm ** 2 + stxforeArm ** 2 + stzArm ** 2) ** 0.5
            subx = eix - dix
            plsx = cmds.getAttr(L_hndCon + ".tx") + subx
            cmds.setAttr(L_hndCon + ".tx", plsx)
            RstxUpArm = cmds.xform(R_uA, q=True, t=True, ws=True)[0] - cmds.xform(R_fA, q=True, t=True, ws=True)[0]
            RstyUpArm = cmds.xform(R_uA, q=True, t=True, ws=True)[1] - cmds.xform(R_fA, q=True, t=True, ws=True)[1]
            RstzArm = cmds.xform(R_fA, q=True, t=True, ws=True)[2] - cmds.xform(R_uA, q=True, t=True, ws=True)[2]
            RstxforeArm = cmds.xform(R_fA, q=True, t=True, ws=True)[0] - cmds.xform(R_hNd, q=True, t=True, ws=True)[0]
            RstyforeArm = cmds.xform(R_hNd, q=True, t=True, ws=True)[1] - cmds.xform(R_fA, q=True, t=True, ws=True)[1]
            Rdix = cmds.xform(R_uA, q=True, t=True, ws=True)[0] - cmds.xform(R_hNd, q=True, t=True, ws=True)[0]
            Reix = (RstxUpArm ** 2 + RstyUpArm ** 2 + RstzArm ** 2) ** 0.5 + (RstyforeArm ** 2 + RstxforeArm ** 2 + RstzArm ** 2) ** 0.5
            Rsubx = Reix - Rdix
            Rplsx = cmds.getAttr(R_hndCon + ".tx") - Rsubx
            cmds.setAttr(R_hndCon + ".tx", Rplsx)
            tx = self.pvls.checkState()
            if str(tx) == "PySide2.QtCore.Qt.CheckState.Unchecked":
                for j in polv_List:
                    cmds.setAttr(self.nsChar + ":" + j + ".localSpace", 0)
            else:
                for j in polv_List:
                    cmds.setAttr(self.nsChar + ":" + j + ".localSpace", 1)
            # 폴벡터를 팔꿈치에서 Z축으로 0 만큼 떨어지도록 만들기
            delCon = cmds.pointConstraint(L_fA, self.nsChar + ":" + "L_IK_handVec_CON")
            cmds.delete(delCon)
            polV = cmds.getAttr(self.nsChar + ":" + "L_IK_handVec_CON.tz")
            cmds.setAttr(self.nsChar + ":" + "L_IK_handVec_CON.tz", polV-3)
            RdelCon = cmds.pointConstraint(R_fA, self.nsChar + ":" + "R_IK_handVec_CON")
            cmds.delete(RdelCon)
            RpolV = cmds.getAttr(self.nsChar + ":" + "R_IK_handVec_CON.tz")
            cmds.setAttr(self.nsChar + ":" + "R_IK_handVec_CON.tz", RpolV-3)
            L_legPolv = cmds.getAttr(self.nsChar + ":" + polv_List[2] + ".tz")
            cmds.setAttr(self.nsChar + ":" + polv_List[2] + ".tz", L_legPolv + 3)
            R_legPolv = cmds.getAttr(self.nsChar + ":" + polv_List[3] + ".tz")
            cmds.setAttr(self.nsChar + ":" + polv_List[3] + ".tz", R_legPolv + 3)

    def unmuteCh(self, restFrame):
        mel.eval("channelBoxCommand -break; ")
        if len(str(restFrame)) != 0:
            tFr = restFrame - 1
        else:
            cmds.warning("Input frame.")
        cmds.currentTime(tFr)
        for i in self.place_list:
            tfName = self.nsChar + ":" + i
            cmds.mute(tfName, d=True)
        for j in range(len(self.trAttr)):
            if self.toggleState[self.trAttr[j]] == 0:
                cmds.cutKey(self.nsChar + ":place_CON", at=self.trAttr[j], t=(tFr, tFr))
            elif self.toggleState[self.trAttr[j]] == 1:
                pass
            cmds.setAttr(self.nsChar + ":place_CON." + self.trAttr[j], self.oldValues[j])

    def muteCh(self, restFrame):
        if len(str(restFrame)) != 0:
            tFr = restFrame - 1
        else:
            cmds.warning("Input frame.")
        cmds.currentTime(tFr)
        self.oldValues = []
        self.toggleState = {'translateX': float(), 'translateY': float(), 'translateZ': float(), 'rotateX': float(), 'rotateY': float(), 'rotateZ': float()}
        for i in self.trAttr:
            self.oldValues.append(cmds.getAttr(self.nsChar + ":place_CON." + i))
            if cmds.keyframe(self.nsChar + ":place_CON", at=i, q=True, t=(tFr, tFr)) != None:
                self.toggleState[i] = 1
            else:
                self.toggleState[i] = 0
        for a in self.place_list:
            tfName = self.nsChar + ":" + a
            for x in self.trAttr:
                cmds.setAttr(tfName + "." + x, 0)
                cmds.setKeyframe(tfName, at=x)
            cmds.mute(tfName)

    def jsmk(self, data, filepath):
        if data:
            f = open(filepath, 'w')
            json.dump(data, f, indent=4, sort_keys=True)
            f.close()

    def saveKeyFrames(self, exAnim, restFrame, startFrame):
        savFilePath = str(exAnim).replace(".anim", ".json")
        jset = dict()
        jset['nk'] = dict()
        jset['pl'] = {'move_CON': dict(), 'direction_CON': dict(), 'place_CON': dict()}
        jset['bcf'] = dict()
        larm = {'L_FK_upArm_CON': dict(), 'L_FK_foreArm_CON': dict(), 'L_FK_hand_CON': dict()}
        rarm = {'R_FK_upArm_CON': dict(), 'R_FK_foreArm_CON': dict(), 'R_FK_hand_CON': dict()}
        lleg = {'L_FK_leg_CON': dict(), 'L_FK_lowLeg_CON': dict(), 'L_FK_foot_CON': dict(), 'L_FK_ball_CON': dict()}
        rleg = {'R_FK_leg_CON': dict(), 'R_FK_lowLeg_CON': dict(), 'R_FK_foot_CON': dict(), 'R_FK_ball_CON': dict()}
        jset['blCon'] = {'L_legBlend_CON': lleg, 'R_legBlend_CON': rleg, 'L_armBlend_CON': larm, 'R_armBlend_CON': rarm}
        if str(cmds.ls(sl=1)[0]).count(":") == 1:
            if cmds.referenceQuery(str(cmds.ls(sl=1)[0]), inr=True):
                jset['refFile'] = str(cmds.referenceQuery(str(cmds.ls(sl=1)[0]), f=True))  # 선택한 요소의 레퍼런스 파일 경로
            else:
                jset['refFile'] = "Imported File"
            # move, direction, place 컨트롤러 키값 저장
            for s in self.place_list:
                cont_Name = self.nsChar + ":" + s
                for v in cmds.listAttr(cont_Name, k=True):
                    if cmds.keyframe(cont_Name, at=str(v), q=True) != None:
                        jset['pl'][s][v] = dict()
                        for n in cmds.keyframe(cont_Name, at=str(v), q=True):
                            jset['pl'][s][v][n] = cmds.getAttr(cont_Name + "." + str(v), t=n)
                    else:
                        jset['pl'][s][v] = cmds.getAttr(cont_Name + "." + str(v))
            bleCon = ['L_legBlend_CON', 'R_legBlend_CON', 'L_armBlend_CON', 'R_armBlend_CON']
            actSTF = int(startFrame)
            # s v n i j k
            for w in bleCon:
                bleConName = self.nsChar + ":" + w
                if cmds.keyframe(bleConName, at='FKIKBlend', q=True) == None:
                    if cmds.getAttr(bleConName + '.FKIKBlend') == 0.0:
                        jset['bcf'][w] = 1
                        for q in jset['blCon'][w].keys():
                            fkcName = self.nsChar + ":" + q
                            for t in self.trAttr:
                                if cmds.keyframe(fkcName, at=t, q=True) != None:
                                    jset['blCon'][w][q][t] = dict()
                                    for o in cmds.keyframe(fkcName, at=t, q=True):
                                        jset['blCon'][w][q][t][o] = cmds.getAttr(fkcName + "." + t, t=o)
                                else:
                                    jset['blCon'][w][q][t] = cmds.getAttr(fkcName + "." + t)
                    else:
                        jset['bcf'][w] = 0
                else:
                    if cmds.getAttr(bleConName + '.FKIKBlend', t=actSTF) == 0.0:
                        jset['bcf'][w] = 1
                        for q in jset['blCon'][w].keys():
                            fkcName = self.nsChar + ":" + q
                            for t in self.trAttr:
                                if cmds.keyframe(fkcName, at=t, q=True) != None:
                                    jset['blCon'][w][q][t] = dict()
                                    for o in cmds.keyframe(fkcName, at=t, q=True):
                                        jset['blCon'][w][q][t][o] = cmds.getAttr(fkcName + "." + t, t=o)
                                else:
                                    jset['blCon'][w][q][t] = cmds.getAttr(fkcName + "." + t)
                    else:
                        jset['bcf'][w] = 0

            # 각 키 탄젠트 옵션값 저장
            cmds.select(cl=True)
            for i in cmds.ls(self.nsChar + ":*_CON"):
                conNm = str(i.split(":")[1])
                jset[conNm] = dict()
                for j in cmds.listAttr(str(i), k=True):
                    jset[conNm][j] = dict()
                    if cmds.keyframe(str(i) + "." + str(j), q=True):
                        for k in cmds.keyframe(str(i), at=str(j), q=True):
                            jset[conNm][j][k] = cmds.keyTangent(str(i), at=str(j), q=True, t=(k, k), ia=True, oa=True, iw=True, ow=True, l=True)
                    else:
                        jset['nk'][conNm + "." + str(j)] = cmds.getAttr(str(i) + "." + str(j))
                    if len(jset[conNm][j]) == 0:
                        del jset[conNm][j]
                    else:
                        pass
                if len(jset[conNm]) == 0:
                    del jset[conNm]
                else:
                    pass
            self.jsmk(jset, savFilePath)  # json 생성
            self.muteCh(restFrame)
        else:
            cmds.warning("Please select an Object that has a Namespace.")
            return

def main():
    global myWindow
    try:
        myWindow.close()
    except:
        pass
    myWindow = Window()
    myWindow.show()
    myWindow.resize(218,430)
    return myWindow

if __name__ == '__main__':
    main()
