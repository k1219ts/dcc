# encoding:utf-8
# !/usr/bin/env python

import os
import maya.cmds as cmds
import maya.mel as mel
import json
import math
from Qt import QtCore, QtGui, QtWidgets, load_ui
import dxUI

mel.eval("cycleCheck -e off")
currentpath = os.path.abspath(__file__)
uiFile = os.path.join(os.path.dirname(currentpath), "ui", "crwAnimBridge.ui")

class Window(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        dxUI.setup_ui(uiFile, self)
        mel.eval('HIKCharacterControlsTool;')

        self.connectSignal()
        self.moduleJoint_list = ['Crw_Hips', 'Crw_Spine', 'Crw_Spine1', 'Crw_Spine2',
                                 'Crw_Spine3', 'Crw_Neck', 'Crw_Head', 'Crw_LeftShoulder',
                                 'Crw_LeftArm', 'Crw_LeftForeArm', 'Crw_LeftHand', 'Crw_LefthandSub1',
                                 'Crw_RightShoulder', 'Crw_RightArm', 'Crw_RightForeArm', 'Crw_RightHand',
                                 'Crw_RightHandSub1', 'Crw_LeftUpLeg', 'Crw_LeftLeg', 'Crw_LeftFoot',
                                 'Crw_LeftToeBase', 'Crw_RightUpLeg', 'Crw_RightLeg', 'Crw_RightFoot',
                                 'Crw_RightToeBase']
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
        self.expBtn.clicked.connect(self.expAnim)
        self.tpBTN.clicked.connect(self.makeTpose)
        self.impBtn.clicked.connect(self.impAnim)
        self.pvls.stateChanged.connect(self.chkState)

    def getNamespace(self):
        if cmds.ls(sl=1):
            sel = cmds.ls(sl=True)[0]
            self.nsChar = str(sel.rsplit(':')[0])
        else:
            cmds.warning('"select Character!!"')

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
        cmds.currentTime(cmds.playbackOptions(q=True, min=True))
        polV = cmds.getAttr(self.nsChar + ":" + "L_IK_handVec_CON.tz")
        cmds.setAttr(self.nsChar + ":" + "L_IK_handVec_CON.tz", polV - 3)
        RpolV = cmds.getAttr(self.nsChar + ":" + "R_IK_handVec_CON.tz")
        cmds.setAttr(self.nsChar + ":" + "R_IK_handVec_CON.tz", RpolV - 3)
        L_legPolv = cmds.getAttr(self.nsChar + ":L_IK_footVec_CON.tz")
        cmds.setAttr(self.nsChar + ":L_IK_footVec_CON.tz", L_legPolv + 3)
        R_legPolv = cmds.getAttr(self.nsChar + ":R_IK_footVec_CON.tz")
        cmds.setAttr(self.nsChar + ":R_IK_footVec_CON.tz", R_legPolv + 3)

    def timelineSet(self):
        self.minCurrent = cmds.playbackOptions(q=1, min=True) - 1
        self.maxCurrent = cmds.playbackOptions(q=1, max=True) + 1

    def chkState(self):
        tx = self.pvls.checkState()
        if str(tx) == "PySide2.QtCore.Qt.CheckState.Unchecked":
            self.pvls.setText("Pole Vector Local Space  :   0")
        else:
            self.pvls.setText("Pole Vector Local Space  :   1")

    def expAnim(self):
        self.getNamespace()
        minTime = cmds.playbackOptions(q=True, min=True)
        maxTime = cmds.playbackOptions(q=True, max=True)
        cmds.select("%s:%s" % (self.nsChar, "Crw_Hips"), hi=True)
        cmds.bakeResults(cmds.ls(sl=True), hi=True, simulation=True, t=(int(minTime), int(maxTime)), sampleBy=1, dic=True, pok=True, sac=False, ral=False, bol=False, mr=True, controlPoints=False, shape=True)
        cmds.selectKey(k=True)
        cmds.filterCurve(f="euler")
        cmds.select(cl=True)
        cmds.select("%s:%s" % (self.nsChar, "Crw_Hips"), hi=True)
        exAnim = cmds.fileDialog2(startingDirectory="/dexter/Cache_DATA/", fileMode=0, fileFilter="animExport (*.anim)", caption="Export Anim")
        cmds.file(exAnim, force=True, options="precision=8;nodeNames=1;verboseUnits=0;whichRange=1;options=keys;hierarchy=below;controlPoints=0;shapes=1;useChannelBox=0;copyKeyCmd=-animation objects -option keys -hierarchy below -controlPoints 0 -shape 1", typ="animExport", eas=True)

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

    def blendCheck(self):
        '''
        IK/FK Blend Check. FK -> IK
        :return: None
        '''
        self.getNamespace()
        for i in cmds.ls(self.nsChar + ":*Blend_CON"):
            if cmds.getAttr(str(i) + ".FKIKBlend") == 0:
                cmds.setAttr(str(i) + ".FKIKBlend", 1)

    def impAnim(self):
        '''
        - Mocap Anim File Import
        :return: None
        '''
        self.targetNS = []
        self.delConstraints = []
        self.getNamespace()
        self.blendCheck()
        cmds.select(cl=True)
        scrP = "/dexter/Cache_DATA/mocap/02_Data/02_CharMocap"
        animDir = cmds.fileDialog2(fileMode=1, fileFilter="animExport (*.anim)", dir=scrP, caption="Import Anim File")
        cmds.file('/dexter/Cache_DATA/animation/A14_Asset/HumanIK_for_AnimBrowser/mocapJoint.ma', i=True,
                  type='mayaAscii', ra=True, mergeNamespacesOnClash=True, namespace="mcp", options='v=0')
        cmds.select("mcp:Crw_Hips")
        cmds.file(animDir[0], i=True)
        mxT = cmds.keyframe("mcp:Crw_Hips", q=True, lastSelected=True, timeChange=True)
        mnT = cmds.keyframe("mcp:Crw_Hips", q=True)[0]
        cmds.playbackOptions(minTime=mnT, maxTime=mxT[0])
        cmds.select(cl=True)
        cmds.currentTime(mnT)
        cmds.file('/dexter/Cache_DATA/animation/A14_Asset/HumanIK_for_AnimBrowser/huType_hikJoint.ma', i=True,
                  type='mayaAscii', ra=True, mergeNamespacesOnClash=True, namespace='mcp_hik', options='v=0')
        mocapJoint_constraint_list = []
        for i in range(len(self.mocapJoint_list)):
            temp_point_con = cmds.pointConstraint("%s:%s" % ("mcp", self.mocapJoint_list[i]),
                                                  "%s:%s" % ("mcp_hik", self.mocapJoint_list[i]), mo=0, w=1)
            mocapJoint_constraint_list.append(temp_point_con[0])  # constrain 일괄 삭제를 위한 리스트 축적
        cmds.delete(mocapJoint_constraint_list)  # Constrain 삭제
        for j in range(len(self.mocapJoint_list)):
            cmds.parentConstraint("%s:%s" % ("mcp", self.mocapJoint_list[j]),
                                  "%s:%s" % ("mcp_hik", self.mocapJoint_list[j]), mo=1, w=1)
        # HIKJoint 뼈 로드해서 선택한 캐릭터에 Constrain 걸기
        if cmds.ls(type='transform').count("%s:HIKJoint_LOC" % self.nsChar) == 1:
            cmds.delete("%s:HIKJoint_LOC" % self.nsChar)
        cmds.file('/dexter/Cache_DATA/animation/A14_Asset/HumanIK_for_AnimBrowser/HIKJoint.ma', i=True,
                  type='mayaAscii', ra=True, mergeNamespacesOnClash=True, namespace=self.nsChar, options='v=0')
        HIKJoint_constraint_list = []
        for u in range(len(self.HIKJoint_list)):
            temp_point_con = cmds.pointConstraint("%s:%s" % (self.nsChar, self.moduleJoint_list[u]),
                                                  "%s:%s" % (self.nsChar, self.HIKJoint_list[u]), mo=0, w=1)
            HIKJoint_constraint_list.append(temp_point_con[0])  # constrain 일괄 삭제를 위한 리스트 축적

        # if (self.nsChar + ":Crw_Sword"):
        # elif (self.nsChar + ":Crw_Spear"):
        # elif (self.naChar + ":Crw_Bow"):
        # 무기 붙이는 위치 <=================================== 컨트롤러 필요

        cmds.delete(HIKJoint_constraint_list)  # Constrain 삭제
        self.targetNS.append(self.nsChar)  # NameSpace 입력

        for r in range(len(self.HIK_attatch_toIK_list)):
            self.toIK_Const = cmds.parentConstraint("%s:%s" % (self.nsChar, self.HIK_attatch_toIK_list[r]),
                                                    "%s:%s" % (self.nsChar, self.IKController_list[r]), mo=1, w=1)
            self.delConstraints.append(self.toIK_Const[0])

        mel.eval('$gHIKCurrentCharacter = "%s:%s"' % (self.nsChar, "set_HIK"))
        mel.eval('hikToggleLockDefinition();')
        mel.eval('refreshAllCharacterLists();')
        mel.eval('mayaHIKsetCharacterInput("%s:%s","%s:%s");' % (self.nsChar, "set_HIK", 'mcp_hik', "Character1"))
        self.IKPOVBake()
        self.timelineSet()
        self.bakeIKControl()
        for i in range(len(self.IKController_list)):
            cmds.select('%s:%s' % (self.nsChar, self.IKController_list[i]), add=True)
        cmds.selectKey(k=True)
        cmds.filterCurve(f="euler")
        cmds.select(cl=True)
        if self.delConstraints == None:
            pass
        else:
            cmds.delete(self.toIK_Const)
            cmds.delete(self.nsChar + ":HIKJoint_LOC")
            delLst = ["mcp:mocap_char", "mcp_hik:mocap_char", "*:set_HIK"]
            for delf in delLst:
                if len(cmds.ls(delf)) != 0:
                    cmds.delete(delf)
                else:
                    pass
        cmds.delete("mcp_hik:*")
        cmds.select(self.nsChar + ":*_CON")
        cmds.delete(sc=True, uac=False, hi="none", cp=False, s=True)
        cmds.select(cl=True)
        cmds.currentTime(cmds.playbackOptions(q=True, min=True))

        # 이동 컨트롤러 잡고 Y축 아래로 -0.72463만큼 내려서 바닥에 발 맞추는 부분
        # 리타겟팅으로 인해 발이 지면에서 뜨게 되는데 이는 HIK Joint Set 의 root joint의 높이차 때문
        if len(cmds.ls(self.nsChar + ":root_CON")) != 0:
            movecon = ["root_CON", "L_IK_hand_CON", "R_IK_hand_CON", "L_IK_foot_CON", "R_IK_foot_CON",
                       "L_IK_footVec_CON", "R_IK_footVec_CON", "L_IK_handVec_CON", "R_IK_handVec_CON"]
        elif len(cmds.ls(self.nsChar + ":C_IK_root_CON")) != 0:
            movecon = ["C_IK_root_CON", "L_IK_foreLeg_CON", "R_IK_foreLeg_CON", "L_IK_knee_CON", "R_IK_knee_CON",
                       "L_IK_hindLeg_CON", "R_IK_hindLeg_CON", "L_IK_edbow_CON", "R_IK_edbow_CON"]
        else:
            pass
        cmds.select(cl=True)
        for i in movecon:
            cmds.keyframe(self.nsChar + ":" + i, at="translateY", e=True, iub=True, r=True, o="over", vc=-0.72463)

    def makeTpose(self):
        self.getNamespace()
        polv_List = ["L_IK_handVec_CON", "R_IK_handVec_CON", "L_IK_footVec_CON", "R_IK_footVec_CON"]
        L_uA = self.nsChar + ":Crw_LeftArm"
        L_fA = self.nsChar + ":Crw_LeftForeArm"
        L_hNd = self.nsChar + ":Crw_LeftHand"
        L_hndCon = self.nsChar + ":L_IK_hand_CON"
        R_uA = self.nsChar + ":Crw_RightArm"
        R_fA = self.nsChar + ":Crw_RightForeArm"
        R_hNd = self.nsChar + ":Crw_RightHand"
        R_hndCon = self.nsChar + ":R_IK_hand_CON"
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
        # for j in cmds.listAttr():
        #     if str(j).count("R_"):
        #         cmds.setAttr(selns + "." + str(j), -90)
        #     else:
        #         cmds.setAttr(selns + "." + str(j), 90)


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