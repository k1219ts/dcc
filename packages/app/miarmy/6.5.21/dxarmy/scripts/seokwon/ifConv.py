# encoding:utf-8
# !/usr/bin/env python

# IK 손 컨트롤러의 움직임을 FK 컨트롤러로 복사

import os
import maya.cmds as cmds
from Qt import QtCore, QtGui, QtWidgets, load_ui
currentpath = os.path.abspath(__file__)
uiFile = os.path.join(os.path.dirname(currentpath), "../ui/ifConv.ui")

class Window(QtGui.QMainWindow):
    def __init__(self, parent = None):
        super(Window, self).__init__(parent)
        self.ui = load_ui(uiFile)

        self.IKPOV_list = []
        self.placeList = ['move_CON', 'direction_CON', 'place_CON']
        self.trAttr = ['translateX', 'translateY', 'translateZ']
        self.rtAttr = ['rotateX', 'rotateY', 'rotateZ']
        self.connectSignal()

    def connectSignal(self):
        '''
        Qt버튼 - 함수 연결
        :return: None 
        '''
        self.ui.itfBtn.clicked.connect(self.main)

    def IKPOVBake(self, POV_set_List):
        '''
        Pole vector bake
        :return: None
        '''
        self.getTimeline()
        nsChar = [self.nsChar]
        decomposedMatrix_list = []
        plusMinusAverage_list = []
        multiplyDivide_list = []
        delCon = []

        # joint part : 4
        for i in range(len(POV_set_List)):
            del decomposedMatrix_list[0:len(decomposedMatrix_list)]
            del plusMinusAverage_list[0:len(plusMinusAverage_list)]
            del multiplyDivide_list[0:len(multiplyDivide_list)]
            # decomposed_Node : 4
            for x in range(4):
                decomposed_node = cmds.createNode('decomposeMatrix',
                                                  n='%s:%s_%s_DCM' % (nsChar[0], POV_set_List[i], x + 1))
                decomposedMatrix_list.append(decomposed_node)
                delCon.append(decomposed_node)
            # plusMinusAverage_Node : 6
            for y in range(6):
                plusMinusAverage_node = cmds.createNode('plusMinusAverage',
                                                        n='%s:%s_%s_PMA' % (nsChar[0], POV_set_List[i], y + 1))
                if y <= 3:
                    cmds.setAttr('%s.operation' % plusMinusAverage_node, 2)
                plusMinusAverage_list.append(plusMinusAverage_node)
                delCon.append(plusMinusAverage_node)
            # multiplyDivide_Node : 2
            for z in range(2):
                multiplyDivide_node = cmds.createNode('multiplyDivide',
                                                      n='%s:%s_%s_MPD' % (nsChar[0], POV_set_List[i], z + 1))
                cmds.setAttr('%s.input2X' % multiplyDivide_node, 2)
                cmds.setAttr('%s.input2Y' % multiplyDivide_node, 2)
                cmds.setAttr('%s.input2Z' % multiplyDivide_node, 2)
                if z <= 0:
                    cmds.setAttr('%s.operation' % multiplyDivide_node, 2)
                multiplyDivide_list.append(multiplyDivide_node)
                delCon.append(multiplyDivide_node)
            # composeMatrix : 1
            composeMatrix_node = cmds.createNode('composeMatrix',
                                                 n='%s:%s_%s_CPM' % (nsChar[0], POV_set_List[i], i + 1))
            delCon.append(composeMatrix_node)
            # multiMatrix : 1
            multMatrix_node = cmds.createNode('multMatrix', n='%s:%s_%s_MMX' % (nsChar[0], POV_set_List[i], i + 1))
            delCon.append(multMatrix_node)

            # connectAttr
            for a in range(3):
                cmds.connectAttr('%s:%s.worldMatrix' % (nsChar[0], self.IKPOV_list[i][a]),
                                 '%s.inputMatrix' % decomposedMatrix_list[a])
                cmds.connectAttr('%s.outputTranslate' % decomposedMatrix_list[a],
                                 '%s.input3D[0]' % plusMinusAverage_list[a])
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
            cmds.connectAttr('%s.outputTranslate' % decomposedMatrix_list[3],
                             '%s:%s.translate' % (nsChar[0], self.IKPOV_list[i][3]))
            cmds.connectAttr('%s:%s.parentInverseMatrix' % (nsChar[0], self.IKPOV_list[i][3]),
                             '%s.matrixIn[1]' % multMatrix_node)

        cmds.currentTime(cmds.playbackOptions(q=True, minTime=True))
        cmds.select(cl=True)
        polVec = {}
        povNd = {"L_FK_lowLeg_JNT": {self.nsChar + ":L_FK_lowLeg_JNT_4_DCM": self.nsChar + ":L_IK_footVec_CON"},
                 "R_FK_lowLeg_JNT": {self.nsChar + ":R_FK_lowLeg_JNT_4_DCM": self.nsChar + ":R_IK_footVec_CON"},
                 "L_FK_upArm_JNT": {self.nsChar + ":L_FK_upArm_JNT_4_DCM": self.nsChar + ":L_IK_handVec_CON"},
                 "R_FK_upArm_JNT": {self.nsChar + ":R_FK_upArm_JNT_4_DCM": self.nsChar + ":R_IK_handVec_CON"}}
        for dn in POV_set_List:
            if povNd[dn]:
                polVec.update(povNd[dn])
            else:
                pass
        for n in polVec.values():
            if len(cmds.ls(str(n))) != 0:
                cmds.select(str(n), add=True)
        cmds.bakeResults(cmds.ls(sl=True), simulation=True, t=(self.minTime, self.maxTime), sampleBy=1, dic=True,
                         pok=True, sac=False, ral=False, bol=False, mr=True, controlPoints=False, shape=True)
        for w in range(len(polVec)):
            cmds.disconnectAttr(str(polVec.keys()[w]) + ".outputTranslate", str(polVec.values()[w]) + ".translate")
            cmds.delete(str(polVec.keys()[w]))

    def getNamespace(self):
        '''
        Namespace Query
        :return: None
        '''
        if len(cmds.ls(sl=True)) != 0:
            self.nsChar = str(cmds.ls(sl=True)[0]).rsplit(":")[0]
        else:
            cmds.warning("Select object have a namespace.")

    def getTimeline(self):
        '''
        Timeline Query
        :return: None
        '''
        self.maxTime = cmds.playbackOptions(q=True, max=True)
        self.minTime = cmds.playbackOptions(q=True, min=True)

    def main(self):
        selCtrl = cmds.ls(sl=True)
        self.getNamespace()
        self.getTimeline()
        povList = {"L_legBlend_CON": "L_FK_lowLeg_JNT", "R_legBlend_CON": "R_FK_lowLeg_JNT",
                   "L_armBlend_CON": "L_FK_upArm_JNT", "R_armBlend_CON": "R_FK_upArm_JNT"}
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
        fkaJnt = {"R_Blend_upArm_JNT": "R_FK_upArm_CON", "R_Blend_foreArm_JNT": "R_FK_foreArm_CON",
                  "R_Blend_hand_JNT": "R_FK_hand_CON",
                  "L_Blend_upArm_JNT": "L_FK_upArm_CON", "L_Blend_foreArm_JNT": "L_FK_foreArm_CON",
                  "L_Blend_hand_JNT": "L_FK_hand_CON",
                  "R_Blend_leg_JNT": "R_FK_leg_CON", "R_Blend_lowLeg_JNT": "R_FK_lowLeg_CON",
                  "R_Blend_foot_JNT": "R_FK_foot_CON",
                  "L_Blend_leg_JNT": "L_FK_leg_CON", "L_Blend_lowLeg_JNT": "L_FK_lowLeg_CON",
                  "L_Blend_foot_JNT": "L_FK_foot_CON"}
        ikConSet = {"L_legBlend_CON": ["L_IK_foot_CON", "L_IK_footVec_CON"],
                    "R_legBlend_CON": ["R_IK_foot_CON", "R_IK_footVec_CON"],
                    "L_armBlend_CON": ["L_IK_hand_CON", "L_IK_handVec_CON"],
                    "R_armBlend_CON": ["R_IK_hand_CON", "R_IK_handVec_CON"]}
        # Joint - Locators Constraint
        cmds.currentTime(self.minTime)
        locL = []  # Locators List
        temConstList = []  # Temp Constrain List
        for i in blendList:
            worldP = cmds.spaceLocator()[0]
            locL.append(worldP)
            temCons = cmds.parentConstraint(self.nsChar + ":" + i, worldP, w=1, mo=False)[0]
            temConstList.append(temCons)
        cmds.select(cl=True)
        # Bake Locators & Delete Constraints
        for k in locL:
            cmds.select(k, add=True)
        cmds.bakeResults(cmds.ls(sl=True), simulation=True, t=(self.minTime, self.maxTime), sampleBy=1, dic=True,
                         pok=True, sac=False, ral=False, bol=False, mr=True, controlPoints=False, shape=True)
        cmds.delete(temConstList)
        # FK Ctrls On
        ikConList = []
        for e in blendCtrl:
            cmds.setAttr(str(e) + ".FKIKBlend", 0)
            ikConList += ikConSet[str(e).split(":")[1]]
        cmds.currentTime(self.minTime)
        # BlendJoint's Attr - FK Ctrls Data minTime Pose Sync
        for q in range(len(blendList)):
            for e in range(len(self.trAttr)):
                rt = cmds.getAttr(self.nsChar + ":" + blendList[q] + "." + self.rtAttr[e])
                cmds.setAttr(self.nsChar + ":" + fkaJnt[blendList[q]] + "." + self.rtAttr[e], rt)
        # Locators - FK Ctrls Constraint
        fkList = []
        for j in range(len(blendList)):
            temCon = cmds.orientConstraint(locL[j], self.nsChar + ":" + fkaJnt[blendList[j]], w=1, mo=False)[0]
            fkList.append(fkaJnt[blendList[j]])
        cmds.select(cl=True)
        # FK Ctrls Bake
        for t in fkList:
            cmds.select(self.nsChar + ":" + t, add=True)
        cmds.bakeResults(cmds.ls(sl=True), simulation=True, t=(self.minTime, self.maxTime), sampleBy=1, dic=True,
                         pok=True, sac=False, ral=False, bol=False, mr=True, controlPoints=False, shape=True)
        # Delete Orient Constraints & Locators
        cmds.delete(temCon)
        cmds.delete(locL)
        cmds.select(cl=True)
        for t in fkList:
            cmds.select(self.nsChar + ":" + t, add=True)
        cmds.selectKey(k=True)
        cmds.filterCurve(f="euler")
        cmds.select(cl=True)
        for sn in ikConList:
            cmds.select(self.nsChar + ":" + sn, add=True)
        cmds.selectKey(k=True)
        cmds.cutKey(cl=True)
        cmds.select(cl=True)

    def fkToik(self, blendCtrl, povSetList):
        fkList = {"L_armBlend_CON": {"L_FK_hand_CON": "L_IK_hand_CON"},
                  "R_armBlend_CON": {"R_FK_hand_CON": "R_IK_hand_CON"},
                  "L_legBlend_CON": {"L_FK_foot_CON": "L_IK_foot_CON"},
                  "R_legBlend_CON": {"R_FK_foot_CON": "R_IK_foot_CON"}}
        fkConSet = {"L_legBlend_CON": ["L_FK_leg_CON", "L_FK_lowLeg_CON", "L_FK_foot_CON", "L_FK_ball_CON"],
                    "R_legBlend_CON": ["R_FK_leg_CON", "R_FK_lowLeg_CON", "R_FK_foot_CON", "R_FK_ball_CON"],
                    "L_armBlend_CON": ["L_FK_upArm_CON", "L_FK_foreArm_CON", "L_FK_hand_CON"],
                    "R_armBlend_CON": ["R_FK_upArm_CON", "R_FK_foreArm_CON", "R_FK_hand_CON"]}
        cmds.currentTime(self.minTime)
        # Joint - Locators Constraint
        locL = []
        temConsc = []
        locD = dict()
        for i in blendCtrl:
            worldP = cmds.spaceLocator()[0]
            locL.append(worldP)
            locD[str(i).split(":")[1]] = worldP
            temConsc.append(
                cmds.pointConstraint(self.nsChar + ":" + fkList[str(i).split(":")[1]].keys()[0], worldP, w=1, mo=False)[
                    0])
            temConsc.append(
                cmds.orientConstraint(self.nsChar + ":" + fkList[str(i).split(":")[1]].keys()[0], worldP, w=1, mo=True)[
                    0])
        cmds.select(cl=True)
        # Bake Locators & Delete Constraints
        for k in locL:
            cmds.select(k, add=True)
        cmds.bakeResults(cmds.ls(sl=True), simulation=True, t=(self.minTime, self.maxTime), sampleBy=1, dic=True,
                         pok=True, sac=False, ral=False, bol=False, mr=True, controlPoints=False, shape=True)
        cmds.delete(temConsc)
        fkConList = []
        for e in blendCtrl:
            cmds.setAttr(str(e) + ".FKIKBlend", 1)
            fkConList += fkConSet[str(e).split(":")[1]]
        stemCon = []
        for j in blendCtrl:
            stemCon.append(cmds.parentConstraint(locD[str(j).split(":")[1]],
                                                 self.nsChar + ":" + fkList[str(j).split(":")[1]].values()[0], w=1,
                                                 mo=False)[0])
        cmds.select(cl=True)
        for c in blendCtrl:
            cmds.select(self.nsChar + ":" + fkList[str(c).split(":")[1]].values()[0], add=True)
        cmds.bakeResults(cmds.ls(sl=True), simulation=False, t=(self.minTime, self.maxTime), sampleBy=1, dic=True,
                         pok=True, sac=False, ral=False, bol=False, mr=True, controlPoints=False, shape=True)
        cmds.delete(stemCon)
        cmds.delete(locL)
        # IK Ctrls On
        cmds.currentTime(self.minTime)
        self.IKPOVBake(povSetList)
        cmds.select(cl=True)
        for r in blendCtrl:
            cmds.select(self.nsChar + ":" + fkList[str(r).split(":")[1]].values()[0], add=True)
        cmds.selectKey(k=True)
        cmds.filterCurve(f="euler")
        cmds.select(cl=True)
        for sn in fkConList:
            cmds.select(self.nsChar + ":" + sn, add=True)
        cmds.selectKey(k=True)
        cmds.cutKey(cl=True)
        cmds.select(cl=True)

def main():
    global myWindow
    myWindow = Window()
    myWindow.ui.show()

if __name__ == '__main__':
    main()
