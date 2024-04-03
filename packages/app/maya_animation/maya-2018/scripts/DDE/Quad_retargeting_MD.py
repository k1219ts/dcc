# encoding=utf-8
#!/usr/bin/env python
import os
import maya.cmds as cmds
import maya.mel as mel

# QuadRuped retargeting "리그 방향은 플러스 Z축이여야하며 원점에서 레스트포즈를 맞춰야한다"는 전제하에 바디
# HIK와 front Leg HIK 두개를 따로 만들어 불러와 소스와 타겟에 각갂 연결해 주는 방식을 취함.  FKIK 노드 연결
# 완료,HIK 두개의 hip위치가 따로노는 이슈 해결 요망.겉운 바율로 이동 해야함.


class QUADRETARGET:
    def __init__(self):
        '''
        컨트롤러, 조인트들 리스트 Bumble Bee Tool 4족기반으로 제작.
        순서가 매우 중요.
        '''
        self.moduleJoint_list = [
            'C_Skin_root_JNT',
            'C_Skin_spine1_JNT',
            'C_Skin_spine2_JNT',
            'C_Skin_spine3_JNT',
            'C_Skin_spine4_JNT',
            'C_Skin_spine5_JNT',
            'C_Skin_spine6_JNT',
            'C_Skin_chest_JNT',
            'C_Skin_neck1_JNT',
            'C_Skin_head_JNT',
            'L_Skin_clavicle_JNT',
            'L_Skin_shoulder_JNT',
            'L_Skin_elbow_JNT',
            'L_Skin_wrist_JNT',
            'L_Skin_foot1_JNT',
            'R_Skin_clavicle_JNT',
            'R_Skin_shoulder_JNT',
            'R_Skin_elbow_JNT',
            'R_Skin_wrist_JNT',
            'R_Skin_foot1_JNT',
            'L_Skin_hip_JNT',
            'L_Skin_knee_JNT',
            'L_Skin_ankle_JNT',
            'L_Skin_ball_JNT',
            'L_Skin_toe_JNT',
            'R_Skin_hip_JNT',
            'R_Skin_knee_JNT',
            'R_Skin_ankle_JNT',
            'R_Skin_ball_JNT',
            'R_Skin_toe_JNT']
        self.HIKJoint_list = [
            'Hips',
            'Spine',
            'Spine1',
            'Spine2',
            'Spine3',
            'Spine4',
            'Spine5',
            'Spine6',
            'Neck',
            'Head',
            'LeftShoulder',
            'LeftArm', 
            'LeftForeArm', 
            'LeftHand', 
            'LeftFingerBase', 
            'RightShoulder', 
            'RightArm', 
            'RightForeArm', 
            'RightHand', 
            'RightFingerBase', 
            'LeftUpLeg',
            'LeftLeg',
            'LeftFoot',
            'LeftToeBase',
            'LeftToe',
            'RightUpLeg',
            'RightLeg',
            'RightFoot',
            'RightToeBase',
            'RightToe']
        self.subFKController_list = [
            'C_IK_root_CON',
            'C_IK_lowBody_CON',
            'C_FKsub_spine2_NUL_CON',
            'C_FKsub_spine3_NUL_CON',
            'C_IK_upBody_CON',
            'C_IK_neck1_CON',
            'C_IK_head_CON',
            'L_FK_foreLeg_clavicle_CON',
            'L_FK_foreLeg_shoulder_CON',
            'L_FK_foreLeg_edbow_CON',
            'L_FK_foreLeg_wrist_CON',
            'L_FK_foreLeg_foot_CON',
            'R_FK_foreLeg_clavicle_CON',
            'R_FK_foreLeg_shoulder_CON',
            'R_FK_foreLeg_edbow_CON',
            'R_FK_foreLeg_wrist_CON',
            'R_FK_foreLeg_foot_CON',
            'L_FK_hindLeg_hip_CON',
            'L_FK_hindLeg_knee_CON',
            'L_FK_hindLeg_ankle_CON',
            'L_FK_hindLeg_ball_CON',
            'L_FK_hindLeg_toe_CON',
            'R_FK_hindLeg_hip_CON',
            'R_FK_hindLeg_knee_CON',
            'R_FK_hindLeg_ankle_CON',
            'R_FK_hindLeg_ball_CON',
            'R_FK_hindLeg_toe_CON']
        self.FKController_list = [
            'C_IK_root_CON',
            'C_IK_lowBody_CON',
            'C_FKsub_spine2_NUL_CON',
            'C_FKsub_spine3_NUL_CON',
            'C_IK_upBody_CON',
            'C_IK_neck1_CON',
            'C_IK_head_CON',
            'L_FK_clavicle_CON',
            'L_FK_shoulder_CON',
            'L_FK_elbow_CON',
            'L_FK_wrist_CON',
            'L_FK_foot1_CON',
            'R_FK_clavicle_CON',
            'R_FK_shoulder_CON',
            'R_FK_elbow_CON',
            'R_FK_wrist_CON',
            'R_FK_foot1_CON',
            'L_FK_hip_CON',
            'L_FK_knee_CON',
            'L_FK_ankle_CON',
            'L_FK_ball_CON',
            'L_FK_toe_CON',
            'R_FK_hip_CON',
            'R_FK_knee_CON',
            'R_FK_ankle_CON',
            'R_FK_ball_CON',
            'R_FK_toe_CON']
        self.HIK_attatch_toFK_list = [
            'Hips',
            'Spine',
            'Spine1',
            'Spine2',
            'Spine3',
            'Neck',
            'Head',
            'LeftShoulder',
            'LeftArm', 
            'LeftForeArm', 
            'LeftHand', 
            'LeftFingerBase', 
            'RightShoulder', 
            'RightArm', 
            'RightForeArm', 
            'RightHand', 
            'RightFingerBase', 
            'LeftUpLeg',
            'LeftLeg',
            'LeftFoot',
            'LeftToeBase',
            'LeftToe',
            'RightUpLeg',
            'RightLeg',
            'RightFoot',
            'RightToeBase',
            'RightToe']
        self.HIK_attatch_tosubFK_list = [
            'Hips',
            'Spine',
            'Spine1',
            'Spine2',
            'Spine3',
            'Neck',
            'Head',
            'LeftShoulder',
            'LeftArm', 
            'LeftForeArm', 
            'LeftHand', 
            'LeftFingerBase', 
            'RightShoulder', 
            'RightArm', 
            'RightForeArm', 
            'RightHand', 
            'RightFingerBase', 
            'LeftUpLeg',
            'LeftLeg',
            'LeftFoot',
            'LeftToeBase',
            'LeftToe',
            'RightUpLeg',
            'RightLeg',
            'RightFoot',
            'RightToeBase',
            'RightToe']
        self.IKController_list = [
            'C_IK_root_CON',
            'C_IK_lowBody_CON',
            'C_FKsub_spine2_NUL_CON',
            'C_FKsub_spine3_NUL_CON',
            'C_IK_upBody_CON',
            'C_IK_neck1_CON',
            'C_IK_head_CON',
            'L_IK_clavicle_CON',
            'L_IK_foreLeg_CON',
            'R_IK_clavicle_CON',
            'R_IK_foreLeg_CON',
            'L_IK_hindLeg_CON',
            'R_IK_hindLeg_CON']
        self.HIK_attatch_toIK_list = [
            'Hips',
            'Spine',
            'Spine1',
            'Spine2',
            'Spine3',
            'Neck',
            'Head',
            'LeftShoulder',
            'LeftFingerBase',
            'RightShoulder',
            'RightFingerBase',
            'LeftToeBase',
            'RightToeBase']
        self.IKPOV_list = [['LeftArm',
                            'LeftForeArm',
                            'LeftHand',
                            'L_IK_elbow_CON'],
                           ['RightArm',
                            'RightForeArm',
                            'RightHand',
                            'R_IK_elbow_CON'],
                           ['LeftUpLeg',
                            'LeftLeg',
                            'LeftFoot',
                            'L_IK_knee_CON'],
                           ['RightUpLeg',
                            'RightLeg',
                            'RightFoot',
                            'R_IK_knee_CON']]
        self.POV_set_List = [
            'L_IK_foreLeg_CON',
            'R_IK_foreLeg_CON',
            'L_IK_hindLeg_CON',
            'R_IK_hindLeg_CON']
        self.HIK_arm_List = [
            'LeftShoulder',
            'LeftArm', 
            'LeftForeArm', 
            'LeftHand', 
            'LeftFingerBase',
            'RightShoulder', 
            'RightArm', 
            'RightForeArm', 
            'RightHand', 
            'RightFingerBase'
            ]
        self.localSpace_list = ['C_IK_head_CON','C_IK_neck1_CON','L_IK_elbow_CON','R_IK_elbow_CON','L_IK_knee_CON','R_IK_knee_CON','L_IK_foreLeg_CON','R_IK_foreLeg_CON']
        self.channel_List = ['tx','ty','tz','rx','ry','rz']
        self.delConstraints = []
        
    def defineSource(self, ns):
        self.sourceNS = ns
   
    def defineTarget(self, ns):
        self.targetNS = ns

    def getSourceHIK(self):
        '''
        source가 되는 리그의 HIK세팅 HIK 조인트들이 스킨 조인트를 따라간다.
        '''
        self.importHIK( self.sourceNS )
        self.skinToHIK(0 , self.sourceNS)
    
    def getTargetHIK(self):
        '''
        target이 되는 리그의 HIK세팅 
        '''
        self.importHIK( self.targetNS )

    def importHIK(self, nsChar):
        '''
        아래 경로의 HIK scene을 가져온다.
        리그의 스킨 조인트 위치에 대입해서 parent Constraint를 걸고 삭제한다.
        만약 네임스페이스가 없는 리그 작업 씬에서 실행된다면 씬을 임포트하고 리그를 T자형을 만들기 위해 컨드롤러들에 수치 값 또한 들어간다.
        '''
        if nsChar != None:
            cmds.file( '/stdrepo/RIG/RND/Tool/DDE/HIKJoint_Quad.mb' , i = True , type = 'mayaBinary' , ra = True , mergeNamespacesOnClash = True ,ns = nsChar , options = 'v=0')
            HIKJoint_constraint_list = []
            for i in range(len(self.HIKJoint_list)):
                if 'elbow' in self.moduleJoint_list[i]:
                    try:
                        cmds.select("%s:%s" %(nsChar, self.moduleJoint_list[i]))
                        elbow = self.moduleJoint_list[i]
                    except :
                        elbow = self.moduleJoint_list[i].replace("elbow", "edbow")
                    temp_point_con = cmds.parentConstraint(
                        "%s:%s" %(nsChar, elbow), 
                        "%s:%s" %(nsChar, self.HIKJoint_list[i]), mo=0, w=1)
                    HIKJoint_constraint_list.append(temp_point_con[0])
                else:
                    temp_point_con = cmds.parentConstraint(
                        "%s:%s" %(nsChar, self.moduleJoint_list[i]), 
                        "%s:%s" %(nsChar, self.HIKJoint_list[i]), mo=0, w=1)
                    HIKJoint_constraint_list.append(temp_point_con[0])
            cmds.delete(HIKJoint_constraint_list)
            mel.eval('$gHIKCurrentCharacter  = "%s:set_HIK"' % (nsChar))
            mel.eval('hikToggleLockDefinition();')
        else:
            cmds.file('/stdrepo/RIG/RND/Tool/DDE/HIKJoint_Quad.mb' , i = True , type = 'mayaBinary' , ra = True , mergeNamespacesOnClash = True ,ns = ":", options = 'v=0')
            HIKJoint_constraint_list = []
            for i in range(len(self.HIKJoint_list)):
                if 'elbow' in self.moduleJoint_list[i]:
                    try:
                        cmds.select("%s" %(self.moduleJoint_list[i]))
                        elbow = self.moduleJoint_list[i]
                    except :
                        elbow = self.moduleJoint_list[i].replace("elbow", "edbow")
                    temp_point_con = cmds.parentConstraint(
                        "%s" % (elbow), 
                        "%s" % (self.HIKJoint_list[i]), mo=0, w=1)
                    HIKJoint_constraint_list.append(temp_point_con[0])
                else:
                    temp_point_con = cmds.parentConstraint(
                        "%s" %(self.moduleJoint_list[i]), 
                        "%s" %(self.HIKJoint_list[i]), mo=0, w=1)
                    HIKJoint_constraint_list.append(temp_point_con[0])
            mel.eval('$gHIKCurrentCharacter  = "set_HIK"')
            # mel.eval('hikToggleLockDefinition();')
            for con in range(len(self.localSpace_list)):
                if not "elbow" in  self.localSpace_list[con]:
                    cmds.setAttr("%s.%s"% (self.localSpace_list[con],'localSpace'),1)
                else:
                    try:
                        cmds.setAttr("%s.%s"% (self.localSpace_list[con],'localSpace'),1)
                    except:
                        elbow = self.localSpace_list[con].replace("edlow", "edbow")
                        cmds.setAttr("%s.%s"% (elbow,'localSpace'),1)
                        
            cmds.setAttr("C_IK_root_CON.rotateX", -90)
            cmds.setAttr("R_wistRoll_CON.fallowRotate",1)
            cmds.setAttr("L_wistRoll_CON.fallowRotate",1)

    def skinToHIK(self, delCon = 0, nsChar = "" ):
        '''
        HIK 조인트들이 스킨조인트들에 parentConstraint 걸린다.
        '''
        for i in range(len(self.HIKJoint_list)):
            if 'elbow' in self.moduleJoint_list[i]:
                try:
                    cmds.select("%s:%s" %
                                (nsChar, self.moduleJoint_list[i]))
                    elbow = self.moduleJoint_list[i]
                except :
                    elbow = self.moduleJoint_list[i].replace("elbow", "edbow")

                self.toSkin_Const = cmds.parentConstraint(
                    "%s:%s" %
                    (nsChar, elbow), "%s:%s" %
                    (nsChar, self.HIKJoint_list[i]), mo=1, w=1)
                self.delConstraints.append(self.toSkin_Const[0])
            else:
                self.toSkin_Const = cmds.parentConstraint(
                    "%s:%s" %
                    (nsChar, self.moduleJoint_list[i]), "%s:%s" %
                    (nsChar, self.HIKJoint_list[i]), mo=1, w=1)
                self.delConstraints.append(self.toSkin_Const[0])
        if delCon == 1:
            cmds.delete(self.delConstraints)

    def HIKtoFK(self):  
        '''
        HIK 조인트와 FK 컨트롤러 연결
        두가지의 컨트롤러 타입이 존재 (사일런스도그/와일드보어)
        '''
        switch_list = [
            'R_foreLeg_switch_CON.IKFKBlend', 'L_foreLeg_switch_CON.IKFKBlend',
            'R_hindLeg_switch_CON.IKFKBlend', 'L_hindLeg_switch_CON.IKFKBlend']
        parent_list = [
            'Hips',
            'Spine',
            'Spine1',
            'Spine2',
            'Spine3',
            'Neck',
            'Head',
            'LeftShoulder',
            'RightShoulder', 
            'LeftUpLeg',
            'RightUpLeg']
        if cmds.objExists("%s:%s" % (self.targetNS,
                          switch_list[0])):
            for a in range(len(switch_list)):
                cmds.setAttr('%s:%s' %(self.targetNS,switch_list[a]),0)
            
            for i in range(len(self.FKController_list)):
                if cmds.objExists("%s:%s" %(self.targetNS,self.FKController_list[i])):
                    if self.HIK_attatch_toFK_list[i] in parent_list:
                        try:
                            self.toFK_Const = cmds.parentConstraint(
                                "%s:%s" %
                                (self.targetNS, self.HIK_attatch_toFK_list[i]), "%s:%s" %
                                (self.targetNS, self.FKController_list[i]), mo=1, w=1)
                            self.delConstraints.append(self.toFK_Const[0])
                        except:
                            self.toFK_Const = cmds.orientConstraint(
                                "%s:%s" %
                                (self.targetNS, self.HIK_attatch_toFK_list[i]), "%s:%s" %
                                (self.targetNS, self.FKController_list[i]), mo=1, w=1)
                            self.delConstraints.append(self.toFK_Const[0])
                    else:
                        self.toFK_Const = cmds.orientConstraint(
                                "%s:%s" %
                                (self.targetNS, self.HIK_attatch_toFK_list[i]), "%s:%s" %
                                (self.targetNS, self.FKController_list[i]), mo=1, w=1)
                        self.delConstraints.append(self.toFK_Const[0])
                else:
                    pass
            
            for ikCon in range(len(self.POV_set_List)):
                cons = "%s_parentConstraint1" % self.POV_set_List[ikCon]
                cmds.delete(cons)
                for ch in self.channel_List:
                    cmds.setAttr("%s:%s.%s" %(self.targetNS ,self.POV_set_List[ikCon], ch),0)
            for pvCon in range(len(self.IKPOV_list)):
                connect = cmds.listConnections("%s:%s.t"%(self.targetNS,self.IKPOV_list[pvCon][3]), s = 1,p = 1, d = 0 )
                cmds.disconnectAttr(connect[0] , "%s:%s.t"%(self.targetNS,self.IKPOV_list[pvCon][3]))
                for ch in range(3):
                    cmds.setAttr( "%s:%s.%s"%(self.targetNS,self.IKPOV_list[pvCon][3],self.channel_List[ch]),0)
            IKRoll_list = [
                'L_wistRoll_CON', 'L_ankleRoll_CON',
                'R_wistRoll_CON', 'R_ankleRoll_CON'
                ]
            for rollCon in range(len(IKRoll_list)):
                cons = "%s_aimConstraint1" % IKRoll_list[rollCon]
                cmds.delete(cons)
                for ch in self.channel_List:
                    cmds.setAttr("%s:%s.%s" %(self.targetNS ,IKRoll_list[rollCon], ch),0)

        elif cmds.objExists('%s:%s.addFK' %(self.targetNS,self.POV_set_List[0])):
            for con in range(len(self.POV_set_List)):
                cmds.setAttr('%s:%s.addFK' %(self.targetNS,self.POV_set_List[con]),1)
            
            for i in range(len(self.subFKController_list)):
                if cmds.objExists("%s:%s" %(self.targetNS,self.subFKController_list[i])):
                    try:
                        self.toFK_Const = cmds.parentConstraint(
                            "%s:%s" %
                            (self.targetNS, self.HIK_attatch_tosubFK_list[i]), "%s:%s" %
                            (self.targetNS, self.subFKController_list[i]), mo=1, w=1)
                        self.delConstraints.append(self.toFK_Const[0])
                    except:
                        self.toFK_Const = cmds.orientConstraint(
                            "%s:%s" %
                            (self.targetNS, self.HIK_attatch_tosubFK_list[i]), "%s:%s" %
                            (self.targetNS, self.subFKController_list[i]), mo=1, w=1)
                        self.delConstraints.append(self.toFK_Const[0])
                else:
                    pass
            for con in range(len(self.localSpace_list)):
                if not "elbow" in  self.localSpace_list[con]:
                    cmds.setAttr('%s:%s.localSpace' %(self.targetNS,self.localSpace_list[con]),1.0)
                else:
                    try:
                        cmds.setAttr('%s:%s.localSpace' %(self.targetNS,self.localSpace_list[con]),1.0)
                    except:
                        elbow = self.localSpace_list[con].replace("elbow", "edbow")
                        cmds.setAttr('%s:%s.localSpace' %(self.targetNS,elbow),1.0)
        else:
            cmds.error("Check FKControlers, please")

        cmds.setAttr("%s:HIKproperties1.RollExtractionMode" % self.targetNS ,1)
        

    def HIKtoIK(self):  
        '''
        HIK 조인트와 IK 컨트롤러 연결
        '''
        switch_list = [
                'R_foreLeg_switch_CON.IKFKBlend', 'L_foreLeg_switch_CON.IKFKBlend',
                'R_hindLeg_switch_CON.IKFKBlend', 'L_hindLeg_switch_CON.IKFKBlend']
        if cmds.objExists('%s:%s' %(self.targetNS,switch_list[0])):
            for a in range(len(switch_list)):
                cmds.setAttr('%s:%s' %(self.targetNS,switch_list[a]),1)
        elif cmds.objExists('%s:%s.addFK' %(self.targetNS,self.POV_set_List[0])):
            for con in range(len(self.POV_set_List)):
                cmds.setAttr('%s:%s.addFK' %(self.targetNS,self.POV_set_List[con]),0)
        for i in range(len(self.IKController_list)):
            if self.IKController_list[i] in self.POV_set_List:
                pass
            # elif 'clavicle' in self.IKController_list[i]:
            #     pass
            else:
                try:
                    self.toIK_Const = cmds.parentConstraint(
                        "%s:%s" %
                        (self.targetNS, self.HIK_attatch_toIK_list[i]), "%s:%s" %
                        (self.targetNS, self.IKController_list[i]), mo=1, w=1)
                    self.delConstraints.append(self.toIK_Const[0])
                except RuntimeError:
                    self.toIK_Const = cmds.orientConstraint(
                        "%s:%s" %
                        (self.targetNS, self.HIK_attatch_toIK_list[i]), "%s:%s" %
                        (self.targetNS, self.IKController_list[i]), mo=1, w=1)
                    self.delConstraints.append(self.toIK_Const[0])
        self.IKPOVBake()
        # self.footGround()
        toeBase_list = [
            'LeftFingerBase',
            'RightFingerBase',
            'LeftToeBase',
            'RightToeBase']
        for i in range(len(self.POV_set_List)):
            cmds.parentConstraint(
                "%s:%s" %(self.targetNS, toeBase_list[i]) , 
                "%s:%s" % (self.targetNS, self.POV_set_List[i]), mo=1, w=1)
        for con in range(len(self.localSpace_list)):
            if "elbow" in  self.localSpace_list[con]:
                if cmds.objExists('%s:%s.localSpace' %(self.targetNS,self.localSpace_list[con])):
                    elbow = self.localSpace_list[con]
                else:
                    elbow = self.localSpace_list[con].replace("elbow", "edbow")
                cmds.setAttr('%s:%s.localSpace' %(self.targetNS,elbow),1.0)
            elif "knee" in  self.localSpace_list[con]:   
                cmds.setAttr('%s:%s.localSpace' %(self.targetNS,self.localSpace_list[con]),1.0)
            else:
                cmds.setAttr('%s:%s.localSpace' %(self.targetNS,self.localSpace_list[con]),0)

        targetCon_List = cmds.ls(self.targetNS+":"+"*"+"CON")
        for con in range(len(targetCon_List)):
            for ch in range(len(self.channel_List)):
                try:
                    cmds.setAttr("%s.%s"% (targetCon_List[con],self.channel_List[ch]),0)
                except:
                    pass

        cmds.currentTime(str(cmds.playbackOptions(q=1, minTime=True)))
        
    def IKPOVBake(self):
        decome_set_List = ['upLeg', 'Foot', 'Leg', 'POV']
        pma_set_List = [
            'upLeg',
            'Foot',
            'Leg',
            'minusLeg',
            'upLegFoot',
            'plusLeg']
        
        mpd_set_List = ['upLegFoot', 'Leg']
        decomposedMatrix_list = []
        plusMinusAverage_list = []
        multiplyDivide_list = []
        # polevectorCON 4
        for i in range(len(self.POV_set_List)):
            del decomposedMatrix_list[0:len(decomposedMatrix_list)]
            del plusMinusAverage_list[0:len(plusMinusAverage_list)]
            del multiplyDivide_list[0:len(multiplyDivide_list)]
            # decomposeMatrixNode 4 (upleg, foot, leg, POVCon)
            for dcm in range(len(decome_set_List)):
                decomposed_node = cmds.createNode(
                    'decomposeMatrix', n='%s:%s_%s_DCM' %
                    (self.targetNS, self.POV_set_List[i], decome_set_List[dcm]))
                decomposedMatrix_list.append(decomposed_node)
                self.delConstraints.append(decomposed_node)
            for pma in range(len(pma_set_List)):
                plusMinusAverage_node = cmds.createNode(
                    'plusMinusAverage', n='%s:%s_%s_PMA' %
                    (self.targetNS, self.POV_set_List[i], pma_set_List[pma]))
                if pma <= 3:
                    cmds.setAttr('%s.operation' % plusMinusAverage_node, 2)
                plusMinusAverage_list.append(plusMinusAverage_node)
                self.delConstraints.append(plusMinusAverage_node)
            for mpd in range(len(mpd_set_List)):
                multiplyDivide_node = cmds.createNode(
                    'multiplyDivide', n='%s:%s_%s_MPD' %
                    (self.targetNS, self.POV_set_List[i], mpd_set_List[mpd]))
                cmds.setAttr('%s.input2X' % multiplyDivide_node, 2)
                cmds.setAttr('%s.input2Y' % multiplyDivide_node, 2)
                cmds.setAttr('%s.input2Z' % multiplyDivide_node, 2)
                if mpd <= 0:
                    cmds.setAttr('%s.operation' % multiplyDivide_node, 2)
                multiplyDivide_list.append(multiplyDivide_node)
                self.delConstraints.append(multiplyDivide_node)
            composeMatrix_node = cmds.createNode(
                'composeMatrix', n='%s:%s_CPM' %
                (self.targetNS, self.POV_set_List[i]))
            self.delConstraints.append(composeMatrix_node)
            multMatrix_node = cmds.createNode(
                'multMatrix', n='%s:%s_MMX' %
                (self.targetNS, self.POV_set_List[i]))
            self.delConstraints.append(multMatrix_node)
            for a in range(3):
                cmds.connectAttr(
                    '%s:%s.worldMatrix'%(self.targetNS, self.IKPOV_list[i][a]),
                    '%s.inputMatrix'%decomposedMatrix_list[a])
                cmds.connectAttr(
                    '%s.outputTranslate'%decomposedMatrix_list[a],
                    '%s.input3D[0]' %plusMinusAverage_list[a])
            cmds.connectAttr(
                '%s.output3D'%plusMinusAverage_list[0],
                '%s.input3D[0]'%plusMinusAverage_list[4])
            cmds.connectAttr(
                '%s.output3D'%plusMinusAverage_list[2],
                '%s.input3D[1]'%plusMinusAverage_list[4])
            cmds.connectAttr(
                '%s.output3D' %plusMinusAverage_list[4],
                '%s.input1' %multiplyDivide_list[0])
            cmds.connectAttr(
                '%s.output' %multiplyDivide_list[0],
                '%s.input3D[1]' %plusMinusAverage_list[3])
            cmds.connectAttr(
                '%s.output3D' %plusMinusAverage_list[1],
                '%s.input3D[0]' %plusMinusAverage_list[3])
            cmds.connectAttr(
                '%s.output3D' %plusMinusAverage_list[3],
                '%s.input1' %multiplyDivide_list[1])
            cmds.connectAttr(
                '%s.output' %multiplyDivide_list[1],
                '%s.input3D[0]' %plusMinusAverage_list[5])
            cmds.connectAttr(
                '%s.output' %multiplyDivide_list[0],
                '%s.input3D[1]' %plusMinusAverage_list[5])
            cmds.connectAttr(
                '%s.output3D' %plusMinusAverage_list[5],
                '%s.inputTranslate' %composeMatrix_node)
            cmds.connectAttr(
                '%s.outputMatrix' %composeMatrix_node,
                '%s.matrixIn[0]' %multMatrix_node)
            cmds.connectAttr(
                '%s.matrixSum' %multMatrix_node,
                '%s.inputMatrix' %decomposedMatrix_list[3])
            if cmds.objExists('%s:%s' %(self.targetNS,self.IKPOV_list[i][3])):
                elbow =self.IKPOV_list[i][3]
            else:
                elbow =self.IKPOV_list[i][3].replace("elbow", "edbow")
            cmds.connectAttr(
                '%s.outputTranslate' %decomposedMatrix_list[3],
                '%s:%s.translate' %(self.targetNS,elbow))
            cmds.connectAttr(
                '%s:%s.parentInverseMatrix' %
                (self.targetNS, elbow), '%s.matrixIn[1]' %multMatrix_node) 
        cmds.group(n="%s:transfer_loc_GRP" %(self.targetNS),em=1)
        
    def set_IKrollCon(self):    

        IKRoll_list = [
            'L_wistRoll_CON',
            'L_ankleRoll_CON',
            'R_wistRoll_CON',
            'R_ankleRoll_CON']
        toeBase_list = [
            'LeftFingerBase',
            'LeftToeBase',
            'RightFingerBase',
            'RightToeBase']
        for con in range(len(IKRoll_list)):
            LocName = IKRoll_list[con].replace("CON", "aim_LOC")
            LocNul = IKRoll_list[con].replace("CON", "aim_NUL")
            cmds.spaceLocator(n=LocName)
            cmds.group(n = LocNul, em =1)
            cmds.parent(LocName, LocNul)
            decomposed_node = cmds.createNode(
                    'decomposeMatrix', n='%s:%s_DCM' %
                    (self.targetNS, toeBase_list[con]))
            self.delConstraints.append(decomposed_node)
            plusMinusAverage_node = cmds.createNode(
                    'plusMinusAverage', n='%s:%s_PMA' %
                    (self.targetNS, toeBase_list[con]))
            cmds.setAttr('%s.operation' % plusMinusAverage_node, 1)
            cmds.setAttr('%s.input3D[1].input3Dz' % plusMinusAverage_node, 5)
            self.delConstraints.append(plusMinusAverage_node)
            composeMatrix_node = cmds.createNode(
                'composeMatrix', n='%s:%s_CPM' %
                (self.targetNS, toeBase_list[con]))
            self.delConstraints.append(composeMatrix_node)
            multMatrix_node = cmds.createNode(
                'multMatrix', n='%s:%s_MMX' %
                (self.targetNS, toeBase_list[con]))
            self.delConstraints.append(multMatrix_node)
            decomposedToanul_node = cmds.createNode(
                    'decomposeMatrix', n='%s:%s_DCM' %
                    (self.targetNS, LocNul))
            self.delConstraints.append(decomposedToanul_node)

            cmds.connectAttr(
                '%s:%s.worldMatrix'%(self.targetNS, toeBase_list[con]),
                '%s.inputMatrix'%decomposed_node)
            cmds.connectAttr(
                '%s.outputTranslate'%decomposed_node,
                '%s.input3D[0]' %plusMinusAverage_node)
            cmds.connectAttr(
                '%s.output3D'%plusMinusAverage_node,
                '%s.inputTranslate' %composeMatrix_node)
            cmds.connectAttr(
                '%s.outputRotate'%decomposed_node,
                '%s.inputRotate' %composeMatrix_node)
            cmds.connectAttr(
                '%s.outputMatrix' %composeMatrix_node,
                '%s.matrixIn[0]' %multMatrix_node)
            cmds.connectAttr(
                '%s.parentInverseMatrix[0]' %LocNul,
                '%s.matrixIn[1]' %multMatrix_node)
            cmds.connectAttr(
                '%s.matrixSum' %multMatrix_node,
                '%s.inputMatrix' %decomposedToanul_node)
            cmds.connectAttr(
                '%s.outputTranslate' %decomposedToanul_node,
                '%s.translate' %LocNul)
            cmds.connectAttr(
                '%s.outputRotate' %decomposedToanul_node,
                '%s.rotate' %LocNul)

            toeJnt = cmds.pickWalk( "%s:%s" % (self.targetNS, toeBase_list[con]), d = 'up')[0]
            cmds.select(
                toeJnt, '%s:%s' %
                (self.targetNS, IKRoll_list[con]), r=1)
            mel.eval(
                'doCreateAimConstraintArgList 1 {"1", "0", "0", "0", "0", "1", "0", "0", "0", "1", "0", "1", "0", "1", "object", "' +
                str(LocName) +
                '", "0", "0", "0", "","1"};')
            cmds.parent(LocNul,"%s:transfer_loc_GRP" %(self.targetNS))
            cmds.hide("%s:transfer_loc_GRP" %(self.targetNS))
        
    
    def set_HIKproperties(self):
        propertiesNode = "%s:HIKproperties1" % self.targetNS
        cmds.setAttr("%s.HandFloorContact" % propertiesNode , 1)
        cmds.setAttr("%s.HandContactType" % propertiesNode , 3)
        cmds.setAttr("%s.LeftElbowKillPitch" % propertiesNode , 1)
        cmds.setAttr("%s.RightElbowKillPitch" % propertiesNode , 1)
        Loc = cmds.spaceLocator(n = "%s:retaegertLegLength_LOC" % self.targetNS)[0]
        oriLoc = "%s:originLegLength_LOC.ty" % self.targetNS
        cmds.matchTransform(Loc , "%s:Spine6" % self.targetNS)
        self.delConstraints.append(Loc)
        self.delConstraints.append(oriLoc)
        cmds.parent(Loc,"%s:transfer_loc_GRP" %(self.targetNS),oriLoc)  
        retarget_height = cmds.getAttr("%s:retaegertLegLength_LOC.ty" % self.targetNS)
        origin_height = cmds.getAttr("%s:originLegLength_LOC.ty" % self.targetNS)
        if origin_height < retarget_height :
            cmds.setAttr("%s.ReachActorChest" % propertiesNode , 0.5)
        else: 
            pass
            
    def bakeHipMatrix(self):
        front_loc = cmds.spaceLocator(
            n="%s:%s" %
            (self.targetNS, "frontHip_loc"))[0]
        toLocCons = cmds.parentConstraint(
            "%s:%s" %
            (self.targetNS, "frontLeg_Hips"), front_loc, mo=0, w=1)
        startFr = cmds.playbackOptions(q=1, minTime=True)
        endFr = cmds.playbackOptions(q=1, maxTime=True)
        # gmainPane = mel.eval('global string $gMainPane; $temp =$gMainPane;')
        # cmds.paneLayout('viewPanes', e=1, manage=0)
        # cmds.refresh(su=1)
        cmds.select(front_loc, r=1)
        cmds.bakeResults(t=(int(startFr), int(endFr)), simulation=True, sb=1)
        # cmds.refresh(su=0)

        # cmds.paneLayout('viewPanes', e=1, manage=1)
        return front_loc

    def frontlocToHIKloc(self):
        frontLoc = self.bakeHipMatrix()
        minusMatrixloc = cmds.spaceLocator(
            n="%s:%s" %
            (self.targetNS, "minusMatrix_loc"))[0]
        spine_decom = cmds.createNode(
            'decomposeMatrix',n='%s:Spine3_DCM' %(self.targetNS))
        frontLoc_decom = cmds.createNode(
            'decomposeMatrix',n='%s:frontHip_loc_DCM' %(self.targetNS))
        forntLeg_PNA = cmds.createNode(
            'plusMinusAverage',n='%s:frontHip_loc_PMA' %(self.targetNS))
        cmds.setAttr("%s.operation" % (forntLeg_PNA), 2)
        forntLeg_MPD = cmds.createNode(
            'multiplyDivide',n='%s:frontHip_loc_MPD' %(self.targetNS))
        cmds.setAttr("%s.operation" % (forntLeg_MPD), 1)
        cmds.setAttr("%s.input2X" % (forntLeg_MPD), 1)
        cmds.setAttr("%s.input2Y" % (forntLeg_MPD), -1)
        cmds.setAttr("%s.input2Z" % (forntLeg_MPD), -1)
        cmds.connectAttr(
            '%s:Spine3.worldMatrix[0]' %(self.targetNS),'%s.inputMatrix' %(spine_decom))
        cmds.connectAttr(
            '%s.worldMatrix[0]' %(frontLoc),'%s.inputMatrix' %(frontLoc_decom))
        cmds.connectAttr(
            '%s.outputTranslate' %(frontLoc_decom),'%s.input3D[0]' %(forntLeg_PNA))
        cmds.connectAttr(
            '%s.outputTranslate' %(spine_decom),'%s.input3D[1]' %(forntLeg_PNA))
        cmds.connectAttr(
            '%s.output3D' %(forntLeg_PNA),'%s.input1' %(forntLeg_MPD))
        cmds.connectAttr(
            '%s.output' %(forntLeg_MPD),'%s.translate' %(minusMatrixloc))
        cmds.connectAttr(
            '%s.translate' %(minusMatrixloc),'%s:frontLeg_HIKJoint_LOC.translate' %(self.targetNS))
        if cmds.objExists("%s:transfer_loc_GRP" %(self.targetNS)):
            cmds.parent(minusMatrixloc,frontLoc,"%s:transfer_loc_GRP" %(self.targetNS))
        else:
            cmds.group(n="%s:transfer_loc_GRP" %(self.targetNS),em=1)
            cmds.parent(minusMatrixloc,frontLoc,"%s:transfer_loc_GRP" %(self.targetNS))

    def footGround(self):
        toeBase_list = [
            'LeftFingerBase',
            'RightFingerBase',
            'LeftToeBase',
            'RightToeBase']
        cmds.group(n="%s:footBridge_loc_GRP" %(self.targetNS),em=1)
        cmds.parent("%s:footBridge_loc_GRP" %(self.targetNS), "%s:transform_GRP" %(self.targetNS))
        cmds.scale(1,1,1, "%s:footBridge_loc_GRP" %(self.targetNS))
        for i in range(len(self.POV_set_List)):
            locname = self.POV_set_List[i].replace("CON","toe_loc") 
            toeLoc = cmds.spaceLocator(
                n = "%s:%s" %
                (self.targetNS, locname))[0]
            toeNul = cmds.group(toeLoc, name=locname.replace("loc", "nul"))
            #지면 인식 노드 생성 추후 추가 예정
            # bridgeLoc = cmds.spaceLocator(
                # n = locname.replace("toe_loc", "bridge_loc"))[0]
            # cmds.parent(bridgeLoc,"%s:footBridge_loc_GRP" %(self.targetNS))
            cmds.parent(toeNul,"%s:footBridge_loc_GRP" %(self.targetNS))
            cmds.scale(1,1,1, toeNul)
            # cmds.scale(1,1,1,bridgeLoc)
            Const = cmds.parentConstraint(
                    "%s:%s" %(self.targetNS, toeBase_list[i]), 
                    toeNul, mo=0, w=1)
            cmds.delete(Const)
            cmds.parentConstraint(
                "%s:%s" %(self.targetNS, toeBase_list[i]), 
                toeLoc, mo=1, w=1)
            # condNode = cmds.createNode(
            #     "condition",n = locname.replace('loc','CND'))
            # cmds.setAttr("%s.operation"%condNode, 4)
            # cmds.connectAttr(
            #     "%s.translateY"%toeLoc,
            #     "%s.firstTerm"%condNode)
            # cmds.connectAttr(
            #     "%s.translateY"%toeLoc,
            #     "%s.colorIfFalseR"%condNode)
            # cmds.connectAttr(
            #     "%s.outColorR"%condNode,
            #     "%s.translateY"%bridgeLoc)  
            cmds.parentConstraint(
                 "%s:%s" %(self.targetNS, toeBase_list[i]) , 
                 "%s:%s" % (self.targetNS, self.POV_set_List[i]), mo=1, w=1)
            
    def selectControler(self,type = ""):
        if type == "IK":
            IKadd_list = [
                'L_wistRoll_CON',
                'L_ankleRoll_CON',
                'R_wistRoll_CON',
                'R_ankleRoll_CON',
                'L_IK_elbow_CON',
                'R_IK_elbow_CON',
                'L_IK_knee_CON',
                'R_IK_knee_CON']
            targetIK_list = []
            for i in range(len(self.IKController_list)):
                target_con = "%s:%s"%(self.targetNS,self.IKController_list[i])
                targetIK_list.append(target_con)
            for roll in range(8):
                target_con = "%s:%s"%(self.targetNS,IKadd_list[roll])
                if not cmds.objExists(target_con):
                    elbowcon = target_con.replace("elbow","edbow")
                    targetIK_list.append(elbowcon)
                else:
                    targetIK_list.append(target_con)
                
            cmds.select(targetIK_list,r=1)
        if type == "FK":
            targetFK_list = []
            if cmds.objExists("%s:%s" % (self.targetNS,
                          self.FKController_list[8])):
                for i in range(len(self.FKController_list)):
                    target_con = "%s:%s"%(self.targetNS,self.FKController_list[i])
                    targetFK_list.append(target_con)
            if cmds.objExists("%s:%s" % (self.targetNS,
                          self.subFKController_list[8])):
                for i in range(len(self.subFKController_list)):
                    target_con = "%s:%s"%(self.targetNS,self.subFKController_list[i])
                    targetFK_list.append(target_con)
            cmds.select(targetFK_list,r=1)
    
    def deleteAll(self):
        print (self.delConstraints)
        try:
            cmds.delete(
                self.delConstraints)
            cmds.delete(
                "%s:originLegLength_LOC" % self.targetNS)
        except: pass
        try:
            cmds.delete(
            "%s:transfer_loc_GRP" %(self.targetNS))
        except: pass
        try:
            cmds.delete("%s:footBridge_loc_GRP" %(self.targetNS))
        except: pass
        self.delConstraints = []

    def mirrorKey(self):
        mirrorCon_List = [
            'IK_foreLeg_CON',
            'IK_hindLeg_CON',
            'IK_clavicle_world_CON',
            'wistRoll_CON',
            'IK_clavicle_CON'
            ]
        reverseList_A = ['translateX', 'rotateY','rotateZ']
        reverseList_B = ['translateX','translateY','translateZ']
        for i in range(len(mirrorCon_List)):
            leftCon = "L_%s"% mirrorCon_List[i]
            rightCon = "R_%s"% mirrorCon_List[i]
            if cmds.objExists(rightCon):
                attrList = cmds.listAttr(leftCon, keyable = 1,unlocked = 1)
                for attr in range(len(attrList)):
                    value = cmds.getAttr("%s.%s"%(leftCon,attrList[attr]))
                    if i == 4:
                        if attrList[attr] in reverseList_B:
                            value *= -1
                        cmds.setAttr("%s.%s"%(rightCon,attrList[attr]), value)
                    else:
                        if attrList[attr] in reverseList_A:
                            value *= -1
                        cmds.setAttr("%s.%s"%(rightCon,attrList[attr]), value)
    
    def muteWorldCon(self):
        
        worldCon_list = ["move_CON", "direction_CON", "place_CON" ]
        for con in range(len(worldCon_list)):
            for ch in range(len(self.channel_List)):
                cmds.mute("%s:%s.%s" % (self.sourceNS, worldCon_list[con], self.channel_List[ch]))
    

                
