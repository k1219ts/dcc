# encoding=utf-8
#!/usr/bin/env python

# Import Maya Modules
import maya.cmds as cmds
import maya.mel as mel

class BIPADRETARGRT:
    def __init__(self):
        self.moduleJoint_list = ['C_Skin_hip_JNT','C_Skin_spine1_JNT','C_Skin_spine2_JNT','C_Skin_spine3_JNT','C_Skin_chest_JNT','C_Skin_neck_JNT','C_Skin_head_JNT','L_Skin_shoulder_JNT','L_Skin_upArm_JNT','L_Skin_foreArm_JNT','L_Skin_hand_JNT','L_Skin_middle1_JNT','R_Skin_shoulder_JNT','R_Skin_upArm_JNT','R_Skin_foreArm_JNT','R_Skin_hand_JNT','R_Skin_middle1_JNT','L_Skin_leg_JNT','L_Skin_lowLeg_JNT','L_Skin_foot_JNT','L_Skin_ball_JNT','R_Skin_leg_JNT','R_Skin_lowLeg_JNT','R_Skin_foot_JNT','R_Skin_ball_JNT']
        self.HIKJoint_list = ['Hips','Spine','Spine1','Spine2','Spine3','Neck','Head','LeftShoulder','LeftArm','LeftForeArm','LeftHand','LeftFingerBase','RightShoulder','RightArm','RightForeArm','RightHand','RightFingerBase','LeftUpLeg','LeftLeg','LeftFoot','LeftToeBase','RightUpLeg','RightLeg','RightFoot','RightToeBase']
        self.FKController_list = ['root_CON','C_IK_lowBody_CON','C_IK_upBodyRot1_CON','C_IK_upBodyRot2_CON','C_IK_upBody_CON','C_IK_neck_CON','C_IK_head_CON','L_FK_shoulder_CON','L_FK_upArm_CON','L_FK_foreArm_CON','L_FK_hand_CON','R_FK_shoulder_CON','R_FK_upArm_CON','R_FK_foreArm_CON','R_FK_hand_CON','L_FK_leg_CON','L_FK_lowLeg_CON','L_FK_foot_CON','R_FK_leg_CON','R_FK_lowLeg_CON','R_FK_foot_CON']
        self.HIK_attatch_toFK_list = ['Hips','Spine','Spine1','Spine2','Spine3','Neck','Head','LeftShoulder','LeftArm','LeftForeArm','LeftHand','RightShoulder','RightArm','RightForeArm','RightHand','LeftUpLeg','LeftLeg','LeftFoot','RightUpLeg','RightLeg','RightFoot']
        self.IKController_list = ['root_CON','C_IK_lowBody_CON','C_IK_upBodyRot1_CON','C_IK_upBodyRot2_CON','C_IK_upBody_CON','C_IK_neck_CON','C_IK_head_CON','L_FK_shoulder_CON','L_IK_hand_CON','R_FK_shoulder_CON','R_IK_hand_CON','L_IK_foot_CON','R_IK_foot_CON']
        self.HIK_attatch_toIK_list = ['Hips','Spine','Spine1','Spine2','Spine3','Neck','Head','LeftShoulder','LeftHand','RightShoulder','RightHand','LeftFoot','RightFoot']
        self.IKPOV_list = [['LeftUpLeg','LeftLeg','LeftFoot','L_IK_footVec_CON'],['RightUpLeg','RightLeg','RightFoot','R_IK_footVec_CON'],['LeftArm','LeftForeArm','LeftHand','L_IK_handVec_CON'],['RightArm','RightForeArm','RightHand','R_IK_handVec_CON']]
        self.delConstraints = []
        self.channel_List = ['tx','ty','tz','rx','ry','rz']
        
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
        cmds.file('/stdrepo/ANI/Asset/HumanIK_for_AnimBrowser/HIKJoint.ma' , i = True , type = 'mayaAscii' , ra = True , mergeNamespacesOnClash = True ,ns = nsChar , options = 'v=0')
        HIKJoint_constraint_list = []
        for i in range(len(self.HIKJoint_list)):
            if 'edbow' in self.moduleJoint_list[i]:
                try:
                    cmds.select("%s:%s" %(nsChar, self.moduleJoint_list[i]))
                    elbow = self.moduleJoint_list[i]
                except :
                    elbow = self.moduleJoint_list[i].replace("edbow", "elbow")
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

    def skinToHIK(self, delCon = 0, nsChar = ""):
        for i in range(len(self.HIKJoint_list)):
            self.toSkin_Const = cmds.parentConstraint( "%s:%s" %(nsChar,self.moduleJoint_list[i]) ,"%s:%s" %(nsChar,self.HIKJoint_list[i]), mo = 1 , w = 1 )
            self.delConstraints.append(self.toSkin_Const[0])  
        if delCon == 1:
            cmds.delete(self.delConstraints)
    
    def HIKtoFK(self):  
        for i in range(len(self.FKController_list)):
            self.toFK_Const = cmds.parentConstraint( "%s:%s" %(self.targetNS,self.HIK_attatch_toFK_list[i]) ,"%s:%s" %(self.targetNS,self.FKController_list[i]), mo = 1 , w = 1 )
            self.delConstraints.append(self.toFK_Const[0])
            cmds.setAttr('%s:R_armBlend_CON.FKIKBlend' %self.targetNS , 0)
            cmds.setAttr('%s:L_armBlend_CON.FKIKBlend' %self.targetNS , 0)
            cmds.setAttr('%s:R_legBlend_CON.FKIKBlend' %self.targetNS , 0)
            cmds.setAttr('%s:L_legBlend_CON.FKIKBlend' %self.targetNS , 0)  

    def HIKtoIK(self): 
        for i in range(len(self.IKController_list)):
            print(self.IKController_list[i])
            self.toIK_Const = cmds.parentConstraint( "%s:%s" %(self.targetNS,self.HIK_attatch_toIK_list[i]) ,"%s:%s" %(self.targetNS,self.IKController_list[i]), mo = 1 , w = 1 )
            self.delConstraints.append(self.toIK_Const[0])
            cmds.setAttr('%s:R_armBlend_CON.FKIKBlend' %self.targetNS , 1)
            cmds.setAttr('%s:L_armBlend_CON.FKIKBlend' %self.targetNS , 1)
            cmds.setAttr('%s:R_legBlend_CON.FKIKBlend' %self.targetNS , 1)
            cmds.setAttr('%s:L_legBlend_CON.FKIKBlend' %self.targetNS , 1)  
    
    def IKPOVBake(self):
 
        POV_set_List = ['LeftLeg' , 'RightLeg' , 'LeftArm' , 'RightArm' ]
        decomposedMatrix_list = []
        plusMinusAverage_list = []
        multiplyDivide_list = []

        # joint part : 4
        for i in range(len(POV_set_List)):
            del decomposedMatrix_list[0:len(decomposedMatrix_list)]
            del plusMinusAverage_list[0:len(plusMinusAverage_list)]
            del multiplyDivide_list[0:len(multiplyDivide_list)]
            # decomposed_Node : 4
            for x in range(len(self.IKPOV_list)):
                decomposed_node = cmds.createNode('decomposeMatrix',n ='%s:%s_%s_DCM' %(self.targetNS,POV_set_List[i],x+1)  ) 
                decomposedMatrix_list.append(decomposed_node)
                self.delConstraints.append(decomposed_node)
            # plusMinusAverage_Node : 6
            for y in range(len(self.IKPOV_list)+2):
                plusMinusAverage_node = cmds.createNode('plusMinusAverage',n ='%s:%s_%s_PMA' %(self.targetNS,POV_set_List[i],y+1)  ) 
                if y <= 3:
                    cmds.setAttr('%s.operation' %plusMinusAverage_node , 2 )
                plusMinusAverage_list.append(plusMinusAverage_node)
                self.delConstraints.append(plusMinusAverage_node)
            # multiplyDivide_Node : 2
            for z in range(len(self.IKPOV_list)-2):
                multiplyDivide_node = cmds.createNode('multiplyDivide',n ='%s:%s_%s_MPD' %(self.targetNS,POV_set_List[i],z+1)  ) 
                cmds.setAttr('%s.input2X' %multiplyDivide_node , 2)
                cmds.setAttr('%s.input2Y' %multiplyDivide_node , 2)
                cmds.setAttr('%s.input2Z' %multiplyDivide_node , 2)
                if z <= 0:
                    cmds.setAttr('%s.operation' %multiplyDivide_node , 2 )
                multiplyDivide_list.append(multiplyDivide_node)
                self.delConstraints.append(multiplyDivide_node)
            # composeMatrix : 1
            composeMatrix_node = cmds.createNode('composeMatrix',n ='%s:%s_%s_CPM' %(self.targetNS,POV_set_List[i],i+1)) 
            self.delConstraints.append(composeMatrix_node)
            # multiMatrix : 1
            multMatrix_node = cmds.createNode('multMatrix',n ='%s:%s_%s_MMX' %(self.targetNS,POV_set_List[i],i+1)) 
            self.delConstraints.append(multMatrix_node)

            # connectAttr 
            for a in range(3):
                cmds.connectAttr('%s:%s.worldMatrix' %(self.targetNS,self.IKPOV_list[i][a]), '%s.inputMatrix' %decomposedMatrix_list[a] )
                cmds.connectAttr('%s.outputTranslate' %decomposedMatrix_list[a] , '%s.input3D[0]' %plusMinusAverage_list[a] )
            cmds.connectAttr('%s.output3D' %plusMinusAverage_list[0] , '%s.input3D[0]' %plusMinusAverage_list[4] )
            cmds.connectAttr('%s.output3D' %plusMinusAverage_list[2] , '%s.input3D[1]' %plusMinusAverage_list[4] )
            cmds.connectAttr('%s.output3D' %plusMinusAverage_list[4] , '%s.input1' %multiplyDivide_list[0] )
            cmds.connectAttr('%s.output' %multiplyDivide_list[0]  , '%s.input3D[1]' %plusMinusAverage_list[3] )
            cmds.connectAttr('%s.output3D' %plusMinusAverage_list[1]  , '%s.input3D[0]' %plusMinusAverage_list[3] )
            cmds.connectAttr('%s.output3D' %plusMinusAverage_list[3] , '%s.input1' %multiplyDivide_list[1] )
            cmds.connectAttr('%s.output' %multiplyDivide_list[1] , '%s.input3D[0]' %plusMinusAverage_list[5] )
            cmds.connectAttr('%s.output' %multiplyDivide_list[0]  , '%s.input3D[1]' %plusMinusAverage_list[5] )
            cmds.connectAttr('%s.output3D' %plusMinusAverage_list[5] , '%s.inputTranslate' %composeMatrix_node)
            cmds.connectAttr('%s.outputMatrix' %composeMatrix_node ,'%s.matrixIn[0]' %multMatrix_node )
            cmds.connectAttr('%s.matrixSum' %multMatrix_node ,'%s.inputMatrix' %decomposedMatrix_list[3] )
            cmds.connectAttr('%s.outputTranslate' %decomposedMatrix_list[3] , '%s:%s.translate' %(self.targetNS,self.IKPOV_list[i][3]))
            cmds.connectAttr('%s:%s.parentInverseMatrix' %(self.targetNS,self.IKPOV_list[i][3]) , '%s.matrixIn[1]' %multMatrix_node )       ## IK poleVector Bake

    def deleteAll(self):
        try:
            cmds.delete(
                self.delConstraints)
        except: pass
        try:
            cmds.delete(
            "%s:transfer_loc_GRP" %(self.targetNS))
        except: pass
        try:
            cmds.delete("%s:footBridge_loc_GRP" %(self.targetNS))
        except: pass
        self.delConstraints = []  

    def selectControler(self,type = ""):
        if type == "IK":
            IKadd_list = [
                'L_IK_handVec_CON',
                'R_IK_handVec_CON',
                'L_IK_footVec_CON',
                'R_IK_footVec_CON']
            targetIK_list = []
            for i in range(len(self.IKController_list)):
                target_con = "%s:%s"%(self.targetNS,self.IKController_list[i])
                targetIK_list.append(target_con)
            for roll in range(len(IKadd_list)):
                target_con = "%s:%s"%(self.targetNS,IKadd_list[roll])
                targetIK_list.append(target_con)
            cmds.select(targetIK_list,r=1)
        if type == "FK":
            targetFK_list = []
            for i in range(len(self.FKController_list)):
                target_con = "%s:%s"%(self.targetNS,self.FKController_list[i])
                targetFK_list.append(target_con)
            cmds.select(targetFK_list,r=1)
    
    def muteWorldCon(self):
        worldCon_list = ["move_CON", "direction_CON", "place_CON" ]
        for con in range(len(worldCon_list)):
            for ch in range(len(self.channel_List)):
                cmds.mute("%s:%s.%s" % (self.sourceNS, worldCon_list[con], self.channel_List[ch]))
    