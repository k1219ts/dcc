# encoding:utf-8
# !/usr/bin/env python

#import json
import maya.cmds as cmds

'''
orientConvertData = [('C_IK_root_CON', 'switch_root', 'tx', 'tz', 1),
                     ('C_IK_root_CON', 'switch_root', 'ty', 'ty', 1),
                     ('C_IK_root_CON', 'switch_root', 'tz', 'tx', -1),
                     ('C_IK_root_CON', 'switch_root', 'rx', 'rz', 1),
                     ('C_IK_root_CON', 'switch_root', 'ry', 'ry', 1),
                     ('C_IK_root_CON', 'switch_root', 'rz', 'rx', -1)
                     ]



neckHead_measure = []
L_foreLeg_measure = []
R_foreLeg_measure = []
L_rearLeg_measure = []
R_rearLeg_measure = []
root_position = []
body_measure = []
poleVector = []

measure_lists = [neckHead_measure, L_foreLeg_measure, R_foreLeg_measure,L_rearLeg_measure,R_rearLeg_measure]
sourceRootPosition = [root_position]
bodyData = [body_measure]
poleData = [poleVector]




class Retarget:

    ###변수들###
    '''
    allControler = ['tail4_CON', 'tail3_CON', 'tail2_CON', 'tail1_CON', 'tail1_move_CON', 'tail2_move_CON', 'tail3_move_CON', 'tail4_move_CON', 'tail5_move_CON',

                    'R_IK_scapula_CON',  'R_IK_foreLeg_CON', 'R_IK_clavicle_CON', 'R_IK_clavicle_world_CON','R_IK_edbow_CON', 'R_wistRoll_CON',

                    'L_IK_scapula_CON', 'L_IK_foreLeg_CON', 'L_IK_clavicle_CON', 'L_IK_clavicle_world_CON','L_IK_edbow_CON', 'L_wistRoll_CON',

                    'R_IK_hindLeg_CON', 'R_ankleRoll_CON', 'R_IK_knee_CON', 'R_IK_hip_CON',

                    'L_IK_hindLeg_CON', 'L_ankleRoll_CON', 'L_IK_knee_CON', 'L_IK_hip_CON',

                    'C_IK_hip_CON',

                    'C_IK_neck1_CON', 'C_IK_neckMiddle_CON', 'neck_switch_CON', 'C_IK_head_CON','C_IK_headSub_CON','C_IK_neck1Sub_CON','C_IK_neckBack_CON','C_IK_neckFront_CON',

                    'C_IK_lowBody_CON', 'spine_switch_CON', 'C_IK_root_CON', 'C_IK_spineMiddle_CON', 'C_IK_upBody_CON','C_IK_upBodySub_CON','C_IK_spineFront_CON','C_IK_spineBack_CON','C_IK_lowBodySub_CON',

                    'move_CON', 'direction_CON', 'place_CON' ]
    '''

    allControler = ['tail4_CON', 'tail3_CON', 'tail2_CON', 'tail1_CON', 'tail1_move_CON', 'tail2_move_CON', 'tail3_move_CON', 'tail4_move_CON', 'tail5_move_CON',

                    'R_IK_scapula_CON',  'R_IK_foreLeg_CON', 'R_IK_clavicle_CON', 'R_IK_clavicle_world_CON','R_IK_edbow_CON', 'R_wistRoll_CON',

                    'L_IK_scapula_CON', 'L_IK_foreLeg_CON', 'L_IK_clavicle_CON', 'L_IK_clavicle_world_CON','L_IK_edbow_CON', 'L_wistRoll_CON',

                    'R_IK_hindLeg_CON', 'R_ankleRoll_CON', 'R_IK_knee_CON', 'R_IK_hip_CON',

                    'L_IK_hindLeg_CON', 'L_ankleRoll_CON', 'L_IK_knee_CON', 'L_IK_hip_CON',

                    'C_IK_hip_CON',

                    'C_IK_neck1_CON', 'C_IK_neckMiddle_CON', 'neck_switch_CON', 'C_IK_head_CON','C_IK_headSub_CON','C_IK_neck1Sub_CON',

                    'C_IK_lowBody_CON', 'spine_switch_CON', 'C_IK_root_CON', 'C_IK_spineMiddle_CON', 'C_IK_upBody_CON','C_IK_upBodySub_CON','C_IK_lowBodySub_CON',

                    'move_CON', 'direction_CON', 'place_CON' ]





    # <4족> 디폴트 포즈를 만들기 위해 셋업해야하는 컨트롤러 리스트
    initPoseCtrl_list = ['L_IK_foreLeg_CON', 'R_IK_foreLeg_CON', 'L_IK_hindLeg_CON', 'R_IK_hindLeg_CON','L_wistRoll_CON','R_wistRoll_CON','L_ankleRoll_CON','R_ankleRoll_CON',
                         'L_IK_knee_CON', 'R_IK_knee_CON', 'L_IK_edbow_CON', 'R_IK_edbow_CON',
                         'C_IK_upBody_CON', 'C_IK_spineMiddle_CON','C_IK_lowBody_CON','C_IK_upBodySub_CON','C_IK_lowBodySub_CON',
                         'C_IK_root_CON','C_IK_hip_CON','L_IK_hip_CON','R_IK_hip_CON',
                         'L_IK_clavicle_world_CON','R_IK_clavicle_world_CON',
                         'C_IK_head_CON', 'C_IK_headSub_CON','C_IK_neck1_CON', 'C_IK_neckMiddle_CON','C_IK_neck1Sub_CON',
                         'C_IK_root_CON', 'move_CON', 'direction_CON', 'place_CON'
                         ]

    IK_controler = ['C_IK_head_CON', 'L_IK_foreLeg_CON', 'R_IK_foreLeg_CON','L_IK_hindLeg_CON', 'R_IK_hindLeg_CON' ]

    neckHead_Jnt = ['C_IK_neck1_JNT','C_IK_neck2_JNT','C_IK_neck3_JNT','C_IK_neck4_JNT','C_IK_neck5_JNT','C_IK_neck6_JNT', 'C_IK_head_JNT']
    L_foreLeg_Jnt = ['L_Skin_shoulder_JNT','L_Skin_edbow_JNT', 'L_Skin_wrist_JNT', 'L_Skin_foot1_JNT']
    R_foreLeg_Jnt = ['R_Skin_shoulder_JNT','R_Skin_edbow_JNT', 'R_Skin_wrist_JNT', 'R_Skin_foot1_JNT']
    L_rearLeg_Jnt = ['L_Skin_hip_JNT','L_Skin_knee_JNT','L_Skin_ankle_JNT','L_Skin_ball_JNT']
    R_rearLeg_Jnt = ['R_Skin_hip_JNT','R_Skin_knee_JNT','R_Skin_ankle_JNT','R_Skin_ball_JNT']

    jnt_lists = [neckHead_Jnt, L_foreLeg_Jnt, R_foreLeg_Jnt, L_rearLeg_Jnt, R_rearLeg_Jnt]

    # 폴벡터 컨트롤러 항목
    polevector_controler = ['L_IK_edbow_CON', 'R_I K_edbow_CON', 'L_IK_knee_CON', 'R_IK_knee_CON']
    polevector_joint = ['L_IK_edbow_JNT', 'R_IK_edbow_JNT', 'L_IK_knee_JNT', 'R_IK_knee_JNT']

    forPoleJnt_lists = [L_foreLeg_Jnt, R_foreLeg_Jnt, L_rearLeg_Jnt, R_rearLeg_Jnt]

    neckHead_Ctrls = ['C_IK_neck1_CON', 'C_IK_neckMiddle_CON', 'C_IK_neck1Sub_CON','C_IK_neckBack_CON','C_IK_neckFront_CON','C_IK_head_CON','C_IK_headSub_CON']
    L_foreLeg_Ctrls = ['L_IK_foreLeg_CON','L_wistRoll_CON','L_IK_edbow_CON','L_IK_clavicle_world_CON']
    R_foreLeg_Ctrls = ['R_IK_foreLeg_CON','R_wistRoll_CON','R_IK_edbow_CON','R_IK_clavicle_world_CON']
    L_rearLeg_Ctrls = ['L_IK_hindLeg_CON', 'L_ankleRoll_CON','L_IK_knee_CON','L_IK_hip_CON']
    R_rearLeg_Ctrls = ['R_IK_hindLeg_CON', 'R_ankleRoll_CON','R_IK_knee_CON','R_IK_hip_CON']

    controler_lists = [neckHead_Ctrls, L_foreLeg_Ctrls, R_foreLeg_Ctrls, L_rearLeg_Ctrls, R_rearLeg_Ctrls]


    # 루트 조인트/컨트롤러 항목
    root_Jnt = ['C_IK_root_CON']
    root_Ctrls = ['C_IK_root_CON', 'move_CON', 'direction_CON', 'place_CON']

    body_Jnt = ['C_IK_spine1_JNT', 'C_IK_spine2_JNT', 'C_IK_spine3_JNT', 'C_IK_spine4_JNT', 'C_IK_spine5_JNT', 'C_IK_spine6_JNT', 'C_IK_chest_JNT']

    upBodyCtrls = ['C_IK_upBody_CON', 'C_IK_upBodySub_CON']
    middleBodyCtrls = ['C_IK_spineMiddle_CON']
    lowBodyCtrls = ['C_IK_lowBody_CON','C_IK_lowBodySub_CON']


    # Attribute list for initial pose(4-leg)
    initPoseAtt_list = {
        # transform attributes
        'tx': 0, 'ty': 0, 'tz': 0, 'rx': 0, 'ry': 0, 'rz': 0, 'sx': 1, 'sy': 1, 'sz': 1,

        # leg etc attributes
        'addFK': 0, 'length1': 1, 'length2': 1, 'front': 1, 'fallowRotate': 1,

        # thumb attributes
        'spread': 10, 'pinkyBend': 0, 'ringBend': 0, 'middleBend': 0, 'indexBend': 0, 'thumbBend': 0,

        # visivility attribute
        'v': 1
    }



    ####메서드들



    def changeKeyScale_root(self, sourceRoot, controler):

        nameSpace = self.getNamespace()
        reScale_atts = ["tx", "ty", "tz"]

        # 타겟 케릭터의 루트 조인트의 위치
        targetRoot = cmds.xform(nameSpace + ":" + self.root_Jnt[0], q=1, ws=1,t=1)
        worldControlerPosition = cmds.xform(nameSpace + ":move_CON", q=1, ws=1,t=1)

        # 월드컨트롤러 기준으로 루트 조인트의 오브젝트스페이스 높이값
        targetRoot_Y = targetRoot[1] - worldControlerPosition[1]

        reScaleValue = targetRoot_Y / sourceRoot

        for att in reScale_atts:
            self.reScale_keyCurve((nameSpace+":"+controler),att,reScaleValue)



    def changeKeyScale_body(self, sourceChest, sourceSpineRoot, sourceSpineMiddle):

        nameSpace = self.getNamespace()
        reScale_atts = ["tx", "ty", "tz"]

        # 가슴,허리,힙의 조인트이름 가져오기
        targetChestJoint = nameSpace + ":" + self.body_Jnt[len(self.body_Jnt)-1]
        targetSpineRootJoint = nameSpace + ":" + self.body_Jnt[0]
        targetSpineMiddleJoint = nameSpace + ":" + self.body_Jnt[int(len(self.body_Jnt)/2)]
        worldControlerPosition = cmds.xform(nameSpace + ":move_CON", q=1, ws=1,t=1)
        targetChest_WS = cmds.xform(targetChestJoint, q=1, ws=1,t=1)
        targetChest_Y = targetChest_WS[1] - worldControlerPosition[1]
        targetSpineRoot_WS = cmds.xform(targetSpineRootJoint, q=1, ws=1,t=1)
        targetSpineRoot_Y = targetSpineRoot_WS[1] - worldControlerPosition[1]

        # 월드컨트롤러 기준으로 허리 조인트의 높이를 구함
        targetSpintMiddle_WS = cmds.xform(targetSpineMiddleJoint, q=1, ws=1,t=1)
        targetSpineMiddle_Y = targetSpintMiddle_WS[1] - worldControlerPosition[1]
        chestHightRatio = targetChest_Y / sourceChest
        spineRootHightRatio = targetSpineRoot_Y / sourceSpineRoot
        spineMiddleHightRatio = targetSpineMiddle_Y / sourceSpineMiddle
        bodyScaleRatio = [chestHightRatio, spineRootHightRatio, spineMiddleHightRatio]


        for i in range(len(self.upBodyCtrls)):
            for att in reScale_atts:
                self.reScale_keyCurve((nameSpace+":"+self.upBodyCtrls[i]),att,chestHightRatio)

        for i in range(len(self.middleBodyCtrls)):
            for att in reScale_atts:
                self.reScale_keyCurve((nameSpace+":"+self.middleBodyCtrls[i]),att,spineMiddleHightRatio)

        for i in range(len(self.lowBodyCtrls)):
            for att in reScale_atts:
                self.reScale_keyCurve((nameSpace+":"+self.lowBodyCtrls[i]),att,spineRootHightRatio)




    def changeKeyScale(self, sourceLength, jntList, controler):

        targetLength = 0.0
        nameSpace = self.getNamespace()
        reScale_atts = ["tx", "ty", "tz"]
        jntStartPoint = cmds.xform(nameSpace+":"+jntList[0], q=1, t=1, ws=1)

        for i in range(len(jntList)):

            # 각각의 조인트 길이를 합산
            #targetLength += abs(cmds.getAttr(nameSpace + ":" + jnt + ".tx"))
            sizeList = len(jntList)
            if(i<sizeList-1):

                child_jnt = nameSpace + ":" + jntList[i]
                parent_jnt = nameSpace + ":" + jntList[i + 1]

                # 기존 4족리깅 조인트리스트에 해당하는 조인트가 존재하지 않을경우
                if (cmds.objExists(child_jnt) != 1):
                    cmds.error("There is no joint to import the length value")

                targetLength += self.getDistance(child_jnt, parent_jnt)


        reScaleValue = targetLength/sourceLength

        for att in reScale_atts:
            self.reScale_keyCurve((nameSpace+":"+controler),att,reScaleValue)


        return targetLength



    def getIKOffsetPosition(self,controler,sourceLength,sourcePosition,jntList):

        targetLength = 0.0
        nameSpace = self.getNamespace()
        jntStartPoint = cmds.xform(nameSpace+":"+jntList[0], q=1, t=1, ws=1)

        for i in range(len(jntList)):

            # 각각의 조인트 길이를 합산
            #targetLength += abs(cmds.getAttr(nameSpace + ":" + jnt + ".tx"))
            sizeList = len(jntList)
            if(i<sizeList-1):

                child_jnt = nameSpace + ":" + jntList[i]
                parent_jnt = nameSpace + ":" + jntList[i + 1]

                # 기존 4족리깅 조인트리스트에 해당하는 조인트가 존재하지 않을경우
                if (cmds.objExists(child_jnt) != 1):
                    cmds.error("There is no joint to import the length value")

                targetLength += self.getDistance(child_jnt, parent_jnt)

        scaleValue = targetLength/sourceLength
        reScaleSourcePosition = [sourcePosition[0]*scaleValue, sourcePosition[1]*scaleValue, sourcePosition[2]*scaleValue]
        targetPosition = [reScaleSourcePosition[0]+jntStartPoint[0], reScaleSourcePosition[1]+jntStartPoint[1], reScaleSourcePosition[2]+jntStartPoint[2]]
        ctrlPosition = cmds.xform(nameSpace + ":" + controler, q=1, ws=1, t=1)
        offsetValue = [targetPosition[0]-ctrlPosition[0], targetPosition[1]-ctrlPosition[1], targetPosition[2]-ctrlPosition[2]]

        return offsetValue



    # 원본캐릭터와 소스 캐릭터간의 폴벡터 거리차이에 따른 키옵셋
    def setPolevectorOffset(self,controler,sourceDistance,jointName, legJointList, sourceLength):

        targetLength = 0.0
        nameSpace = self.getNamespace()

        for i in range(len(legJointList)):

            # 각각의 조인트 길이를 합산
            sizeList = len(legJointList)
            if(i<sizeList-1):

                child_jnt = nameSpace + ":" + legJointList[i]
                parent_jnt = nameSpace + ":" + legJointList[i + 1]

                # 기존 4족리깅 조인트리스트에 해당하는 조인트가 존재하지 않을경우
                if (cmds.objExists(child_jnt) != 1):
                    cmds.error("There is no joint to import the length value")

                targetLength += self.getDistance(child_jnt, parent_jnt)


        lengthRatio = targetLength / sourceLength
        poleControler = nameSpace +":"+controler
        poleJoint = nameSpace +":"+jointName
        targetDistance = self.getDistance(poleControler, poleJoint)
        DistanceDiff = targetDistance - sourceDistance
        offset_distance = DistanceDiff * lengthRatio

        # 폴벡터에 옵셋을 시킬 키커브가 존재하는지 확인 
        if(cmds.keyframe(poleControler, attribute = 'tz', tc=1, q=1) != None):

            cmds.selectKey((poleControler + ".tz"), replace=1, keyframe=1)
            if(poleControler.find('knee')!= -1):
                cmds.keyframe(animation="keys", option="over", relative=1, valueChange=-1*(offset_distance))
            else:
                cmds.keyframe(animation="keys", option="over", relative=1, valueChange=(offset_distance))




    def transformKey(self,sourceCtrl,targetCtrl,att):

        cmds.select(sourceCtrl, r=1)
        cmds.selectKey()
        keyPosition = cmds.keyframe(sourceCtrl, at=att, sl=1, q=1, tc=1)
        if keyPosition != None:
            cmds.copyKey(sourceCtrl, attribute=att, option="curve")
            cmds.pasteKey(targetCtrl, attribute=att, option="replace")
        else:
            cmds.warning("Key does not exist on selected object(controler)")




    def reScale_keyCurve(self, controler, att, reScaleValue):

        if (cmds.keyframe( controler, attribute=att, tc=1, q=1) != None):
            cmds.selectKey((controler + "." + att), replace=1, keyframe=1)
            cmds.scaleKey(ssk=1, timeScale=1, timePivot=0, floatScale=1, floatPivot=0,
                          valueScale=reScaleValue, valuePivot=0)
        else:
            cmds.warning("Key does not exist on selected " + controler)







    ##네임스페이스 가져오기
    def getNamespace(self):
        name_space = []
        listCtrls = cmds.ls(sl=1)

        #아무것도 선택하지 않았을 경우 에러메세지
        if(listCtrls == []):
            cmds.error("you must first select the controler or the object")


        else:

            # ":"가 존재하지 않는경우 -1을 반환해주는 find 사용
            sign =  listCtrls[0].find(':')


            #네임스페이스가 없는 경우
            if(sign == -1):
                return ''

            #네임스페이스가 있는경우
            else:
                for ctrl in listCtrls:
                    name_space.append(ctrl.split(":")[0])

                return name_space[0]


    ##주어진 컨트롤러에 전체키
    def setKeyAll(self, controlers):
        nameSpace = self.getNamespace()
        cmds.select(cl=1)
        for controler in controlers:
            cmds.select(nameSpace + ":" + controler, add=1)

        cmds.setKeyframe(breakdown=0, hierarchy='none', controlPoints=0, shape=0)


    ##두위치 사이의 거리값가져오기
    def getDistance(self, point1, point2):

        selected_Obj = cmds.ls(sl=1)
        pst1 = cmds.xform(point1, q=1, ws=1, t=1)
        pst2 = cmds.xform(point2, q=1, ws=1, t=1)

        cmds.createNode('distanceBetween', n="tempDistNode")
        cmds.setAttr("tempDistNode.p1x", pst1[0])
        cmds.setAttr("tempDistNode.p1y", pst1[1])
        cmds.setAttr("tempDistNode.p1z", pst1[2])

        cmds.setAttr("tempDistNode.p2x", pst2[0])
        cmds.setAttr("tempDistNode.p2y", pst2[1])
        cmds.setAttr("tempDistNode.p2z", pst2[2])

        dist = cmds.getAttr("tempDistNode.distance")
        cmds.delete("tempDistNode")

        cmds.select(selected_Obj,r=1)

        return dist




    def offsetPosition(self, target, offsetValue):

        nameSpace = self.getNamespace()
        #print (nameSpace + ":" + target + ".tx")

        cmds.selectKey((nameSpace + ":" + target + ".tx"), replace=1, keyframe=1)
        cmds.keyframe(animation="keys", option="over", relative=1, valueChange=offsetValue[0])

        cmds.selectKey((nameSpace + ":" + target + ".ty"), replace=1, keyframe=1)
        cmds.keyframe(animation="keys", option="over", relative=1, valueChange=offsetValue[1])

        cmds.selectKey((nameSpace + ":" + target + ".tz"), replace=1, keyframe=1)
        cmds.keyframe(animation="keys", option="over", relative=1, valueChange=offsetValue[2])




    def findMinMaxKey(self, controlers):

        nameSpace = self.getNamespace()
        cmds.select(cl=1)


        for controler in controlers:
            if(cmds.objExists(nameSpace + ":" + controler):
                cmds.select(nameSpace + ":" + controler, add=1)

        sel_ctrls = cmds.ls(sl=1)
        minKeyTimes = []
        maxKeyTimes = []
        cmds.selectKey()

        for num in range(len(sel_ctrls)):
            keyPosition = cmds.keyframe(sel_ctrls[num], sl=1, q=1, tc=1)
            if keyPosition != None:
                minKeyTimes.append(min(keyPosition))
                maxKeyTimes.append(max(keyPosition))

        if (minKeyTimes == []):
            cmds.error("key does not exist on this controller(s)")

        minKeyPst = min(minKeyTimes)
        maxKeyPst = max(maxKeyTimes)

        return minKeyPst, maxKeyPst


    ##주어진 컨트롤러들의 초기화 포즈셋팅
    def makeInitPose(self, controlers, attributes):
        nameSpace = self.getNamespace()
        attList = attributes.keys()

        for controler in controlers:

            for att in attList:
                attExist = cmds.attributeQuery(att, n=nameSpace + ":" + controler, ex=1)

                if (attExist == 1):
                    lockStatus = cmds.getAttr(nameSpace + ":" + controler + "." + att, l=1)
                    # connectStatus = cmds.connectionInfo(nameSpace + ":" + controler + "." + att, ies=True)
                    KeyStatus = cmds.keyframe(nameSpace + ":" + controler, attribute=att, tc=1, q=1)
                    if (lockStatus == 0 and KeyStatus != None):
                        cmds.setAttr(nameSpace + ":" + controler + "." + att, attributes[att])





    def delKeyInitial(self, controlerList):

        nameSpace = self.getNamespace()
        currentTime = cmds.currentTime(q=1)

        for controler in controlerList:
            controlerName = nameSpace + ":" + controler
            cmds.cutKey(controlerName, clear=1, t=(currentTime,currentTime))





    def delKeyCurrent(self, controlerList, current):

        nameSpace = self.getNamespace()

        for controler in controlerList:
            controlerName = nameSpace + ":" + controler
            cmds.cutKey(controlerName, clear=1, t=(current,current))



#########################################################################################################
#########################################################################################################

def fixRotateOrient(orientData):

    print orientData




def getSourceData(*argv):

    #######변수들###
    componentList = ['neck_head','L_foreLeg', 'R_foreLeg', 'L_rearLeg', 'R_rearLeg']



    neckHead_Jnt = ['C_IK_neck1_JNT','C_IK_neck2_JNT','C_IK_neck3_JNT','C_IK_neck4_JNT','C_IK_neck5_JNT','C_IK_neck6_JNT', 'C_IK_head_JNT']
    L_foreLeg_Jnt = ['L_Skin_shoulder_JNT','L_Skin_edbow_JNT', 'L_Skin_wrist_JNT', 'L_Skin_foot1_JNT']
    R_foreLeg_Jnt = ['R_Skin_shoulder_JNT','R_Skin_edbow_JNT', 'R_Skin_wrist_JNT', 'R_Skin_foot1_JNT']
    L_rearLeg_Jnt = ['L_Skin_hip_JNT','L_Skin_knee_JNT','L_Skin_ankle_JNT','L_Skin_ball_JNT']
    R_rearLeg_Jnt = ['R_Skin_hip_JNT','R_Skin_knee_JNT','R_Skin_ankle_JNT','R_Skin_ball_JNT']

    jnt_lists = [neckHead_Jnt, L_foreLeg_Jnt, R_foreLeg_Jnt, L_rearLeg_Jnt, R_rearLeg_Jnt]



    polevector_controler = ['L_IK_edbow_CON', 'R_I K_edbow_CON', 'L_IK_knee_CON', 'R_IK_knee_CON']



    body_Jnt = ['C_IK_spine1_JNT', 'C_IK_spine2_JNT', 'C_IK_spine3_JNT', 'C_IK_spine4_JNT', 'C_IK_spine5_JNT', 'C_IK_spine6_JNT', 'C_IK_chest_JNT']


    # 네임스페이스를 가져오기 위해 선택된 소스 캐릭터의 컨트롤러 이름 가져오기
    selectedNam = cmds.ls(sl=1)
    # 네임스페이스가져오기
    sourceNam = getNamespace(selectedNam)




    for i in range(len(measure_lists)):

        jntLength = 0.0
        ctrlPositionRatio = 0.0
        endJointPosition = [0.0, 0.0, 0.0]
        bodyJntLength = 0.0
        upBodyHight = 0.0
        lowBodyHight = 0.0

        # 각 부위별 조인트들의 길이의 총합 구하기
        for j in range(len(jnt_lists[i])):

            if(j<len(jnt_lists[i])-1):
                jntLength += getDistance(sourceNam+":"+jnt_lists[i][j], sourceNam+":"+jnt_lists[i][j+1])


        numJnt = len(jnt_lists[i])


        poseLength = getDistance(sourceNam+":"+jnt_lists[i][0], sourceNam+":"+jnt_lists[i][numJnt-1])
        ctrlPositionRatio = poseLength / jntLength
        endJointPosition = findLengthPosition(sourceNam+":"+jnt_lists[i][0], sourceNam+":"+jnt_lists[i][numJnt-1])

        measure_lists[i] = [jntLength, ctrlPositionRatio, endJointPosition]
        print componentList[i]   + " : ", measure_lists[i]


    # 루트조인트의 위치값;
    sourceRootPosition[0] = cmds.xform(sourceNam + ":C_IK_root_JNT", q=1, ws=1, t=1 )
    print "root position : ", sourceRootPosition[0]

    for i in range(len(body_Jnt)):

        if (i < len(body_Jnt)-1):
            bodyJntLength += getDistance(sourceNam + ":" + body_Jnt[i], sourceNam + ":" + body_Jnt[i + 1])

    chestJoint =  sourceNam + ":" + body_Jnt[len(body_Jnt)-1]
    spineRootJoint = sourceNam + ":" + body_Jnt[0]
    spineMiddleJoint = sourceNam + ":" + body_Jnt[int(len(body_Jnt)/2)]
    worldControler =  sourceNam + ":move_CON"


    worldControlerPosition = cmds.xform(worldControler, q=1, ws=1, t=1)
    chest_WS_Position = cmds.xform(chestJoint, q=1, ws=1, t=1)
    chest_OS_Position = [chest_WS_Position[0]-worldControlerPosition[0], chest_WS_Position[1]-worldControlerPosition[1], chest_WS_Position[2]-worldControlerPosition[2]]

    spineRoot_WS_Position = cmds.xform(chestJoint, q=1, ws=1, t=1)
    spineRoot_OS_Position = [spineRoot_WS_Position[0]-worldControlerPosition[0], spineRoot_WS_Position[1]-worldControlerPosition[1], spineRoot_WS_Position[2]-worldControlerPosition[2]]

    spineMiddle_WS_Position = cmds.xform(spineMiddleJoint, q=1, ws=1, t=1)
    spineMiddle_OS_Position = [spineMiddle_WS_Position[0]-worldControlerPosition[0], spineMiddle_WS_Position[1]-worldControlerPosition[1], spineMiddle_WS_Position[2]-worldControlerPosition[2]]


    bodyData[0] = [bodyJntLength, chest_OS_Position[1], spineRoot_OS_Position[1], spineMiddle_OS_Position[1]]
    print "body data : ", bodyData[0]


    L_elbow_poleDistance = getDistance(sourceNam + ":L_IK_edbow_JNT", sourceNam + ":L_IK_edbow_CON")
    R_elbow_poleDistance = getDistance(sourceNam + ":R_IK_edbow_JNT", sourceNam + ":R_IK_edbow_CON")
    L_knee_poleDistance = getDistance(sourceNam + ":L_IK_knee_JNT", sourceNam + ":L_IK_knee_CON")
    R_knee_poleDistance = getDistance(sourceNam + ":R_IK_knee_JNT", sourceNam + ":R_IK_knee_CON")


    poleData[0] = [L_elbow_poleDistance, R_elbow_poleDistance,L_knee_poleDistance,R_knee_poleDistance]
    print "pole dintance : ", poleData[0]







def findLengthPosition(point1, point2):

    pst1 = cmds.xform(point1, ws=1, t=1, q=1)
    pst2 = cmds.xform(point2, ws=1, t=1, q=1)

    pstX = pst2[0] - pst1[0]
    pstY = pst2[1] - pst1[1]
    pstZ = pst2[2] - pst1[2]

    lengthPosition = [pstX, pstY, pstZ]
    return lengthPosition







def exeRetargetClass(inst):


    initialSetup(inst)

    for i in range(len(inst.controler_lists)):
        for controler in inst.controler_lists[i]:
            inst.changeKeyScale(measure_lists[i][0], inst.jnt_lists[i], controler)

    for controler in inst.root_Ctrls:
        inst.changeKeyScale_root(sourceRootPosition[0][1],controler)

    inst.changeKeyScale_body(bodyData[0][1], bodyData[0][2], bodyData[0][3])

    for i in range(len(inst.IK_controler)):

        offset = inst.getIKOffsetPosition(inst.IK_controler[i], measure_lists[i][0],measure_lists[i][2],inst.jnt_lists[i])
        inst.offsetPosition(inst.IK_controler[i], offset)



    for i in range(len(inst.polevector_controler)):

        inst.setPolevectorOffset(inst.polevector_controler[i], poleData[0][i], inst.polevector_joint[i], inst.forPoleJnt_lists[i], measure_lists[i+1][0])



    # 리타겟팅을 위해 셋팅한 이니셜포즈 키값 삭제
    inst.delKeyInitial(inst.initPoseCtrl_list)






def retarget_EXE(*argv):

    selected = cmds.ls(sl=1)

    #리타겟팅을 실행하기 위해 2개이상의 컨트롤러의 선택을 했는지 체크
    if(len(selected)<2):
        cmds.error("리타겟팅을 실행하려면 하나의 소스캐릭터와 하나이상의 타겟케릭터를 선택해야 합니다")

    #선택된 컨트롤러중 첫번째 컨트롤러의 이름에서 네임스페이스를 가져와서 소스캐릭터의 네임스페이스로 지정;
    cmds.select(cl=1)
    cmds.select(selected[0], r=1)
    sourceCtrl = cmds.ls(sl=1)
    sourceNamespace = getNamespace(sourceCtrl)

    sourceChar = Retarget()
    initialSetup(sourceChar)


    cmds.select(cl=1)
    cmds.select(selected[0], r=1)
    getSourceData()


    # 리타겟팅을 위해 셋팅한 소스캐릭터의 이니셜포즈 키값 삭제
    keyPosition = sourceChar.findMinMaxKey(sourceChar.initPoseCtrl_list)
    sourceChar.delKeyCurrent(sourceChar.initPoseCtrl_list, keyPosition[0])


    #타겟캐릭터들의 네임스페이스들을 저장할 리스트 선언;
    targetNamespaces = []
    targetWorldPosition = []

    #첫번째 선택된 컨트롤러를 제외한 나머지 컨트롤러들(타겟캐릭터들)의 이름에서 네임스페이스만 가져와서 타겟캐릭터 네임스페이스리스트 만들기
    for i in range(len(selected)):
        if(i<len(selected)-1):
            cmds.select(cl=1)
            cmds.select(selected[i+1], r=1)
            targetCtrl = cmds.ls(sl=1)
            namespace = getNamespace(targetCtrl)

            #타겟캐릭터들의 월드포지션 구하기
            targetPosition = getWorldPostion(targetCtrl)
            targetWorldPosition.append(targetPosition)

            #print targetPosition

            if namespace not in targetNamespaces:
                targetNamespaces.append(namespace)


    print targetWorldPosition



    #소스캐릭터의 전체키를 타겟 캐릭터로 복사
    for targetNamespace in targetNamespaces:
        for controler in Retarget.allControler:
            source = sourceNamespace + ":" + controler
            target = targetNamespace + ":" + controler

            #컨트롤러가 실제로 존재하는지 확인;
            if (cmds.objExists(source)):
                #컨트롤러에 키가 존재하는지 확인;
                if (cmds.keyframe(source, tc=1, q=1) != None):
                    cmds.copyKey(source)
                    if(cmds.objExists(target)):
                        cmds.pasteKey(target)




    for i in range(len(selected)):
        if(i<len(selected)-1):
            cmds.select(cl=1)

            cmds.select(selected[i+1], r=1)

            selectedName = cmds.ls(sl=1)
            currentTargetNamespace = getNamespace(selectedName)
            target = Retarget()

            '''
            exeRetargetClass(target)
            '''

            #타겟캐릭터의 원래 월드포지션으로 위치를 옮겨주기
            targetKeyMinMax = target.findMinMaxKey(target.allControler)
            cmds.currentTime(targetKeyMinMax[0])

            attribute = ['tx','ty', 'tz', 'rx', 'ry', 'rz']
            worldCON_offsetValue = []

            for j in range(len(attribute)):

                value = targetWorldPosition[i][j] - cmds.getAttr(currentTargetNamespace + ":place_CON." + attribute[j])

                print targetWorldPosition[i][j], attribute[j]

                if (cmds.keyframe(currentTargetNamespace + ":place_CON", attribute=attribute[j], tc=1, q=1) != None):
                    cmds.selectKey((currentTargetNamespace + ":place_CON." + attribute[j]), replace=1, keyframe=1)
                    cmds.keyframe(animation="keys", option="over", relative=1, valueChange=value)

                else:
                    cmds.setAttr(currentTargetNamespace + ":place_CON." + attribute[j], targetWorldPosition[i][j])





##### 월드컨트롤러 위치값을 저장하고 현재 위치값을 원점으로 설정 #############################################################

def getWorldPostion(controlerList):

    atts = ("tx","ty","tz","rx","ry","rz")
    pos = []
    nsChar = getNamespace(controlerList)
    for att in atts:
        value = cmds.getAttr(nsChar + ":place_CON." + att)
        pos.append(value)

    return pos



def exeFixOrient(*argv):

    fixRotateOrient(orientConvertData)


##전체 키프레임의 가장 앞쪽위치를 찾아낸 뒤 거기로부터 -50프레임 앞쪽에 이니셜포즈 셋팅 및 전체키
##로컬스페이스 채널의 키 초기화방법 찾아야 함- need need need need need need need need need need need need need need need need need need
def initialSetup(inst):

    # get min keyframe position
    minMaxValue = inst.findMinMaxKey(inst.initPoseCtrl_list)

    # move to first key position
    cmds.currentTime(minMaxValue[0])

    # set key All controler
    inst.setKeyAll(inst.initPoseCtrl_list)

    initKeyframe = minMaxValue[0] - 50

    # move to create initial pose
    cmds.currentTime(initKeyframe)

    # make initial pose
    inst.makeInitPose(inst.initPoseCtrl_list, inst.initPoseAtt_list)

    # set key All controler
    inst.setKeyAll(inst.initPoseCtrl_list)





##네임스페이스 가져오기
def getNamespace(selected_list):

    name_space = []
    # 아무것도 선택하지 않았을 경우 에러메세지
    if (selected_list == []):
        cmds.error("you must first select the controler or the object")

    else:

        # ":"가 존재하지 않는경우 -1을 반환해주는 find 사용
        sign = selected_list[0].find(':')

        # 네임스페이스가 없는 경우
        if (sign == -1):
            cmds.error("namespace does not exist")

    # 네임스페이스가 있는경우
        else:
            for ctrl in selected_list:
                name_space.append(ctrl.split(":")[0])

            return name_space[0]






##두위치 사이의 거리값가져오기
def getDistance(point1, point2):

    selected_Obj = cmds.ls(sl=1)
    pst1 = cmds.xform(point1, q=1, ws=1, t=1)
    pst2 = cmds.xform(point2, q=1, ws=1, t=1)

    cmds.createNode('distanceBetween', n="tempDistNode")
    cmds.setAttr("tempDistNode.p1x", pst1[0])
    cmds.setAttr("tempDistNode.p1y", pst1[1])
    cmds.setAttr("tempDistNode.p1z", pst1[2])

    cmds.setAttr("tempDistNode.p2x", pst2[0])
    cmds.setAttr("tempDistNode.p2y", pst2[1])
    cmds.setAttr("tempDistNode.p2z", pst2[2])

    dist = cmds.getAttr("tempDistNode.distance")
    cmds.delete("tempDistNode")

    cmds.select(selected_Obj,r=1)

    return dist




def makeUI():
    if (cmds.window('animRetargeter', q=1, ex=1)):
        cmds.deleteUI('animRetargeter', window=True)

    windowUI = cmds.window('animRetargeter', t="anim retargeter", rtf=1)
    cmds.columnLayout(adjustableColumn=True)
    cmds.button("fix_Button", w=300, l='Change orient', command=exeFixOrient)
    #cmds.button("SSR_Button", w=300, l='Get source data', command=getSourceData)
    #cmds.button("GCD1_Button", w=300, l='Retarget', command=exeClass)
    cmds.button("retarget_Button", w=300, l='RetargetEXE', command=retarget_EXE)


    cmds.setParent('..')
    cmds.showWindow(windowUI)


def mm(*argv):
    wolf = Retarget()
    wolf.test()