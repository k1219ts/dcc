# encoding:utf-8
# !/usr/bin/env python


import sys
import maya.cmds as cmds
import maya.mel as mel


from HumanRig import humanRigClass
reload(humanRigClass)

from Module import aniModule1
reload(aniModule1)

retargetCheckBox = 0





def getNamespace():
    name_space = []
    listCtrls = cmds.ls(sl=1)

    if(listCtrls == []):
        cmds.error("you must select the controler or the object")
    else:
        sign =  listCtrls[0].find(':')
        if(sign == -1):
            return ''

        else:
            for ctrl in listCtrls:
                name_space.append(ctrl.split(":")[0])

            return name_space[0]






def deleteRig(nameSpace, rigElements, skinJointList):


    cmds.select(nameSpace+":"+"SkinJoint_GRP", hi=1)

    allNode_SkinJointGRP = cmds.ls(sl=1)

    for node in allNode_SkinJointGRP:
        nodeType = cmds.nodeType(node)
        if(nodeType == 'parentConstraint' or nodeType == 'pointConstraint' or nodeType == 'orientConstraint'):
            cmds.delete(node)


    cmds.select(cl=1)
    cmds.select(ado=1)
    topNode = cmds.ls(sl=1)

    cmds.parent(nameSpace+":SkinJoint_GRP", w=1)
    cmds.parent(nameSpace+":SkinJoint_GRP", topNode[0])

    cmds.parent(nameSpace+":geometry_GRP", w=1)
    cmds.parent(nameSpace+":geometry_GRP", topNode[0])

    cmds.select(ado=1, hi=1)
    cmds.select(topNode[0], d=1)
    cmds.select(nameSpace+":SkinJoint_GRP", d=1, hi=1)
    cmds.select(nameSpace+":geometry_GRP", d=1, hi=1)

    cmds.delete()

    cmds.setAttr( nameSpace+":SkinJoint_GRP.v", 1)



    '''
    cmds.select(nameSpace+":"+"SkinJoint_GRP", hi=1)

    allNode_SkinJointGRP = cmds.ls(sl=1)

    for node in allNode_SkinJointGRP:
        nodeType = cmds.nodeType(node)
        if(nodeType == 'parentConstraint' or nodeType == 'pointConstraint' or nodeType == 'orientConstraint'):
            cmds.delete(node)

    for jnt in skinJointList:
        if(cmds.objExists(nameSpace+":"+jnt)):
            cmds.parent(nameSpace+":"+jnt, w=1)

    for rigElement in rigElements:
        if cmds.objExists((nameSpace + ":" + rigElement)):
            cmds.delete(nameSpace + ":" + rigElement)
    '''





def HIK_armJointArrange(nameSpace,skinJointList, HIKjointList, aim_vec):

    for i in range(len(HIKjointList)):
        temp_point_con = cmds.pointConstraint("%s:%s" % (nameSpace, skinJointList[i]),
                                              "%s:%s" % (nameSpace, HIKjointList[i]), mo=0, w=1)
        cmds.delete(temp_point_con[0])

        if(i<len(HIKjointList)-1):
            temp_aim_con =  cmds.aimConstraint("%s:%s" % (nameSpace, skinJointList[i+1]),
                                               "%s:%s" % (nameSpace, HIKjointList[i]), aim=aim_vec, w=1)
            cmds.delete(temp_aim_con[0])






def crd_jointArrange(nameSpace,skinJointList, targetJointList, aim_vec, up_vec):

    for i in range(len(targetJointList)):
        temp_point_con = cmds.pointConstraint("%s:%s" % (nameSpace, skinJointList[i]),
                                              "%s:%s" % (nameSpace, targetJointList[i]), mo=0, w=1)
        cmds.delete(temp_point_con[0])

        if(i<len(targetJointList)-1):
            temp_aim_con =  cmds.aimConstraint("%s:%s" % (nameSpace, skinJointList[i+1]),
                                               "%s:%s" % (nameSpace, targetJointList[i]), aim=aim_vec, w=1, wu=up_vec)
            cmds.delete(temp_aim_con[0])






def MCP_makeHierachy(inst, nameSpace):

    if cmds.ls(type='transform').count("%s:HIKJoint_LOC" % nameSpace) == 1:
        cmds.delete("%s:HIKJoint_LOC" % nameSpace)

    cmds.file('/stdrepo/ANI/Asset/HumanIK_for_AnimBrowser//HIKJoint.ma', i=True,
        type='mayaAscii', ra=True, mergeNamespacesOnClash=True, namespace=nameSpace, options='v=0')

    HIKJoint_constraint_list = []
    for u in range(len(inst.HIKJoint_list)):
        temp_point_con = cmds.pointConstraint("%s:%s" % (nameSpace, inst.connectSkinJoint_list[u]),
                                              "%s:%s" % (nameSpace, inst.HIKJoint_list[u]), mo=0, w=1)
        HIKJoint_constraint_list.append(temp_point_con[0])
    cmds.delete(HIKJoint_constraint_list)

    L_arm_aimVec = [1,0,0]
    R_arm_aimVec = [-1,0,0]
    HIK_armJointArrange(nameSpace, inst.L_SkinArmJoint_list, inst.HIK_L_ArmJoint_list,L_arm_aimVec)
    HIK_armJointArrange(nameSpace, inst.R_SkinArmJoint_list, inst.HIK_R_ArmJoint_list,R_arm_aimVec)

    skinJNT_list = inst.skin_HIK_hierachyList.keys()
    for skinJNT in skinJNT_list:
        if(inst.skin_HIK_hierachyList[skinJNT] != ''):
            if(cmds.objExists(nameSpace+':'+skinJNT)):
                cmds.parent((nameSpace+':'+skinJNT), w=1)
                cmds.parent((nameSpace+':'+skinJNT), (nameSpace+':'+inst.skin_HIK_hierachyList[skinJNT]))




def MCP_makeHierachy_HindLeg(inst, nameSpace):

    if cmds.ls(type='transform').count("%s:HIKJoint_LOC" % nameSpace) == 1:
        cmds.delete("%s:HIKJoint_LOC" % nameSpace)

    cmds.file('/stdrepo/ANI/Asset/HumanIK_for_AnimBrowser//HIKJoint.ma', i=True,
        type='mayaAscii', ra=True, mergeNamespacesOnClash=True, namespace=nameSpace, options='v=0')

    HIKJoint_constraint_list = []
    for u in range(len(inst.HIKJoint_list)):
        temp_point_con = cmds.pointConstraint("%s:%s" % (nameSpace, inst.connectSkinJoint_list[u]),
                                              "%s:%s" % (nameSpace, inst.HIKJoint_list[u]), mo=0, w=1)
        HIKJoint_constraint_list.append(temp_point_con[0])
    cmds.delete(HIKJoint_constraint_list)

    L_arm_aimVec = [1,0,0]
    R_arm_aimVec = [-1,0,0]
    HIK_armJointArrange(nameSpace, inst.L_SkinArmJoint_list, inst.HIK_L_ArmJoint_list,L_arm_aimVec)
    HIK_armJointArrange(nameSpace, inst.R_SkinArmJoint_list, inst.HIK_R_ArmJoint_list,R_arm_aimVec)

    skinJNT_list = inst.skin_HIK_hierachyList.keys()
    for skinJNT in skinJNT_list:
        if(inst.skin_HIK_hierachyList[skinJNT] != ''):
            if(cmds.objExists(nameSpace+':'+skinJNT)):
                cmds.parent((nameSpace+':'+skinJNT), (nameSpace+':'+inst.skin_HIK_hierachyList[skinJNT]))


    if(cmds.objExists(nameSpace+":L_hindLeg_GRP") and cmds.objExists(nameSpace+":R_hindLeg_GRP")):
        cmds.delete(nameSpace+":L_hindLeg_GRP", nameSpace+":R_hindLeg_GRP")

    if(cmds.objExists(nameSpace+":L_Skin_toe_JNT") and cmds.objExists(nameSpace+":R_Skin_toe_JNT")):
        cmds.parent(nameSpace+":L_Skin_toe_JNT", nameSpace+":LeftToeBase")
        cmds.parent(nameSpace+":R_Skin_toe_JNT", nameSpace+":RightToeBase")





def CRD_makeHierachy(inst, nameSpace):


    '''
    if cmds.ls(type='transform').count("%s:mocap_char" % nameSpace + "_mcp") == 1:
        cmds.delete("%s:mocap_char" % nameSpace + "_mcp")
    cmds.file('/stdrepo/ANI/Asset/HumanIK_for_AnimBrowser//crdJoint.mb', i=True, type='mayaBinary', ra=True, mergeNamespacesOnClash=True, namespace=nameSpace, options='v=0')

    crdJoint_constraint_list = []
    for u in range(len(inst.crd_BodyJoint_list)):
        temp_point_con = cmds.pointConstraint("%s:%s" % (nameSpace, inst.SkinBodyJoint_list[u]),
                                              "%s:%s" % (nameSpace, inst.crd_BodyJoint_list[u]), mo=0, w=1)
        crdJoint_constraint_list.append(temp_point_con[0])
    cmds.delete(crdJoint_constraint_list)



    L_arm_aimVec = [1,0,0]
    L_arm_upVec = [0,1,0]
    R_arm_aimVec = [-1,0,0]
    R_arm_upVec = [0,-1,0]

    crd_jointArrange(nameSpace, inst.L_SkinArmJoint_list, inst.crd_L_ArmJoint_list,L_arm_aimVec,L_arm_upVec)
    crd_jointArrange(nameSpace, inst.R_SkinArmJoint_list, inst.crd_R_ArmJoint_list,R_arm_aimVec,R_arm_upVec)

    L_leg_aimVec = [1,0,0]
    L_leg_upVec = [0,0,1]
    R_leg_aimVec = [-1,0,0]
    R_leg_upVec = [0,0,-1]
    crd_jointArrange(nameSpace, inst.L_SkinLegJoint_list, inst.crd_L_LegJoint_list,L_leg_aimVec,L_leg_upVec)
    crd_jointArrange(nameSpace, inst.R_SkinLegJoint_list, inst.crd_R_LegJoint_list,R_leg_aimVec,R_leg_upVec)
    '''

    #make skinJont hierachy
    skinJNT_list = inst.skinJNT_hierachy.keys()
    for skinJNT in skinJNT_list:
        if(inst.skinJNT_hierachy[skinJNT] != ''):
            if(cmds.objExists(nameSpace+':'+skinJNT)):
                cmds.parent((nameSpace+':'+skinJNT), (nameSpace+':'+inst.skinJNT_hierachy[skinJNT]))




def parentTo_HIK(nameSpace, hierachyList):

    skinJNT_list = hierachyList.keys()
    for skinJNT in skinJNT_list:
        if(hierachyList[skinJNT] != ''):
            if(cmds.objExists(nameSpace+':'+skinJNT)):
                cmds.parent((nameSpace+':'+skinJNT), (nameSpace+':'+hierachyList[skinJNT]))




def CRD_makeHierachy_skinJoint(nameSpace, hierachyList):

    skinJNT_list = hierachyList.keys()
    for skinJNT in skinJNT_list:
        if(hierachyList[skinJNT] != ''):
            if(cmds.objExists(nameSpace+':'+skinJNT)):
                cmds.parent((nameSpace+':'+skinJNT), w=1)
                cmds.parent((nameSpace+':'+skinJNT), (nameSpace+':'+hierachyList[skinJNT]))





def HIK_makeHierachy_withFinger(inst, nameSpace):

    if cmds.ls(type='transform').count("%s:HIKJoint_LOC" % nameSpace) == 1:
        cmds.delete("%s:HIKJoint_LOC" % nameSpace)

    if(nameSpace == ''):
        nameSpace = ":"

    cmds.file('/stdrepo/ANI/Asset/HumanIK_for_AnimBrowser//HIKJoint_withFinger.ma', i=True,
        type='mayaAscii', ra=True, mergeNamespacesOnClash=True, namespace=nameSpace, options='v=0')

    HIKJoint_constraint_list = []
    for u in range(len(inst.HIKJoint_list)):
        temp_point_con = cmds.pointConstraint("%s:%s" % (nameSpace, inst.connectSkinJoint_list[u]),
                                              "%s:%s" % (nameSpace, inst.HIKJoint_list[u]), mo=0, w=1)
        HIKJoint_constraint_list.append(temp_point_con[0])
    #cmds.delete(HIKJoint_constraint_list)

    L_arm_aimVec = [1,0,0]
    R_arm_aimVec = [-1,0,0]
    HIK_armJointArrange(nameSpace, inst.L_SkinArmJoint_list, inst.HIK_L_ArmJoint_list,L_arm_aimVec)
    HIK_armJointArrange(nameSpace, inst.R_SkinArmJoint_list, inst.HIK_R_ArmJoint_list,R_arm_aimVec)

    for u in range(len(inst.HIK_fingerJoint_list)):
        if(cmds.objExists(nameSpace+":"+inst.fingerJoint_list[u])):
            temp_point_con = cmds.pointConstraint("%s:%s" % (nameSpace, inst.fingerJoint_list[u]),
                                                  "%s:%s" % (nameSpace, inst.HIK_fingerJoint_list[u]), mo=0, w=1)
            HIKJoint_constraint_list.append(temp_point_con[0])
    cmds.delete(HIKJoint_constraint_list)





def HIK_makeHierachy_temp(inst, nameSpace):

    if cmds.ls(type='transform').count("%s:HIKJoint_LOC" % nameSpace) == 1:
        cmds.delete("%s:HIKJoint_LOC" % nameSpace)

    if(nameSpace == ''):
        nameSpace = ":"

    cmds.file('/home/sungoh.moon/Desktop/project/scripts/MC_rig/test_scene/export/HumanIK_for_AnimBrowser/HIKJoint_withFinger_edit.ma', i=True,
        type='mayaAscii', ra=True, mergeNamespacesOnClash=True, namespace=nameSpace, options='v=0')

    HIKJoint_constraint_list = []
    for u in range(len(inst.HIKJoint_list)):
        temp_point_con = cmds.pointConstraint("%s:%s" % (nameSpace, inst.connectSkinJoint_list[u]),
                                              "%s:%s" % (nameSpace, inst.HIKJoint_list[u]), mo=0, w=1)
        HIKJoint_constraint_list.append(temp_point_con[0])
    #cmds.delete(HIKJoint_constraint_list)

    L_arm_aimVec = [1,0,0]
    R_arm_aimVec = [-1,0,0]
    HIK_armJointArrange(nameSpace, inst.L_SkinArmJoint_list, inst.HIK_L_ArmJoint_list,L_arm_aimVec)
    HIK_armJointArrange(nameSpace, inst.R_SkinArmJoint_list, inst.HIK_R_ArmJoint_list,R_arm_aimVec)

    for u in range(len(inst.HIK_fingerJoint_list)):
        if(cmds.objExists(nameSpace+":"+inst.fingerJoint_list[u])):
            temp_point_con = cmds.pointConstraint("%s:%s" % (nameSpace, inst.fingerJoint_list[u]),
                                                  "%s:%s" % (nameSpace, inst.HIK_fingerJoint_list[u]), mo=0, w=1)
            HIKJoint_constraint_list.append(temp_point_con[0])
    cmds.delete(HIKJoint_constraint_list)






def HIK_makeHierachy(inst, nameSpace):

    if cmds.ls(type='transform').count("%s:HIKJoint_LOC" % nameSpace) == 1:
        cmds.delete("%s:HIKJoint_LOC" % nameSpace)

    if(nameSpace == ''):
        nameSpace = ":"

    cmds.file('/stdrepo/ANI/Asset/HumanIK_for_AnimBrowser//HIKJoint.ma', i=True,
        type='mayaAscii', ra=True, mergeNamespacesOnClash=True, namespace=nameSpace, options='v=0')


    skinRoot_pst = cmds.xform(nameSpace + ":C_Skin_hip_JNT", q=1, t=1, ws=1)
    HIKRoot_pst = cmds.xform(nameSpace + ":Hips", q=1, t=1, ws=1)

    Root_dif_pst = skinRoot_pst[1] - HIKRoot_pst[1]


    HIKJoint_constraint_list = []
    for u in range(len(inst.HIKJoint_list)):
        temp_point_con = cmds.pointConstraint("%s:%s" % (nameSpace, inst.connectSkinJoint_list[u]),
                                              "%s:%s" % (nameSpace, inst.HIKJoint_list[u]), mo=0, w=1)
        HIKJoint_constraint_list.append(temp_point_con[0])
    cmds.delete(HIKJoint_constraint_list)

    L_arm_aimVec = [1,0,0]
    R_arm_aimVec = [-1,0,0]
    HIK_armJointArrange(nameSpace, inst.L_SkinArmJoint_list, inst.HIK_L_ArmJoint_list,L_arm_aimVec)
    HIK_armJointArrange(nameSpace, inst.R_SkinArmJoint_list, inst.HIK_R_ArmJoint_list,R_arm_aimVec)


    return Root_dif_pst


    '''
    for u in range(len(inst.HIK_fingerJoint_list)):
        if(cmds.objExists(nameSpace+":"+inst.fingerJoint_list[u])):
            temp_point_con = cmds.pointConstraint("%s:%s" % (nameSpace, inst.fingerJoint_list[u]),
                                                  "%s:%s" % (nameSpace, inst.HIK_fingerJoint_list[u]), mo=0, w=1)
            HIKJoint_constraint_list.append(temp_point_con[0])
    cmds.delete(HIKJoint_constraint_list)
    '''




def HIK_T_pose(*argv):

    human_set = humanRigClass.human_rig()

    for jnt in human_set.HIK_L_ArmJoint_list:
        cmds.setAttr(human_set.nameSpace+":"+jnt+".rx", 0)
        cmds.setAttr(human_set.nameSpace+":"+jnt+".ry", 0)
        cmds.setAttr(human_set.nameSpace+":"+jnt+".rz", 0)

    for jnt in human_set.HIK_R_ArmJoint_list:
        cmds.setAttr(human_set.nameSpace+":"+jnt+".rx", 0)
        cmds.setAttr(human_set.nameSpace+":"+jnt+".ry", 0)
        cmds.setAttr(human_set.nameSpace+":"+jnt+".rz", 0)



def crd_T_pose(*argv):

    human_set = humanRigClass.human_rig()

    for jnt in human_set.crd_L_ArmJoint_list:
        cmds.setAttr(human_set.nameSpace+":"+jnt+".rx", 0)
        cmds.setAttr(human_set.nameSpace+":"+jnt+".ry", 0)
        cmds.setAttr(human_set.nameSpace+":"+jnt+".rz", 0)

    for jnt in human_set.crd_R_ArmJoint_list:
        cmds.setAttr(human_set.nameSpace+":"+jnt+".rx", 0)
        cmds.setAttr(human_set.nameSpace+":"+jnt+".ry", 0)
        cmds.setAttr(human_set.nameSpace+":"+jnt+".rz", 0)


    for jnt in human_set.crd_L_LegJoint_list:
        cmds.setAttr(human_set.nameSpace+":"+jnt+".rx", 0)
        cmds.setAttr(human_set.nameSpace+":"+jnt+".ry", 0)
        cmds.setAttr(human_set.nameSpace+":"+jnt+".rz", 0)

    for jnt in human_set.crd_R_LegJoint_list:
        cmds.setAttr(human_set.nameSpace+":"+jnt+".rx", 0)
        cmds.setAttr(human_set.nameSpace+":"+jnt+".ry", 0)
        cmds.setAttr(human_set.nameSpace+":"+jnt+".rz", 0)













def export_HIKAnim_retarget(inst):

    exportAnim = cmds.fileDialog2(startingDirectory="/stdrepo/MCP/Data/01_CrowdMocap", fileMode=0, fileFilter="animExport (*.anim)", caption="Export Anim")

    if (exportAnim != None):

        initialSetup(inst)
        root_dif_pst = HIK_makeHierachy(inst, inst.nameSpace)


        HIK_ball_pst = cmds.xform(inst.nameSpace+":"+"RightToeBase", q=1, t=1, ws=1)

        minMaxValue = aniModule1.findMinMaxKey(inst.initPoseCtrl_list, inst.nameSpace)

        minCurrent = minMaxValue[0]
        maxCurrent = minMaxValue[1]

        skinJointList = [inst.connectSkinJoint_list, inst.L_SkinArmJoint_list, inst.R_SkinArmJoint_list]#, inst.fingerJoint_list]
        HIKJointList =  [inst.HIKJoint_list, inst.HIK_L_ArmJoint_list, inst.HIK_R_ArmJoint_list]#, inst.HIK_fingerJoint_list]


        HIKJoint_constraint_list = []

        for i in range(len(skinJointList)):
            for j in range(len(skinJointList[i])):
                if(cmds.objExists(inst.nameSpace+":"+skinJointList[i][j])):
                    temp_parent_con = cmds.parentConstraint("%s:%s" % (inst.nameSpace, skinJointList[i][j]),
                                                            "%s:%s" % (inst.nameSpace, HIKJointList[i][j]), mo=1, w=1)
                HIKJoint_constraint_list.append(temp_parent_con[0])

        cmds.select(cl=1)
        for ctrl in inst.initPoseCtrl_list:
            cmds.select(inst.nameSpace+":"+ctrl, add=1)

        cmds.cutKey(clear=1, t=(minCurrent,minCurrent))


        #cmds.select(cl=True)
        #cmds.select("%s:%s" % (inst.nameSpace, "Hips"), hi=True)

        #for jnt in inst.fingerJoint_list:
        #    if(cmds.objExists(inst.nameSpace+":"+jnt)):
        #        cmds.select(inst.nameSpace+":"+jnt, add=1)

        #cmds.bakeResults(cmds.ls(sl=True), simulation=True, t=(minCurrent, maxCurrent), sampleBy=1, dic=True, pok=True, sac=False, ral=False, bol=False, mr=True, controlPoints=False, shape=True)

        cmds.currentTime(minCurrent)

        #for jointList in HIKJointList:
        #    for HIKjoint in jointList:
        #        if(cmds.objExists(HIKjoint)):
        #            cmds.setAttr(HIKjoint+".rx", 0)
        #            cmds.setAttr(HIKjoint+".ry", 0)
        #            cmds.setAttr(HIKjoint+".rz", 0)


        # Human IK input setting
        cmds.file('/stdrepo/ANI/Asset/HumanIK_for_AnimBrowser//huType_hikJoint.ma', i=True,type='mayaAscii', ra=True, mergeNamespacesOnClash=True, namespace='mcp_hik', options='v=0')
        mel.eval('$gHIKCurrentCharacter = "%s:%s";' % ('mcp_hik', "Character1"))
        mel.eval('hikToggleLockDefinition();')
        mel.eval('refreshAllCharacterLists();')
        mel.eval('mayaHIKsetCharacterInput("%s:%s", "%s:%s");' % ('mcp_hik', "Character1", inst.nameSpace,"set_HIK"))
        cmds.setAttr("mcp_hik:HIKproperties1.CtrlPullLeftFoot", 1)
        cmds.setAttr("mcp_hik:HIKproperties1.CtrlPullRightFoot", 1)
        cmds.setAttr("mcp_hik:HIKproperties1.ForceActorSpace", 1)
        cmds.setAttr("mcp_hik:HIKproperties1.AnkleHeightCompensationMode", 0)



        # Bake HIK set
        cmds.select(cl=True)
        for jnt in inst.HIK_Joint_list_All:
            if(cmds.objExists("mcp_hik:Crw_"+jnt)):
                cmds.select("mcp_hik:Crw_"+jnt, add=1)


        cmds.bakeResults(cmds.ls(sl=True), simulation=True, t=(minCurrent, maxCurrent), sampleBy=1, dic=True, pok=True, sac=False, ral=False, bol=False, mr=True, controlPoints=False, shape=True)

        cmds.keyframe("mcp_hik:Crw_Hips.ty", e=1, iub=1, r=1, o='over', vc=(-1)*0.609)
        cmds.keyframe("mcp_hik:Crw_Hips.ty", e=1, iub=1, r=1, o='over', vc=HIK_ball_pst[1])
        cmds.filterCurve(f="euler")


        cmds.file(exportAnim, force=True, options="precision=8;nodeNames=1;verboseUnits=0;whichRange=1;options=keys;hierarchy=below;controlPoints=0;shapes=1;useChannelBox=0;copyKeyCmd=-animation objects -option keys -hierarchy below -controlPoints 0 -shape 1", typ="animExport", eas=True)

        cmds.delete("mcp_hik:mocap_char", (inst.nameSpace+":HIKJoint_LOC"))
        cmds.delete("mcp_hik:Character1")
        cmds.delete(inst.nameSpace+":set_HIK")








def export_HIKAnim(inst):

    exportAnim = cmds.fileDialog2(startingDirectory="/stdrepo/MCP/Data/01_CrowdMocap", fileMode=0, fileFilter="animExport (*.anim)", caption="Export Anim")

    if (exportAnim != None):

        initialSetup(inst)
        HIK_makeHierachy_withFinger(inst, inst.nameSpace)

        minMaxValue = aniModule1.findMinMaxKey(inst.initPoseCtrl_list, inst.nameSpace)

        minCurrent = minMaxValue[0]
        maxCurrent = minMaxValue[1]

        skinJointList = [inst.connectSkinJoint_list, inst.L_SkinArmJoint_list, inst.R_SkinArmJoint_list, inst.fingerJoint_list]
        HIKJointList =  [inst.HIKJoint_list, inst.HIK_L_ArmJoint_list, inst.HIK_R_ArmJoint_list, inst.HIK_fingerJoint_list]


        HIKJoint_constraint_list = []

        for i in range(len(skinJointList)):
            for j in range(len(skinJointList[i])):
                if(cmds.objExists(inst.nameSpace+":"+skinJointList[i][j])):
                    temp_parent_con = cmds.parentConstraint("%s:%s" % (inst.nameSpace, skinJointList[i][j]),
                                                            "%s:%s" % (inst.nameSpace, HIKJointList[i][j]), mo=1, w=1)
                HIKJoint_constraint_list.append(temp_parent_con[0])


        cmds.select(cl=1)
        for ctrl in inst.initPoseCtrl_list:
            cmds.select(inst.nameSpace+":"+ctrl, add=1)

        cmds.cutKey(clear=1, t=(minCurrent,minCurrent))

        cmds.select(cl=True)
        cmds.select("%s:%s" % (inst.nameSpace, "Hips"), hi=True)

        for jnt in inst.fingerJoint_list:
            if(cmds.objExists(inst.nameSpace+":"+jnt)):
                cmds.select(inst.nameSpace+":"+jnt, add=1)

        cmds.bakeResults(cmds.ls(sl=True), simulation=True, t=(minCurrent, maxCurrent), sampleBy=1, dic=True, pok=True, sac=False, ral=False, bol=False, mr=True, controlPoints=False, shape=True)
        cmds.playbackOptions( minTime=minCurrent, maxTime=maxCurrent )
        cmds.filterCurve(f="euler")
        cmds.delete(HIKJoint_constraint_list)

        cmds.select(cl=True)
        cmds.select("%s:%s" % (inst.nameSpace, "Hips"), hi=True)

        for jnt in inst.fingerJoint_list:
            if(cmds.objExists(inst.nameSpace+":"+jnt)):
                cmds.select(inst.nameSpace+":"+jnt, add=1)

        cmds.file(exportAnim, force=True, options="precision=8;nodeNames=1;verboseUnits=0;whichRange=1;options=keys;hierarchy=below;controlPoints=0;shapes=1;useChannelBox=0;copyKeyCmd=-animation objects -option keys -hierarchy below -controlPoints 0 -shape 1", typ="animExport", eas=True)

        cmds.delete(inst.nameSpace+":HIKJoint_LOC")
        cmds.delete(inst.nameSpace+":set_HIK")







def importAnim(inst,nameSpace):

    scrP = "/stdrepo/ANI/Library/Mocap_Library/02_Data/02_CharMocap"
    animDir = str(cmds.fileDialog2(fileMode=1, fileFilter="animExport (*.anim)", dir=scrP, caption="Import Anim File")[0])

    if animDir:
        HIK_makeHierachy_withFinger(inst, nameSpace)

        if cmds.ls(type='transform').count("%s:mocap_char" % nameSpace + "_mcp") == 1:
            cmds.delete("%s:mocap_char" % nameSpace + "_mcp")
        cmds.file('/stdrepo/ANI/Asset/HumanIK_for_AnimBrowser//crdJoint.mb', i=True, type='mayaBinary', ra=True, mergeNamespacesOnClash=True, namespace=nameSpace, options='v=0')


        crdJoint_constraint_list = []
        for u in range(len(inst.crd_BodyJoint_list)):
            temp_point_con = cmds.pointConstraint("%s:%s" % (nameSpace, inst.SkinBodyJoint_list[u]),
                                                  "%s:%s" % (nameSpace, inst.crd_BodyJoint_list[u]), mo=0, w=1)
            crdJoint_constraint_list.append(temp_point_con[0])
        cmds.delete(crdJoint_constraint_list)

        L_arm_aimVec = [1,0,0]
        L_arm_upVec = [0,1,0]
        R_arm_aimVec = [-1,0,0]
        R_arm_upVec = [0,-1,0]

        crd_jointArrange(nameSpace, inst.L_SkinArmJoint_list, inst.crd_L_ArmJoint_list,L_arm_aimVec,L_arm_upVec)
        crd_jointArrange(nameSpace, inst.R_SkinArmJoint_list, inst.crd_R_ArmJoint_list,R_arm_aimVec,R_arm_upVec)

        L_leg_aimVec = [1,0,0]
        L_leg_upVec = [0,0,1]
        R_leg_aimVec = [-1,0,0]
        R_leg_upVec = [0,0,-1]
        crd_jointArrange(nameSpace, inst.L_SkinLegJoint_list, inst.crd_L_LegJoint_list,L_leg_aimVec,L_leg_upVec)
        crd_jointArrange(nameSpace, inst.R_SkinLegJoint_list, inst.crd_R_LegJoint_list,R_leg_aimVec,R_leg_upVec)


        for i in range(len(inst.crdJnt_constraint)):
            cmds.parentConstraint(nameSpace+":"+inst.crdJnt_constraint[i], nameSpace+":"+inst.skinJnt_constraint[i], mo=1)

    else:
        return

    HIKJoint_constraint_list = []
    for u in range(len(inst.HIK_Joint_list_All)):
        temp_parent_con = cmds.parentConstraint("%s:%s" % (nameSpace, inst.HIK_Joint_list_All[u]),
                                                "%s:%s" % (nameSpace, inst.crd_Joint_list_All[u]), mo=1, w=1)
        HIKJoint_constraint_list.append(temp_parent_con[0])
    #cmds.delete(HIKJoint_constraint_list)

    HIKfinger_constraint_list = []
    for u in range(len(inst.fingerJoint_list)):
        if(cmds.objExists(nameSpace+":"+inst.fingerJoint_list[u])):
            temp_parent_con = cmds.parentConstraint("%s:%s" % (nameSpace, inst.HIK_fingerJoint_list[u]),
                                                    "%s:%s" % (nameSpace, inst.fingerJoint_list[u]), mo=1, w=1)
            HIKfinger_constraint_list.append(temp_parent_con[0])

    cmds.select(inst.nameSpace+":"+"Hips", r=1)
    cmds.file(animDir, i=True, applyTo=inst.nameSpace+":"+"Hips")

    minMaxValue = aniModule1.findMinMaxKey(inst.HIK_Joint_list_All, nameSpace)

    minCurrent = minMaxValue[0]
    maxCurrent = minMaxValue[1]


    ## bake ##
    cmds.select(cl=True)
    #cmds.select("%s:%s" % (inst.nameSpace, "Crw_Hips"), hi=True)
    cmds.select("%s:%s" % (inst.nameSpace, "C_Skin_hip_JNT"), hi=True)
    cmds.bakeResults(cmds.ls(sl=True), simulation=True, t=(minCurrent, maxCurrent), sampleBy=1, dic=True, pok=True, sac=False, ral=False, bol=False, mr=True, controlPoints=False, shape=True)

    cmds.playbackOptions( minTime=minCurrent, maxTime=maxCurrent )
    cmds.filterCurve(f="euler")


    cmds.delete(HIKJoint_constraint_list)
    cmds.delete(HIKfinger_constraint_list)
    cmds.delete(nameSpace+":HIKJoint_LOC")
    cmds.delete(nameSpace+":crowd_char")
    cmds.delete(nameSpace+":set_HIK")











def import_HIKAnim_retarget(inst, nameSpace):

    scrP = "/stdrepo/ANI/Library/Mocap_Library/02_Data/02_CharMocap"
    animDir = str(cmds.fileDialog2(fileMode=1, fileFilter="animExport (*.anim)", dir=scrP, caption="Import Anim File")[0])

    if animDir:
        HIK_makeHierachy(inst, nameSpace)

        cmds.file('/stdrepo/ANI/Asset/HumanIK_for_AnimBrowser//huType_hikJoint.ma', i=True,type='mayaAscii', ra=True, mergeNamespacesOnClash=True, namespace='mcp_hik', options='v=0')

        cmds.select("mcp_hik:Crw_Hips", r=1)
        cmds.file(animDir, i=True, applyTo="mcp_hik:Crw_Hips")


        skinJointList = [inst.connectSkinJoint_list, inst.L_SkinArmJoint_list, inst.R_SkinArmJoint_list]#, inst.fingerJoint_list]
        HIKJointList =  [inst.HIKJoint_list, inst.HIK_L_ArmJoint_list, inst.HIK_R_ArmJoint_list]#, inst.HIK_fingerJoint_list]

        HIKJoint_constraint_list = []

        for i in range(len(skinJointList)):
            for j in range(len(skinJointList[i])):
                if(cmds.objExists(inst.nameSpace+":"+skinJointList[i][j])):
                    temp_parent_con = cmds.parentConstraint("%s:%s" % (inst.nameSpace, HIKJointList[i][j]),
                                                            "%s:%s" % (inst.nameSpace, skinJointList[i][j]), mo=1, w=1)
                HIKJoint_constraint_list.append(temp_parent_con[0])

        for i in range(len(HIKJointList)):
            for j in range(len(HIKJointList[i])):
                if(cmds.objExists(nameSpace+":"+HIKJointList[i][j])):
                    cmds.setAttr(nameSpace+":"+HIKJointList[i][j]+".rx", 0)
                    cmds.setAttr(nameSpace+":"+HIKJointList[i][j]+".ry", 0)
                    cmds.setAttr(nameSpace+":"+HIKJointList[i][j]+".rz", 0)



        #cmds.file('/stdrepo/ANI/Asset/HumanIK_for_AnimBrowser//huType_hikJoint.ma', i=True,type='mayaAscii', ra=True, mergeNamespacesOnClash=True, namespace='mcp_hik', options='v=0')
        mel.eval('$gHIKCurrentCharacter = "%s:%s";' % (nameSpace,"set_HIK"))
        mel.eval('hikToggleLockDefinition();')
        mel.eval('refreshAllCharacterLists();')
        mel.eval('mayaHIKsetCharacterInput("%s:%s", "%s:%s");' % (nameSpace,"set_HIK",'mcp_hik', "Character1"))
        #cmds.setAttr(nameSpace+":HIKproperties1.CtrlPullLeftFoot", 1)
        #cmds.setAttr(nameSpace+":HIKproperties1.CtrlPullRightFoot", 1)
        cmds.setAttr(nameSpace+":HIKproperties1.ForceActorSpace", 1)
        cmds.setAttr(nameSpace+":HIKproperties1.AnkleHeightCompensationMode", 0)


        minMaxValue = aniModule1.findMinMaxKey(inst.crd_Joint_list_All, 'mcp_hik')

        minCurrent = minMaxValue[0]
        maxCurrent = minMaxValue[1]


        cmds.select(cl=1)

        for i in range(len(skinJointList)):
            for j in range(len(skinJointList[i])):
                if(cmds.objExists(inst.nameSpace+":"+skinJointList[i][j])):

                    cmds.select(inst.nameSpace+":"+skinJointList[i][j], add=1)


        cmds.bakeResults(cmds.ls(sl=True), simulation=True, t=(minCurrent, maxCurrent), sampleBy=1, dic=True, pok=True, sac=False, ral=False, bol=False, mr=True, controlPoints=False, shape=True)

        cmds.playbackOptions( minTime=minCurrent, maxTime=maxCurrent )
        cmds.filterCurve(f="euler")

        cmds.delete(HIKJoint_constraint_list)
        cmds.delete('mcp_hik:mocap_char')
        cmds.delete(nameSpace+":HIKJoint_LOC")
        cmds.delete("mcp_hik:Character1")
        cmds.delete(nameSpace+":set_HIK")


    else:
        return










def rebuild_CRD_human(*argv):

    human_set = humanRigClass.human_rig()
    nameSpace = human_set.nameSpace
    deleteRig(nameSpace, human_set.delRIG_list,human_set.skinJNT_list)
    #CRD_makeHierachy(human_set, nameSpace)



def CRD_exportAnim(*argv):

    retarget_CBox = cmds.checkBoxGrp("retargetCheckBox", q=1, v1=1)
    human_set = humanRigClass.human_rig()
    nameSpace = human_set.nameSpace

    if(retarget_CBox == 0):
        export_HIKAnim(human_set)

    else:
        export_HIKAnim_retarget(human_set)




def rebuild_MCP_human(*argv):

    human_set = humanRigClass.human_rig()
    nameSpace = human_set.nameSpace
    deleteRig(nameSpace, human_set.delRIG_list,human_set.skinJNT_list)
    MCP_makeHierachy(human_set,nameSpace)



def rebuild_MCP_humanHind(*argv):

    human_set = humanRigClass.human_rig()
    nameSpace = human_set.nameSpace
    deleteRig(nameSpace, human_set.delRIG_list,human_set.skinJNT_list)
    MCP_makeHierachy_HindLeg(human_set, nameSpace)




def makeSkinHierachy_CRD_human(*argv):

    human_set = humanRigClass.human_rig()
    nameSpace = human_set.nameSpace
    CRD_makeHierachy_skinJoint(nameSpace, human_set.skinJNT_hierachy)



def CRD_importAnim(*argv):

    retarget_CBox = cmds.checkBoxGrp("retargetCheckBox", q=1, v1=1)
    human_set = humanRigClass.human_rig()
    nameSpace = human_set.nameSpace

    if(retarget_CBox == 0):
        importAnim(human_set, nameSpace)

    else:
        import_HIKAnim_retarget(human_set, nameSpace)

    '''
    crdRootHight = cmds.getAttr(human_set.nameSpace+":Crw_Hips.ty")
    scaleRatio = human_set.crdBaseJnt_hight / crdRootHight
    currentScale = cmds.getAttr(human_set.nameSpace+":Crw_Hips.sx")

    scaleValue = currentScale * scaleRatio

    cmds.setAttr(human_set.nameSpace+":crowd_char.sx", scaleValue)
    cmds.setAttr(human_set.nameSpace+":crowd_char.sy", scaleValue)
    cmds.setAttr(human_set.nameSpace+":crowd_char.sz", scaleValue)
    '''


######################
def initialSetup(inst):

    minMaxValue = aniModule1.findMinMaxKey(inst.initPoseCtrl_list, inst.nameSpace)
    cmds.currentTime(minMaxValue[0])
    aniModule1.setKeyAll(inst.nameSpace, inst.initPoseCtrl_list)
    initKeyframe = minMaxValue[0] - 50
    cmds.currentTime(initKeyframe)
    aniModule1.makeInitPose(inst.nameSpace, inst.initPoseCtrl_list, inst.initPoseAtt_list)
    aniModule1.setKeyAll(inst.nameSpace, inst.initPoseCtrl_list)




def makeUI():

    if (cmds.window('mocapCrowd_charSetUp', q=1, ex=1)):
        cmds.deleteUI('mocapCrowd_charSetUp', window=True)

    windowUI = cmds.window('mocapCrowd_charSetUp', t="MC rig   <20211021>", s=0, widthHeight=(454,250))
    cmds.window(windowUI, edit=1, rtf=1, widthHeight=(454,555))

    cmds.columnLayout("frameLayout_column", w=465, h=200, ebg=1, cat=("left",10), adjustableColumn=False, io=1)
    cmds.separator(h=10)
    cmds.frameLayout("MPC_HM_frame",l="Motion Capture",w=445, h=71, bv=1, bgs=1)
    cmds.rowColumnLayout("MPC_human_column",w=445,h=50,numberOfRows=1, rowHeight=[(1,46)], cs=(1,1))
    cmds.button("MPC_human_button", w=110, l='Human IK', command=rebuild_MCP_human)
    cmds.button("MPC_humanHind_button", w=110, l='HIK (HindLeg)', command=rebuild_MCP_humanHind)
    cmds.button("T_pose_button", w=220, l='Make T-Pose', command=HIK_T_pose)
    cmds.setParent('..')
    cmds.setParent('..')

    cmds.separator(h=20)

    cmds.frameLayout("CRD_human_frame", l="Crowd",w=445, h=100, bv=1, bgs=1)
    cmds.rowColumnLayout("CRD_retargetCB_column",w=445,h=50, numberOfRows=2, rowHeight=[(30,46)], cs=(1,1))
    cmds.checkBoxGrp("retargetCheckBox", cal=[1,"center"], rat=[1,"both",10], numberOfCheckBoxes=1, label='HIK retargeting', v1=1) #retargetCheckBox)
    cmds.setParent('..')

    cmds.rowColumnLayout("CRD_human_column",w=445,h=50, numberOfRows=1, rowHeight=[(30,46)], cs=(1,1))
    cmds.button("CRD_human_button", w=110, l='Delete RIG', command=rebuild_CRD_human)
    cmds.button("CRD_make_hierachy", w=110, l='Make Hierachy', command=makeSkinHierachy_CRD_human)
    cmds.button("CRD_exportAnim_button", w=109, l='export Anim', command=CRD_exportAnim)
    cmds.button("CRD_importAnim_button", w=109, l='import Anim', command=CRD_importAnim)
    cmds.setParent('..')
    cmds.setParent('..')

    cmds.showWindow(windowUI)
