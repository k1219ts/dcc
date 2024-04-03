import sys
import maya.cmds as cmds
import maya.mel as mel


sys.path.append("/stdrepo/ANI/Library/Script/Class/HumanRig")
import humanRigClass
reload(humanRigClass)

sys.path.append("/stdrepo/ANI/Library/Script/Module")
import aniModule1
reload(aniModule1)
 



def retargetEXE(*argv):
    
    selected_char = cmds.ls(sl=1)

    cmds.select(selected_char[0], r=1)
    sourceHuman = humanRigClass.human_rig()

    cmds.select(selected_char[1], r=1)
    targetHuman = humanRigClass.human_rig()


    HIK_retarget(sourceHuman, targetHuman)





def HIK_retarget(source, target):


    # get world position
    targetPosition = cmds.xform(target.nameSpace+":place_CON", q=1, ws=1, t=1)
    sourcePosition = cmds.xform(source.nameSpace+":place_CON", q=1, ws=1, t=1)
    

    # source charcter setting ############################
    initialSetup(source)
  
    root_dif_pst = HIK_makeHierachy(source, source.nameSpace)

    HIK_ball_pst = cmds.xform(source.nameSpace+":"+"RightToeBase", q=1, t=1, ws=1)

    minMaxValue = aniModule1.findMinMaxKey(source.initPoseCtrl_list, source.nameSpace)

    minCurrent = minMaxValue[0]
    maxCurrent = minMaxValue[1]

    skinJointList = [source.connectSkinJoint_list, source.L_SkinArmJoint_list, source.R_SkinArmJoint_list]#, source.fingerJoint_list]
    HIKJointList =  [source.HIKJoint_list, source.HIK_L_ArmJoint_list, source.HIK_R_ArmJoint_list]#, source.HIK_fingerJoint_list]


    HIKJoint_constraint_list = []

    for i in range(len(skinJointList)):
        for j in range(len(skinJointList[i])):
            if(cmds.objExists(source.nameSpace+":"+skinJointList[i][j])):
                temp_parent_con = cmds.parentConstraint("%s:%s" % (source.nameSpace, skinJointList[i][j]),
                                                        "%s:%s" % (source.nameSpace, HIKJointList[i][j]), mo=1, w=1)
            HIKJoint_constraint_list.append(temp_parent_con[0])

    cmds.select(cl=1)
    for ctrl in source.initPoseCtrl_list:
        cmds.select(source.nameSpace+":"+ctrl, add=1)

    # delete rest pose ###
    cmds.cutKey(clear=1, t=(minCurrent,minCurrent))



    # target charcter setting ##############################
    cmds.setAttr(target.nameSpace+":place_CON.tx", sourcePosition[0])
    cmds.setAttr(target.nameSpace+":place_CON.ty", sourcePosition[1])
    cmds.setAttr(target.nameSpace+":place_CON.tz", sourcePosition[2])

    HIK_makeHierachy(target, target.nameSpace)
    constraint_targetRigCtrl(target, target.nameSpace)
    HIK_T_pose(target)

    cmds.currentTime(minCurrent)


    # retargeting ###########################################
    mel.eval('$gHIKCurrentCharacter = "%s:%s";' % (target.nameSpace, "set_HIK"))
    mel.eval('hikToggleLockDefinition();')
    mel.eval('refreshAllCharacterLists();')
    mel.eval('mayaHIKsetCharacterInput("%s:%s", "%s:%s");' % (target.nameSpace, "set_HIK", source.nameSpace,"set_HIK"))
    cmds.setAttr(target.nameSpace+":HIKproperties1.CtrlPullLeftFoot", 1)
    cmds.setAttr(target.nameSpace+":HIKproperties1.CtrlPullRightFoot", 1)
    cmds.setAttr(target.nameSpace+":HIKproperties1.ForceActorSpace", 1)
    cmds.setAttr(target.nameSpace+":HIKproperties1.AnkleHeightCompensationMode", 0)



    IK_blend = cmds.checkBox("IK_Blend", q=1, v=1)
    
    if(IK_blend == 0):
        cmds.setAttr(target.nameSpace + ":HIKproperties1.ReachActorRightAnkle", 0)
        cmds.setAttr(target.nameSpace + ":HIKproperties1.ReachActorLeftAnkle", 0)
        cmds.setAttr(target.nameSpace + ":HIKproperties1.ReachActorRightAnkleRotation", 0)
        cmds.setAttr(target.nameSpace + ":HIKproperties1.ReachActorLeftAnkleRotationRotation", 0)


    cmds.select(cl=True)
    minMaxSource = aniModule1.findMinMaxKey(source.initPoseCtrl_list, source.nameSpace)
    minSource = minMaxSource[0]
    maxSource = minMaxSource[1]

    print minSource, maxSource


    # bake RigControler of target character
    cmds.select(cl=True)
    for ctrl in target.initPoseCtrl_list:
        if(cmds.objExists(target.nameSpace+":"+ctrl)):
            cmds.select(target.nameSpace+":"+ctrl, add=1)

    cmds.select(target.nameSpace+":place_CON", d=1)
    cmds.select(target.nameSpace+":direction_CON", d=1)
    cmds.select(target.nameSpace+":move_CON", d=1)


    cmds.bakeResults(cmds.ls(sl=True), simulation=True, t=(minSource, maxSource), sampleBy=1, dic=True, pok=True, sac=False, ral=False, bol=False, mr=True, controlPoints=False, shape=True)





    # move to the original position 
    cmds.setAttr(target.nameSpace+":place_CON.tx", targetPosition[0])
    cmds.setAttr(target.nameSpace+":place_CON.ty", targetPosition[1])
    cmds.setAttr(target.nameSpace+":place_CON.tz", targetPosition[2])


    #delete humanIK sets
    cmds.delete(source.nameSpace+":HIKJoint_LOC", target.nameSpace+":HIKJoint_LOC")
    cmds.delete(source.nameSpace+":set_HIK", target.nameSpace+":set_HIK")



    #Move the key of the finger controller
    att_list = ['rx','ry','rz']
    for Source_Ctrl in source.fingerControler_list:
        for att in att_list:
            if(cmds.objExists(source.nameSpace+":"+Source_Ctrl)):
                transformKey(source.nameSpace+":"+Source_Ctrl, target.nameSpace+":"+Source_Ctrl, att)






def transformKey(sourceCtrl,targetCtrl,att):

    cmds.select(sourceCtrl, r=1)
    cmds.selectKey()
    keyPosition = cmds.keyframe(sourceCtrl, at=att, sl=1, q=1, tc=1)
    if keyPosition != None:
        cmds.copyKey(sourceCtrl, attribute=att, option="curve")
        cmds.pasteKey(targetCtrl, attribute=att, option="replace")

    else:
        cmds.warning("Key does not exist on selected object(controler)")



def HIK_T_pose(inst): 

    for jnt in inst.HIK_L_ArmJoint_list: 
        cmds.setAttr(inst.nameSpace+":"+jnt+".rx", 0) 
        cmds.setAttr(inst.nameSpace+":"+jnt+".ry", 0)
        cmds.setAttr(inst.nameSpace+":"+jnt+".rz", 0)

    for jnt in inst.HIK_R_ArmJoint_list:
        cmds.setAttr(inst.nameSpace+":"+jnt+".rx", 0)
        cmds.setAttr(inst.nameSpace+":"+jnt+".ry", 0)
        cmds.setAttr(inst.nameSpace+":"+jnt+".rz", 0)





def constraint_targetRigCtrl(inst, nameSpace):
    
    print inst.const_list
    print inst.pole_list


    for i in range(len(inst.const_list)):
        jointNam =  nameSpace + ":" + inst.const_list[i][0]
        ctrlNam = nameSpace + ":" + inst.const_list[i][1]
        cmds.parentConstraint(jointNam, ctrlNam, mo=1)

    for j in range(len(inst.pole_list)):
        poleJnt =  nameSpace + ":" + inst.pole_list[j][0]
        poleCtrl = nameSpace + ":" + inst.pole_list[j][1]
        cmds.pointConstraint(poleJnt, poleCtrl, mo=1)  






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





def HIK_armJointArrange(nameSpace,skinJointList, HIKjointList, aim_vec):

    for i in range(len(HIKjointList)):
        temp_point_con = cmds.pointConstraint("%s:%s" % (nameSpace, skinJointList[i]),
                                              "%s:%s" % (nameSpace, HIKjointList[i]), mo=0, w=1)
        cmds.delete(temp_point_con[0])

        if(i<len(HIKjointList)-1):
            temp_aim_con =  cmds.aimConstraint("%s:%s" % (nameSpace, skinJointList[i+1]),
                                               "%s:%s" % (nameSpace, HIKjointList[i]), aim=aim_vec, w=1)   
            cmds.delete(temp_aim_con[0])




def initialSetup(inst):


    minMaxValue = aniModule1.findMinMaxKey(inst.initPoseCtrl_list, inst.nameSpace)
    cmds.currentTime(minMaxValue[0])
    aniModule1.setKeyAll(inst.nameSpace, inst.initPoseCtrl_list)
    initKeyframe = minMaxValue[0] - 50
    cmds.currentTime(initKeyframe)
    aniModule1.makeInitPose(inst.nameSpace, inst.initPoseCtrl_list, inst.initPoseAtt_list)
    aniModule1.setKeyAll(inst.nameSpace, inst.initPoseCtrl_list)




def makeUI():

    if (cmds.window('humanRetargetWin', q=1, ex=1)):
        cmds.deleteUI('humanRetargetWin', window=True)

    windowUI = cmds.window('humanRetargetWin', t="Retargeter", s=0, widthHeight=(454,250))
    cmds.window(windowUI, edit=1, rtf=1, widthHeight=(454,555))

    cmds.columnLayout("frameLayout_column", w=465, h=500, ebg=1, cat=("left",10), adjustableColumn=False, io=1)
    cmds.separator(h=10)
    cmds.frameLayout("h_RetargetFrame",l="Biped",w=445, h=125, bv=1, bgs=1)
    cmds.rowColumnLayout("h_RetargetColumn",w=445,h=350,numberOfRows=1, rowHeight=[(1,100)], cs=(1,1))
    cmds.button("hik_RetargetButton", w=220, l='Human IK', command=retargetEXE)
    cmds.button("crs_RetargetButton", w=220, l='Curve Rescale', en=0, command='')
    #cmds.button("T_pose_button", w=220, l='Make T-Pose', command='')
    cmds.setParent('..')    
    cmds.setParent('..')



    '''
    cmds.separator(h=20)

    cmds.frameLayout("CRD_human_frame", l="Quadruped",w=445, h=100, bv=1, bgs=1)
    cmds.rowColumnLayout("CRD_retargetCB_column",w=445,h=50, numberOfRows=2, rowHeight=[(30,46)], cs=(1,1))
    cmds.checkBoxGrp("retargetCheckBox", cal=[1,"center"], rat=[1,"both",10], numberOfCheckBoxes=1, label='HIK retargeting', v1=1) #retargetCheckBox)
    cmds.setParent('..') 

    cmds.rowColumnLayout("CRD_human_column",w=445,h=50, numberOfRows=1, rowHeight=[(30,46)], cs=(1,1))
    cmds.button("CRD_human_button", w=110, l='Delete RIG', command='')
    cmds.button("CRD_make_hierachy", w=110, l='Make Hierachy', command='')
    cmds.button("CRD_exportAnim_button", w=109, l='export Anim', command='')
    cmds.button("CRD_importAnim_button", w=109, l='import Anim', command='')
    cmds.setParent('..') 
    cmds.setParent('..') 
    '''


    cmds.showWindow(windowUI)
    
    
    
#makeUI()    
    
    