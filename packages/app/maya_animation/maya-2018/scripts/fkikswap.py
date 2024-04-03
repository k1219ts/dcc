# coding=utf-8

import maya.cmds as cmds

IKjnt=['IK_upArm_JNT', 'IK_foreArm_JNT', 'IK_hand_JNT'] 
FKjnt=['FK_upArm_JNT', 'FK_foreArm_JNT', 'FK_hand_JNT']
IKctrl=['IK_hand_CON', 'IK_handSub_CON', 'IK_handVec_CON']
FKctrl=['FK_upArm_CON', 'FK_foreArm_CON', 'FK_hand_CON']
IKlegjnt=['IK_leg_JNT', 'IK_lowLeg_JNT', 'IK_foot_JNT']
FKlegjnt=['FK_leg_JNT', 'FK_lowLeg_JNT', 'FK_foot_JNT']
IKlegctrl=['IK_foot_CON', 'IK_footSub_CON', 'IK_footVec_CON']
FKlegctrl=['FK_leg_CON', 'FK_lowLeg_CON', 'FK_foot_CON']    
############################################################################################################################################################################################################################################
############################################################################################################################################################################################################################################
#Tangent switch
############################################################################################################################################################################################################################################
############################################################################################################################################################################################################################################
def armIKFK_Tan():
    
    IKjnt=['IK_upArm_JNT', 'IK_foreArm_JNT', 'IK_hand_JNT'] 
    FKjnt=['FK_upArm_JNT', 'FK_foreArm_JNT', 'FK_hand_JNT']
    IKctrl=['IK_hand_CON', 'IK_handSub_CON', 'IK_handVec_CON']
    FKctrl=['FK_upArm_CON', 'FK_foreArm_CON', 'FK_hand_CON']
    
    sel=cmds.ls(sl=1)[0]
    nameSpace=sel.split(":")[0]+":"
    
    obj=sel.split(":")[1]
    dir=obj[0:2]
               
    snap_Tan(IKjnt,FKjnt,IKctrl,FKctrl,nameSpace, dir)

    objnames = [nameSpace+dir+FKctrl[0], nameSpace+dir+FKctrl[1], nameSpace+dir+FKctrl[2], nameSpace+dir+IKctrl[0], nameSpace+dir+IKctrl[1], nameSpace+dir+IKctrl[2]]
    legObjnames = [nameSpace+dir+FKlegctrl[0], nameSpace+dir+FKlegctrl[1], nameSpace+dir+FKlegctrl[2], nameSpace+dir+IKlegctrl[0], nameSpace+dir+IKlegctrl[1], nameSpace+dir+IKlegctrl[2]]

    for objname in objnames:
    
        rotation = cmds.xform(objname, q=1, rotation=1)

        if any(abs(angle) > 180 for angle in rotation):
            nor = [(angle + 180) % 360 - 180 for angle in rotation]
    
            cmds.xform(objname, rotation=nor)
       
        new_rot = cmds.xform(objname, q=1, rotation=1)
    
    armEuler(IKjnt, FKjnt, IKctrl, FKctrl, nameSpace, dir)
     
def snap_Tan(IKjnt, FKjnt, IKctrl, FKctrl, ns, dir):
    
    IKFKblendCtrl = ns+dir+"armBlend_CON"
    IKFKstate = cmds.getAttr(IKFKblendCtrl+'.FKIKBlend')
    current_time = cmds.currentTime(q=1)
    
            
   
    if IKFKstate == 1:
        lastKeyframeIK = None
        for i in range(len(IKjnt)):
            ctrlName = ns+dir+FKctrl[i]
            IKctrlName = ns+dir+IKctrl[i]
            
            
            
            keyframe_times = cmds.keyframe(IKctrlName, q=1, timeChange=1)
            
            if keyframe_times:
                for keyframe_time in keyframe_times:
                    if keyframe_time < current_time:
                        lastKeyframeIK = max(lastKeyframeIK, keyframe_time)
        
        if lastKeyframeIK is not None:
            cmds.setKeyframe(IKFKblendCtrl, attribute='FKIKBlend', time=lastKeyframeIK)
            cmds.keyTangent(IKFKblendCtrl, at='FKIKBlend', inTangentType='stepnext', outTangentType='step') 
            
        for i in range(len(IKjnt)):
            ctrlName = ns+dir+FKctrl[i]
            IKctrlName = ns+dir+IKctrl[i]
                
            cmds.setKeyframe(IKctrlName, at='translate')
            cmds.setKeyframe(IKctrlName, at='rotate')
            cmds.keyTangent(IKctrlName, time=(current_time, current_time), lock = False, outTangentType='stepnext')
            
            const=cmds.orientConstraint(ns+dir+IKjnt[i], ctrlName)
            cmds.setKeyframe(ctrlName, at='rotate')
            cmds.delete(const)            
  
        cmds.setAttr(IKFKblendCtrl+'.FKIKBlend', 0)
            
    elif IKFKstate == 0:
        lastKeyframeFK = None
        for i in range(len(FKjnt)):
            FKctrlName = ns+dir+FKctrl[i]
            keyframe_times = cmds.keyframe(FKctrlName, q=1, timeChange=1)
            
            if keyframe_times:
                for keyframe_time in keyframe_times:
                    if keyframe_time < current_time:
                        lastKeyframeFK = max(lastKeyframeFK, keyframe_time)
                
        if lastKeyframeFK is not None:
            cmds.setKeyframe(IKFKblendCtrl, attribute='FKIKBlend', time=lastKeyframeFK)
            cmds.keyTangent(IKFKblendCtrl, at='FKIKBlend', inTangentType='stepnext', outTangentType='step') 
            
        for i in range(len(FKjnt)):
            ctrlName = ns+dir+FKctrl[i]
            
            FKhand = ns+dir+FKjnt[2]
            FKforeArm = ns+dir+FKjnt[1]
         
                        
            IKarm = ns+dir+IKctrl[0]
            IKsubHand = ns+dir+IKctrl[1]
            armPole = ns+dir+IKctrl[2]
            
            
            cmds.setKeyframe(ctrlName, at='rotate')
            cmds.keyTangent(ctrlName, time=(current_time, current_time), lock = False, outTangentType='stepnext')
            
            const1 = cmds.pointConstraint(FKhand, IKarm)
            const2 = cmds.orientConstraint(FKhand, IKsubHand)
            const3 = cmds.pointConstraint(FKforeArm, armPole)
            cmds.setKeyframe(IKarm, at='translate')
            cmds.setKeyframe(IKarm, at='rotate')
            cmds.setKeyframe(IKsubHand, at='rotate')
            cmds.setKeyframe(armPole, at='translate')
            cmds.delete(const1, const2, const3)
        
                
            cmds.setAttr(IKFKblendCtrl+'.FKIKBlend', 1)   
    
    
    
    cmds.setKeyframe(IKFKblendCtrl, at='FKIKBlend')
    cmds.keyTangent(IKFKblendCtrl, at='FKIKBlend', inTangentType='stepnext', outTangentType='step') 
    

def armEuler(IKjnt, FKjnt, IKctrl, FKctrl, nameSpace, dir):
    
    IKctrl=['IK_hand_CON', 'IK_handSub_CON', 'IK_handVec_CON']
    FKctrl=['FK_upArm_CON', 'FK_foreArm_CON', 'FK_hand_CON']

    sel=cmds.ls(sl=1)[0]
    nameSpace=sel.split(":")[0]+":"

    obj=sel.split(":")[1]
    dir=obj[0:2]
    

    objnames = [nameSpace+dir+FKctrl[0], nameSpace+dir+FKctrl[1], nameSpace+dir+FKctrl[2], nameSpace+dir+IKctrl[0], nameSpace+dir+IKctrl[1], nameSpace+dir+IKctrl[2]]
    cmds.filterCurve(objnames)
    
#############################################################################################################
#Tangent Leg 
#############################################################################################################
def legIKFK_Tan():
    
    IKlegjnt=['IK_leg_JNT', 'IK_lowLeg_JNT', 'IK_foot_JNT']
    FKlegjnt=['FK_leg_JNT', 'FK_lowLeg_JNT', 'FK_foot_JNT']
    IKlegctrl=['IK_foot_CON', 'IK_footSub_CON', 'IK_footVec_CON']
    FKlegctrl=['FK_leg_CON', 'FK_lowLeg_CON', 'FK_foot_CON']
    
    sel=cmds.ls(sl=1)[0]
    nameSpace=sel.split(":")[0]+":"
    
    obj=sel.split(":")[1]
    dir=obj[0:2]
    
    snapLeg_Tan(IKlegjnt,FKlegjnt,IKlegctrl,FKlegctrl,nameSpace, dir)

    objnames = [nameSpace+dir+FKlegctrl[0], nameSpace+dir+FKlegctrl[1], nameSpace+dir+FKlegctrl[2], nameSpace+dir+IKlegctrl[0], nameSpace+dir+IKlegctrl[1], nameSpace+dir+IKlegctrl[2]]

    for objname in objnames:
    
        rotation = cmds.xform(objname, q=1, rotation=1)

        if any(abs(angle) > 180 for angle in rotation):
            nor = [(angle + 180) % 360 - 180 for angle in rotation]
    
            cmds.xform(objname, rotation=nor)
        new_rot = cmds.xform(objname, q=1, rotation=1)

    legEuler(IKjnt, FKjnt, IKctrl, FKctrl, nameSpace, dir)
         
def snapLeg_Tan(IKlegjnt, FKlegjnt, IKlegctrl, FKlegctrl, ns, dir):

    IKFKblendCtrl = ns+dir+"legBlend_CON"
    IKFKstate = cmds.getAttr(IKFKblendCtrl+'.FKIKBlend')
    current_time = cmds.currentTime(q=1)
       
    if IKFKstate == 1:
        lastKeyframeIK = None
        for i in range(len(IKjnt)):
            ctrlName = ns + dir + FKlegctrl[i]
            IKctrlName = ns + dir + IKlegctrl[i]
            
            keyframe_times = cmds.keyframe(IKctrlName, q=1, timeChange=1)
            
            if keyframe_times:
                for keyframe_time in keyframe_times:
                    if keyframe_time < current_time:
                        lastKeyframeIK = max(lastKeyframeIK, keyframe_time)
        
        if lastKeyframeIK is not None:
            cmds.setKeyframe(IKFKblendCtrl, attribute='FKIKBlend', time=lastKeyframeIK)
            cmds.keyTangent(IKFKblendCtrl, at='FKIKBlend', inTangentType='stepnext', outTangentType='step') 
        
        
        
        
        for i in range(len(IKlegjnt)):
            ctrlName = ns + dir + FKlegctrl[i]
            IKctrlName = ns + dir + IKlegctrl[i]
            
            cmds.setKeyframe(IKctrlName, at='translate')
            cmds.setKeyframe(IKctrlName, at='rotate')
            cmds.keyTangent(IKctrlName, time=(current_time, current_time), lock = False, outTangentType='stepnext')
                        
            const=cmds.orientConstraint(ns+dir+IKlegjnt[i], ctrlName)
            cmds.setKeyframe(ctrlName, at='rotate')
            cmds.delete(const)            

        cmds.setAttr(IKFKblendCtrl+'.FKIKBlend', 0)
        cmds.setAttr(IKFKblendCtrl+'.FKIKBlend', 0)
            
    elif IKFKstate == 0:
        lastKeyframeFK = None  
        for i in range(len(FKjnt)):
            FKctrlName = ns+dir+FKlegctrl[i]
            
            keyframe_times = cmds.keyframe(FKctrlName, q=1, timeChange=1)
            
            if keyframe_times:
                for keyframe_time in keyframe_times:
                    if keyframe_time < current_time:
                        lastKeyframeFK = max(lastKeyframeFK, keyframe_time)
                
        if lastKeyframeFK is not None:
            cmds.setKeyframe(IKFKblendCtrl, attribute='FKIKBlend', time=lastKeyframeFK)
            cmds.keyTangent(IKFKblendCtrl, at='FKIKBlend', inTangentType='stepnext', outTangentType='step') 
    
    
    
        for i in range(len(IKlegjnt)):
            ctrlName = ns + dir + FKlegctrl[i]
            cmds.setKeyframe(ctrlName, at='rotate')
            cmds.keyTangent(ctrlName, time=(current_time, current_time), lock = False, outTangentType='stepnext')
            
            
            FKhand = ns+dir+FKlegjnt[2]
            FKforeArm = ns+dir+FKlegjnt[1]
        
            IKleg = ns+dir+IKlegctrl[0]
            IKsubHand = ns+dir+IKlegctrl[1]
            armPole = ns+dir+IKlegctrl[2]

     
            const1 = cmds.pointConstraint(FKhand, IKleg)
            const2 = cmds.orientConstraint(FKhand, IKsubHand)
            const3 = cmds.pointConstraint(FKforeArm, armPole)
            cmds.setKeyframe(IKleg, at='translate')
            cmds.setKeyframe(IKleg, at='rotate')
            cmds.setKeyframe(IKsubHand, at='rotate')
            cmds.setKeyframe(armPole, at='translate')
            cmds.delete(const1, const2, const3)

            cmds.setAttr(IKFKblendCtrl+'.FKIKBlend', 1)
            cmds.setAttr(IKFKblendCtrl+'.FKIKBlend', 1)
     
    cmds.setKeyframe(IKFKblendCtrl, at='FKIKBlend')
    cmds.keyTangent(IKFKblendCtrl, at='FKIKBlend', inTangentType='stepnext', outTangentType='step') 

def legEuler(IKjnt, FKjnt, IKctrl, FKctrl, nameSpace, dir):
    
    IKlegctrl=['IK_foot_CON', 'IK_footSub_CON', 'IK_footVec_CON']
    FKlegctrl=['FK_leg_CON', 'FK_lowLeg_CON', 'FK_foot_CON']

    sel=cmds.ls(sl=1)[0]
    nameSpace=sel.split(":")[0]+":"

    obj=sel.split(":")[1]
    dir=obj[0:2]
    

    objnames = [nameSpace+dir+FKlegctrl[0], nameSpace+dir+FKlegctrl[1], nameSpace+dir+FKlegctrl[2], nameSpace+dir+IKlegctrl[0], nameSpace+dir+IKlegctrl[1], nameSpace+dir+IKlegctrl[2]]
    cmds.filterCurve(objnames)            

############################################################################################################################################################################################################################################
############################################################################################################################################################################################################################################
#setKeyFrame at current Frame and Before 1 Frame to BlendCON_ARM
############################################################################################################################################################################################################################################
############################################################################################################################################################################################################################################

def armIKFK_STPKY():
    
    IKjnt=['IK_upArm_JNT', 'IK_foreArm_JNT', 'IK_hand_JNT'] 
    FKjnt=['FK_upArm_JNT', 'FK_foreArm_JNT', 'FK_hand_JNT']
    IKctrl=['IK_hand_CON', 'IK_handSub_CON', 'IK_handVec_CON']
    FKctrl=['FK_upArm_CON', 'FK_foreArm_CON', 'FK_hand_CON']
    
    sel=cmds.ls(sl=1)[0]
    nameSpace=sel.split(":")[0]+":"
    
    obj=sel.split(":")[1]
    dir=obj[0:2]
    
    snap_STPKY(IKjnt,FKjnt,IKctrl,FKctrl,nameSpace, dir)

    objnames = [nameSpace+dir+FKctrl[0], nameSpace+dir+FKctrl[1], nameSpace+dir+FKctrl[2], nameSpace+dir+IKctrl[0], nameSpace+dir+IKctrl[1], nameSpace+dir+IKctrl[2]]

    for objname in objnames:
    
        rotation = cmds.xform(objname, q=1, rotation=1)

        if any(abs(angle) > 180 for angle in rotation):
            nor = [(angle + 180) % 360 - 180 for angle in rotation]
    
            cmds.xform(objname, rotation=nor)
    
        new_rot = cmds.xform(objname, q=1, rotation=1)

    armEuler(IKjnt, FKjnt, IKctrl, FKctrl, nameSpace, dir)       

def snap_STPKY(IKjnt, FKjnt, IKctrl, FKctrl, ns, dir):
    
    IKFKblendCtrl = ns+dir+"armBlend_CON"
    IKFKstate = cmds.getAttr(IKFKblendCtrl+'.FKIKBlend')

    current_frame = cmds.currentTime(q=1)
    minusOneFrame = current_frame - 1     
    plusOneFrame = current_frame + 1
    
    if IKFKstate == 1:
        for i in range(len(IKjnt)):
            ctrlName = ns+dir+FKctrl[i]
            IKctrlName = ns+dir+IKctrl[i]

            cmds.setKeyframe(IKctrlName, at='translate')
            cmds.setKeyframe(IKctrlName, at='rotate')
            cmds.keyTangent(IKctrlName, at='translate', time=(current_frame, current_frame), lock = False , inTangentType='linear', outTangentType='linear')
            cmds.keyTangent(IKctrlName, at='rotate', time=(current_frame, current_frame), lock = False , inTangentType='linear', outTangentType='linear') 


            
            
            const=cmds.orientConstraint(ns+dir+IKjnt[i], ctrlName)
            cmds.setKeyframe(ctrlName, at='rotate')
            cmds.delete(const)            

        cmds.setKeyframe(IKFKblendCtrl, at='FKIKBlend', time=minusOneFrame)   
        cmds.setAttr(IKFKblendCtrl+'.FKIKBlend', 0)
          
    elif IKFKstate == 0:
        
        for i in range(len(IKjnt)):
            
            ctrlName = ns+dir+FKctrl[i]
            FKhand = ns+dir+FKjnt[2]
            FKforeArm = ns+dir+FKjnt[1]
        
            IKarm = ns+dir+IKctrl[0]
            IKsubHand = ns+dir+IKctrl[1]
            armPole = ns+dir+IKctrl[2]
            

            cmds.setKeyframe(ctrlName, at='rotate')
        
            cmds.keyTangent(ctrlName, at='rotate', time=(current_frame, current_frame), lock = False , inTangentType='linear', outTangentType='linear') 

            
            const1 = cmds.pointConstraint(FKhand, IKarm)
            const2 = cmds.orientConstraint(FKhand, IKsubHand)
            const3 = cmds.pointConstraint(FKforeArm, armPole)
            cmds.setKeyframe(IKarm, at='translate')
            cmds.setKeyframe(IKarm, at='rotate')
            cmds.setKeyframe(IKsubHand, at='rotate')
            cmds.setKeyframe(armPole, at='translate')
            cmds.delete(const1, const2, const3)
        
        cmds.setKeyframe(IKFKblendCtrl, at='FKIKBlend', time=minusOneFrame)
            
        cmds.setAttr(IKFKblendCtrl+'.FKIKBlend', 1)
        
            
    cmds.setKeyframe(IKFKblendCtrl, at='FKIKBlend')
    cmds.keyTangent(IKFKblendCtrl, at='FKIKBlend', inTangentType='stepnext', outTangentType='step')
def armEuler(IKjnt, FKjnt, IKctrl, FKctrl, nameSpace, dir):
    
    IKctrl=['IK_hand_CON', 'IK_handSub_CON', 'IK_handVec_CON']
    FKctrl=['FK_upArm_CON', 'FK_foreArm_CON', 'FK_hand_CON']

    sel=cmds.ls(sl=1)[0]
    nameSpace=sel.split(":")[0]+":"

    obj=sel.split(":")[1]
    dir=obj[0:2]
    

    objnames = [nameSpace+dir+FKctrl[0], nameSpace+dir+FKctrl[1], nameSpace+dir+FKctrl[2], nameSpace+dir+IKctrl[0], nameSpace+dir+IKctrl[1], nameSpace+dir+IKctrl[2]]
    cmds.filterCurve(objnames)
    
#############################################################################################################
#setKeyFrame at current Frame and Before 1 Frame to BlendCON_LEG
#############################################################################################################
def legIKFK_STPKY():
    
    IKlegjnt=['IK_leg_JNT', 'IK_lowLeg_JNT', 'IK_foot_JNT']
    FKlegjnt=['FK_leg_JNT', 'FK_lowLeg_JNT', 'FK_foot_JNT']
    IKlegctrl=['IK_foot_CON', 'IK_footSub_CON', 'IK_footVec_CON']
    FKlegctrl=['FK_leg_CON', 'FK_lowLeg_CON', 'FK_foot_CON']
    
    sel=cmds.ls(sl=1)[0]
    nameSpace=sel.split(":")[0]+":"
    
    obj=sel.split(":")[1]
    dir=obj[0:2]
    
    snapLeg_STPKY(IKlegjnt,FKlegjnt,IKlegctrl,FKlegctrl,nameSpace, dir)

    objnames = [nameSpace+dir+FKlegctrl[0], nameSpace+dir+FKlegctrl[1], nameSpace+dir+FKlegctrl[2], nameSpace+dir+IKlegctrl[0], nameSpace+dir+IKlegctrl[1], nameSpace+dir+IKlegctrl[2]]

    for objname in objnames:
    
        rotation = cmds.xform(objname, q=1, rotation=1)

        if any(abs(angle) > 180 for angle in rotation):
            nor = [(angle + 180) % 360 - 180 for angle in rotation]
    
            cmds.xform(objname, rotation=nor)
        new_rot = cmds.xform(objname, q=1, rotation=1)
        
def snapLeg_STPKY(IKlegjnt, FKlegjnt, IKlegctrl, FKlegctrl, ns, dir):

    IKFKblendCtrl = ns+dir+"legBlend_CON"
    IKFKstate = cmds.getAttr(IKFKblendCtrl+'.FKIKBlend')

    current_frame = cmds.currentTime(q=1)
    minusOneFrame = current_frame - 1     

    if IKFKstate == 1:
        for i in range(len(IKlegjnt)):
            ctrlName = ns+dir+FKlegctrl[i]
            IKctrlName = ns+dir+IKlegctrl[i]
            

            cmds.setKeyframe(IKctrlName, at='translate')
            cmds.setKeyframe(IKctrlName, at='rotate')
            cmds.keyTangent(IKctrlName, at='translate', time=(current_frame, current_frame), lock = False , inTangentType='linear', outTangentType='linear') 
            cmds.keyTangent(IKctrlName, at='rotate', time=(current_frame, current_frame), lock = False , inTangentType='linear', outTangentType='linear') 
            
            
            const=cmds.orientConstraint(ns+dir+IKlegjnt[i], ctrlName)
            cmds.setKeyframe(ctrlName, at='rotate')
            cmds.delete(const)            

        cmds.setKeyframe(IKFKblendCtrl, at='FKIKBlend', time=minusOneFrame)   
        cmds.setAttr(IKFKblendCtrl+'.FKIKBlend', 0)
            
    elif IKFKstate == 0:  
        for i in range(len(FKlegjnt)):
            ctrlName = ns + dir + FKlegctrl[i]
            

            cmds.setKeyframe(ctrlName, at='rotate')
            cmds.keyTangent(ctrlName, at='rotate', time=(current_frame, current_frame), lock = False , inTangentType='linear', outTangentType='linear') 


            
            FKhand = ns+dir+FKlegjnt[2]
            FKforeArm = ns+dir+FKlegjnt[1]
        
            IKleg = ns+dir+IKlegctrl[0]
            IKsubHand = ns+dir+IKlegctrl[1]
            armPole = ns+dir+IKlegctrl[2]
        
            const1 = cmds.pointConstraint(FKhand, IKleg)
            const2 = cmds.orientConstraint(FKhand, IKsubHand)
            const3 = cmds.pointConstraint(FKforeArm, armPole)
            cmds.setKeyframe(IKleg, at='translate')
            cmds.setKeyframe(IKleg, at='rotate')
            cmds.setKeyframe(IKsubHand, at='rotate')
            cmds.setKeyframe(armPole, at='translate')
            cmds.delete(const1, const2, const3)
        
        cmds.setKeyframe(IKFKblendCtrl, at='FKIKBlend', time=minusOneFrame)    
        cmds.setAttr(IKFKblendCtrl+'.FKIKBlend', 1)
             
    cmds.setKeyframe(IKFKblendCtrl, at='FKIKBlend')
    
def legEuler(IKjnt, FKjnt, IKctrl, FKctrl, nameSpace, dir):
    
    IKlegctrl=['IK_foot_CON', 'IK_footSub_CON', 'IK_footVec_CON']
    FKlegctrl=['FK_leg_CON', 'FK_lowLeg_CON', 'FK_foot_CON']

    sel=cmds.ls(sl=1)[0]
    nameSpace=sel.split(":")[0]+":"

    obj=sel.split(":")[1]
    dir=obj[0:2]
    

    objnames = [nameSpace+dir+FKlegctrl[0], nameSpace+dir+FKlegctrl[1], nameSpace+dir+FKlegctrl[2], nameSpace+dir+IKlegctrl[0], nameSpace+dir+IKlegctrl[1], nameSpace+dir+IKlegctrl[2]]
    cmds.filterCurve(objnames)            
#################################################################################
#setKeyFrame at current Frame and Before 1 Frame to BlendCON_LEG
#################################################################################
def legIKFK_STPKY():
    
    IKlegjnt=['IK_leg_JNT', 'IK_lowLeg_JNT', 'IK_foot_JNT']
    FKlegjnt=['FK_leg_JNT', 'FK_lowLeg_JNT', 'FK_foot_JNT']
    IKlegctrl=['IK_foot_CON', 'IK_footSub_CON', 'IK_footVec_CON']
    FKlegctrl=['FK_leg_CON', 'FK_lowLeg_CON', 'FK_foot_CON']
    
    sel=cmds.ls(sl=1)[0]
    nameSpace=sel.split(":")[0]+":"
    
    obj=sel.split(":")[1]
    dir=obj[0:2]
    
    snapLeg_STPKY(IKlegjnt,FKlegjnt,IKlegctrl,FKlegctrl,nameSpace, dir)

    objnames = [nameSpace+dir+FKlegctrl[0], nameSpace+dir+FKlegctrl[1], nameSpace+dir+FKlegctrl[2], nameSpace+dir+IKlegctrl[0], nameSpace+dir+IKlegctrl[1], nameSpace+dir+IKlegctrl[2]]

############################################################################################################################################################################################################################################
############################################################################################################################################################################################################################################
#just switch
############################################################################################################################################################################################################################################
############################################################################################################################################################################################################################################

def armSwitch():
    
    IKjnt=['IK_upArm_JNT', 'IK_foreArm_JNT', 'IK_hand_JNT'] 
    FKjnt=['FK_upArm_JNT', 'FK_foreArm_JNT', 'FK_hand_JNT']
    IKctrl=['IK_hand_CON', 'IK_handSub_CON', 'IK_handVec_CON']
    FKctrl=['FK_upArm_CON', 'FK_foreArm_CON', 'FK_hand_CON']
    
    
    sel=cmds.ls(sl=1)[0]
    nameSpace=sel.split(":")[0]+":"
    
    obj=sel.split(":")[1]
    dir=obj[0:2]
    
    armSwitch_nonKY(IKjnt,FKjnt,IKctrl,FKctrl,nameSpace, dir)

    objnames = [nameSpace+dir+FKctrl[0], nameSpace+dir+FKctrl[1], nameSpace+dir+FKctrl[2], nameSpace+dir+IKctrl[0], nameSpace+dir+IKctrl[1], nameSpace+dir+IKctrl[2]]


   
def armSwitch_nonKY(IKjnt, FKjnt, IKctrl, FKctrl, ns, dir):
    
    IKFKblendCtrl = ns+dir+"armBlend_CON"
    IKFKstate = cmds.getAttr(IKFKblendCtrl+'.FKIKBlend')
    current_time = cmds.currentTime(q=1)
    IKFKkey = cmds.keyframe(IKFKblendCtrl, q=1, time=(current_time, current_time)) 

    if IKFKstate == 1 and IKFKkey:
        for i in range(len(IKjnt)):
            ctrlName = ns+dir+FKctrl[i]
            const=cmds.orientConstraint(ns+dir+IKjnt[i], ctrlName)
            cmds.delete(const)            
  
        cmds.setAttr(IKFKblendCtrl+'.FKIKBlend', 0)


    elif IKFKstate == 1 and not IKFKkey:
        for i in range(len(IKjnt)):
            ctrlName = ns+dir+FKctrl[i]
            const=cmds.orientConstraint(ns+dir+IKjnt[i], ctrlName)
            cmds.delete(const)            
  
        cmds.setAttr(IKFKblendCtrl+'.FKIKBlend', 0)
        cmds.cutKey(IKFKblendCtrl+'.FKIKBlend', time=(current_time, current_time), clear=True)
        
            
    elif IKFKstate == 0 and IKFKkey:  
        FKhand = ns+dir+FKjnt[2]
        FKforeArm = ns+dir+FKjnt[1]
        
        IKarm = ns+dir+IKctrl[0]
        IKsubHand = ns+dir+IKctrl[1]
        armPole = ns+dir+IKctrl[2]
        
        const1 = cmds.pointConstraint(FKhand, IKarm)
        const2 = cmds.orientConstraint(FKhand, IKsubHand)
        const3 = cmds.pointConstraint(FKforeArm, armPole)

        cmds.delete(const1, const2, const3)
        
        cmds.setAttr(IKFKblendCtrl+'.FKIKBlend', 1)

    elif IKFKstate == 0 and not IKFKkey:  
        FKhand = ns+dir+FKjnt[2]
        FKforeArm = ns+dir+FKjnt[1]
        
        IKarm = ns+dir+IKctrl[0]
        IKsubHand = ns+dir+IKctrl[1]
        armPole = ns+dir+IKctrl[2]
        
        const1 = cmds.pointConstraint(FKhand, IKarm)
        const2 = cmds.orientConstraint(FKhand, IKsubHand)
        const3 = cmds.pointConstraint(FKforeArm, armPole)

        cmds.delete(const1, const2, const3)
        
        cmds.setAttr(IKFKblendCtrl+'.FKIKBlend', 1)
        cmds.cutKey(IKFKblendCtrl+'.FKIKBlend', time=(current_time, current_time), clear=True)
        
   
### nonKey



def legSwitch():
    
    IKlegjnt=['IK_leg_JNT', 'IK_lowLeg_JNT', 'IK_foot_JNT']
    FKlegjnt=['FK_leg_JNT', 'FK_lowLeg_JNT', 'FK_foot_JNT']
    IKlegctrl=['IK_foot_CON', 'IK_footSub_CON', 'IK_footVec_CON']
    FKlegctrl=['FK_leg_CON', 'FK_lowLeg_CON', 'FK_foot_CON']
    
    
    sel=cmds.ls(sl=1)[0]
    nameSpace=sel.split(":")[0]+":"
    
    obj=sel.split(":")[1]
    dir=obj[0:2]
    
    legSwitch_nonKY(IKlegjnt,FKlegjnt,IKlegctrl,FKlegctrl,nameSpace, dir)



    
def legSwitch_nonKY(IKlegjnt, FKlegjnt, IKlegctrl, FKlegctrl, ns, dir):

    IKFKblendCtrl = ns+dir+"legBlend_CON"
    IKFKstate = cmds.getAttr(IKFKblendCtrl+'.FKIKBlend')
    current_time = cmds.currentTime(q=1)
    IKFKkey = cmds.keyframe(IKFKblendCtrl, q=1, time=(current_time, current_time)) 
    

    if IKFKstate == 1 and IKFKkey:
        for i in range(len(IKlegjnt)):
            ctrlName = ns+dir+FKlegctrl[i]
            const=cmds.orientConstraint(ns+dir+IKlegjnt[i], ctrlName)
            cmds.delete(const)            

        cmds.setAttr(IKFKblendCtrl+'.FKIKBlend', 0)

    if IKFKstate == 1 and not IKFKkey:
        for i in range(len(IKlegjnt)):
            ctrlName = ns+dir+FKlegctrl[i]
            const=cmds.orientConstraint(ns+dir+IKlegjnt[i], ctrlName)
            cmds.delete(const)            

        cmds.setAttr(IKFKblendCtrl+'.FKIKBlend', 0)
        cmds.cutKey(IKFKblendCtrl+'.FKIKBlend', time=(current_time, current_time), clear=True)

            
    elif IKFKstate == 0 and IKFKkey:  
        FKhand = ns+dir+FKlegjnt[2]
        FKforeArm = ns+dir+FKlegjnt[1]
        
        IKleg = ns+dir+IKlegctrl[0]
        IKsubHand = ns+dir+IKlegctrl[1]
        armPole = ns+dir+IKlegctrl[2]
        
        const1 = cmds.pointConstraint(FKhand, IKleg)
        const2 = cmds.orientConstraint(FKhand, IKsubHand)
        const3 = cmds.pointConstraint(FKforeArm, armPole)
   
        cmds.delete(const1, const2, const3)

        cmds.setAttr(IKFKblendCtrl+'.FKIKBlend', 1)
        
    elif IKFKstate == 0 and not IKFKkey:  
        FKhand = ns+dir+FKlegjnt[2]
        FKforeArm = ns+dir+FKlegjnt[1]
        
        IKleg = ns+dir+IKlegctrl[0]
        IKsubHand = ns+dir+IKlegctrl[1]
        armPole = ns+dir+IKlegctrl[2]
        
        const1 = cmds.pointConstraint(FKhand, IKleg)
        const2 = cmds.orientConstraint(FKhand, IKsubHand)
        const3 = cmds.pointConstraint(FKforeArm, armPole)
   
        cmds.delete(const1, const2, const3)

        cmds.setAttr(IKFKblendCtrl+'.FKIKBlend', 1)
        cmds.cutKey(IKFKblendCtrl+'.FKIKBlend', time=(current_time, current_time), clear=True)


################################################################################################################################################################################
#Radio Box commands
################################################################################################################################################################################

   
def tangent():
       
    cmds.cycleCheck(e=False)
    sel_tan = cmds.ls(sl=1)[0]
    nameSpace=sel_tan.split(":")[0]+":"
    
    obj=sel_tan.split(":")[1]
    dir=obj[0:2]
    
    
    armcon = [nameSpace+dir+"IK_hand_CON", nameSpace+dir+"IK_handSub_CON", nameSpace+dir+"IK_handVec_CON", nameSpace+dir+"FK_hand_CON", nameSpace+dir+"FK_foreArm_CON", nameSpace+dir+"FK_upArm_CON"] 
    legcon = [nameSpace+dir+"IK_foot_CON", nameSpace+dir+"IK_footSub_CON", nameSpace+dir+"IK_footVec_CON", nameSpace+dir+"FK_foot_CON", nameSpace+dir+"FK_lowLeg_CON", nameSpace+dir+"FK_leg_CON"]
 
    #Local_ON = cmds.checkBox("FIK_arm_SWCH_button", q=1, v=1)
    #Local_OFF = cmds.checkBox("FIK_leg_SWCH_button", q=1, v=1)
    
    if sel_tan in armcon:
        armIKFK_Tan()
        cmds.cycleCheck(e=False)
    
    elif sel_tan in legcon:
        legIKFK_Tan()
        cmds.cycleCheck(e=False)    

def stepKey():

    cmds.cycleCheck(e=False)
    sel_stp = cmds.ls(sl=1)[0]    
    nameSpace=sel_stp.split(":")[0]+":"
    
    obj=sel_stp.split(":")[1]
    dir=obj[0:2]
    #Local_ON = cmds.checkBox("FIK_arm_SWCH_button", q=1, v=1)
    #Local_OFF = cmds.checkBox("FIK_leg_SWCH_button", q=1, v=1)    
    
    armcon = [nameSpace+dir+"IK_hand_CON", nameSpace+dir+"IK_handSub_CON", nameSpace+dir+"IK_handVec_CON", nameSpace+dir+"FK_hand_CON", nameSpace+dir+"FK_foreArm_CON", nameSpace+dir+"FK_upArm_CON"] 
    legcon = [nameSpace+dir+"IK_foot_CON", nameSpace+dir+"IK_footSub_CON", nameSpace+dir+"IK_footVec_CON", nameSpace+dir+"FK_foot_CON", nameSpace+dir+"FK_lowLeg_CON", nameSpace+dir+"FK_leg_CON"]
 
    
    if sel_stp in armcon:
        armIKFK_STPKY()
        cmds.cycleCheck(e=False)
        
    elif sel_stp in legcon:
        legIKFK_STPKY()
        cmds.cycleCheck(e=False)

def justSwitch():


    cmds.cycleCheck(e=False)
    sel_swc = cmds.ls(sl=1)[0]
    nameSpace=sel_swc.split(":")[0]+":"
    
    obj=sel_swc.split(":")[1]
    dir=obj[0:2]
    
    
    armcon = [nameSpace+dir+"IK_hand_CON", nameSpace+dir+"IK_handSub_CON", nameSpace+dir+"IK_handVec_CON", nameSpace+dir+"FK_hand_CON", nameSpace+dir+"FK_foreArm_CON", nameSpace+dir+"FK_upArm_CON"] 
    legcon = [nameSpace+dir+"IK_foot_CON", nameSpace+dir+"IK_footSub_CON", nameSpace+dir+"IK_footVec_CON", nameSpace+dir+"FK_foot_CON", nameSpace+dir+"FK_lowLeg_CON", nameSpace+dir+"FK_leg_CON"]

    if sel_swc in armcon:
        armSwitch()
        print("ARM Switched")
    
    elif sel_swc in legcon:
        legSwitch()
        print("LEG Switched")

    else:
        armSwitch()
        legSwitch()


# launch 버튼이 눌렸을 때 실행될 함수
def launch_button_command(radio_button_grp):
    radio_button_grp = 'radiogrp'
    index = cmds.radioButtonGrp(radio_button_grp, q=True, select=True)
    #Local_ON = cmds.checkBox("FIK_arm_SWCH_button", q=1, v=1)
    #Local_OFF = cmds.checkBox("FIK_leg_SWCH_button", q=1, v=1)
    
    if index:
        
        if index == 1:
            tangent()
        elif index == 2:
            stepKey()
        elif index == 3:
            justSwitch()
    
    else:
        cmds.error("please check BodyParts or Keytype to Switch")

def makeUI():

    if (cmds.window('FKIK_EZswitch', q=1, ex=1)):
        cmds.deleteUI('FKIK_EZswitch', window=True)

    windowUI = cmds.window('FKIK_EZswitch', t="FKIK_Switch(240325)", s=0, widthHeight=(460,200))
    cmds.window(windowUI, edit=1, rtf=1, widthHeight=(300,155))

    cmds.columnLayout("frameLayout_column", w=200, h=150, ebg=1, adjustableColumn=True, columnAlign="left", io=1)
    
    cmds.separator(h=10) 
    cmds.frameLayout("FIK_frame",l=" BlendCON Key Type ",w=200, h=71, bv=1, bgs=1)
    cmds.rowColumnLayout("FIK_column",w=100,h=50,numberOfRows=1, rowHeight=[(1,46)], cs=(1,1))
    radio_button_grp = cmds.radioButtonGrp('radiogrp', label=('Type : '), numberOfRadioButtons=3,
                                           labelArray3=[' Step ', ' Linear', 'just Switch'], select=3, cw4=[55, 60, 65, 65])
    #cmds.text(label=' ', width=3) #이거 그냥 체크박스를 레이아웃 벽에서 떨어뜨릴라고쓰는거임
    #cmds.checkBox("FIK_arm_SWCH_button", w=157, l='Local Space ON',)
    #cmds.separator(w=10, style = ("single"))
    #cmds.checkBox("FIK_leg_SWCH_button", w=157, l='Local Space OFF',)

    cmds.setParent('..')    
    cmds.setParent('..')


    cmds.separator(h=15)
    
    cmds.rowColumnLayout("launch_column",w=20,h=70,numberOfRows=1, rowHeight=[(1,46)], cs=(1,1), cat=[1,"both",115])
    cmds.button(label="Switch!", w=80, h=20, command=launch_button_command)
    
    cmds.separator(height=10, style='none')
    
    
    cmds.showWindow(windowUI)
    
    #print("Radio button group name", radio_button_grp) 
    