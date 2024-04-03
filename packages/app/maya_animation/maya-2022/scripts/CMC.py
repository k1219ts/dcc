# encoding:utf-8
# !/usr/bin/env python
#from __future__ import print_function


import maya.cmds as cmds
import random
import os

from dxBlockUtils import extra
from pxr import Sdf, Usd
import DXUSD.Utils as utl

locatorList=[]
targetEntityList=[]
scaleV=''
offsetV=''
timeScaleMethod_checkBox=''
velocity_field=''
timeRandom_field=''


class GetClipInfo:
    def __init__(self, show, asset):
        self.assetfile = '/show/%s/_3d/asset/%s/clip/clip.usd' % (show, asset)

    def readFile(self):
        srclyr = utl.AsLayer(self.assetfile)
        if not srclyr:
            print('# ERROR : not found asset clip!')
        return srclyr

    def clips(self):
        srclyr = self.readFile()
        if not srclyr:
            return

        result = list()
        with utl.OpenStage(srclyr) as stage:
            dprim = stage.GetDefaultPrim()
            result= dprim.GetVariantSet('clip').GetVariantNames()
        return result

    def timeScales(self, clip):
        srclyr = self.readFile()
        if not srclyr:
            return

        result = list()
        with utl.OpenStage(srclyr) as stage:
            dprim = stage.GetDefaultPrim()
            dprim.GetVariantSet('clip').SetVariantSelection(clip)
            result = dprim.GetVariantSet('timeScale').GetVariantNames()
        return result

def disableVelocityUI(*argv):
    global velocity_field
    global timeRandom_field
    global timeScaleMethod_checkBox

    cmds.floatFieldGrp(velocity_field, e=1, en=0)
    cmds.floatFieldGrp(timeRandom_field, e=1, en=1)
    cmds.checkBoxGrp(timeScaleMethod_checkBox, e=1, v2=0)

def disableRandomTimescaleUI(*argv):
    global velocity_field
    global timeRandom_field
    global timeScaleMethod_checkBox

    cmds.floatFieldGrp(velocity_field, e=1, en=1)
    cmds.floatFieldGrp(timeRandom_field, e=1, en=0)
    cmds.checkBoxGrp(timeScaleMethod_checkBox, e=1, v1=0)

def getLocatorList(*argv):
    
    global locatorList
    locatorList = []
    sel = cmds.ls(sl=1)
    
    for obj in sel: 
        if cmds.objectType(obj) == 'transform':
            shapes = cmds.listRelatives(obj, shapes=1, fullPath=1)
            
            if shapes:
                for shape in shapes:
                    obj_type = cmds.nodeType(shape)
                    if obj_type == 'locator':
                        locatorList.append(obj)
                        
    cmds.select(cl=1)
    
    if locatorList:
        cmds.confirmDialog(t="locator select", m="locator selected ", button="OK")
             
    else:
        cmds.error("please select Locator")

def select_hierachy(*argv):
    
    selected_nodes = cmds.ls(sl=1, long=1) or []
    
    all_nodes = []
    for node in selected_nodes:
        all_nodes.extend(cmds.listRelatives(node, allDescendents=1, fullPath=1) or [])
    
    all_nodes.extend(selected_nodes)
     
        
    cmds.select(all_nodes, replace=1)

def getTargetEntityList(*argv):

    global targetEntityList
    ent_list = []
    
    sel = cmds.ls(sl=1)
    
    select_hierachy()
    group_selection = cmds.ls(sl=1)
       
      
    for node in sel:
        existUsdAtt = cmds.attributeQuery('usdVariantSet_clip',node=node,ex=1)
        if existUsdAtt:
            cmds.select(node, deselect=1)
            ent_list.append(node)
        else:
            continue

    for node in ent_list:
        sel.remove(node)

    for node in group_selection:
        existUsdAtt = cmds.attributeQuery('usdVariantSet_clip',node=node,ex=1)
        if existUsdAtt:
            cmds.select(node, deselect=1)
            ent_list.append(node)
        else:
            continue            
        
    if sel:
        cmds.select(sel)
             
    if group_selection:
        cmds.select(group_selection)
        
        
    cmds.select(cl=1)
    
    targetEntityList = list(set(ent_list))
    
    
    if targetEntityList:
        cmds.confirmDialog(t="USD Clip select", m="USD Clip selected", button="OK")
        print(targetEntityList)
        cmds.warning("selected Clip Successfully")
    else:
        cmds.error("please select USD Clip")
    
def substituteEntity(*argv):

    global scaleV
    global targetEntityList

    clipMesh = cmds.ls(sl=1)
    clipMeshNum = len(clipMesh)

    for i in range(len(targetEntityList)):
        
        sourceNum = random.randint(0, (clipMeshNum-1))
        substituteMeshList = []
        
        if(cmds.objExists(clipMesh[sourceNum])):
            cmds.setAttr(clipMesh[sourceNum]+".tx",0)
            cmds.setAttr(clipMesh[sourceNum]+".ty",0)
            cmds.setAttr(clipMesh[sourceNum]+".tz",0)
            cmds.setAttr(clipMesh[sourceNum]+".rx",0)
            cmds.setAttr(clipMesh[sourceNum]+".ry",0)
            cmds.setAttr(clipMesh[sourceNum]+".rz",0)

            substituteClipMesh = clipMesh[sourceNum]+"_copy_"+str(i)
            copyNum=i
            
            while (cmds.objExists(clipMesh[sourceNum]+"_copy_"+str(copyNum))):
                copyNum=copyNum+1
                if (cmds.objExists(clipMesh[sourceNum]+"_copy_"+str(copyNum))==0):
                    substituteClipMesh=clipMesh[sourceNum]+"_copy_"+str(copyNum)

            cmds.duplicate(clipMesh[sourceNum],n=substituteClipMesh)
            substituteMeshList.append(substituteClipMesh)
            extra.RandomizeOffsetByDxTimeOffset(substituteMeshList, minOffset=0.0, maxOffset=5, step=1.0)

            substitute_TimeConnect=cmds.connectionInfo(substituteMeshList[0]+'.time',sfd=1)
            substitute_OffsetNodeConnect = substitute_TimeConnect.split('.')
            substitute_OffsetNode = substitute_OffsetNodeConnect[0]

            entity_TimeConnect=cmds.connectionInfo(targetEntityList[i]+'.time',sfd=1)
            entity_OffsetNodeConnect = entity_TimeConnect.split('.')
            entity_OffsetNode = entity_OffsetNodeConnect[0]
            entity_offsetValue = cmds.getAttr(entity_OffsetNode+".offset") 

            substituteMeshGroupName = substituteClipMesh + "_grp"
            cmds.group(n=substituteMeshGroupName, em=1)

            targetEntityGroupName = cmds.listRelatives(targetEntityList[i], p=1)
            crowdGroupName = cmds.listRelatives(targetEntityGroupName, p=1)

            targetEntityGrp_t = targetEntityGroupName[0]+'.t'
            targetEntityGrp_r = targetEntityGroupName[0]+'.r'
            source_translate = cmds.connectionInfo(targetEntityGrp_t, sfd=True)
            source_rotate = cmds.connectionInfo(targetEntityGrp_r, sfd=True)

            cmds.connectAttr(source_translate, substituteMeshGroupName+'.t', f=1)
            cmds.connectAttr(source_rotate, substituteMeshGroupName+'.r', f=1)

            targetEntityTranslate = cmds.xform(targetEntityList[i],q=1,ws=1,t=1)
            targetEntityRotate = cmds.xform(targetEntityList[i],q=1,ws=1,ro=1)
            targetEntityScale = cmds.xform(targetEntityList[i],q=1,ws=1,s=1)
 
            cmds.setAttr(substituteClipMesh+'.tx',targetEntityTranslate[0])
            cmds.setAttr(substituteClipMesh+'.ty',targetEntityTranslate[1]) 
            cmds.setAttr(substituteClipMesh+'.tz',targetEntityTranslate[2]) 

            cmds.setAttr(substituteClipMesh+'.rx',targetEntityRotate[0])
            cmds.setAttr(substituteClipMesh+'.ry',targetEntityRotate[1]) 
            cmds.setAttr(substituteClipMesh+'.rz',targetEntityRotate[2]) 

            cmds.setAttr(substituteClipMesh+'.sx',targetEntityScale[0])
            cmds.setAttr(substituteClipMesh+'.sy',targetEntityScale[1]) 
            cmds.setAttr(substituteClipMesh+'.sz',targetEntityScale[2])

            cmds.addAttr(substituteClipMesh, ln='velocity', at='double', dv=0)
            cmds.setAttr(substituteClipMesh+".velocity", e=1, keyable=1)

            entityTimeScale = cmds.getAttr(targetEntityList[i]+'.usdVariantSet_timeScale')
            #entityClipVer = cmds.getAttr(targetEntityList[i]+'.usdVariantSet_clipVer')
            entityClip = cmds.getAttr(targetEntityList[i]+'.usdVariantSet_clip')
            entityVelocity = cmds.getAttr(targetEntityList[i]+'.velocity')
            
            clipMeshClip = cmds.getAttr(clipMesh[0]+'.usdVariantSet_clip')
            
            
              
            cmds.setAttr(substituteClipMesh+".usdVariantSet_timeScale",entityTimeScale, type='string')
            #cmds.setAttr(substituteClipMesh+".usdVariantSet_clipVer",entityClipVer, type='string')
            cmds.setAttr(substituteClipMesh+".usdVariantSet_clip",clipMeshClip, type='string') 
                     
            
            cmds.setAttr(substituteClipMesh+".velocity",entityVelocity)
            
            cmds.setAttr(substitute_OffsetNode+'.offset',entity_offsetValue)
            cmds.delete(targetEntityList[i], targetEntityGroupName)

            cmds.parent(substituteClipMesh,substituteMeshGroupName)
            cmds.parent(substituteMeshGroupName,crowdGroupName)


def attachMesh(*argv):

    global scaleV
    global locatorList

    clipMesh = cmds.ls(sl=1)

    crowdGroupName = clipMesh[0]+"_crowdGroup"
    cmds.group(n=crowdGroupName, em=1)

    clipMeshNum = len(clipMesh)
    copyClipList = []


    for i in range(len(locatorList)):

        sourceNum = random.randint(0, (clipMeshNum-1))

        if(cmds.objExists(clipMesh[sourceNum])):
            cmds.setAttr(clipMesh[sourceNum]+".tx",0)
            cmds.setAttr(clipMesh[sourceNum]+".ty",0)
            cmds.setAttr(clipMesh[sourceNum]+".tz",0)
            cmds.setAttr(clipMesh[sourceNum]+".rx",0)
            cmds.setAttr(clipMesh[sourceNum]+".ry",0)
            cmds.setAttr(clipMesh[sourceNum]+".rz",0)

            copyClipMesh = clipMesh[sourceNum]+"_copy_"+str(i)
            cmds.duplicate(clipMesh[sourceNum],n=copyClipMesh)
            copyClipList.append(copyClipMesh)


            clipMeshGroupName = copyClipMesh + "_grp"
            cmds.group(copyClipMesh, n=clipMeshGroupName)
            cmds.connectAttr(locatorList[i]+".translate", clipMeshGroupName+".translate")
            cmds.connectAttr(locatorList[i]+".rotate", clipMeshGroupName+".rotate")

            scaleMin = cmds.floatFieldGrp(scaleV, q=1, v1=1)
            scaleMax = cmds.floatFieldGrp(scaleV, q=1, v2=1)

            scaleValue = random.uniform(scaleMin, scaleMax)
            cmds.setAttr(copyClipMesh+".sx", scaleValue)
            cmds.setAttr(copyClipMesh+".sy", scaleValue)
            cmds.setAttr(copyClipMesh+".sz", scaleValue)

            cmds.parent(clipMeshGroupName, crowdGroupName)

            velocity = getVelocity(locatorList[i])

            cmds.addAttr(copyClipMesh, ln='velocity', at='double', dv=velocity)
            cmds.setAttr(copyClipMesh+".velocity", e=1, keyable=1)

    #apply offset value
    extra.RandomizeOffsetByDxTimeOffset(copyClipList, minOffset=0.0, maxOffset=5, step=1.0)




def getVelocity(object):

    cmds.createNode('frameCache', n=object+'_frameCache_X')
    cmds.createNode('frameCache', n=object+'_frameCache_Y')
    cmds.createNode('frameCache', n=object+'_frameCache_Z')
    cmds.createNode('distanceBetween', n=object+'_dist')
    cmds.createNode('multiplyDivide', n=object+'_timeDivide')

    cmds.connectAttr(object+'.translate', object+'_dist.point1',f=1)
    cmds.connectAttr(object+'.tx',object+'_frameCache_X.stream',f=1)
    cmds.connectAttr(object+'.ty',object+'_frameCache_Y.stream',f=1)
    cmds.connectAttr(object+'.tz',object+'_frameCache_Z.stream',f=1)
    cmds.connectAttr(object+'_frameCache_X.past[1]',object+'_dist.point2X',f=1)
    cmds.connectAttr(object+'_frameCache_Y.past[1]',object+'_dist.point2Y',f=1)
    cmds.connectAttr(object+'_frameCache_Z.past[1]',object+'_dist.point2Z',f=1)
    cmds.connectAttr(object+'_dist.distance', object+'_timeDivide.input1X',f=1)
    cmds.setAttr(object+'_timeDivide.operation',2)
    cmds.setAttr(object+'_timeDivide.input2X',24)

    velocity = cmds.getAttr(object+'_timeDivide.outputX')
    cmds.delete(object+'_frameCache_X',object+'_frameCache_Y',object+'_frameCache_Z',object+'_dist',object+'_timeDivide')

    return velocity





def applyScale(*argv):

    global scaleV

    entityList = cmds.ls(sl=1)

    for entity in entityList:

        scaleMin = cmds.floatFieldGrp(scaleV, q=1, v1=1)
        scaleMax = cmds.floatFieldGrp(scaleV, q=1, v2=1)

        scaleValue = random.uniform(scaleMin, scaleMax)
        cmds.setAttr(entity+".sx", scaleValue)
        cmds.setAttr(entity+".sy", scaleValue)
        cmds.setAttr(entity+".sz", scaleValue)




###

def applyOffset(*argv):

    global offsetV


    sel = cmds.ls(sl=1)
    
    select_hierachy()
    group_selection = cmds.ls(sl=1)
       
      
    for node in sel:
        existUsdAtt = cmds.attributeQuery('usdVariantSet_clip',node=node,ex=1)
        if existUsdAtt:
            entityTimeConnect=cmds.connectionInfo(node+'.time',sfd=1)
            offsetNodeConnect = entityTimeConnect.split('.')
            offsetNode = offsetNodeConnect[0]

            offsetMin = cmds.intFieldGrp(offsetV, q=1, v1=1)
            offsetMax = cmds.intFieldGrp(offsetV, q=1, v2=1)

            offsetValue = random.randint(offsetMin, offsetMax)

            cmds.setAttr(offsetNode+".offset", offsetValue)
            cmds.setAttr(offsetNode+".offset", offsetValue)
            cmds.setAttr(offsetNode+".offset", offsetValue)

            
        else:
            continue


    for node in group_selection:
        existUsdAtt = cmds.attributeQuery('usdVariantSet_clip',node=node,ex=1)
        if existUsdAtt:
            entityTimeConnect=cmds.connectionInfo(node+'.time',sfd=1)
            offsetNodeConnect = entityTimeConnect.split('.')
            offsetNode = offsetNodeConnect[0]

            offsetMin = cmds.intFieldGrp(offsetV, q=1, v1=1)
            offsetMax = cmds.intFieldGrp(offsetV, q=1, v2=1)

            offsetValue = random.randint(offsetMin, offsetMax)

            cmds.setAttr(offsetNode+".offset", offsetValue)
            cmds.setAttr(offsetNode+".offset", offsetValue)
            cmds.setAttr(offsetNode+".offset", offsetValue)

            
        else:
            continue            

def setTimeScale(*argv):

    global timeScaleMethod_checkBox
    global timeRandom_field
    #global velocity_field

    
    usd_clip = []
    selected_nodes = cmds.ls(sl=1, long=1) or []
    
    for entity in selected_nodes:
        if cmds.attributeQuery('usdVariantSet_clip', node=entity, ex=1):
            usd_clip.append(entity)
            continue
        
        else:
            all_des = cmds.listRelatives(entity, allDescendents=1, fullPath=1) or []
            for descendant in all_des:
                if cmds.attributeQuery('usdVariantSet_clip', node=descendant, ex=1):
                    usd_clip.append(descendant)
                    continue
    
    cmds.select(usd_clip, replace=1)
    print(usd_clip)
    
    entityList = cmds.ls(sl=1)


    for entity in entityList:

        timeScaleAtt = entity+".usdVariantSet_timeScale"

        currentTimeScale = cmds.getAttr(timeScaleAtt)


        assetInfo = getAssetInfo(entity)
        clipInfo = GetClipInfo(assetInfo[0],assetInfo[1])
        clipList = clipInfo.clips()


        clipName_att = entity+".usdVariantSet_clip"
        clipName = cmds.getAttr(clipName_att)
        #print clipName


        timeScaleList = clipInfo.timeScales(clipName)

        #randomTimeScale_method = cmds.checkBoxGrp(timeScaleMethod_checkBox, q=1, v1=1)
        #speedTimeScale_method = cmds.checkBoxGrp(timeScaleMethod_checkBox, q=1, v2=1)

        cvtTimeScaleList = []

        for timeScale in timeScaleList:

            cvtTimeScaleList.append(float(timeScale.replace('_','.')))




        if cvtTimeScaleList:

            randMin = cmds.floatFieldGrp(timeRandom_field, q=1, v1=1)
            randMax = cmds.floatFieldGrp(timeRandom_field, q=1, v2=1)

            TimeSclaeInRange = []
            for cvtTimeScale in cvtTimeScaleList:

                if(randMin <= cvtTimeScale <= randMax):
                    TimeSclaeInRange.append(cvtTimeScale)

            numTimeScale = len(TimeSclaeInRange)
            randNum = random.randint(0, numTimeScale-1)
            applyTimeScale = str(TimeSclaeInRange[randNum])
            str_applyTimeScale = applyTimeScale.replace('.','_')

            cmds.setAttr(entity+".usdVariantSet_timeScale",str_applyTimeScale,type='string')



    '''
        if(speedTimeScale_method == 1):

            defaultVelocity =  cmds.floatFieldGrp( velocity_field, q=1, value1=1)

            entityVelocity = cmds.getAttr(entity+".velocity")
            orig_timeScale = entityVelocity / defaultVelocity
            velocityTimeScale = findNear(cvtTimeScaleList, orig_timeScale)
            applyTimeScale = str(velocityTimeScale)
            str_applyTimeScale = applyTimeScale.replace('.','_')

            cmds.setAttr(entity+".usdVariantSet_timeScale",str_applyTimeScale,type='string')
    '''




def getAssetInfo(entity):

    cachePath = cmds.getAttr(entity+".filePath")
    pathComponent = cachePath.split('/')

    numComponent = len(pathComponent)
    showName = pathComponent[2]
    assetName = pathComponent[(numComponent-1)-1]

    return showName, assetName




def getAssetName(Name):

    rName = (''.join(reversed(Name)))
    lengthName = len(rName)

    for i in range(len(rName)):

        isCh = rName[i].isdigit()
        if(isCh == 0):
            NameEndIndex = (lengthName - i)
            assetName = Name[0:NameEndIndex]
            return assetName




def findNear(timeScaleList, value):

    for i in range(len(timeScaleList)):
        if(timeScaleList[i]<=value):
            if(timeScaleList[i+1]>=value):
                subtractValue = ((timeScaleList[i+1] - timeScaleList[i])/2)+timeScaleList[i]

                if(subtractValue <= value):
                    return timeScaleList[i+1]

                else:
                    return timeScaleList[i]




def makeUI():

    global scaleV
    global offsetV
    global timeScaleMethod_checkBox
    global velocity_field
    global timeRandom_field

    
    if (cmds.window('clipmeshCrowdWin', q=1, ex=1)):
        cmds.deleteUI('clipmeshCrowdWin', window=True)

    windowUI = cmds.window('clipmeshCrowdWin', t="ClipMesh attach tool", w=300, h=500, rtf=1, sizeable=1)
    cmds.columnLayout(adjustableColumn=True, rowSpacing = 10)

    cmds.separator( height=40, style='in')
    cmds.button("getLocator", w=300, h=30, l='Select locators', command=getLocatorList)
    cmds.button("attachMesh", w=300, h=30, l='attach Clip', command=attachMesh)

    cmds.separator( height=40, style='in')
    cmds.button("getEntitys", w=300, h=30, l='Select USD Clip', command=getTargetEntityList)
    cmds.button("substituteEntity", w=300, h=30, l='Substiute Clip', command=substituteEntity)

    cmds.separator( height=40, style='in' )
    scaleV = cmds.floatFieldGrp( numberOfFields=2, label='Scale', value1=0.9, value2=1.1)
    cmds.button("applyScale", w=300, h=50, l='Apply scale', command=applyScale)

    cmds.separator( height=40, style='in' )
    offsetV = cmds.intFieldGrp( numberOfFields=2, label='Offset', value1=1, value2=7)
    cmds.button("applyOffset", w=300, h=50, l='Apply offset', command=applyOffset)

    cmds.separator( height=40, style='in' )
    #timeScaleMethod_checkBox = cmds.checkBoxGrp( numberOfCheckBoxes=1, label='Apply method', label1='Random',onc=disableVelocityUI, v1=1 )
    #velocity_field = cmds.floatFieldGrp( numberOfFields=1, label='Default velocity', value1=0.5, en=0)
    timeRandom_field = cmds.floatFieldGrp( numberOfFields=2, label='Random range', value1=1, value2=2.0)
    cmds.button("applyTimescale", w=300, h=50, l='Apply timeScale', command=setTimeScale)

    #cmds.setParent('..')
    cmds.showWindow(windowUI)


# makeUI()