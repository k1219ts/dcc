import hou
import os
import json
import bson
import math
import _alembic_hom_extensions as abc

def camScale(axis):
    node = hou.pwd()
    fileName = hou.node("../").parm("fileName").eval()
    objectPath = hou.node("../").parm("objectPath").eval()
    time = node.parm("frame").eval()/node.parm("fps").eval()
    xform=abc.getWorldXform(fileName,objectPath,time)[0]
    return 1/hou.Matrix4(xform).explode()["scale"][axis]

def camZoom(node,pattern,axis):
    fileName = node.parent().parm("fileName").eval()
    objectPath = node.parent().parm("objectPath").eval()+"/"+pattern
    time = node.parm("frame").eval()/node.parm("fps").eval()
    dict = abc.alembicGetCameraDict(fileName,objectPath,time)
    ap=[]
    
    ap.append(dict["aperture"]/25.4)
    ap.append(dict["aperture"]/dict["filmaspectratio"]/25.4)
    
    return ap[axis]
'''
def initScale():
    fileName = hou.node("../").parm("fileName").eval()
    objectPath = hou.node("../").parm("objectPath").eval()
    time=hou.node("../").parm("frame").eval()/hou.node("../").parm("fps").eval()
    initScale = abc.alembicArbGeometry(fileName,objectPath,"initScale",time)[0][0]
    return initScale
'''
def initScale():
    fileName = hou.node("../").parm("fileName").eval()
    objectPath = hou.node("../").parm("objectPath").eval()
    time=hou.node("../").parm("frame").eval()/hou.node("../").parm("fps").eval()
    if(abc.alembicArbGeometry(fileName,objectPath,"initScale",time)[2]=='unknown'):
        initScale=1
    else:
        initScale = abc.alembicArbGeometry(fileName,objectPath,"initScale",time)[0][0]
    return initScale

def readJson(jsonPath):
    file = open(jsonPath,'r')
    js = json.loads(file.read())
    file.close()
    return js

def readBson(bsonPath):
    file = open(bsonPath,'r')
    bs = file.read()
    file.close()
    return bs

def expandChild(root, child, objectHierarchy, objectType):
    objectHierarchy.append(root + child[0])
    objectType.append(child[1])
    if len(child[2])==0:
        return
    else:
        return expandChild(root+child[0] + "/", child[2][0],objectHierarchy,objectType)


def setKey(tempJs,tempParm,fps,type,scale=1.0):

    if(type==-1):
        if(tempJs["value"]):
            tempParm.set(1)
        else:
            tempParm.set(0)        

    
    if(type==0):
        tempParm.set(tempJs["value"]*scale)
                        
    elif(type==1):
        tempFrame = tempJs['frame']
    
        for j in range(len(tempFrame)):
            hou_keyframe = hou.Keyframe()
            time  = hou.frameToTime( tempFrame[j] )
            value = tempJs['value'][j]*scale
            hou_keyframe.setTime( time )
            hou_keyframe.setValue( value )
            tempParm.setKeyframe( hou_keyframe )
        keys = tempParm.keyframes()
    
   
        if tempJs.has_key('weight'):
            for j in range(len(keys)):            
                hou_keyframe = keys[j]
                inAngle  = tempJs['angle'][j*2]
                outAngle = tempJs['angle'][j*2+1]
                inWeight = tempJs['weight'][j*2]
                outWeight= tempJs['weight'][j*2+1]
                inSlope = fps*math.tan(math.radians(inAngle))
                outSlope = fps*math.tan(math.radians(outAngle))
                inAccel = math.sqrt(math.pow(inWeight,2)*(math.pow(inSlope,2)+1)/(math.pow(fps,2)+math.pow(inSlope,2)))
                outAccel = math.sqrt(math.pow(outWeight,2)*(math.pow(outSlope,2)+1)/(math.pow(fps,2)+math.pow(outSlope,2)))
                hou_keyframe.setSlope( outSlope )
                hou_keyframe.setInSlope( inSlope )
                hou_keyframe.setAccel( outAccel )
                hou_keyframe.setInAccel( inAccel )
                hou_keyframe.setExpression( 'bezier()', hou.exprLanguage.Hscript )
                tempParm.setKeyframe( hou_keyframe )
    
        else:
            for j in range(len(keys)):
                hou_keyframe = keys[j]
                inAngle  = tempJs['angle'][j*2]
                outAngle = tempJs['angle'][j*2+1]
                inSlope = fps*math.tan(math.radians(inAngle))
                outSlope = fps*math.tan(math.radians(outAngle))
                hou_keyframe.setSlope( outSlope )
                hou_keyframe.setInSlope( inSlope )
                hou_keyframe.setExpression( 'cubic()', hou.exprLanguage.Hscript )            
                tempParm.setKeyframe( hou_keyframe )

    elif(type==2):
        tempFrame = tempJs.keys()

        for j in range(len(tempFrame)):
            hou_keyframe = hou.Keyframe()
            time  = hou.frameToTime( int(tempFrame[j]))
            hou_keyframe.setTime( time )
            
            matrix = hou.Matrix4(tempJs[tempFrame[j]])

            t = matrix.explode()['translate']
            r = matrix.explode()['rotate']            
            s = matrix.explode()['scale']

            hou_keyframe.setValue(t[0])
            tempParm.parm("tx").setKeyframe( hou_keyframe )
            
            hou_keyframe.setValue(t[1])
            tempParm.parm("ty").setKeyframe( hou_keyframe )
            
            hou_keyframe.setValue(t[2])            
            tempParm.parm("tz").setKeyframe( hou_keyframe )
            
            hou_keyframe.setValue(r[0])
            tempParm.parm("rx").setKeyframe( hou_keyframe )
            
            hou_keyframe.setValue(r[1])
            tempParm.parm("ry").setKeyframe( hou_keyframe )
            
            hou_keyframe.setValue(r[2])
            tempParm.parm("rz").setKeyframe( hou_keyframe )
            
            hou_keyframe.setValue(s[0])
            tempParm.parm("sx").setKeyframe( hou_keyframe )

            hou_keyframe.setValue(s[1])
            tempParm.parm("sy").setKeyframe( hou_keyframe )

            hou_keyframe.setValue(s[2])
            tempParm.parm("sz").setKeyframe( hou_keyframe )
      

























