import math
import random
import maya.cmds as cmds
import maya.OpenMaya as om

def GetPosition( nodeName ):
    tx = cmds.getAttr( nodeName + '.tx' )
    ty = cmds.getAttr( nodeName + '.ty' )
    tz = cmds.getAttr( nodeName + '.tz' )
    return om.MPoint( tx, ty, tz )

def SetPosition( nodeName, p ):
    cmds.setAttr( nodeName + '.tx', p.x )
    cmds.setAttr( nodeName + '.ty', p.y )
    cmds.setAttr( nodeName + '.tz', p.z )
    cmds.setKeyframe( nodeName + '.tx' )
    cmds.setKeyframe( nodeName + '.ty' )
    cmds.setKeyframe( nodeName + '.tz' )

def GetQuaternion( nodeName ):
    rx = math.radians( cmds.getAttr( nodeName + '.rx' ) )
    ry = math.radians( cmds.getAttr( nodeName + '.ry' ) )
    rz = math.radians( cmds.getAttr( nodeName + '.rz' ) )
    e = om.MEulerRotation( rx, ry, rz )
    q = e.asQuaternion()
    q.normalizeIt()
    return q

def SetQuaternion( nodeName, q ):
    e = q.asEulerRotation()
    cmds.setAttr( nodeName + '.rx', math.degrees(e.x) )
    cmds.setAttr( nodeName + '.ry', math.degrees(e.y) )
    cmds.setAttr( nodeName + '.rz', math.degrees(e.z) )
    cmds.setKeyframe( nodeName + '.rx' )
    cmds.setKeyframe( nodeName + '.ry' )
    cmds.setKeyframe( nodeName + '.rz' )

def GetRotationAxis( currentPosition, previousPosition, upVector ):
    forward = currentPosition - previousPosition
    axis = forward ^ upVector
    axis.normalize()
    return axis

def GetRotationAngle( currentPosition, previousPosition, radius ):
    distance = ( currentPosition - previousPosition ).length()
    angle = distance / radius
    return -angle

radius = 10.0
move = om.MVector(1,0,0)
upVector = om.MVector(0,1,0)

cmds.polySphere( r=radius )
nodeName = 'pSphere1'

for i in range(1,100):
    
    cmds.currentTime( i )

    previousPosition = GetPosition( nodeName )
    currentPosition = previousPosition + move
    SetPosition( nodeName, currentPosition )

    axis = GetRotationAxis( currentPosition, previousPosition, upVector )
    angle = GetRotationAngle( currentPosition, previousPosition, radius )

    q = om.MQuaternion( angle, axis ) # rotation axis -> quaternion
    q.normalizeIt()
    
    Q = GetQuaternion( nodeName )
    Q = Q * q; # add rotation
    Q.normalizeIt()
    
    SetQuaternion( nodeName, Q )
