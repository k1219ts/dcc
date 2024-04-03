import pymel.core as pm
import maya.cmds as mc
import os
import numpy as np
import math
from math import e
import re
import collections
import pprint
import Zelos
import maya.cmds as cmds
import maya.OpenMaya as om
import maya.OpenMayaAnim as oma
import maya.mel as mel
import time 
from pprint import pprint

def undo(func):
    def wrapper(*args, **kwargs):
        cmds.undoInfo(openChunk=True)
        try:
            ret = func(*args, **kwargs)
        finally:
            cmds.undoInfo(closeChunk=True)
        return ret
    return wrapper
    
class BVHImporter(object):
    def __init__(self):
        self.rootNode = None  # Used for targeting
        self.filename = ""
        self.channels = {}
        self.linearData = []
        self.frame = 0
        self.isCON = False
        self.conData = {}
        self.namespace = ""
        self.target_ns = ""
        
    def read_bvh(self, filename):
        self.skeleton = Zelos.skeleton()
        self.skeleton.load(filename)
        self.frame = self.skeleton.numFrames()
        self.root = self.skeleton.getRoot()
        rootname = str(self.skeleton.jointName(self.root))
        # import key to joint
        if self.isCON == False:
            # create joint set
            # num = 0
            # while cmds.namespace( ex = 'proxy%s'%num ) == True:
            #     num += 1
            # proxy = 'proxy' + str(num)
            # cmds.namespace( add = proxy )
            # cmds.namespace( set = ':' + proxy )
            self.createRecur( self.target_ns, self.root)
            cmds.namespace( set = ':' )
            
        # import key to controllers
        if self.isCON == True and self.conData:
            time1 = time.time()
            # connect fk to joint
            self.connectIktoProxyFk(self.conData)
            self.readRecurKeyToController(self.root, self.conData)  
            print 'Animbrowser: retargetting time ', time.time() - time1, ' sec.'

    @undo            
    def createRecur(self, ns, arachneJoint):
        joint_n = str( self.skeleton.jointName(arachneJoint) )
        if not "_End" in joint_n:
            joint_name = cmds.ls("%s:%s" % (ns, joint_n), type = "joint")
            if joint_name:
                joint_name = joint_name[0]
            # joint_name = cmds.joint( name = ns + ':' + joint_n ) # create joint
            # if arachneJoint !=  self.root:
            #     parent = self.skeleton.getParent(arachneJoint)
            #     parentname = ns + ':' +  str( self.skeleton.jointName(parent) )
            #     try:
            #         cmds.parent(joint_name, w=1)
            #         cmds.parent(joint_name, parentname)
            #         cmds.joint( joint_name, e=True, o=[0,0,0] )
            #     except:
            #         pass

            # keying     
            self.keyAnimCurves( arachneJoint, joint_name, self.frame )
                                
        for i in range(self.skeleton.childNum(arachneJoint)):
            child = self.skeleton.childAt(arachneJoint, i)
            self.createRecur( ns, child)

    def keyAnimCurves(self, joint=None, target="", frames=0, rotation=None, translation=None ):
        TL_list = ['translateX', 'translateY', 'translateZ']
        TA_list = ['rotateX', 'rotateY', 'rotateZ']

        # create anim curve
        dgmod = om.MDGModifier()
        objList = list()
        for i in TL_list:
            obj = dgmod.createNode( 'animCurveTL' )
            dgmod.renameNode( obj, '%s_%s' % (target, i) )
            objList.append( obj )
        for i in TA_list:
            obj = dgmod.createNode( 'animCurveTA' )
            dgmod.renameNode( obj, '%s_%s' % (target, i) )
            objList.append( obj )

        dgmod.doIt()

        # modified: create -> find
        # objList = list()
        # for i in TL_list:
        #     objList.append('%s_%s' % (target, i))
        # for i in TA_list:
        #     objList.append('%s_%s' % (target, i))

        keyObjList = list()
        for o in objList:
            obj = oma.MFnAnimCurve()
            obj.setObject( o )
            keyObjList.append( obj )

        # keyframe set
        for f in range(frames):
            if joint:
                rotation = self.rotation(joint, f)
                translation = self.translation(joint, f)

            mtime = om.MTime( f, om.MTime.uiUnit() )

            for x in range(3):
                keyObjList[x].addKey( mtime, translation[x] )

            for x in range(3):
                keyObjList[x+3].addKey( mtime, math.radians(rotation[x]) )

        curveNames = list()

        for i in TL_list:
            index = TL_list.index(i)
            mfn   = om.MFnDependencyNode( objList[index] )
            name  = mfn.name()
            curveNames.append( name )
            # cmds.connectAttr( '%s.output' % name, '%s.%s' % (target, i) )

        for i in TA_list:
            index = TA_list.index(i)
            mfn   = om.MFnDependencyNode( objList[index+3] )
            name  = mfn.name()
            curveNames.append( name )
            # cmds.connectAttr( '%s.output' % name, '%s.%s' % (target, i) )

        print curveNames
        cmds.filterCurve( curveNames )

    def translation(self, joint, frame):
        return [float(self.skeleton.translation(joint, frame).x),
                float(self.skeleton.translation(joint, frame).y),
                float(self.skeleton.translation(joint, frame).z)]

    def rotation(self, joint, frame):
        return [(self.skeleton.orientation(joint, frame).x),
                (self.skeleton.orientation(joint, frame).y),
                (self.skeleton.orientation(joint, frame).z)]

    '''
    upper code confirm success
    '''
        
    @undo      
    def readRecurKeyToController(self, joint, data):
        joint_n = str( self.skeleton.jointName(joint) )
        rotation  = self.rotation(joint, 0)
        offset = self.skeleton.getOffsets(joint)
        if not "_End" in joint_n:
            if joint_n in data:
                fkconName = data[joint_n]['proxy_fk'][0]
                startFrame = cmds.playbackOptions(q=1,min=1)
                if cmds.objExists(fkconName):   
                    self.keyAnimCurves_rotate(joint=joint, conName=fkconName, start=startFrame, frames=self.frame)   
                    self.keyAnimCurves_translate(joint=joint, conName=fkconName, start=startFrame, frames=self.frame)
                        
                if 'ik' in data[joint_n]:
                    ikconName = data[joint_n]['ik'][0]
                    if cmds.objExists(ikconName):
                        matrixList = self.copyMatrix( source=ikconName, start=startFrame, frames=self.frame)
                        try:
                            cmds.delete(data[joint_n]['constraint'][0],cn=1)
                        except:
                            pass
                        
                        self.setMatrix ( matrixList=matrixList, target=ikconName, start=startFrame, frames=self.frame)
                        
                cmds.filterCurve(fkconName)
        for i in range(self.skeleton.childNum(joint)):
            child = self.skeleton.childAt(joint, i)
            self.readRecurKeyToController(child, data)

    def createFKSet_recur(self, joint):
        jointName = str( self.skeleton.jointName(joint) )
        if '_JNT_End' not in jointName:
            self.createRemapFKControllerSet(joint, jointName)
            if joint == self.skeleton.getRoot():
                self.rootFKNull = conNull  
            for i in range(self.sourceSkel.childNum(joint)):
                child = self.sourceSkel.childAt(joint, i)
                self.createFKSet_recur(child)   
                
    def createRemapFKControllerSet(self, joint, jointName):
        if jointName.find('_Skin_') > -1:
            if self.target_ns:
                name = self.target_ns+':'+jointName
            else:
                name = jointName

            parentJoint = self.skeleton.getParent(joint)
            conName = 'proxy_' + jointName.replace('_Skin_','_FK_').replace('_JNT','_CON')
            self.conData[jointName] = { 'proxy_fk': [ conName ] }
            if not cmds.objExists(conName):
                con = cmds.circle(n=conName, nr=[1,0,0], d=3, s=8, ch=0)[0]
                conNull = cmds.group(con, n=conName.replace('_CON','_NUL'))
                temp = cmds.parentConstraint(name, conNull, mo=0, w=1)
                cmds.delete(temp)
                if parentJoint:
                    parentJoint = str(self.skeleton.jointName(parentJoint))
                    parentJoint = self.remapData[parentJoint][0]
                    pCON = 'proxy_' + parentJoint.replace('_Skin_','_FK_').replace('_JNT','_CON')
                    cmds.parent(conNull, pCON)
                    
    @undo        
    def connectIktoProxyFk(self, data):
        constraints = []
        tobake = []
        for i in data:
            jointName = i
            proxyFkName = data[i]['proxy_fk'][0]
            if not '_Skin_' in i:
                continue
                
            if 'ik' in data[i]:
                ikName = data[i]['ik'][0]
                if cmds.objExists(ikName):
                    if cmds.getAttr('%s.rotateX'%ikName, l=1) == False:
                        c = cmds.parentConstraint(proxyFkName, ikName, mo=True, w=1 )
                    else:
                        c = cmds.pointConstraint(proxyFkName, ikName, mo=True, w=1 )
                    
                    data[i]['constraint'] = c
                    constraints += c
                    tobake.append(ikName)
                    
        return tobake, constraints

    def rotationByOrientation(self, rotation, orient):
        x = y = z = 0
        e = om.MEulerRotation(math.radians(rotation[0]),
                              math.radians(rotation[1]),
                              math.radians(rotation[2]))
        e1 = om.MEulerRotation(math.radians(orient[0]),
                              math.radians(orient[1]),
                              math.radians(orient[2]))     
                                                            
        ro = e * e1.inverse()
        return ro
        
    def keyAnimCurves_rotate(self, joint=None, conName="", start=0, frames=0):
        # create anim curve
        dgmod = om.MDGModifier()
        TA_list = ['rotateX', 'rotateY', 'rotateZ']
        objList = list()
        for i in TA_list:
            obj = dgmod.createNode( 'animCurveTA' )
            dgmod.renameNode( obj, '%s_%s' % (conName, i) )
            objList.append( obj )
            
        dgmod.doIt()
    
        keyObjList = list()
        for o in objList:
            obj = oma.MFnAnimCurve()
            obj.setObject( o )
            keyObjList.append( obj )
            
        conNull = conName.replace('_CON','_NUL')
                        
        # keyframe set  
        for f in range(frames):   
            frame = start + f
            if joint:  
                rotation = self.rotation(joint, f)
                if cmds.objExists(conNull):
                    orient =  cmds.getAttr('%s.rotate'%conNull)[0]
                    rotation = self.rotationByOrientation(rotation, orient)
                mtime = om.MTime( frame, om.MTime.uiUnit() )
                for x in range(3):
                    keyObjList[x].addKey( mtime, rotation[x] )
                    
        curveNames = list()
        for i in TA_list:
            index = TA_list.index(i)
            mfn   = om.MFnDependencyNode( objList[index] )
            name  = mfn.name()
            curveNames.append( name )
            cmds.connectAttr( '%s.output' % name, '%s.%s' % (conName, i) )
                
        cmds.filterCurve( curveNames )
                
    def keyAnimCurves_translate(self, joint=None, conName="", start=0, frames=0):        
        # create anim curve
        dgmod = om.MDGModifier()
        TL_list = ['translateX', 'translateY', 'translateZ']
        objList = list()
        for i in TL_list:
            obj = dgmod.createNode( 'animCurveTL' )
            dgmod.renameNode( obj, '%s_%s' % (conName, i) )
            objList.append( obj )
        dgmod.doIt()
    
        keyObjList = list()
        for o in objList:
            obj = oma.MFnAnimCurve()
            obj.setObject( o )
            keyObjList.append( obj )
            
        if joint:
            translation0 = self.translation(joint, 0)
                
        # keyframe set  
        for f in range(frames):   
            frame = start + f
            mtime = om.MTime( frame, om.MTime.uiUnit() )
 
            translation = self.translation(joint, f)
                
            if joint != self.root:
                for x in range(3):
                    keyObjList[x].addKey( mtime, translation[x]-translation0[x] )
            else:
                conNull = conName.replace('CON','NUL')
                keyObjList[0].addKey( mtime, translation[1]-cmds.getAttr(conNull+'.ty') )
                keyObjList[1].addKey( mtime, translation[2]-cmds.getAttr(conNull+'.tz') )                      
                keyObjList[2].addKey( mtime, translation[0]-cmds.getAttr(conNull+'.tx') )                
            
        curveNames = list()
        for i in TL_list:
            index = TL_list.index(i)
            mfn   = om.MFnDependencyNode( objList[index] )
            name  = mfn.name()
            curveNames.append( name )
            if joint != self.root:
                cmds.connectAttr( '%s.output' % name, '%s.%s' % (conName, i) )
          
        ns = ''
        if self.target_ns:
            ns = self.target_ns+':'
              
        if joint == self.root:
            cmds.connectAttr( '%s.output' % curveNames[0], '%s.%s' % (conName, 'translateX') ) # y
            cmds.connectAttr( '%s.output' % curveNames[2], '%s.%s' % (ns+'move_CON', 'translateX') )
            cmds.connectAttr( '%s.output' % curveNames[1], '%s.%s' % (ns+'move_CON', 'translateZ') )
            # copy translate TX,TZ key to move_CON
                
        cmds.filterCurve( curveNames )
        
    def copyMatrix(self, source="", start=0, frames=0):     
        matrixList = list()
        selection = om.MSelectionList()
        selection.add( source )
        dagPath = om.MDagPath()
        selection.getDagPath(0, dagPath)
        mobj = om.MObject()
        selection.getDependNode(0, mobj)
        transFn = om.MFnTransform(dagPath)

        # check alembic node frames
        for f in range(frames):
            frame = start + f
            mtxAttr = transFn.attribute( 'worldMatrix' )
            mtxPlug = om.MPlug( mobj, mtxAttr )
            mtxPlug = mtxPlug.elementByLogicalIndex( 0 )
            frameCtx = om.MDGContext( om.MTime( frame, om.MTime.uiUnit()) )       
            mtxObj   = mtxPlug.asMObject( frameCtx )
            mtxData  = om.MFnMatrixData( mtxObj )
            mtxValue = mtxData.matrix()

            wrdMtx = mtxValue

            # get parent inverse matrix
            mtxAttr = transFn.attribute('parentInverseMatrix')
            mtxPlug = om.MPlug( mobj, mtxAttr )
            mtxPlug = mtxPlug.elementByLogicalIndex( 0 )
            frameCtx = om.MDGContext( om.MTime( frame, om.MTime.uiUnit()) )       
            mtxObj   = mtxPlug.asMObject( frameCtx )
            mtxData  = om.MFnMatrixData( mtxObj )
            mtxValue = mtxData.matrix()

            piMtx = mtxValue

            localMtx = wrdMtx * piMtx
            matrixList.append( localMtx )
            
        return matrixList
        
    def setMatrix(self, matrixList=[], target="", start=0, frames=0):
        '''
        ##########
        set matrix
        ##########
        '''
        dgmod = om.MDGModifier()
        TL_list = ['translateX', 'translateY', 'translateZ']
        TA_list = ['rotateX', 'rotateY', 'rotateZ']

        objList = list()
        for i in TL_list:
            obj = dgmod.createNode( 'animCurveTL' )
            dgmod.renameNode( obj, '%s_%s' % (target, i) )
            objList.append( obj )
        for i in TA_list:
            obj = dgmod.createNode( 'animCurveTA' )
            dgmod.renameNode( obj, '%s_%s' % (target, i) )
            objList.append( obj )

        dgmod.doIt()

        keyObjList = list()
        for o in objList:
            obj = oma.MFnAnimCurve()
            obj.setObject( o )
            keyObjList.append( obj )

        for f in range(frames):
            frame = start + f
            #tmtx  = om.MTransformationMatrix( self.m_matrix[i] )
            localMtx  = om.MMatrix( matrixList[f] )
            tmtx = om.MTransformationMatrix( localMtx )
            mtime = om.MTime( frame, om.MTime.uiUnit() )
            space = om.MSpace.kWorld
            
            tr = tmtx.translation( space )
            for x in range(3):
                keyObjList[x].addKey( mtime, tr[x] )

            ro = tmtx.rotation().asEulerRotation()
            for x in range(3):
                keyObjList[x+3].addKey( mtime, ro[x] )

        curveNames = list()

        for i in TL_list:
            index = TL_list.index(i)
            mfn   = om.MFnDependencyNode( objList[index] )
            name  = mfn.name()
            curveNames.append( name )        
            if cmds.getAttr('%s.%s' % (target, i), l=True) == False:
                if not target.split(':')[-1] in ['root_CON','C_IK_root_CON']:
                    cmds.connectAttr( '%s.output' % name, '%s.%s' % (target, i) )
                            
        if target.split(':')[-1] in ['root_CON','C_IK_root_CON']:
            cmds.connectAttr( '%s.output' % curveNames[1], '%s.%s' %( target, 'translateY') )
            # copy translateY key to root_CON

        for i in TA_list:
            index = TA_list.index(i)
            mfn   = om.MFnDependencyNode( objList[index+3] )
            name  = mfn.name()
            curveNames.append( name )
            if cmds.getAttr('%s.%s' % (target, i), l=True) == False:
                cmds.connectAttr( '%s.output' % name, '%s.%s' % (target, i) )
            
        cmds.filterCurve( curveNames )

