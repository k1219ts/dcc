# encoding=utf-8

# Import Package Modules
from splineUtil.splineUtil import *

from util.controller import *
from util.homeNul import *

# Import Python Modules

# Import Maya Modules
import maya.cmds as cmds
import maya.mel as mel

"""
startConstNode = cmds.listConnections(type='constraint')[0]
startJointNode = cmds.listConnections(startConstNode, type='joint')[0]

endConstNode = cmds.listConnections(type='constraint')[0]
endtJointNode = cmds.listConnections(endConstNode, type='joint')[0]


pos1 = cmds.xform(startJointNode, ws=True, t=True, q=True)
pos2 = cmds.xform(endtJointNode, ws=True, t=True, q=True)

curveLength = getDistance(pos1, pos2)
"""    
    
class RibbonRig:
    def __init__(self, curveLength, prefix, jointList, conNum, nameSpace, fkCon=False, scaleSpace=0):

        self.curveLength = curveLength
        
        #side = 'left'
        self.jointList = [x for x in jointList]
        self.prefix = prefix   

        self.jointNum = len(self.jointList)
        self.scaleSpace = scaleSpace
        self.conNum = conNum

        self.nameSpace = nameSpace

        self.fkCon = fkCon
    
    def createIkRig(self):
        #mel.eval('nurbsPlane -p 0 0 0 -ax 0 1 0 -w 24 -lr 0.04166666667 -d 3 -u 12 -v 2 -ch 1; objectMoveCommand;')
        #mel.eval('nurbsPlane -p 0 0 0 -ax 0 1 0 -w %s -lr 0.05 -d 3 -u %s -v 2 -ch 1; objectMoveCommand;' % (curveLength, conNum * 2) )
        #self.prefix = self.jointList[0].split(':')[-1].split('_')[0]
        nPlane = cmds.nurbsPlane(p=[0, 0, 0], ax=[0, 0, 1], w=self.curveLength, lr=0.05, d=3, u=int(self.conNum * 2), v=2, ch=False, )
        nsfName = cmds.rename( nPlane,  self.nameSpace + '_' + self.prefix + '_NSF' )

        cmds.setAttr('%s.visibility' % nsfName, 0)

        nsfShape = cmds.listRelatives( nsfName, s=True )[0]
        cmds.setAttr('%s.template' % nsfShape, 1)

        tentacleCurve = cmds.duplicateCurve(self.nameSpace + '_' + self.prefix + "_NSF.v[0.5]", ch=True, n= self.nameSpace + '_' + self.prefix + '_CRV')[0]
        print tentacleCurve

        cmds.setAttr('%s.visibility' % tentacleCurve, 0)

        # curve seting
        moNode = pathToU( self.nameSpace + '_' + self.prefix, tentacleCurve, self.jointNum )
        outputLocator = [ i for i in moNode[0] ]
        motionPathNode = [ i for i in moNode[1] ]
        #print outputLocator
        #print motionPathNode


        cmds.delete( outputLocator, cn=True )
        """
        follicleList = cmds.listConnections( nsfName + 'Shape', type='follicle' )
        FL = list( set(follicleList) )
        follocleShapeList = cmds.listRelatives( FL, s=True )
        """
        follocleShapeList = []
        # create follicle node and connections
        for x in range( len(outputLocator) ):
            #create follicle
            flc = cmds.createNode( 'follicle', name=outputLocator[x].replace( 'LOC', 'FLCShape' ) )
            follocleShapeList.append( flc )

            cmds.connectAttr( '%s.outTranslate' % flc, '%s.translate' % flc[:-5] )
            cmds.connectAttr( '%s.outRotate' % flc, '%s.rotate' % flc[:-5] )
            
            cmds.connectAttr( '%s.local' % nsfShape, '%s.inputSurface' % flc )
            cmds.connectAttr( '%s.worldMatrix[0]' % nsfShape, '%s.inputWorldMatrix' % flc )
            
            cmds.setAttr( '%s.parameterV' % flc, 0.5 )
            
            
            
            #
            dcm = cmds.createNode( 'decomposeMatrix', n=outputLocator[x].replace('LOC', 'DCM') )
            cps = cmds.createNode( 'closestPointOnSurface', n=outputLocator[x].replace('LOC', 'CPS') )
            cmds.connectAttr( '%s.worldMatrix[0]' % outputLocator[x], '%s.inputMatrix' % dcm )
            cmds.connectAttr( '%s.outputTranslate' % dcm, '%s.inPosition' % cps )
            
            cmds.connectAttr( '%s.worldSpace[0]' % nsfShape, '%s.inputSurface' % cps )
            cmds.connectAttr( '%s.parameterU' % cps, '%s.parameterU' % flc, f=True )




        # controller set
        selectCurve = cmds.duplicate( tentacleCurve, n=tentacleCurve.replace( 'CRV', 'dupleCurve' ) )[0]
        #mel.eval( 'rebuildCurve -ch 1 -rpo 1 -rt 0 -end 1 -kr 0 -kcp 0 -kep 1 -kt 0 -s 6 -d 3 -tol 1e-06 "%s";' % selectCurve )
        mel.eval( 'rebuildCurve -ch 1 -rpo 1 -rt 0 -end 1 -kr 0 -kcp 0 -kep 1 -kt 0 -s %s -d 3 -tol 1e-06 "%s";' % ( (self.conNum - 1), selectCurve ) )
        curveShape = cmds.listRelatives( selectCurve, s=True )





        ############  IK Controller     ##########################################333
        cvNum = cmds.getAttr( '%s.cp' % selectCurve, s=True )

        pos = cmds.pointPosition( '%s.cv[0]' % selectCurve )
        #print pos
        ikStartConName = controllerShape( 'IK_' + self.nameSpace + '_' + self.prefix + '_1_CON', 'rombus', 'red' )
        cmds.move( pos[0], pos[1], pos[2], ikStartConName )

        startNurbsBindJoint = cmds.joint( n=ikStartConName.replace( 'CON', 'JNT' ) )
        cmds.select( cl=True )

        ikStartConNulName = homeNul( ikStartConName )


        fkStartConNameList = []
        fkStartConNulNameList = []
        if self.fkCon == True:
            # create FK node
            fkStartConName = controllerShape( 'FK_' + self.nameSpace + '_' + self.prefix + '_1_CON', 'dubleOctagon', 'blue' )
            cmds.rotate(0, 0, -90, '%sShape.cv[0:32]' % fkStartConName, r=True, ocp=True, os=True, fo=True)
            cmds.move( pos[0], pos[1], pos[2], fkStartConName )
            fkStartConNameList.append(fkStartConName)
            fkStartConNulNameList = homeNul( fkStartConName )


        #cvPosition = []
        nurbsBindJointList = []
        conNulNameList = []
        fkConNameList = []
        fkConNulNameList = []
        for i in range( cvNum-4 ):
            
            pos = cmds.pointPosition( '%s.cv[%s]' % (selectCurve, (i+2)) )
            #print pos
            #cvPosition.append( pos )
            conName = controllerShape( 'IK_' + self.nameSpace + '_' + self.prefix + '_%s_CON' % (i+2), 'rombus', 'red' )
            cmds.move( pos[0], pos[1], pos[2], conName )
            
            nurbsBindJoint = cmds.joint( n=conName.replace( 'CON', 'JNT' ) )
            nurbsBindJointList.append( nurbsBindJoint )
            cmds.select( cl=True )
            
            conNulName = homeNul( conName )
            conNulNameList.append( conNulName )


            if self.fkCon == True:
                # create FK node
                fkConName = controllerShape( 'FK_' + self.nameSpace + '_' + self.prefix + '_%s_CON' % (i+2), 'dubleOctagon', 'blue' )
                cmds.rotate(0, 0, -90, '%sShape.cv[0:32]' % fkConName, r=True, ocp=True, os=True, fo=True)
                
                
                fkConNameList.append( fkConName )
                cmds.move( pos[0], pos[1], pos[2], fkConName )
                
                fkConNulName = homeNul( fkConName )
                fkConNulNameList.append( fkConNulName )




            
            
        pos = cmds.pointPosition( '%s.cv[%s]' % (selectCurve, cvNum) )
        #print pos
        endConName = controllerShape( 'IK_' + self.nameSpace + '_' + self.prefix + '_%s_CON' % (cvNum-2), 'rombus', 'red' )
        cmds.move( pos[0], pos[1], pos[2], endConName )

        endNurbsBindJoint = cmds.joint( n=endConName.replace( 'CON', 'JNT' ) )
        cmds.select( cl=True )

        endConNulName = homeNul( endConName )



        fkEndConNameList = []
        fkEndConNulNameList = []
        if self.fkCon == True:
            # create FK node
            fkEndConName = controllerShape( 'FK_' + self.nameSpace + '_' + self.prefix + '_%s_CON' % (cvNum-2), 'dubleOctagon', 'blue' )
            cmds.rotate(0, 0, -90, '%sShape.cv[0:32]' % fkEndConName, r=True, ocp=True, os=True, fo=True)
            
            cmds.move( pos[0], pos[1], pos[2], fkEndConName )
            fkEndConNameList.append(fkEndConName)
            fkEndConNulNameList = homeNul( fkEndConName )

        nurbsBindJointList = [startNurbsBindJoint] + nurbsBindJointList + [endNurbsBindJoint]
        for x in nurbsBindJointList:
            cmds.setAttr('%s.visibility' % x, 0)
            
        conNulNameList = [ikStartConNulName] + conNulNameList + [endConNulName]


        if self.fkCon == True:       
            fkConNameList = fkStartConNameList + fkConNameList + fkEndConNameList
            fkConNulNameList = [fkStartConNulNameList] + fkConNulNameList + [fkEndConNulNameList]

            for x in range(len(fkConNameList)-1):
                cmds.parent(fkConNulNameList[x+1], fkConNameList[x])

















            
        cmds.delete( selectCurve )

        """
        # controller align
        for x in range( len(conNulNameList) ):
            FL = list( set(follicleList) )
            nFollicle = nearestJointsList( conNulNameList[x], FL )[0]
            cmds.delete( cmds.orientConstraint( nFollicle, conNulNameList[x] ) )
        """






        # add start end controller

        startConName = controllerShape(  self.nameSpace + '_' + self.prefix + 'Start_CON', 'cube', 'yellow' )
        startConNulName = homeNul( startConName )

        cmds.delete( cmds.parentConstraint( conNulNameList[0], startConNulName )  )
        """
        endConName = controllerShape(  prefix + 'End_CON', 'cube', 'yellow' )
        endConNulName = homeNul( endConName )

        cmds.delete( cmds.parentConstraint( conNulNameList[-1], endConNulName )  )
        """

        """
        cmds.parent( conNulNameList[:2], startConName )
        cmds.parent( conNulNameList[-2:], endConName )
        """

        # addAttr
        cmds.addAttr( startConName, at='float', dv=0, min=0, max=10, ln='stretch' )
        cmds.addAttr( startConName, at='float', dv=0, min=-10, max=0, ln='squash' )
        cmds.addAttr( startConName, at='float', dv=10, min=-10, max=10, ln='squashStretch', k=True )
        # create dg node
        srgStretchNode = cmds.createNode( 'setRange', n='stretch_SRG' )
        cmds.setAttr( "%s.maxX" % srgStretchNode, 1)
        cmds.setAttr( "%s.oldMaxX" % srgStretchNode, 10)

        srgSquashNode = cmds.createNode( 'setRange', n='squash_SRG' )
        cmds.setAttr( "%s.maxX" % srgSquashNode, cmds.getAttr( '%s.restCurveLength' % tentacleCurve ))
        cmds.setAttr( "%s.oldMinX" % srgSquashNode, -10)

        rvNode = cmds.createNode( 'reverse' )

        # connectAttr
        cmds.connectAttr( '%s.stretch' % startConName, '%s.valueX' % srgStretchNode )
        cmds.connectAttr( '%s.outValueX' % srgStretchNode, '%s.inputX' % rvNode )
        cmds.connectAttr( '%s.outputX' % rvNode, '%s.preserveLength' % tentacleCurve )

        
        cmds.connectAttr( '%s.squash' % startConName, '%s.valueX' % srgSquashNode )
        cmds.connectAttr( '%s.outValueX' % srgSquashNode, '%s.restCurveLength' % tentacleCurve )


        cmds.connectAttr( '%s.squashStretch' % startConName, '%s.squash' % startConName )
        cmds.connectAttr( '%s.squashStretch' % startConName, '%s.stretch' % startConName )






        # outputLocator rotateUp
        for x in range(len(motionPathNode)):
            cmds.setAttr( "%s.worldUpType" % motionPathNode[x], 2 )
            cmds.connectAttr( '%s.worldMatrix[0]' % startConName, '%s.worldUpMatrix' % motionPathNode[x] )




        ##############################################
        # add path offset ############################
        ##############################################
        cmds.addAttr( startConName, at='float', dv=0, min=-10, max=10, ln='pathOffset', k=True )

        offset_MPD = cmds.createNode('multiplyDivide' , n='pathOffset_MPD')
        cmds.setAttr('%s.operation'%offset_MPD , 2)
        cmds.setAttr('%s.input2X'%offset_MPD , 10)

        motionPath = []
        resultUvalueMPD = []
        resultUvalueADL = []

        for i in range(len(outputLocator)):
            list1 = cmds.listConnections(outputLocator[i] , t='motionPath')[0]
            motionPath.append(list1)
            list2 = cmds.listConnections(motionPath[i] , t='multiplyDivide')[0]
            resultUvalueMPD.append(list2)
            list3 = cmds.createNode( 'addDoubleLinear', n='resultUvalue_%s'%i+'_ADL' )
            resultUvalueADL.append(list3)


        cmds.connectAttr('%s.pathOffset' % startConName , '%s.input1X'%offset_MPD)
        for x in range(len(outputLocator)):
            cmds.connectAttr('%s.outputX'%resultUvalueMPD[x] , '%s.input1'%resultUvalueADL[x])
            cmds.connectAttr('%s.outputX'%offset_MPD , '%s.input2'%resultUvalueADL[x])
            cmds.connectAttr('%s.output'%resultUvalueADL[x] , '%s.uValue'%motionPath[x] , f=True)

        ##############################################
        ##############################################
        ##############################################





        """
        ### attach master fk slave ik

        fkSlaveList = [ startConNulName ] + conNulNameList[2:-2] + [ endConNulName ]

        for x in range( len( fkSlaveList ) ):
            cmds.parentConstraint( fkConNameList[1:-1][x], fkSlaveList[x] )
        """    
            
            
            
        """    
        if side == 'R':
            for x in conNulNameList[:2]:
                cmds.rotate( 0, 0, 180, x )
            for x in conNulNameList[-2:]:
                cmds.rotate( 0, 0, 180, x )    
        """


        """
        cmds.parent( conNulNameList[:2], startConName )
        cmds.parent( conNulNameList[-2:], endConName )
        """





        # arrangemrnt

        closestPointLocatorGRP = cmds.group( outputLocator, n= self.nameSpace + '_' + self.prefix + '_closestPointLocator_GRP' )
        cmds.setAttr('%s.visibility' % closestPointLocatorGRP, 0)
        #controllerGRP = cmds.group( [ startConNulName ] + conNulNameList[2:-2] + [ endConNulName ] + [fkConNulNameList[1]], n= prefix + '_Controller_GRP' )
        cmds.parent( conNulNameList, startConName )
        controllerGRP = cmds.group( [ startConNulName ], n= self.nameSpace + '_' + self.prefix + '_Controller_GRP' )
        outputGRP = cmds.group( follocleShapeList, n= self.nameSpace + '_' + self.prefix + '_output_GRP' )
        cmds.setAttr('%s.visibility' % outputGRP, 0)


        # global scale
        if cmds.objExists( 'glovalScale_MPD' ):
            cmds.connectAttr( 'glovalScale_MPD.outputX', '%s.maxX' % srgSquashNode )
        else:
            gs = cmds.createNode( 'multiplyDivide', n='glovalScale_MPD' )
            cmds.setAttr( '%s.input1X' % gs, cmds.getAttr( '%s.restCurveLength' % tentacleCurve ))
            
            cmds.connectAttr( '%s.outputX' % gs, '%s.maxX' % srgSquashNode )
            
            #########################################################################
            if cmds.objExists( self.nameSpace + ':' + 'place_CON' ):
                cmds.connectAttr( self.nameSpace + ':' + 'place_CON.initScale', '%s.input2X' % gs )
            else:
                print 'No object matches name: "place_CON"'
            #########################################################################    
            


        # skin joint
        skinJointList = []
        for x in range(len(follocleShapeList)):
            cmds.select( cl=True )
            cmds.select( follocleShapeList[x][:-5] )
            skinJoint = cmds.joint( n='Skin_' + self.nameSpace + '_' + self.prefix + '_%s_JNT' % x )
            skinJointList.append( skinJoint )    
            cmds.parentConstraint( follocleShapeList[x][:-5], skinJoint )
            #cmds.scaleConstraint( follocleShapeList[x][:-5], skinJoint )
            #cmds.connectAttr( '%s.scaleY' % follocleShapeList[x][:-5], '%s.scaleY' % skinJoint )
            #cmds.connectAttr( '%s.scaleZ' % follocleShapeList[x][:-5], '%s.scaleZ' % skinJoint )
            
        
        skinJointGroup = cmds.group( skinJointList, n= self.nameSpace + '_' + self.prefix + '_skinJoint_GRP', w=True )
        cmds.setAttr('%s.visibility' % skinJointGroup, 0)
        

        #hierarchy
        for x in range(len(skinJointList)-1):
            cmds.parent(skinJointList[x+1], skinJointList[x])

        











        ####################################################################################################################
        ####################################################################################################################
        ####################################################################################################################
        ####################################################################################################################
        ####################################################################################################################
        ####################################################################################################################
        # nurbsBind
        #cmds.skinCluster( nurbsBindJointList, nsfName, mi=2, dr=4 )
        



        meshObj = mel.eval('nurbsToPoly -mnd 1  -ch 1 -f 0 -pt 1 -pc 10000 -chr 0.1 -ft 0.01 -mel 0.001 -d 0.1 -ut 1 -un 3 -vt 1 -vn 3 -uch 0 -ucr 0 -cht 0.2 -es 0 -ntr 0 -mrt 0 -uss 1 %s;' % nsfName)[0]


        posList = []
        for x in range(len(nurbsBindJointList)):
            pos = cmds.xform(nurbsBindJointList[x], ws=True, t=True, q=True)
            posList.append(tuple(pos))

        wCurve = cmds.curve( p=posList )


        wDeform = cmds.wire(meshObj, w=wCurve, dropoffDistance=[0, 300])[0]
        cmds.setAttr( '%s.rotation' % wDeform, 0)





        cmds.select(cl=True)
        for x in nurbsBindJointList:
            cmds.select(x, add=True)

        cmds.select(wCurve, add=True)
        cmds.SmoothBindSkin()


        wireToSkinCluster(wCurve)


        meshToNurbsSkinCopy(meshObj, nsfName)

        cmds.delete(meshObj, wCurve)
        
        ####################################################################################################################
        ####################################################################################################################
        ####################################################################################################################
        ####################################################################################################################
        ####################################################################################################################
        ####################################################################################################################





















        if self.scaleSpace > 0:
            ###  scale 

            #outputLocator
            #follocleShapeList

            #outputLocator
            #follocleShapeList

            scaleConNulList = []
            scaleConList = []
            scalePartJointList = []



            count = 0
            while count < self.jointNum-1:
                stepList = outputLocator[count:count+self.scaleSpace]
                #stepfollicleList = follocleShapeList[count:count+scaleSpace]
                scalePartJoint = self.jointList[count:count+self.scaleSpace]
                
                
                scalePartJointList.append(scalePartJoint)
                
                scaleConStart = controllerShape( self.nameSpace + '_' + self.prefix + 'Scale_%s_CON' % count, 'square', 'skyBlue' )
                scaleConNulStart = homeNul(scaleConStart)
                scaleConNulList.append(scaleConNulStart)
                scaleConList.append(scaleConStart)
                #scaleConEnd = controllerShape( 'scale_' + prefix + 'End_%s_CON' % count, 'square', 'skyBlue' )
                #scaleConNulEnd = homeNul(scaleConEnd)
                #scaleConNulList.append(scaleConNulEnd)
                
                cmds.parentConstraint( stepList[0], scaleConNulStart )
                #cmds.parentConstraint( stepList[-1], scaleConNulEnd )
                
                if len(stepList) < (self.scaleSpace-1):
                    scaleConStart = controllerShape( self.nameSpace + '_' + self.prefix + 'Scale_%s_CON' % (self.jointNum-1), 'square', 'skyBlue' )
                    scaleConNulStart = homeNul(scaleConStart)
                    scaleConNulList.append(scaleConNulStart)
                    scaleConList.append(scaleConStart)
                    
                    cmds.parentConstraint( stepList[-1], scaleConNulStart )

                count = count + (self.scaleSpace-1)
                
                
                
            # scaleConStart = controllerShape( self.prefix + 'Scale_%s_CON' % self.jointNum, 'square', 'skyBlue' )
            # scaleConNulStart = homeNul(scaleConStart)
            # scaleConNulList.append(scaleConNulStart)
            # scaleConList.append(scaleConStart)
            
            cmds.parentConstraint( stepList[-1], scaleConNulStart )     
            # Partial Scale Constraints
            count = 0
            while count < len(scaleConList)-1:
                #print '======================\n%s\n======================' % scaleConList[count]
                
                startWeight = 1.0 /  len(scalePartJointList[count])
                a = 0
                b = 1
                for x in range(len(scalePartJointList[count])):
                    
                    cmds.scaleConstraint(scaleConList[count], scalePartJointList[count][x], weight=b, mo=True, skip='x')
                    cmds.scaleConstraint(scaleConList[count+1], scalePartJointList[count][x], weight=a, mo=True, skip='x')   
                    a = a + startWeight
                    b = b - startWeight
                    
                count = count + 1

            scaleControllerGRP = cmds.group( scaleConNulList, n= self.nameSpace + '_' + self.prefix + '_scaleController_GRP' )
            cmds.parent( scaleControllerGRP, controllerGRP )


            # scale controller vis set
            cmds.addAttr( startConName, ln="scaleConVis", at="enum", en="off:on:", k=True )
            cmds.connectAttr( '%s.scaleConVis' % startConName, '%s.visibility' % scaleControllerGRP )

        grp = cmds.group(nsfName, tentacleCurve, controllerGRP, closestPointLocatorGRP, skinJointGroup, outputGRP, n=self.nameSpace + '_' + self.prefix + '_splineRig_GRP')
        





        if self.fkCon == True:
            cmds.parent( fkConNulNameList[0], controllerGRP )




            #fkConNameList, conNulNameList
            # for x in range(len(fkConNameList)):
            #     print fkConNameList[x], conNulNameList[x]
            #     cmds.parentConstraint(fkConNameList[x], conNulNameList[x], mo=True)







        

        #vis


            
        return startConNulName, skinJointList, skinJointGroup, endConName, conNulNameList, follocleShapeList, fkConNulNameList, grp, controllerGRP



