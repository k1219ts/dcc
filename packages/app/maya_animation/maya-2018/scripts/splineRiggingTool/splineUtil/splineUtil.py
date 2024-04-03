# coding=utf-8
import maya.api.OpenMaya as om
import maya.api.OpenMayaAnim as oma
import maya.cmds as cmds
import maya.mel as mel

# 
def increaseNum( nodeName, NodeType, starNum = 1 ):
    num = str(starNum)
    if cmds.objExists( nodeName + num + '_' + NodeType ):
        
        x = 1
        while cmds.objExists( nodeName + str(x) + '_' + NodeType ):
            x += 1
            
        num = x
        
    return num

##
def pathToU( prefix, curveName, divide ):
    # node name == name_CRV
    #curveSel = cmds.ls( selection = True )[0]
    curveSel = curveName
    curveShape = cmds.listRelatives( curveSel, s=True )[0]
    
    curveInfo = cmds.arclen( curveShape, ch=True )
    curveInfoNode = cmds.rename( curveInfo, curveShape.replace( 'CRVShape', 'CIF' ) )
    
    arcLength = cmds.getAttr( curveInfoNode + '.arcLength' )
    
    # addAttr
    cmds.addAttr( curveSel, ln='preserveLength', at='double', dv=1, k=True )
    cmds.addAttr( curveSel, ln ='restCurveLength', at='double', dv=arcLength, k=True )
    
    # create condition
    preserveLengthCND = cmds.createNode( 'condition', n='preserveLength%s_CND' % increaseNum( 'preserveLength', 'CND' ) )
    cmds.setAttr( '%s.operation' % preserveLengthCND, 3 )
    cmds.setAttr( '%s.colorIfTrueR' % preserveLengthCND, 1 )
    cmds.setAttr( '%s.colorIfFalseR' % preserveLengthCND, 0 )
    # connetAttr
    cmds.connectAttr( curveInfoNode + '.arcLength', preserveLengthCND + '.firstTerm' )
    cmds.connectAttr( curveSel + '.restCurveLength', preserveLengthCND + '.secondTerm' )
    
    # create multiplyDivide == mult
    preserveLengthMPD = cmds.createNode( 'multiplyDivide', n='preserveLength_MPD' )
    # connetAttr
    cmds.connectAttr( curveSel + '.preserveLength', preserveLengthMPD + '.input1X' )
    cmds.connectAttr( preserveLengthCND + '.outColorR', preserveLengthMPD + '.input2X' )
    
    # create blendTwoAttr
    preserveLengthIndexBTA = cmds.createNode( 'blendTwoAttr', n='preserveLengthIndex_BTA' )
    # connetAttr
    cmds.connectAttr( preserveLengthMPD + '.outputX', preserveLengthIndexBTA + '.attributesBlender' )
    cmds.connectAttr( curveSel + '.restCurveLength', preserveLengthIndexBTA + '.input[0]' )
    cmds.connectAttr( curveInfoNode + '.arcLength', preserveLengthIndexBTA + '.input[1]' )
    
    
    div = divide - 1
    # create multiplyDivide == divide
    lengthDivideMPD = cmds.createNode( 'multiplyDivide', n='lengthDivide_MPD' )
    cmds.setAttr( lengthDivideMPD + '.operation', 2 )
    cmds.connectAttr( preserveLengthIndexBTA + '.output', lengthDivideMPD + '.input1X' )
    cmds.setAttr( lengthDivideMPD + '.input2X', div )
    
    outputLOCList = []
    motionPathMPTList = []
    for i in range( div + 1 ):
        # create multDoubleLinear == ratio
        lengthRatioMDL = cmds.createNode( 'multDoubleLinear', n='length_Ratio%s_MDL' % i )
        cmds.connectAttr( lengthDivideMPD + '.outputX', lengthRatioMDL + '.input1' )
        if i == 0:
            cmds.setAttr( lengthRatioMDL + '.input2', i + 0.0001 )
        else:
            cmds.setAttr( lengthRatioMDL + '.input2', i )
        
        # create multiplyDivide == attr rest length divide / rest length
        restLengthDivideMPD = cmds.createNode( 'multiplyDivide', n='restLengthDivide%s_MPD' % i )
        cmds.setAttr( restLengthDivideMPD + '.operation', 2 )
        cmds.connectAttr( curveSel + '.restCurveLength', restLengthDivideMPD + '.input1X' )
        cmds.connectAttr( preserveLengthIndexBTA + '.output', restLengthDivideMPD + '.input2X' )
        
        # create multiplyDivide == ratio result divide
        ratioResultDivideMPD = cmds.createNode( 'multiplyDivide', n='ratioResultDivide%s_MPD' % i )
        cmds.setAttr( ratioResultDivideMPD + '.operation', 2 )
        cmds.connectAttr( preserveLengthIndexBTA + '.output', ratioResultDivideMPD + '.input1X' )
        cmds.connectAttr( lengthRatioMDL + '.output', ratioResultDivideMPD + '.input2X' )
        
        # create multiplyDivide == result U value
        resultUvalueMPD = cmds.createNode( 'multiplyDivide', n='resultUvalue%s_MPD' % i )
        cmds.setAttr( resultUvalueMPD + '.operation', 2 )
        cmds.connectAttr( restLengthDivideMPD + '.outputX', resultUvalueMPD + '.input1X' )
        cmds.connectAttr( ratioResultDivideMPD + '.outputX', resultUvalueMPD + '.input2X' )
        
        # create motionPathNode
        motionPathMPT = cmds.createNode( 'motionPath', n='motionPath%s_MPT' % i )
        motionPathMPTList.append( motionPathMPT )
        cmds.setAttr( motionPathMPT + '.fractionMode', True )# maya bug???
        cmds.connectAttr( resultUvalueMPD + '.outputX', motionPathMPT + '.uValue' )
        cmds.connectAttr( curveShape + '.worldSpace[0]', motionPathMPT + '.geometryPath' )
        
        # create output locator
        outputLOC = cmds.spaceLocator( n=prefix + '_output%s_LOC' % i )[0]
        outputLOCList.append( outputLOC )
        cmds.connectAttr( motionPathMPT + '.allCoordinates', outputLOC + '.translate' )
        cmds.connectAttr( motionPathMPT + '.rotate', outputLOC + '.rotate' )
        
    return outputLOCList, motionPathMPTList




def getDistance( pos1, pos2 ):
    distance = ( ( pos1[0] - pos2[0] ) **2 + ( pos1[1] - pos2[1] ) **2 + ( pos1[2] - pos2[2] ) **2 ) **0.5
    distance = round( distance, 4 )
    return distance
"""    
def get_dis( pos1, pos2 ):
    
    x1, y1, z1 = pos1
    x2, y2, z2 = pos2
    
    distance = math.sqrt(math.pow(x2-x1, 2)
                    + math.pow(y2-y1, 2)
                    + math.pow(z2-z1, 2))
    return distance
"""

def lineVis( startName, endName, template = True ):
    """
    cmds.ls( sl=True )

    startName = sel[0]
    endName = sel[1]
    """
    curveName = cmds.curve( d=1, p=[(0,0,0),(0,0,0)], name = endName.replace( '_' + endName.split('_')[-1], 'GuideVis_CRV' ) )
    
    curveShape = cmds.listRelatives( curveName, s=True )[0]
    curveShapeName = cmds.rename( curveShape, '%sShape' % curveName )
    
    if template == True:
        cmds.setAttr( '%s.template' % curveShapeName, True )
    else:
        cmds.setAttr( '%s.template' % curveShapeName, False )
    
    startMMX = cmds.createNode( 'multMatrix', n=startName.replace( '_%s' % startName.split('_')[-1], 'Vis_MMX' ) )
    endDCM = cmds.createNode( 'decomposeMatrix', n=endName.replace( '_%s' % endName.split('_')[-1], 'Vis_DCM' ) )
    endMMX = cmds.createNode( 'multMatrix', n=endName.replace( '_%s' % endName.split('_')[-1], 'Vis_MMX' ) )
    startDCM = cmds.createNode( 'decomposeMatrix', n=startName.replace( '_%s' % startName.split('_')[-1], 'Vis_DCM' ) )
    
    # Connection Node
    cmds.connectAttr( '%s.worldMatrix[0]' % startName, '%s.matrixIn[0]' % startMMX )
    cmds.connectAttr( '%s.matrixSum' % startMMX, '%s.inputMatrix' % endDCM )
        
    cmds.connectAttr( '%s.worldInverseMatrix[0]' % curveName, '%s.matrixIn[1]' % startMMX )
    
    cmds.connectAttr( '%s.outputTranslate' % endDCM, '%s.controlPoints[0]' %  curveShapeName )
        
        
    cmds.connectAttr( '%s.worldMatrix[0]' % endName, '%s.matrixIn[0]' % endMMX )
    cmds.connectAttr( '%s.matrixSum' % endMMX, '%s.inputMatrix' % startDCM )
        
    cmds.connectAttr( '%s.worldInverseMatrix[0]' % curveName, '%s.matrixIn[1]' % endMMX )
    
    cmds.connectAttr( '%s.outputTranslate' % startDCM, '%s.controlPoints[1]' %  curveShapeName )

    return curveName, startMMX, endDCM, endMMX, startDCM









def duplicateWithoutChildren(nodes, keepShapes=False):

    if not nodes:
        return []

    duplicates = []
    for obj in nodes:
        dup = cmds.duplicate(obj, rc=True)[0] # sometimes maya is buggy without renameChildren==true
       
        # Note: The *shapes* argument can only be set to True to filter
        # to only children shapes, yet it can't be set to False to False
        # to return only non-shapes.
        # Therefore we need to filter out all shapes ourselves (see below)
        children = cmds.listRelatives(dup, fullPath=True)
                                   
        if children:
            if keepShapes:
                # Don't care about preserving order, but do care about removing
                # in a fast way where one *could* possibly have many shapes.
                # It's more likely a premature optimization, but `set` should
                # perform much better on the cases with many many children.
                # Note: We're forcing full path names (fullPath==True and long==True) in
                # `listRelatives` and `ls` because we're filtering based on string names.
                # This avoids any hierarchy/unique name issues. Where speed isn't critical
                # but you would rather code faster give a go with Pymel.
                children = list(set(children) - set(cmds.ls(children, shapes=True, long=True)))
            if children:
                cmds.delete(children)
        duplicates.append(dup)
   
    if duplicates:
        cmds.select(duplicates, r=1) # select newly created nodes
        
    return duplicates










# Add FK



def addFkLayer(parentNodeList, chaldNodeList):
    #parentNode = cmds.ls( sl=True )
    #chaldNodeList = cmds.ls( sl=True )

    for i in range( len(parentNodeList) ):
        mm = cmds.createNode( 'multMatrix' )
        dm = cmds.createNode( 'decomposeMatrix' )

        cmds.connectAttr( '%s.matrixSum' % mm, '%s.inputMatrix' % dm )

        PN = cmds.listRelatives( parentNodeList[i], p=True )

        if PN == None:
            cmds.connectAttr( '%s.worldMatrix[0]' % parentNodeList[i], '%s.matrixIn[0]' % mm )

            cmds.connectAttr( '%s.parentInverseMatrix[0]' % chaldNodeList[i], '%s.matrixIn[1]' % mm )

            cmds.connectAttr( '%s.outputTranslate' % dm, '%s.translate' % chaldNodeList[i] )
            cmds.connectAttr( '%s.outputRotate' % dm, '%s.rotate' % chaldNodeList[i] )
        else:
            cmds.connectAttr( '%s.worldMatrix[0]' % parentNodeList[i], '%s.matrixIn[0]' % mm )

            cmds.connectAttr( '%s.worldInverseMatrix[0]' % PN[0], '%s.matrixIn[1]' % mm )

            cmds.connectAttr( '%s.outputTranslate' % dm, '%s.translate' % chaldNodeList[i] )
            cmds.connectAttr( '%s.outputRotate' % dm, '%s.rotate' % chaldNodeList[i] )





def controllerResize(controller, size):
    """
    size ==>  -0.5 ~ 0.5  is best value..
    """
    stepSize = 1.0 + size

    if len(controller) < 0:
        pass
    else:
        curveShape = cmds.listRelatives(controller, s=True)
        
        
        
        for x in range(len(curveShape)):        
            cvNum = cmds.getAttr('%s.cp' % curveShape[x], s=True)
            cmds.select(cl=True)
            cmds.select('%s.cv[0:%s]' % (curveShape[x], cvNum))
            cmds.scale(stepSize, stepSize, stepSize, r=True, ocp=True)
            cmds.select(cl=True)
            cmds.select(controller)




def meshToNurbsSkinCopy(meshObj, nurbsObj):

    skinClusterO = mel.eval('findRelatedSkinCluster ' + meshObj)
    joints = cmds.listConnections(skinClusterO + '.matrix')

    cmds.select(cl=True)
    for x in joints:
        cmds.select(x, add=True)

    cmds.select(nurbsObj, add=True)
    cmds.SmoothBindSkin()

    cmds.select(cl=True)
    cmds.nurbsSelect(nurbsObj)
    getCV = cmds.ls(sl=True)
    cmds.select(meshObj)
    cmds.select(getCV, add=True)
    
    cmds.CopySkinWeights()
    cmds.select(cl=True)




# def nearestJointsList( base, objList ):
#     nDisList = []
#     disList = []
#     for x in range( len( objList ) ):
#         basePos = cmds.xform( base, ws=True, t=True, q=True )
#         pos = cmds.xform( objList[x], ws=True, t=True, q=True )
#         dis = getDistance( basePos, pos )
#         disList.append( dis )
#     for i in range( len( disList ) ):
#         if len( disList ) != 0:
#             disIndex = disList.index( min( disList ) )
#             nDisList.append( objList[ disIndex ] )
#             #print objList[ disIndex ]
#             del disList[ disIndex ]
#             del objList[ disIndex ]
            
#     return nDisList

###############################################################################################################
# getUParam( pos, curve )
"""
guideCurve = lineVis( startConName, endConName, template=True )[0]

guideCurveShape = cmds.listRelatives( guideCurve, s=True )[0]

for x in range( len( conNulNameList ) ):
    
    mpt = cmds.createNode( 'motionPath' )
    guideLocator = cmds.spaceLocator( n='guide%s_LOC' % x )[0]
    cmds.connectAttr( '%s.worldSpace[0]' % guideCurveShape, '%s.geometryPath' % mpt )
    cmds.connectAttr( '%s.allCoordinates' % mpt, '%s.translate' % guideLocator )
    cmds.connectAttr( '%s.rotate' % mpt, '%s.rotate' % guideLocator )
    
    pos = cmds.xform( conNulNameList[x], ws=True, t=True, q=True )
    uValue = getUParam( pos, guideCurve )
    
    cmds.setAttr( '%s.uValue' % mpt, uValue )
    
    cmds.parentConstraint( guideLocator, conNulNameList[x], mo=True )
"""
###############################################################################################################



def wireToSkinCluster(crv):
    # cmds.select curve
    selectList = [crv] #cmds.ls (sl=1)
    curveShape = cmds.listRelatives (selectList[0], type='nurbsCurve')[0]
    wireNode = cmds.listConnections (curveShape, type='wire')[0]
    skincluster = cmds.listConnections (curveShape, type='skinCluster')[0]
    geometry = cmds.wire (curveShape, q=1, g=1)[0]
    jointList = cmds.skinCluster (skincluster, q=1, inf=1)
    
    # Geometry to Dag Path
    meshMSelection = om.MSelectionList ()
    meshMSelection.add (geometry)
    meshDagPath = meshMSelection.getDagPath (0)
    
    # Get the mesh orgin position
    mFnMesh = om.MFnMesh (meshDagPath)
    geoPosition = mFnMesh.getPoints (om.MSpace.kObject)
    
    # Get the weight from each joint
    weightList = []
    for index in range (len(jointList)) :    
        jntParent = cmds.listRelatives (jointList[index], p=1)
        jntChild = cmds.listRelatives (jointList[index], c=1)
        
        if jntParent :
            cmds.parent (jointList[index], w=1)
        if jntChild :
            cmds.parent (jntChild[0], w=1)
                
        jointMSelection = om.MSelectionList ()
        jointMSelection.add (jointList[index])
        jointDagPath = jointMSelection.getDagPath (0)
        
        # Set and reset the deformation value to joint
        mFnTransform = om.MFnTransform (jointDagPath)
        world = mFnTransform.translation (om.MSpace.kWorld)
        moveWorld = om.MVector (world.x + 1, world.y, world.z)
        mFnTransform.setTranslation (moveWorld, om.MSpace.kWorld)
        
        movePosition = mFnMesh.getPoints (om.MSpace.kObject)    
        jointWeights = []       
        for vertexIndex in range (len(movePosition)) :
            length = movePosition[vertexIndex] - geoPosition[vertexIndex]
            weight = length.length ()
            jointWeights.append (weight)        
        weightList.append (jointWeights)    
        mFnTransform.setTranslation (world, om.MSpace.kWorld)
        
        if jntParent :
            cmds.parent (jointList[index], jntParent[0])
        if jntChild :
            cmds.parent (jntChild[0], jointList[index])      
    
    # Set join weight to geometry
    geoSkinCluster = cmds.skinCluster (jointList, geometry)[0]
    skinMSelection = om.MSelectionList ()    
    skinMSelection.add (geoSkinCluster)
    skinMObject = skinMSelection.getDependNode (0)
    
    mfnSkinCluster = oma.MFnSkinCluster (skinMObject)   
    
    # Vertex components
    vertexIndexList = range (len(geoPosition))
    mfnIndexComp = om.MFnSingleIndexedComponent ()
    vertexComp = mfnIndexComp.create (om.MFn.kMeshVertComponent)
    mfnIndexComp.addElements (vertexIndexList)
    
    # influences
    influenceObjects = mfnSkinCluster.influenceObjects ()
    influenceList = om.MIntArray ()
    for eachInfluenceObject in influenceObjects :
        currentIndex = mfnSkinCluster.indexForInfluenceObject (eachInfluenceObject)
        influenceList.append (currentIndex)    
    
    # weights
    mWeightList = om.MDoubleArray ()
    for wIndex in range (len(weightList[0])) :
        for jntIndex in range (len(weightList)) :  
            mWeightList.append (weightList[jntIndex][wIndex]) 
    
    mfnSkinCluster.setWeights (meshDagPath, vertexComp, influenceList, mWeightList)
    cmds.setAttr ('%s.envelope' % skincluster, 0)
    cmds.setAttr ('%s.envelope' % wireNode, 0)
    
    print '\nWire weights successfully transfer to skincluster'




#lastCon = 'tentacle:addFk_078_CON'
def fkTentacleSpaceBlend(lastCon):
    """
    lastCon ==> 선택한 마지막 컨트롤러 ( oirentConstrains 노드가 있어야함 )

    """
    parentLocator = cmds.spaceLocator(n='%s_parent_%s' % (lastCon.split(':')[0], lastCon.split(':')[1].replace('_CON','_LOC') ) )[0]
    cmds.setAttr("%s.visibility" % parentLocator, 0)
    
    splitList = lastCon.split('_')
    # Result: ['tentacle:addFk', '085', 'CON'] # 
    
    numPadding = int('%s' % splitList[1].zfill(3)) - 1
    numPaddingToString = str(numPadding).zfill(3)
    parentNode = splitList[0] + '_' + numPaddingToString + '_' + splitList[-1].replace('CON','JNT')
    
    cmds.parentConstraint( parentNode, parentLocator )
    
    oriConNode = cmds.orientConstraint( parentLocator, lastCon )[0]
    cmds.setAttr('%s.interpType' % oriConNode, 2)
    targetWeightAttrList = cmds.orientConstraint( oriConNode, weightAliasList=True, q=True )
    
    
    numPadding = int('%s' % splitList[1].zfill(3)) + 1
    numPaddingToString = str(numPadding).zfill(3)
    controlNode = splitList[0] + '_' + numPaddingToString + '_' + splitList[-1]
    
    # addAttr
    cmds.addAttr(controlNode, ln='follow', at='double', min=0, max=10, dv=10, k=True )
    # Create Node
    reMapNode = cmds.createNode('remapValue')
    cmds.setAttr( '%s.inputMax' % reMapNode, 10 )
    
    reverseNode = cmds.createNode('reverse')
    
    # connectAttr
    cmds.connectAttr( '%s.follow' % controlNode, '%s.inputValue' % reMapNode, f=True )
    cmds.connectAttr( '%s.outValue' % reMapNode, '%s.inputX' % reverseNode, f=True )
    cmds.connectAttr( '%s.outValue' % reMapNode, '%s.%s' % (oriConNode, targetWeightAttrList[1]), f=True )
    cmds.connectAttr( '%s.outputX' % reverseNode, '%s.%s' % (oriConNode, targetWeightAttrList[0]), f=True )
    
    return parentLocator
    
    #parent tentacle_parent_addFk_085_LOC tentacle_addFk_GRP ;