import string
import numpy
import math
import maya.cmds as cmds
import maya.api.OpenMaya as OpenMaya

import aniCommon
reload(aniCommon)

CON_LIST = {
    'FK': {
        'ARM': [
            '{side}_FK_upArm_CON',
            '{side}_FK_foreArm_CON',
            '{side}_FK_hand_CON'
        ],
        'LEG': [
            '{side}_FK_leg_CON',
            '{side}_FK_lowLeg_CON',
            '{side}_FK_foot_CON'
        ]
    },
    'IK': {
        'ARM': [
            '{side}_IK_hand_CON',
            '{side}_IK_handVec_CON',
            '{side}_IK_handSub_CON'
        ],
        'LEG': [
            '{side}_IK_foot_CON',
            '{side}_IK_footVec_CON'
        ]
    }
}

_JOINT = {
    'ARM': [
        '{side}_{type}_upArm_JNT',
        '{side}_{type}_foreArm_JNT',
        '{side}_{type}_hand_JNT',
        '{side}_{type}_hand_snapSpace_NUL'
    ],
    'LEG': [
        '{side}_{type}_leg_JNT',
        '{side}_{type}_lowLeg_JNT',
        '{side}_{type}_foot_JNT',
        '{side}_{type}_foot_snapSpace_NUL'
    ]
}


def addNamespace(node, namespace):
    fullName = node
    if namespace:
        fullName = string.join([namespace, node], ":")
    return fullName


def getControlerInfo(node, namespace=None):
    rpart = None
    splitnodename = node.split("|")[-1]

    if not splitnodename.find("L_") == -1:
        side = "L"
    else:
        side = "R"

    for type in CON_LIST:
        for part in CON_LIST[type]:
            for con in CON_LIST[type][part]:
                conName = con.format(side=side)
                fullName = addNamespace(conName, namespace)
                if node == fullName:
                    rpart = part
                    break

    return {'side': side, 'part': rpart}


def radiansToDegrees(eulerRotation):
    angles = [math.degrees(angle) for angle in (eulerRotation.x, eulerRotation.y, eulerRotation.z)]
    return angles


def plusMinusAvg(input1D=None,
                 input2D=None,
                 input3D=None,
                 operation='Sum'):
    output1D = []
    output2D = []
    output3D = []
    outValue = dict()

    if input1D:
        for num in input1D:
            if not output1D:
                output1D = num
            if operation == "Sum":
                output1D += num
            elif operation == "Subtract":
                output1D -= num
        outValue['output1D'] = output1D

    if input3D:
        for array in input3D:
            npArray = numpy.array(array)

            if not list(output3D):
                output3D = npArray
                continue
            if operation == "Sum":
                output3D += npArray
            elif operation == "Subtract":
                output3D -= npArray
        outValue['output3D'] = list(output3D)

    return outValue


def multipleDivide(input1=list(),
                   input2=list(),
                   operation='Multiply'):
    output = []
    npInput1 = numpy.array(input1)
    npInput2 = numpy.array(input2)

    if operation == 'Multiply':
        output = npInput1 * npInput2
    elif operation == 'Divide':
        output = npInput1 / npInput2

    return output


def getMatrix(node, attr='worldMatrix'):
    sel = OpenMaya.MSelectionList()
    sel.add(node)

    mobj = sel.getDependNode(0)
    mfn = OpenMaya.MFnDependencyNode(mobj)

    mtxAttr = mfn.attribute(attr)
    mtxPlug = OpenMaya.MPlug(mobj, mtxAttr)
    mtxPlug = mtxPlug.elementByLogicalIndex(0)

    mtxObj = mtxPlug.asMObject()
    mtxData = OpenMaya.MFnMatrixData(mtxObj)
    mtxValue = mtxData.matrix()

    return mtxValue


def decompMatrix(node, matrix, euler=True):
    rotOrderAttr = OpenMaya.MSelectionList()
    rotOrderAttr.add(node + '.rotateOrder')
    rotOrderPlug = rotOrderAttr.getPlug(0)
    rotOrder = rotOrderPlug.asInt()

    mTransformMtx = OpenMaya.MTransformationMatrix(matrix)
    trans = mTransformMtx.translation(OpenMaya.MSpace.kWorld)
    eulerRot = mTransformMtx.rotation()
    eulerRot.reorderIt(rotOrder)

    angles = radiansToDegrees(eulerRotation=eulerRot)
    scale = mTransformMtx.scale(OpenMaya.MSpace.kWorld)

    if euler:
        angles = eulerRot

    return [trans.x, trans.y, trans.z], angles, scale


def getPoleVectorInfo(namespace, side, part):
    up_jnt = addNamespace(_JOINT[part][0].format(side=side, type='FK'), namespace)
    middle_jnt = addNamespace(_JOINT[part][1].format(side=side, type='FK'), namespace)
    end_jnt = addNamespace(_JOINT[part][2].format(side=side, type='FK'), namespace)
    vectorCon = addNamespace(CON_LIST['IK'][part][1].format(side=side), namespace)

    up_jnt_mtx = getMatrix(up_jnt)
    middle_jnt_mtx = getMatrix(middle_jnt)
    end_jnt_mtx = getMatrix(end_jnt)

    up_jnt_trans = decompMatrix(up_jnt, up_jnt_mtx)
    middle_jnt_trans = decompMatrix(middle_jnt, middle_jnt_mtx)
    end_jnt_trans = decompMatrix(end_jnt, end_jnt_mtx)

    end_sum_up = plusMinusAvg(input3D=[up_jnt_trans[0], end_jnt_trans[0]])['output3D']
    end_sum_up_halfdivide = multipleDivide(input1=end_sum_up,
                                           input2=[2, 2, 2],
                                           operation='Divide')

    middle_jnt_vector = plusMinusAvg(input3D=[middle_jnt_trans[0], end_sum_up_halfdivide],
                                     operation='Subtract')['output3D']

    middle_jnt_vector_sum = plusMinusAvg(input3D=[end_sum_up_halfdivide, middle_jnt_vector],
                                         operation='Sum')['output3D']

    matrixList = [0, 0, 0, 0,
                  0, 0, 0, 0,
                  0, 0, 0, 0,
                  middle_jnt_vector_sum[0], middle_jnt_vector_sum[1], middle_jnt_vector_sum[2], 1]

    poleVector_worldMatrix = OpenMaya.MMatrix(matrixList)

    parentInverseMatrix = getMatrix(vectorCon, 'parentInverseMatrix')
    absMatrix = poleVector_worldMatrix * parentInverseMatrix

    outTransformMtx = OpenMaya.MTransformationMatrix(absMatrix)

    return {'con': vectorCon, 'matrix': outTransformMtx}


def setNodeToMFnTransform(node):
    sel = OpenMaya.MSelectionList()
    sel.add(node)
    dagpath = sel.getDependNode(0)
    transform_node = OpenMaya.MFnTransform(dagpath)

    return transform_node


def setTransformByMatrix(node, matrix):
    transform_node = setNodeToMFnTransform(node)
    transform_node.setTransformation(matrix)


class SnapKinematics():
    def __init__(self):
        self._mode = 'FK'
        self._node = None
        self._setHandLocalSpace = False
        self._startTime = None
        self._endTime = None

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value

    @property
    def node(self):
        return self._node

    @node.setter
    def node(self, value):
        self._node = value

    @property
    def setHandLocalSpace(self):
        return self._setHandLocalSpace

    @setHandLocalSpace.setter
    def setHandLocalSpace(self, value):
        self._setHandLocalSpace = value

    @property
    def startTime(self):
        return self._startTime

    @startTime.setter
    def startTime(self, value):
        self._startTime = value

    @property
    def endTime(self):
        return self._endTime

    @endTime.setter
    def endTime(self, value):
        self._endTime = value

    def getJoint(self,
                 node,
                 jointString,
                 side,
                 part
                 ):
        namespace = aniCommon.getNameSpace(node)
        joints = _JOINT[part]
        jointType = 'IK'
        jointName = jointString.format(side=side, type=jointType)

        if self.mode == 'IK':
            jointType = 'FK'
            jointName = joints[3].format(side=side, type=jointType)
            # local space
            if part == 'ARM' and self.setHandLocalSpace:
                jointName = joints[2].format(side=side, type=jointType)

        joint = addNamespace(jointName, namespace)
        return joint

    @aniCommon.undo
    def snap(self):
        namespace = None
        node = self.node.split("|")[-1]

        if cmds.referenceQuery(node, isNodeReferenced=True):
            namespace = aniCommon.getNameSpace(node)

        side = getControlerInfo(node, namespace)['side']
        part = getControlerInfo(node, namespace)['part']
        jointStringlist = _JOINT[part]
        conlist = CON_LIST[self.mode][part]

        for num, jointString in enumerate(jointStringlist):
            joint = self.getJoint(node, jointString, side, part)
            controlerName = conlist[num].format(side=side)
            controler = addNamespace(controlerName, namespace)

            mtx = getMatrix(joint)
            pInvMtx = getMatrix(controler, 'parentInverseMatrix')
            absMatrix = mtx * pInvMtx
            trans, eulerRotation, scale = decompMatrix(controler, absMatrix)

            # --------------- to localspace or not ----------------------------
            if self.setHandLocalSpace and part == 'ARM' and side == 'R':
                eulerRotation.x += math.radians(90)
            elif self.setHandLocalSpace and part == 'ARM' and side == 'L':
                eulerRotation.x += math.radians(-90)
            # -----------------------------------------------------------------

            rotation = radiansToDegrees(eulerRotation=eulerRotation)
            cmds.xform(controler, t=trans, ro=rotation)

            if self.mode == 'IK':
                poleVectorInfo = getPoleVectorInfo(namespace, side, part)
                pvTrans, pvRot, pvScale = decompMatrix(poleVectorInfo['con'], poleVectorInfo['matrix'])
                cmds.xform(poleVectorInfo['con'], t=pvTrans)
                break


    def getControlers(self):
        namespace = None
        node = self.node.split("|")[-1]
        controlers = list()

        if cmds.referenceQuery(node, isNodeReferenced=True):
            namespace = aniCommon.getNameSpace(node)

        side = getControlerInfo(node, namespace)['side']
        part = getControlerInfo(node, namespace)['part']

        for mode in CON_LIST:
            controlers += CON_LIST[mode][part]
        controlers = [addNamespace(con.format(side=side), namespace) for con in controlers]

        return controlers


    def bake(self, controlers):
        cmds.bakeResults(controlers,
                         simulation=True,
                         t=(self.startTime, self.endTime),
                         sampleBy=1,
                         disableImplicitControl=True,
                         preserveOutsideKeys=True,
                         sparseAnimCurveBake=False,
                         removeBakedAttributeFromLayer=False,
                         removeBakedAnimFromLayer=False,
                         bakeOnOverrideLayer=False,
                         minimizeRotation=True,
                         controlPoints=False,
                         shape=True)




#
# @aniCommon.undo
# def switch(type='FK', node=None):
#     namespace = None
#     node = node.split( "|" )[-1]
#
#     if cmds.referenceQuery( node, isNodeReferenced=True ):
#         namespace = aniCommon.getNameSpace( node )
#
#     side = getControlerInfo( node, namespace )['side']
#     part = getControlerInfo( node, namespace )['part']
#     jointlist = _JOINT[part]
#     conlist = CON_LIST[type][part]
#
#     for num, i in enumerate( jointlist ):
#         jointtype = 'IK'
#         jointname = i.format( side=side, type=jointtype )
#
#         if type == 'IK':
#             num = 0
#             jointtype = 'FK'
#             jointname = jointlist[3].format( side=side, type=jointtype )
#
#         print jointname
#         joint = addNamespace( jointname, namespace )
#         controlername = conlist[num].format( side=side )
#         con = addNamespace( controlername, namespace )
#
#         mtx = getMatrix( joint )
#         pInvMtx = getMatrix( con, 'parentInverseMatrix' )
#         absMatrix = mtx * pInvMtx
#
#         # con_transform_node = setNodeToMFnTransform(con)
#         trans, eulerRotation, scale = decompMatrix( con, absMatrix )
#         if type == 'IK':
#             trans = OpenMaya.MVector( trans )
#
#             # if part == 'ARM' and side == 'R':
#             #     eulerRotation.x += math.radians(90)
#             # elif part == 'ARM' and side == 'L':
#             #     eulerRotation.x += math.radians(-90)
#         # using maya.cmds for undo
#         rotation = radiansToDegrees( eulerRotation=eulerRotation )
#         cmds.xform( con, t=trans, ro=rotation )
#
#         # if type == 'IK' and part == 'LEG' and side == 'L':
#         #     cmds.rotate( 0, 90, 90, con, r=True, os=True, fo=True )
#         # elif type == 'IK' and part == 'LEG' and side == 'R':
#         #     cmds.rotate( 0, 90, -90, con, r=True, os=True, fo=True )
#
#         # con_transform_node.setTranslation(trans, OpenMaya.MSpace.kTransform)
#         # con_transform_node.setRotation(eulerRotation, OpenMaya.MSpace.kTransform)
#         # mtransformMtrix = OpenMaya.MTransformationMatrix(absMatrix)
#         # setTransformByMatrix(con, mtransformMtrix)
#
#         if type == 'IK':
#             poleVectorInfo = getPoleVectorInfo( namespace, side, part )
#             pvTrans, pvRot, pvScale = decompMatrix( poleVectorInfo['con'], poleVectorInfo['matrix'] )
#             cmds.xform( poleVectorInfo['con'], t=pvTrans )
#             # setTransformByMatrix(poleVectorInfo['con'], poleVectorInfo['matrix'])
#             break
