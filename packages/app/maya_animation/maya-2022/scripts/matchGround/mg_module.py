# encoding:utf-8

import logging
import string
import maya.cmds as cmds
import ANI_common
import dxRigUI as drg
import GHSnapKey.skModule as snapKey

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not cmds.pluginInfo('ghProjectMesh', q=True, l=True):
    cmds.loadPlugin('ghProjectMesh')

CON_LIST = {'Wolf' : {'L_IK_foreLeg_CON' : 'L_foreLeg_ground_CON',
                      'R_IK_foreLeg_CON' : 'R_foreLeg_ground_CON',
                      'L_IK_hindLeg_CON' : 'L_hindLeg_ground_CON',
                      'R_IK_hindLeg_CON' : 'R_hindLeg_ground_CON',
                      'C_IK_upBody_CON' : None,
                      'C_IK_lowBody_CON' : None
                      },
            'Normal Spider' : {'L_IK_MidLeg_CON': [17.138671287512327, 0.2754767133032181, -3.9474009951960123],
                               'R_IK_FrontLeg_CON': [-16.316789643475136, 0.2561000386087757, 3.5989260314146003],
                               'L_IK_FrontLeg_CON': [16.316789643475136, 0.2561000386087757, 3.5989260314146003],
                               'R_IK_MidLeg_CON': [-17.138671287512327, 0.2754767133032181, -3.9474009951960123],
                               'L_IK_BackLeg_CON': [18.457355625423215, 0.2790040863731855, -16.561804708696464],
                               'R_IK_BackLeg_CON': [-18.457355625423215, 0.2790040863731855, -16.561804708696464],
                               'root_CON':None
                               }
            }


def getNameSpace(node):
    nameSpace = string.join(node.split(":")[:-1], ":")
    logger.debug(u'Get namespace of : {0} : < {1} >'.format(node, nameSpace))
    return nameSpace


def getWorldTranslate(node):
    value = cmds.xform(node, q=True, rp=True, ws=True)
    return value


def createNormalBase(namespace):
    """Create polygon to get normal direction
    
    :param namespace: Namespace of dxRig node
    :return: Parent Group and polygon name
    """
    spiderLegs = ['R_IK_FrontLeg_CON',
                  'L_IK_FrontLeg_CON',
                  'L_IK_BackLeg_CON',
                  'R_IK_BackLeg_CON']
    nodeGroup = cmds.createNode('transform', n=namespace + "_NOR")
    legPositionNodeInfo = dict()
    for leg in spiderLegs:
        legCon = namespace + ":" + leg
        legPosition = cmds.xform(legCon, q=True, rp=True, ws=True)
        legPositionNodeInfo[ legCon ] = legPosition
    legPosition_FR = legPositionNodeInfo[namespace + ':R_IK_FrontLeg_CON']
    legPosition_FL = legPositionNodeInfo[namespace + ':L_IK_FrontLeg_CON']
    legPosition_RR = legPositionNodeInfo[namespace + ':R_IK_BackLeg_CON']
    legPosition_RL = legPositionNodeInfo[namespace + ':L_IK_BackLeg_CON']

    normalBase = cmds.polyCreateFacet(
        ch=True, tx=0, s=1, n=namespace + 'NormalBase',
        p=[(legPosition_FR[0], legPosition_FR[1], legPosition_FR[2]),
           (legPosition_FL[0], legPosition_FL[1], legPosition_FL[2]),
           (legPosition_RL[0], legPosition_RL[1], legPosition_RL[2]),
           (legPosition_RR[0], legPosition_RR[1], legPosition_RR[2])])[0]

    for num, leg in enumerate(spiderLegs):
        clusterHandle = cmds.cluster(normalBase + '.vtx[{0}]'.format(num))[1]
        cmds.pointConstraint(namespace + ":" + leg, clusterHandle, mo=False)
        cmds.parent(clusterHandle, nodeGroup)
        # clusterDcompMatrix = cmds.createNode('decomposeMatrix')
        # cmds.connectAttr(namespace + ":" + leg + '.worldMatrix[0]',
        #                  clusterDcompMatrix + '.inputMatrix')
        # cmds.connectAttr(clusterDcompMatrix + '.outputTranslate',
        #                  clusterHandle + '.translate')
    cmds.parent(normalBase, nodeGroup)
    logger.debug(u'Normal Base Object : {0}'.format(normalBase))
    return nodeGroup, normalBase


def matchNormal(namespace):
    """노말 방향과 바디의 로테이션 매칭
    
    :param namespace: Namespace of dxRig node
    :return:
    """
    rootCon = namespace + ":root_CON"
    rootConNul = cmds.listRelatives(rootCon, p=True)[0]
    #offsetNode = cmds.createNode('characterOffset', n=rootCon + "_RDN")
    offsetRotateNode = cmds.createNode('transform', n=namespace + '_OSR')
    offsetRotateNodeParent = cmds.createNode('transform', n=namespace + '_OSR_NUL')
    decomposeNode = cmds.createNode('decomposeMatrix', n=namespace + '_MG_DCM')
    cmds.parent(offsetRotateNode, offsetRotateNodeParent)
    cmds.connectAttr(namespace + ":move_CON.worldMatrix[0]", decomposeNode + ".inputMatrix")
    cmds.connectAttr(decomposeNode + ".outputRotateX", offsetRotateNodeParent + ".rotateX")
    cmds.connectAttr(decomposeNode + ".outputRotateY", offsetRotateNodeParent + ".rotateY")
    cmds.connectAttr(decomposeNode + ".outputRotateZ", offsetRotateNodeParent + ".rotateZ")
    cmds.connectAttr(rootCon + '.translateX', rootConNul + '.rotatePivotX')
    cmds.connectAttr(rootCon + '.translateY', rootConNul + '.rotatePivotY')
    cmds.connectAttr(rootCon + '.translateZ', rootConNul + '.rotatePivotZ')
    cmds.orientConstraint(offsetRotateNode,
                          rootConNul,
                          w=1, mo=True)
    return offsetRotateNode, offsetRotateNodeParent


def bakeControlers(nodes, startTime, endTime):
    logger.debug(u'Bake Controlers : {0}'.format(nodes))
    cmds.bakeResults(nodes,
                     simulation=False,
                     t=(startTime, endTime),
                     sampleBy=1, oversamplingRate=1,
                     disableImplicitControl=True,
                     preserveOutsideKeys=True,
                     sparseAnimCurveBake=False,
                     removeBakedAttributeFromLayer=False,
                     bakeOnOverrideLayer=False,
                     minimizeRotation=True,
                     controlPoints=False,
                     shape=True)
    cmds.filterCurve()


def matchWorldMatix(target, node, translate, rotate):
    dcmNode = cmds.createNode('decomposeMatrix', n=target + '_MG_DCM')
    cmds.connectAttr(target + '.worldMatrix[0]', dcmNode + '.inputMatrix')
    if translate:
        cmds.connectAttr(dcmNode + '.outputTranslateX', node + '.translateX')
        cmds.connectAttr(dcmNode + '.outputTranslateY', node + '.translateY')
        cmds.connectAttr(dcmNode + '.outputTranslateZ', node + '.translateZ')
    if rotate:
        cmds.connectAttr(dcmNode + '.outputRotateX', node + '.rotateX')
        cmds.connectAttr(dcmNode + '.outputRotateY', node + '.rotateY')
        cmds.connectAttr(dcmNode + '.outputRotateZ', node + '.rotateZ')
    return dcmNode


def gcdWolf(dxNodes, minTime, maxTime):
    """Bake Wolf rig's body controlers 
    and delete root controler's keyframes
    
    :param dxNodes: dxRig nodes
    :param minTime: Start time to bake keys
    :param maxTime: End time to bake keys
    """
    delList = list()
    bodyCons = list()
    bodyNuls = dict()
    for node in dxNodes:
        nameSpace = getNameSpace(node)
        upBodyCon = nameSpace + ':C_IK_upBody_CON'
        lowBodyCon = nameSpace + ':C_IK_lowBody_CON'

        bodyNuls[upBodyCon] = cmds.createNode('transform', n=upBodyCon + '_MG_NUL')
        upDcm = matchWorldMatix(target=upBodyCon,
                                node=bodyNuls[upBodyCon],
                                translate=True,
                                rotate=True)
        bodyNuls[lowBodyCon] = cmds.createNode('transform', n=lowBodyCon + '_MG_NUL')
        lowDcm = matchWorldMatix(target=lowBodyCon,
                                 node=bodyNuls[lowBodyCon],
                                 translate=True,
                                 rotate=True)
        bodyCons += [upBodyCon, lowBodyCon]
        delList += [upDcm, lowDcm]
        delList += [bodyNuls[upBodyCon], bodyNuls[lowBodyCon]]
    bakeControlers(nodes=bodyNuls.values(), startTime=minTime, endTime=maxTime)

    logger.debug(u'Delete Keyframes Of Root, Body CON')
    for node in dxNodes:
        nameSpace = getNameSpace(node)
        moveCon = nameSpace + ':C_IK_root_CON'
        upBodyCon = nameSpace + ':C_IK_upBody_CON'
        lowBodyCon = nameSpace + ':C_IK_lowBody_CON'
        cmds.cutKey([moveCon, upBodyCon, lowBodyCon])
        cmds.xform(moveCon, t=(0,0,0), ro=(0,0,0))
        cmds.parentConstraint(bodyNuls[upBodyCon], upBodyCon, w=1, mo=False)
        cmds.parentConstraint(bodyNuls[lowBodyCon], lowBodyCon, w=1, mo=False)
    bakeControlers(nodes=bodyCons, startTime=minTime, endTime=maxTime)
    logger.debug(u'Delete List : {}'.format(delList))
    cmds.delete(bodyNuls.values())


def matchGround(nodes, character, ground, minTime, maxTime):
    """Match character controler's transform to ground
    
    :param nodes: Selected nodes
    :param character: 'Wolf' or 'Normal Spider'
    :param ground: Ground mesh object
    :param minTime: 
    :param maxTime: 
    """
    dxNodes = list()
    for node in nodes:
        rootNode = ANI_common.getRootNode(node, type='dxNode')
        if rootNode not in dxNodes:
            dxNodes.append(rootNode)
    if character == 'Wolf':
        logger.debug(u'Bake GCD Wolf Body Controlers')
        gcdWolf(dxNodes, minTime, maxTime)

    cmds.autoKeyframe(state=False)
    cmds.currentTime(cmds.playbackOptions(q=True, min=True) - 51)
    for dxNode in dxNodes:
        drg.controlersInit("{0}.controlers".format(dxNode))
        ls_locateNode = list()
        nameSpace = getNameSpace(dxNode)
        cmds.undoInfo(ock=True)
        for controler_lib in CON_LIST[character]:
            controler = nameSpace + ":" + controler_lib
            locateNode_back = None

            if character == "Normal Spider" and controler_lib == "root_CON":
                controlerNul = controler
                while controlerNul.find("_NUL") == -1:
                    controlerNul = cmds.listRelatives(controlerNul, p=True)[0]
            else:
                controlerNul = cmds.listRelatives(controler, p=True)[0]

            logger.debug(u'Create Nodes For {0}'.format(controler))
            locateNode = cmds.createNode('transform', n=controler + "_LON")
            controlerDcm = cmds.createNode('decomposeMatrix', n=controler + "_DCM")
            moveConPos = cmds.xform(nameSpace + ':move_CON', q=True, rp=True, ws=True)
            if character == "Normal Spider" and controler_lib == "root_CON":
                locateNode_center = cmds.createNode('transform', n=controler + "_LON_C")
                locateNode_back = cmds.createNode('transform', n=controler + "_LON_B")
                pmaNode_front = cmds.createNode('plusMinusAverage', n=locateNode + "_PMA_F")
                pmaNode_back = cmds.createNode('plusMinusAverage', n=locateNode_back + "_PMA_B")
                cmds.xform(locateNode_center, t=(0, moveConPos[1], 0))
                cmds.xform(locateNode_back, t=(0, moveConPos[1], 0))
            cmds.xform(locateNode, t=(0, moveConPos[1], 0))

            logger.debug(u'Connect Nodes')
            cmds.connectAttr(controler + '.worldMatrix[0]',
                             controlerDcm + '.inputMatrix')
            cmds.connectAttr(controlerDcm + '.outputTranslateX',
                             locateNode + '.translateX')

            if character == "Normal Spider" and controler_lib == "root_CON":
                cmds.connectAttr(controlerDcm + '.outputTranslateX',
                                 locateNode_center + '.translateX')
                cmds.connectAttr(controlerDcm + '.outputTranslateZ',
                                 locateNode_center + '.translateZ')
                cmds.connectAttr(controlerDcm + '.outputTranslateX',
                                 locateNode_back + '.translateX')
                cmds.connectAttr(controlerDcm + '.outputTranslateZ',
                                 pmaNode_front + '.input1D[0]')
                cmds.setAttr(pmaNode_front + '.input1D[1]', 10)
                cmds.connectAttr(controlerDcm + '.outputTranslateZ',
                                 pmaNode_back + '.input1D[0]')
                cmds.setAttr(pmaNode_back + '.input1D[1]', -10)
                cmds.connectAttr(pmaNode_front + '.output1D', locateNode + '.translateZ')
                cmds.connectAttr(pmaNode_back + '.output1D', locateNode_back + '.translateZ')
            else:
                cmds.connectAttr(controlerDcm + '.outputTranslateZ',
                                 locateNode + '.translateZ')
            foot_groundCon = CON_LIST[character][controler_lib]
            if character == "Wolf" and foot_groundCon:
                foot_groundCon = nameSpace + ':' + foot_groundCon
                locateNodeDcm = cmds.createNode('decomposeMatrix', n=controler + '_LON_DCM')
                locateNodeMmx = cmds.createNode('multMatrix', n=controler + '_LON_MMX')
                cmds.connectAttr(locateNode + '.worldMatrix[0]',
                                 locateNodeMmx + '.matrixIn[0]')
                cmds.connectAttr(foot_groundCon + '.parentInverseMatrix',
                                 locateNodeMmx + '.matrixIn[1]')
                cmds.connectAttr(locateNodeMmx + '.matrixSum',
                                 locateNodeDcm + '.inputMatrix')
                cmds.connectAttr(locateNodeDcm + '.outputTranslateY',
                                 foot_groundCon + '.translateY', f=True)
            if character == "Normal Spider" and controler_lib == "root_CON":
                cmds.addAttr(controler, ln="weight_front", at="double", min=0, max=1, dv=0.35)
                cmds.setAttr(controler + '.weight_front', e=True, k=True)
                cmds.addAttr(controler, ln="weight_center", at="double", min=0, max=1, dv=0.3)
                cmds.setAttr(controler + '.weight_center', e=True, k=True)
                cmds.addAttr(controler, ln="weight_back", at="double", min=0, max=1, dv=0.35)
                cmds.setAttr(controler + '.weight_back', e=True, k=True)
                pConstraintNode = cmds.pointConstraint(locateNode,
                                                       controlerNul,
                                                       skip=('x', 'z'),
                                                       weight=0.35, mo=True)[0]
                cmds.pointConstraint(locateNode_center,
                                     controlerNul,
                                     skip=('x', 'z'),
                                     weight=0.3, mo=True)
                cmds.pointConstraint(locateNode_back,
                                     controlerNul,
                                     skip=('x', 'z'),
                                     weight=0.35, mo=True)
                topLegNul_left = cmds.listRelatives(nameSpace + ':L_IK_TopFrontLeg_CON', p=True)
                topLegNul_right = cmds.listRelatives(nameSpace + ':R_IK_TopFrontLeg_CON', p=True)
                cmds.pointConstraint(controler,
                                     topLegNul_left,
                                     skip=('x', 'z'),
                                     weight=1, mo=True)
                cmds.pointConstraint(controler,
                                     topLegNul_right,
                                     skip=('x', 'z'),
                                     weight=1, mo=True)
                cmds.connectAttr(controler + '.weight_front',
                                 pConstraintNode + '.' + locateNode.split(":")[-1] + 'W0')
                cmds.connectAttr(controler + '.weight_center',
                                 pConstraintNode + '.' + locateNode_center.split(":")[-1] + 'W1')
                cmds.connectAttr(controler + '.weight_back',
                                 pConstraintNode + '.' + locateNode_back.split(":")[-1] + 'W2')
                # cmds.setAttr(controler + '.weight_front', 0.5)
                # cmds.setAttr(controler + '.weight_back', 0.5)
            else:
                pConstraintNode = cmds.pointConstraint(locateNode,
                                                       controlerNul,
                                                       skip=('x', 'z'),
                                                       weight=1, mo=True)[0]
            logger.debug(u'Constraint To Ground : {}'.format(ground))
            cmds.geometryConstraint(ground, locateNode, w=1)

            if locateNode not in ls_locateNode:
                ls_locateNode.append(locateNode)
            if locateNode_back and locateNode_back not in ls_locateNode:
                cmds.geometryConstraint(ground, locateNode_center, w=1)
                cmds.geometryConstraint(ground, locateNode_back, w=1)
                ls_locateNode += [locateNode_back, locateNode_center]

        ostGroup = cmds.group(ls_locateNode, n=nameSpace + "_OST")

        if dxNode.find('normalSpider') != -1:
            logger.debug(u'Match Normal Direction (normalSpider Only)')
            logger.debug(u'     Node : {0}'.format(dxNode))
            matchNormalNodeGroup, normalBase = createNormalBase(namespace=nameSpace)
            rotateOffsetNode, rotateOffsetNodeNul = matchNormal(namespace=nameSpace)
            cmds.normalConstraint(normalBase,
                                  rotateOffsetNode,
                                  weight=1,
                                  aimVector=(0, 1, 0),
                                  upVector=(0, 1, 0),
                                  worldUpType="vector",
                                  worldUpVector=(0, 1, 0))
            cmds.parent(ostGroup, matchNormalNodeGroup)
            cmds.parent(rotateOffsetNodeNul, matchNormalNodeGroup)
            cmds.setAttr(matchNormalNodeGroup + '.visibility', 0)
    cmds.undoInfo(cck=True)
    cmds.autoKeyframe(state=True)


def zeroNul(character, controlers):
    for con in controlers:
        nulNode = cmds.listRelatives(con, ap=1)
        if nulNode and nulNode[0].find("_NUL") != -1:
            con_lib = con.split(":")[-1]
            if con_lib not in CON_LIST[character]:
                continue
            translate = CON_LIST[character][con_lib]
            if character == "Normal Spider" and translate:
                cmds.setAttr(nulNode[0] + '.translateX', translate[0])
                cmds.setAttr(nulNode[0] + '.translateY', translate[1])
                cmds.setAttr(nulNode[0] + '.translateZ', translate[2])
            else:
                cmds.setAttr(nulNode[0] + '.translateX', 0)
                cmds.setAttr(nulNode[0] + '.translateY', 0)
                cmds.setAttr(nulNode[0] + '.translateZ', 0)
            cmds.setAttr(nulNode[0] + '.rotateX', 0)
            cmds.setAttr(nulNode[0] + '.rotateY', 0)
            cmds.setAttr(nulNode[0] + '.rotateZ', 0)


def bakeDeleteNodes(nodes, character, minTime, maxTime):
    controlers = list()
    dxRigNodes = list()
    offsetNodes = list()
    mgNodes = list()
    controlers_lib = CON_LIST[character].keys()

    for node in nodes:
        dxRigNode = ANI_common.getRootNode(node, type='dxNode')
        if dxRigNode not in dxRigNodes:
            dxRigNodes.append(dxRigNode)
    for dxRig in dxRigNodes:
        namespace = getNameSpace(dxRig)
        for con in controlers_lib:
            conFullName = namespace + ":" + con
            groundCon = CON_LIST[character][con]
            if groundCon and isinstance(groundCon, str):
                groundConName = namespace + ":" + groundCon
                if groundConName not in controlers:
                    controlers.append(groundConName)
            if conFullName not in controlers:
                controlers.append(conFullName)

        if character == "Normal Spider":
            for mg in cmds.ls("{0}_NOR*|{0}_OST*|{0}:*_LON*".format(namespace)):
                if mg not in offsetNodes:
                    offsetNodes.append(mg)
            for mgr in cmds.ls("{0}_NOR*|{0}_OSR_NUL*|{0}_OSR*".format(namespace)):
                if mgr not in offsetNodes:
                    offsetNodes.append(mgr)
        elif character == "Wolf":
            for mg in cmds.ls("{0}_OST*|{0}:*_LON*".format(namespace)):
                if mg not in offsetNodes:
                    offsetNodes.append(mg)

        if character == "Normal Spider":
            for mg in cmds.ls("{0}_NOR*".format(namespace)):
                if mg not in mgNodes:
                    mgNodes.append(mg)
        elif character == "Wolf":
            for mg in cmds.ls("{0}_OST*".format(namespace)):
                if mg not in mgNodes:
                    mgNodes.append(mg)

    snapKey.bakeLocators(nodes=controlers, minTime=minTime, maxTime=maxTime)
    for node in offsetNodes:
        for mg in cmds.listConnections(node):
            if mg not in mgNodes:
                mgNodes.append(mg)
    cmds.delete(mgNodes)
    zeroNul(character, controlers)
    snapKey.loc2ctrl(minTime=minTime, maxTime=maxTime)


def matchGroundNew(nodes, character, ground):
    dxNodes = list()
    for node in nodes:
        rootNode = ANI_common.getRootNode(node, type='dxNode')
        if rootNode not in dxNodes:
            dxNodes.append(rootNode)

    groundShape = cmds.listRelatives(ground, s=True)[0]
    for node in dxNodes:
        cmds.undoInfo(ock=True)
        namespace = getNameSpace(node)

        for con_lib in CON_LIST[character]:
            con = namespace + ":" + con_lib
            controlerNul = cmds.listRelatives(con, p=True)[0]
            projectMeshNode = cmds.createNode('ghProjectMesh', n=con + "_GPM")
            cmds.setAttr(projectMeshNode + ".WeightRotate", 0)
            cmds.connectAttr(groundShape + ".worldMesh",
                             projectMeshNode + ".inputMeshTarget")
            cmds.connectAttr(con + ".worldMatrix[0]",
                             projectMeshNode + ".inputMatrix")
            cmds.connectAttr(projectMeshNode + ".outputTranslateY",
                             controlerNul + ".translateY")
            cmds.connectAttr(controlerNul + ".parentInverseMatrix[0]",
                             projectMeshNode + ".inParentInverseMatrix")
            if character == 'Wolf':
                groundCon_lib = CON_LIST[character][con_lib]
                if groundCon_lib:
                    groundCon = namespace + ":" + groundCon_lib
                    pmaNode = cmds.createNode('plusMinusAverage', n=groundCon + "_PMA")
                    cmds.setAttr(pmaNode + ".input1D[0]", 0.411)
                    cmds.connectAttr(projectMeshNode + ".outputTranslateY",
                                     pmaNode + ".input1D[1]")
                    cmds.connectAttr(pmaNode + ".output1D",
                                     groundCon + ".translateY", f=True)
                if con_lib == 'C_IK_upBody_CON' or \
                                con_lib == 'C_IK_lowBody_CON':
                    cmds.setAttr(projectMeshNode + ".offsetY", 7.469827)
                else:
                    cmds.setAttr(projectMeshNode + ".offsetY", 0.588)
            elif character == 'Normal Spider':
                if con_lib == 'root_CON':
                    cmds.setAttr(projectMeshNode + ".offsetY", 7.406172)
        cmds.undoInfo(cck=True)