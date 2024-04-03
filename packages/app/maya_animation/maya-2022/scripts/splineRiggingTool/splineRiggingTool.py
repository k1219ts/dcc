# encoding=utf-8
# Spline Tool ver. 0.05

# Import Package Modules
from splineUtil.splineUtil import *
from util.homeNul import *
import ribbonRig as RR
reload(RR)

# Import Python Modules
import os

# Import Maya Modules
import maya.cmds as cmds
import maya.mel as mel
import maya.OpenMayaUI as omui

# Import PySide2 Modules
import PySide2.QtGui as QtGui
import PySide2.QtCore as QtCore
import PySide2.QtWidgets as QtWidgets
import PySide2.QtUiTools as QtUiTools
from PySide2.QtWidgets import QMessageBox

# Import Shiboken2 Modules
from shiboken2 import wrapInstance

# Base path
BASE_PATH = os.path.dirname(__file__)

def getMayaWin():
    ptr = omui.MQtUtil.mainWindow()
    
    mayaMainWindow = wrapInstance(long(ptr), QtWidgets.QMainWindow)
    return mayaMainWindow

def undo(func):
    def wrapper(*args, **kwargs):
        cmds.undoInfo(openChunk=True)
        try:
            ret = func(*args, **kwargs)
        finally:
            cmds.undoInfo(closeChunk=True)
        return ret
    return wrapper

class SplineRig(QtWidgets.QMainWindow):
    def __init__(self, parent=getMayaWin()):
        super(SplineRig, self).__init__(parent)

        uiFile = os.path.join(BASE_PATH, 'splineRiggingTool.ui')
        loadUI = QtUiTools.QUiLoader()

        # load plugin
        if not cmds.pluginInfo('matrixNodes.so', q=True, loaded=True):
            cmds.loadPlugin('matrixNodes.so')

        self.ui = loadUI.load(uiFile)
        self.ui.setWindowFlags(QtCore.Qt.Window)
        self.ui.setWindowTitle('Spline Rigging Tool')

        #self.rootConRadioB()
        self.createConnections()

    # def rootConRadioB(self):
    #     self.ui.FKIK_radioButton.clicked.connect(self.radioButtonState)
    #     self.ui.IKFK_radioButton.clicked.connect(self.radioButtonState)


    def createConnections(self):
        self.ui.startCon_pushButton.clicked.connect(self.addConName)
        self.ui.endCon_pushButton.clicked.connect(self.addConName)

        self.ui.riggingCreate_pushButton.clicked.connect(self.createRig)

        self.ui.deleteRig_pushButton.clicked.connect(self.deleteRig)

    # def radioButtonState(self):
    #     print self.ui.FKIK_radioButton.objectName()
    #     print self.ui.FKIK_radioButton.isChecked()
    #     print self.ui.IKFK_radioButton.objectName()
    #     print self.ui.IKFK_radioButton.isChecked()
    def addConName(self):
        conList = cmds.ls(sl=True, type='nurbsCurve', dag=True)

        uiName = self.sender().objectName()

        if len(conList) > 0:
            eval('self.ui.%s_lineEdit.setText(conList[0][:-5])' % uiName.split('_')[0])
        else:
            eval('self.ui.%s_lineEdit.clear()' % uiName.split('_')[0])
    @undo
    def createRig(self):
        # Get length
        startConNode = self.ui.startCon_lineEdit.text()
        endConNode = self.ui.endCon_lineEdit.text()

        startConstNode = cmds.listConnections(startConNode, type='constraint')[0]
        startJointNode = cmds.listConnections(startConstNode, type='joint')[0]

        endConstNode = cmds.listConnections(endConNode, type='constraint')[0]
        endtJointNode = cmds.listConnections(endConstNode, type='joint')[0]

        pos1 = cmds.xform(startJointNode, ws=True, t=True, q=True)
        pos2 = cmds.xform(endtJointNode, ws=True, t=True, q=True)

        curveLength = getDistance(pos1, pos2)


        # Joint List
        startConNum = startConNode.split('_')[-2].split('addFk')[-1]

        endConNum = endConNode.split('_')[-2].split('addFk')[-1]

        jointNum = len(range(int(endConNum) - int(startConNum))) + 1


        partName = startConNode.split('_')[0]
        jointList = []
        for i in range(jointNum):
            #jointList.append('addFk%s_%s' % (i + jointNum, 'JNT'))
            #jointList.append('%s_%s_JNT' % (partName, str(i+1).zfill(3)))
            jointList.append('%s_%s_JNT' % (partName, str(int(startConNum)+i).zfill(3)))


        # Controller Num
        conNum = self.ui.conNum_spinBox.value()

        # Scale Space
        #scaleSpace = self.ui.scaleSpace_spinBox.value()

        
        
        # Prefix
        #get top node ===> Asset Name
        rigTopNode = cmds.ls(startConNode, fl=True, l=True, o=True)[0].split('|')[1]
        nameSpace = rigTopNode.split(':')[0]
        assetPartName = startConNode.split(':')[-1].split('_')[0]

        prefix = assetPartName



        # RUN ~ ~ !!
        # curveLength, prefix, jointList, conNum, nameSpace, fkCon=False, scaleSpace=0
        rrI = RR.RibbonRig(curveLength, prefix, jointList, conNum, nameSpace, False, scaleSpace=0)
        rootConNul = rrI.createIkRig()





        # SCALE SCALE SCALE
        if cmds.objExists( nameSpace + ':' + 'place_CON' ):
            #cmds.connectAttr( 'place_CON.initScale', '%s.input1X' % gs )
            #print topConRig[2]  ===   'topAddfk_skinJoint_GRP'
            #print topConRig[8]  ===   'topAddfk_Controller_GRP'
            cmds.connectAttr( nameSpace + ':' + 'place_CON.initScale', '%s.scaleX' % rootConNul[2] )
            cmds.connectAttr( nameSpace + ':' + 'place_CON.initScale', '%s.scaleY' % rootConNul[2] )
            cmds.connectAttr( nameSpace + ':' + 'place_CON.initScale', '%s.scaleZ' % rootConNul[2] )
            cmds.connectAttr( nameSpace + ':' + 'place_CON.initScale', '%s.scaleX' % rootConNul[8] )
            cmds.connectAttr( nameSpace + ':' + 'place_CON.initScale', '%s.scaleY' % rootConNul[8] )
            cmds.connectAttr( nameSpace + ':' + 'place_CON.initScale', '%s.scaleZ' % rootConNul[8] )






        ## match position....
        #cmds.delete(cmds.pointConstraint(startConNode, rootConNul[0]))
        cmds.delete(cmds.parentConstraint(startConNode, rootConNul[0]))





        #hi
        conStartP = cmds.listRelatives(startConNode, p=True)
        
        conStartPP = cmds.listRelatives(conStartP, p=True)

        
        conStartPosNode = conStartP + conStartPP
        sNode = duplicateWithoutChildren(conStartPosNode)
        # duplicateWithoutchildren: 노드리스트를 넣으면 해당 노드들의 하위를 제외 한 상태로 복사 후 리스트로 리턴


        cmds.parent(sNode, w=True)

        for x in range(len(sNode)-1):
            cmds.parent(sNode[x+1], sNode[x])


        cmds.delete(cmds.parentConstraint(rootConNul[1][0], sNode[0]))
        cmds.parent(rootConNul[1][0], sNode[-1])
        

        cmds.parent(sNode[0], rootConNul[2])

        #rename
        cmds.rename(sNode[0], '%s_%s'%(nameSpace, sNode[0].replace('_NUL', '_refNUL')))
        refTopNul = cmds.rename(sNode[1], '%s_%s'%(nameSpace, sNode[1].replace('_CON', '_refNUL')))


        #conStartPP




        # Attach
        #parentNodeList, chaldNodeList
        #testL = [ 'test%s' % i for i in range(10,20)  ]
        startNum = startConNode.split('_')[1]
        endNum = endConNode.split('_')[1]
        #chaldNodeList = [nameSpace + ':%s_%s_NUL' % (partName, str(i).zfill(3)) for i in range(int(startNum), int(endNum)+1)]
        chaldNodeList = ['%s_%s_NUL' % (partName, str(i).zfill(3)) for i in range(int(startNum), int(endNum)+1)]

        addFkLayer(rootConNul[1], chaldNodeList)



        # parentConstraints and Scale
        # cmds.parentConstraint(nameSpace +':'+ sNode[1], rootConNul[0], mo=True)
        # cmds.parentConstraint(nameSpace +':'+ sNode[1], refTopNul, mo=True)
        cmds.parentConstraint(nameSpace +':'+ sNode[1], rootConNul[0], mo=True)
        cmds.parentConstraint(nameSpace +':'+ sNode[1], refTopNul, mo=True)

        cmds.setAttr('%s.scaleX' % refTopNul, l=False, k=True)
        cmds.setAttr('%s.scaleY' % refTopNul, l=False, k=True)
        cmds.setAttr('%s.scaleZ' % refTopNul, l=False, k=True)

        #cmds.scaleConstraint(nameSpace + ':' + 'transform_GRP', refTopNul, mo=True)
        #cmds.scaleConstraint(nameSpace + ':' + 'transform_GRP', rootConNul[0], mo=True)



        #rootConNul[-1] ======== endConName
        #cmds.orientConstraint( rootConNul[3], nameSpace + ':%s_%s_CON' % (partName, str(endConNum).zfill(3)) )
        cmds.orientConstraint( rootConNul[3], '%s_%s_CON' % (partName, str(endConNum).zfill(3)) )



        # vis addFkCon
        jointName = cmds.listRelatives(startConNode.replace('_CON','_JNT'))[0]
        skinJointList = cmds.ls(jointName.replace(jointName.split('_')[-2], '*'))
        if len(chaldNodeList) == len(skinJointList):
            pass
        else:

            # visSet
            for x in range(len(chaldNodeList)):
                chaldFkCon = chaldNodeList[x].replace('_NUL', '_CON')
                chaldFkConShape = cmds.listRelatives( chaldFkCon, s=True )[0]
                #print(chaldFkConShape)
                #tentacle:addFk_001_CONShape
                # 첫번째 컨트롤러는 항상 숨기지 않음 
                if chaldFkConShape.split('_')[-2] == '001':
                    pass
                else: 
                    cmds.setAttr('%s.visibility' % chaldFkConShape, 0)







        ####
        

        topConNum = self.ui.fkIkConNum_spinBox.value()


        # curveLength, prefix, jointList, conNum, nameSpace, fkCon=False, scaleSpace=0
        testRig = RR.RibbonRig(curveLength, 'top' + prefix.capitalize(), range(conNum), topConNum, nameSpace, True, scaleSpace=0)
        topConRig = testRig.createIkRig()







        
        














        ### Layer Attach

        cmds.delete(cmds.parentConstraint(rootConNul[0], topConRig[0]))
        cmds.setAttr('%sShape.visibility' % topConRig[0].replace('_NUL', '_CON'), 0)





        for x in range(len(rootConNul[4])):
            cmds.parentConstraint( cmds.listRelatives(topConRig[5][x], p=True), rootConNul[4][x], mo=True)


        




        cmds.delete(cmds.parentConstraint(rootConNul[0], topConRig[6][0]))






        topConNulList = topConRig[4] + topConRig[6]
        topConList = []
        for x in range(len(topConNulList)):
            topConShape = cmds.ls(topConNulList[x], dag=True, type='nurbsCurve')[0]
            topCon = cmds.listRelatives(topConShape, p=True)[0]
            topConList.append(topCon)



        controllerResize(topConList, 0.5)








        #fkConNameList, conNulNameList
        for x in range(len(topConRig[6])):
            
            cmds.parentConstraint( topConRig[6][x].replace('_NUL', '_CON'), topConRig[4][x], mo=True )

        cmds.parentConstraint(nameSpace +':'+ sNode[1], topConRig[6][0], mo=True)








        grp = cmds.group(rootConNul[7], topConRig[7], name=nameSpace + '_' + prefix + '_GRP')
        #outlinerColor set
        cmds.setAttr('%s.useOutlinerColor' % grp, 1)
        cmds.setAttr('%s.outlinerColor' % grp, 1, 1, 0 )



        if cmds.objExists(nameSpace + '_GRP'):
            cmds.parent(grp, nameSpace + '_GRP')
        else:
            rigGrp = cmds.group(n=nameSpace + '_GRP', em=True)
            cmds.parent(grp, nameSpace + '_GRP')





        # SCALE SCALE SCALE
        if cmds.objExists( nameSpace + ':' + 'place_CON' ):
            #cmds.connectAttr( 'place_CON.initScale', '%s.input1X' % gs )
            #print topConRig[2]  ===   'topAddfk_skinJoint_GRP'
            #print topConRig[8]  ===   'topAddfk_Controller_GRP'
            cmds.connectAttr( nameSpace + ':' + 'place_CON.initScale', '%s.scaleX' % topConRig[2] )
            cmds.connectAttr( nameSpace + ':' + 'place_CON.initScale', '%s.scaleY' % topConRig[2] )
            cmds.connectAttr( nameSpace + ':' + 'place_CON.initScale', '%s.scaleZ' % topConRig[2] )
            cmds.connectAttr( nameSpace + ':' + 'place_CON.initScale', '%s.scaleX' % topConRig[8] )
            cmds.connectAttr( nameSpace + ':' + 'place_CON.initScale', '%s.scaleY' % topConRig[8] )
            cmds.connectAttr( nameSpace + ':' + 'place_CON.initScale', '%s.scaleZ' % topConRig[8] )

        # skinJointList ConnectAttr 부분이 롤백되어 비활성화 하였습니다. -- 12/30 상경

        # skinJointList = cmds.ls('%s:%sJoint_NUL' % (nameSpace, prefix), dag=True, type='joint')
        # print(skinJointList)
        # if cmds.objExists( nameSpace + ':' + 'place_CON' ):
        #     for x in range(len(skinJointList)):
        #         cmds.connectAttr( nameSpace + ':' + 'place_CON.initScale', '%s.scaleX' % skinJointList[x] )
        #         cmds.connectAttr( nameSpace + ':' + 'place_CON.initScale', '%s.scaleY' % skinJointList[x] )
        #         cmds.connectAttr( nameSpace + ':' + 'place_CON.initScale', '%s.scaleZ' % skinJointList[x] )



        
        # NurbsPlane VIS
        #rootConNul[7], topConRig[7]
        topIkNurbsShape = cmds.ls(rootConNul[7], dag=True, type='nurbsSurface', ni=True )
        topIkNurbsPlane = cmds.listRelatives(topIkNurbsShape, p=True)[0]
        subIkNurbsShape = cmds.ls(topConRig[7], dag=True, type='nurbsSurface', ni=True )
        subIkNurbsPlane = cmds.listRelatives(subIkNurbsShape, p=True)[0]
        cmds.addAttr( rootConNul[0].replace('NUL', 'CON'), ln="topIkNurbsVis", at="enum", en="off:on:", k=True )
        cmds.addAttr( rootConNul[0].replace('NUL', 'CON'), ln="subIkNurbsVis", at="enum", en="off:on:", k=True )

        cmds.connectAttr('%s.topIkNurbsVis' % rootConNul[0].replace('NUL', 'CON'), '%s.visibility' % topIkNurbsPlane)
        cmds.connectAttr('%s.subIkNurbsVis' % rootConNul[0].replace('NUL', 'CON'), '%s.visibility' % subIkNurbsPlane)

                




        print('The %s_GRP rigging is complete!' % prefix)




        # addFk follow space setup
        #endConNode, chaldNodeList
        #print ( '%s_______%s' % (endConNode, chaldNodeList[-1].replace('_NUL', '_CON')) )
        if endConNode == chaldNodeList[-1].replace('_NUL', '_CON'):
            pass
        else:
            fkSpaceLocator = fkTentacleSpaceBlend(endConNode)
            cmds.parent(fkSpaceLocator, rootConNul[7])

            cmds.select(cl=True)





    def deleteRig(self):

        selRigGrp = cmds.ls(sl=True)

        #rigTopNode = cmds.ls(startConNode, fl=True, l=True, o=True)[0].split('|')[1]
        
        if len(selRigGrp) == 1:
            signal = self.messageCall(selRigGrp)
            if signal == 'Yes':

                rigTop = cmds.ls(selRigGrp[0], fl=True, l=True, o=True)[0].split('|')[1]
                nameSpace = rigTop.split('_')[0]


                rigAssetName = selRigGrp[0].split('_')[1]

                #gangsterKssO:addFkA_NUL
                disConShape = cmds.ls(nameSpace + ':' + rigAssetName + '_NUL', dag=True, type='nurbsCurve')[1:]

                #mel.eval('source "generateChannelMenu.mel./";')
                mel.eval('source "/usr/autodesk/maya2018/scripts/startup/channelBoxCommand.mel";')

                for x in disConShape:
                    cmds.setAttr('%s.visibility' % x, 1)

                    reN = x.replace('_CONShape', '_NUL')


                    mel.eval( 'CBdeleteConnection "%s.tx";' % reN )
                    mel.eval( 'CBdeleteConnection "%s.ty";' % reN )
                    mel.eval( 'CBdeleteConnection "%s.tz";' % reN )
                    
                    mel.eval( 'CBdeleteConnection "%s.rx";' % reN )
                    mel.eval( 'CBdeleteConnection "%s.ry";' % reN )
                    mel.eval( 'CBdeleteConnection "%s.rz";' % reN )
                    
                    mel.eval( 'CBdeleteConnection "%s.sx";' % reN )
                    mel.eval( 'CBdeleteConnection "%s.sy";' % reN )
                    mel.eval( 'CBdeleteConnection "%s.sz";' % reN )
                    
                    mel.eval( 'CBdeleteConnection "%s.v";' % reN )

                cmds.delete(selRigGrp[0])

                print('Delete  <  %s  >  Rigging' % selRigGrp[0])

            else:
                pass
                #cmds.warning("경고메시지")# 혹은 작동 정지

        else:
            cmds.warning("하나의 리깅 그룹을 선택 해주세요!")# 혹은 작동 정지



    def messageCall(self, selRigGrp):
        title = '리깅을 제거 하시겠습니까?'
        message = "warning!!warning!!\n %s delete Rigging!!"%selRigGrp[0]
        result = cmds.confirmDialog(t=title, icn='warning', b=['Yes', 'No'], m=message, ds='No')
        # cmds.confirmDialog(t=팝업창 타이틀, icn=아이콘타입(warning, information, question 등), b=[버튼 선택지 문자열: 실행 시 리턴], m=내용, ds=디폴트 선택지 문자열)

        return result





        # for x in disConShape:
        #     cmds.setAttr('%s.visibility' % x, 1)

        # disCon = cmds.listRelatives(disConShape, p=True)

        # disConNul = cmds.listRelatives(disCon, p=True)

        # for x in disConNul:
        #     mel.eval( 'CBdeleteConnection "%s.tx"' % x )
        #     mel.eval( 'CBdeleteConnection "%s.ty"' % x )
        #     mel.eval( 'CBdeleteConnection "%s.tz"' % x )
            
        #     mel.eval( 'CBdeleteConnection "%s.rx"' % x )
        #     mel.eval( 'CBdeleteConnection "%s.ry"' % x )
        #     mel.eval( 'CBdeleteConnection "%s.rz"' % x )
            
        #     mel.eval( 'CBdeleteConnection "%s.sx"' % x )
        #     mel.eval( 'CBdeleteConnection "%s.sy"' % x )
        #     mel.eval( 'CBdeleteConnection "%s.sz"' % x )
            
        #     mel.eval( 'CBdeleteConnection "%s.v"' % x )









        


def openUI():
    global Window
    try:
        Window.close()
        Window.deleteLater()
    except: pass
    Window = SplineRig()
    Window.ui.show()