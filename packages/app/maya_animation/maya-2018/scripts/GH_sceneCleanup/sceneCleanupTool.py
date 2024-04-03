# -*- coding:utf-8 -*-

"""
import sceneCleanupTool as sct
reload(sct)
sct.SceneCleanupMain().SceneCleanupUI()
"""

import maya.cmds as cmds
import maya.mel as mm

createHUIscriptNodeString = """
import HeadsUpDisplay as hud

if cmds.objExists('frameCounterUpdate') == 1:
    allExpr = cmds.ls("frameCounterUpdate", type = "expression")
    for curExpr in allExpr:
        cmds.delete(curExpr)

hud.HeadsUpDisplayMain().offAllHud()
hud.HeadsUpDisplayMain().mg_removeHUD()
hud.expressionRemover()
"""

"""
dispLayers = cmds.ls(type = 'displayLayer')

for dislyr in dispLayers:
    if dislyr != 'defaultLayer':
        cmds.delete(dislyr)
"""


# 레퍼런스가 아닌 컨스트레인 노드 삭제
class SceneCleanupMain:
    def __init__(self):
        self.TWtmp = cmds.ls('tw_*')

    def SceneCleanupUI(self):
        if cmds.window('SceneCleanup_win', ex=1): cmds.deleteUI('SceneCleanup_win')
        cmds.window('SceneCleanup_win', title="BARIQUANT", sizeable=0)

        cmds.columnLayout('SceneCleanupMainColumn')
        cmds.frameLayout('SceneCleanup_layout', labelVisible = False, label="   main", collapsable=True, collapse=0, marginHeight=10, marginWidth=10)
        cmds.rowColumnLayout('SceneCleanupRCL', numberOfColumns=2, columnWidth=[(1, 130),(2, 100)])

        self.CleanupOptionBox = cmds.checkBox(l = "custom", h = 20, v = True, cc = self.optionChangeCommand)
        cmds.button(l = "ReBoot", c = "sct.SceneCleanupMain().SceneCleanupUI()")

        cmds.setParent("..")

        cmds.rowColumnLayout('SceneCleanupRCL2', numberOfColumns=1, columnWidth=[(1, 200)], columnSpacing = [1, 15])
        cmds.frameLayout(labelVisible = False, marginWidth = 10, mh = 10)

        self.bkcChBox = cmds.checkBox(l = "Bake, Delete Constraint", h = 20, en = True, v = False)
        self.isplyrChBox = cmds.checkBox(l = "Delete Display Layers", h = 20, en = True, v = False)
        self.DelNodeChBox = cmds.checkBox( l = "Delete Unused Node", h = 20, en = True, v = True)
        self.MrgAnmLyrChBox = cmds.checkBox( l = "Merge Anim Layers", h = 20, en = True, v = False)
        self.DelBAR_planeChBox = cmds.checkBox( l = "Delete imagePlane Bar", h = 20, en = True, v = False)
        self.DelUnkownNodeChBox = cmds.checkBox( l = "Delete Unkown Node", h = 20, en = True, v = True)

        #self.DelScrtNodeChBox = cmds.checkBox( l = "Delete ScriptNode", h = 20, en = False, v = True)

        cmds.setParent("..")
        cmds.setParent("..")

        cmds.separator(style = "in", h = 30)

        cmds.rowColumnLayout( numberOfColumns=1, columnWidth=[(1, 230)])
        cmds.button(l = "CleanUp Scene", h = 30,c = self.DoItCmd)
        cmds.text(l = "")

        cmds.separator(style = "in", h = 30)

        cmds.text(l = "| Delete TimeWarp Node |",  h = 20, fn = "fixedWidthFont")
        cmds.text(l = "")

        if self.TWtmp != []:
            cmds.text(l = "Time Warp 노드가 있습니다.\n베이크하고 버튼을 눌러주세요")
            cmds.button(l = "DELETE TIMEWARP", h = 30,c = self.DeleteWarp)
        else:
            cmds.text(l = "Time Warp 노드 없음..")
            cmds.button(l = "DELETE TIMEWARP", en = False, h = 30,c = self.DeleteWarp)

        cmds.showWindow('SceneCleanup_win')


    def optionChangeCommand(self, *args):
        if cmds.checkBox(self.CleanupOptionBox, q = True, v = True):
            cmds.checkBox(self.bkcChBox, e = True, en = True)
            cmds.checkBox(self.isplyrChBox, e = True, en = True)
            cmds.checkBox(self.DelNodeChBox, e = True, en = True)
            cmds.checkBox(self.MrgAnmLyrChBox, e = True, en = True)
            cmds.checkBox(self.DelBAR_planeChBox, e = True, en = True)
            cmds.checkBox(self.DelUnkownNodeChBox, e = True, en = True)
        else:
            cmds.checkBox(self.bkcChBox, e = True, en = False)
            cmds.checkBox(self.isplyrChBox, e = True, en = False)
            cmds.checkBox(self.DelNodeChBox, e = True, en = False)
            cmds.checkBox(self.MrgAnmLyrChBox, e = True, en = False)
            cmds.checkBox(self.DelBAR_planeChBox, e = True, en = False)
            cmds.checkBox(self.DelUnkownNodeChBox, e = True, en = False)

    def DoItCmd(self, *args):
        if cmds.checkBox(self.bkcChBox, q = 1, v = True):
            self.delConstrains()

        if cmds.checkBox(self.isplyrChBox, q = 1, v = True):
            self.delDispLYR()

        if cmds.checkBox(self.DelNodeChBox, q = 1, v = True):
            self.deleteUnusedNodeCmd()

        if cmds.checkBox(self.MrgAnmLyrChBox, q = 1, v = True):
            self.AnimLYRmerge()

        if cmds.checkBox(self.DelBAR_planeChBox, q = 1, v = True):
            self.DeleteBarPlane()

        if cmds.checkBox(self.DelUnkownNodeChBox, q = 1, v = True):
            self.deleteUnknownNode()

        cmds.confirmDialog( title='info', message='cleanup Complete.', button=['close'] )

        #if cmds.checkBox(self.DelScrtNodeChBox, q = 1, v = True):
            #self.DelScriptNode()
        #self.DeleteWarp()

    def delConstrains(self, *args):
        StartTime = cmds.playbackOptions(q=1, min = True) - 1
        EndTime = cmds.playbackOptions(q=1, max = True) + 1

        consListA = cmds.ls(type = 'parentConstraint')
        consListB = cmds.ls(type = 'pointConstraint')
        consListC = cmds.ls(type = 'orientConstraint')
        consListD = cmds.ls(type = 'aimConstraint')
        consListE = cmds.ls(type = 'scaleConstraint')

        AllConstrainList = consListA + consListB + consListC + consListD + consListE

        for tempPrCons in AllConstrainList:
            if cmds.referenceQuery( tempPrCons, isNodeReferenced=True ) == 0:
                ConnetedObj = cmds.listRelatives(tempPrCons, parent=True, type='transform')[0]
                cmds.bakeResults( ConnetedObj, simulation = False, dic = False, t=(StartTime, EndTime) )
                mm.eval("performEulerFilter graphEditor1FromOutliner")
                cmds.keyTangent( ConnetedObj, itt = 'linear',  ott = 'linear', animation = 'objects' )
                cmds.delete(tempPrCons)


    # 디스플레이 레이어 삭제

    def delDispLYR(self, *args):
        dispLayers = cmds.ls(type = 'displayLayer')

        for dislyr in dispLayers:
            if dislyr != 'defaultLayer':
                if cmds.referenceQuery( dislyr, isNodeReferenced=True ) == 0:
                    cmds.delete(dislyr)


    # 사용하지 않는 노드 삭제
    def deleteUnusedNodeCmd(self, *args):
        mm.eval('hyperShadePanelMenuCommand("hyperShadePanel1", "deleteUnusedNodes");')

        # delete turtle Node
        turtleNode = ['TurtleBakeLayerManager','TurtleDefaultBakeLayer','TurtleRenderOptions','TurtleUIOptions']
        rmanList = cmds.ls("rman*")
        rendermanList = cmds.ls("renderMan*")

        if rmanList or rendermanList:
            cmds.delete(rmanList, rendermanList)

        for i in turtleNode:
            try:
                cmds.lockNode( i, lock=False )
                cmds.delete(i)
            except:
                pass


        #self.deleteUnknownNode()

    def deleteUnknownNode(self):
        #unknownNode = cmds.ls(type = ("unknown", "unknownDag", "unknownTransform"))
        unknownNode = cmds.ls(type = ("unknown"))

        for i in unknownNode:
            lockState = cmds.lockNode(i, q = True)[0]

            if lockState:
                cmds.lockNode(i, l = False)

        try:
            cmds.delete(unknownNode)
        except:
            pass

        print "Delete : ",unknownNode

        #cmds.confirmDialog( title='info', message='cleanup Complete.', button=['close'] )



 # AnimLayer 체크, merge

    def AnimLYRmerge(self, *args):
        animLayers = cmds.ls(type = 'animLayer')

        if animLayers != ['BaseAnimation'] and animLayers != []:
            LYRstr = ""

            for al in animLayers:
                LYRstr = LYRstr + '"%s"' %al
                if al != animLayers[-1]:
                    LYRstr = LYRstr  + ","

            mm.eval( 'string $layers[]={%s}; layerEditorMergeAnimLayer( $layers, 0 )' % LYRstr )


    def DelScriptNode(self, *args):
        if cmds.ls('HUIdelNode*') == []:
            cmds.scriptNode( scriptType = 1, beforeScript = createHUIscriptNodeString, name = 'HUIdelNode', sourceType = "python")

    #delConstrains()
    #delDispLYR()
    #AnimLYRmerge()


    # time warp 삭제
    def DeleteWarp(self, *args):
        cmds.delete(cmds.ls('tw_*'))

    def DeleteBarPlane(self):
        imgpln = cmds.ls(type = "imagePlane")

        for i in imgpln:
            if i.find("BAR_plane") != -1:
                i_trans = cmds.listRelatives(i, p=1)
                cmds.delete(i_trans)

                print "Delete \"" + str(i_trans[0]) + "\""


#SceneCleanupMain().SceneCleanupUI()
