
import maya.cmds as cmds
import maya.mel as mm
import sys

filepath = sys.argv[1]

class backgroundMayaProcess:
    def __init__(self, _filePath):
        self.filePath = _filePath
        self.cleanUp()

    def cleanUp(self):
        cmds.file(self.filePath, f=True, options=("v=0;"), typ="mayaBinary", o=True)
        #mm.eval('animLayerEditorOnSelect "BaseAnimation" 0;animLayerEditorOnSelect "BaseAnimation" 1;')
        #self.mergeAnimLyr()
        self.bakeConstrains()
        self.setRestPose()
        cmds.file(s=True)

    def bakeConstrains(self):
        StartTime = cmds.playbackOptions(q=1, min=True) - 1
        EndTime = cmds.playbackOptions(q=1, max=True) + 1

        AllConstrainList = cmds.ls(type='parentConstraint')
        AllConstrainList += cmds.ls(type='pointConstraint')
        AllConstrainList += cmds.ls(type='orientConstraint')
        AllConstrainList += cmds.ls(type='aimConstraint')
        AllConstrainList += cmds.ls(type='scaleConstraint')

        for tempPrCons in AllConstrainList:
            if cmds.referenceQuery(tempPrCons, isNodeReferenced=True) == 0:
                ConnectedObj = cmds.listRelatives(tempPrCons, parent=True, type='transform')[0]
                cmds.bakeResults(ConnectedObj, simulation=False, dic=False, t=(StartTime, EndTime))
                #mm.eval('performEulerFilter graphEditor1FromOutliner')
                cmds.filterCurve()
                cmds.keyTangent(ConnectedObj, itt='linear', ott='linear', animation='objects')
                cmds.delete(tempPrCons)
                print "# Bake {} with constraint".format(ConnectedObj)
        print "# Bake Constraint Done\n"

    def mergeAnimLyr(self):
        animLayers = cmds.ls(type='animLayer')

        if animLayers != ['BaseAnimation'] and animLayers != []:
            LYRstr = ""

            for al in animLayers:
                LYRstr = LYRstr + '"%s"' % al
                if al != animLayers[-1]:
                    LYRstr = LYRstr + ","

            mm.eval('string $layers[]={%s}; layerEditorMergeAnimLayer( $layers, 0 )' % LYRstr)

    def setRestPose(self):
        #mm.eval('source "layerEditor";')
        #mm.eval('source buildSetAnimLayerMenu;')
        #mm.eval('selectLayer("BaseAnimation");')

        dxRigNodes = cmds.ls(type="dxRig")
        cmds.currentTime(950)
        import dxRigUI as drg

        for dxnode in dxRigNodes:
            conAttr = "{node}.controlers".format(node=dxnode)
            #mm.eval('dxrigControlersInit("{}");'.format(conAttr))
            drg.controlersInit(conAttr)
            drg.selectAttributeObjects(conAttr)
            mm.eval("SetKeyAnimated;")
            cmds.currentTime(u=True)
            print "# Set Rest Pose"

if __name__ == "__main__":
    from pymel.all import *
    bmp = backgroundMayaProcess(filepath)