import sys
import dxRigUI
import maya.cmds as cmds

#sys.path.append("/stdrepo/ANI/Library/Script/Module")
from Module import aniModule1


def dxRigMeshesAdd(attr):
    meshType = ['surfaceShape', 'nurbsCurve']
    if 'simMeshes' in attr:
        meshType = ['surfaceShape', 'nurbsCurve','locator']

    addedMeshes = cmds.getAttr(attr)
    meshes = cmds.ls(sl=True, dag=True, type=meshType, ni=True, l=True)
    for i in meshes:
        trans_node = cmds.listRelatives(i, p=True)
        if trans_node[0] not in addedMeshes:
            addedMeshes.append(trans_node[0])

    cmds.setAttr(attr, *([len(addedMeshes)] + addedMeshes), type='stringArray')

def MakeGeomGrp(*argv):

    selected = cmds.ls(sl=1)
    cmds.select(selected[0], r=1)
    nameSP1 = aniModule1.getNamespace()

    cmds.select(selected[1], r=1)
    nameSP2 = aniModule1.getNamespace()

    child_HighGrp = nameSP1+":"+nameSP1+"_model_GRP"
    child_lowGrp = nameSP1+":"+nameSP1+"_model_low_GRP"
    parent_HighGrp = nameSP2+":"+nameSP2+"_model_GRP"
    parent_lowGrp = nameSP2+":"+nameSP2+"_model_low_GRP"
    parent_rigGrp = nameSP2+":"+nameSP2+"_rig_GRP"



    cmds.parent(child_HighGrp,parent_HighGrp)

    if(cmds.objExists(parent_lowGrp)):
        cmds.parent(child_lowGrp,parent_lowGrp)

    #delete child namespace
    cmds.namespace(rm=nameSP1, mnr=1)


    cmds.setAttr(parent_rigGrp+".editable", 1)

    cmds.select(nameSP1+"_model_GRP", hi=1)
    # mel.eval("dxRigMeshesAdd(\""+parent_rigGrp+".renderMeshes\", \"drigRenderMeshesListWidget\");")
    dxRigMeshesAdd(parent_rigGrp+'.renderMeshes')

    cmds.select(cl=1)
    cmds.select(nameSP2+":"+nameSP2+"_model_GRP", hi=1)
    # mel.eval("dxRigMeshesAdd(\""+parent_rigGrp+".renderMeshes\", \"drigRenderMeshesListWidget\");")
    dxRigMeshesAdd(parent_rigGrp+'.renderMeshes')



    if(cmds.objExists(parent_lowGrp)):
        cmds.select(nameSP1+"_model_low_GRP", hi=1)
        # mel.eval("dxRigMeshesAdd(\""+parent_rigGrp+".lowMeshes\", \"drigLowMeshesListWidget\");")
        dxRigMeshesAdd(parent_rigGrp+'.lowMeshes')

        cmds.select(cl=1)
        cmds.select(nameSP2+":"+nameSP2+"_model_low_GRP", hi=1)
        # mel.eval("dxRigMeshesAdd(\""+parent_rigGrp+".lowMeshes\", \"drigLowMeshesListWidget\");")
        dxRigMeshesAdd(parent_rigGrp+'.lowMeshes')


    motionName = cmds.textFieldButtonGrp('pfc_but', q=1, text=1)


    cmds.namespace(ren=(nameSP2, motionName))

    cmds.select(cl=1)
    cmds.select(nameSP1+"_model_GRP", hi=1)
    cmds.select(nameSP1+"_model_low_GRP", hi=1, add=1)


    sel_objs = cmds.ls(sl=1, type='transform')


    for obj in sel_objs:
        cmds.rename(obj, motionName+":"+obj)
        print obj


    cmds.setAttr(motionName+":"+nameSP2+"_rig_GRP"+".editable", 0)









def makeUI():

    if (cmds.window('PresetForClip', q=1, ex=1)):
        cmds.deleteUI('PresetForClip', window=True)

    windowUI = cmds.window('PresetForClip', t="Preset for Clip", rtf=1, s=0, widthHeight=(454,220))
    columnUI = cmds.columnLayout("frameLayout_column", w=463, h=48, ebg=1, cat=("left",10), adjustableColumn=False, io=1)
    formUI = cmds.formLayout(numberOfDivisions=100)
    textFieldUI = cmds.textFieldButtonGrp( 'pfc_but', label='Motion Name', text='motion', buttonLabel='preset' ,bc=MakeGeomGrp)
    cmds.formLayout( formUI, edit=True, attachForm=[(textFieldUI, 'top', 10)] )

    cmds.showWindow(windowUI)




#makeUI()
