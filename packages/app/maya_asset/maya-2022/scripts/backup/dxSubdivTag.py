#encoding=utf-8

import maya.cmds as cmds
import maya.mel as mel

SubdivSchemeAttr                = 'rman__torattr___subdivScheme'
SubdivFacevaryingAttr   = 'rman__torattr___subdivFacevaryingInterp'

def getMesh():
    meshs = cmds.ls(sl=True, dag=True, type='mesh', ni=True)
    return meshs

def tagAddOn(arg=None):
    meshs = getMesh()
    if not meshs:
        return
    for mesh in meshs:
        #add attributes
        if not cmds.attributeQuery( SubdivSchemeAttr, n=mesh, ex=True ):
            cmds.addAttr( mesh, ln=SubdivSchemeAttr, at='long' )
        if not cmds.attributeQuery( SubdivFacevaryingAttr, n=mesh, ex=True ):
            cmds.addAttr( mesh, ln=SubdivFacevaryingAttr, at='long' )
        # set attributes
        cmds.setAttr( '%s.%s' % (mesh, SubdivSchemeAttr), 0 )
        cmds.setAttr( '%s.%s' % (mesh, SubdivFacevaryingAttr), 3 )

#select objects set to Catmull-Clark
def tagOnSelect(arg=None):
    cmds.select( cl=True )

    select = list()
    meshs = cmds.ls( type='mesh', ni=True )
    for i in meshs:
        if cmds.attributeQuery( SubdivSchemeAttr, n=i, ex=True ):
            if cmds.getAttr('%s.%s' % (i, SubdivSchemeAttr) ) == 0:
                select.append(i)
    cmds.select( select )

#clear None, Loop attributes
def noneLoopClear(arg=None):
    meshs = cmds.ls( type='mesh', ni=True )
    for i in meshs:
        if cmds.attributeQuery( SubdivSchemeAttr, n=i, ex=True ):
            scheme = cmds.getAttr( '%s.%s' % (i, SubdivSchemeAttr) )
            if scheme == 100 or scheme == 1:
                cmds.deleteAttr( '%s.%s' % (i, SubdivSchemeAttr) )
                if cmds.attributeQuery( SubdivFacevaryingAttr, n=i, ex=True ):
                    cmds.deleteAttr( '%s.%s' % (i, SubdivFacevaryingAttr) )

#clear attributes
def tagClear(arg=None):
    meshs = getMesh()
    if not meshs:
        return
    for i in meshs:
        for a in [SubdivSchemeAttr, SubdivFacevaryingAttr]:
            if cmds.attributeQuery( a, n=i, ex=True ):
                cmds.deleteAttr( '%s.%s' % (i, a) )

def show_ui():

    if cmds.window('win', q = True, ex = True ):
        cmds.deleteUI('win')

    win= cmds.window('win', t = 'RenderMan Subdiv Tool', wh= [310, 125], s= False)
    form= cmds.formLayout(nd= 100)
    b1= cmds.button( l = 'Tag Add/On', c= tagAddOn, w = 149, h = 30 )
    b2= cmds.button( l = 'Tag On Select', c= tagOnSelect, w = 149, h = 30 )
    column = cmds.columnLayout(rs= 1)
    cmds.separator( w = 305, h = 10)
    b3= cmds.button( l = 'Selected Tag Clear', c= tagClear, w = 300, h = 30)
    cmds.separator( w = 305, h = 10)
    b4= cmds.button( l = 'None/Loop Tag Clear', c= noneLoopClear, w = 300, h = 30)
    cmds.formLayout(form, edit= True, af= [(b1, 'top', 5), (b1, 'left', 5), (b2, 'top', 5), (b2, 'right', 5), (column, 'bottom', 5), (column, 'left', 4), (column, 'right', 5)])

    cmds.showWindow(win)
