#encoding=utf-8
#!/usr/bin/env python

#-------------------------------------------------------------------------------
#
#   Dexter CG Supervisor
#
#		sanghun.kim		rman.td@gmail.com
#
#	2017.01.16	$2
#-------------------------------------------------------------------------------

import string

import maya.cmds as cmds
import maya.mel as mel


#-------------------------------------------------------------------------------
#
#	Meshes
#
#-------------------------------------------------------------------------------
# add meshes stringArray attribute
def addSelectedMeshes( attr ):
    selected = cmds.ls( sl=True )
    if len(selected) < 1:
        return

    dxrigs = cmds.ls( sl=True, type='dxRig' )
    selected = list( set(selected) - set(dxrigs) )

    result   = cmds.getAttr( attr )
    node     = attr.split('.')[0]
    attrName = attr.split('.')[-1]

    if cmds.getAttr( '%s.rigType' % node ) == 2:
        result += selected
    else:
        meshType = ['surfaceShape', 'nurbsCurve']
        if attrName == 'simMeshes':
            meshType = ['surfaceShape', 'nurbsCurve','locator']
        meshes = cmds.ls( selected, dag=True,type=meshType, ni=True, l=True )
        for i in meshes:
            if i.find( node ) > -1:
                trans_node = cmds.listRelatives( i, p=True )
                result.append( trans_node[0] )

    result = list( set(result) )
    result.sort()
    cmds.setAttr( attr, *([len(result)] + result), type='stringArray' )
    return result


# select stringArray attribute
def selectAttributeObjects( attr ):
    currentValues = cmds.getAttr( attr )

    node = attr.split('.')[0]

    selections = list()
    for i in currentValues:
        for x in cmds.ls( i, l=True, r=True ):
            if x.find( node ) > -1:
                selections.append( x )

    if selections:
        cmds.select( selections )


# select menu items
def selectItemsObjects( attr, items ):
    node = attr.split('.')[0]

    selections = list()
    for i in items.split(','):
        for x in cmds.ls( i, l=True, r=True ):
            if x.find( node ) > -1:
                selections.append( x )
    if selections:
        cmds.select( selections )


# remove menu items
def removeItemsObjects( attr, items ):
    node = attr.split('.')[0]

    currentValues = cmds.getAttr( attr )
    for i in items.split(','):
        if i:
            currentValues.remove( i )

    cmds.setAttr( attr, *([len(currentValues)] + currentValues), type='stringArray' )



#-------------------------------------------------------------------------------
#
#	Controllers
#
#-------------------------------------------------------------------------------
def addControllers(attr):
    node = attr.split('.')[0]
    result = cmds.getAttr(attr)

    selected = cmds.ls(sl=True)
    if len(selected) == 1:
        for i in cmds.ls('*_CON', l=True):
            if i.find(node) > -1:
                result += cmds.ls(i)
    else:
        selected.remove(node)
        result += selected

    if not result:
        return
    result = list(set(result))
    result.sort()
    cmds.setAttr(attr, *([len(result)] + result), type='stringArray')
    return result

def controllersInit( attr ):
    node = attr.split('.')[0]
    ns_name = ns = ""
    src = node.split(':')
    if len(src) > 1:
        ns_name = string.join( src[:-1], ':' )
    if ns_name:
        ns = ns_name + ":"

    data = cmds.getAttr( '%s.controllersData' % node )
    if not data:
        mel.eval( 'print "# Debug : Not found controllers initialize data"' )
        return

    data = eval(data)
    for i in data:
        # print(i, data[i]['value'], data[i]['type'])
        nodeName = i
        if ns_name:
            nodeName = '%s:%s' % (ns_name, i)
        if nodeName.split('.')[0] not in [ns+'move_CON', ns+'direction_CON', ns+'place_CON'] and i.find('Type') == -1:
            try:
                if data[i]['type'] == 'string':
                    cmds.setAttr( nodeName, data[i]['value'], type='string' )
                else:
                    cmds.setAttr( nodeName, data[i]['value'] )
            except:
                pass

def controllersInitWorld( attr ):
    node = attr.split('.')[0]
    ns_name = ns = ""
    src = node.split(':')
    if len(src) > 1:
        ns_name = string.join( src[:-1], ':' )
    if ns_name:
        ns = ns_name + ":"

    data = cmds.getAttr( '%s.controllersData' % node )
    if not data:
        return

    data = eval( data )
    for i in data:
        nodeName = i
        if ns_name:
            nodeName = '%s:%s' % (ns_name, i)
        if nodeName.split('.')[0] in [ns+'move_CON', ns+'direction_CON', ns+'place_CON']:
            try:
                cmds.setAttr( nodeName, data[i]['value'] )
            except:
                pass



#-------------------------------------------------------------------------------
#
#	Joints
#
#-------------------------------------------------------------------------------
def addSkinJoints( attr ):
    node = attr.split('.')[0]

    result = list()
    for i in cmds.ls( '*_Skin_*_JNT', l=True ):
        if i.find( node ) > -1:
            result += cmds.ls( i, type='joint' )

    if not result:
        return
    cmds.setAttr( attr, *([len(result)] + result), type='stringArray' )
    return result


#-------------------------------------------------------------------------------
#
#	StepMeshes
#
#-------------------------------------------------------------------------------
def addStepMeshes(attr):
    node = attr.split('.')[0]
    result = cmds.getAttr(attr)

    selected = cmds.ls(sl=True)
    selected.remove(node)

    result += selected
    result  = list(set(result))
    result.sort()
    cmds.setAttr(attr, *([len(result)] + result), type='stringArray')
    return result


#-------------------------------------------------------------------------------
#
#	Old Attributes update
#
#-------------------------------------------------------------------------------
def updateAttributes():
    selected = cmds.ls(sl=True, type='dxRig')
    if not selected:
        selected = cmds.ls(type='dxRig')

    for node in selected:
        # controlers -> controllers
        val = cmds.getAttr('%s.controlers' % node)
        if val:
            cmds.setAttr('%s.controllers' % node, *([len(val)] + val), type='stringArray')
            cmds.setAttr('%s.controlers' % node, 0, type='stringArray')

        # controlersData -> controllersData
        val = cmds.getAttr('%s.controlersData' % node)
        if val:
            cmds.setAttr('%s.controllersData' % node, val, type='string')
            cmds.setAttr('%s.controlersData' % node, '', type='string')

        print('# update attributes -', node)
