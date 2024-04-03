#encoding=utf-8
#!/usr/bin/env python

#-------------------------------------------------------------------------------
#
#   Dexter CG Supervisor
#
#		sanghun.kim		rman.td@gmail.com
#
#	dxCamera custom node procedural
#
#	2017.02.03	$2
#-------------------------------------------------------------------------------
#
#	for AETemplate
#	- import_cameraFile
#
#	Main Procedural
#	- import_mayaCamera, import_alembicCamera,
#	- alembicCamera_2DPanZoom, alembicCamera_imagePlane,
#	- importCameraDialog, exportCameraDialog
#
#-------------------------------------------------------------------------------

import os, sys
import re
import json

import maya.cmds as cmds
import maya.mel as mel

import sgCommon
import sgCamera

#-------------------------------------------------------------------------------
#
#	for AETemplate
#
#-------------------------------------------------------------------------------
def import_cameraFile( attr, fileName, panzoomFile=None ):
    node = attr.split('.')[0]

    # clear current nodes
    child = cmds.listRelatives( node )
    if child:
        if cmds.referenceQuery( child[0], inr=True ):
            refFile = cmds.referenceQuery( child[0], f=True )
            cmds.file( refFile, rr=True )
        else:
            cmds.delete( child )

    cmds.setAttr( node+'.fileName', fileName, type='string' )
    # get version
    version = 0
    src_tmp = re.compile( '_v(\d+)?.mb' ).findall( fileName )
    if src_tmp:
        version = int(src_tmp[0])
    cmds.setAttr( node+'.version', version )

    importNode = None
    if fileName.split('.')[-1] == 'mb':
        importNode = import_mayaCamera( fileName )
    elif fileName.split('.')[-1] == 'abc':
        importNode = import_alembicCamera( fileName, panzoomFile )

    if not importNode:
        return
    src_nodes = list()
    for i in cmds.ls( importNode, l=True, dag=True, type=['camera', 'surfaceShape'] ):
        src_nodes.append( cmds.listRelatives(i, f=True, p=True)[0] )
    cmds.parent( src_nodes, node )
    cmds.delete( importNode )
    cmds.setAttr( '%s.action' % node, 1 )
# debug
    impCamList = []
    for i in cmds.ls( node, dag=True, type='camera' ):
        camNode = cmds.listRelatives(i, p=True)[0]
        mel.eval( 'print "# import camera : %s\\n"' % camNode )
        impCamList.append( camNode )
    return impCamList

#-------------------------------------------------------------------------------
#
#	Main Procedural
#
#-------------------------------------------------------------------------------
def import_mayaCamera( fileName ):
    baseName = os.path.splitext( os.path.basename(fileName) )[0]
    cmds.file( fileName, i=True, type='mayaBinary', iv=True, mnc=False,
               rpr=baseName, gr=True, gn=baseName+'_import',
               op='v=0;p=17;f=0', pr=True )
    return baseName+'_import'


def import_alembicCamera( fileName, panzoomFile=None ):
    baseName = os.path.splitext( os.path.basename(fileName) )[0]
    root = cmds.group( n=baseName+'_imoprt', em=True )
    mel.eval( 'AbcImport -d -m import -rpr "%s" "%s"' % (root, fileName) )

# 2DPanZoom
    alembicCamera_2DPanZoom( fileName, panzoomFile )

# ImagePlane
    alembicCamera_imagePlane( fileName )
    return root

def alembicCamera_2DPanZoom( abcFile , panzoomFile=None):
    if panzoomFile:
        fn = panzoomFile
        shakeFn = panzoomFile.replace('.panzoom', '.shake')
    else:
        fn = abcFile.replace('.abc', '.panzoom')
        shakeFn = abcFile.replace('.abc', '.shake')

    if not os.path.exists(fn):
        return

    f = open(fn, 'r')
    body = json.load(f)
    f.close()

    shakeBody = dict()

    if os.path.exists(shakeFn):
        f = open(shakeFn, 'r')
        shakeBody = json.load(f)
        f.close()

    if not body.has_key('2DPanZoom'):
        return

    data = body['2DPanZoom']
    shakeData = {}
    if '2DShake' in shakeBody:
        shakeData = shakeBody['2DShake']

    for node in data:
        curShapes = cmds.ls(node.split(':')[-1], r=True, type='camera')
        attrs = data[node].keys()
        for shape in curShapes:
            for a in attrs:
                #
                cmds.setAttr('%s.panZoomEnabled' % shape, 1)
                cmds.setAttr('%s.renderPanZoom' % shape, 1)
                #
                keyData = data[node][a]
                if type(keyData).__name__ == 'dict':
                    if keyData.has_key('frame'):
                        sgCommon.coreKeyLoad(shape, a, keyData)
                    else:
                        gv = keyData['value']
                        gt = keyData['type']
                        if gt == 'string':
                            cmds.setAttr('%s.%s' % (shape, a), gv, type='string')
                        else:
                            cmds.setAttr('%s.%s' % (shape, a), gv)
                else:
                    cmds.setAttr('%s.%s' % (shape, a), keyData)

            if shakeBody:
                cmds.animLayer('cameraShakeLyer#',
                               at=[shape + '.hpn', shape + '.vpn'])
                animBlendNodeH = cmds.listConnections(shape + '.hpn', s=True, d=False)
                animBlendNodeV = cmds.listConnections(shape + '.vpn', s=True, d=False)
                if shakeData:
                    sgCommon.coreKeyLoad(animBlendNodeH, 'inputB', shakeData[node]['horizontal'])
                    sgCommon.coreKeyLoad(animBlendNodeV, 'inputB', shakeData[node]['vertical'])


def alembicCamera_imagePlane( abcFile, jsonPath=None):
    fn = abcFile.replace( '.abc', '.imageplane' )

    # .imageplane file and .json file compatibility
    if not(jsonPath):
        if not os.path.exists( fn ):
            return
    else:
        fn = jsonPath

    f = open( fn, 'r' )
    body = json.load( f )
    f.close()

    if not body.has_key('ImagePlane'):
        return

    data = body['ImagePlane']
    for cShape in data:
        if cmds.ls(cShape, r=True):
            cameraShape = cmds.ls( cShape, r=True )[0]
            for imp in data[cShape]:
                attrDict = data[cShape][imp]
                impTrans, impShape = cmds.imagePlane( camera=cameraShape )
                sgCommon.attributesKeyLoad( impShape, attrDict )
                if attrDict['useFrameExtension']['value']:
                    if not cmds.listConnections( '%s.frameExtension' % impShape ):
                        cmds.expression( n='%s_expression' % impShape, s='%s.frameExtension=frame;' % impShape )


def imagePlaneFromJson(jsonPath):
    fn = jsonPath
    f = open( fn, 'r' )
    body = json.load( f )
    f.close()

    if not body.has_key('ImagePlane'):
        return

    data = body['ImagePlane']
    for cShape in data:
        print cShape
        cameraShape = cmds.ls( cShape, r=True )[0]
        for imp in data[cShape]:
            attrDict = data[cShape][imp]
            impTrans, impShape = cmds.imagePlane( camera=cameraShape )
            sgCommon.attributesKeyLoad( impShape, attrDict )
            if attrDict['useFrameExtension']['value']:
                if not cmds.listConnections( '%s.frameExtension' % impShape ):
                    cmds.expression( n='%s_expression' % impShape, s='%s.frameExtension=frame;' % impShape )

#-------------------------------------------------------------------------------
# import camera
def importCameraDialog():
    fn = cmds.fileDialog2( fm=4,
                           ff='Alembic,Maya (*.abc *.mb);;Alembic (*.abc);;MayaBinary (*.mb)',
                           cap='Import Camera (Select Camera File)' )
    if not fn:
        return

    dxcam = cmds.createNode( 'dxCamera' )
    import_cameraFile( '%s.fileName' % dxcam, fn[0] )
    cmds.select( cl=True )


#-------------------------------------------------------------------------------
# export camera
def exportCameraDialog():
    fn = cmds.fileDialog2( fm=3,
                           cap='Export Camera (Select Directory)',
                           okc='export',
                           ocr='dxsgnExportFrame_UICreate',
                           oin='dxsgnExportFrame_UIInit',
                           ocm='dxsgnExportFrame_UICommit' )
    if not fn:
        return

    mode  = cmds.optionVar( q='dxExportFrameMode' )
    start = int( cmds.optionVar(q='dxExportFrameStart') ) -1
    end   = int( cmds.optionVar(q='dxExportFrameEnd') ) +1

    current = int( cmds.currentTime(q=True) )

    if mode == 'sgncurrenttime':
        camClass = sgCamera.ExportCamera( Path=fn[0], Start=current, End=current )
        camClass.doIt()
    else:
        camClass = sgCamera.ExportCamera( Path=fn[0], Start=start, End=end )
        camClass.doIt()
