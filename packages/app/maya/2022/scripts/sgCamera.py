#encoding=utf-8
#!/usr/bin/env python

#-------------------------------------------------------------------------------
#
#   Dexter CG Supervisor
#
#       Sanghun Kim, rman.td@gmail.com
#
#	Camera pipe-line procedural
#
#	2017.03.05 $3
#-------------------------------------------------------------------------------
#
#   CameraMtxCopy
#
#	ImagePlane
#	- createPolyImagePlane, getImagePlaneAttributes
#
#	Camera
#	- ExportCamera
#		- doIt, getCameras, duplicateCameras,
#		- getImagePlane, keyFrameOffset,
#		- export_abc, export_maya, export_file
#
#	2DPanZoom
#	- exportPanZoomDialog, exportPanZoom, toNuke_2DPanZoom
#-------------------------------------------------------------------------------

import os, sys
import math
import re
import string
import getpass

import maya.api.OpenMaya as OpenMaya
import maya.api.OpenMayaAnim as OpenMayaAnim

import maya.cmds as cmds
import maya.mel as mel

import dplCommon
import sgCommon

import sgAnimation
import sgAlembic

from dxname import rulebook
from dxname import tag_parser
import datetime

#-------------------------------------------------------------------------------
#
#	Camera World Matrix Copy
#
#-------------------------------------------------------------------------------
#	sourceNode : original camera transform node
#	targetNode : new camera transform node
class CameraMtxCopy:
    def __init__( self, sourceNode, targetNode, start, end, step ):
        self.m_sourceNode = sourceNode
        self.m_targetNode = targetNode
        self.m_start = start
        self.m_end   = end
        self.m_step  = step
        self.m_rotateOrder = cmds.getAttr( '%s.rotateOrder' % self.m_targetNode )

        # doIt
        self.m_frames = self.getFrames()
        self.getMatrix()
        self.setMatrix()

    def getAlembicFrame( self ):
        start = end = None
        alembicNode = cmds.listConnections(self.m_sourceNode, s=1, type='AlembicNode')
        if alembicNode:
            print(alembicNode)
            start = cmds.getAttr('%s.startFrame'%alembicNode[0])
            end = cmds.getAttr('%s.endFrame'%alembicNode[0])

        return start, end

    def findKeyFrames( self ):
        parentPath = cmds.listRelatives( self.m_sourceNode, c=True, f=True )[0]
        src = parentPath.split('|')
        total_frames = list()
        for i in range(1, len(src)):
            node = string.join( src[:i+1], '|' )
            frames = cmds.keyframe( node, q=True )
            if frames:
                total_frames += frames
            constNodes = cmds.listConnections( node, type='constraint', s=True, d=False )
            if constNodes:
                for c in list(set(constNodes)):
                    keyNodes = cmds.listConnections( c, s=True, d=False )
                    keyNodes = list( set(keyNodes)-set([node, c]) )
                    for k in keyNodes:
                        frames = cmds.keyframe( k, q=True )
                        if frames:
                            total_frames += frames
        return list(set(total_frames))

    def getFrames( self ):
        frames = sgCommon.get_frames( self.m_start, self.m_end, self.m_step )
        frames = list( set(frames + self.findKeyFrames()) )
        frames.sort()
        return frames


    def getMatrix( self ):
        self.m_matrix = list()
        selection = OpenMaya.MSelectionList()
        selection.add( self.m_sourceNode )

        mobj = selection.getDependNode( 0 )
        mfn  = OpenMaya.MFnDependencyNode( mobj )

        mtxAttr = mfn.attribute( 'worldMatrix' )
        mtxPlug = OpenMaya.MPlug( mobj, mtxAttr )
        mtxPlug = mtxPlug.elementByLogicalIndex( 0 )

        # check alembic node frames
        alembic_start, alembic_end = self.getAlembicFrame()
        keyed = cmds.keyframe(self.m_sourceNode, q=1, t=(self.m_start, self.m_end))
        for f in self.m_frames:
            # copy last alembic key if there is no key
            if keyed == None:
                if alembic_start and alembic_end:
                    if f > alembic_end:
                        f = alembic_end
                    elif f < alembic_start:
                        f = alembic_start

            frameCtx = OpenMaya.MDGContext( OpenMaya.MTime(f, OpenMaya.MTime.uiUnit()) )
            mtxObj   = mtxPlug.asMObject( frameCtx )
            mtxData  = OpenMaya.MFnMatrixData( mtxObj )
            mtxValue = mtxData.matrix()
            self.m_matrix.append( mtxValue )


    def setMatrix( self ):
        node = self.m_targetNode

        dgmod = OpenMaya.MDGModifier()

        TL_list = ['translateX', 'translateY', 'translateZ', 'scaleX', 'scaleY', 'scaleZ']
        TA_list = ['rotateX', 'rotateY', 'rotateZ']

        objList = list()
        for i in TL_list:
            obj = dgmod.createNode( 'animCurveTL' )
            dgmod.renameNode( obj, '%s_%s' % (node, i) )
            objList.append( obj )
        for i in TA_list:
            obj = dgmod.createNode( 'animCurveTA' )
            dgmod.renameNode( obj, '%s_%s' % (node, i) )
            objList.append( obj )

        dgmod.doIt()

        keyObjList = list()
        for o in objList:
            obj = OpenMayaAnim.MFnAnimCurve()
            obj.setObject( o )
            keyObjList.append( obj )

        for i in xrange( len(self.m_frames) ):
            #tmtx  = OpenMaya.MTransformationMatrix( self.m_matrix[i] )
            mtx  = OpenMaya.MMatrix( self.m_matrix[i] )
            tmtx = OpenMaya.MTransformationMatrix( mtx )
            mtime = OpenMaya.MTime( self.m_frames[i], OpenMaya.MTime.uiUnit() )
            space = OpenMaya.MSpace.kWorld

            tr = tmtx.translation( space )
            for x in range(3):
                keyObjList[x].addKey( mtime, tr[x] )

            sc = tmtx.scale( space )
            for x in range(3):
                keyObjList[x+3].addKey( mtime, sc[x] )

            ro = tmtx.rotation()
            ro.reorderIt( self.m_rotateOrder )
            for x in range(3):
                keyObjList[x+6].addKey( mtime, ro[x] )

        curveNames = list()
        for i in TL_list:
            index = TL_list.index(i)
            mfn   = OpenMaya.MFnDependencyNode( objList[index] )
            name  = mfn.name()
            curveNames.append( name )
            cmds.connectAttr( '%s.output' % name, '%s.%s' % (node, i) )
        for i in TA_list:
            index = TA_list.index(i)
            mfn   = OpenMaya.MFnDependencyNode( objList[index+6] )
            name  = mfn.name()
            curveNames.append( name )
            cmds.connectAttr( '%s.output' % name, '%s.%s' % (node, i) )
        cmds.filterCurve( curveNames )



#-------------------------------------------------------------------------------
#
#	ImagePlane
#
#-------------------------------------------------------------------------------
def createPolyImagePlane( camShape, imageplanes ):
    camTrans = cmds.listRelatives( camShape, p=True, f=True )[0]
    hfa = cmds.camera( camShape, q=True, hfa=True )
    vfa = cmds.camera( camShape, q=True, vfa=True )
    fov = math.radians( cmds.camera(camShape, q=True, hfv=True) )
    aspect = vfa/hfa

    tmpPlaneList = list()
    for i in imageplanes:
        overscan = 1.0
        # temporary statement
        tmp_coverageX = cmds.getAttr( '%s.coverageX' % i )
        if tmp_coverageX == 2420 or tmp_coverageX == 1210 or tmp_coverageX == 3168 or tmp_coverageX == 1584:
            overscan = 1.1

        pname = '%s_polyImagePlane_tmp' % camTrans.split(':')[-1]
        tmpPlane = cmds.polyPlane( name=pname,
                                   w=1.0, h=aspect, sx=1, sy=1, ax=(0,-1,0), cuv=1, ch=1 )[0]
        cmds.polyFlipUV( tmpPlane, ft= 0 )
        cmds.polyFlipUV( tmpPlane, ft= 1 )
        tmpPlaneList.append( tmpPlane )

    for i in tmpPlaneList:
        cmds.parent( i, camTrans )

    xpos = cmds.getAttr( '%s.tx' % camTrans )
    ypos = cmds.getAttr( '%s.ty' % camTrans )
    zpos = cmds.getAttr( '%s.tz' % camTrans )

    expressionString = "float $hfa = `camera -q -hfa {camera}`;\n"
    expressionString += "float $vfa = `camera -q -vfa {camera}`;\n"
    expressionString += "float $fov = `camera -q -hfv {camera}`;\n"
    expressionString += "float $bbminx = {imageplane}.boundingBoxMinX;\n"
    expressionString += "float $bbmaxx = {imageplane}.boundingBoxMaxX;\n"
    expressionString += "float $bbminy = {imageplane}.boundingBoxMinY;\n"
    expressionString += "float $bbmaxy = {imageplane}.boundingBoxMaxY;\n\n"
    expressionString += "{polyplane}.scaleX = {polyplane}.scaleY = {polyplane}.scaleZ = 2*({imageplane}.depth)*(tand($fov/2.0));\n\n"
    # (offset 비율) * (imageplane boundingbox 크기)
    expressionString += "{polyplane}.translateX = ({imageplane}.offsetX / $hfa) * ($bbmaxx - $bbminx);\n"
    expressionString += "{polyplane}.translateY = ({imageplane}.offsetY / $vfa) * ($bbmaxy - $bbminy);\n"
    expressionString += "{polyplane}.translateZ = -1*{imageplane}.depth;\n"
    expressionString += "{polyplane}.rotateZ = 180 + -1*{imageplane}.rotate;\n"

    newPlaneList = list()

    for i in tmpPlaneList:
        depth = cmds.getAttr( '%s.depth' % imageplanes[tmpPlaneList.index(i)] ) * -1.0
        cmds.setAttr( '%s.rx' % i, -90.0 )
        cmds.setAttr( '%s.ry' % i, 0.0 )

        imagePlaneShape = cmds.listRelatives(imageplanes[tmpPlaneList.index(i)], s=True, p=False)[0]
        exprString = expressionString.format(
            camera=camTrans,
            imageplane=imagePlaneShape,
            polyplane=i
        )
        cmds.expression(s=exprString, o=i, ae=True, uc='all')
        cmds.refresh()
        newPlane = cmds.duplicate(i, name=i.replace('_tmp', ''))[0]
        cmds.parent(newPlane, world=True)

        newPlaneList.append( newPlane )

    return newPlaneList, tmpPlaneList


def getImagePlaneAttributes( shapeName ):
    attrs = [ 'displayMode', 'type', 'textureFilter', 'imageName',
              'offsetX', 'offsetY', 'useFrameExtension', 'frameOffset', 'frameCache',
              'fit', 'displayOnlyIfCurrent', 'depth', 'frameExtension',
              'coverageX', 'coverageY', 'coverageOriginX', 'coverageOriginY',
              'imageCenterX', 'imageCenterY', 'imageCenterZ',
              'width', 'height', 'maintainRatio', 'alphaGain' ]

    connected = cmds.listConnections( shapeName, plugs=True, type='animCurve' )
    if connected:
        for i in connected:
            plug = cmds.connectionInfo( i, dfs=True )
            if plug:
                attrs.append( plug[0].split('.')[-1] )

    data = sgCommon.attributesKeyDump( shapeName, list(set(attrs)) )
    return data



#-------------------------------------------------------------------------------
#
# camera export v2.0
#
#	Start, End input just frame, output +,- 1frame offset auto key
#
#-------------------------------------------------------------------------------
class ExportCamera:
    def __init__( self, Path=None, Start=None, End=None, Step=1.0 ):
        # plug-in setup
        plugins = ['AbcExport', 'backstageMenu']
        for p in plugins:
            if not cmds.pluginInfo( p, q=True, l=True ):
                cmds.loadPlugin( p )

        if not Path:
            return
        currentfile = cmds.file( q=True, sn=True )
        if not currentfile:
            mel.eval( 'print "Error : save scene\\n"' )
            return

        self.m_baseName = os.path.splitext( os.path.basename(currentfile) )[0]
        self.m_Path = Path
        cmds.autoKeyframe( state=False )

        # members
        self.m_enable_maya	 = False
        self.m_cleanUp		 = True
        self.m_username		 = None

        self.m_createCameras = list()	# transform nodes, final export nodes
        self.m_exportCameras = list()	# transform nodes
        self.m_renderCameras = list()	# transform nodes
        self.m_polyPlanes	 = list()

        self.m_logDict = dict()
        self.m_logDict['render_camera'] = list()
        self.m_imagePlane = dict()

        self.m_start = None; self.m_end = None
        if Start:
            self.m_start = int(Start)
        if End:
            self.m_end = int(End);
        # frame range
        if not self.m_start:
            self.m_start = int( cmds.playbackOptions(q=True, min=True) )
        if not self.m_end:
            self.m_end   = int( cmds.playbackOptions(q=True, max=True) )
        self.m_step = Step

    def doIt( self ):
        # log : username
        if not self.m_username:
            self.m_username = getpass.getuser()

        self.getCameras()
        self.bakeKeys()
        self.duplicateCameras()

        if not self.m_createCameras:
            return

        self.keyFrameOffset( self.m_createCameras, 1 )
        self.export_file()

        self.m_logDict['cameras'] = list()
        for c in self.m_exportCameras:
            self.m_logDict['cameras'].append( c.split('|')[-1] )

        # cleanup
        if self.m_cleanUp:
            cmds.delete( self.m_createCameras )
            cmds.delete( 'export_camera_GRP' )
            if self.m_polyPlanes:
                cmds.delete( self.m_polyPlanes )


    def getCameras( self ):
        # added '|' for prevent overlapping with mmv left, right cameras
        delList = ['frontShape', 'topShape', 'perspShape', 'backShape',
                   'sideShape', '|leftShape', '|rightShape', 'bottomShape']
        currentCameras = list()

        # dxCamera
        for dc in cmds.ls( type='dxCamera' ):
            action   = cmds.getAttr( '%s.action' % dc )
            fileName = cmds.getAttr( '%s.fileName' % dc )
            if action == 0 and fileName:	# not export just logging
                fext = fileName.split('.')[-1]
                if fext == 'abc':
                    self.m_logDict['abc_camera'] = fileName
                if fext == 'mb':
                    self.m_logDict['maya_camera'] = fileName
                # renderable
                for cShape in cmds.ls( dc, dag=True, type='camera', l=True ):
                    delList.append( cShape )
                    if cmds.getAttr( '%s.renderable' % cShape ):
                        cTrans = cmds.listRelatives( cShape, p=True )[0]
                        name = cTrans.split(':')[-1]
                        self.m_logDict['render_camera'].append( name )

        for c in cmds.ls( type='camera', l=True ):
            ifever = 0
            for d in delList:
                if c.find(d) > -1:
                    ifever += 1
            if ifever == 0:
                currentCameras.append( c )

        for cShape in cmds.ls( currentCameras, l=True ):
            cTrans = cmds.listRelatives( cShape, p=True, f=True )[0]
            self.m_exportCameras.append( cTrans )
            if cmds.getAttr( '%s.renderable' % cShape ):
                trans = cmds.listRelatives( cShape, p=True )[0]
                name = trans.split(':')[-1]
                self.m_logDict['render_camera'].append( name )

    def bakeKeys( self ):
        tobake = []
        for cam in self.m_exportCameras:
            camShape = cmds.listRelatives( cam, c=True, f=True, type='camera' )
            if camShape:
                camShape = camShape[0]
                imagePlanes = cmds.listRelatives( camShape, type='imagePlane', ad=True)
                tobake.append(camShape)
                if imagePlanes:
                    tobake += imagePlanes
        if tobake:
            try:
                attrs = ['tx', 'ty', 'tz',
                         'rx', 'ry', 'rz',
                         'sx', 'sy', 'sz',
                         'focalLength', 'zom',
                         'vfa', 'hfa', 'hfv',
                         'depth', 'frameExtension']
                cmds.bakeResults( tobake, t=(self.m_start,self.m_end), at=attrs)
            except:
                pass

    def duplicateCameras( self ):
        exportGroup = cmds.group( n='export_camera_GRP', em=True )
        for c in self.m_exportCameras:
            orig_camTrans = c
            orig_camShape = cmds.listRelatives( c, c=True, f=True, type='camera' )[0]

            newCamera = cmds.duplicate( c, rr=True )[0]
            cmds.parent( newCamera, exportGroup )
            # rename
            name = cmds.ls( newCamera, l=True )[0]
            new_camTrans = cmds.rename( name, c.split('|')[-1] )
            new_camShape = cmds.listRelatives( new_camTrans, c=True, f=True, type='camera' )[0]
            new_camShape = cmds.rename(new_camShape, orig_camShape.split('|')[-1])

            # unlock
            attrs = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz', 'sx', 'sy', 'sz']
            attrs+= ['hfa', 'vfa', 'fl', 'lsr', 'fs', 'fd', 'sa', 'coi']
            sgCommon.unlockAttrs( new_camTrans, attrs )

            # copy key
            connections = cmds.listConnections( orig_camShape, p=True, type='animCurve' )
            if connections:
                for i in connections:
                    dest = cmds.connectionInfo( i, dfs=True )[0]
                    attr = dest.split('.')[-1]
                    cmds.copyKey( orig_camShape, attribute=attr )
                    cmds.pasteKey( new_camShape, attribute=attr )

            # init pivot
            cmds.xform( new_camTrans, ztp=True )

            # key copy
            CameraMtxCopy( orig_camTrans, new_camTrans, self.m_start, self.m_end, self.m_step )


            # get imagePlane data
            self.getImagePlane( orig_camShape )

            # create camera
            self.m_createCameras.append( new_camTrans )



    def getImagePlane( self, sourceCamera ):
        imagePlanes = cmds.listConnections( sourceCamera, type='imagePlane', d=False )
        if not imagePlanes:
            return

        for i in list(set(imagePlanes)):
            # get attributes
            attrData = getImagePlaneAttributes( i )
            if attrData:
                name = sourceCamera.split('|')[-1].split(':')[-1]
                if not self.m_imagePlane.has_key( name ):
                    self.m_imagePlane[name] = dict()
                self.m_imagePlane[name][ i.split(':')[-1] ] = attrData

        # create poly imagePlane
        polymeshes, tmpmeshes = createPolyImagePlane( sourceCamera, list(set(imagePlanes)) )
        self.m_polyPlanes += tmpmeshes

        '''
        # key copy by python API 2.0
        for i in range(len(polymeshes)):
            matrixs, frames = sgCommon.getMtx( tmpmeshes[i], self.m_start, self.m_end, self.m_step )
            sgCommon.setMtx( polymeshes[i], matrixs, frames )
        cmds.select( polymeshes )
        cmds.filterCurve()

        if self.m_cleanUp:
            cmds.delete( tmpmeshes )
        '''

    def keyFrameOffset( self, objects, offsetframe ):
        attrList = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
        for i in objects:
            for a in attrList:
                sgCommon.offsetKeyAttr( i, a, offsetframe )

    #-------------------------------------------------------------------------------------
    # export file
    def export_abc( self ):
        start = self.m_start - 1
        end   = self.m_end + 1
        abcFile = os.path.join( self.m_Path, '%s_camera.abc' % self.m_baseName )
        options = '-v -j "-ef -worldSpace -fr %s %s -s %s' % ( start, end, self.m_step )
        for i in self.m_createCameras:
            options += ' -rt %s' % i
        options += ' -f %s"' % abcFile

        self.impFile = ''
        if self.m_polyPlanes:
            self.impFile = os.path.join( self.m_Path, '%s_imagePlane.abc' % self.m_baseName )
            options += ' -j "-uv -worldSpace -dataFormat ogawa -fr %s %s' % ( start, end )
            for i in self.m_polyPlanes:
                options += ' -rt %s' % i
            options += ' -f %s"' % self.impFile

        mel.eval( 'AbcExport %s' % options )
        self.m_logDict['abc_camera'] = abcFile
        self.m_logDict['imageplane_path'] = self.impFile

    def export_maya( self ):
        mayaFile = os.path.join( self.m_Path, '%s_camera.mb' % self.m_baseName )
        cmds.select( self.m_createCameras + self.m_polyPlanes )
        cmds.file( mayaFile, force=True, options='v=0;', typ='mayaBinary', pr=False, es=True )
        self.m_logDict['maya_camera'] = mayaFile

    def export_file( self ):
        if not os.path.exists( self.m_Path ):
            os.makedirs( self.m_Path )

        if self.m_enable_maya:
            self.export_maya()

        self.export_abc()

        pzFile = os.path.join( self.m_Path, '%s_camera.panzoom' % self.m_baseName )
        exportPanZoom( pzFile, self.m_exportCameras, self.m_start, self.m_end, self.m_username )

        # imageplane
        if self.m_imagePlane:
            fn = os.path.join( self.m_Path, '%s_camera.imageplane' % self.m_baseName )
            dplCommon.writeJsonLog( File=fn,
                                    Data={'ImagePlane': self.m_imagePlane},
                                    Context=cmds.file(q=True, sn=True),
                                    User=self.m_username )
            self.m_logDict['imageplane_json_path'] = fn

class SgCameraMMV(object):
    # def __init__(self, info, nameRoot,  Step=1.0,):
    def __init__(self, info, Step=1.0, ):
        # plug-in setup
        plugins = ['AbcExport', 'backstageMenu']
        for p in plugins:
            if not cmds.pluginInfo(p, q=True, l=True):
                cmds.loadPlugin(p)

        # currentfile = cmds.file(q=True, sn=True)
        # if not currentfile:
        #     mel.eval('print "Error : save scene\\n"')
        #     return

        self.info = info
        ##################### DB RECORD & NAMING MUDULE ####################
        self.dbRecord = {'show': self.info['show'],
                         'sequence': self.info['sequence'],
                         'shot': self.info['shot'],
                         'task': self.info['task'],
                         'version': self.info['version'],
                         'data_type': 'camera',
                         'time': datetime.datetime.now().isoformat(),
                         'artist': self.info['user'],
                         'files': {},
                         'enabled':True,
                         'task_publish': {'camera_type': self.info['camera_type'],
                                          'stereo': self.info['stereo'],
                                          'overscan': self.info['overscan'],
                                          'overscan_value': self.info['overscan_value'],
                                          'render_width': self.info['render_width'],
                                          'render_height': self.info['render_height'],
                                          'dx_camera': self.info['camera_list'],
                                          'camera_only': self.info['camera_only'],
                                          'startFrame': self.info['startFrame'],
                                          'endFrame': self.info['endFrame']
                                          }
                         }
        self.dxNodeName = self.info['camera_list'][0].split('|')[1]

        self.nameRoot = rulebook.Coder()
        self.nameRoot.load_rulebook('/backstage/libs/python_lib/dxname/name_for_publish.yaml')
        self.nameRoot.flag['PROJECT'] = self.info["show"]
        self.nameRoot.flag['SEQUENCE'] = self.info['sequence']
        self.nameRoot.flag['SHOT'] = self.info['shot']
        self.nameRoot.flag['PUBDEV'] = 'pub'

        self.nameRoot.camera.shot_camera.flag['TEAM'] = self.info['task']
        if self.info.has_key('plate'):
            self.nameRoot.camera.shot_camera.flag['PLATE'] = self.info['plate']
            self.dbRecord['task_publish']['plateType'] = self.info['plate']
        self.nameRoot.flag['VER'] = 'v' + str(self.info['version']).zfill(2)
        ######################################################################

        self.m_step = Step

        self.m_cleanUp = True

        self.m_polyPlanes = list()

        self.m_logDict = dict()
        self.m_logDict['render_camera'] = list()
        self.m_imagePlane = dict()

        self.m_start = int(float(self.info['startFrame']))
        self.m_end = int(float(self.info['endFrame']));

    def doIt(self):
        # create dir if not exists
        print("self.nameRoot : ",self.nameRoot)
        basePath = self.nameRoot.camera.shot_camera.product['root']
        if not os.path.exists(basePath):
            os.makedirs(basePath)

        # export maya file (save as)
        maya_scene_path = self.nameRoot.camera.shot_camera.product['maya_pub_file']
        cmds.file(maya_scene_path, ea=True, options='v=0;', typ='mayaBinary', pr=True)
        cmds.file(rn=maya_scene_path)
        cmds.file(s=1)

        self.dbRecord['files']['maya_dev_file'] = [cmds.file(q=True, sn=True)]
        self.dbRecord['files']['maya_pub_file'] = [maya_scene_path]

        # self.m_createCameras = list()	# transform nodes, final export nodes
        # self.m_exportCameras = list()	# transform nodes
        # self.m_polyPlanes	 = list()

        # self.m_exportCameras -> orgCamTrans
        # self.m_createCameras -> newCamTrans

        # collect transform node from camera node under dxCamera
        # orgCamTrans = self.getActualCamera(self.info['camera_list'])
        orgCamTrans = self.info['camera_list']
        print(orgCamTrans)
        # duplicate transform from orgCamTrans
        newCamTrans = self.duplicateCameras(orgCamTrans)
        print(newCamTrans)

        if not newCamTrans:
            return

        # set keyframe offset + - 1
        self.keyFrameOffset(newCamTrans + self.m_polyPlanes, 1)

        # export abc with polyPlanes
        start = self.m_start - 1
        end = self.m_end + 1

        out_camera_path = self.nameRoot.camera.shot_camera.product['camera_path']
        options = '-v -j "-ef -worldSpace -fr %s %s -s %s' % (start, end, self.m_step)
        for i in newCamTrans:
            options += ' -rt %s' % i
        options += ' -f %s"' % out_camera_path
        self.dbRecord['files']['camera_path'] = [out_camera_path]

        # TAG FROM ABC CAMERA FILE
        self.dbRecord['tags'] = tag_parser.run(out_camera_path)

        if self.m_polyPlanes:
            print("export image plane!!")
            out_imgplane_path = self.nameRoot.camera.shot_camera.product['imageplane_path']
            # ADDED -writeVisibility -wv for image plane
            options += ' -j "-uv -wv -worldSpace -fr %s %s' % (start, end)
            for i in self.m_polyPlanes:
                options += ' -rt %s' % i
            options += ' -f %s"' % out_imgplane_path
            self.dbRecord['files']['imageplane_path'] = [out_imgplane_path]

        # export other geo loc asset if 'camera_only' is False
        if not (self.info['camera_only']):
            # IF 'cam_geo' under dxCamera node
            if [i for i in cmds.listRelatives(self.dxNodeName, f=1) if i.endswith('cam_geo')]:
                out_camgeo_path = self.nameRoot.camera.shot_camera.product['camera_geo_path']
                options += ' -j "-uv -worldSpace -fr %s %s' % (start, end)
                #options += ' -rt cam_geo'
                options += ' -rt %s' % [i for i in cmds.listRelatives(self.dxNodeName, f=1) if i.endswith('cam_geo')][0]
                options += ' -f %s"' % out_camgeo_path
                self.dbRecord['files']['camera_geo_path'] = [out_camgeo_path]

            # IF 'cam_loc' under dxCamera node
            if [i for i in cmds.listRelatives(self.dxNodeName, f=1) if i.endswith('cam_loc')]:
            #if cmds.ls('cam_loc'):
                out_camloc_path = self.nameRoot.camera.shot_camera.product['camera_loc_path']
                options += ' -j "-uv -worldSpace -fr 1 %s' % end
                #options += ' -j "-uv -worldSpace -fr %s %s' % (start, end)
                #options += ' -rt cam_loc'
                options += ' -rt %s' % [i for i in cmds.listRelatives(self.dxNodeName, f=1) if i.endswith('cam_loc')][0]

                options += ' -f %s"' % out_camloc_path
                self.dbRecord['files']['camera_loc_path'] = [out_camloc_path]

            # ASSET GEO / ASSET LOC
            # assetGeoList = cmds.ls('*_geo')
            # assetLocList = cmds.ls('*_loc')
            assetGeoList = [i for i in cmds.listRelatives(self.dxNodeName, f=1)
                            if (i.endswith('_geo') and not(i.endswith('cam_geo')))]
            assetLocList = [i for i in cmds.listRelatives(self.dxNodeName, f=1)
                            if (i.endswith('_loc') and not(i.endswith('cam_loc')))]

            # if 'cam_geo' in assetGeoList:
            #     assetGeoList.remove('cam_geo')
            # if 'cam_loc' in assetLocList:
            #     assetLocList.remove('cam_loc')

            if assetGeoList:
                self.dbRecord['files']['camera_asset_geo_path'] = []

                for geo in assetGeoList:
                    # ADDED -writeVisibility -wv for geoAsset
                    options += ' -j "-uv -wv -worldSpace -fr %s %s' % (start, end)
                    options += ' -rt %s' % geo
                    self.nameRoot.camera.shot_camera.flag['ASSET_GEO'] = geo.split('|')[-1]
                    geoPath = self.nameRoot.camera.shot_camera.product['camera_asset_geo_path']
                    options += ' -f %s"' % geoPath
                    self.dbRecord['files']['camera_asset_geo_path'].append(geoPath)

            if assetLocList:
                self.dbRecord['files']['camera_asset_loc_path'] = []

                for loc in assetLocList:
                    options += ' -j "-uv -worldSpace -fr %s %s' % (start, end)
                    options += ' -rt %s' % loc
                    self.nameRoot.camera.shot_camera.flag['ASSET_LOC'] = loc.split('|')[-1]
                    locPath = self.nameRoot.camera.shot_camera.product['camera_asset_loc_path']
                    options += ' -f %s"' % locPath
                    self.dbRecord['files']['camera_asset_loc_path'].append(locPath)

            # TODO: 6. export rig key json if exists
            if (self.info.has_key('rig_assets')) and (self.info['rig_assets']):
                self.dbRecord['files']['camera_asset_key_path'] = []
                self.dbRecord['files']['camera_asset_key_abc_path'] = []

                for assetKey in self.info['rig_assets']:
                    # [SINGLE RIG], JSON FILE NAME -> sgAnimation.write()
                    self.nameRoot.camera.shot_camera.flag['ASSET_KEY'] = assetKey
                    rigJson = self.nameRoot.camera.shot_camera.product['camera_asset_key_path']
                    sgAnimation.write([assetKey], rigJson)
                    self.dbRecord['files']['camera_asset_key_path'].append(rigJson)

                    # RIG ABC EXPORT FOR FX TEAM?
                    # TEMP DISALBE THIS AS THIS CAUSE ERROR EXPORTING DXRIG TO ABC.
                    # options += ' -j "-uv -worldSpace -fr %s %s' % (start, end)
                    # options += ' -rt %s' % assetKey
                    # options += ' -f %s"' % rigAbcPath
                    rigAbcPath = self.nameRoot.camera.shot_camera.product['root']
                    abcClass = sgAlembic.CacheExport(
                                FilePath = rigAbcPath,
                                Nodes = [assetKey],
                                Start = start,
                                End = end,
                                Step = self.m_step
                                )

                    abcClass.m_username  = getpass.getuser()
                    abcClass.m_meshTypes = ['render','mid','low']
                    abcClass.doIt()
                    if abcClass.m_logDict['render']:
                        self.dbRecord['files']['camera_asset_key_abc_path'] += abcClass.m_logDict['render']

            # TODO: 6 export json (deprecated version of mmv publish)
            # imageplane
            if self.m_imagePlane:
                out_imgplane_json_path = self.nameRoot.camera.shot_camera.product['imageplane_json_path']
                dplCommon.writeJsonLog(File=out_imgplane_json_path,
                                       Data={'ImagePlane': self.m_imagePlane},
                                       Context=cmds.file(q=True, sn=True),
                                       User=self.info['user'])
                self.dbRecord['files']['imageplane_json_path'] = [out_imgplane_json_path]


        # panzoom to export (camera_only option exception)
        out_panzoom_path = os.path.splitext(self.nameRoot.camera.shot_camera.product['panzoom_json_path'])[0]

        if export_2DPanZoom(out_panzoom_path, orgCamTrans,
                            self.m_start, self.m_end,
                            self.info['user']):
            self.dbRecord['files']['panzoom_json_path'] = [out_panzoom_path + '.json']
            self.dbRecord['files']['panzoom_nuke_path'] = [out_panzoom_path + '.nk']
        print("abc export option : ", options)
        mel.eval('AbcExport %s' % options)

        # cleanup
        if self.m_cleanUp:
            cmds.delete(newCamTrans)
            cmds.delete('export_camera_GRP')
            if self.m_polyPlanes:
                cmds.delete(self.m_polyPlanes)

        return self.dbRecord, self.nameRoot

    def getActualCamera(self, dxCameraNames):
        """
        :param dxCameraNames: list of dxcamera name (ex:[dxCamera1,dxCamera2]
        :return:
        """
        cTransList = []
        for dxCameraName in dxCameraNames:
            # for c in cmds.listRelatives(['dxCamera1'], ad=1, type='camera', fullPath=1):
            for c in cmds.listRelatives([dxCameraName], ad=1, type='camera', fullPath=1):
                for cShape in cmds.ls(c, l=True):
                    cTrans = cmds.listRelatives(cShape, p=True, f=True)[0]
                    cTransList.append(cTrans)
        return cTransList

    def duplicateCameras(self, transList):
        createCameras = []
        exportGroup = cmds.group(n='export_camera_GRP', em=True)
        for c in transList:
            orig_camTrans = c
            orig_camShape = cmds.listRelatives(c, c=True, f=True, type='camera')[0]

            newCamera = cmds.duplicate(c, rr=True)[0]
            cmds.parent(newCamera, exportGroup)
            # rename
            name = cmds.ls(newCamera, l=True)[0]
            new_camTrans = cmds.rename(name, c.split('|')[-1])
            new_camShape = cmds.listRelatives(new_camTrans, c=True, f=True, type='camera')[0]

            # unlock
            attrs = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz', 'sx', 'sy', 'sz']
            attrs += ['hfa', 'vfa', 'fl', 'lsr', 'fs', 'fd', 'sa', 'coi']
            sgCommon.unlockAttrs(new_camTrans, attrs)

            # copy key
            connections = cmds.listConnections(orig_camShape, p=True, type='animCurve')
            if connections:
                for i in connections:
                    dest = cmds.connectionInfo(i, dfs=True)[0]
                    attr = dest.split('.')[-1]
                    cmds.copyKey(orig_camShape, attribute=attr)
                    cmds.pasteKey(new_camShape, attribute=attr)

            # init pivot
            cmds.xform(new_camTrans, ztp=True)
            # key copy
            CameraMtxCopy(orig_camTrans, new_camTrans, self.m_start, self.m_end, self.m_step)

            # get imagePlane data
            print("getImagePlane", orig_camShape, new_camShape)
            self.getImagePlane(orig_camShape)

            # create camera
            createCameras.append(new_camTrans)

        return createCameras

    def getImagePlane(self, sourceCamera):
        imagePlanes = cmds.listConnections(sourceCamera, type='imagePlane', d=False)
        if not imagePlanes:
            return

        for i in list(set(imagePlanes)):
            # get attributes
            attrData = getImagePlaneAttributes(i)
            if attrData:
                name = sourceCamera.split('|')[-1].split(':')[-1]
                if not self.m_imagePlane.has_key(name):
                    self.m_imagePlane[name] = dict()
                self.m_imagePlane[name][i.split(':')[-1]] = attrData

        # create poly imagePlane
        polymeshes, tmpmeshes = createPolyImagePlane(sourceCamera, list(set(imagePlanes)))
        self.m_polyPlanes += polymeshes
        # key copy by python API 2.0
        for i in range(len(polymeshes)):
            print(polymeshes)
            matrixs, frames = sgCommon.getMtx(tmpmeshes[i], self.m_start, self.m_end, self.m_step)
            sgCommon.setMtx(polymeshes[i], matrixs, frames)
        cmds.select(polymeshes)
        cmds.filterCurve()

        if self.m_cleanUp:
            cmds.delete(tmpmeshes)

    def keyFrameOffset(self, objects, offsetframe):
        attrList = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']

        for i in objects:
            for a in attrList:
                sgCommon.offsetKeyAttr(i, a, offsetframe)



#-------------------------------------------------------------------------------
#
#	2DPanZoom
#
#------------------------------------------------------------------------------
def exportPanZoomDialog():
    startpath = cmds.workspace( q=True, rd=True )
    fn = cmds.fileDialog2( fm=0,
                           cap='Export 2D Pan/Zoom',
                           okc='export',
                           dir=startpath )
    if not fn:
        return

    min = int( cmds.playbackOptions(q=True, min=True) )
    max = int( cmds.playbackOptions(q=True, max=True) )

    selections = cmds.ls( sl=True, dag=True, ca=True )
    if not selections:
        selections = cmds.ls( dag=True, ca=True )

    fileName = fn[0].split('.')[0] + '.panzoom'
    exportPanZoom( fileName, selections, min, max, None )
    # debug
    mel.eval( 'print "# Result : 2D Pan/Zoom export <%s>\\n"' % fileName )


def backup_exportPanZoom( fileName, cameras, start, end, username ):
    keyData = dict()
    nukeScene = ''
    for c in cameras:
        shape = cmds.ls( c, dag=True, type='camera' )[0]
        if cmds.getAttr( '%s.panZoomEnabled' % shape ):
            data = sgCommon.attributesKeyDump( shape, ['hpn', 'vpn', 'zom'] )
            keyData[shape.split('|')[-1]] = data
            # for Nuke
            nukeNode = toNuke_2DPanZoom( shape, start-1, end+1 )
            if nukeNode:
                nukeScene += nukeNode
    if keyData:
        dplCommon.writeJsonLog( File=fileName,
                                Data={'2DPanZoom': keyData},
                                Context=cmds.file(q=True, sn=True),
                                User=username )
    if nukeScene:
        fn = fileName.replace( '.panzoom', '_panzoom.nk' )
        f  = open( fn, 'w' )
        f.write( nukeScene )
        f.close()


def export_2DPanZoom( fileName, cameras, start, end, username ):
    print("export_2DPanZoom called!!")
    keyData = dict()
    nukeScene = ''
    jsonFile = fileName + '.json'
    nukeFile = fileName + '.nk'
    for c in cameras:
        shape = cmds.ls( c, dag=True, type='camera' )[0]
        if cmds.getAttr( '%s.panZoomEnabled' % shape ):
            data = sgCommon.attributesKeyDump( shape, ['hpn', 'vpn', 'zom'] )
            keyData[shape.split('|')[-1]] = data
            # for Nuke
            nukeNode = toNuke_2DPanZoom( shape, start-1, end+1 )
            if nukeNode:
                nukeScene += nukeNode
    print("keyData", keyData)

    if keyData:
        dplCommon.writeJsonLog( File=jsonFile,
                                Data={'2DPanZoom': keyData},
                                Context=cmds.file(q=True, sn=True),
                                User=username )
    if nukeScene:
        f  = open( nukeFile, 'w' )
        f.write( nukeScene )
        f.close()

    if keyData:
        return True
    else:
        return False

# to Nuke
def toNuke_2DPanZoom( cameraShape, start, end ):
    nukeNode = None
    cameraName = cmds.listRelatives( cameraShape, p=True )[0]
    if not cmds.getAttr( '%s.panZoomEnabled' % cameraShape ):
        return

    hfa = cmds.getAttr( '%s.hfa' % cameraShape )

    # Nuke Camera
    nukeNode  = 'Camera {\n'
    nukeNode += '  inputs 0\n'

    # horizontalPan
    connections = cmds.listConnections( '%s.hpn' % cameraShape, type='animCurve' )
    if connections:
        tx = '{curve'
        for f in range( start, end+1 ):
            value = cmds.getAttr( '%s.hpn' % cameraShape, t=f )
            tx += ' x%d' % f
            tx += ' %.8f' % ( value / hfa * 2 )
        tx += '}'
    else:
        value = cmds.getAttr( '%s.hpn' % cameraShape )
        tx = '%.8f' % ( value / hfa * 2 )

    # verticalPan
    connections = cmds.listConnections( '%s.vpn' % cameraShape, type='animCurve' )
    if connections:
        ty = '{curve'
        for f in range( start, end+1 ):
            value = cmds.getAttr( '%s.vpn' % cameraShape, t=f )
            ty += ' x%d' % f
            ty += ' %.8f' % ( value / hfa * 2 )
        ty += '}'
    else:
        value = cmds.getAttr( '%s.vpn' % cameraShape )
        ty = '%.8f' % ( value / hfa * 2 )

    nukeNode += '  win_translate {%s %s}\n' % ( tx, ty )

    # zoom
    connections = cmds.listConnections( '%s.zom' % cameraShape, type='animCurve' )
    if connections:
        sc = '{curve'
        for f in range( start, end+1 ):
            value = cmds.getAttr( '%s.zom' % cameraShape, t=f )
            sc += ' x%d' % f
            sc += ' %.8f' % value
        sc += '}'
    else:
        sc = '%.8f' % cmds.getAttr( '%s.zom' % cameraShape )

    nukeNode += '  win_scale {%s %s}\n' % ( sc, sc )

    nukeNode += '  name %s_2DPanZoom\n' % cameraName
    nukeNode += '}\n'

    return nukeNode



################################################################################################

def createNukeString2DShake(
        cameraShape,
        start,
        end,
        hfa,
        hpnSource,
        vpnSource,
        zomSource,
        panzoomShakeAttr
):
    # Nuke Camera
    nukeNode = 'Camera {\n'
    nukeNode += '  inputs 0\n'

    # horizontalPan
    if hpnSource:
        tx = '{curve'
        for f in range(start, end + 1):
            attr = panzoomShakeAttr.get('attr', '.hpn')
            value = cmds.getAttr(hpnSource + attr, t=f)
            tx += ' x%d' % f
            tx += ' %.8f' % (value / hfa * 2)
        tx += '}'
    else:
        value = cmds.getAttr(cameraShape + '.hpn')
        tx = '%.8f' % (value / hfa * 2)

    # verticalPan
    if vpnSource:
        ty = '{curve'
        for f in range(start, end + 1):
            attr = panzoomShakeAttr.get('attr', '.vpn')
            value = cmds.getAttr(hpnSource + attr, t=f)
            ty += ' x%d' % f
            ty += ' %.8f' % (value / hfa * 2)
        ty += '}'
    else:
        value = cmds.getAttr(cameraShape + '.vpn')
        ty = '%.8f' % (value / hfa * 2)

    nukeNode += '  win_translate {%s %s}\n' % (tx, ty)

    # zoom
    if not panzoomShakeAttr and zomSource:
        sc = '{curve'
        for f in range(start, end + 1):
            value = cmds.getAttr('%s.zom' % cameraShape, t=f)
            sc += ' x%d' % f
            sc += ' %.8f' % value
        sc += '}'
    else:
        sc = '%.8f' % cmds.getAttr('%s.zom' % cameraShape)

    nukeNode += '  win_scale {%s %s}\n' % (sc, sc)

    nukeNode += '  name {0}_{1}\n'.format(cameraShape, panzoomShakeAttr.get('name', '2DPanZoom'))
    nukeNode += '}\n'

    return nukeNode


def toNuke_2DShake(cameraShape, start, end):
    panzoomString = str()
    shakeString = str()

    if not cmds.getAttr('%s.panZoomEnabled' % cameraShape):
        return

    hfa = cmds.getAttr('%s.hfa' % cameraShape)
    hpnSource = cmds.listConnections(cameraShape + '.hpn', s=True, d=False)
    vpnSource = cmds.listConnections(cameraShape + '.vpn', s=True, d=False)
    zomSource = cmds.listConnections(cameraShape + '.zom', s=True, d=False)

    panzoomAttr = dict()
    panzoomShakeAttr = dict()

    if hpnSource:
        hpnSource = hpnSource[0]

        if cmds.nodeType(hpnSource) == 'animBlendNodeAdditive':
            panzoomAttr['attr'] = '.inputA'
            panzoomShakeAttr['attr'] = '.inputB'
            panzoomShakeAttr['name'] = 'shake'
        else:
            hpnSource = cameraShape

    if vpnSource:
        vpnSource = vpnSource[0]

        if cmds.nodeType(vpnSource) != 'animBlendNodeAdditive':
            vpnSource = cameraShape

    panzoomString = createNukeString2DShake(
        cameraShape,
        start=start,
        end=end,
        hfa=hfa,
        hpnSource=hpnSource,
        vpnSource=vpnSource,
        zomSource=zomSource,
        panzoomShakeAttr=panzoomAttr
    )
    if panzoomAttr:
        shakeString = createNukeString2DShake(
            cameraShape,
            start=start,
            end=end,
            hfa=hfa,
            hpnSource=hpnSource,
            vpnSource=vpnSource,
            zomSource=zomSource,
            panzoomShakeAttr=panzoomShakeAttr
        )
    panzoomString += shakeString

    return panzoomString

def createShakeKeyLog(node, start, end, attr):
    logDict = {'frame': list(),
               'value': list(),
               'infinity': ['constant', 'constant']}

    for i in range(start, end+1):
        value = cmds.getAttr(node + '.{}'.format(attr), t=i)
        logDict['frame'].append(i)
        logDict['value'].append(value)
        # logDict['angle'].append(0.0)
    return logDict

def exportPanZoom( fileName, cameras, start, end, username ):
    keyData = dict()
    shakeData = dict()
    nukeScene = ''
    for c in cameras:
        additiveData = dict()
        shape = cmds.ls( c, dag=True, type='camera' )[0]
        if cmds.getAttr( '%s.panZoomEnabled' % shape ):
            data = sgCommon.attributesKeyDump(shape, ['hpn', 'vpn', 'zom'])
            # --------   gh edit --------------------------------------
            shapeSourceH = cmds.listConnections(shape + '.hpn', s=True, d=False)
            shapeSourceV = cmds.listConnections(shape + '.vpn', s=True, d=False)
            if shapeSourceH:
                if cmds.nodeType(shapeSourceH[0]) == 'animBlendNodeAdditive':
                    shakeDataH = createShakeKeyLog(shapeSourceH[0], start, end, 'inputB')
                    shakeDataV = createShakeKeyLog(shapeSourceV[0], start, end, 'inputB')
                    panDataH = sgCommon.attributesKeyDump(shapeSourceH[0], ['inputA'])
                    panDataV = sgCommon.attributesKeyDump(shapeSourceV[0], ['inputA'])
                    zomData = sgCommon.attributesKeyDump(shape, ['zom'])
                    additiveData[shape.split('|')[-1]] = {
                        'hpn': panDataH['inputA'],
                        'vpn': panDataV['inputA'],
                        'zom': zomData['zom']
                    }
                    shakeData[shape.split('|')[-1]] = {'horizontal': shakeDataH,
                                                       'vertical': shakeDataV}
            if not additiveData:
                # anim key
                keyData[shape.split('|')[-1]] = data
            else:
                # additive key
                keyData.update(additiveData)

            # for Nuke
            nukeNode = toNuke_2DShake(shape, start - 1, end + 1)
            if nukeNode:
                nukeScene += nukeNode
    if keyData:
        dplCommon.writeJsonLog( File=fileName,
                                Data={'2DPanZoom': keyData},
                                Context=cmds.file(q=True, sn=True),
                                User=username )
    if shakeData:
        dplCommon.writeJsonLog(File=fileName.replace('.panzoom', '.shake'),
                               Data={'2DShake': shakeData},
                               Context=cmds.file(q=True, sn=True),
                               User=username)

    if nukeScene:
        fn = fileName.replace( '.panzoom', '_panzoom.nk' )
        f  = open( fn, 'w' )
        f.write( nukeScene )
        f.close()
