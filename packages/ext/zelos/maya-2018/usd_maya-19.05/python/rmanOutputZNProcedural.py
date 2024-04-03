#encoding=utf-8
#!/usr/bin/env python
#---------------------------#
#   Jang, Julie             #
#   2017.03.23  v00         #
#---------------------------#

import os, sys, shutil
import numpy as np

import maya.cmds as cmds
import maya.mel as mel

#import lgtCommon

from maya.OpenMaya import MMatrix
from maya.OpenMaya import MPoint

def getRealCameraShape( name ):
    if cmds.objExists( name ):
        return name

    nsList = list()
    for i in cmds.namespaceInfo( lon=True ):
        if i.find('UI') == -1 and i.find('shared') == -1:
            nsList.append( i )
    if not nsList:
        return name
    camName = None
    for ns in nsList:
        if name.find( '%s_' % ns ) > -1:
            tmp = name.replace( '%s_' % ns, '%s:' % ns  )
            if cmds.objExists( tmp ):
                camName = tmp
                break
    return camName


#---------------------------------------- common
def getAttrWrapper(node, attr_name):
    node = str(node)
    value = cmds.getAttr('%s.%s' % (node, attr_name))
    return value

def offsetViewFrustum(s, ntr, ntl, nbr, nbl, ftr, ftl, fbr, fbl):
    u = (nbr - nbl) / np.linalg.norm(nbr - nbl)
    v = (ntl - nbl) / np.linalg.norm(ntl - nbl)
    ntl = ntl - s*u + s*v
    ntr = ntr + s*u + s*v
    nbr = nbr + s*u - s*v
    nbl = nbl - s*u - s*v
    
    u = (fbr - fbl) / np.linalg.norm(fbr - fbl)
    v = (ftl - fbl) / np.linalg.norm(ftl - fbl)
    ftl = ftl - s*u + s*v
    ftr = ftr + s*u + s*v
    fbr = fbr + s*u - s*v
    fbl = fbl - s*u - s*v
    
    return [ntr, ntl, nbr, nbl, ftr, ftl, fbr, fbl]

def cacheOutViewFrustumPoints(node, cam_node):
    cwm = cmds.getAttr('%s.wm' % cam_node)

    cs = cmds.camera(cam_node, q=True, cameraScale=True)
    lsr = cmds.camera(cam_node, q=True, lensSqueezeRatio=True)
    ar = cmds.camera(cam_node, q=True, aspectRatio=True)

    hfa = cmds.camera(cam_node, q=True, horizontalFilmAperture=True) #fov
    vfa = cmds.camera(cam_node, q=True, verticalFilmAperture=True)

    ncp = cmds.camera(cam_node , q=True, nearClipPlane=True)
    fcp = cmds.camera(cam_node, q=True, farClipPlane=True)

    fl = cmds.camera(cam_node, q=True, focalLength=True)
    o = cmds.camera(cam_node, q=True, orthographic=True)
    ow = cmds.camera(cam_node, q=True, orthographicWidth=True)
    
    
    h_fov = hfa * 0.5 / (fl * 0.03937) * cs * lsr
    v_fov = vfa * 0.5 / (fl * 0.03937) * cs

    near_right  = ncp * h_fov
    near_top    = ncp * v_fov
    far_right   = fcp * h_fov
    far_top     = fcp * v_fov
    if o == True:
        near_right = near_top = far_right = far_top = ow / 2.0
    
    ntr = np.array([ near_right,  near_top, -ncp, 1.0])
    ntl = np.array([-near_right,  near_top, -ncp, 1.0])
    nbr = np.array([ near_right, -near_top, -ncp, 1.0])
    nbl = np.array([-near_right, -near_top, -ncp, 1.0])
    
    ftr = np.array([ far_right,  far_top, -fcp, 1.0])
    ftl = np.array([-far_right,  far_top, -fcp, 1.0])
    fbr = np.array([ far_right, -far_top, -fcp, 1.0])
    fbl = np.array([-far_right, -far_top, -fcp, 1.0])
    
    cwm = np.array([[cwm[0], cwm[4], cwm[8],  cwm[12]],
                    [cwm[1], cwm[5], cwm[9],  cwm[13]],
                    [cwm[2], cwm[6], cwm[10], cwm[14]],
                    [cwm[3], cwm[7], cwm[11], cwm[15]]])

    #print 'col-major'
    ntr = cwm.dot(ntr)
    ntl = cwm.dot(ntl)
    nbr = cwm.dot(nbr)
    nbl = cwm.dot(nbl)
    ftr = cwm.dot(ftr)
    ftl = cwm.dot(ftl)
    fbr = cwm.dot(fbr)
    fbl = cwm.dot(fbl)
    
    offsetScale = 5.0
    #return [ntr, ntl, nbr, nbl, ftr, ftl, fbr, fbl]
    return offsetViewFrustum(offsetScale, ntr, ntl, nbr, nbl, ftr, ftl, fbr, fbl)

def rmanOutputDynamicLoad(frame, path, node, bbox):
    dsoName= 'ZennDSO'
    #dso option
    renderGlobals       = 'renderManRISGlobals'

    cache_path          = os.path.dirname(path)
    cache_name          = os.path.basename(path)

    start_frame         = cmds.playbackOptions(query=True, minTime=True)
    end_frame           = cmds.playbackOptions(query=True, maxTime=True)
    current_frame       = frame

    #TODO::get value from node
    motion_blur         = cmds.getAttr('%s.rman__torattr___motionBlur' % renderGlobals)
    motion_blur_type    = cmds.getAttr('%s.rman__toropt___motionBlurType' % renderGlobals)
    shutter_angle       = cmds.getAttr('%s.rman__toropt___shutterAngle' % renderGlobals)
    motion_sample       = cmds.getAttr('%s.rman__torattr___motionSamples' % renderGlobals)
    step_size           = '%0.2f' % (shutter_angle / 360.0)
    
    #cacheout view-frustum 8-point position
    cam_node = mel.eval('rman getvar CAMERA')
    #print 'before %s' % cam_node
    cam_node = getRealCameraShape(cam_node)
    #print 'camera name : %s' % cam_node
    
    [ntr, ntl, nbr, nbl, ftr, ftl, fbr, fbl] = cacheOutViewFrustumPoints(node, cam_node)

    options =  '--cpt %s' % cache_path                 #cachePath
    options += ' --cnm %s' % cache_name                    #cacheName
    options += ' --stf %s' % start_frame
    options += ' --enf %s' % end_frame
    options += ' --cfr %s' % current_frame             #currentFrame
    options += ' --mbr %s' % motion_blur                   #motionBlur
#    options += ' --sta %s' % shutter_angle
#    options += ' --mos %s' % motion_sample
#    #options += ' -sts %s' % step_size

#    options += ' --ntr %s %s %s' % (ntr[0], ntr[1], ntr[2])
#    options += ' --ntl %s %s %s' % (ntl[0], ntl[1], ntl[2])
#    options += ' --nbr %s %s %s' % (nbr[0], nbr[1], nbr[2])
#    options += ' --nbl %s %s %s' % (nbl[0], nbl[1], nbl[2])

#    options += ' --ftr %s %s %s' % (ftr[0], ftr[1], ftr[2])
#    options += ' --ftl %s %s %s' % (ftl[0], ftl[1], ftl[2])
#    options += ' --fbr %s %s %s' % (fbr[0], fbr[1], fbr[2])
#    options += ' --fbl %s %s %s' % (fbl[0], fbl[1], fbl[2])
    # RiProcedure
    mel.eval('RiArchiveRecord("comment", "dynamicLoadInjection::%s")' % node )
    mel.eval('RiArchiveRecord("verbatim", "##RLF Inject SurfaceShading -attribute sets@,initialShadingGroup,\\n")' )
    mel.eval('RiTransformBegin' )
    mel.eval('RiProcedural "DynamicLoad" "%s" %s "%s"' % (dsoName, bbox, options) )
    #TODO
    #mel.eval('RiArchiveRecord("verbatim", "RiProcedural2\\n")' )
    mel.eval('RiTransformEnd' )

def rmanOutputRenderStat(node):
    mel.eval('RiArchiveRecord("comment", "renderStat::%s")' % node )
    rman__riattr__visibility_camera = getAttrWrapper(node, 'rman__riattr__visibility_camera')
    rman__riattr__visibility_indirect = getAttrWrapper(node, 'rman__riattr__visibility_indirect')
    rman__riattr__visibility_transmission = getAttrWrapper(node, 'rman__riattr__visibility_transmission')

def readBBoxData(frame, cachePath):
    bboxFilename = ''
    if frame < 0:
        bboxFilename = 'bbox_m%s' % frame
    else:
        bboxFilename = 'bbox_p%s' % frame

    bboxPath = os.path.join(cachePath, bboxFilename)
    if not os.path.exists(bboxPath):
        print 'Error@readBBoxData()::not exist bbox file = %s' % bboxPath
        return

    bboxf = open(bboxPath, 'r')
    bbox = bboxf.read().split('\n')[0]
    bboxf.close()

    return bbox

#--------------------------------------- ZNGroup
def getTempCachePath():

    path = mel.eval('rman getvar rfmRIBs').replace('$STAGE', mel.eval('rman getvar STAGE'))
    path = mel.eval('rman getvar RMSPROJ') + path + '/cache'
    return path

def rmanOutputZNGroupProcedural():
    if 'ZENNForMaya' not in cmds.pluginInfo(query=True, listPlugins=True):
        print 'Error@rmanOutputZNGroupProcedural()::not loaded ZENNForMaya'
        return

    if cmds.about(batch=True):
        print 'Warrning@rmanOutputZNGroupProcedural::not rendered in batch render mode'
        return
    
    node    = mel.eval('rman ctxGetObject')
    print 'node name = %s' % node
    frame   = int(mel.eval('rman getvar F4'))
    path    = getTempCachePath();
    
    print 'path = %s' % path
    
    base_path = path
    cache_path = path + '/' + node

    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
        os.makedirs(cache_path)
    else:
        os.makedirs(cache_path)

    cmds.ZN_CacheGen2Cmd(startFrame = frame, endFrame = frame, step = 1.0, cachePath=base_path, nodeNames=node)
    cmds.currentTime(frame)
    bbox = readBBoxData(frame, cache_path)

    mel.eval( 'RiAttribute "shade" "string transmissionhitmode" "shader"' )

    if getAttrWrapper(node, 'rman__riattr__visibility_transmission'):
        mel.eval( 'RiAttribute "grouping" "string membership" "+reflection,refraction,shadow"' )
    else:
        mel.eval( 'RiAttribute "grouping" "string membership" "+reflection,refraction"' )

    #rmanOutputRenderStat(node)
    rmanOutputDynamicLoad(frame, cache_path, node, bbox)

    #print 'log@rmanOutputZNGroupProcedural(): frame = %d, node = %s rib injection complete' % (frame, node)

#-------------------------------------------------related to ZNArchive
def rmanOutputZNArchiveProcedural():
    if 'ZENNForMaya' not in cmds.pluginInfo(query=True, listPlugins=True):
        print 'Error@rmanOutputZNArchiveProcedural()::not loaded ZENNForMaya'
        return

    node  = mel.eval('rman ctxGetObject')
    frame = int(getAttrWrapper(node, 'renderFrame'))
    path  = getAttrWrapper(node, 'cachePath')

    if path == '':
        print 'Warnning@rmanOutputZNArchiveProcedural():Cache Path is empty.'
        return

    bbox = readBBoxData(frame, path)

    #rmanOutputRenderStat(node)
    rmanOutputDynamicLoad(frame, path, node, bbox)
    print 'log@rmanOutputZNArchiveProcedural(): frame = %d, node = %s rib injection complete' % (frame, node)


#------------------------------------------------- related To Test
def rmanOutputTestProcedural():
    node  = mel.eval('rman ctxGetObject')
    cam = mel.eval('rman getvar CAMERA')
    mel.eval('RiArchiveRecord("comment", "dynamicLoadInjection::%s")' % node )

def getCachePath():
    path= mel.eval('rman getvar rfmRIBs').replace( '$STAGE', mel.eval('rman getvar STAGE') )
    path= mel.eval('rman getvar RMSPROJ') + path + '/cache'
    return path

def getTargetNode( node ):
    target = None
    if cmds.nodeType(node) == 'ZN_StrandsViewer':
        target= cmds.connectionInfo( '%s.inStrands' % node, sfd=True )
        if not target:
            return None
        target= target.split('.')[0]
        if len( cmds.connectionInfo('%s.outStrands' % target, dfs=True) ) > 1:
            return None

    elif cmds.nodeType(node) == 'ZN_FeatherSetViewer':
        target= cmds.connectionInfo('%s.inFeatherSet' % node, sfd=True)
        if not target:
            return None
        target= target.split('.')[0]

    return target

def readBBoxData(frame, cachePath):
    bboxFilename = ''
    if frame < 0:
        bboxFilename = 'bBox_m%s' % frame
    else:
        bboxFilename = 'bBox_p%s' % frame

    bboxPath = os.path.join(cachePath, bboxFilename)
    if not os.path.exists(bboxPath):
        print 'Error@readBBoxData()::not exist bbox file = %s' % bboxPath
        return

    bboxf = open(bboxPath, 'r')
    bbox = bboxf.read().split('\n')[0]
    bboxf.close()

    return bbox

def rmanOutputZNViewerProcedural():
    if 'ZENNForMaya' not in cmds.pluginInfo(query=True, listPlugins=True):
        cmds.loadPlugin('ZENNForMaya')


    if cmds.about(batch=True):
        print 'Warrning@rmanOutputZEProcedural::not rendered in batch render mode'
        return

    node    = mel.eval('rman ctxGetObject')
    time    = int(mel.eval('rman getvar F4'))
    path    = getCachePath()

    # target node
    target = getTargetNode(node)
    if not target:
        return

    targetpath = getCachePath() + '/%s' % target

    if os.path.exists(targetpath):
        shutil.rmtree(targetpath)
        os.makedirs(targetpath)

    cmds.ZN_CacheGenCmd(startFrame=time, endFrame=time, cachePath=path, nodeNames=target)

    # check cache path exists
    path = getCachePath() + '/%s' % target
    if not os.path.exists(path):
        print 'Warrning@rmanOutputZEProcedural::cache gen error'
        return

    # bound
    if time < 0:
        filename = 'bbox_m%s' % time
    else:
        filename = 'bbox_p%s' % time
    fn = os.path.join( path, filename )
    if not os.path.exists( fn ):
        return

    f = open( fn, 'r' )
    bound = f.read().split('\n')[0]
    f.close()

    # DSO options
    dsoName         = ''
    rootWidth       = 1.0
    tipWidth        = 1.0
    opacity         = 1.0
    motionBlur      = cmds.getAttr('renderManRISGlobals.rman__torattr___motionBlur')
    options         =  '-cpt %s' % os.path.dirname(path)
    options         += ' -cnm %s' % os.path.basename(path)

    if cmds.nodeType(node) == 'ZN_StrandsViewer':
        dsoName = 'ZennStrandsDSO'
        options+= ' -cfr %s -nfr %s' % (time, time+1)
        options+= ' -rr0 1.0'# renderRatio0
        options+= ' -rr1 0.0'# renderRatio1
        options+= ' -sp0 1'# specularVisibility0
        options+= ' -df0 1'# diffuseVisibility0
        options+= ' -tm0 1'# transmissionVisibility0
        options+= ' -sp1 0'# specularVisibility1
        options+= ' -df1 0'# diffuseVisibility1
        options+= ' -tm1 0'# transmissionVisibility1
        options+= ' -col 0.0'# cutoffLength
        options+= ' -rws %0.3f' % rootWidth# rootWidthScale
        options+= ' -tws %0.3f' % tipWidth# tipWidthScale
        options+= ' -ops %0.3f' % opacity# opacityScale
        options+= ' -mbr %s' % motionBlur# motionBlur
        options+= ' -lns 1.0'# lengthScale
        options+= ' -prt 0'# printReport

    elif cmds.nodeType(node) == 'ZN_FeatherSetViewer':
        dsoName = 'ZennFeathersDSO'
        options+= ' -cfr %s -nfr %s' % (time, time+1)
        options+= ' -rws %0.3f' % rootWidth# rootWidthScale
        options+= ' -tws %0.3f' % tipWidth# tipWidthScale
        options+= ' -ops %0.3f' % opacity# opacityScale
        options+= ' -mbr %s' % motionBlur# motionBlur
        options+= ' -prt 0'# printReport

    # RiProcedure
    mel.eval( 'RiAttributeEnd' )
    mel.eval( 'RiAttributeEnd' )
    mel.eval( 'RiAttributeEnd' )
    mel.eval( 'RiArchiveRecord("comment", "ZennArchive %s")' % node )
    mel.eval( 'RiAttributeBegin' )
    mel.eval( 'RiAttributeBegin' )
    mel.eval( 'RiAttributeBegin' )
    mel.eval( 'RiAttribute "identifier" "string name" "ZennArchive_%s"' % target )
    mel.eval( 'RiAttribute "visibility" "int camera" 1' )

    # ZennStrands or ZennFeather
    if cmds.nodeType(node) == 'ZN_StrandsViewer' or cmds.nodeType(node) == 'ZN_FeatherSetViewer':
        mel.eval( 'RiAttribute "user" "int zhair" 1' )
        mel.eval( 'RiAttribute "trace" "int maxspeculardepth" 0' )
        mel.eval( 'RiAttribute "dice" "int roundcurve" 1 "int hair" 1' )
        mel.eval( 'RiArchiveRecord("verbatim", "##RLF Inject SurfaceShading -attribute sets@,initialShadingGroup,zfur,\\n")' )
        mel.eval( 'RiTransformBegin' )
        #mel.eval( 'RiAttribute "procedural" "int immediatesubdivide" 1' )
        mel.eval( 'RiProcedural "DynamicLoad" "%s" %s "%s"' % (dsoName, bound, options) )
        mel.eval( 'RiTransformEnd' )
        # do not RiAttributeEnd * 3




