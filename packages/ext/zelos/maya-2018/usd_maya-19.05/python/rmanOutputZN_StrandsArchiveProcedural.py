#encoding=utf-8
#!/usr/bin/env python
#---------------------------------------------------#
#   author		 : Dohyeon Yang @Dexter Studios		#
#	last updates : 2017.05.29						#
#---------------------------------------------------#

import os, sys, shutil
import numpy as np

import maya.cmds as cmds
import maya.mel as mel

def ls(directory):
    """full-featured solution, via wrapping find"""
    import os
    files = os.popen4('find %s' % directory)[1].read().rstrip().split('\n')
    files.remove(directory)
    n = len(directory)
    if directory[-1] != os.path.sep:
        n += 1
    files = [f[n:] for f in files] # remove dir prefix
    return [f for f in files if os.path.sep not in f] # remove files in sub-directories

#-------------------------------------------------related to ZEnvArchiveV2
def getExisitingFrame(cfr, path):
    filenames = os.listdir(path)

    frames = []
    for filename in filenames:
        if os.path.isdir('%s/%s' % (path, filename)):
            continue
        if filename == 'follicles':
            continue
        if filename == 'info':
            continue
        frames.append(filename.split('_')[1][1:])

    existingFrame = -1
    if float(cfr) < float(min(frames)):
        existingFrame = min(frames)
    elif float(max(frames)) < float(cfr):
        existingFrame = max(frames)
    else:
        existingFrame  = cfr

    #if existingFrame  == max(frames):
    #    print str(int(existingFrame) - 1)
    return existingFrame

def readBBoxData(frame, cpt):
    bboxFilename = ''
    if frame < 0:
        bboxFilename = 'bbox_m%s' % frame
    else:
        bboxFilename = 'bbox_p%s' % frame

    bboxPath = os.path.join(cpt, bboxFilename)
    if not os.path.exists(bboxPath):
        print 'Error@readBBoxData()::not exist bbox file = %s' % bboxPath
        return

    bboxf = open(bboxPath, 'r')
    bbox = bboxf.read().split('\n')[0]
    bboxf.close()

    offset = 100.0

    bounds = bbox.split(' ')

    bounds[0] = str(float(bounds[0]) - offset)
    bounds[1] = str(float(bounds[1]) + offset)
    bounds[2] = str(float(bounds[2]) - offset)
    bounds[3] = str(float(bounds[3]) + offset)
    bounds[4] = str(float(bounds[4]) - offset)
    bounds[5] = str(float(bounds[5]) + offset)

    bound = '%s %s %s %s %s %s' % (bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5])
    return bound

def rmanOutputZN_StrandsArchiveStandardBindingProcedrual():
    print "rmanOutputZN_StrandsArchiveStandardBindingProcedrual"

def rmanOutputZN_StrandsArchiveRigidBindingProcedrual():
    if 'ZENNForMaya' not in cmds.pluginInfo(query=True, listPlugins=True):
        print 'Error@rmanOutputZN_StrandsArchiveRigidBindingProcedrual()::not loaded ZENNForMaya'
        return

    renderGlobals       = 'renderManRISGlobals'

    node = mel.eval('rman ctxGetObject')
    cfr = int(mel.eval('rman getvar F4'))

    path = "%s/%s" % (cmds.getAttr('%s.inZennCachePath' % node), cmds.getAttr('%s.inZennCacheName' % node))




    #------------------------------------------------------------------------------------------ set opt
    cpt = os.path.dirname(path)
    cnm = os.path.basename(path)
    cfr = cfr
    sts = '%0.2f' % (cmds.getAttr('%s.rman__toropt___shutterAngle' % renderGlobals) / 360.0)

    rr0 = 1.0 #cmds.getAttr('%s.rr0' % node)
    sp0 = 1
    df0 = 1
    tm0 = 1

    rr1 = 0.0
    sp1 = 0
    df1 = 0
    tm1 = 0

    col = 0.0
    rws = cmds.getAttr('%s.rws' % node)
    tws = cmds.getAttr('%s.tws' % node)
    ops = cmds.getAttr('%s.ops' % node)
    mbr = cmds.getAttr('%s.rman__torattr___motionBlur' % renderGlobals)
    lns = cmds.getAttr('%s.lns' % node)

    bmod = 1

    efr     = int(cmds.getAttr('%s.efr' % node) + 0.5)         #getExisitingFrame(cfr, path)

    #abcpath = cmds.getAttr('%s.outAbcCachePath' % node)
    abcpath = cmds.getAttr('%s.inAbcCachePath' % node)

    bound = readBBoxData(efr, path)

    dsoName= 'ZennStrandsDSO_rigidBinding'

    opt  = '--cpt %s'                       % cpt
    opt += ' --cnm %s'                      % cnm
    opt += ' --cfr %s'                      % cfr
    opt += ' --sts %s'                      % sts
    opt += ' --rr0 %.3f'                    % rr0	    # main ratio
    opt += ' --sp0 %.3f --df0 %s --tm0 %s'  % (sp0, df0, tm0)
    opt += ' --rr1 %.3f'                    % rr1
    opt += ' --sp1 %.3f --df1 %s --tm1 %s'  % (sp1, df1, tm1)
    opt += ' --col %.3f'                    % col	    # cutoffLength
    opt += ' --rws %.3f'                    % rws	    # rootWidthScale
    opt += ' --tws %.3f'                    % tws	    # tipWidthScale
    opt += ' --ops %.3f'                    % ops	    # opacityScale
    opt += ' --mbr %s'                      % mbr       # motionBlur
    opt += ' --lns %.3f'                    % lns	    # lengthScale
    opt += ' --bmod %s'                     % bmod
    opt += ' --efr %s'                      % efr
    opt += ' --abcpath %s'                  % abcpath	# abcpath

    mel.eval( 'RiArchiveRecord("verbatim", "##RLF Inject SurfaceShading -attribute sets@,initialShadingGroup,\\n")' )

    #camvis   = int(cmds.getAttr('%s.cameraVisibility' % node))
    #indirvis = int(cmds.getAttr('%s.indirectVisibility' % node))
    #transvis = int(cmds.getAttr('%s.transmissionVisibility' % node))
    #mel.eval( 'RiAttribute "visibility" "int camera" %s "int indirect" %s "int transmission" %s' % (camvis, indirvis, transvis))

    mel.eval('RiAttributeBegin' )
    mel.eval('RiAttribute "user" "int zhair" 1' )
    mel.eval('RiAttribute "dice" "int roundcurve" 1 "int hair" 1' )

    mel.eval('RiAttributeBegin' )
    mel.eval('RiAttribute "identifier" "string name" %s' % cnm )
    mel.eval('RiArchiveRecord("verbatim", "##RLF Inject SurfaceShading -attribute sets@,initialShadingGroup,zfur,\\n")' )
    mel.eval('RiTransformBegin' )
    mel.eval('RiAttribute "procedural" "int immediatesubdivide" 1' )

    mel.eval('RiProcedural "DynamicLoad" "%s" %s "%s"' % (dsoName, bound, opt))

    mel.eval('RiTransformEnd' )
    mel.eval('RiAttributeEnd' )
    mel.eval('RiAttributeEnd' )
