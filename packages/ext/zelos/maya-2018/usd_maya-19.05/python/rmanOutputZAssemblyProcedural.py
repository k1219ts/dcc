#encoding=utf-8
#!/usr/bin/env python
#---------------------------#
#   Yang, dohyeon           #
#   2017.03.29  v00         #
#---------------------------#

import os, sys, shutil

import maya.cmds as cmds
import maya.mel as mel

import lgtCommon
import ribCommon




def rmanOutputZAssemblyArchiveProcedural():
    if 'ZMayaTools' not in cmds.pluginInfo(query=True, listPlugins=True):
        print 'Error@rmanOutputZAssemblyArchiveProcedural()::not loaded ZMayaTools'
        return

    renderGlobals       = 'renderManRISGlobals'

    node  = mel.eval('rman ctxGetObject')

    afp = cmds.getAttr('%s.afp' % node)
    cfr = int(mel.eval('rman getvar F4'))
    gsc = cmds.getAttr('%s.gsc' % node)

    mbr             = cmds.getAttr('%s.rman__torattr___motionBlur' % renderGlobals)
    mbrtype         = cmds.getAttr('%s.rman__toropt___motionBlurType' % renderGlobals)
    motionsample    = cmds.getAttr('%s.rman__torattr___motionSamples' % renderGlobals)

    shutterangle    = cmds.getAttr('%s.rman__toropt___shutterAngle' % renderGlobals)
    shutteropen     = cmds.getAttr('%s.rman__riopt__Camera_shutteropening0' % renderGlobals)
    shutterclose    = cmds.getAttr('%s.rman__riopt__Camera_shutteropening1' % renderGlobals )

    # FPS
    fps = 24
    tunit = cmds.currentUnit( time=True, q=True )
    fpsMap = {
        '2fps': 2,
        '3fps': 3,
        '4fps': 4,
        '5fps': 5,
        '6fps': 6,
        '8fps': 8,
        '10fps': 10,
        '12fps': 12,
        'game': 15,
        '16fps': 2,
        '20fps': 2,
        '23.976fps': 2,
        'film': 24,
        'pal' : 25,
        '29.97fps': 2,
        'ntsc': 30,
        '40fps': 30,
        '47.952fps': 30,
        'show': 48,
        'palf': 50,
        '59.94fps': 50,
        'ntscf' : 60,
        '75fps' : 70,
        '80fps' : 80,
        '100fps' : 100,
        '120fps' : 120,
        '125fps' : 125,
        '150fps' : 150,
        '200fps' : 200,
        '240fps' : 240,
        '250fps' : 250,
        '300fps' : 300,
        '375fps' : 375,
        '400fps' : 400,
        '500fps' : 500,
        '600fps' : 600,
        '750fps' : 750,
        '1200fps' : 1200,
        '1500fps' : 1500,
        '2000fps' : 2000,
        '3000fps' : 3000,
        '6000fps' : 6000,
        '44100fps' : 44100,
        '48000fps' : 48000,
    }    
    if fpsMap.has_key(tunit):
        fps = fpsMap[tunit]


    # Alembic Attributes
    vdt  = float('%.3f' % cmds.getAttr('%s.vdt' % node))
    fsdv = int(cmds.getAttr('%s.fsdv' % node))
    foid = int(cmds.getAttr('%s.foid' % node))
    fgid = int(cmds.getAttr('%s.fgid' % node))
    fcpv = int(cmds.getAttr('%s.fcpv' % node))
    ftxv = int(cmds.getAttr('%s.ftxv' % node))

    # render attribute
    groupId = cmds.getAttr('%s.groupId' % node)

    # Instance Attributes
    dln  = cmds.getAttr('%s.dln' % node)
    slod = cmds.getAttr('%s.slod' % node)

    xmin, ymin, zmin = cmds.getAttr('%s.boundingBoxMin' % node)[0]
    xmax, ymax, zmax = cmds.getAttr('%s.boundingBoxMax' % node)[0]
    bound = [xmin, xmax, ymin, ymax, zmin, zmax]

    mel.eval('RiAttributeBegin')
    mel.eval('RiAttribute "dice" "string instancestrategy" "worlddistance"')
    mel.eval('RiAttribute "instance" "int singlelod" %s' % slod)
    #mel.eval('RiReverseOrientation')

    procRi = ribCommon.Procedural2Archive( 'AssemblyRiProcedural' )
    procRi.addBound(bound)
    procRi.addArg( 'string', 'filename', afp)
    procRi.addArg( 'float', 'frame', cfr )
    procRi.addArg( 'float', 'fps', fps )
    procRi.addArg( 'int', 'subdiv', fsdv)
    procRi.addArg( 'int', 'oid', foid )
    procRi.addArg( 'int', 'gid', fgid )
    procRi.addArg( 'int', 'pid', groupId )
    procRi.addArg( 'int', 'primvar', fcpv )
    procRi.addArg( 'int', 'txvar', ftxv )
    if mbr:
        procRi.addArg( 'float', 'shutteropen', shutteropen)
        procRi.addArg( 'float', 'shutterclose', shutterclose)
        if mbrtype == 'subframe':
            procRi.addArg( 'int', 'subframe', 1 )
        if vdt > 0.0:
            procRi.addArg( 'float', 'dt', vdt)

    #print procRi.getRi();

    mel.eval( 'RiArchiveRecord("verbatim", "%s\\n")' % procRi.getRi() )
    mel.eval( 'RiAttributeEnd' )

