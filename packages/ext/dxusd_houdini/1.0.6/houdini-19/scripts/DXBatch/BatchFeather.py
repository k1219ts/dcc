#coding:utf-8
from __future__ import print_function

import hou, os, sys
from enum import Enum, unique

from pxr import Sdf

import DXRulebook.Interface as rb
import DXUSD_HOU.Utils as utl
import DXUSD_HOU.Vars as var
import DXUSD_HOU.Message as msg


@unique
class eImportType(Enum):
    Designer = 0
    Groomer = 1
    Wings = 2
    Deformer = 3


def SetVariant(node, kind, i, v):
    parm = '%svariants%d'%(kind, i)
    node.parm(parm).set(v)
    kwargs = {
        'node':node,
        'script_parm':parm,
        'script_multiparm_index':str(i)
    }
    node.hdaModule().UI_LoadVariants(kwargs, True)


def FeatherDefomerSceneSetup(inputCache, groomFile,
                             rootBlendPower=0.2, fr=(0, 0), step=1.0,
                             primPattern=None):
    _name = 'FeatherDefomerSceneSetup'
    ropNode  = '{SHOW}_{SEQ}_{SHOT}_ROP'

    # --------------------------------------------------------------------------
    # resolve inputCache
    sflg = rb.Flags(pub='_3d', dcc='USD')
    aflg = rb.Flags(pub='_3d', dcc='USD')

    sflg.D.SetDecode(utl.DirName(inputCache))

    srclyr = utl.AsLayer(inputCache)
    if not srclyr:
        msg.errmsg('Input cache does not exist (%s)'%inputCache)
        sys.exit(1)

    # find feather variants
    if sflg.task == var.T.CROWD:
        if not primPattern or '/' not in primPattern:
            msg.errmsg(e)
            m = 'If crowd groom, must have primPattern '
            m += '(Eg. /Crowd/cacheProxyShape1/Agent_2)'
            msg.errmsg(m)
            sys.exit(1)

        sflg.nslyr = primPattern.split('/')[-1]
        aflg.D.SetDecode(utl.DirName(groomFile))
        aflg.pop('nslyr')
        groomTaskFile = utl.SJoin(aflg.D.TASK, aflg.F.TASK)
        groomTaskLyr = utl.AsLayer(groomTaskFile)
        if not groomTaskLyr:
            msg.errmsg('Cannot find groom task layer(%s)'%groomTaskFile)

        prim = groomTaskLyr.GetPrimAtPath('/%s'%groomTaskLyr.defaultPrim)
        aflg.nslyr = prim.variantSelections[var.T.VAR_RIGVER]

    else:
        rigfile = srclyr.customLayerData.get(var.T.CUS_RIGFILE)

        if not rigfile or utl.NotExist(rigfile):
            msg.errmsg('@%s :'%_name, 'Cannot find rig file path')
            sys.exit(1)

        aflg.D.SetDecode(utl.DirName(rigfile))
        if rigfile.endswith('.mb'):
            aflg.F.MAYA.rig.SetDecode(utl.BaseName(rigfile), 'WORK')
            aflg.nslyr = aflg.N.RIGVER
            aflg.show = sflg.show
        else:
            aflg.D.SetDecode(utl.DirName(rigfile))

    if sflg.IsShot():
        shotusd = utl.SJoin(sflg.D.SHOT, sflg.F.SHOT)
    else:
        shotusd = utl.SJoin(sflg.D.ASSET, sflg.F.ASSET)

    # --------------------------------------------------------------------------
    # create network nodes
    geomNode = '%s_%s_%s_GEOM'%(sflg.show, sflg.seq, sflg.shot)
    geomNode = hou.node("/obj").createNode("geo", geomNode)

    ropNode = '%s_ROP'%sflg.show
    ropNode = hou.node("/obj").createNode("ropnet", ropNode)

    # --------------------------------------------------------------------------
    # SOP - shot import node
    shotNode = 'import_shot'
    shotNode = geomNode.createNode('dxusdSOP_import', shotNode)
    kwargs = {'node':shotNode}

    # load shot usd
    shotNode.parm("usdpath").set(shotusd)
    shotNode.hdaModule().UI_LoadUSD(kwargs)

    if sflg.task == var.T.CROWD:
        crowdgrp = primPattern.split('/')[1]

        shotNode.parm('shotcategory').set(crowdgrp)
        shotNode.hdaModule().UI_LoadShotNamespace(kwargs)
        shotNode.parm('shotnamespace').set(crowdgrp)
        shotNode.hdaModule().UI_LoadPrim(kwargs)
        shotNode.parm('assignedtask').set(sflg.task)
        shotNode.hdaModule().UI_LoadTaskUsd(kwargs)
    else:
        # todo : edit params for clip
        # specify catetor, nslyr, task
        shotNode.parm('shotcategory').set('Rig')
        shotNode.hdaModule().UI_LoadShotNamespace(kwargs)
        shotNode.parm('shotnamespace').set(sflg.nslyr)
        shotNode.hdaModule().UI_LoadPrim(kwargs)
        shotNode.parm('assignedtask').set(sflg.task)
        shotNode.hdaModule().UI_LoadTaskUsd(kwargs)

    if primPattern:
        if not primPattern.startswith('/World'):
            primPattern = '/World%s'%primPattern
        shotNode.parm('primpattern').set(primPattern)

    # set task version
    for i in range(shotNode.parm('shotvariantfolder').evalAsInt()):
        vset = shotNode.parm('shotvariantset%d'%i).evalAsString()
        if vset == sflg.N.TASKVAR:
            SetVariant(shotNode, var.T.SHOT, i, sflg.nsver)
            break
    # set other params
    shotNode.parm('purpose').set('render')
    shotNode.parm('applyworldxform').set(False)

    # --------------------------------------------------------------------------
    # SOP - groom import node
    assetusd = utl.SJoin(aflg.D.ASSET, aflg.F.ASSET)
    groomNode = 'import_groom'

    groomNode = geomNode.createNode('dxusdSOP_import', groomNode)
    kwargs = {'node':groomNode}

    # load shot usd
    groomNode.parm("usdpath").set(assetusd)
    groomNode.hdaModule().UI_LoadUSD(kwargs)

    kind = var.T.ASSET if aflg.IsAsset() else var.T.BRANCH

    # set task
    for i in range(groomNode.parm('%svariantfolder'%kind).evalAsInt()):
        vset = groomNode.parm('%svariantset%d'%(kind, i)).evalAsString()
        if vset == var.T.TASK:
            SetVariant(groomNode, kind, i, var.T.GROOM)
            break
    # set rigver
    for i in range(groomNode.parm('%svariantfolder'%kind).evalAsInt()):
        vset = groomNode.parm('%svariantset%d'%(kind, i)).evalAsString()
        if vset == var.T.VAR_RIGVER:
            rigvers = groomNode.userData('%svariantdata%d'%(kind, i))
            rigvers = rigvers.split(' ')
            if aflg.nslyr in rigvers:
                rigver = aflg.nslyr
            else:
                flgas = {'asset':aflg.asset, 'ver':utl.Ver(0)}
                rigver = var.N.rig.RIGVER.Encode(**flgas)
            SetVariant(groomNode, kind, i, rigver)
            break
    # set other params
    groomNode.parm('purpose').set('render guide')

    # --------------------------------------------------------------------------
    # SOP - attach node
    attachNode = geomNode.createNode('dxSOP_FeatherAttach')

    attachNode.setInput(0, groomNode)
    attachNode.setInput(1, shotNode)
    attachNode.parm('gamma').set(rootBlendPower)


    # --------------------------------------------------------------------------
    # ROP
    # create layer node
    layerNode = ropNode.createNode("dxusdROP_layer")
    layerNode.parm("lyrtype").set("feather")
    layerNode.parm("feather_exporttype").set(eImportType.Deformer.value)
    layerNode.parm("feather_objpath").set(geomNode.path())
    layerNode.parm("feather_deformerpath").set(attachNode.path())
    layerNode.parm("feather_reload").pressButton()
    # layerNode.parm("mergeseqlayers").set(False)

    # create publish node
    pubNode = ropNode.createNode("dxusdROP_publish")
    pubNode.setInput(0, layerNode)

    pubNode.parm("trange").set(1)
    pubNode.parm("f1").deleteAllKeyframes()
    pubNode.parm("f1").set(fr[0])

    pubNode.parm("f2").deleteAllKeyframes()
    pubNode.parm("f2").set(fr[1])

    pubNode.parm("f3").deleteAllKeyframes()
    pubNode.parm("f3").set(step)

    pubNode.parm("show").set(sflg.show)
    pubNode.parm("seq").set(sflg.seq)
    pubNode.parm("shot").set(sflg.shot)
    pubNode.parm("shot").pressButton()

    layerNode.hdaModule().UI_UpdateInputs({'node':layerNode})

    if sflg.task == var.T.CROWD:
        layerNode.parm('nslyrtgl').set(True)
        layerNode.parm('nslyr').set(primPattern.split('/')[-1])

        layerNode.hdaModule().UI_UpdateOutputs({'node':layerNode})

    if msg.DEV:
        if sflg.IsShot():
            f = '%s_%s_%s_%s.hip'%(sflg.show, sflg.seq, sflg.shot, sflg.nslyr)
        else:
            f = '%s_%s_%s.hip'%(sflg.show, sflg.ABName(), sflg.task)
        hou.hipFile.save(utl.SJoin('/tmp/houBatch', f))

    pubNode.render(verbose=True, output_progress=True)
