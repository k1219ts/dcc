#coding:utf-8
from __future__ import print_function

from pxr import Usd, Sdf

import DXUSD.Message as msg
import DXUSD_HOU.Utils as utl
import DXUSD_HOU.Vars as var
import DXRulebook.Interface as rb

if msg.DEV:
    rb.Reload()


def ReloadStage(usdpath):
    lyr = utl.AsLayer(usdpath)
    if lyr:
        with utl.OpenStage(lyr) as stg:
            stg.Reload()


def IsShot(usdpath):
    flags = rb.Flags()
    try:
        flags.D.SetDecode(utl.DirName(usdpath))
        return flags.IsShot()
    except Exception as e:
        msg.debug(e)
        return False


def GetShotChildren(usdpath, prim=None, hasVSet=False, vsel={var.ORDER:[]}):
    '''
    vsel = {
        var.ORDER:[vset1, vset2, ...],
        vset1:[sel],
        vset2:[sel], ...
    }
    '''
    lyr = utl.AsLayer(usdpath)
    if not lyr:
        return []

    with utl.OpenStage(lyr) as stg:
        target = stg.GetDefaultPrim()
        if prim:
            target = stg.GetPrimAtPath(target.GetPath().AppendChild(prim))

        for k in vsel[var.ORDER]:
            vset = target.GetVariantSet(k)
            vset.SetVariantSelection(vsel[k][0])

        res = []
        for child in target.GetChildren():
            if not hasVSet or child.HasVariantSets():
                res.append(child.GetName())
        return res


def GetVariantPrim(usdpath, primpath=None):
    lyr = utl.AsLayer(usdpath)
    if not lyr:
        return ''

    with utl.OpenStage(lyr) as stg:
        # auto find prim that has variantSet
        prims = [stg.GetPrimAtPath(primpath or '/')]
        while prims and not prims[0].HasVariantSets():
            prims.extend(prims.pop(0).GetChildren())

        return prims[0].GetPath().pathString if prims else ''


def GetAssetLayer(usdpath, primpath):
    lyr = utl.AsLayer(usdpath)
    if not lyr:
        return '', '', ''

    assetpath  = ''
    assetnslyr = ''
    assettask  = ''
    with utl.OpenStage(lyr) as stg:
        prim = stg.GetPrimAtPath(primpath)
        if not prim:
            return '', '', ''

        stacks = prim.GetPrimStack()
        stacks.reverse()
        for stack in stacks:
            # ani(sim) > RIGFILE, groom > GROOMFILE, feather > FEATHERFILE
            filenames = {var.T.RIG:var.T.RIGFILE,
                         var.T.GROOM:var.T.GROOMFILE,
                         var.T.FEATHER:var.T.FEATHERFILE}
            for task, key in filenames.items():
                if stack.layer.customLayerData.has_key(key):
                    filename = stack.layer.customLayerData[key]
                    assetnslyr = utl.BaseName(filename).split('.')[0]
                    assetargs  = var.D.KINDS.Decode(utl.DirName(filename))

                    assetusd   = var.F.ABNAME.Encode(**assetargs)
                    assetpath  = var.D.KINDS.Encode(**assetargs)
                    assetpath  = utl.SJoin(assetpath, assetusd)
                    assettask  = task
                    break

    return assetpath, assettask, assetnslyr


def GetPrimVariants(usdpath, primpath=None, vsels={var.ORDER:[]}):
    lyr = utl.AsLayer(usdpath)
    vsets = {var.ORDER:[]}
    # vsets = {
    #     setname1  : [selected, v1, v2, v3...], ...
    #     __order__ : [setname1, setname2, ...]
    # }

    if not lyr:
        return vsets

    updated = False
    with utl.OpenStage(lyr, True) as stg:
        trglyr = stg.GetEditTarget().GetLayer()
        trglyr.SetPermissionToEdit(True)

        if not primpath:
            dprim = stg.GetDefaultPrim()
        else:
            dprim = stg.GetPrimAtPath(primpath)

        # set variant selections
        for name in vsels[var.ORDER]:
            if name in dprim.GetVariantSets().GetNames():
                vset = dprim.GetVariantSet(name)
                vset.SetVariantSelection(vsels[name])

        # get changed variant sets
        vsetlist = dprim.GetVariantSets()
        for name in vsetlist.GetNames():
            vset = vsetlist.GetVariantSet(name)
            vars = vset.GetVariantNames()
            if vsels.has_key(name):
                sel = vsels[name]
            else:
                sel  = vset.GetVariantSelection()
                if not sel in vars:
                    sel = vars[-1]
                    vsels[name] = sel
                    vsels[var.ORDER].append(name)
                    updated = True

            vars.insert(0, sel)
            vsets[name] = vars
            vsets[var.ORDER].append(name)

    if updated:
        return GetPrimVariants(usdpath, primpath, vsels)
    else:
        return vsets


def GetChildren(usdpath, primpath):
    res = []
    lyr = utl.AsLayer(usdpath)
    if not lyr:
        return res

    with utl.OpenStage(lyr, True) as stg:
        prim = stg.GetPrimAtPath(primpath)
        if not prim:
            return res

        for child in prim.GetChildren():
            res.append(child.GetPath().pathString)

    return res



#
