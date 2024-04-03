#coding:utf-8
from __future__ import print_function


import os, re, string, gc, copy, types, json, glob
from pxr import Usd, UsdGeom, UsdShade, UsdUtils, Sdf, Kind
import os
import DXUSD.Message as msg
import DXUSD.Utils as utl
from DXUSD.Structures import Arguments
import DXUSD.Vars as var
import TUtils as tutl
# ------------------------------------------------------------------------------
# For ETC : Copy Utils
# ------------------------------------------------------------------------------

def jsonDump(jsonpath, data):
    with open(jsonpath, "w") as json_file:
        json.dump(data, json_file)


def jsonRead(jsonpath):
    with open(jsonpath, "r") as json_file:
        data = json.load(json_file)
        return data


def renameAsset(name):
    splitName = name.split('_')
    name = splitName[0]
    for i in splitName[1:]:
        if i[0].islower():
            for index, s in enumerate(i):
                if index == 0:
                    s = s.upper()
                name += s
        else:
            name += i
    return name

def GetBasePath(asset, branch=''):
    '''
    :param asset:
    :param branch:
    :return: txBasePath for Attribute
    '''
    basepath = "asset/%s/texture" %asset
    if branch:
        basepath = "asset/%s/branch/%s/texture" %(asset, branch)

    return basepath


def MakeDir(dstdir):
    if not os.path.exists(dstdir):
        if '/assetlib/' in dstdir:
            suCmd = 'echo dexter2019 | su render -c "mkdir -p %s"' % dstdir
            os.system(suCmd)
        else:
            os.system("mkdir -p %s" % dstdir)
        msg.debug('Make Directory : %s' % dstdir)


def CopyFile(source, dstdir):
    MakeDir(dstdir)
    target = os.path.join(dstdir,os.path.basename(source))
    if os.path.exists(source):
        if not os.path.exists(target):
            os.system("cp -rf %s %s" % (source, dstdir))
            msg.debug('Copy file : %s' % source, dstdir)


def AppendList(list, target):
    if os.path.exists(target):
        if not target in list:
            list.append(target)

# ------------------------------------------------------------------------------
# For ETC : Usd Query Utils
# ------------------------------------------------------------------------------

def SetModelVersion(spec, ver):
    spec.variantSelections.update({'modelVer': ver})
    utl.GetAttributeSpec(spec, 'primvars:modelVersion', ver, Sdf.ValueTypeNames.String,
                         info={'interpolation': 'constant'})


def GetUsdPath(setpath):
    layer = utl.AsLayer(setpath)
    pathList = []
    with utl.OpenStage(layer) as stage:
        dprim = stage.GetDefaultPrim()
        for p in dprim.GetChildren():
            for c in p.GetChildren():
                if c.GetName() == 'scatter':
                    prototypes = utl.UsdGeom.PointInstancer(c).GetPrototypesRel().GetTargets()
                    for i in range(len(prototypes)):
                        prim = stage.GetPrimAtPath(prototypes[i])
                        refs = prim.GetMetadata('references')
                        if refs:
                            filePath = refs.GetAddedOrExplicitItems()[0].assetPath
                            if '/mach/' in filePath:
                                filePath = filePath.split('/mach')[-1]
                            pathList.append(filePath)
                else:
                    refs = c.GetMetadata('references')
                    if refs:
                        filePath = refs.GetAddedOrExplicitItems()[0].assetPath
                        pathList.append(filePath)
    return pathList




# ------------------------------------------------------------------------------
# Common
# ------------------------------------------------------------------------------
def GetTaskDir(path, task):
    '''
    :param path: '/show/prat2/_3d/asset/agwi/agwi.usd'
    :param task:  var.T.RIG #'rig'
    :return: '/show/prat2/_3d/asset/agwi/rig/rig.usd'
    '''

    lyr = utl.AsLayer(path)
    spec = lyr.rootPrims[0]

    if not spec.variantSets['task']:
        return

    for var in spec.variantSets['task'].variantList:
        if var.name == task:
            s = var.primSpec
            fullPath = GetRefFullPath(s)
            if fullPath:
                return fullPath



def GetVerDir(taskmaster, taskVer):
    '''
    :param path: '/show/prat2/_3d/asset/agwi/rig/rig.usd'
    :param taskVer: var.T.VAR_RIGVER #'rigVer'
    :return: '/show/prat2/_3d/asset/agwi/rig/agwi_rig_v001/agwi_rig.usd'
    '''
    if not taskmaster:
        return

    nslyr =''
    stage = Usd.Stage.Open(taskmaster)
    dPrim = stage.GetDefaultPrim()
    var = dPrim.GetVariantSets().GetAllVariantSelections()
    if var.has_key(taskVer):
        nslyr = var[taskVer]

    if nslyr:
        lyr = utl.AsLayer(taskmaster)
        for spec in lyr.rootPrims:
            vset = spec.variantSets.get(taskVer)
            s = vset.variants[nslyr]
            s = s.primSpec
            fullPath = GetRefFullPath(s)

            if fullPath:
                return fullPath

def GetRefFullPath(spec):
    identifier = spec.layer.identifier
    assetPath = spec.referenceList.prependedItems[0].assetPath
    if assetPath:
        fullPath = os.path.abspath(os.path.join(utl.DirName(identifier), assetPath))
        return fullPath


# def GetVarVersion(taskusd, vtype):
#     stage = Usd.Stage.Open(taskusd)
#     dPrim = stage.GetDefaultPrim()
#     vsets = dPrim.GetVariantSets().GetAllVariantSelections()
#     if vsets.has_key(vtype):
#         value = vsets[vtype]
#         return value





# ------------------------------------------------------------------------------
# Groom
# ------------------------------------------------------------------------------

def GetGroomSceneFile(path):
    import DXUSD.Vars as var
    groomVer = ''
    taskMaster = GetTaskDir(path, var.T.GROOM)

    stage = Usd.Stage.Open(taskMaster)
    dPrim = stage.GetDefaultPrim()
    nslyr = dPrim.GetVariantSet('rigVer').GetVariantSelection()

    ##
    stack = dPrim.GetPrimStack()
    primSpec = stack[0]
    vsetSpec = primSpec.variantSets.get('rigVer')
    data = vsetSpec.variants
    if data.has_key(nslyr):
        vspec = data[nslyr]
        groomVer = vspec.GetVariantNames('groomVer')[0]

    mayafile = os.path.join(utl.DirName(taskMaster), 'scenes', '%s.mb' % groomVer)

    return mayafile

# ------------------------------------------------------------------------------
# Rig
# ------------------------------------------------------------------------------

def GetRigSceneFile(path):
    import DXUSD.Vars as var
    nslyr = ''
    taskMaster = GetTaskDir(path,var.T.RIG) # Result: /show/prat2/_3d/asset/agwi/rig/rig.usd #
    print('taskMaster:',taskMaster)

    if not taskMaster:
        return
    arg = Arguments()
    arg.D.SetDecode(taskMaster)

    stage = Usd.Stage.Open(taskMaster)
    dPrim = stage.GetDefaultPrim()
    var = dPrim.GetVariantSets().GetAllVariantSelections()
    if var.has_key('rigVer'):
        nslyr = var['rigVer']

    arg = Arguments()
    arg.D.SetDecode(taskMaster)
    assetname = arg.asset
    if arg.branch:
        assetname = arg.branch
    nslyrMaster = os.path.join(utl.DirName(taskMaster), nslyr, '%s_rig.usd' % assetname)
    layer = Sdf.Layer.FindOrOpen(nslyrMaster)
    customData = layer.customLayerData

    # if branch rig
    try:
        mayafile = os.path.join(utl.DirName(taskMaster), 'scenes', '%s.mb' % nslyr)  # _3d pub dir
    except:
        mayafile = customData['sceneFile']  # works pub dir
    print('mayafile :rig :',mayafile)
    return mayafile


# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------
def GetModelDir(path):
    print('>>>path:',path)
    taskdir = GetTaskDir(path, var.T.MODEL)
    verusd = GetVerDir(taskdir, var.T.VAR_MODELVER)
    verdir = utl.DirName(verusd)
    return verdir

def GetMaster(path, task, taskProduct, ver=None, nslyr=None, nsver=None):
    arg = Arguments()
    arg.D.SetDecode(path)
    arg.task = task
    arg.taskProduct = taskProduct
    if ver:
        arg.ver = ver
    if nslyr:
        arg.nslyr = nslyr
    if nsver:
        arg.nsver = nsver

    if ver or nsver:
        master = os.path.join(arg.D[arg.taskProduct], arg.F.MASTER)
    else:
        master = os.path.join(arg.D.TASK, arg.F.TASK)
    return master


# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------
def GetLidarDir(path):
    print('>>>path:',path)
    taskdir = GetTaskDir(path, var.T.LIDAR)
    verusd = GetVerDir(taskdir, var.T.VAR_LIDARVER)
    print('>>>verusd:', verusd)
    verdir = utl.DirName(verusd)
    return verdir


# ------------------------------------------------------------------------------
# Shot : Layout
# ------------------------------------------------------------------------------

def GetLayoutMaster(usdpath):
    # usdpath = '/show/slc/_3d/shot/HEA/HEA_0020/HEA_0020.usd'
    lyr = Sdf.Layer.FindOrOpen(usdpath)
    spec = lyr.rootPrims[0]

    for sublayer in lyr.subLayerPaths:
        if '/layout/' in sublayer:
            identifier = spec.layer.identifier
            fullPath = os.path.abspath(os.path.join(utl.DirName(identifier), sublayer))
            return fullPath


# ------------------------------------------------------------------------------
# Shot : Ani Reference
# ------------------------------------------------------------------------------
def getCustomData(key, path, pathList):
    # path = /show/slc/_3d/shot/DAG/DAG_0350/ani/carBrokenSA/v002/carBrokenSA_ani.usd
    layer = Sdf.Layer.FindOrOpen(path)
    customData = layer.customLayerData
    file = ''

    if key == 'rigFile':
        rigFile = customData['rigFile']

        arg = Arguments()
        arg.D.SetDecode(rigFile)
        usdpath = os.path.join(arg.D.ASSET, arg.F.ASSET)
        file = usdpath

    if key == 'sceneFile':
        sceneFile = customData['sceneFile']
        file = sceneFile

    if not file in pathList:
        pathList.append(file)


def getRigFile(path, pathList, sceneFileList):
    # path = '/show/slc/_3d/shot/DAG/DAG_0350/ani/carBrokenSA/carBrokenSA.usd'
    layer = utl.AsLayer(path)
    with utl.OpenStage(layer) as stage:
        dprim = stage.GetDefaultPrim()
        for p in dprim.GetChildren():
            if p.GetName() == 'Rig':
                for c in p.GetChildren():
                    stack = c.GetPrimStack()
                    try:
                        spec = stack[1]
                        id = spec.layer.identifier
                        assetPath = spec.referenceList.prependedItems[0].assetPath
                        fullPath = os.path.abspath(os.path.join(utl.DirName(id), assetPath))
                        # print fullPath
                    except: # groom task
                        spec = stack[2]
                        id = spec.layer.identifier
                        assetPath = spec.referenceList.prependedItems[0].assetPath
                        fullPath = os.path.abspath(os.path.join(utl.DirName(id), assetPath))

                    # getCustomData('rigFile', fullPath, rigFileList)
                    getCustomData('rigFile', fullPath, pathList)
                    getCustomData('sceneFile', fullPath, sceneFileList)


def GetAniReferenceList(usdpath):
    lyr = Sdf.Layer.FindOrOpen(usdpath)
    pathList = []
    sceneFileList = []

    with utl.OpenStage(lyr) as stage:
        dprim = stage.GetDefaultPrim()
        for p in dprim.GetChildren():
            if p.GetName() == 'Rig':
                for c in p.GetChildren():
                    stack = c.GetPrimStack()
                    spec = stack[1]
                    identifier = spec.layer.identifier
                    if 'groom' in identifier:
                        stack = c.GetPrimStack()
                        spec = stack[2]
                        identifier = spec.layer.identifier
                    getRigFile(identifier, pathList, sceneFileList )

    return pathList, sceneFileList

# ------------------------------------------------------------------------------
# Branch
# ------------------------------------------------------------------------------
def getbranchlist(usdpath):
    pathlist = []
    taskMaster =GetTaskDir(usdpath, 'branch')
    if not os.path.exists(taskMaster):
        return []

    stage = Usd.Stage.Open(taskMaster)
    dPrim = stage.GetDefaultPrim()

    ##
    stack = dPrim.GetPrimStack()
    primSpec = stack[0]
    vsetSpec = primSpec.variantSets.get('branch')
    data = vsetSpec.variants
    for k, v in data.items():
        branchpath = os.path.join(os.path.dirname(usdpath), 'branch', k, '%s.usd' % k)
        if os.path.exists(branchpath):
            pathlist.append(branchpath)
            # rigtaskpath = GetTaskDir(branchpath, 'rig')
            # if rigtaskpath:
            #     if os.path.exists(rigtaskpath):
            #         pathlist.append(branchpath)
    return pathlist


def istask(usdpath, task=''):
    from pxr import Usd
    isTask = False
    stage = Usd.Stage.Open(usdpath)
    dPrim = stage.GetDefaultPrim()
    if task in dPrim.GetVariantSet('task').GetVariantNames():
        isTask = True
    return isTask


def GetAgentMayafile(usdpath):
    taskMaster = GetTaskDir(usdpath, 'agent')  # /show/emd/_3d/asset/car/agent/agent.usd
    layer = utl.AsLayer(taskMaster)
    arg = Arguments()
    arg.D.SetDecode(taskMaster)

    with utl.OpenStage(layer) as stage:
        dprim = stage.GetDefaultPrim()
        nslyr = dprim.GetVariantSet('agent').GetVariantSelection()
        stack = dprim.GetPrimStack()
        spec = stack[4]
        identifier = spec.layer.identifier
        dir = os.path.dirname(identifier)
        mbpath = os.path.join(dir, 'character', '%s.mb' % arg.asset)
        if os.path.exists(mbpath):
            return mbpath
        else:
            return

def GetReferenceList(usdpath):
    reflist = []
    stage = Usd.Stage.Open(usdpath)
    dprim = stage.GetDefaultPrim()
    tutl.walk(stage, dprim, reflist, usdpath)
    return reflist
