import os

from pxr import Usd, UsdGeom, Sdf

import SessionUtils
import dxsUsdUtils
import PathUtils


#-------------------------------------------------------------------------------
#
#   ASSET
#
#-------------------------------------------------------------------------------
def AssetPackage(showDir, assetName, sourceFile):
    # ASSET.USD
    assetFile = '{DIR}/asset/{NAME}/{NAME}.usd'.format(DIR=showDir, NAME=assetName)
    stage = SessionUtils.MakeInitialStage(assetFile)
    prim  = stage.DefinePrim('/' + assetName, 'Xform')
    stage.SetDefaultPrim(prim)

    timeInfo, defaultPath, firstRoot, primTypeName = dxsUsdUtils.GetMetaData(sourceFile)
    if timeInfo:
        if timeInfo[0] and timeInfo[1]:
            stage.SetStartTimeCode(timeInfo[0])
            stage.SetEndTimeCode(timeInfo[1])
            if timeInfo[2]:
                stage.SetFramesPerSecond(timeInfo[2])
    prim.GetReferences().AddReference(Sdf.Reference(PathUtils.GetRelPath(assetFile, sourceFile)))

    modelCheck = False
    items = prim.GetMetadata('references').prependedItems
    for i in range(len(items)):
        if items[i].assetPath.find('model') > -1:
            modelCheck = True
    if modelCheck:
        vset = prim.GetVariantSets().GetVariantSet('taskVariant')
        vset.SetVariantSelection('model')

    stage.GetRootLayer().Save()
    del stage

    # ASSET.LGT.USD
    assetLgtFile = '{DIR}/asset/{NAME}/{NAME}.lgt.usd'.format(DIR=showDir, NAME=assetName)
    outLayer = Sdf.Layer.FindOrOpen(assetFile)
    SessionUtils.InsertSubLayer(outLayer, '/assetlib/3D/material/usd/lights/Materials.usd')
    outLayer.Export(assetLgtFile, args={'format': 'usda'})
    del outLayer

    # ASSET.PAYLOAD.USD
    assetPayloadFile = '{DIR}/asset/{NAME}/{NAME}.payload.usd'.format(DIR=showDir, NAME=assetName)
    SessionUtils.MakeReferenceStage(assetPayloadFile, [(assetFile, None)], SdfPath='/asset{assetVariant=%s}' % assetName, clear=True)

    # asset.usd
    UpdateAsset(showDir, assetPayloadFile)

def UpdateAsset(showDir, sourceFile):
    assetFile = '{DIR}/asset/asset.usd'.format(DIR=showDir)
    outLayer  = Sdf.Layer.FindOrOpen(assetFile)
    if not outLayer:
        outLayer = Sdf.Layer.CreateNew(assetFile, args={'format': 'usda'})
        stage = Usd.Stage.Open(outLayer, load=Usd.Stage.LoadNone)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

    SessionUtils.InsertSubLayer(outLayer, PathUtils.GetRelPath(assetFile, sourceFile))
    outLayer.Save()
    del outLayer



#-------------------------------------------------------------------------------
#
#   SHOT
#
#-------------------------------------------------------------------------------
def getShotReferenceIndex(refPath):
    # fx, zenn, cloth, ani, set, cam
    if refPath.find('/fx.') > -1:
        return 1
    elif refPath.find('/zenn.') > -1:
        return 2
    elif refPath.find('/sim.') > -1:
        return 3
    elif refPath.find('/ani.') > -1:
        return 4
    elif refPath.find('/set.') > -1:
        return 5
    elif refPath.find('/cam.') > -1:
        return 6
    else:
        return 0

def ShotPackage(showDir, seqName, shotName, sourceFile, fr=(0, 0), fps=24.0):
    shotFile = '{DIR}/shot/{SEQ}/{SHOT}/{SHOT}.usd'.format(DIR=showDir, SEQ=seqName, SHOT=shotName)
    stage = SessionUtils.MakeInitialStage(shotFile, fr=fr, fps=fps)

    references = list()

    shotprim = stage.DefinePrim('/shot', 'Xform')
    stage.SetDefaultPrim(shotprim)
    dxsUsdUtils.SetModelAPI(shotprim, kind='group', name=shotName)

    # add sourceFile
    references.append(PathUtils.GetRelPath(shotFile, sourceFile))

    # current references
    currefs = shotprim.GetMetadata('references')
    if currefs:
        for i in currefs.prependedItems:
            references.append(i.assetPath)

    # reference re-order
    newrefs = [None] * 7
    extrs   = list()
    for r in references:
        index = getShotReferenceIndex(r)
        if index:
            newrefs[index] = r
        else:
            extrs.append(r)

    shotprim.GetReferences().ClearReferences()
    for r in extrs + newrefs:
        if r:
            shotprim.GetReferences().AddReference(Sdf.Reference(r))

    stage.GetRootLayer().Save()
    del stage

    # SHOT.LGT.USD
    shotLgtFile = shotFile.replace('.usd', '.lgt.usd')
    outLayer = Sdf.Layer.FindOrOpen(shotFile)
    SessionUtils.InsertSubLayer(outLayer, '/assetlib/3D/material/usd/lights/Materials.usd')
    outLayer.Export(shotLgtFile, args={'format': 'usda'})
    del outLayer

    # SHOT.PAYLOAD.USD
    shotpayloadFile = shotFile.replace('.usd', '.payload.usd')
    stage = SessionUtils.MakeInitialStage(shotpayloadFile, fr=fr, fps=fps)
    shotprim = stage.DefinePrim('/shot', 'Xform')
    stage.SetDefaultPrim(shotprim)
    dxsUsdUtils.SetModelAPI(shotprim, kind='group', name=shotName)
    # set frameRange
    start = stage.GetStartTimeCode()
    end   = stage.GetEndTimeCode()
    if (start and end) and (start != end):
        varset = dxsUsdUtils.VariantSelection(shotprim, 'shotVariant', shotName)
        with varset.GetVariantEditContext():
            relpath = './{SHOT}.usd'.format(SHOT=shotName)
            dxsUsdUtils.SetModelAPI(shotprim, name=shotName, identifier=relpath)
            shotprim.SetPayload(Sdf.Payload(relpath))
            itimeAttr = shotprim.CreateAttribute('userProperties:inTime', Sdf.ValueTypeNames.Int)
            itimeAttr.Set(start)
            otimeAttr = shotprim.CreateAttribute('userProperties:outTime', Sdf.ValueTypeNames.Int)
            otimeAttr.Set(end)
    stage.GetRootLayer().Save()
    del stage

    # update to shot.usd
    UpdateShot(showDir, shotpayloadFile)

def UpdateShot(showDir, sourceFile):
    outFile = '{DIR}/shot/shot.usd'.format(DIR=showDir)
    outLayer= SessionUtils.MakeInitialLayer(outFile)
    relpath = PathUtils.GetRelPath(outFile, sourceFile)
    if len(outLayer.subLayerPaths) == 0:
        stage = Usd.Stage.Open(outLayer, load=Usd.Stage.LoadNone)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    SessionUtils.InsertSubLayer(outLayer, relpath)
    outLayer.Save()



#-------------------------------------------------------------------------------
#
#   CleanUp
#
#-------------------------------------------------------------------------------
def CleanUpSubLayers(filename):
    outLayer = Sdf.Layer.FindOrOpen(filename)
    if not outLayer:
        return
    basePath = os.path.dirname(filename)
    for lyr in outLayer.subLayerPaths:
        full = os.path.abspath(os.path.join(basePath, lyr))
        if not os.path.exists(full):
            outLayer.subLayerPaths.remove(lyr)
    outLayer.Save()



#-------------------------------------------------------------------------------
#
#   Version Select
#
#-------------------------------------------------------------------------------
def VersionSelect(filename, primPath, variantName, variantValue):
    stage = Usd.Stage.Open(filename, load=Usd.Stage.LoadNone)
    prim  = stage.OverridePrim(primPath)
    if prim:
        vset = prim.GetVariantSets().GetVariantSet(variantName)
        vset.SetVariantSelection(variantValue)
    stage.GetRootLayer().Save()
    del stage
