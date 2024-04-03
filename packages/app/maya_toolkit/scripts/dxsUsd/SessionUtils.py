import os
import re
import string

from pxr import Usd, UsdGeom, UsdShade, UsdUtils, Sdf, Kind

import Arguments
import dxsUsdUtils
import PathUtils

_PurposeMap = {
    'render': UsdGeom.Tokens.render, 'proxy': UsdGeom.Tokens.proxy
}

def MakeInitialLayer(filename, usdformat='usda', clear=False, comment=None):
    outLayer = Sdf.Layer.FindOrOpen(filename)
    if not outLayer:
        outLayer = Sdf.Layer.CreateNew(filename, args={'format': usdformat})
    if clear:
        outLayer.Clear()
    if comment:
        curcomment = outLayer.comment
        if curcomment:
            if not comment in curcomment:
                comment = curcomment + ', ' + comment
        outLayer.comment = comment
    return outLayer


def MakeInitialStage(filename, usdformat='usda', clear=False, comment=None, fr=(None, None), fps=24.0):
    outLayer = MakeInitialLayer(filename, usdformat, clear, comment)
    stage = Usd.Stage.Open(outLayer)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    TimeCodeSetup(stage, fr, fps)
    return stage


# def InsertSubLayer(outLayer, addFile, index=0):
#     if outLayer.subLayerPaths.index(addFile) == -1:
#         outLayer.subLayerPaths.insert(index, addFile)
def InsertSubLayer(outLayer, addFile, index=0):
    layers = list()
    for lyr in outLayer.subLayerPaths:
        if lyr.startswith('/') or lyr.startswith('.'):
            if not lyr in layers:
                layers.append(lyr)
        else:
            nlyr = './' + lyr
            if not nlyr in layers:
                layers.append(nlyr)
    if not addFile in layers:
        layers.insert(index, addFile)
    outLayer.subLayerPaths.clear()
    for i in layers:
        outLayer.subLayerPaths.append(i)



def TimeCodeSetup(stage, fr=(None, None), fps=24.0):
    '''
    start is min value, end is max value
    '''
    if not (fr[0] != None and fr[1] != None):
        return
    cstart = stage.GetStartTimeCode()
    cstart = cstart if cstart else fr[0]
    cend   = stage.GetEndTimeCode()
    cend   = cend if cend else fr[1]
    # if (fr[0] and fr[1]) and (fr[0] != fr[1]):
    if fr[0] != fr[1]:
        start = fr[0] if fr[0] < cstart else cstart
        end   = fr[1] if fr[1] > cend else cend
        stage.SetStartTimeCode(start)
        stage.SetEndTimeCode(end)

        stage.SetFramesPerSecond(fps)
        stage.SetTimeCodesPerSecond(fps)

#-------------------------------------------------------------------------------
#
#   Pipe-Line Payload, Geom Package
#
#-------------------------------------------------------------------------------
class MakeSubLayerStage(Arguments.MakeArgs):
    '''
    Collect USD Files by subLayers
    '''
    def __init__(self, outputFile, sourceFiles, defaultPrimSet=True, **kwargs):
        Arguments.MakeArgs.__init__(self, **kwargs)
        self.outputFile  = outputFile
        self.sourceFiles = sourceFiles
        self.defaultPrimSet = defaultPrimSet
        self.doIt()

    def doIt(self):
        outLayer = MakeInitialLayer(self.outputFile, self.usdformat, self.clear, self.comment)

        startTimes = list(); endTimes = list(); framesPerSecond = None
        targetPath = None
        for f in self.sourceFiles:
            timeInfo, defaultPath, firstRoot, primTypeName = dxsUsdUtils.GetMetaData(f)
            if timeInfo:
                startTimes.append(timeInfo[0])
                endTimes.append(timeInfo[1])
                framesPerSecond = timeInfo[2]
            targetPath = defaultPath if defaultPath else firstRoot
            InsertSubLayer(outLayer, PathUtils.GetRelPath(self.outputFile, f))

        stage = Usd.Stage.Open(outLayer, load=Usd.Stage.LoadNone)
        if self.defaultPrimSet:
            stage.SetDefaultPrim(stage.GetPrimAtPath(targetPath))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

        # Frame
        if startTimes and not self.fr[0]:
            startTimes.sort(); endTimes.sort()
            self.fr = (startTimes[0], endTimes[-1])
            self.fps= framesPerSecond
        if self.fr[0] and self.fr[1]:
            stage.SetStartTimeCode(self.fr[0])
            stage.SetEndTimeCode(self.fr[1])
            if self.fps:
                stage.SetFramesPerSecond(self.fps)
                stage.SetTimeCodesPerSecond(self.fps)

        if self.SdfPath:
            ntyp = type(self.SdfPath).__name__
            if ntyp == 'str' or ntyp == 'unicode':
                self.SdfPath = Sdf.Path(self.SdfPath)
            self.makeOverridePrim(stage)

        outLayer.Save()

    def makeOverridePrim(self, stage):
        if self.SdfPath.ContainsPrimVariantSelection():
            name, value = self.SdfPath.GetVariantSelection()
            prim = stage.OverridePrim(self.SdfPath.StripAllVariantSelections())
            vset = prim.GetVariantSets().GetVariantSet(name)
            vset.SetVariantSelection(value)


class MakeReferenceStage(Arguments.MakeArgs):
    '''
    Args:
        outputFile  (str): output filename
        sourceData (list): [(src file, purpose), (src file, purpose), ...]
        composite   (str): payload or reference

        SdfPath (Sdf.Path): str, unicode change to Sdf.Path
        addChild    (bool): add (xform)? prim by archive source root
        Kind (str): defaultPrim Kind
        Name (str): defaultPrim AssetName
        Type (str): defaultPrim Geom Type
        customLayerData (dict): custom layer meta data
    '''
    def __init__(self, outputFile, sourceData, composite='payload', clearVariantSet=False, **kwargs):
        Arguments.MakeArgs.__init__(self, **kwargs)
        self.outputFile = outputFile
        self.sourceData = sourceData
        self.composite  = composite
        self.clearVariantSet = clearVariantSet

        if not self.SdfPath:
            assert False, '# Not defined SdfPath'
        ntyp = type(self.SdfPath).__name__
        if ntyp == 'str' or ntyp == 'unicode':
            self.SdfPath = Sdf.Path(self.SdfPath)

        self.mStartTimes = list(); self.mEndTimes = list(); self.mFramesPerSecond = None

        self.doIt()

    def doIt(self):
        stage = MakeInitialStage(self.outputFile, self.usdformat, self.clear, self.comment, self.fr, self.fps)
        outLayer = stage.GetRootLayer()
        stripPath = self.SdfPath.StripAllVariantSelections()

        # Clear current
        prim = stage.GetPrimAtPath(stripPath)
        if prim and self.clearVariantSet:
            _isClear = False
            if self.SdfPath.ContainsPrimVariantSelection():
                name, value = self.SdfPath.GetVariantSelection()
                if not prim.GetVariantSets().HasVariantSet(name):
                    _isClear = True
            else:
                if prim.GetVariantSets().GetNames():
                    _isClear = True
            if _isClear:
                edit = Sdf.BatchNamespaceEdit()
                edit.Add(stripPath.pathString, Sdf.Path.emptyPath)
                outLayer.Apply(edit)

        prefixes  = stripPath.GetPrefixes()
        for i in range(len(prefixes)):
            p = prefixes[i]
            # defaultPrim type set
            if i == 0 and self.Type != None:
                prim = stage.DefinePrim(p, self.Type)
            # last prim type set
            if i == len(prefixes)-1 and self.pType != None:
                prim = stage.DefinePrim(p, self.pType)

            prim = stage.GetPrimAtPath(p)
            if not prim:
                prim = UsdGeom.Xform.Define(stage, p)

        # Add outLayer customData
        if self.customLayerData:
            outLayer.customLayerData = self.customLayerData
        primSpec = outLayer.rootPrims[0]
        prim = stage.GetPrimAtPath(primSpec.path)
        dxsUsdUtils.SetModelAPI(prim, kind=self.Kind, name=self.Name)
        stage.SetDefaultPrim(prim)
        # Add defaultPrim customData
        if self.customPrimData:
            prim.SetCustomData(self.customPrimData)

        prim = stage.GetPrimAtPath(stripPath)
        if self.SdfPath.ContainsPrimVariantSelection():
            name, value = self.SdfPath.GetVariantSelection()
            vset = dxsUsdUtils.VariantSelection(prim, name, value)
            with vset.GetVariantEditContext():
                dxsUsdUtils.SetModelAPI(prim, name=value)
                self.compositeArchive(stage, prim)
        else:
            self.compositeArchive(stage, prim)

        # Frame
        self.SetFrameInfo(stage)

        outLayer.Save()

    def compositeArchive(self, stage, parent):
        stageFile = stage.GetRootLayer().identifier

        for filename, purpose in self.sourceData:
            if not filename:
                continue
            # source metadata
            timeInfo, defaultPath, firstRoot, primTypeName = dxsUsdUtils.GetMetaData(filename)
            firstPath = defaultPath if defaultPath else firstRoot
            if timeInfo:
                self.mStartTimes.append(timeInfo[0])
                self.mEndTimes.append(timeInfo[1])
                self.mFramesPerSecond = timeInfo[2]

            prim = parent
            if self.addChild and defaultPath:
                defaultName = stage.GetDefaultPrim().GetPath().name
                name = firstPath.name.split(defaultName + '_')[-1]
                if not name:
                    name = firstPath.name
                # prim = stage.DefinePrim(parent.GetPath().AppendChild(name), 'Xform')
                prim = stage.DefinePrim(parent.GetPath().AppendChild(name), primTypeName)

            if purpose:
                self.clearArchive(prim) # clear current archive
                prim = stage.DefinePrim(prim.GetPath().AppendChild(purpose), 'Xform')
                UsdGeom.Scope(prim).CreatePurposeAttr(_PurposeMap[purpose])
            else:
                children = prim.GetChildren()   # clear children
                if children:
                    self.clearChildren(stage.GetRootLayer(), children)

            if prim.GetTypeName() != primTypeName and primTypeName:
                prim = stage.DefinePrim(prim.GetPath().AppendChild(firstPath.name), primTypeName)

            targetPath = firstRoot if not defaultPath else None
            relpath = PathUtils.GetRelPath(self.outputFile, filename)
            if self.composite == 'payload':
                payload = Sdf.Payload(relpath) if not targetPath else Sdf.Payload(relpath, targetPath)
                dxsUsdUtils.SetPayload(prim, payload)
            else:
                reference = Sdf.Reference(relpath) if not targetPath else Sdf.Reference(relpath, targetPath)
                dxsUsdUtils.SetReference(prim, reference)


    def SetFrameInfo(self, stage):
        if self.mStartTimes and not self.fr[0]:
            self.mStartTimes.sort()
            self.mEndTimes.sort()
            stage.SetStartTimeCode(int(self.mStartTimes[0]))
            stage.SetEndTimeCode(int(self.mEndTimes[-1]))
            stage.SetFramesPerSecond(self.mFramesPerSecond)
            stage.SetTimeCodesPerSecond(self.mFramesPerSecond)


    def clearArchive(self, prim):
        payload = prim.GetMetadata('payload')
        if payload:
            prim.ClearPayload()


    def clearChildren(self, rootLayer, children):
        edit = Sdf.BatchNamespaceEdit()
        for c in children:
            edit.Add(c.GetPrimStack()[0].path.pathString, Sdf.Path.emptyPath)
        rootLayer.Apply(edit)
