import os
import re
import glob
import string

from pxr import Sdf, Usd, UsdGeom, UsdUtils

import dxsMsg
import PathUtils
import SessionUtils
import dxsMayaUtils
import dxsUsdUtils


def GetPerFrameFiles(ruleStr, fr):
    '''
    Args
        ruleStr (str): find file rule. /path/filename.*.usd
        fr (tuple): (start, end)
    '''
    source = glob.glob(ruleStr)
    source.sort()

    start_digit = len(str(int(fr[0])))
    end_digit   = len(str(int(fr[1])))
    if start_digit != end_digit or start_digit > 4:
        for f in source:
            frame = re.findall(r'\.(\d+)?\.', f)
            if frame and len(frame[0]) < end_digit:
                splitFile = f.split('.')
                splitFile[-2] = splitFile[-2].zfill(end_digit)
                newFileName   = '.'.join(splitFile)
                os.rename(f, newFileName)

    source = glob.glob(ruleStr)
    source.sort()

    frames = list()
    files  = list()
    for f in source:
        frame = re.findall(r'\.(\d+)?\.', f)
        if frame:
            iframe = int(frame[0])
            if not iframe in frames:
                frames.append(iframe)
                if fr[0] <= iframe <= fr[1]:
                    files.append(f)
    return files


def ActualExtentCompute(filename):
    stage = Usd.Stage.Open(filename)
    startTime = stage.GetStartTimeCode()
    endTime   = stage.GetEndTimeCode()
    if startTime and endTime and startTime == endTime:
        dprim = stage.GetDefaultPrim()
        for p in iter(Usd.PrimRange.AllPrims(dprim)):
            if p.GetTypeName() == 'Mesh':
                geom = UsdGeom.Mesh(p)
                extAttr = geom.GetExtentAttr()
                if not extAttr.GetTimeSamples():
                    # getVal = extAttr.Get()
                    # print '->', getVal
                    extAttr.Clear()
                    getVal = UsdGeom.Boundable.ComputeExtentFromPlugins(geom, Usd.TimeCode(startTime))
                    extAttr.Set(getVal, Usd.TimeCode(startTime))
    stage.GetRootLayer().Save()


def InsertInitFile(stage, refFile):
    frame = re.findall(r'\.(\d+)?\.', refFile)
    if frame:
        digit = len(frame[0])
        splitFile = refFile.split('.')
        splitFile[-2] = '0'.zfill(digit)
        initFile = '.'.join(splitFile)
        if os.path.exists(initFile):
            outLayer = stage.GetRootLayer()
            relpath  = PathUtils.GetRelPath(outLayer.identifier, initFile)
            SessionUtils.InsertSubLayer(outLayer, relpath)
            outLayer.Save()




#-------------------------------------------------------------------------------
#
#   Coalesce per frame files
#
#-------------------------------------------------------------------------------
def CoalesceFiles(frameFiles, frameRange, step=1, outFile=None, activeOffset=0.0, clipSet='default'):
    if outFile:
        topologyFile = outFile.replace('.usd', '.topology.usd')
    else:
        fileSplit = frameFiles[0].split('.')
        fileSplit[-2] = 'valueclip'
        outFile = '.'.join(fileSplit)
        topologyFile = outFile.replace('.valueclip.', '.topology.')

    topologyLayer= SessionUtils.MakeInitialLayer(topologyFile, usdformat='usdc', clear=True)
    UsdUtils.StitchClipsTopology(topologyLayer, frameFiles)

    clipPath = topologyLayer.rootPrims[0].path

    outLayer = SessionUtils.MakeInitialLayer(outFile, clear=True)
    stride = 1
    baseName = os.path.basename(frameFiles[0])
    for i in range(2):
        m = re.search('(^|\.)([0-9]+)\.', baseName)
        if m:
            fstr = m.groups()[1]
            baseName = baseName.replace(fstr, '#' * len(fstr))
            if i == 1:
                stride = step
    templatePath = './' + baseName

    UsdUtils.StitchClipsTemplate(
        outLayer, topologyLayer, clipPath, templatePath, frameRange[0], frameRange[1], stride, activeOffset, clipSet
    )
    stage = Usd.Stage.Open(outLayer, load=Usd.Stage.LoadNone)
    prim  = stage.GetPrimAtPath(clipPath)
    stage.SetDefaultPrim(prim)
    InsertInitFile(stage, frameFiles[0])
    outLayer.Save()
    return outFile, topologyFile




#-------------------------------------------------------------------------------
#
#   stitch frames files by valueclip
#
#-------------------------------------------------------------------------------
class StitchFiles:
    '''
    Args
        modify (bool): modify timeSamples and primvar cleanup
    '''
    def __init__(self, inputFiles, outFile=None, modify=True, clipSet='default'):
        self.inputFiles = inputFiles
        self.outFile = outFile
        if not outFile:
            splitFile = inputFiles[0].split('.')
            del splitFile[-2]
            self.outFile = '.'.join(splitFile)
        self.modify = modify
        self.clipSet= clipSet

        self.topologyFile = self.outFile.replace('.usd', '.topology.usd')

        # Member variables
        self._deletePrimvars = ['primvars:st']


    def doIt(self):
        if len(self.inputFiles) > 1:
            if self.modify:
                self.timeSampleModify(self.inputFiles[-1])

            self.MakeTopologyFile(self.inputFiles[-1])

            stage = self.MakeClipFile()
            InsertInitFile(stage, self.inputFiles[0])

            if self.modify:
                # Delete Attributes
                for f in self.inputFiles:
                    self.CleanUp(f)
        else:
            refLayer = Sdf.Layer.FindOrOpen(self.inputFiles[0])
            fr = (refLayer.startTimeCode, refLayer.endTimeCode)
            stage = SessionUtils.MakeInitialStage(self.outFile, fr=fr, clear=True)
            SessionUtils.InsertSubLayer(stage.GetRootLayer(), PathUtils.GetRelPath(self.outFile, self.inputFiles[0]))
            InsertInitFile(stage, self.inputFiles[0])
            root = stage.GetPrimAtPath('/')
            prims= root.GetChildren()
            stage.SetDefaultPrim(prims[0])
            stage.GetRootLayer().Save()

        return self.outFile


    def GetInfo(self):
        times = list()
        dpath = None
        for f in self.inputFiles:
            layer = Sdf.Layer.FindOrOpen(f)
            start = layer.startTimeCode
            end   = layer.endTimeCode
            times.append((start, end))
            dpath = layer.defaultPrim
        return dpath, times


    def MakeTopologyFile(self, filename):
        topologyLayer= SessionUtils.MakeInitialLayer(self.topologyFile, usdformat='usdc', clear=True)
        UsdUtils.StitchClipsTopology(topologyLayer, [filename])


    def MakeClipFile(self):
        dpath, times = self.GetInfo()
        defaultPath = '/' + dpath

        stage = SessionUtils.MakeInitialStage(self.outFile, clear=True)
        SessionUtils.InsertSubLayer(stage.GetRootLayer(), './' + os.path.basename(self.topologyFile))

        prim = stage.OverridePrim(defaultPath)

        clipObj = Usd.ClipsAPI(prim)
        assetPaths = list()
        for f in self.inputFiles:
            relpath = PathUtils.GetRelPath(self.outFile, f)
            assetPaths.append(Sdf.AssetPath(relpath))
        clipObj.SetClipAssetPaths(assetPaths, self.clipSet)
        clipObj.SetClipManifestAssetPath(Sdf.AssetPath('./' + os.path.basename(self.topologyFile)), self.clipSet)
        clipObj.SetClipPrimPath(defaultPath, self.clipSet)

        activeTimes = list()
        for i in range(len(times)):
            activeTimes.append((times[i][0], i))
        clipObj.SetClipActive(activeTimes, self.clipSet)

        setTimes = list()
        for i in range(len(times)):
            setTimes.append((times[i][0], times[i][0]))
            setTimes.append((times[i][1], times[i][1]))
            setTimes.append((times[i][1] + 0.5, times[i][1] + 0.5))
        clipObj.SetClipTimes(setTimes, self.clipSet)

        stage.SetDefaultPrim(prim)
        stage.SetStartTimeCode(times[0][0])
        stage.SetEndTimeCode(times[-1][1])
        stage.GetRootLayer().Save()

        return stage


    def timeSampleModify(self, filename):
        stage = Usd.Stage.Open(filename, load=Usd.Stage.LoadNone)
        dprim = stage.GetDefaultPrim()
        for p in iter(Usd.PrimRange.AllPrims(dprim)):
            ptype = p.GetTypeName()
            if ptype == 'Mesh':
                geom = UsdGeom.Mesh(p)

                vtxAttrs = [geom.GetFaceVertexCountsAttr(), geom.GetFaceVertexIndicesAttr()]
                for attr in vtxAttrs:
                    times = attr.GetTimeSamples()
                    if times:
                        value = attr.Get(Usd.TimeCode(times[0]))
                        p.RemoveProperty(attr.GetName())
                        attr.Set(value)

                primvarAPI = UsdGeom.PrimvarsAPI(p)
                for primvar in primvarAPI.GetAuthoredPrimvars():
                    times = primvar.GetTimeSamples()
                    if times:
                        name = primvar.GetName()
                        if name in self._deletePrimvars:
                            type = primvar.GetTypeName()
                            interpolate = primvar.GetInterpolation()
                            value = primvar.Get(Usd.TimeCode(times[0]))
                            index_value = None
                            if primvar.IsIndexed():
                                index_value = primvar.GetIndices(Usd.TimeCode(times[0]))
                                p.RemoveProperty(primvar.GetIndicesAttr().GetName())
                            p.RemoveProperty(name)
                            new = primvarAPI.CreatePrimvar(name, type)
                            new.SetInterpolation(interpolate)
                            if index_value:
                                new.SetIndices(index_value)
                            new.Set(value)

        self.stageSave(stage)


    def CleanUp(self, filename):
        stage = Usd.Stage.Open(filename, load=Usd.Stage.LoadNone)
        for p in iter(Usd.PrimRange.AllPrims(stage.GetDefaultPrim())):
            ptype = p.GetTypeName()
            if ptype == 'Mesh':
                geom = UsdGeom.Mesh(p)
                p.RemoveProperty('faceVertexCounts')
                p.RemoveProperty('faceVertexIndices')

                primvarAPI = UsdGeom.PrimvarsAPI(p)
                for primvar in primvarAPI.GetAuthoredPrimvars():
                    if primvar.IsIndexed():
                        p.RemoveProperty(primvar.GetIndicesAttr().GetName())
                    p.RemoveProperty(primvar.GetName())
        self.stageSave(stage)


    def stageSave(self, stage):
        rootLayer = stage.GetRootLayer()
        filename = rootLayer.identifier
        outfile  = filename.replace('.usd', '_tmp.usd')
        rootLayer.Export(outfile)
        os.remove(filename)
        os.rename(outfile, filename)




#-------------------------------------------------------------------------------
#
#   Flatten merge per frame files and stitch files
#
#-------------------------------------------------------------------------------
class MergeCoalesceFiles:
    '''
    Merge per frame files by size (max 10GB), and stitch

    Args
        fileRule (str): /path/filename.*.usd
        mergeFrames (list): pre-computed 'computeMergeFrames' result
        modify (bool): StitchFiles modify option
    '''
    def __init__(self, fileRule, frameRange, step=1, outFile=None, mergeFrames=None, modify=True, clipSet='default'):
        self.frameFiles = GetPerFrameFiles(fileRule, frameRange)
        self.fr   = frameRange
        self.step = step
        self.outFile = outFile
        if not self.outFile:
            self.outFile = fileRule.replace('.*.usd', '.usd')
        self.modify = modify
        self.clipSet= clipSet

        self.mergeFrames = list()
        if self.frameFiles:
            if mergeFrames != None:
                self.mergeFrames = mergeFrames
            else:
                self.mergeFrames = self.computeMergeFrames()


    def doIt(self):
        if not self.frameFiles:
            return

        # for mesh extent timesample modify ( will be remove by ZENN bugfixed )
        for f in self.frameFiles:
            if '/zenn/' in f:
                ActualExtentCompute(f)

        if self.mergeFrames:
            mergeFiles = list()
            for fr in self.mergeFrames:
                filename = self.flattenMerge(fr)
                mergeFiles.append(filename)
            StitchFiles(mergeFiles, outFile=self.outFile, modify=self.modify, clipSet=self.clipSet).doIt()

        else: # big size per frame file
            CoalesceFiles(self.frameFiles, self.fr, step=self.step, outFile=self.outFile, clipSet=self.clipSet)

        return self.outFile


    def getSourceFiles(self, fr):
        files = list()
        for f in self.frameFiles:
            frame = re.findall(r'\.(\d+)?\.', f)
            if frame:
                iframe = int(frame[0])
                if fr[0] <= iframe <= fr[1]:
                    files.append(f)
        return files

    def flattenMerge(self, fr):
        sourceFiles = self.getSourceFiles(fr)

        dxsMsg.Print('info', "[MergeCoalesceFiles.flattenMerge]\n\t-> sample file : %s\n\t-> frameRange : %s" % (sourceFiles[0], fr))

        clipFile, topologyFile = CoalesceFiles(sourceFiles, fr, step=self.step)

        outFile = sourceFiles[0].replace('.usd', '_tmp.usd')
        stage = Usd.Stage.Open(clipFile)
        stage.Flatten()
        stage.Export(outFile)

        # clean up files
        os.remove(clipFile)
        os.remove(topologyFile)
        for f in sourceFiles:
            os.remove(f)
        os.rename(outFile, sourceFiles[0])
        return sourceFiles[0]


    def computeMergeFrames(self):
        '''
        Return
            (list): [(start, end), (start, end), ...]
        '''
        refSize = os.path.getsize(self.frameFiles[0])
        refSize = refSize / (1000.0 * 1000.0 * 1000.0) # GB

        numframes = self.fr[1] - self.fr[0] + 1
        totalSize = refSize * numframes
        limitSize = 10.0 # GB

        if refSize < (limitSize * 0.5):
            framesPerFile = int(numframes / (totalSize / limitSize))
            frames = list()
            for frame in range(self.fr[0], self.fr[1] + 1, framesPerFile):
                endframe = frame + (framesPerFile - 1)
                if endframe >= self.fr[1]:
                    endframe = self.fr[1]
                frames.append((frame, endframe))
            dxsMsg.Print('info', "[MergeCoalesceFiles.computeMergeFrames]\n\t-> limit per-file size : %.2fGB\n\t-> estimate total size : %.2fGB\n\t-> frames : %s" % (limitSize, totalSize, frames))
            return frames




#-------------------------------------------------------------------------------
#
#   Make LoopClip
#
#-------------------------------------------------------------------------------
class LoopClip:
    '''
    Args
        inputFile (str): geometry master file
        fr (tuple): export loop range
    '''
    def __init__(self, inputFile, scales=[1.0], fr=(0, 0), fs=[0.0]):
        # Member variables
        self.showDir  = None
        self.assetName= None
        self.clipDir  = None
        self.loopLayer= None
        self.version  = None

        self.timeScales = scales
        self.fs = fs    # frameSample (list)
        self.fr = fr    # frameRange (tuple)
        self.expfr = self.fr

        self.isPurpose = False
        self.isLod = False

        self.fileParse(inputFile)
        self.readClipData(inputFile)

        self.expfr = self.fr
        if not fr[0] or not fr[1]:
            self.expfr = (self.clip_start, self.clip_start + 1000)
        if self.expfr[0] > 100:
            self.expfr = (self.expfr[0] - 100, self.expfr[1])

    def fileParse(self, filename):
        splitPath = filename.split('/')
        if '/assetlib/' in splitPath:
            self.showDir = '/assetlib/3D'
        else:
            if '/show/' in filename:
                self.showDir = string.join(splitPath[:splitPath.index('show')+2], '/')
        if '/asset/' in filename:
            self.assetName = splitPath[splitPath.index('asset')+1]

        if not self.showDir or not self.assetName:
            assert False, '# [ASSERT] ClipUtils.LoopClip : file path error!'

        assetPath = '{DIR}/asset/{ASSET}'.format(DIR=self.showDir, ASSET=self.assetName)
        if filename.find(assetPath) == -1:
            assert False, '# [ASSERT] ClipUtils.LoopClip : file assetPath error!'

        self.clipDir = assetPath + '/clip'
        splitPath = filename.split(self.clipDir)[-1].split('/')
        self.version  = splitPath[1]
        self.loopLayer= splitPath[2].replace('clip', 'loop')

    def readClipData(self, filename):
        stage = Usd.Stage.Open(filename)

        self.clip_start = int(stage.GetStartTimeCode())
        self.clip_end   = int(stage.GetEndTimeCode())
        self.clip_fps   = int(stage.GetFramesPerSecond())

        dprim = stage.GetDefaultPrim()

        self.clip_subLayers = list()
        dirpath = os.path.dirname(filename)
        for lyr in stage.GetRootLayer().subLayerPaths:
            self.clip_subLayers.append(os.path.abspath(os.path.join(dirpath, lyr)))

        self.clip_geomFiles = list()
        vsetNames = dprim.GetVariantSets().GetNames()
        if 'lodVariant' in vsetNames:
            self.isLod = True
            # vset = dprim.GetVariantSet('lodVariant')
            current = vsetNames.GetVariantSelection()
            for val in current.GetVariantNames():
                current.SetVariantSelection(val)
                for s in dprim.GetPrimStack():
                    sfile = s.layer.identifier
                    if '_geom.usd' in sfile:
                        self.clip_geomFiles.append(sfile)
                        break
        else:
            # just payload
            renderPrim = stage.GetPrimAtPath(dprim.GetPath().AppendChild('render'))
            proxyPrim = stage.GetPrimAtPath(dprim.GetPath().AppendChild('proxy'))
            if renderPrim:
                self.clip_geomFiles.append(dxsUsdUtils.GetPayloadFile(renderPrim))
                self.isPurpose = True
            if proxyPrim:
                self.clip_geomFiles.append(dxsUsdUtils.GetPayloadFile(proxyPrim))
                self.isPurpose = True

            srcfile = dxsUsdUtils.GetPayloadFile(dprim)
            if srcfile:
                self.clip_geomFiles.append(srcfile)

        self.clip_geomFiles = list(set(self.clip_geomFiles))

    def getPayloadFile(self, prim):
        payload = prim.GetMetadata('payload')
        if payload:
            assetPath = prim.GetAssetInfoByKey('identifier')
            return assetPath.resolvedPath

    def getChildrenPayloadFiles(self, prim):
        result = list()
        for p in prim.GetChildren():
            result.append(self.getPayloadFile(p))
            purposeAttr = UsdGeom.Scope(p).GetPurposeAttr()
            if purposeAttr.Get() != 'default':
                self.isPurpose = True
        return result

    def getGeomType(self, filename):
        baseName = os.path.basename(filename)
        splitName= baseName.split('.')
        return splitName[-2].split('_')[0]

    @staticmethod
    def ComputeClipTimes(frameRange, frameSample, clipRange, timeScale):
        '''
        Args
            sample: (1, 100), [0.0], (1, 12), 1
        Return
            (list): active times [(start, end), (start, end), ...]
        '''
        times = list()

        step = int(round( (clipRange[1] - clipRange[0] + 1) * (1.0 / timeScale) ))

        for i in range((frameRange[1] - frameRange[0]) / step):
            c_start = frameRange[0] + step * i
            c_end   = c_start + step - 1
            if len(frameSample) > 1:
                # start
                for t in frameSample:
                    timeset = (c_start + t, clipRange[0] + t)
                    if not timeset in times:
                        times.append(timeset)
                # end
                for t in frameSample:
                    timeset = (c_end + t, clipRange[1] + t)
                    if not timeset in times:
                        times.append(timeset)
            else:
                times.append((c_start, clipRange[0]))
                times.append((c_end, clipRange[1]))
                times.append((c_end + 0.5, clipRange[1] + 0.5))
        return times

    @staticmethod
    def GetClipTimes(filename):
        stage = Usd.Stage.Open(filename, load=Usd.Stage.LoadNone)
        dprim = stage.GetDefaultPrim()
        clipObj = Usd.ClipsAPI(dprim)
        times   = clipObj.GetClipTimes()
        return times


    def doIt(self):
        if not self.clip_geomFiles:
            # assert False, '# [ASSERT] ClipUtils.LoopClip : not found clip geometry files.'
            return None

        layerMasters = list()
        for ts in self.timeScales:
            scaleName = str(ts).replace('.', '_')
            loopName  = self.loopLayer + scaleName
            geomDir   = '{DIR}/{VER}/{LAYER}'.format(DIR=self.clipDir, VER=self.version, LAYER=loopName)
            for f in self.clip_geomFiles:
                baseName = os.path.basename(f)
                baseName = baseName.replace('.usd', '_loop.usd')
                geomfile = os.path.join(geomDir, baseName)
                self.makeValueClipGeom(geomfile, f, ts)
                masterFile = self.makeGeomPackage(geomfile, loopName)
                layerMasters.append(masterFile)
                # preview
                previewFile= masterFile.replace('.payload.usd', '.preview.usd')
                self.makeGeomPreview(previewFile, geomfile)
        return layerMasters


    def makeValueClipGeom(self, outFile, clipFile, timeScale, clipTimes=None):
        tmpLayer   = Sdf.Layer.FindOrOpen(clipFile)
        defaultPath= tmpLayer.rootPrims[0].path.pathString
        del tmpLayer
        if not clipTimes:
            clipTimes = LoopClip.ComputeClipTimes(self.expfr, self.fs, (self.clip_start, self.clip_end), timeScale)

        stage  = SessionUtils.MakeInitialStage(outFile, clear=True)
        relpath= PathUtils.GetRelPath(outFile, clipFile)
        SessionUtils.InsertSubLayer(stage.GetRootLayer(), relpath)

        prim = stage.OverridePrim(defaultPath)
        clipObj = Usd.ClipsAPI(prim)
        clipObj.SetClipAssetPaths([Sdf.AssetPath(relpath)], 'default')
        clipObj.SetClipPrimPath(defaultPath)
        clipObj.SetClipActive([(0, 0)])
        clipObj.SetClipTimes(clipTimes)

        stage.SetDefaultPrim(prim)
        stage.SetStartTimeCode(clipTimes[0][0])
        stage.SetEndTimeCode(clipTimes[-1][0])
        stage.SetFramesPerSecond(self.clip_fps)
        stage.SetTimeCodesPerSecond(self.clip_fps)
        stage.GetRootLayer().Save()


    def makeGeomPackage(self, geomFile, layer):
        geomDir = os.path.dirname(geomFile)

        masterFile = '{DIR}/{NAME}.usd'.format(DIR=geomDir, NAME=layer)
        geomType = self.getGeomType(geomFile)
        purpose = None
        if self.isPurpose:
            purpose = 'proxy' if geomType == 'low' else 'render'
        sourceFiles = [(geomFile, purpose)]
        SdfPath = Sdf.Path('/' + self.assetName)
        if self.isLod:
            SdfPath = SdfPath.AppendVariantSelection('lodVariant', geomType)
        SessionUtils.MakeReferenceStage(masterFile, sourceFiles, SdfPath=SdfPath, Name=self.assetName + '_' + layer)

        if self.isPurpose and self.isLod:
            dxsUsdUtils.PerfectionLodVariantPackage(masterFile)
        if self.isLod:
            dxsUsdUtils.LodVariantDefaultSelection(masterFile)

        if self.clip_subLayers:
            outLayer = Sdf.Layer.FindOrOpen(masterFile)
            for f in self.clip_subLayers:
                relpath = PathUtils.GetRelPath(masterFile, f)
                SessionUtils.InsertSubLayer(outLayer, relpath)
            outLayer.Save()

        masterPayloadFile = masterFile.replace('.usd', '.payload.usd')
        SdfPath = '/' + self.assetName + '{loopVariant=%s}' % layer
        SessionUtils.MakeReferenceStage(masterPayloadFile, [(masterFile, None)], SdfPath=SdfPath, clear=True)
        return masterPayloadFile

    def makeGeomPreview(self, outFile, refFile):
        sourceFile = outFile.replace('.preview.usd', '.usd')
        # reference file query
        times = LoopClip.GetClipTimes(refFile)
        fr = (times[0][0], times[1][0])

        SessionUtils.MakeReferenceStage(outFile, [(sourceFile, None)], SdfPath='/'+self.assetName, fr=fr, fps=self.clip_fps, clear=True)




#-------------------------------------------------------------------------------
#
#   pxrUsdReferenceAssembly clip timeline Edit
#
#-------------------------------------------------------------------------------
class ClipEdit:
    '''
    pxrUsdReferenceAssembly clip timeline Edit by visibility.
    Args:
        rootNode (str)  - root group node
        firstLoop (int) - loop count
        setTimes (list) - specified edit point
    '''
    def __init__(self, rootNode=None, firstLoop=2, setTimes=list()):
        if not rootNode:
            rootNode = cmds.ls(sl=True)[0]
        self.firstLoop = firstLoop
        self.setTimes  = setTimes

        # member variables
        self.source  = list()
        self.rootNode= rootNode
        if rootNode:
            self.source = cmds.listRelatives(rootNode, c=True, type='pxrUsdReferenceAssembly')
        self.globalOffset = 0
        if cmds.attributeQuery('GlobalOffset', n=self.rootNode, ex=True):
            self.globalOffset = cmds.getAttr('%s.GlobalOffset' % self.rootNode)
            cmds.setAttr('%s.GlobalOffset' % self.rootNode, 0)

    def doIt(self):
        if not self.source:
            assert False, '# Error : Not found source nodes.'
            return
        self.clearRig()

        editPoint = None    # clip start frame
        for i in range(len(self.source)):
            node = self.source[i]
            ClipEdit.ClearVisibility(node)
            dxsMayaUtils.ConnectTimeOffset(node)

            times = ClipEdit.GetClipTime(node)

            index, next = self.getClipInfo(times)
            startFrame = times[index][0]
            endFrame   = times[next][0]
            duration   = endFrame - startFrame

            if not editPoint:
                startFrame = int(times[index * self.firstLoop][0])
                endFrame   = int(times[next * self.firstLoop][0])
                if self.setTimes and len(self.setTimes) > i:
                    endFrame = self.setTimes[i]

            if editPoint:
                offset = editPoint + 1 - startFrame
                ClipEdit.SetOffset(node, offset * -1)
                ClipEdit.SetVisibility(node, editPoint, 0, 1)
                editPoint += duration
            else:
                editPoint = endFrame

            if self.source[-1] != node:
                ClipEdit.SetVisibility(node, editPoint, 1, 0)

        self.rigSetup()
        cmds.setAttr('%s.GlobalOffset' % self.rootNode, self.globalOffset)
        cmds.select(self.rootNode)


    def getClipInfo(self, times):
        clipTimes = list()
        for i in range(len(times)):
            clipTimes.append(times[i][1])
        clipTimes = list(set(clipTimes))
        clipTimes.sort()
        clipRange = (int(clipTimes[0]), int(clipTimes[-1]))

        index = 0
        for i in range(1, len(times)):
            if times[0][1] == times[i][1]:
                index = i
                break
        next = 1
        for i in range(index, len(times)):
            if clipRange[1] == times[i][1]:
                next = i
                break
        return index, next

    @staticmethod
    def GetClipTime(node):
        filePath = cmds.getAttr('%s.filePath' % node)
        stage = Usd.Stage.Open(filePath)
        dprim = stage.GetDefaultPrim()

        variantAttrs = cmds.listAttr(node, st='usdVariantSet_*')
        if variantAttrs:
            for attr in variantAttrs:
                variantSetName = attr.replace('usdVariantSet_', '')
                variantName    = cmds.getAttr('%s.%s' % (node, attr))
                vset = dprim.GetVariantSets().GetVariantSet(variantSetName)
                vset.SetVariantSelection(variantName)

        clipApi = Usd.ClipsAPI(dprim)
        times = clipApi.GetClipTimes()
        return times

    @staticmethod
    def SetVisibility(node, frame, cval, nval):
        # cval : current frame, nval : next frame
        cmds.setKeyframe(node, at='visibility', t=frame, v=cval, s=False)
        cmds.setKeyframe(node, at='visibility', t=frame+1, v=nval, s=False)

    @staticmethod
    def ClearVisibility(node):
        animCurve = cmds.listConnections('%s.visibility' % node, type='animCurve')
        if animCurve:
            cmds.delete(animCurve)
            cmds.setAttr('%s.visibility' % node, 1)

    @staticmethod
    def SetOffset(node, offset):
        offsetNode = cmds.listConnections('%s.time' % node, s=True, d=False, type='dxTimeOffset')
        if offsetNode:
            cmds.setAttr('%s.offset' % offsetNode[0], offset)


    #---------------------------------------------------------------------------
    def SetAttr(self, node, name, value=0.0):
        if not cmds.attributeQuery(name, n=node, ex=True):
            cmds.addAttr(node, ln=name, at='double')
        cmds.setAttr('%s.%s' % (node, name), value)

    def rigSetup(self):
        rootName = self.rootNode.split(':')[-1].split('|')[-1]

        visAnimNodes = list()
        offsetNodes  = list()
        for n in self.source:
            animCurve = cmds.listConnections('%s.visibility' % n, s=True, d=False, type='animCurve')
            if animCurve:
                visAnimNodes += animCurve
            offsetNode= cmds.listConnections('%s.time' % n, s=True, d=False, type='dxTimeOffset')
            if offsetNode:
                offsetNodes += offsetNode

        self.SetAttr(self.rootNode, 'GlobalOffset', 0.0)

        # GlobalOffset - Visibility
        globalOffset_pmaNode = cmds.createNode('plusMinusAverage', name='%s_Visibility_GlobalOffset' % rootName)
        cmds.setAttr('%s.operation' % globalOffset_pmaNode, 2)
        cmds.connectAttr('time1.outTime', '%s.input1D[0]' % globalOffset_pmaNode, f=True)
        cmds.connectAttr('%s.GlobalOffset' % self.rootNode, '%s.input1D[1]' % globalOffset_pmaNode, f=True)
        for n in visAnimNodes:
            cmds.connectAttr('%s.output1D' % globalOffset_pmaNode, '%s.input' % n, f=True)

        # dxTimeOffset
        for offsetNode in offsetNodes:
            name = offsetNode.split(':')[-1].split('|')[-1]
            value= cmds.getAttr('%s.offset' % offsetNode)

            pmaNode = cmds.createNode('plusMinusAverage', name=name.replace('TimeOffset', 'plusMinusAverage'))
            cmds.setAttr('%s.operation' % pmaNode, 2)
            cmds.setAttr('%s.input1D[0]' % pmaNode, value)
            cmds.connectAttr('%s.GlobalOffset' % self.rootNode, '%s.input1D[1]' % pmaNode, f=True)
            cmds.connectAttr('%s.output1D' % pmaNode, '%s.offset' % offsetNode, f=True)

    def clearRig(self):
        targetNodes = list()
        initAttrs   = dict()
        historyNodes= cmds.listHistory(self.rootNode, future=True, bf=True)
        for n in historyNodes[1:]:
            ntype = cmds.nodeType(n)
            if ntype.find('pxrUsd') == -1 and ntype.find('animCurve') == -1 and ntype.find('dxTime') == -1:
                targetNodes.append(n)
            if ntype == 'plusMinusAverage':
                connected = cmds.listConnections(n, s=False, d=True, p=True, type='dxTimeOffset')
                if connected:
                    value = cmds.getAttr('%s.input1D[0]' % n)
                    initAttrs[connected[0]] = value

        cmds.delete(targetNodes)
        for ln in initAttrs:
            cmds.setAttr(ln, initAttrs[ln])
