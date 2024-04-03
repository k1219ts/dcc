import maya.api.OpenMaya as OpenMaya
import maya.cmds as cmds

from pxr import Usd, UsdGeom, UsdUtils, Sdf, Kind, Vt, Gf

def copySpecByXform(stage, srcPath, dstPath):
    # Debug.Info('copySpecByXform')
    srcPrim = stage.GetPrimAtPath(srcPath)
    xformOpAttrs = srcPrim.GetPropertiesInNamespace('xformOp')
    if xformOpAttrs:
        for attr in xformOpAttrs:
            attrPath = attr.GetPath()
            attrName = attr.GetName()
            Sdf.CopySpec(stage.GetRootLayer(), attrPath, stage.GetRootLayer(), Sdf.Path('%s.%s' % (dstPath, attrName)))
        # Sdf.CopySpec(self.outLayer, Sdf.Path('%s.xformOp' % srcPath),
        #              self.outLayer, Sdf.Path('%s.xformOpOrder' % tp))
        print srcPath, dstPath
        Sdf.CopySpec(stage.GetRootLayer(), Sdf.Path('%s.xformOpOrder' % srcPath), stage.GetRootLayer(), Sdf.Path('%s.xformOpOrder' % dstPath))

def convertUsdPathToMayaPath(primPath, nsLayer=''):
    # Debug.Info('convertUsdPathToMayaPath')
    nodePath = primPath
    if nsLayer:
        nodePath = nodePath.replace('/%s_' % nsLayer.replace(':', '_'), '|%s:' % nsLayer)
    nodePath = nodePath.replace('/', '|')
    return nodePath


def GetMayaApiCurveFn(shape):
    '''
    Args:
        shape (str): shapename
    Returns:
        MFnNurbsCurve
    '''
    selection = OpenMaya.MSelectionList()
    selection.add(shape)
    mobj = selection.getDependNode(0)
    curveFn = OpenMaya.MFnNurbsCurve(mobj)
    curveFn.updateCurve()
    return curveFn


def GetSplineData(input):
    '''
    Args:
        input (MFnNurbsCurve) : if str, get MFnNurbsCurve
    Returns:
        list : Gf.Vec3f points
        list : float widths
    '''
    baseWidth = 0.01; tipWidth = 0.01

    if type(input).__name__ == 'MFnNurbsCurve':
        curveFn = input
    else:
        curveFn = GetMayaApiCurveFn(input)
    shape = curveFn.getPath().fullPathName()
    if cmds.attributeQuery('rman__torattr___curveBaseWidth', n=shape, ex=True):
        baseWidth = round(cmds.getAttr('%s.rman__torattr___curveBaseWidth' % shape), 3)
    if cmds.attributeQuery('rman__torattr___curveTipWidth', n=shape, ex=True):
        tipWidth = round(cmds.getAttr('%s.rman__torattr___curveTipWidth' % shape), 3)

    points = list(); widths = list()

    cvPoints = curveFn.cvPositions(OpenMaya.MSpace.kObject)
    for p in cvPoints:
        points.append(Gf.Vec3f(p.x, p.y, p.z))
    points.insert(0, points[0])
    points.insert(0, points[0])
    points.append(points[-1])
    points.append(points[-1])

    if baseWidth != tipWidth:
        sumLen = list()
        length = 0
        for i in range(len(cvPoints) - 1):
            v = cvPoints[i] - cvPoints[i+1]
            l = v.length()
            length += l
            sumLen.append(length)
        for l in sumLen:
            w = baseWidth - (l / sumLen[-1]) * (baseWidth - tipWidth)
            widths.append(w)
        widths.insert(0, baseWidth)
        widths.insert(0, baseWidth)
        widths.append(tipWidth)
    else:
        widths.append(baseWidth)
    return points, widths



# class NurbsToBasis:
#     '''
#     Change NurbsCurves to BasisCurves
#     Args:
#         inputFile (str): usd geometry file
#         ns (str) : maya namespace
#     Returns:
#         None : just overwrite inputFile
#     '''
#     def __init__(self, inputFile, ns=None):
#         self.nameSpace= ns
#         self.outLayer = Sdf.Layer.FindOrOpen(inputFile)
#         if not self.outLayer:
#             assert False, '# msg : not found file'
#         self.stage = Usd.Stage.Open(self.outLayer, load=Usd.Stage.LoadNone)
#         self.defaultPrim = self.stage.GetDefaultPrim()
#         if not self.defaultPrim:
#             return
#         self.doIt()
#         self.stage.GetRootLayer().Save()
#
#     def doIt(self):
#         self.MergeXformAndGeom()
#         self.ChangeCurve()
#
#
#     def ChangeCurve(self):
#         '''
#         Render NurbsCurves -> BasisCurves
#         '''
#         RemoveNurbsCurvesEdit = Sdf.BatchNamespaceEdit()
#         RenameBasisCurvesEdit = Sdf.BatchNamespaceEdit()
#
#         deformedGeoms = list()
#         allFrames     = list()
#
#         for p in Usd.PrimRange.AllPrims(self.defaultPrim):
#             if p.GetTypeName() == 'NurbsCurves':
#                 pathStr = p.GetPath().pathString
#                 nodePath= self.getNodePath(pathStr)
#                 nodeType= cmds.nodeType(nodePath)
#                 if nodeType == 'nurbsCurve':
#                     pathStr = p.GetPath().GetParentPath().pathString
#                 curveShape  = cmds.ls(nodePath, dag=True, type='nurbsCurve', ni=True)[0]
#
#                 # render curve : nurbsCurve -> basisCurve
#                 renderAttrs = cmds.listAttr(curveShape, st=['*BaseWidth', '*TipWidth'])
#                 if not renderAttrs:
#                     continue
#
#                 curveFn = GetMayaApiCurveFn(curveShape)
#
#                 # remove nurbsCurve
#                 RemoveNurbsCurvesEdit.Add(pathStr, Sdf.Path.emptyPath)
#
#                 newPath = Sdf.Path(pathStr + '_basisCurve')
#                 newPrim, newGeom = self.makeBasisCurve(newPath)
#                 RenameBasisCurvesEdit.Add(newPath.pathString, pathStr)
#
#                 # Copy XformOp
#                 self.CopyXform(self.stage.GetPrimAtPath(pathStr), newPrim)
#
#                 # Set Geometry Data
#                 pointsAttr = p.GetAttribute('points')
#                 frames = pointsAttr.GetTimeSamples()
#                 if frames:
#                     deformedGeoms.append((curveFn, newGeom, frames))
#                     allFrames += frames
#                     self.SetBasisCurveInit(curveFn, newGeom, setData=False)
#                 else:
#                     self.SetBasisCurveInit(curveFn, newGeom, setData=True)
#
#         # Set Deformed Geometry Data
#         if deformedGeoms and allFrames:
#             allFrames = list(set(allFrames))
#             allFrames.sort()
#             for f in allFrames:
#                 cmds.currentTime(f)
#                 for cfn, geom, frames in deformedGeoms:
#                     if f in frames:
#                         points, widths = GetSplineData(cfn)
#                         self.SetBasisCurveData(geom, points, widths, time=Usd.TimeCode(f))
#
#         self.outLayer.Apply(RemoveNurbsCurvesEdit)
#         self.outLayer.Apply(RenameBasisCurvesEdit)
#
#
#     def MergeXformAndGeom(self):
#         createEdit = Sdf.BatchNamespaceEdit()
#         removeEdit = Sdf.BatchNamespaceEdit()
#         renameEdit = Sdf.BatchNamespaceEdit()
#         tmpList = list()
#         for p in Usd.PrimRange.AllPrims(self.defaultPrim):
#             if p.GetTypeName() == 'NurbsCurves':
#                 xformOpAttr= p.GetPropertiesInNamespace('xformOp')
#                 parentPrim = p.GetParent()
#                 parentPath = parentPrim.GetPath().pathString
#                 if not xformOpAttr and parentPrim.GetTypeName() == 'Xform':
#                     tmpPath = parentPath + '_tmp'
#                     tmpList.append(tmpPath)
#                     createEdit.Add(p.GetPath().pathString, tmpPath)
#                     removeEdit.Add(parentPath, Sdf.Path.emptyPath)
#                     renameEdit.Add(tmpPath, parentPath)
#
#         self.outLayer.Apply(createEdit)
#
#         for tp in tmpList:
#             sourcePath = tp.replace('_tmp', '')
#             sourcePrim = self.stage.GetPrimAtPath(sourcePath)
#             targetPrim = self.stage.GetPrimAtPath(tp)
#             self.CopyXform(sourcePrim, targetPrim)
#
#         self.outLayer.Apply(removeEdit)
#         self.outLayer.Apply(renameEdit)
#
#
#     def getNodePath(self, primPath):
#         nodePath = primPath
#         if self.nameSpace:
#             nodePath = nodePath.replace('/%s_' % self.nameSpace.replace(':', '_'), '|%s:' % self.nameSpace)
#         nodePath = nodePath.replace('/', '|')
#         return nodePath
#
#
#     def CopyXform(self, sourcePrim, targetPrim):
#         xformOpOrder = UsdGeom.Xform(sourcePrim).GetXformOpOrderAttr().Get()
#         if xformOpOrder:
#             for name in xformOpOrder:
#                 opAttr = sourcePrim.GetAttribute(name)
#                 if opAttr:
#                     newAttr = self.AddXformOp(targetPrim, name)
#                     if newAttr:
#                         frames  = opAttr.GetTimeSamples()
#                         if frames:
#                             for f in frames:
#                                 val = opAttr.Get(Usd.TimeCode(f))
#                                 newAttr.Set(val, Usd.TimeCode(f))
#                         else:
#                             val = opAttr.Get()
#                             newAttr.Set(val)
#             # invert xformOp translate pivot
#             if 'xformOp:translate:pivot' in xformOpOrder:
#                 UsdGeom.Xform(targetPrim).AddTranslateOp(UsdGeom.XformOp.PrecisionFloat, 'pivot', isInverseOp=True)
#
#
#     def AddXformOp(self, prim, attrName):
#         geom = UsdGeom.Xform(prim)
#         if attrName == 'xformOp:translate:pivot':
#             return geom.AddTranslateOp(UsdGeom.XformOp.PrecisionFloat, 'pivot')
#         if attrName == 'xformOp:translate':
#             return geom.AddTranslateOp()
#         if attrName == 'xformOp:scale':
#             return geom.AddScaleOp()
#         if attrName == 'xformOp:rotateX':
#             return geom.AddRotateXOp()
#         if attrName == 'xformOp:rotateXYZ':
#             return geom.AddRotateXYZOp()
#         if attrName == 'xformOp:rotateXZY':
#             return geom.AddRotateXZYOp()
#         if attrName == 'xformOp:rotateY':
#             return geom.AddRotateYOp()
#         if attrName == 'xformOp:rotateYXZ':
#             return geom.AddRotateYXZOp()
#         if attrName == 'xformOp:rotateYZX':
#             return geom.AddRotateYZXOp()
#         if attrName == 'xformOp:rotateZ':
#             return geom.AddRotateZOp()
#         if attrName == 'xformOp:rotateZXY':
#             return geom.AddRotateZXYOp()
#         if attrName == 'xformOp:rotateZYX':
#             return geom.AddRotateZYXOp()
#
#
#     def makeBasisCurve(self, primPath):
#         geom = UsdGeom.BasisCurves.Define(self.stage, primPath)
#         geom.CreateBasisAttr(UsdGeom.Tokens.bspline)
#         geom.CreateTypeAttr(UsdGeom.Tokens.cubic)
#         geom.CreateCurveVertexCountsAttr()
#         geom.CreatePointsAttr()
#         geom.CreateExtentAttr()
#         return geom.GetPrim(), geom
#
#     def SetBasisCurveInit(self, curveFn, curveGeom, setData=True):
#         points, widths = GetSplineData(curveFn)
#         curveGeom.GetCurveVertexCountsAttr().Set(Vt.IntArray([curveFn.numCVs+4]))
#         if len(widths) > 1:
#             curveGeom.SetWidthsInterpolation(UsdGeom.Tokens.varying)
#         else:
#             curveGeom.SetWidthsInterpolation(UsdGeom.Tokens.constant)
#             curveGeom.GetWidthsAttr().Set(widths)
#         if setData:
#             self.SetBasisCurveData(curveGeom, points, widths)
#
#     def SetBasisCurveData(self, curveGeom, points, widths, time=Usd.TimeCode.Default()):
#         curveGeom.GetPointsAttr().Set(Vt.Vec3fArray(points), time)
#         if len(widths) > 1:
#             curveGeom.GetWidthsAttr().Set(widths, time)
#         actualExtent = UsdGeom.Boundable.ComputeExtentFromPlugins(curveGeom, time)
#         curveGeom.GetExtentAttr().Set(actualExtent, time)


class NurbsToBasis():
    '''
    Convert NurbsCurves to BasisCurves
    Args:
        string inputFile : usd Geometry File
        string nsLayer : maya namespace
    Returns:
        None
    '''
    def __init__(self, inputFile, ns=""):
        self.nsLayer = ns
        self.outLayer = Sdf.Layer.FindOrOpen(inputFile)
        if not self.outLayer:
            assert False, "# Error MSG : Not found file"

        self.stage = Usd.Stage.Open(self.outLayer, load=Usd.Stage.LoadNone)
        self.defaultPrim = self.stage.GetDefaultPrim()
        if not self.defaultPrim:
            return

        self.doIt()

    def doIt(self):
        # Debug.Info('NurbsToBasis.doIt()')
        self.mergeXformAndGeom()
        self.convert()

        self.stage.GetRootLayer().Save()

    def mergeXformAndGeom(self):
        # Debug.Info('NurbsToBasis.mergeXformAndGeom()')
        createEdit = Sdf.BatchNamespaceEdit()
        removeEdit = Sdf.BatchNamespaceEdit()
        renameEdit = Sdf.BatchNamespaceEdit()
        tmpList = list()
        for p in Usd.PrimRange.AllPrims(self.defaultPrim):
            if p.GetTypeName() == 'NurbsCurves':
                xformOpAttrs = p.GetPropertiesInNamespace('xformOp')
                parentPrim = p.GetParent()
                parentPath = parentPrim.GetPath().pathString
                if not xformOpAttrs and parentPrim.GetTypeName() == 'Xform':
                    tmpPath = '%s_tmp' % parentPath
                    tmpList.append(tmpPath)
                    createEdit.Add(p.GetPath().pathString, tmpPath)
                    removeEdit.Add(parentPath, Sdf.Path.emptyPath)
                    renameEdit.Add(tmpPath, parentPath)

        self.outLayer.Apply(createEdit)

        for tp in tmpList:
            copySpecByXform(self.stage, tp.replace('_tmp', ''), tp)

        self.outLayer.Apply(removeEdit)
        self.outLayer.Apply(renameEdit)

    def convert(self):
        '''
        Render Convert NurbsCurves to BasisCurves
        '''
        # Debug.Info('NurbsToBasis.convert()')
        removeNurbsCurvesEdit = Sdf.BatchNamespaceEdit()
        renameBasisCurvesEdit = Sdf.BatchNamespaceEdit()

        deformedGeoms = list()
        allFrames = list()

        for p in Usd.PrimRange.AllPrims(self.defaultPrim):
            if p.GetTypeName() == 'NurbsCurves':
                pathStr = p.GetPath().pathString
                nodePath = convertUsdPathToMayaPath(pathStr, self.nsLayer)
                nodeType = cmds.nodeType(nodePath)
                if nodeType == 'nurbsCurve':
                    pathStr = p.GetPath().GetParentPath().pathString

                curveShape = cmds.ls(nodePath, dag=True, type='nurbsCurve', ni=True)
                if not curveShape:
                    continue

                curveShape = curveShape[0]

                # render curve : nurbsCurve to basisCurve
                renderAttrs = cmds.listAttr(curveShape, st=['*BaseWidth', '*TipWidth'])
                if not renderAttrs:
                    continue

                curveFn = GetMayaApiCurveFn(curveShape)

                # remove nurbsCurve
                removeNurbsCurvesEdit.Add(pathStr, Sdf.Path.emptyPath)

                newPath = Sdf.Path('%s_tmp' % pathStr)
                newPrim, newGeom = self.makeBasisCurve(newPath)
                renameBasisCurvesEdit.Add(newPath.pathString, pathStr)

                copySpecByXform(self.stage, pathStr, newPath)

                # Set Geometry Data
                pointsAttr = p.GetAttribute('points')
                frames = pointsAttr.GetTimeSamples()
                if frames:
                    deformedGeoms.append((curveFn, newGeom, frames))
                    allFrames += frames
                    self.evalBasisCurve(curveFn, newGeom, setData=False)
                else:
                    self.evalBasisCurve(curveFn, newGeom)

        # Set Deformed Geometry Data
        if deformedGeoms and allFrames:
            allFrames = list(set(allFrames))
            allFrames.sort()
            for f in allFrames:
                cmds.currentTime(f)
                for curveFn, curveGeom, frames in deformedGeoms:
                    if f in frames:
                        points, widths = GetSplineData(curveFn)
                        self.setBasisCurveData(curveGeom, points, widths, time=Usd.TimeCode(f))

        self.outLayer.Apply(removeNurbsCurvesEdit)
        self.outLayer.Apply(renameBasisCurvesEdit)

    def makeBasisCurve(self, primPath):
        geom = UsdGeom.BasisCurves.Define(self.stage, primPath)
        geom.CreateBasisAttr(UsdGeom.Tokens.bspline)
        geom.CreateTypeAttr(UsdGeom.Tokens.cubic)
        geom.CreateCurveVertexCountsAttr()
        geom.CreatePointsAttr()
        geom.CreateExtentAttr()
        return geom.GetPrim(), geom

    def evalBasisCurve(self, curveFn, curveGeom, setData=True):
        # Debug.Info('NurbsToBasis.evalBasisCurve()')
        points, widths = GetSplineData(curveFn)
        curveGeom.GetCurveVertexCountsAttr().Set(Vt.IntArray([curveFn.numCVs + 4]))
        if len(widths) > 1:
            curveGeom.SetWidthsInterpolation(UsdGeom.Tokens.varying)
        else:
            curveGeom.SetWidthsInterpolation(UsdGeom.Tokens.constant)
            curveGeom.GetWidthsAttr().Set(widths)
        if setData:
            self.setBasisCurveData(curveGeom, points, widths)

    def setBasisCurveData(self, curveGeom, points, widths, time=Usd.TimeCode.Default()):
        # Debug.Info('NurbsToBasis.setBasisCurveData()')
        curveGeom.GetPointsAttr().Set(Vt.Vec3fArray(points), time)
        if len(widths) > 1:
            curveGeom.GetWidthsAttr().Set(widths, time)
        actualExtent = UsdGeom.Boundable.ComputeExtentFromPlugins(curveGeom, time)
        curveGeom.GetExtentAttr().Set(actualExtent, time)
