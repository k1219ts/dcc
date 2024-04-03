import os
from pxr import Usd, UsdGeom, Sdf

#-------------------------------------------------------------------------------
#
#   File Info
#
#-------------------------------------------------------------------------------
def GetMetaData(filename):
    '''
    Grab some meta data from a USD File
    :return
        timeInfo (start, end, fps)
        defaultPath (Sdf.Path)
        firstRoot   (Sdf.Path)
        primTypeName (str)
    '''
    timeInfo    = None # (start, end, fps)
    defaultPath = None
    firstRoot   = None
    primTypeName= None

    rootLayer = Sdf.Layer.FindOrOpen(filename)
    if not rootLayer:
        return None, None, None, None
    stage = Usd.Stage.Open(rootLayer, load=Usd.Stage.LoadNone)
    if stage.HasAuthoredTimeCodeRange():
        timeInfo = (
            stage.GetStartTimeCode(), stage.GetEndTimeCode(), stage.GetFramesPerSecond()
        )
    if stage.HasDefaultPrim():
        defaultPrim = stage.GetDefaultPrim()
        defaultPath = defaultPrim.GetPath()
        primTypeName= defaultPrim.GetTypeName()
    else:
        firstRoot = rootLayer.rootPrims[0].path
        primTypeName = stage.GetPrimAtPath(firstRoot).GetTypeName()
    return timeInfo, defaultPath, firstRoot, primTypeName


def GetPrimStackFilePath(prim, filename):
    stack = prim.GetPrimStack()
    target = None
    for s in stack:
        f = s.layer.identifier
        if f.endswith(filename):
            target = f
    return target


#-------------------------------------------------------------------------------
#
#   Variant
#
#-------------------------------------------------------------------------------
def VariantSelection(prim, name, value):
    vset = prim.GetVariantSets().GetVariantSet(name)
    if not vset:
        vset = prim.GetVariantSets().AddVariantSet(name)
    vset.AddVariant(value)
    vset.SetVariantSelection(value)
    return vset


def PerfectionLodVariantPackage(filename):
    stage = Usd.Stage.Open(filename, load=Usd.Stage.LoadNone)
    prim  = stage.GetDefaultPrim()
    vset  = prim.GetVariantSets().GetVariantSet('lodVariant')
    names = vset.GetVariantNames()
    if 'low' in names:
        vset.SetVariantSelection('low')
        pxprim = prim.GetChild('proxy')
        if pxprim:
            payloads = pxprim.GetMetadata('payload')
            assetPath = None
            try:
                if isinstance(payloads, Sdf.Payload):
                    assetPath = payloads.assetPath
                    payload = payloads
                elif isinstance(payloads, Sdf.PayloadListOp):
                    assetPath = payloads.GetAddedOrExplicitItems()[0].assetPath
                    payload = payloads.GetAddedOrExplicitItems()[0]
                else:
                    raise()
            except:
                print "# Error: Not found 'payload'"

            for n in names:
                vset.SetVariantSelection(n)
                if n == 'low':
                    with vset.GetVariantEditContext():
                        p = stage.DefinePrim(prim.GetPath().AppendChild('render'), 'Xform')
                        UsdGeom.Scope(p).CreatePurposeAttr(UsdGeom.Tokens.render)
                        p.SetPayload(payload)
                        SetModelAPI(p, identifier=assetPath)
                else:
                    with vset.GetVariantEditContext():
                        p = stage.DefinePrim(prim.GetPath().AppendChild('proxy'), 'Xform')
                        UsdGeom.Scope(p).CreatePurposeAttr(UsdGeom.Tokens.proxy)
                        p.SetPayload(payload)
                        SetModelAPI(p, identifier=assetPath)
    stage.GetRootLayer().Save()

def LodVariantDefaultSelection(filename):
    stage = Usd.Stage.Open(filename, load=Usd.Stage.LoadNone)
    prim  = stage.GetDefaultPrim()
    vset  = prim.GetVariantSets().GetVariantSet('lodVariant')
    names = vset.GetVariantNames()
    if 'high' in names:
        vset.SetVariantSelection('high')
    stage.GetRootLayer().Save()



#-------------------------------------------------------------------------------
#
#   ModelAPI
#
#-------------------------------------------------------------------------------
def SetModelAPI(prim, kind=None, name=None, identifier=None):
    api = Usd.ModelAPI(prim)
    if kind:
        api.SetKind(kind)
    if name:
        api.SetAssetName(name)
    if identifier:
        api.SetAssetIdentifier(identifier)



#-------------------------------------------------------------------------------
#
#   Attributes
#
#-------------------------------------------------------------------------------
def AddPrimvar(geom, name, type, interpolation, value):
    primvar = geom.CreatePrimvar('primvars:' + name, type)
    primvar.SetInterpolation(interpolation)
    if value != None:
        primvar.Set(value)

def CreateRiAttribute(prim, RiAttrName, Value, SdfValueType):
    UsdGeom.PrimvarsAPI(prim).CreatePrimvar('ri:attributes:'+RiAttrName, SdfValueType).Set(Value)

def CreateConstPrimvar(prim, VarName, VarValue, SdfValueType):
    UsdGeom.PrimvarsAPI(prim).CreatePrimvar(VarName, SdfValueType, UsdGeom.Tokens.constant).Set(VarValue)

def CreateUserProperty(prim, AttrName, AttrValue, SdfValueType):
    prim.CreateAttribute('userProperties:'+AttrName, SdfValueType).Set(AttrValue)


#-------------------------------------------------------------------------------
#
#   Composite Archive
#
#-------------------------------------------------------------------------------
def SetPayload(prim, payload):
    prim.SetPayload(payload)
    api = Usd.ModelAPI(prim)
    api.SetAssetIdentifier(payload.assetPath)

def SetReference(prim, reference):
    prim.GetReferences().AddReference(reference)
    SetModelAPI(prim, identifier=reference.assetPath)

# Query
def GetPayloadFile(prim):
    payload = prim.GetMetadata('payload')
    if payload:
        stagedir = os.path.dirname(prim.GetStage().GetRootLayer().identifier)
        srcfile   = payload.explicitItems[0].assetPath
        return os.path.abspath(os.path.join(stagedir, srcfile))

#-------------------------------------------------------------------------------
#
#   Xform
#
#-------------------------------------------------------------------------------
