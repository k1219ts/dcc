from __future__ import print_function

import soho, hou
from sohog import SohoGeometry
import os, sys
from pxr import Usd,UsdGeom,UsdUtils,Sdf,Kind,Gf,Vt



class Attribute:
    def __init__(self, t=None, n=None):
        self.n = n
        self.t = t if t == 'etc' else Sdf.ValueTypeNames.Find(t)

        if not self.t:
            soho.error('Given Attribute type(%s) is not available.'%t)

        self.v = dict() # {instance name : [values]}
        self.i = dict() # {instance name : 0}

    def count(self, inst):
        return len(self.v[inst])

    def setValue(self, inst, v):
        v = self.asGf(v)
        if inst in self.v.keys():
            self.v[inst].append(v)
        else:
            self.v.update({inst:[v]})
            self.i.update({inst:0})

    def asGf(self, v):
        if self.t in ['int[]']:
            return v
        elif self.t in ['float[]']:
            return v
        elif self.t in ['float2[]']:
            return Gf.Vec2f(*v)
        elif self.t in ['float3[]', 'point3f[]', 'vector3f[]', 'color3f[]']:
            return Gf.Vec3f(*v)
        elif self.t in ['quatf[]']:
            return Gf.Quatf(*v)
        elif self.t in ['quatd[]']:
            return Gf.Quatd(*v)
        else:
            soho.error('Type is not available.')

    def getValue(self, inst):
        v = self.v[inst][self.i[inst]]
        self.i[inst] += 1

        return v

    def __repr__(self):
        return '%s(%s) : %s'%(self.n, self.t, str(self.v))

# ------------------------------------------------------------------------------
# get information by hou

rop       = None
usdRop    = None
batchNode = None

sopPath   = None
srcFile   = None

attrlistNum = 0
attrTypes   = None
attrList    = {}

# get rop node
rop     = hou.pwd()

# get input usd rop node
inputs = rop.inputs()
batchNode = None

if inputs and inputs[0].type().name() == 'batch':
    batchNode = inputs[0]
    inputs = batchNode.inputs()
    if inputs and inputs[0].type().name() == 'pixar::usdoutput':
        usdRop = inputs[0]
elif inputs and inputs[0].type().name() == 'pixar::usdoutput':
    usdRop = inputs[0]

# get attrlist num
attrlistNum = rop.parm('attrlist').evalAsInt()

if attrlistNum > 0:
    attrTypes   = rop.parm('attrtype1').menuLabels()
else:
    soho.error('Need one more attribute lists')

# get sop path
if usdRop:
    sopPath = usdRop.parm('soppath').evalAsString()
    srcFile = usdRop.parm('usdfile').evalAsString()


# ------------------------------------------------------------------------------
# init soho
parameterDefines = {
    'trange'    : soho.SohoParm('trange',     'int',    [0],       False),
    'f'         : soho.SohoParm('f',          'real',   [1, 1, 1], False),
    'now'       : soho.SohoParm('state:time', 'real',   [0],       False, key='now'),
    'fps'       : soho.SohoParm('state:fps',  'real',   [0],       False, key='fps'),
    'hassoppath': soho.SohoParm('hassoppath', 'bool',   [0],       False),
    'hassrcfile': soho.SohoParm('hassrcfile', 'bool',   [0],       False),
    'soppath'   : soho.SohoParm('soppath',    'string', [''],      False),
    'srcfile'   : soho.SohoParm('srcfile',    'string', [''],      False)
}

for i in range(1, attrlistNum + 1):
    parameterDefines.update({'attrtype%d'%i :soho.SohoParm('attrtype%d'%i,  'int',    [''], False)})
    parameterDefines.update({'attrs%d'%i:soho.SohoParm('attrs%d'%i, 'string', [''], False)})

params = soho.evaluate(parameterDefines)
now    = params['now'].Value[0]
fps    = params['fps'].Value[0]
sf = ef = int(now * params['fps'].Value[0] + 1)

if params['trange'].Value[0] > 0:
    if not batchNode and usdRop:
        soho.error('Must be connected with batch node between usdrop and this node.')

    sf = int(params['f'].Value[0])
    ef = int(params['f'].Value[1])

if params['hassoppath'].Value[0]:
    sopPath = params['soppath'].Value[0]

if params['hassrcfile'].Value[0]:
    srcFile = params['srcfile'].Value[0]

if not sopPath:
    soho.error('Specify soppath or connect usdrop as input.')

if not srcFile:
    soho.error('Specify srcfile or connect usdrop as input.')


attrs = []
for i in range(1, attrlistNum + 1):
    type  = attrTypes[params['attrtype%d'%i].Value[0]]
    for name in params['attrs%d'%i].Value[0].split(' '):
        name = name.strip()
        if name:
            attrs.append(Attribute(t=type, n=name))

# ------------------------------------------------------------------------------
# load usd

srcLayer = Sdf.Layer.FindOrOpen(srcFile)
stage = Usd.Stage.Open(srcLayer)

if not srcLayer:
    soho.error('There is no usd file (%s)'%srcFile)

stage = Usd.Stage.Open(srcLayer)
rootPrim = None


# ------------------------------------------------------------------------------
# loop for time range
for f in range(sf, ef+1):
    now = (f-1)/fps
    timecode  = f

    soho.initialize(now, '')
    soho.lockObjects(now)

    # ------------------------------------------------------------------------------
    # get attributes

    gdp = SohoGeometry(sopPath, now)
    primCount     = gdp.globalValue('geo:primcount')[0]
    usdInstPath  = gdp.attribute('geo:prim', 'usdinstancepath')

    for attr in attrs:
        gdpattr = gdp.attribute('geo:prim', attr.n)
        if gdpattr > 0:
            for i in range(primCount):
                inst  = gdp.value(usdInstPath, i)[0]
                value = gdp.value(gdpattr, i)

                attr.setValue(inst, value)


    # ------------------------------------------------------------------------------
    # set attributes

    if stage.HasDefaultPrim():
        rootPrim = stage.GetDefaultPrim()
    else:
        rootPrim = stage.GetPrimAtPath('/')



    for p in Usd.PrimRange(rootPrim):
        if p.GetTypeName() != 'PointInstancer':
            continue

        pointInst     = UsdGeom.PointInstancer(p)
        protoIdcsAttr = pointInst.GetProtoIndicesAttr()
        protoTypes    = []

        for v in pointInst.GetPrototypesRel().GetTargets():
            protoTypes.append(stage.GetPrimAtPath(v).GetName())

        for attr in attrs:
            values = []
            primvar = None

            for i in protoIdcsAttr.Get(timecode):
                v = attr.getValue(protoTypes[i])
                values.append(v)

            # values = attr.asUSD(v, isVal=False)

            if pointInst.HasPrimvar(attr.n):
                primvar = pointInst.GetPrimvar(attr.n)
            else:
                primvar = pointInst.CreatePrimvar(attr.n, attr.t)

            # try:
            primvar.Set(values, timecode)
            # except:
            #     soho.error('Mismatch attribute type. (%s)'%attr.n)


stage.Save()







#
