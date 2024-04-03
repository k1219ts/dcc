import pymel.core as pm
import pymel.core.nodetypes as nt
import pymel.core.datatypes as dt

import json, math

import ch_cmn as cmn


ZEROV = dt.Vector([0, 0, 0])
ONEV  = dt.Vector([1, 1, 1])
XUP   = dt.Vector([1, 0, 0])
YUP   = dt.Vector([0, 1, 0])
ZUP   = dt.Vector([0, 0, 1])

def getSelTransformNode(errMsg=None, err=True):
    try:
        obj =  pm.ls(sl=True)[0]
        if not isinstance(obj, nt.Transform): raise
        return obj
    except:
        if err:
            errMsg = errMsg if errMsg else 'select a transform node.'
            cmn.confirmDialog(errMsg)
        else:
            return None

def getSelAttribute(multi=False, errMsg=None):
    try:
        attrs = pm.channelBox('mainChannelBox', q=True, sma=True)
        if not attrs: raise
        return attrs if multi else attrs[0]
    except:
        errMsg = errMsg if errMsg else 'select an attribute in the channel box.'
        cmn.confirmDialog(errMsg)

def resetTrasform(obj, t=True, r=True, s=True):
    try:
        if t: obj.t.set(ZEROV)
        if r: obj.r.set(ZEROV)
        if s: obj.s.set(ONEV)
    except:
        pm.error('resetTrasform --> Given object is not transform node')

def lockHideAttributes(obj, t=True, r=True, s=True, v=True,
                       lock=True, hide=True):
    trs  = 't' if t else ''
    trs += 'r' if r else ''
    trs += 's' if s else ''

    for a in 'xyz':
        for b in trs:
            attr = b + a
            obj.attr(attr).setLocked(lock)
            obj.attr(attr).setKeyable(not hide)

    if v:
        obj.v.setLocked(lock)
        obj.v.setKeyable(not hide)


def createCtr(type, name, parent=None, color=None):
    jsonpath = __file__.split('/')
    jsonpath[-1] = 'ctrs.json'
    jsonpath = '/'.join(jsonpath)
    with open(jsonpath, 'r') as f:
        ctrs = json.load(f)

    if not type in ctrs.keys():
        pm.error('creaetCtr --> Given type does not exist(%s)'%type)

    ctr = pm.curve(
        name=name,
        degree=ctrs[type]['degree'],
        knot=ctrs[type]['knot'],
        point=ctrs[type]['point']
    )

    if parent:
        ctr.setParent(parent)

    if color:
        setColor(ctr, color)

    return ctr

def setColor(obj, color=None):
    if not color:
        obj.overrideEnabled.set(False)
        return

    obj.overrideEnabled.set(True)
    obj.setObjectColorType('RGBColor')
    obj.overrideRGBColors.set(1)

    colors = {
        'red':dt.Color(1, 0, 0),
        'green':dt.Color(0, 1, 0),
        'blue':dt.Color(0, 0, 1),
        'yellow':dt.Color(1, 1, 0),
        'puple':dt.Color(1, 0, 1),
        'skyblue':dt.Color(0, 1, 1),
        'white':dt.Color(1, 1, 1),
        'black':dt.Color(0, 0, 0)
    }

    obj.overrideColorRGB.set(colors[color])


# quaternion ###############################################################

def quatToEuler(q):
    x, y, z, w = q.get()
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = math.degrees(math.atan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = math.degrees(math.atan2(t3, t4))

    return [X, Y, Z]


def quat(v, r, rad=True):
    v = dt.Vector(v)
    v.normalize()
    r = r if rad else math.radians(r)

    cos = math.cos(r * 0.5)
    sin = math.sin(r * 0.5)
    qv = sin * v

    return dt.Quaternion(qv.x, qv.y, qv.z, cos)
