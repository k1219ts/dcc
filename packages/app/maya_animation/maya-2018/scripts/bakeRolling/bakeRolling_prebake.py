import pymel.core as pm
import pymel.core.nodetypes as nt
import pymel.core.datatypes as dt

import math

import ch_cmn as cmn
import ch_rig as rig

def doit(rotateGrp, groundGrp, ctr, subCtr, cachedGrp, name, r):
    # get ground controllers
    groundCtrs = []
    for obj in groundGrp.getChildren():
        if isinstance(obj, nt.Transform):
            groundCtrs.append(obj)

    if not groundCtrs:
        cmn.confirmDialog('No ground controllers.')

    # delete all cache groups
    pm.delete(cachedGrp.getChildren())

    # create cache group
    cachingGrps = []
    groundMtxs  = []
    groundIvsMtxs = []
    for obj in groundCtrs:
        # create empty groups
        n = name + '_cached1'
        g = pm.group(em=True, name=n, parent=cachedGrp)
        cachingGrps.append(g)

        # add attribute for store quaternion
        for attr in 'XYZW':
            attr = 'outputQuat' + attr
            g.addAttr(attr, at='float')

        # set scale to default (1, 1, 1) before getting matrix
        orgScale = obj.scale.get()
        obj.scale.set(rig.ONEV)
        m = dt.Matrix(obj.getMatrix(ws=True))

        groundMtxs.append(m)
        groundIvsMtxs.append(m.inverse())

        # set scale to original
        obj.scale.set(orgScale)

    # get frames
    sframe = pm.playbackOptions(q=True, min=True)
    eframe = pm.playbackOptions(q=True, max=True)
    step   = pm.playbackOptions(q=True, by=True)

    # set attributes
    ctr.groundChange.setMax(len(cachingGrps))
    ctr.minFrame.set(sframe)
    ctr.maxFrame.set(eframe)
    ctr.stepFrame.set(step)

    # caching
    groundCnt = len(groundCtrs)
    prevPos = [None] * groundCnt
    up      = rig.YUP


    for t in range(int(sframe), int(eframe/step)+1):
        t = t * step

        for i in range(groundCnt):
            pos = getPos_onGround(rotateGrp, t, groundIvsMtxs[i])
            tpMtx = dt.Matrix(pm.getAttr(rotateGrp.parentInverseMatrix, t=t))
            tpMtx.translate = rig.ZEROV

            q = dt.Quaternion()
            if not prevPos[i]:
                prevPos[i] = pos
            else:
                dsv  = pos - prevPos[i]
                axis = up.cross(dsv)
                axis *= groundMtxs[i] * tpMtx
                axis.normalize()

                ds   = dsv.length()
                roll = ds / r

                prevPos[i] = pos
                q = rig.quat(axis, roll)

            cachingGrps[i].outputQuatX.setKey(v=q.x, t=t)
            cachingGrps[i].outputQuatY.setKey(v=q.y, t=t)
            cachingGrps[i].outputQuatZ.setKey(v=q.z, t=t)
            cachingGrps[i].outputQuatW.setKey(v=q.w, t=t)

    return cachingGrps


def quat(v, r, rad=True):
    v = dt.Vector(v)
    v.normalize()
    r = r if rad else math.radians(r)

    cos = math.cos(r * 0.5)
    sin = math.sin(r * 0.5)
    qv = sin * v

    return dt.Quaternion(qv.x, qv.y, qv.z, cos)


def getPos_onGround(obj, t, gMtx):
    mtx = dt.Matrix(pm.getAttr(obj.worldMatrix, t=t))
    mtx = mtx * gMtx

    return dt.Vector(mtx.translate)















#
