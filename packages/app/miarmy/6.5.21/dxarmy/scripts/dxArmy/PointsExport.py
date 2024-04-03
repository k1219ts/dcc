import os
import sys
import time

# for alembic python
from imath import *
from alembic.AbcCoreAbstract import *
from alembic.Abc import *
from alembic.AbcGeom import *
from alembic.Util import *
kWrapExisting = WrapExistingFlag.kWrapExisting

import maya.api.OpenMaya as OpenMaya

import maya.cmds as cmds
import maya.mel as mel


_timeUnitMap = {'game':15.0, 'film':24.0, 'pal':25.0,
                'ntsc':30.0, 'show':48.0, 'palf':50.0, 'ntscf':60.0}


class PointsExport:
    '''
    McdAgent alembic points export
    '''
    def __init__(self, filename, nodes, name, start=None, end=None):
        self.m_filename = filename
        self.m_nodes = nodes
        self.m_size  = len(self.m_nodes)
        self.m_name  = name

        self.m_start = start
        self.m_end   = end

        self.getFrameRange()

    def getFrameRange(self):
        if not self.m_start:
            self.m_start = int(cmds.playbackOptions(q=True, min=True))
        if not self.m_end:
            self.m_end = int(cmds.playbackOptions(q=True, max=True))

    def doIt(self):
        startTime = time.time()

        if not os.path.exists(os.path.dirname(self.m_filename)):
            os.makedirs(os.path.dirname(self.m_filename))
        if os.path.exists(self.m_filename):
            os.remove(self.m_filename)

        self.Selection = OpenMaya.MSelectionList()
        for n in self.m_nodes:
            self.Selection.add(n)

        tunit = cmds.currentUnit(t=True, q=True)
        timeSamp = TimeSampling(
            1.0 / _timeUnitMap[tunit] * 1.0, self.m_start / _timeUnitMap[tunit]
        )

        oarch = OArchive(str(self.m_filename), asOgawa=True)
        root = oarch.getTop()
        obj  = OPoints(root, str(self.m_name), timeSamp)
        schema = obj.getSchema()
        arb = schema.getArbGeomParams()
        scaleparam  = OFloatGeomParam(arb, 'scale', False, GeometryScope.kVaryingScope, 3, timeSamp)
        orientparam = OQuatfGeomParam(arb, 'orient', False, GeometryScope.kVaryingScope, 1, timeSamp)

        for f in xrange(self.m_start, self.m_end+1):
            cmds.currentTime(f)
            self.setFrame(schema, scaleparam, orientparam, f)

        endTime = time.time()
        # debug
        print '# Miarmy PointsExport : %s, %s' % (self.m_filename, endTime-startTime)

    def setFrame(self, PointSchema, ScaleParam, OrientParam, frame):
        positions = V3fArray(self.m_size)
        ids       = IntArray(self.m_size)
        scales    = FloatArray(self.m_size * 3)
        orients   = QuatfArray(self.m_size)

        i = 0
        itSelection = OpenMaya.MItSelectionList(self.Selection)
        while not itSelection.isDone():
            mdag = itSelection.getDagPath()
            mobj = mdag.transform()
            mfn = OpenMaya.MFnDependencyNode(mobj)

            mtxAttr = mfn.attribute('worldMatrix')
            mtxPlug = OpenMaya.MPlug(mobj, mtxAttr)
            mtxPlug = mtxPlug.elementByLogicalIndex(0)
            mtxObj  = mtxPlug.asMObject()
            mtxData = OpenMaya.MFnMatrixData(mtxObj)
            transMtx= mtxData.transformation()

            tr = transMtx.translation(OpenMaya.MSpace.kWorld)
            positions[i] = V3f(tr.x, tr.y, tr.z)

            sc = transMtx.scale(OpenMaya.MSpace.kWorld)
            for x in xrange(3):
                scales[i*3+x] = sc[x]

            ro = transMtx.rotation(asQuaternion=True)
            orients[i] = Quatf(ro.x, ro.y, ro.z, ro.w)

            ids[i] = i

            i += 1
            itSelection.next()

        schemaSamp = OPointsSchemaSample()
        schemaSamp.setPositions(positions)
        schemaSamp.setIds(ids)

        scalesSamp = OFloatGeomParamSample()
        scalesSamp.setScope(GeometryScope.kVertexScope)
        scalesSamp.setVals(scales)

        orientsSamp = OQuatfGeomParamSample()
        orientsSamp.setScope(GeometryScope.kVertexScope)
        orientsSamp.setVals(orients)

        PointSchema.set(schemaSamp)
        ScaleParam.set(scalesSamp)
        OrientParam.set(orientsSamp)



def AgentPointsExport(filename, start=None, end=None):
    nodes = cmds.ls(type='McdAgent')
    exp = PointsExport(filename, nodes, 'McdAgents', start, end)
    exp.doIt()
