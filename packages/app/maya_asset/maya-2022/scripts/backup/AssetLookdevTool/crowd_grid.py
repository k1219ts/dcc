import os
import sys
import string
import random
import math

import maya.api.OpenMaya as OpenMaya

from alembic.AbcCoreAbstract import *
from alembic.Abc import *
from alembic.AbcGeom import *
from alembic.Util import *

import maya.cmds as cmds

current = os.path.abspath(__file__)
currentDir = os.path.dirname(current)

gridAbcPath = {'400': os.path.join(currentDir, 'data', "crowd_grid.abc"),
               '2000': os.path.join(currentDir, 'data', "output_2000.abc")}


def AgentLookDev():
    ldvNode = cmds.ls(type='dxLdvNode')[0]
    root = cmds.listRelatives(ldvNode, p = True)[0]

    if not cmds.objExists(root):
        root = cmds.group(n=root, em=True)

    agentPoint = 'AgentPointsShape'
    if not cmds.objExists(agentPoint):
        pointsShape = cmds.createNode('ZAbcPtcViewer', n = agentPoint, ss=True)
        cmds.setAttr('%s.displayColor' % pointsShape, 1, 0, 0, type='double3')
        cmds.setAttr('%s.pointSize' % pointsShape, 3)
        cmds.setAttr('%s.file' % pointsShape, gridAbcPath['2000'], type='string')
        cmds.parent(cmds.listRelatives(pointsShape, p=True), root)

    source = 'AgentSource_GRP'
    if not cmds.objExists(source):
        source = cmds.group(n=source, p=root, em=True)

    cmds.select(cl=True)

    return agentPoint, source


def AgentGridSetup(pointsShape = None, sourceGRP = None, *args):
    try:
        if pointsShape == None:
            pointsShape = cmds.ls('AgentPointsShape')
            pointsShape = pointsShape[0]
        if sourceGRP == None:
            sourceGRP   = cmds.ls('AgentSource_GRP')
            sourceGRP = sourceGRP[0]
    except Exception as e:
        cmds.error(e.message)
        pointsShape = None
        sourceGRP = None

    if not pointsShape or not sourceGRP:
        return

    abcfile = cmds.getAttr('%s.file' % pointsShape)
    source  = cmds.listRelatives(sourceGRP, c=True, f=True)

    # print abcfile, source
    crowd = CrowdGrid(abcfile, source)
    crowd.initScale = cmds.getAttr('AgentPoints.scale')[0]
    if args:
        crowd.maxCount = args[0]
    crowd.doIt()

    cmds.select(cl=True)


class CrowdGrid:
    def __init__(self, abcfile, source):
        self.abcfile = abcfile
        self.source  = source
        self.initScale = (1, 1, 1)
        self.maxCount = -1

    def doIt(self):
        if self.abcfile and len(self.source) > 0:
            self.readAbc(self.abcfile)

    def readAbc(self, filename):
        iarch = IArchive(str(filename))
        root = iarch.getTop()
        self.visitObject(root)

    def visitObject(self, iobj):
        for obj in iobj.children:
            ohead = obj.getHeader()
            if IPoints.matches(ohead):
                self.visitPoints(obj)
            self.visitObject(obj)

    def visitPoints(self, iobj):
        icompoundProp = iobj.getProperties()
        prop = icompoundProp.getProperty(0)

        # .pointsIds
        idProp = prop.getProperty('.pointIds')
        ids = idProp.getValue()

        geomprop = prop.getProperty('.arbGeomParams')

        pointsProp = prop.getProperty('P')
        scaleProp  = geomprop.getProperty('scale')
        orientProp = geomprop.getProperty('orient')

        P = pointsProp.getValue()
        Scale = scaleProp.getValue()
        Orient = orientProp.getValue()

        if self.maxCount < 0:
            self.maxCount = len(ids)

        if cmds.objExists("Agents"):
            cmds.delete("Agents")
        for i in xrange(self.maxCount):
            trans = P[i]
            scale = Scale[i*3:i*3+3]

            orient = Orient[i]
            orient = [orient.r()] + list(orient.v())
            quat   = OpenMaya.MQuaternion(orient)
            rotate = quat.asEulerRotation()

            self.createAgent(trans, scale, rotate, i)


    def createAgent(self, trans, scale, rotate, id):
        # print 'trans :', trans[0], trans[1], trans[2]
        # print 'scale :', scale[0], scale[1], scale[2]
        # print 'rotate :', rotate
        root = 'Agents'
        if not cmds.objExists(root):
            root = cmds.group(n=root, em=True)

        target_index = self.getSource(id)
        new = cmds.duplicate(self.source[target_index], rr=True, ic=True)

        cmds.setAttr('%s.tx' % new[0], trans[0]*self.initScale[0])
        cmds.setAttr('%s.tz' % new[0], trans[2]*self.initScale[2])
        cmds.setAttr('%s.scale' % new[0], scale[0], scale[1], scale[2], type='double3')
        cmds.setAttr('%s.ry' % new[0], math.degrees(rotate[1]))

        cmds.setAttr('%s.objectid' % new[0], 0)
        cmds.setAttr('%s.groupid' % new[0], 0)
        cmds.setAttr('%s.primid' % new[0], id+1)

        cmds.parent(new, root)


    def getSource(self, index):
        random.seed(index)
        return random.randrange(0, len(self.source))
