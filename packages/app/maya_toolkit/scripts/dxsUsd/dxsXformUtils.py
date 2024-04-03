import string, math

from pxr import Usd, UsdGeom, Vt, Gf
import maya.api.OpenMaya as OpenMaya
import maya.api.OpenMayaAnim as OpenMayaAnim
import maya.cmds as cmds

import dxsMayaUtils

def Get4x4MatrixByXformCmd(node, start, end, step=1.0):
    '''
    Returns:
        matrix (list): [[4x4 matrix], [...], [...]]
        frame  (list): [1.0, 1.25, 1.5, ...]
    '''
    mtxList = list()
    frmList = list()
    orgFrame = cmds.currentTime(q=True)

    # # retime bake
    # isRetime = False
    # if cmds.getAttr("time1.enableTimewarp"):
    #     isRetime = True
    #     cmds.currentTime(end)
    #     end = int(cmds.getAttr("time1.outTime")) + 1
    #     cmds.setAttr("time1.enableTimewarp", False)

    for f in range(start, end + 1):
        for s in dxsMayaUtils.GetFrameSample(step):
            frame = f + s

            cmds.currentTime(frame)

            mtxValue = cmds.xform(node, q=True, ws=True, m=True)

            mtxList.append(list(mtxValue))
            frmList.append(frame)

    # if isRetime:
    #     cmds.setAttr("time1.enableTimewarp", True)
    cmds.currentTime(orgFrame)
    return mtxList, frmList

def Get4x4Matrix(node, start, end, step=1.0, space='worldMatrix'):
    '''
    Returns:
        matrix (list): [[4x4 matrix], [...], [...]]
        frame  (list): [1.0, 1.25, 1.5, ...]
    '''
    mobj = dxsMayaUtils.GetMObject(node, dag=False)
    mfn  = OpenMaya.MFnDependencyNode(mobj)

    mtxAttr = mfn.attribute(space)
    mtxPlug = OpenMaya.MPlug(mobj, mtxAttr)
    if space == 'worldMatrix':
        mtxPlug = mtxPlug.elementByLogicalIndex(0)

    # time wrap
    timeWrap = cmds.listConnections('time1', d=False, s=True)

    mtxList = list()
    frmList = list()    # 2019.05.14 frame list change. maya api query wraped time, but usd write unwraped time
    for f in range(start, end+1):
        for s in dxsMayaUtils.GetFrameSample(step):
            frame = f + s
            wrapframe = frame
            if timeWrap:
                wrapframe = cmds.getAttr('%s.output' % timeWrap[0], time=frame)

            frameCtx = OpenMaya.MDGContext(OpenMaya.MTime(wrapframe, OpenMaya.MTime.uiUnit()))
            mtxObj   = mtxPlug.asMObject(frameCtx)
            mtxData  = OpenMaya.MFnMatrixData(mtxObj)
            mtxValue = mtxData.matrix()

            mtxList.append(list(mtxValue))
            frmList.append(frame)
    return mtxList, frmList


def GetMatrixByGf(position, orient, scale):
    tmtx = OpenMaya.MTransformationMatrix()
    tmtx.setScale([scale[0], scale[1], scale[2]], OpenMaya.MSpace.kWorld)

    img = orient.imaginary
    quat= OpenMaya.MQuaternion([img[0], img[1], img[2], orient.real])
    tmtx.setRotation(quat.asEulerRotation())

    tmtx.setTranslation(OpenMaya.MVector(*position), OpenMaya.MSpace.kWorld)
    return tmtx.asMatrix()

class GetMatrix:
    def __init__(self, node, fr=(0, 0), step=1.0, space='worldMatrix'):
        self.node = node
        self.fr   = fr
        self.step = step
        self.space= space

    def doIt(self):
        isConstant = GetMatrix.isConstant(self.node)
        if isConstant or self.fr[0] == self.fr[1]:
            return cmds.xform(self.node, q=True, m=True, ws=True), None
        else:
            return self.getData()

    def getData(self):
        mobj = dxsMayaUtils.GetMObject(self.node, dag=False)
        mfn  = OpenMaya.MFnDependencyNode(mobj)

        mtxAttr = mfn.attribute(self.space)
        mtxPlug = OpenMaya.MPlug(mobj, mtxAttr)
        if self.space == 'worldMatrix':
            mtxPlug = mtxPlug.elementByLogicalIndex(0)

        # time wrap
        timeWrap = cmds.listConnections('time1', d=False, s=True)

        mtxList = list()
        frmList = list()
        for f in range(self.fr[0], self.fr[1]+1):
            for s in dxsMayaUtils.GetFrameSample(self.step):
                frame = f + s
                if timeWrap:
                    frame = cmds.getAttr('%s.output' % timeWrap[0], time=frame)
                frameCtx = OpenMaya.MDGContext(OpenMaya.MTime(frame, OpenMaya.MTime.uiUnit()))
                mtxObj   = mtxPlug.asMObject(frameCtx)
                mtxData  = OpenMaya.MFnMatrixData(mtxObj)
                mtxValue = mtxData.matrix()
                mtxList.append(list(mtxValue))
                frmList.append(frame)
        return mtxList, frmList


    @staticmethod
    def isConstant(node):
        longPath = cmds.ls(node, l=True)[0]
        splitPath= longPath.split('|')
        for i in range(1, len(splitPath)):
            path = string.join(splitPath[:i+1], '|')
            animCurve = cmds.listConnections(path, type='animCurve', s=True, d=False, c=True)
            expression= cmds.listConnections(path, type='expression', s=True, d=False)
            if expression:
                return False
            if animCurve:
                for i in range(len(animCurve)/2):
                    attribute = animCurve[i*2]  # node.attr
                    attrname  = attribute.split('.')[-1]
                    values = list(set(cmds.keyframe(node, at=attrname, q=True, vc=True)))
                    angles = list(set(cmds.keyTangent(node, at=attrname, q=True, ia=True, oa=True)))
                    if len(values) > 1 or len(angles) > 1:
                        return False
        return True


def Set4x4Matrix(node, mtxList, frmList, space=OpenMaya.MSpace.kWorld):
    size = len(frmList)
    if size == 1:
        if space == 2:  # kObject
            cmds.xform(node, m=mtxList[0], os=True)
        else:
            cmds.xform(node, m=mtxList[0], ws=True)
    else:
        # key data check
        mtxvals = list()
        for i in xrange(size):
            val = mtxList[i]
            if not val in mtxvals:
                mtxvals.append(val)
        if len(mtxvals) == 1:
            if space == 2:
                cmds.xform(node, m=mtxList[0], os=True)
            else:
                cmds.xform(node, m=mtxList[0], ws=True)
            return

        dgmod = OpenMaya.MDGModifier()
        TL_list = ['translateX', 'translateY', 'translateZ', 'scaleX', 'scaleY', 'scaleZ']
        TA_list = ['rotateX', 'rotateY', 'rotateZ']

        objList = list()
        for i in TL_list:
            node_name = '%s_%s' % (node, i)
            if cmds.objExists(node_name):
                cmds.delete(node_name)
            obj = dgmod.createNode('animCurveTL')
            dgmod.renameNode(obj, node_name)
            objList.append(obj)
        for i in TA_list:
            node_name = '%s_%s' % (node, i)
            if cmds.objExists(node_name):
                cmds.delete(node_name)
            obj = dgmod.createNode('animCurveTA')
            dgmod.renameNode(obj, node_name)
            objList.append(obj)
        dgmod.doIt()

        keyObjList = list()
        for o in objList:
            obj = OpenMayaAnim.MFnAnimCurve()
            obj.setObject(o)
            keyObjList.append(obj)

        rotateOrder = cmds.getAttr('%s.rotateOrder' % node)

        for i in xrange(size):
            mtx  = OpenMaya.MMatrix(mtxList[i])
            tmtx = OpenMaya.MTransformationMatrix(mtx)
            mtime= OpenMaya.MTime(frmList[i], OpenMaya.MTime.uiUnit())

            tr = tmtx.translation(space)
            for x in range(3):
                keyObjList[x].addKey(mtime, tr[x])

            sc = tmtx.scale(space)
            for x in range(3):
                keyObjList[x+3].addKey(mtime, sc[x])

            ro = tmtx.rotation()
            ro.reorderIt(rotateOrder)
            for x in range(3):
                keyObjList[x+6].addKey(mtime, ro[x])

        curveNames = list()
        for i in TL_list:
            index = TL_list.index(i)
            mfn   = OpenMaya.MFnDependencyNode(objList[index])
            name  = mfn.name()
            curveNames.append(name)
            cmds.connectAttr('%s.output' % name, '%s.%s' % (node, i), f=True)
        for i in TA_list:
            index = TA_list.index(i)
            mfn   = OpenMaya.MFnDependencyNode(objList[index+6])
            name  = mfn.name()
            curveNames.append(name)
            cmds.connectAttr('%s.output' % name, '%s.%s' % (node, i), f=True)




class GetGfXform:
    '''
    Compute transformation for USD
    Args:
        matrix : 4x4 list
        transform : Translate(list), Scale(list), Rotate(list), Rotate Order
    Returns:
        postion (Gf.Vec3f):
        scale (Gf.Vec3f):
        orient (Gf.Quath or Gf.Vec4f): by scheme
    '''
    def __init__(self, matrix=None, transform=None):
        if matrix:
            self.asMatrix(matrix)
        elif transform:
            self.asTransform(transform)

    def asTransform(self, transform):
        self.translate = transform[0]
        self.scale = transform[1]
        self.rotate= OpenMaya.MEulerRotation(transform[2], transform[3]).asQuaternion()

    def asMatrix(self, matrix):
        tmx = OpenMaya.MTransformationMatrix(OpenMaya.MMatrix(matrix))
        self.translate = tmx.translation(OpenMaya.MSpace.kWorld)
        self.scale = tmx.scale(OpenMaya.MSpace.kWorld)
        self.rotate= tmx.rotation(asQuaternion=True)


    def Get(self, scheme='PointInstancer'):
        if scheme == 'PointInstancer':
            pos = Gf.Vec3f(*self.translate)
            scl = Gf.Vec3f(*self.scale)
            ort = Gf.Quath(self.rotate.w, self.rotate.x, self.rotate.y, self.rotate.z)
            return pos, scl, ort
        if scheme == 'Points':
            pos = Gf.Vec3f(*self.translate)
            scl = Gf.Vec3f(*self.scale)
            ort = Gf.Vec4f(self.rotate.x, self.rotate.y, self.rotate.z, self.rotate.w)
            return pos, scl, ort


class AddXformOp:
    def __init__(self, prim, matrix=None, frames=None):
        self.geom = UsdGeom.Xform(prim)
        if matrix:
            self.asMatrix(matrix, frames)

    def asMatrix(self, matrix, frames):
        if frames:
            xformAttr = self.geom.MakeMatrixXform()
            for i in range(len(frames)):
                xformAttr.Set(Gf.Matrix4d(*matrix[i]), Usd.TimeCode(frames[i]))
        else:
            if matrix != list(OpenMaya.MMatrix()):
                self.geom.MakeMatrixXform().Set(Gf.Matrix4d(*matrix))
