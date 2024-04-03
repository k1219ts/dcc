import maya.OpenMayaMPx as OpenMayaMPx
import maya.OpenMaya as OpenMaya
import maya.cmds as cmds

'''
# blendshape
import pymel.core as pm
import pymel.core.nodetypes as nt

sels = pm.ls(sl=True)
base = sels.pop(-1)
dfm  = pm.deformer(base, type=nt.FexBlendShape)[0]

for i, sel in enumerate(sels):
    pm.blendShape(dfm, t=(base, i, sel, 1.0), e=True)
'''

class fexBlendShape(OpenMayaMPx.MPxBlendShape):
    kPluginNodeId = OpenMaya.MTypeId(0x00000002)

    def __init__(self):
        OpenMayaMPx.MPxBlendShape.__init__(self)

    def deformData(self, block, geomData, groupId, m, multiIndex):
        '''
        Description:   Deforms the point with a simple smooth skinning algorithm
        Arguments:
           block      : the datablock of the node
           geomData   : a handle to the geometry to be deformed
           groupId    : the group ID of the geometry to deform
           m          : matrix to transform the point into world space
           multiIndex : the index of the geometry that we are deforming
        '''
        weightMH   = block.inputArrayValue(self.weight)
        numWeights = weightMH.elementCount()
        weights    = OpenMaya.MFloatArray()

        for w in range(numWeights):
            weightMH.jumpToElement(w)
            weights.append(weightMH.inputValue().asFloat())

        inputTargetMH = block.inputArrayValue(self.inputTarget)
        try:
            inputTargetMH.jumpToElement(multiIndex)
        except:
            return

        inputTargetH = inputTargetMH.inputValue()
        inputTargetGroupMH = OpenMaya.MArrayDataHandle(inputTargetH.child(self.inputTargetGroup))

        offsets = {}

        for w in range(numWeights):
            # inputPointsTarget is computed on pull,
            # so can't just read it out of the datablock
            plug = OpenMaya.MPlug(self.thisMObject(), self.inputPointsTarget)
            plug.selectAncestorLogicalIndex(multiIndex, self.inputTarget)
            plug.selectAncestorLogicalIndex(w, self.inputTargetGroup)
            # ignore deformer chains here and just take the first one
            plug.selectAncestorLogicalIndex(6000, self.inputTargetItem)
            pointArray = plug.asMObject()
            pts = OpenMaya.MFnPointArrayData(pointArray).array()
            # get the component list
            plug = plug.parent()
            plug = plug.child(self.inputComponentsTarget)

            compList = OpenMaya.MFnComponentListData(plug.asMObject())

            if not compList.length():
                continue

            # iterate over the components
            defWgt = weights[w]
            inputTargetGroupMH.jumpToArrayElement(w)
            targetWeightsMH = OpenMaya.MArrayDataHandle(inputTargetGroupMH.inputValue().child(self.targetWeights))

            ptIndex = 0
            itGeo = OpenMaya.MItGeometry(geomData, compList[0], False)
            while not itGeo.isDone():
                compIndex = itGeo.index()
                wgt       = defWgt

                try:
                    targetWeightsMH.jumpToElement(compIndex)
                    wgt *= targetWeightsMH.inputValue().asFloat()
                except:
                    pass

                A = OpenMaya.MVector(pts[ptIndex] * wgt)
                _A = A.length()
                if wgt == 0 or _A == 0:
                    ptIndex += 1
                    itGeo.next()
                    continue
                elif not offsets.has_key(compIndex):
                    offsets[compIndex] = A
                else:
                    B  = offsets[compIndex]
                    C  = A + B
                    _C = C.length()
                    if _C == 0:
                        offsets.pop(compIndex)
                    else:
                        _B  = B.length()
                        if _A > _B:
                            big = A
                            _big = _A
                        else:
                            big = B
                            _big = _B

                        _D   = big.x*C.x + big.y*C.y + big.z*C.z
                        _cos = _D / (_big*_C)

                        if _cos == 1 and _big > _C:
                            offsets[compIndex] = OpenMaya.MVector(C)
                        elif _cos == 1 and _big < _C:
                            offsets[compIndex] = OpenMaya.MVector(big)
                        else:
                            _D /= _C*_C
                            offsets[compIndex] = OpenMaya.MVector(_D*C.x, _D*C.y, _D*C.z)

                ptIndex += 1
                itGeo.next()

        for w in range(numWeights):
            # inputPointsTarget is computed on pull,
            # so can't just read it out of the datablock
            plug = OpenMaya.MPlug(self.thisMObject(), self.inputPointsTarget)
            plug.selectAncestorLogicalIndex(multiIndex, self.inputTarget)
            plug.selectAncestorLogicalIndex(w, self.inputTargetGroup)
            # ignore deformer chains here and just take the first one
            plug.selectAncestorLogicalIndex(6000, self.inputTargetItem)
            # get the component list
            plug = plug.parent()
            plug = plug.child(self.inputComponentsTarget)

            compList = OpenMaya.MFnComponentListData(plug.asMObject())
            if not compList.length():
                continue

            itGeo = OpenMaya.MItGeometry(geomData, compList[0], False)
            while not itGeo.isDone():
                compIndex = itGeo.index()
                if compIndex not in offsets.keys():
                    itGeo.next()
                    continue

                pt     = OpenMaya.MPoint(itGeo.position())
                itGeo.setPosition(pt + offsets.pop(compIndex))
                itGeo.next()


def creator():
    return OpenMayaMPx.asMPxPtr(fexBlendShape())


def initialize():
    pass


def initializePlugin(obj):
    plugin = OpenMayaMPx.MFnPlugin(obj, 'Chad Vernon', '1.0', 'Any')
    try:
        plugin.registerNode(
            'fexBlendShape',
            fexBlendShape.kPluginNodeId,
            creator,
            initialize,
            OpenMayaMPx.MPxNode.kBlendShape)
    except:
        raise RuntimeError('Failed to register node')


def uninitializePlugin(obj):
    plugin = OpenMayaMPx.MFnPlugin(obj)
    try:
        plugin.deregisterNode(BlendNode.kPluginNodeId)
    except:
        raise RuntimeError('Failed to deregister node')
