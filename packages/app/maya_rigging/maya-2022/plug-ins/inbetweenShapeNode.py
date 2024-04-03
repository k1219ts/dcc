import sys

import maya.api.OpenMaya as OpenMaya
from maya.api.OpenMaya import (MPxNode, MFnPlugin, MTypeId, MObject,
                               MFnAttribute, MFnNumericAttribute, MFnNumericData,
                               MFnCompoundAttribute)


def maya_useNewAPI():
    pass


kPluginNodeName = "pyInbetweenShape"
kPluginNodeClassify = "utility/general"
kPluginNodeId = MTypeId(0x10400000)


class InbetweenNode(MPxNode):

    inputAttr = MObject()

    minWeightAttr = MObject()
    maxWeightAttr = MObject()
    inbetweenAttr = MObject()
    inputWeightAttr = MObject()

    outputAttr = MObject()

    minDefault = 0.0
    maxDefault = 1.0
    inbetweenDefault = 0.5
    inputDefault = 0.5
    outputDefaultVal = 0.5

    def __init__(self):
        MPxNode.__init__(self)

    @staticmethod
    def _get_curve(w0, w1, f):
        x = (f - w0) / (w1 - w0)
        return x * x * (-2.0 * x + 3)

    def compute(self, plug, data_block):
        if plug == InbetweenNode.outputAttr:
            input_array = data_block.inputArrayValue(self.inputAttr)
            out_val = 1.0
            while not input_array.isDone():
                input_element = input_array.inputValue()
                min_val = input_element.child(self.minWeightAttr).asFloat()
                max_val = input_element.child(self.maxWeightAttr).asFloat()
                input_val = input_element.child(self.inputWeightAttr).asFloat()
                if min_val <= input_val <= max_val:
                    btw_val = input_element.child(self.inbetweenAttr).asFloat()
                    if input_val != btw_val:
                        w0 = min_val if input_val < btw_val else max_val
                        out_val *= self._get_curve(w0, btw_val, input_val)
                    input_array.next()
                else:
                    out_val = 0.0
                    break

            output_handle = data_block.outputValue(self.outputAttr)
            output_handle.setFloat(out_val)
            data_block.setClean(plug)


def nodeCreator():
    return InbetweenNode()


def nodeInitializer():
    num_attr = MFnNumericAttribute()
    comp_attr = MFnCompoundAttribute()

    # input attribute
    InbetweenNode.minWeightAttr = num_attr.create("minWeight", "min",
                                                  MFnNumericData.kFloat, InbetweenNode.minDefault)
    InbetweenNode.maxWeightAttr = num_attr.create("maxWeight", "max",
                                                  MFnNumericData.kFloat, InbetweenNode.maxDefault)
    InbetweenNode.inbetweenAttr = num_attr.create("inbetweenWeight", "btw",
                                                  MFnNumericData.kFloat, InbetweenNode.inbetweenDefault)
    InbetweenNode.inputWeightAttr = num_attr.create("inputWeight", 'i',
                                                    MFnNumericData.kFloat, InbetweenNode.inputDefault)
    num_attr.storable = True
    num_attr.keyable = True
    # num_attr.internal = True

    InbetweenNode.inputAttr = comp_attr.create("input", "in")
    comp_attr.addChild(InbetweenNode.minWeightAttr)
    comp_attr.addChild(InbetweenNode.maxWeightAttr)
    comp_attr.addChild(InbetweenNode.inbetweenAttr)
    comp_attr.addChild(InbetweenNode.inputWeightAttr)
    comp_attr.array = True
    comp_attr.hidden = False
    comp_attr.disconnectBehavior = MFnAttribute.kDelete

    # output weight attribute
    InbetweenNode.outputAttr = num_attr.create("outputWeight", 'o',
                                               MFnNumericData.kFloat, InbetweenNode.outputDefaultVal)
    num_attr.storable = False
    num_attr.writable = False
    num_attr.readable = True
    num_attr.hidden = False

    # add attributes
    InbetweenNode.addAttribute(InbetweenNode.inputAttr)
    InbetweenNode.addAttribute(InbetweenNode.outputAttr)

    # node attribute dependencies
    InbetweenNode.attributeAffects(InbetweenNode.minWeightAttr, InbetweenNode.outputAttr)
    InbetweenNode.attributeAffects(InbetweenNode.maxWeightAttr, InbetweenNode.outputAttr)
    InbetweenNode.attributeAffects(InbetweenNode.inbetweenAttr, InbetweenNode.outputAttr)
    InbetweenNode.attributeAffects(InbetweenNode.inputWeightAttr, InbetweenNode.outputAttr)
    InbetweenNode.attributeAffects(InbetweenNode.inputAttr, InbetweenNode.outputAttr)


def initializePlugin(mobject):
    plugin = MFnPlugin(mobject)
    try:
        plugin.registerNode(kPluginNodeName,
                            kPluginNodeId,
                            nodeCreator,
                            nodeInitializer,
                            MPxNode.kDependNode,
                            kPluginNodeClassify)
    except:
        sys.stderr.write("Failed to register node: {}".format(kPluginNodeName))
        raise


def uninitializePlugin(mobject):
    plugin = MFnPlugin(mobject)
    try:
        plugin.deregisterNode(kPluginNodeId)
    except:
        sys.stderr.write("Failed to deregister node: {}".format(kPluginNodeName))
        raise
