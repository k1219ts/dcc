import sys
try:
    from typing import List, Set, Dict, Tuple, Text, Optional, Union
except ImportError:
    pass

from maya.api.OpenMaya import (MPxCommand, MArgParser, MSyntax,
                               MGlobal, MObject, MVector, MPlug, MPointArray, MIntArray,
                               MFn, MFnPlugin, MFnAttribute, MFnDependencyNode, MFnPointArrayData,
                               MFnCompoundAttribute, MFnSingleIndexedComponent, MFnComponentListData,
                               MItMeshVertex, MItMeshFaceVertex)


def maya_useNewAPI():
    """
	The presence of this function tells Maya that the plugin produces, and
	expects to be passed, objects created using the Maya Python API 2.0.
	"""
    pass


class setPointsTarget(MPxCommand):
    NAME = "setPointsTarget"
    TARGET_FLAG = "-t"
    TARGET_FLAG_LONG = "-target"
    POINT_FLAG = "-p"
    POINT_FLAG_LONG = "-points"
    COMPONENT_FLAG = "-c"
    COMPONENT_FLAG_LONG = "-components"

    def __init__(self):
        MPxCommand.__init__(self)

        self.plugPoints = None      # type: Optional[MPlug]
        self.oldPointData = None    # type: Optional[MObject]
        self.pointData = None       # type: Optional[MObject]

        self.plugComponents = None      # type: Optional[MPlug]
        self.oldComponentData = None    # type: Optional[MObject]
        self.componentData = None       # type: Optional[MObject]

        self.done = False

    def doIt(self, args):
        path, base, target, index, points, components = self.parse_arguments(args)

        selection = MGlobal.getSelectionListByName(path)
        if selection.isEmpty():
            sys.stderr.write("{} is not found.".format(path))

        blendshape = selection.getDependNode(0)
        if not blendshape.hasFn(MFn.kBlendShape):
            sys.stderr.write("{} is not a blendShape node.".format(path))

        fn = MFnDependencyNode(blendshape)
        plug_base = fn.findPlug("inputTarget", False).elementByLogicalIndex(base)
        plug_group = plug_base.child(fn.attribute("inputTargetGroup"))
        plug_target = plug_group.elementByLogicalIndex(target)
        plug_item = plug_target.child(fn.attribute("inputTargetItem"))
        plug_inbetween = plug_item.elementByLogicalIndex(index)

        self.plugPoints = plug_inbetween.child(fn.attribute("inputPointsTarget"))
        if self.plugPoints.isDefaultValue(forceEval=True):
            fn_data = MFnPointArrayData()
            self.oldPointData = fn_data.create()
        else:
            self.oldPointData = self.plugPoints.asMObject()
            # fn_points_data = MFnPointArrayData(self.plugPoints.asMObject())
            # fn_points_data.copyTo(self.oldPointData)
        fn_data = MFnPointArrayData()
        self.pointData = fn_data.create()
        fn_data.set(points)

        self.plugComponents = plug_inbetween.child(fn.attribute("inputComponentsTarget"))
        if self.plugComponents.isDefaultValue(forceEval=True):
            fn_vert = MFnSingleIndexedComponent()
            fn_data = MFnComponentListData(fn_vert.create(MFn.kMeshVertComponent))
            self.oldComponentData = fn_data.create()
        else:
            self.oldComponentData = self.plugComponents.asMObject()
            # fn_data = MFnComponentListData(self.plugComponents.asMObject())
            # if fn_data.length():
            #     self.oldComponentData = MFnSingleIndexedComponent(fn_data.get(0)).getElements()
        fn_vert = MFnSingleIndexedComponent()
        obj_vert = fn_vert.create(MFn.kMeshVertComponent)
        if components:
            fn_vert.addElements(components)
        fn_data = MFnComponentListData()
        self.componentData = fn_data.create()
        fn_data.add(obj_vert)

        self.redoIt()

    def parse_arguments(self, args):
        args = MArgParser(self.syntax(), args)
        path = args.commandArgumentString(0)

        if not args.isFlagSet(self.TARGET_FLAG):
            sys.stderr.write("target parameter is not found.")
        # if args.numberOfFlagUses(self.TARGET_FLAG) != 3:
        #     sys.stderr.write("target parameter: [base, target, index]")
        base = args.flagArgumentInt(self.TARGET_FLAG, 0)
        target = args.flagArgumentInt(self.TARGET_FLAG, 1)
        index = args.flagArgumentInt(self.TARGET_FLAG, 2)

        points = MPointArray()
        if args.isFlagSet(self.POINT_FLAG):
            for i in range(args.numberOfFlagUses(self.POINT_FLAG)):
                flag_args = args.getFlagArgumentList(self.POINT_FLAG, i)
                points.append(flag_args.asPoint(0))

        components = None
        if args.isFlagSet(self.COMPONENT_FLAG):
            n = args.numberOfFlagUses(self.COMPONENT_FLAG)
            if n != len(points):
                sys.stderr.write("each count of points and components does not match.")
            components = MIntArray([args.getFlagArgumentList(self.COMPONENT_FLAG, i).asInt(0)
                                    for i in range(n)])

        return path, base, target, index, points, components

    def redoIt(self):
        self.plugPoints.isDefaultValue(forceEval=True)
        self.plugPoints.setMObject(self.pointData)
        self.plugComponents.isDefaultValue(forceEval=True)
        self.plugComponents.setMObject(self.componentData)
        self.done = True

    def undoIt(self):
        self.plugPoints.isDefaultValue(forceEval=True)
        self.plugPoints.setMObject(self.oldPointData)
        self.plugComponents.isDefaultValue(forceEval=True)
        self.plugComponents.setMObject(self.oldComponentData)

    def isUndoable(self):
        return self.done

    @classmethod
    def syntax_creator(cls):
        syntax = MSyntax()
        syntax.addArg(MSyntax.kString)

        syntax.addFlag(cls.TARGET_FLAG, cls.TARGET_FLAG_LONG,
                       [MSyntax.kUnsigned, MSyntax.kUnsigned, MSyntax.kUnsigned])

        syntax.addFlag(cls.POINT_FLAG, cls.POINT_FLAG_LONG,
                       [MSyntax.kDouble, MSyntax.kDouble, MSyntax.kDouble])
        syntax.makeFlagMultiUse(cls.POINT_FLAG)

        syntax.addFlag(cls.COMPONENT_FLAG, cls.COMPONENT_FLAG_LONG, MSyntax.kUnsigned)
        syntax.makeFlagMultiUse(cls.COMPONENT_FLAG)

        return syntax


# Plug-in initialization
def initializePlugin(mobject):
    mplugin = MFnPlugin(mobject)
    try:
        mplugin.registerCommand(setPointsTarget.NAME, setPointsTarget,
                                setPointsTarget.syntax_creator)
    except:
        sys.stderr.write("Failed to register {}".format(setPointsTarget.NAME))


def uninitializePlugin(mobject):
    mplugin = MFnPlugin(mobject)
    try:
        mplugin.deregisterCommand(setPointsTarget.NAME)
    except:
        sys.stderr.write("Failed to unregister {}".format(setPointsTarget.NAME))
