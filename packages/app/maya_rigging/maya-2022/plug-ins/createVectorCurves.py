import sys
try:
    from typing import Optional
except ImportError:
    pass

from maya.api.OpenMaya import (MArgDatabase, MDagPath, MPxCommand, MArgParser, MSpace, MSyntax,
                               MGlobal, MObject, MPlug, MDGModifier, MPoint, MVector,
                               MIntArray, MPointArray, MVectorArray, MSelectionList,
                               MFn, MFnPlugin, MFnNurbsCurve, MFnTransform,
                               MFnSingleIndexedComponent, MFnComponentListData,
                               MItMeshVertex)


def maya_useNewAPI():
    """
    The presence of this function tells Maya that the plugin produces, and
    expects to be passed, objects created using the Maya Python API 2.0.
    """
    pass


class createVectorCurves(MPxCommand):
    NAME = "createVectorCurves"
    NAME_FLAG = "-n"
    NAME_FLAG_LONG = "-name"
    VECTOR_FLAG = "-v"
    VECTOR_FLAG_LONG = "-vectors"
    VERT_ID_FLAG = "-ids"
    VERT_ID_FLAG_LONG = "-vertexids"

    def __init__(self):
        MPxCommand.__init__(self)

        self.selection = None   # type: Optional[MSelectionList]
        self.name = None        # type: Optional[str]
        self.vertexIds = MIntArray()
        self.vectors = MVectorArray()
        self.matrix = None
        self.pathMesh = None
        self.points = MPointArray()
        self.transform = None

        self.done = False

    @classmethod
    def syntax_creator(cls):
        syntax = MSyntax()
        syntax.addArg(MSyntax.kString)
        syntax.setMaxObjects(1)

        syntax.addFlag(cls.NAME_FLAG, cls.NAME_FLAG_LONG, MSyntax.kString)

        syntax.addFlag(cls.VERT_ID_FLAG, cls.VERT_ID_FLAG_LONG, MSyntax.kUnsigned)
        syntax.makeFlagMultiUse(cls.VERT_ID_FLAG)

        syntax.addFlag(cls.VECTOR_FLAG, cls.VECTOR_FLAG_LONG,
                       [MSyntax.kDouble, MSyntax.kDouble, MSyntax.kDouble])
        syntax.makeFlagMultiUse(cls.VECTOR_FLAG)

        return syntax

    def parse_arguments(self, args):
        data = MArgDatabase(self.syntax(), args)
        self.selection = data.commandArgumentMSelectionList(0)
        self.name = data.flagArgumentString(self.NAME_FLAG, 0)
        for i in range(data.numberOfFlagUses(self.VERT_ID_FLAG)):
            flag_args = data.getFlagArgumentList(self.VERT_ID_FLAG, i)
            self.vertexIds.append(flag_args.asInt(0))

        for i in range(data.numberOfFlagUses(self.VECTOR_FLAG)):
            flag_args = data.getFlagArgumentList(self.VECTOR_FLAG, i)
            self.vectors.append(flag_args.asVector(0))

    def setup_mesh(self):
        path = self.selection.getDagPath(0)
        if path.apiType() == MFn.kTransform:
            self.pathMesh = MDagPath(path)
            self.pathMesh.extendToShape()
        elif path.apiType() == MFn.kMesh:
            self.pathMesh = path
            path = MDagPath.getAPathTo(path.transform())
        else:
            sys.stderr.write("no mesh")

        fn_trans = MFnTransform(path)
        self.matrix = fn_trans.transformation()

    def doIt(self, args):
        self.parse_arguments(args)
        self.setup_mesh()

        fn_ids = MFnSingleIndexedComponent()
        obj_ids = fn_ids.create(MFn.kMeshVertComponent)
        if len(self.vertexIds):
            fn_ids.addElements(self.vertexIds)
        else:
            fn_ids.setCompleteData()
            it_vert = MItMeshVertex(self.pathMesh)

        it_vert = MItMeshVertex(self.pathMesh, obj_ids)
        while not it_vert.isDone():
            pos = it_vert.position()
            self.points.append(pos)
            it_vert.next()

        self.redoIt()

    def create_curve(self, i):
        fn_trans = MFnTransform()
        obj_trans = fn_trans.create(self.transform)
        fn_trans.setName("{}{}".format(self.name, i))
        fn_curve = MFnNurbsCurve()
        points = [MPoint(), self.vectors[i]]
        fn_curve.createWithEditPoints(points, 1, MFnNurbsCurve.kOpen,
                                      False, False, False, obj_trans)
        fn_curve.setName("{}Shape{}".format(self.name, i))
        fn_trans.setTranslation(MVector(self.points[i]), MSpace.kObject)

    def redoIt(self):
        fn_trans = MFnTransform()
        self.transform = fn_trans.create()
        fn_trans.setName(self.name)

        for i in range(len(self.points)):
            self.create_curve(i)

        fn_trans.setTransformation(self.matrix)

        path = MDagPath.getAPathTo(self.transform)
        self.clearResult()
        self.setResult(path.partialPathName())

        self.done = True

    def undoIt(self):
        mod = MDGModifier()
        mod.deleteNode(self.transform)
        # self.done = False

    def isUndoable(self):
        return self.done


# Plug-in initialization
def initializePlugin(mobject):
    mplugin = MFnPlugin(mobject)
    try:
        mplugin.registerCommand(createVectorCurves.NAME, createVectorCurves,
                                createVectorCurves.syntax_creator)
    except RuntimeError:
        sys.stderr.write("Failed to register {}".format(createVectorCurves.NAME))


def uninitializePlugin(mobject):
    mplugin = MFnPlugin(mobject)
    try:
        mplugin.deregisterCommand(createVectorCurves.NAME)
    except RuntimeError:
        sys.stderr.write("Failed to unregister {}".format(createVectorCurves.NAME))
