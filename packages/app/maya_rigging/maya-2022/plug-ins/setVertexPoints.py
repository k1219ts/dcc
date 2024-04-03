import sys

from maya import cmds
from maya.api.OpenMaya import (MFnPlugin, MPxCommand, MArgParser, MSyntax,
                               MGlobal, MPointArray, MVectorArray, MIntArray,
                               MFn, MFnSingleIndexedComponent,
                               MItMeshVertex)


def maya_useNewAPI():
    """
	The presence of this function tells Maya that the plugin produces, and
	expects to be passed, objects created using the Maya Python API 2.0.
	"""
    pass


class setVertexPoints(MPxCommand):
    NAME = "setVertexPoints"
    POINT_FLAG = "-p"
    POINT_FLAG_LONG = "-points"
    VERTEX_ID_FLAG = "-v"
    VERTEX_ID_FLAG_LONG = "-vertices"

    def __init__(self):
        MPxCommand.__init__(self)

        self.it_vert = None

        self.old_points = MPointArray()
        self.vectors = None

        self.done = False

    def doIt(self, args):
        path, vectors, vertices = self.parse_arguments(args)

        selection_list = MGlobal.getSelectionListByName(path)
        if selection_list.isEmpty():
            sys.stderr.write("{} is not found.".format(path))

        dag = selection_list.getDagPath(0)
        dag_type = dag.apiType()
        if dag_type == MFn.kTransform:
            dag = dag.extendToShape()
            dag_type = dag.apiType()
        if dag_type != MFn.kMesh:
            sys.stderr.write("{} is not a mesh.".format(path))

        fn = MFnSingleIndexedComponent()
        obj = fn.create(MFn.kMeshVertComponent)
        fn.addElements(vertices)

        old_points = self.old_points
        self.it_vert = it_vert = MItMeshVertex(dag, obj)
        while not it_vert.isDone():
            old_points.append(it_vert.position())
            it_vert.next()

        self.vectors = vectors
        self.redoIt()

    def parse_arguments(self, args):
        args = MArgParser(self.syntax(), args)
        path = args.commandArgumentString(0)

        vectors = MVectorArray()
        vertices = MIntArray()
        if args.isFlagSet(self.POINT_FLAG):
            for i in range(args.numberOfFlagUses(self.POINT_FLAG)):
                flag_args = args.getFlagArgumentList(self.POINT_FLAG, i)
                vectors.append(flag_args.asVector(0))
        if args.isFlagSet(self.VERTEX_ID_FLAG):
            for i in range(args.numberOfFlagUses(self.VERTEX_ID_FLAG)):
                flag_args = args.getFlagArgumentList(self.VERTEX_ID_FLAG, i)
                vertices.append(flag_args.asInt(0))

        n_points = len(vectors)
        n_vertices = len(vertices)
        if not n_points or not n_vertices or n_points != n_vertices:
            sys.stderr.write("points and ids need to match.")

        return path, vectors, vertices

    def redoIt(self):
        it_vert = self.it_vert.reset()
        for v in self.vectors:
            it_vert.translateBy(v)
            it_vert.next()
        self.done = True

    def undoIt(self):
        it_vert = self.it_vert.reset()
        for p in self.old_points:
            it_vert.setPosition(p)
            it_vert.next()

    def isUndoable(self):
        return self.done

    @classmethod
    def syntax_creator(cls):
        syntax = MSyntax()
        syntax.addArg(MSyntax.kString)

        syntax.addFlag(cls.POINT_FLAG, cls.POINT_FLAG_LONG,
                       [MSyntax.kDouble, MSyntax.kDouble, MSyntax.kDouble])
        syntax.makeFlagMultiUse(cls.POINT_FLAG)

        syntax.addFlag(cls.VERTEX_ID_FLAG, cls.VERTEX_ID_FLAG_LONG, MSyntax.kUnsigned)
        syntax.makeFlagMultiUse(cls.VERTEX_ID_FLAG)

        return syntax


# Plug-in initialization
def initializePlugin(mobject):
    mplugin = MFnPlugin(mobject)
    try:
        mplugin.registerCommand(setVertexPoints.NAME, setVertexPoints,
                                setVertexPoints.syntax_creator)
    except:
        sys.stderr.write("Failed to register {}".format(setVertexPoints.NAME))


def uninitializePlugin(mobject):
    mplugin = MFnPlugin(mobject)
    try:
        mplugin.deregisterCommand(setVertexPoints.NAME)
    except:
        sys.stderr.write("Failed to unregister {}".format(setVertexPoints.NAME))
