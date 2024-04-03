import sys

from maya.api.OpenMaya import (MFnPlugin, MPxCommand, MArgParser, MSyntax,
                               MGlobal, MPointArray, MColorArray, MIntArray,
                               MFn, MFnMesh)


def maya_useNewAPI():
    """
	The presence of this function tells Maya that the plugin produces, and
	expects to be passed, objects created using the Maya Python API 2.0.
	"""
    pass


class selectVertices(MPxCommand):
    NAME = "setMeshPoints"
    POINT_FLAG = "-p"
    POINT_FLAG_LONG = "-points"
    COLOR_FLAG = "-c"
    COLOR_FLAG_LONG = "-colors"
    VERTEX_ID_FLAG = "-v"
    VERTEX_ID_FLAG_LONG = "-vertices"

    def __init__(self):
        MPxCommand.__init__(self)

        self.fn_mesh = None

        self.orig_points = None
        self.new_points = None

        self.orig_colors = None
        self.new_colors = None

        self.done = False

    def doIt(self, args):
        path, points, colors = self.parse_arguments(args)
        if not points and not colors:
            sys.stderr.write("no essential flag is set.")

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

        fn_mesh = MFnMesh(dag)
        n = fn_mesh.numVertices
        if (points and len(points) != n) or (colors and len(colors) != n):
            sys.stderr.write("number of points provided "
                             "does not meet with {}.".format(path))

        self.fn_mesh = fn_mesh
        self.orig_points = fn_mesh.getPoints() if points else None
        self.new_points = points
        self.orig_colors = fn_mesh.getVertexColors() if colors else None
        self.new_colors = colors

        self.redoIt()

    def parse_arguments(self, args):
        args = MArgParser(self.syntax(), args)
        path = args.commandArgumentString(0)

        points = MPointArray()
        colors = MColorArray()
        if args.isFlagSet(self.POINT_FLAG):
            for i in range(args.numberOfFlagUses(self.POINT_FLAG)):
                flag_args = args.getFlagArgumentList(self.POINT_FLAG, i)
                points.append(flag_args.asPoint(0))
        if args.isFlagSet(self.COLOR_FLAG):
            for i in range(args.numberOfFlagUses(self.COLOR_FLAG)):
                flag_args = args.getFlagArgumentList(self.COLOR_FLAG, i)
                colors.append(flag_args.asPoint(0))

        if not len(points):
            points = None
        if not len(colors):
            colors = None

        return path, points, colors

    def redoIt(self):
        if self.new_points:
            self.fn_mesh.setPoints(self.new_points)
        if self.new_colors:
            self.fn_mesh.clearColors()
            self.fn_mesh.setColors(self.new_colors)
            self.fn_mesh.assignColors()
        self.done = True

    def undoIt(self):
        if self.orig_points:
            self.fn_mesh.setPoints(self.orig_points)
        if self.orig_colors:
            self.fn_mesh.setPoints(self.orig_colors)

    def isUndoable(self):
        return self.done

    @classmethod
    def syntax_creator(cls):
        syntax = MSyntax()
        syntax.addArg(MSyntax.kString)

        syntax.addFlag(cls.POINT_FLAG, cls.POINT_FLAG_LONG,
                       [MSyntax.kDouble, MSyntax.kDouble, MSyntax.kDouble])
        syntax.makeFlagMultiUse(cls.POINT_FLAG)

        syntax.addFlag(cls.COLOR_FLAG, cls.COLOR_FLAG_LONG,
                       [MSyntax.kDouble, MSyntax.kDouble, MSyntax.kDouble, MSyntax.kDouble])
        syntax.makeFlagMultiUse(cls.COLOR_FLAG)

        return syntax


# Plug-in initialization
def initializePlugin(mobject):
    mplugin = MFnPlugin(mobject)
    try:
        mplugin.registerCommand(selectVertices.NAME, selectVertices,
                                selectVertices.syntax_creator)
    except:
        sys.stderr.write("Failed to register {}".format(selectVertices.NAME))


def uninitializePlugin(mobject):
    mplugin = MFnPlugin(mobject)
    try:
        mplugin.deregisterCommand(selectVertices.NAME)
    except:
        sys.stderr.write("Failed to unregister {}".format(selectVertices.NAME))
