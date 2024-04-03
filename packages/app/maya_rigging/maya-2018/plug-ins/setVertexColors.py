import sys

from maya.api.OpenMaya import (MFnPlugin, MPxCommand, MArgParser, MSyntax,
                               MFn, MGlobal, MFnMesh,
                               MIntArray, MColor, MColorArray, MPointArray)


def maya_useNewAPI():
    """
	The presence of this function tells Maya that the plugin produces, and
	expects to be passed, objects created using the Maya Python API 2.0.
	"""
    pass


class setVertexColors(MPxCommand):
    NAME = "setVertexColors"
    COLORSET_FLAG = "-s"
    COLORSET_FLAG_LONG = "-colorSet"
    COLOR_FLAG = "-c"
    COLOR_FLAG_LONG = "-colors"
    RGBA_FLAG = "-a"
    RGBA_FLAG_LONG = "-rgbas"
    ID_FLAG = "-ids"
    ID_FLAG_LONG = "-vertIds"

    def __init__(self):
        MPxCommand.__init__(self)

        self.fnMesh = None

        self.origColorSet = None
        self.origColors = None

        self.colorSet = None
        self.colors = None
        self.ids = None
        self.colorRepresentation = MFnMesh.kRGB

        self.done = False

    def doIt(self, args):
        path, color_set, ids, colors = self.parse_arguments(args)
        if not colors:
            sys.stderr.write("colors flag is not set.")

        num_colors = len(colors)
        if not ids:
            ids = list(range(num_colors))

        num_ids = len(ids)
        if num_ids != num_colors:
            sys.stderr.write("numbers of colors and ids do not match.")

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
        if not color_set:
            color_set = fn_mesh.currentColorSetName()
            if not color_set:
                sys.stderr.write("{} does not have any colorset.".format(path))
        elif color_set not in fn_mesh.getColorSetNames():
            sys.stderr.write("{} does not have colorset named {}".format(path, color_set))

        num_verts = fn_mesh.numVertices
        if num_colors > num_verts:
            sys.stderr.write("too many colors")
        if num_ids > num_verts or min(ids) < 0 or max(ids) >= num_verts:
            sys.stderr.write("id error")

        self.fnMesh = fn_mesh

        self.origColorSet = fn_mesh.currentColorSetName()
        mesh_colors = fn_mesh.getVertexColors()
        orig_colors = MColorArray()
        for i in ids:
            orig_colors.append(mesh_colors[i])
        self.origColors = orig_colors

        self.colorSet = color_set
        self.colors = colors
        self.ids = ids

        self.redoIt()

    def parse_arguments(self, arg_parser):
        arg_parser = MArgParser(self.syntax(), arg_parser)
        path = arg_parser.commandArgumentString(0)

        color_set = ''
        ids = MIntArray()
        colors = MColorArray()
        if arg_parser.isFlagSet(self.COLORSET_FLAG):
            color_set = arg_parser.flagArgumentString(self.COLORSET_FLAG, 0)
        if arg_parser.isFlagSet(self.ID_FLAG):
            for i in range(arg_parser.numberOfFlagUses(self.ID_FLAG)):
                flag_args = arg_parser.getFlagArgumentList(self.ID_FLAG, i)
                ids.append(flag_args.asInt(0))
        if arg_parser.isFlagSet(self.COLOR_FLAG):
            for i in range(arg_parser.numberOfFlagUses(self.COLOR_FLAG)):
                flag_args = arg_parser.getFlagArgumentList(self.COLOR_FLAG, i)
                colors.append(MColor(flag_args.asVector(0)))
        if arg_parser.isFlagSet(self.RGBA_FLAG):
            points = MPointArray()
            self.colorRepresentation = MFnMesh.kRGBA
            for i in range(arg_parser.numberOfFlagUses(self.RGBA_FLAG)):
                flag_args = arg_parser.getFlagArgumentList(self.RGBA_FLAG, i)
                colors.append(flag_args.asPoint(0, 4))

        if not len(ids):
            ids = []
        if not len(colors):
            colors = None

        return path, color_set, ids, colors

    def redoIt(self):
        self.fnMesh.setCurrentColorSetName(self.colorSet)
        self.fnMesh.setVertexColors(self.colors, self.ids, rep=self.colorRepresentation)
        self.fnMesh.setCurrentColorSetName(self.origColorSet)
        self.done = True

    def undoIt(self):
        self.fnMesh.setCurrentColorSetName(self.colorSet)
        self.fnMesh.setVertexColors(self.origColors, self.ids, rep=MFnMesh.kRGBA)
        self.fnMesh.setCurrentColorSetName(self.origColorSet)
        # self.done = False

    def isUndoable(self):
        return self.done

    @classmethod
    def syntax_creator(cls):
        syntax = MSyntax()
        syntax.addArg(MSyntax.kString)

        syntax.addFlag(cls.COLORSET_FLAG, cls.COLORSET_FLAG_LONG, MSyntax.kString)

        syntax.addFlag(cls.ID_FLAG, cls.ID_FLAG_LONG, MSyntax.kLong)
        syntax.makeFlagMultiUse(cls.ID_FLAG)

        syntax.addFlag(cls.COLOR_FLAG, cls.COLOR_FLAG_LONG,
                       [MSyntax.kDouble, MSyntax.kDouble, MSyntax.kDouble])
        syntax.makeFlagMultiUse(cls.COLOR_FLAG)

        syntax.addFlag(cls.RGBA_FLAG, cls.RGBA_FLAG_LONG,
                       [MSyntax.kDouble, MSyntax.kDouble, MSyntax.kDouble, MSyntax.kDouble])
        syntax.makeFlagMultiUse(cls.RGBA_FLAG)

        return syntax


# Plug-in initialization
def initializePlugin(mobject):
    mplugin = MFnPlugin(mobject)
    try:
        mplugin.registerCommand(setVertexColors.NAME, setVertexColors, setVertexColors.syntax_creator)
    except:
        sys.stderr.write("Failed to register {}".format(setVertexColors.NAME))


def uninitializePlugin(mobject):
    mplugin = MFnPlugin(mobject)
    try:
        mplugin.deregisterCommand(setVertexColors.NAME)
    except:
        sys.stderr.write("Failed to unregister {}".format(setVertexColors.NAME))
