from contextlib import contextmanager
from maya import cmds


@contextmanager
def maya_namespace(
        namespace='', create_if_missing=True, restore_current_namespace=True):
    """Context manager to temporarily set a namespace"""
    initial_namespace = ':' + cmds.namespaceInfo(currentNamespace=True)
    if not namespace.startswith(':'):
        namespace = ':' + namespace
    try:
        if not cmds.namespace(absoluteName=True, exists=namespace):
            if create_if_missing:
                cmds.namespace(setNamespace=':')
                namespace = cmds.namespace(addNamespace=namespace)
            else:
                cmds.namespace(initial_namespace)
                raise ValueError(namespace + " doesn't exist.")
        cmds.namespace(setNamespace=namespace)
        yield namespace
    finally:
        if restore_current_namespace:
            cmds.namespace(setNamespace=initial_namespace)


def select_targets(shapes, selection_mode='replace'):
    shapes = [s for s in shapes if s.targets()]
    hovered = [s for s in shapes if s.hovered]
    targets = {t for s in hovered for t in s.targets() if cmds.objExists(t)}

    if selection_mode in ('add', 'replace'):
        return cmds.select(list(targets), add=selection_mode == 'add')
    elif selection_mode == 'remove':
        selection = [n for n in cmds.ls(sl=True) if n not in targets]
        return cmds.select(selection)

    # Invert selection
    selected = [s for s in shapes if s.selected]
    to_select = [s for s in shapes if s in hovered and s not in selected]
    # List targets unaffected by selection
    targets = {
        t for s in selected for t in s.targets()
        if cmds.objExists(t) and not s.hovered}
    # List targets in reversed selection
    invert_t = {t for s in to_select for t in s.targets() if cmds.objExists(t)}
    targets.union(invert_t)
    cmds.select(targets)
    return


def select_shapes_from_selection(shapes):
    selection = cmds.ls(sl=True)
    for shape in shapes:
        if not shape.targets():
            shape.selected = False
            continue
        for target in shape.targets():
            if target not in selection:
                shape.selected = False
                break
        else:
            shape.selected = True


def switch_namespace(name, namespace):
    basename = name.split("|")[-1]
    name = basename if ":" not in basename else basename.split(":")[-1]
    if not namespace:
        return name
    return namespace + ":" + name


class Selection():
    def __init__(self):
        self.shapes = []
        self.mode = 'replace'

    def set(self, shapes):
        if self.mode == 'add':
            if shapes is None:
                return
            return self.add(shapes)
        elif self.mode == 'replace':
            if shapes is None:
                return self.clear()
            return self.replace(shapes)
        elif self.mode == 'invert':
            if shapes is None:
                return
            return self.invert(shapes)
        elif self.mode == 'remove':
            if shapes is None:
                return
            for shape in shapes:
                if shape in self.shapes:
                    self.remove(shape)

    def replace(self, shapes):
        self.shapes = shapes

    def add(self, shapes):
        self.shapes.extend([s for s in shapes if s not in self])

    def remove(self, shape):
        self.shapes.remove(shape)

    def invert(self, shapes):
        for shape in shapes:
            if shape not in self.shapes:
                self.add([shape])
            else:
                self.remove(shape)

    def clear(self):
        self.shapes = []

    def __bool__(self):
        return bool(self.shapes)
    __nonzero__=__bool__

    def __iter__(self):
        return self.shapes.__iter__()


def get_selection_mode(ctrl, shift):
    if not ctrl and not shift:
        return 'replace'
    elif ctrl and shift:
        return 'invert'
    elif shift:
        return 'add'
    return 'remove'
