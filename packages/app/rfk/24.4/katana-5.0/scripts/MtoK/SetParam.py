from Katana import NodegraphAPI

def SetCameraTransform(tx, ty, tz, rx, ry, rz):
    node = NodegraphAPI.GetNode('MtK_Camera')
    if not node:
        print('[ERROR MtoK]: Not found "MtK_Camera"')
        return
    node.getParameter('transform.translate.x').setValue(tx, 0)
    node.getParameter('transform.translate.y').setValue(ty, 0)
    node.getParameter('transform.translate.z').setValue(tz, 0)

    node.getParameter('transform.rotate.x').setValue(rx, 0)
    node.getParameter('transform.rotate.y').setValue(ry, 0)
    node.getParameter('transform.rotate.z').setValue(rz, 0)


def SetCameraFov(value):
    node = NodegraphAPI.GetNode('MtK_Camera')
    if not node:
        print('[ERROR MtoK]: Not found "MtK_Camera"')
        return
    node.getParameter('fov').setValue(value, 0)

def SetCameraBase(fov, hfa, vfa):
    node = NodegraphAPI.GetNode('MtK_Camera')
    if not node:
        print('[ERROR MtoK]: Not found "MtK_Camera"')
        return

    node.getParameter('fov').setValue(fov, 0)

    val = vfa / hfa
    node.getParameter('screenWindow.left').setValue(-1, 0)
    node.getParameter('screenWindow.right').setValue(1, 0)
    node.getParameter('screenWindow.bottom').setValue(val * -1, 0)
    node.getParameter('screenWindow.top').setValue(val, 0)


def SetCameraPanZoom(panx, pany, zoom, hfa):
    node = NodegraphAPI.GetNode('MtK_Camera')
    if not node:
        print('[ERROR MtoK]: Not found "MtK_Camera"')
        return

    left  = node.getParameter('screenWindow.left').getValue(0)
    right = node.getParameter('screenWindow.right').getValue(0)
    bottom= node.getParameter('screenWindow.bottom').getValue(0)
    top   = node.getParameter('screenWindow.top').getValue(0)

    new_left  = left * zoom + panx / hfa * 2
    new_right = right * zoom + panx / hfa * 2
    new_bottom= bottom * zoom + pany / hfa * 2
    new_top   = top * zoom + pany / hfa * 2

    node.getParameter('screenWindow.left').setValue(new_left, 0)
    node.getParameter('screenWindow.right').setValue(new_right, 0)
    node.getParameter('screenWindow.bottom').setValue(new_bottom, 0)
    node.getParameter('screenWindow.top').setValue(new_top, 0)


def SetCacheFile(filename=None):
    node = NodegraphAPI.GetNode('MtK_CacheIn')
    if not node:
        print('[ERROR MtoK]: Not found "MtK_CacheIn"')
        return
    if filename:
        node.getParameter('fileName').setValue(filename, 0)
