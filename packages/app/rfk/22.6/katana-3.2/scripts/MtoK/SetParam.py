from Katana import NodegraphAPI

def SetCameraTransform(tx, ty, tz, rx, ry, rz):
    node = NodegraphAPI.GetNode('MtK_Camera')
    if not node:
        print '[ERROR MtoK]: Not found "MtK_Camera"'
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
        print '[ERROR MtoK]: Not found "MtK_Camera"'
        return
    node.getParameter('fov').setValue(value, 0)


def SetCacheFile(filename=None):
    node = NodegraphAPI.GetNode('MtK_CacheIn')
    if not node:
        print '[ERROR MtoK]: Not found "MtK_CacheIn"'
        return
    if filename:
        node.getParameter('fileName').setValue(filename, 0)
