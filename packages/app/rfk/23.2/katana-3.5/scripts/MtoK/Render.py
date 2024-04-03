from Katana import NodegraphAPI, RenderManager

def PreviewRender():
    node = NodegraphAPI.GetNode('MtK_Render')
    if not node:
        print '[ERRPR MtoK]: Not found "MtK_Render"'
        return

    vsetnode = NodegraphAPI.GetNode('MtoKVariableSet')
    if vsetnode:
        vsetnode.setBypassed(False)

    RenderManager.StartRender('previewRender', node=node)


def LiveRender():
    node = NodegraphAPI.GetNode('MtK_Render')
    if not node:
        print '[ERRPR MtoK]: Not found "MtK_Render"'
        return
    # rs = RenderManager.RenderingSettings()
    # rs.asynch = True
    # rs.interactiveMode = True
    # RenderManager.StartRender('liveRender', node=node, settings=rs)
    # r = RenderManager.StartRender('liveRender', node=node)
    # print r


def CancelRender():
    RenderManager.CancelRender()
