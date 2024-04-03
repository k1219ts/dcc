import hou

def get():
    desktop = hou.ui.curDesktop()
    sceneViewer = desktop.paneTabOfType(hou.paneTabType.SceneViewer)
    viewport = sceneViewer.curViewport()
    
    if viewport.camera(): 
        cam = viewport.camera().path()
    else:
        cam = None

    return cam