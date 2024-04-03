import hou
from utils import setvariable
from network_editor import footprints

count = 0
debug = 0

def scene_event_callback(event_type):
    if event_type == hou.hipFileEventType.AfterLoad:

        #footprints.viewportcallbacks().addcamcallback()
        setvariable.version(hou.hipFile.path())

def resettoplevel(event_type):
    if event_type == hou.hipFileEventType.AfterLoad:
        tabs = hou.ui.paneTabs()
        editors = [t.setFootprints([]) for t in tabs 
            if t.type().name() == 'NetworkEditor' 
            and t.pwd().path().count('/')==1]

def waitforUI():
    global count

    if count == 0:

        ## Add camera callback

        try:
            if debug:
                print 'trying to add cam callback'
            footprints.viewportcallbacks()
            if debug:
                print 'added cam callback'
        except:
            if debug:
                print 'failed to add cam callback'
            return

        ## Add light callback

        try:
            if debug:
                print 'trying to add light callback'
            footprints.lightfootprints()
            if debug:
                print 'added light callback'
        except:
            if debug:
                print 'failed to light cam callback'
            return
            
    else:
        try:
            hou.ui.removeEventLoopCallback(waitforUI)
            if debug:
                print 'removed waitforui ui callback'
        except:
            if debug:
                print 'failed to remove waitforui ui callback'
            pass

    count +=1

def main():
    if hou.isUIAvailable():
        hou.ui.addSelectionCallback(
            footprints.selectioncallback)

        hou.ui.addEventLoopCallback(waitforUI)

        hou.hipFile.addEventCallback(scene_event_callback)

        #hou.hipFile.addEventCallback(resettoplevel)

main()