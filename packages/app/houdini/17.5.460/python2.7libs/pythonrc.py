import hou

# Aftersave callback
def savedCallback(event_type):
    if event_type == hou.hipFileEventType.AfterSave:
        print("save")

hou.hipFile.addEventCallback(savedCallback)

# Afteropen callback
opend = hou.hipFile.name()
def opendCallback(event_type):
    if event_type == hou.hipFileEventType.AfterLoad:
        print "open file: ", opend

hou.hipFile.addEventCallback(opendCallback)

# Hide old FileIO
hou.hscript("ophide Sop DXC_FileIO")

'''
# for test by leeys
def scene_was_loaded(event_type):
    if event_type == hou.hipFileEventType.AfterLoad:
        print("The user loaded", hou.hipFile.path())

hou.hipFile.addEventCallback(scene_was_loaded)
'''