import hou, getpass, datetime, os, json

# Aftersave callback
def savedCallback(event_type):
    if event_type == hou.hipFileEventType.AfterSave:
        hipFile = hou.hipFile.path()

        dataDict = {'artist': getpass.getuser(),
                    'time': datetime.datetime.now().isoformat(),
                    'file': hipFile,
                    'frameRange': (hou.playbar.frameRange()[0], hou.playbar.frameRange()[1])
                    }
        if os.environ.has_key('REZ_USED_REQUEST'):
            dataDict['rezRequest'] = os.environ['REZ_USED_REQUEST'].split()

        with open(hipFile.replace('.hip', '.json'), 'w') as f:
            json.dump(dataDict, f, indent=4)

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