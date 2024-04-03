import json, os
import nukeCommon as comm

import nuke, nukescripts
import dxpublish.insertDB as insertDB

def onCreate():
    # mov setting - Rec.709
    node = nuke.thisNode()
    try:
        if '.mov' in node['file'].getValue():
            colorM = nuke.root().knob('colorManagement').value()
            if 'OCIO' == colorM:
                node['colorspace'].setValue("Output - Rec.709")
            else:
                node['colorspace'].setValue("sRGB")
    # exr setting - ACES 2065-1
        if 'plates' in node['file'].getValue():
            colorM = nuke.root().knob('colorManagement').value()
            if 'OCIO' == colorM:
                node['colorspace'].setValue('ACES - ACES2065-1')
            else:
                pass

    except:
        pass

# set onCreate callback
nuke.addOnCreate(onCreate, nodeClass='Read')

def dropCallback(mimeType, path):
    try:
        print('dropCallback!')
        print('path:', path)
        # delete 3DEqulizer cache file
        if os.path.isdir(path):
            for file in os.listdir(path):
                if '.3de_bcompress' in file:
                    os.remove(os.path.join(path, file))
                    print('# deleteFile: %s' % file)
    except Exception as e:
        print("nuke dropCallback error")
        print(e)

def saveCallBack():
    result = insertDB.recordWork('nuke', 'save', nuke.root().name())
    print('result:', result)

def loadCallBack():
    try:
        result = insertDB.recordWork('nuke', 'open', nuke.root().name())
        fullPath = nuke.root().name()

        for i in ['/netapp/dexter/show/', '/mach', '/knot']:
            fullPath = fullPath.replace(i, '')
            print("fullpath : ", fullPath)

        if not '/show/' in fullPath:
            return

        configData = comm.getDxConfig()
        if configData:
            if 'ACES' in configData['colorSpace']:
                nuke.root().knob('colorManagement').setValue('OCIO')
            nuke.root()['fps'].setValue(configData['delivery']['fps'])
            print('fps:', configData['delivery']['fps'])

        # old path resolve
        comm.resolveOldPath()

        # only open defaultFile
        if not nuke.allNodes('Read'):
            show, seq, shot = comm.readPlates(fullPath)
            print('shotInfo:', show, seq, shot)

            shotInfo = comm.getShotInfo(show, seq, shot)
            if shotInfo:
                nuke.root().knob("first_frame").setValue(shotInfo['frame_in'])
                nuke.root().knob("last_frame").setValue(shotInfo['frame_out'])
                print('frameRange:', shotInfo['frame_in'], shotInfo['frame_out'])

            if configData:
                res = '%s %s' % (str(configData['works']['resolution'][0]), str(configData['works']['resolution'][1]))
                prjFormat = nuke.addFormat(res)
                nuke.root()['format'].setValue(prjFormat)
                print('resolution:', res)
    except Exception as e:
        print("nuke loadCallBack error")
        print(e)
