#
#
# 3DE4.script.name:  2. dxImport Project Plates...
#
# 3DE4.script.version:  v1.1
#
# 3DE4.script.gui:  Main Window::dx_Setup
#
# 3DE4.script.comment:  Import Project Plates.
#
# DXTER STUDIOS:  CG Supervisor Kwantae.Kim

import os
import glob, re
import json
import DD_common


def _import_plates_callback(requester, widget, action):
    if widget == 'plateType':
        slPlateType = DD_common.find_list_item(requester, 'plateType')
        verList = DD_common.get_dir_list(os.path.join(shotPath, slPlateType))

        tde4.removeAllListWidgetItems(requester, 'ver')
        for i in verList:
            count = 0
            tde4.insertListWidgetItem(requester, 'ver', i, count)
            count += 1

    if widget == 'ver':
        slPlateType = DD_common.find_list_item(requester, 'plateType')
        slVersion = DD_common.find_list_item(requester, 'ver')

        if not slVersion.startswith('Select'):
            fileList = DD_common.getSeqFileList(os.path.join(shotPath, slPlateType, slVersion))

            tde4.removeAllListWidgetItems(requester, 'fileList')
            for i in fileList:
                count = 0

                #print i

                tde4.insertListWidgetItem(requester, 'fileList', '%s :[%s-%s]' % (i[0], i[1], i[2]), count)

                # type(i)
                # if type(i) == tuple:
                #     tde4.insertListWidgetItem(requester, 'fileList', '%s :[%s-%s]'%(i[0], i[1], i[2]), count)
                # else:
                #     tde4.insertListWidgetItem(requester, 'fileList', i, count)
                count += 1

#
# main...
if os.environ.has_key('show'):

    shotPath = os.path.join('/', 'show', os.environ['show'], '_2d', 'shot', os.environ['seq'], os.environ['shot'], 'plates')
    platesList = DD_common.get_dir_list(shotPath)
    verList = DD_common.get_dir_list(shotPath)

    req = tde4.createCustomRequester()
    tde4.addTextFieldWidget(req, 'show', 'Show', os.environ['show'])
    tde4.setWidgetSensitiveFlag(req, 'show', 0)
    tde4.addTextFieldWidget(req, 'seq', 'Sequence', os.environ['seq'])
    tde4.setWidgetSensitiveFlag(req, 'seq', 0)
    tde4.addTextFieldWidget(req, 'shot', 'Shot', os.environ['shot'])
    tde4.setWidgetSensitiveFlag(req, 'shot', 0)

    tde4.addListWidget(req, 'plateType', 'Plate Type', 0, 70)
    for i in platesList:
        count = 0
        tde4.insertListWidgetItem(req, 'plateType', i, count)
        count += 1
    tde4.setWidgetCallbackFunction(req, 'plateType', '_import_plates_callback')

    tde4.addListWidget(req, 'ver', 'Version', 0, 70)
    tde4.insertListWidgetItem(req, 'ver', 'Select plate type first.', 0)
    tde4.setWidgetCallbackFunction(req, 'ver', '_import_plates_callback')

    tde4.addListWidget(req, 'fileList', 'File List', 1, 130)
    tde4.insertListWidgetItem(req, 'fileList', 'Select version first.', 0)
    tde4.setWidgetCallbackFunction(req, 'fileList', '_import_plates_callback')

    ret = tde4.postCustomRequester(req, 'Import Project Plates', 700, 0, 'Ok', 'Cancel')
    if ret == 1:
        slPlates = DD_common.findListItems(req, 'fileList')
        slPlateType = DD_common.find_list_item(req, 'plateType')
        slVersion = DD_common.find_list_item(req, 'ver')

        gamma = DD_common.get_show_config(os.environ['show'], 'gamma')

        for i in slPlates:
            if '#' and ' :[' and ']' in i:
                fileName, frameRange = i.split(' :')    # result: 'SHS_0420_main_v02.0101.jpg', '[101-103]'
                start, end = frameRange.split('-')    # result: '[101', '103]'
                num = DD_common.extractNumber(fileName)    # result: 0101
                pad = '#' * len(num)    # result: '####'
                fileName2 = fileName.replace(num, pad)    # result: 'SHS_0420_main_v02.####.jpg'

                frameIndex = fileName.rfind(num)    # result: 18
                camName = fileName[:frameIndex-1]    # result: 'SHS_0420_main_v01'

                cam = tde4.createCamera('SEQUENCE')
                if gamma:
                    tde4.setCamera8BitColorGamma(cam, float(gamma))
                tde4.setCameraPath(cam, os.path.join(shotPath, slPlateType, slVersion, fileName))
                tde4.setCameraPath(cam, os.path.join(shotPath, slPlateType, slVersion, fileName2))
                tde4.setCameraName(cam, camName)
                tde4.setCameraSequenceAttr(cam, int(start[1:]), int(end[:-1]), 1)
            else:
                cam = tde4.createCamera('REF_FRAME')
                tde4.setCameraPath(cam, os.path.join(shotPath, slPlateType, slVersion, i))
                tde4.setCameraName(cam, os.path.splitext(i)[0])

else:
    tde4.postQuestionRequester('Import Project Plate.', 'Please open a project using \'Open Project\' script first.', 'Ok')
