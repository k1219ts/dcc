#
# 3DE4.script.name:  shotChange Tool ...
#
# 3DE4.script.version:  v1.0.0
#
# 3DE4.script.gui:  Main Window::Dexter
#
# 3DE4.script.comment:  Shot change Tool.
#
# DXTER STUDIOS:  CG Supervisor Kwantae.Kim


import os
import shutil
import tde4
import DD_common
reload(DD_common)


class dxShotChange:
    def __init__(self, requester, windowTitle):
        self.req = requester
        self.windowTitle = windowTitle

        self.SHOW_ROOT = '/show'
        self.show = os.environ['show']
        self.seq = os.environ['seq']
        self.shot = os.environ['shot']
        self.shotPath = os.path.join(self.SHOW_ROOT, self.show, '_2d', 'shot', '{SEQ}', '{SHOT}', 'plates')
        self.mmvpubPath = os.path.join(self.SHOW_ROOT, self.show, '_3d', 'shot', '{SEQ}', '{SHOT}', 'cam', 'scenes')
        self.mmvPath = os.path.join(self.SHOW_ROOT, self.show, 'works', 'MMV', 'shot', '{SEQ}', '{SHOT}')

    def _toShot_callback(self, requester, widget, action):
        try:
            self.toShot = tde4.getWidgetValue(requester, "toShot")
            self.toSeq = self.toShot.split('_')[0]

            shotInfo = DD_common.getShotInfo(self.show, [os.environ['shot'], self.toShot])

            tde4.clearTextAreaWidget(requester, 'shotInfo')
            tde4.appendTextAreaWidgetString(requester, 'shotInfo', shotInfo)

            shotPlatePath = os.path.join(self.SHOW_ROOT, self.show, '_2d', 'shot', self.toSeq, self.toShot, 'plates')

            platesList = DD_common.get_dir_list(shotPlatePath)
            verList = DD_common.get_dir_list(shotPlatePath)

            tde4.removeAllListWidgetItems(requester, 'platetype')
            for i in platesList:
                count = 0
                tde4.insertListWidgetItem(requester, 'platetype', i, count)
                count += 1
        except:
            tde4.clearTextAreaWidget(requester, 'shotInfo')
            tde4.appendTextAreaWidgetString(requester, 'shotInfo', 'input shotName.')


    def copyMayaScene(self, toPath):

        if os.path.isdir(self.mmvpubPath):
            i = os.listdir(self.mmvpubPath)

            try:
                if len(i) > 1:
                    i.sort(reverse=True)

                lastVer = i[0]
                from_pubPath = os.path.join(self.mmvpubPath, lastVer)
                from_fileList = DD_common.getFileList(from_pubPath, '', '*.mb')
                fromPath = os.path.join(from_pubPath, from_fileList[0])

                shutil.copy(fromPath, toPath)
            except:
                print 'MayaScene Copy Error!'

    def doIt(self):
        self.toShot = tde4.getWidgetValue(self.req, 'toShot')
        self.toSeq = self.toShot.split('_')[0]
        platetype = DD_common.find_list_item(self.req, 'platetype')

        if platetype!=None:
            if not platetype.startswith('Input ') or platetype.startswith('No '):    gogo = 1
        else:    gogo = 0

        if gogo == 0:
            tde4.postQuestionRequester(self.windowTitle, 'Select plateType.', 'Ok')
        else:
            shotMmvRoot = self.mmvPath.format(SEQ=self.toSeq, SHOT=self.toShot)
            newShotPath = os.path.join(self.mmvPath.format(SEQ=self.toSeq, SHOT=self.toShot), '3de', self.toShot + '_%s' % platetype + '_matchmove_v001.3de')

            if os.path.isfile(newShotPath):
                path = os.path.dirname(newShotPath)   # /show/slc/works/MMV/shot/MET/MET_0080//3de
                selList = DD_common.getFileList(path)
                selList.sort(reverse=True)
                file = selList[0]                       # MET_0080_main1_matchmove_v001.3de
                fileName = file.split('.')[0]           # MET_0080_main1_matchmove_v001
                splitFileName = fileName.split('_')     # ['MET', '0080', 'main1', matchmove', 'v001']
                newReVer = int(splitFileName[-1].replace('v', '')) + 1   # 001 + 1
                splitFileName[-1] = 'v%.3d' % newReVer   # ['MET', '0080', 'main1', matchmove', 'v002']
                newFileName = '_'.join(splitFileName)   # MET_0080_main1_matchmove_v002

                newShotPath = os.path.join(path, newFileName+'.3de')

            # print 'newShotPath:', newShotPath

            msg = '"%s"\nDo you want to Save the project?' % newShotPath
            ans = tde4.postQuestionRequester(self.windowTitle, msg, 'Yes', 'No')

            if ans == 1:
                if not os.path.isdir(shotMmvRoot):
                    DD_common.makeMMVdirs(shotMmvRoot)

                if tde4.getWidgetValue(self.req, 'copy_mayaScene') == 1:
                    self.copyMayaScene(os.path.join(shotMmvRoot, 'scenes', self.toShot + '_%s' % platetype + '_matchmove_v001.mb'))

                if tde4.getWidgetValue(self.req, 'add_plate') == 1:
                    shotPath = os.path.join(self.shotPath.format(SEQ=self.toSeq, SHOT=self.toShot), platetype)
                    slVersion = DD_common.get_dir_list(shotPath)
                    slVersion.sort(reverse=True)
                    fileList = DD_common.getSeqFileList(os.path.join(shotPath, slVersion[0]))

                    cam = ''
                    gamma = DD_common.get_show_config(self.show, 'gamma')

                    for i in fileList:
                        tmp = '%s :[%s-%s]' % (i[0], i[1], i[2])
                        fileName, frameRange = tmp.split(' :')  # result: 'SHS_0420_main_v02.0101.jpg', '[101-103]'
                        start = int(i[1])
                        end = int(i[2])  # result: '[101', '103]'
                        num = str(DD_common.extractNumber(fileName))  # result: 0101
                        pad = '#' * len(num)  # result: '####'
                        fileName2 = fileName.replace(num, pad)  # result: 'SHS_0420_main_v02.####.jpg'

                        frameIndex = fileName.rfind(num)  # result: 18
                        camName = fileName[:frameIndex - 1]  # result: 'SHS_0420_main_v01'

                        cam = tde4.createCamera('SEQUENCE')
                        if gamma:
                            tde4.setCamera8BitColorGamma(cam, float(gamma))
                        tde4.setCameraPath(cam, os.path.join(shotPath, slVersion[0], fileName2))
                        tde4.setCameraName(cam, camName)
                        tde4.setCameraSequenceAttr(cam, start, end, 1)

                tde4.saveProject(newShotPath)

                os.environ['show'] = self.show
                os.environ['seq'] = self.toSeq
                os.environ['shot'] = self.toShot
                os.environ['platetype'] = platetype
