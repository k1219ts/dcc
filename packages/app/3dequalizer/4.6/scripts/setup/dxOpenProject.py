# DXTER STUDIOS:  CG Supervisor Kwantae.Kim

import os
import tde4
import DD_common
import DXRulebook.Interface as rb


class dxOpenPrj:
    def __init__(self, requester):
        self.showRoot = '/show'
        self.req = requester
        self.teamMembers = DD_common.getTeamMembers()
        self.myTaskList = DD_common.getMyTask()

        self.show = ''
        self.seq = ''
        self.shot = ''
        self.plateType = ''
        self.windowTitle = ''

    def getShotInfo(self):
        tmp = DD_common.find_list_item(self.req, 'shotlist')
        info = tmp.split(' ')
        self.show = info[0].replace('[', '').replace(']', '').lower()

        if 'SHOW' in os.environ:
            if not self.show in os.environ['SHOW']:
                tde4.postQuestionRequester(self.windowTitle, 'Please 3DE run again, "%s" in dxRunner.'% self.show, 'Ok')
                return

        try:
            coder = rb.Coder()
            argv = coder.N.SHOTNAME.Decode(info[1])
            self.seq = argv.seq
            self.shot = info[1]
        except:
            tde4.postQuestionRequester(self.windowTitle, 'Please 3DE run again, "%s" in dxRunner.'% self.show, 'Ok')
            return

        # mmv training seq dir
        if 'mmv' in info[0]:
            self.seq = 'MMT'

        if self.seq.count('pos') > 0:
            self.seq = 'POS'

        if self.show == 'testshot':
            self.show = 'test_shot'
        elif self.show == 'cdh':
            self.show = 'cdh1'

    def setListWidgetMyTask(self):
        tde4.removeAllListWidgetItems(self.req, 'shotlist')
        mode = tde4.getWidgetValue(self.req, 'userlist') - 1
        user = self.teamMembers[mode]
        self.myTaskList = DD_common.getMyTask(user)

        if not self.myTaskList:
            tde4.insertListWidgetItem(self.req, 'shotlist', 'No Tasks.', 0)
            return

        for idx, i in enumerate(self.myTaskList):
            if not i['priority']:
                i['priority'] = 0
            tmp = '[' + i['project_code'].upper() + '] ' + i['extra_name'] + '  ..' + str(i['status']) + '  ' + str(
                i['end_date']) + '  ' + str(i['priority'])
            tde4.insertListWidgetItem(self.req, 'shotlist', tmp, 0)
            if i['status'] == 'Ready':
                tde4.setListWidgetItemColor(self.req, 'shotlist', idx, 1, 1, 0.6)
            elif i['status'] == 'In-Progress':
                tde4.setListWidgetItemColor(self.req, 'shotlist', idx, 0.75, 1, 0.15)
            elif i['status'] == 'OK':
                tde4.setListWidgetItemColor(self.req, 'shotlist', idx, 0, 1, 1)
            elif i['status'] == 'Retake':
                tde4.setListWidgetItemColor(self.req, 'shotlist', idx, 1, 0, 0)
            elif i['status'] == 'Review':
                tde4.setListWidgetItemColor(self.req, 'shotlist', idx, 0, 0.8, 0)

    def setShotinfo(self):
        os.environ['show'] = self.show
        os.environ['seq'] = self.seq
        os.environ['shot'] = self.shot
        os.environ['platetype'] = self.plateType

    def _searchShotKeyword(self, requester, widget, action):
        keyword = tde4.getWidgetValue(requester, 'keyword')
        shots = DD_common.getShotList(keyword=keyword)

        tde4.removeAllListWidgetItems(requester, 'shotlist')
        if shots:
            for i in shots:
                tde4.insertListWidgetItem(requester, 'shotlist', i, 0)
        else:
            tde4.insertListWidgetItem(requester, 'shotlist', 'No Shots', 0)

        tde4.removeAllListWidgetItems(requester, 'platetype')
        tde4.removeAllListWidgetItems(requester, 'filelist')
        tde4.insertListWidgetItem(requester, 'platetype', 'Select Shot.', 0)
        tde4.insertListWidgetItem(requester, 'filelist', 'Select Plate Type.', 0)

    def _clearListWidgetShot(self, requester, widget, action):
        tde4.setWidgetValue(requester, 'keyword', '')
        tde4.removeAllListWidgetItems(requester, 'shotlist')

    def _setListWidgetShot(self, requester, widget, action):
        self.setListWidgetMyTask()

    def _setListWidgetPlateTypeTask(self, requester, widget, action):
        chk = DD_common.find_list_item(requester, 'shotlist')
        if chk.startswith('No ') or chk.startswith('Select '):
            return

        self.getShotInfo()
        # /show/pipe/_2d/shot/PKL/PKL_0290/plates/main1/v001
        shotPath = os.path.join(self.showRoot, self.show, '_2d/shot',
                                self.seq, self.shot, 'plates')
        platesList = DD_common.get_dir_list(shotPath)
        tde4.removeAllListWidgetItems(requester, 'platetype')

        for i in platesList:
            tde4.insertListWidgetItem(requester, 'platetype', i, 0)

        tde4.removeAllListWidgetItems(requester, 'filelist')
        tde4.insertListWidgetItem(requester, 'filelist', 'Select Plate Type.', 0)

    def _setListWidgetFile(self, requester, widget, action):
        chk = DD_common.find_list_item(requester, 'platetype')
        if chk.startswith('No ') or chk.startswith('Select '):
            return

        self.plateType = DD_common.find_list_item(requester, 'platetype')
        file_list = DD_common.getFileList(os.path.join(self.showRoot, self.show, 'works/MMV/shot', self.seq, self.shot, '3de'), self.plateType, '*.3de')
        file_list.sort(reverse=True)
        tde4.removeAllListWidgetItems(requester, 'filelist')
        if file_list[0].startswith('No '):
            file_list = ['Create New Project.']
        for i in file_list:
            tde4.insertListWidgetItem(requester, 'filelist', i, 0)

    def doIt(self):
        filename = DD_common.find_list_item(self.req, 'filelist')

        if self.show != None:
            if not self.show.startswith('Select ') or self.show.startswith('No '):    gogo = 1
        else:
            gogo = 0
        if self.shot != None:
            if not self.shot.startswith('Select ') or self.shot.startswith('No '):    gogo = 1
        else:
            gogo = 0
        if self.plateType != None:
            if not self.plateType.startswith('Select ') or self.plateType.startswith('No '):    gogo = 1
        else:
            gogo = 0
        if filename != None:
            if not filename.startswith('Select ') or filename.startswith('No '):    gogo = 1
        else:
            gogo = 0

        if gogo == 0:
            tde4.postQuestionRequester(self.windowTitle, 'Select Sequence or Shot or File.', 'Ok')
        else:
            shotRoot = os.path.join(self.showRoot, self.show, '_2d/shot', self.seq, self.shot)
            mmvRoot = os.path.join(self.showRoot, self.show, 'works/MMV/shot', self.seq, self.shot)

            projPath = tde4.getProjectPath()
            if not projPath == None:
                ans = tde4.postQuestionRequester(self.windowTitle, 'Do you want to Save the project?', 'Yes', 'No')
                if ans == 1:
                    tde4.saveProject(projPath)
                #else:
                    #tde4.postQuestionRequester(self.windowTitle, 'save Cancelled.', 'Ok')

            if filename == 'Create New Project.':
                if not os.path.isdir(mmvRoot):
                    ans = tde4.postQuestionRequester(self.windowTitle,
                                                     'There is no matchmove folder in shot folder. Create matchmove folder?',
                                                     'Yes', 'No')
                    if ans == 1:
                        DD_common.makeMMVdirs(mmvRoot)
                    else:
                        tde4.postQuestionRequester(self.windowTitle, 'Cancelled.', 'Ok')
                        return

                DD_common.clearProject(1)
                shotPath = os.path.join(shotRoot, 'plates', self.plateType)
                slVersion = DD_common.get_dir_list(shotPath)
                slVersion.sort(reverse=True)
                fileList = DD_common.getSeqFileList(os.path.join(shotPath, slVersion[0]))
                # print shotPath, '\n', self.plateType, '\n', slVersion[0]

                gamma = DD_common.get_show_config(self.show, 'gamma')

                for i in fileList:
                    tmp = '%s :[%s-%s]' % (i[0], i[1], i[2])
                    # print tmp
                    fileName, frameRange = tmp.split(' :')  # result: 'APT_0020_main1_v001.1001.jpg', '[1001-1003]'
                    start, end = frameRange.split('-')  # result: '[1001', '1003]'
                    num = DD_common.extractNumber(fileName)  # result: 1001
                    pad = '#' * len(num)  # result: '####'
                    fileName2 = fileName.replace(num, pad)  # result: 'APT_0020_main1_v001.####.jpg'

                    frameIndex = fileName.rfind(num)  # result: 18
                    camName = fileName[:frameIndex - 1]  # result: 'APT_0020_main1_v001'

                    # print fileName, '\n', fileName2

                    cam = tde4.createCamera('SEQUENCE')
                    if gamma:
                        tde4.setCamera8BitColorGamma(cam, float(gamma))
                    tde4.setCameraPath(cam, os.path.join(shotPath, slVersion[0], fileName))
                    tde4.setCameraPath(cam, os.path.join(shotPath, slVersion[0], fileName2))
                    tde4.setCameraName(cam, camName)
                    tde4.setCameraSequenceAttr(cam, int(start[1:]), int(end[:-1]), 1)

                tde4.saveProject(os.path.join(mmvRoot, '3de', self.shot + '_%s' % self.plateType + '_matchmove_v001.3de'))
                self.setShotinfo()
            else:
                prj = os.path.join(mmvRoot, '3de', filename)
                DD_common.clearProject(0)
                tde4.loadProject(prj)
                self.setShotinfo()
                if tde4.getWidgetValue(self.req, 'import_cache') == 1:
                    DD_common.importBcompress()
