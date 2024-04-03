import os, sys, datetime
import subprocess, getpass
from pymongo import MongoClient
import pprint
from imp import reload

import tractor.api.author as author

import tde4
import TDE4_common
import DD_common
import dxUIcommon
reload(dxUIcommon)
import dxExportNuke
reload(dxExportNuke)

import DXRulebook.Interface as rb
reload(rb)
from dxname import tag_parser
from dxpublish import insertDB
from dxConfig import dxConfig

DB_IP = dxConfig.getConf('DB_IP')
DB_NAME = 'PIPE_PUB'

TRACTOR_IP = '10.0.0.106'
PORT = 80

class dxExportImp:
    def __init__(self, requester, camera):
        self.req = requester
        if isinstance(camera, list):
            self.cam = camera[0]
            self.camList = camera
        else:
            self.cam = camera

        if os.environ['show']:
            self.show = os.environ['show']
            self.seq = os.environ['seq']
            self.shot = os.environ['shot']
            self.plateType = os.environ['platetype']
        else:
            # case: pmodel
            scenePath = tde4.getProjectPath()
            coder = rb.Coder()
            argv = coder.D.TDE.WORKS.Decode(os.path.dirname(scenePath))
            self.show = argv.show

            if argv.seq:    self.seq = argv.seq
            else:           self.seq = ''

            if argv.shot:   self.shot = coder.N.SHOTNAME.Encode(**argv)
            else:           self.shot = ''

            self.plateType = ''

        self.seqPath = os.path.split(tde4.getCameraPath(self.cam))[0]
        self.seqWidth = tde4.getCameraImageWidth(self.cam)
        self.seqHeight = tde4.getCameraImageHeight(self.cam)
        self.seqAttr = tde4.getCameraSequenceAttr(self.cam) #start, end, step
        self.colorspaceR = DD_common.get_show_config(self.show, 'in')
        self.colorspaceW = DD_common.get_show_config(self.show, 'out')
        if not self.colorspaceR: self.colorspaceR = 'default'

        seqFiles = os.path.split(tde4.getCameraPath(self.cam))[1]
        ext = seqFiles.split('.')[-1]
        self.jpgFile = seqFiles.replace(ext, 'jpg')
        self.frames = tde4.getCameraNoFrames(self.cam)
        self.windowTitle = ''

    def doItPmodel(self):
        data = {}
        data['show'] = self.show
        data['seq'] = self.seq
        data['shot'] = self.shot
        data['plateType'] = self.plateType

        data['seqColorspaceR'] = 'default'
        data['seqColorspaceW'] = 'default'
        data['startFrame'] = 0

        data['filePath'] = tde4.getWidgetValue(self.req, 'file_path')

        data['overscan'] = 0
        data['jpgSize'] = tde4.getWidgetValue(self.req, 'size')
        data['burnIn'] = 0

        onlyScript = tde4.getWidgetValue(self.req, 'only_script')

        # write a nuke python script per ref camera.
        scriptFile = os.path.basename(tde4.getProjectPath()).replace('.3de', '.py')
        tde4.postProgressRequesterAndContinue(self.windowTitle, 'Undistorting, ...', len(self.camList), 'Stop')
        nukePyScriptFile = TDE4_common.valid_name(scriptFile)

        for frame, cam in enumerate(self.camList):
            data['seqFile'] = tde4.getCameraPath(cam)
            data['cameraId'] = cam
            data['camName'] = tde4.getCameraName(cam)

            data['seqWidth'] = tde4.getCameraImageWidth(cam)
            data['seqHeight'] = tde4.getCameraImageHeight(cam)
            data['seqAttr'] = tde4.getCameraSequenceAttr(cam)

            # get a camera, lens data.
            data['lens'] = tde4.getCameraLens(cam)
            data['lensLDModel'] = tde4.getLensLDModel(data['lens'])
            data['filmbackWidth'] = tde4.getLensFBackWidth(data['lens'])
            data['filmbackHeight'] = tde4.getLensFBackHeight(data['lens'])
            data['pixelAspect'] = tde4.getLensPixelAspect(data['lens'])
            data['lensOffsetX'] = tde4.getLensLensCenterX(data['lens'])
            data['lensOffsetY'] = tde4.getLensLensCenterY(data['lens'])
            data['fov'] = tde4.getCameraFOV(cam)

            seqFiles = os.path.split(tde4.getCameraPath(cam))[1]
            ext = seqFiles.split('.')[-1]
            data['fileName'] = seqFiles.replace(ext, 'jpg')
            data['jpgFile'] = os.path.join(data['filePath'], data['fileName'])

            tde4.setCameraProxyFootage(cam, 3)
            tde4.setCameraSequenceAttr(cam, data['seqAttr'][0], data['seqAttr'][1], data['seqAttr'][2])
            tde4.setCameraPath(cam, data['jpgFile'])
            tde4.setCameraProxyFootage(cam, 0)

            dxExportNuke.createNukePyForUndistortImage(self.req, cam, nukePyScriptFile, data, 1, pmodel=True, onlyScript=onlyScript)
            cont = tde4.updateProgressRequester(frame, "Undistorting, %s..." % data['camName'])

            if onlyScript == 0:
                cmds = [os.environ['DCCPROC'],
                        'rez-env',
                        'pylibs-2.7',
                        'openexr-2.4.2',
                        'baselib-2.5',
                        'ocio_configs-1.2',
                        'ldpk-2.4',
                        '--',
                        '/opt/Nuke12.2v4/Nuke12.2',
                        '--nukex',
                        '-i',
                        '-t',
                        os.path.join('/tmp', nukePyScriptFile)]
                # print('nuke_cmd', ' '.join(cmds))
                nuke_cmd = ' '.join(cmds)
                if not cont: break

                run = subprocess.Popen(nuke_cmd, shell=True)
                run.wait()

    def doIt(self):
        data = {}
        data['show'] = self.show
        data['seq'] = self.seq
        data['shot'] = self.shot
        data['plateType'] = self.plateType

        data['seqFile'] = tde4.getCameraPath(self.cam)
        data['cameraId'] = self.cam
        data['camName'] = tde4.getCameraName(self.cam)
        data['seqWidth'] = self.seqWidth
        data['seqHeight'] = self.seqHeight
        data['seqAttr'] = self.seqAttr

        data['seqColorspaceR'] = tde4.getWidgetValue(self.req, 'colorspaceR')
        data['seqColorspaceW'] = tde4.getWidgetValue(self.req, 'colorspaceW')
        data['startFrame'] = int(tde4.getWidgetValue(self.req, 'start_frame'))
        data['numFrames'] = tde4.getCameraNoFrames(self.cam)

        data['filePath'] = tde4.getWidgetValue(self.req, 'file_path')
        data['fileName'] = tde4.getWidgetValue(self.req, 'file_name')

        data['overscan'] = tde4.getWidgetValue(self.req, 'overscan')
        data['overscanValue'] = self.overscanValue
        data['overscanWidth'] = int(tde4.getWidgetValue(self.req, 'os_width'))
        data['overscanHeight'] = int(tde4.getWidgetValue(self.req, 'os_height'))

        data['jpgFile'] = os.path.join(data['filePath'], data['fileName'])
        data['jpgSize'] = tde4.getWidgetValue(self.req, 'size')
        data['burnIn'] = tde4.getWidgetValue(self.req, 'burnin')

        # get a camera, lens data.
        data['lens'] = tde4.getCameraLens(self.cam)
        data['lensLDModel'] = tde4.getLensLDModel(data['lens'])
        data['filmbackWidth'] = tde4.getLensFBackWidth(data['lens'])
        data['filmbackHeight'] = tde4.getLensFBackHeight(data['lens'])
        data['pixelAspect'] = tde4.getLensPixelAspect(data['lens'])
        data['lensOffsetX'] = tde4.getLensLensCenterX(data['lens'])
        data['lensOffsetY'] = tde4.getLensLensCenterY(data['lens'])
        data['fov'] = tde4.getCameraFOV(self.cam)

        # set undistorted image to camera proxy footage.
        tde4.setCameraProxyFootage(self.cam, 3)
        tde4.setCameraSequenceAttr(self.cam, self.seqAttr[0], self.seqAttr[1], self.seqAttr[2])
        tde4.setCameraPath(self.cam, data['jpgFile'])
        tde4.setCameraProxyFootage(self.cam, 0)

        sendTractor = tde4.getWidgetValue(self.req, 'send_tractor')
        # sendTractor = 0
        onlyScript = tde4.getWidgetValue(self.req, 'only_script')
        db_publish = tde4.getWidgetValue(self.req, 'database_publish')

        cmds = ['rez-env',
                'pylibs-2.7',
                'openexr-2.4.2',
                'baselib-2.5',
                'ocio_configs-1.2',
                'ldpk-2.4',
                '--show',
                self.show,
                '--',
                '/opt/Nuke12.2v4/Nuke12.2',
                '--nukex',
                '-i',
                '-t']

        if sendTractor == 1:
            job = author.Job()
            job.title = os.path.basename(tde4.getProjectPath())
            job.comment = 'sourcefile : ' + tde4.getProjectPath()
            job.service = 'nuke'
            job.maxactive = 0
            job.tier = 'comp'
            job.tags = ['2d']
            job.projects = ['comp']
            job.priority = 500

            rootTask = author.Task(title='MMV Imageplane')
            job.addChild(rootTask)

            # Notification Cmd
            jobMsgCmd = ['/backstage/dcc/DCC', 'rez-env', 'rocketchattoolkit', '--', 'TrBotMsg']
            job.newPostscript(argv=jobMsgCmd + ['-b', 'BadBot'], when='error')  # Error
            job.newPostscript(argv=jobMsgCmd + ['-b', 'GoodBot'], when='done')  # Done

        else:
            # write a nuke python script per frame.
            tde4.postProgressRequesterAndContinue(self.windowTitle, 'Undistorting, Frame 1...', self.frames, 'Stop')
            nukePyScriptFile = TDE4_common.valid_name(data['camName'])+'.py'

        for frame in range(1, data['numFrames']+1):
            if onlyScript == 0:
                if sendTractor == 1:
                    nukePyScriptFile = TDE4_common.valid_name(data['camName'] + '_%s' % str(frame+self.seqAttr[0]-1).zfill(4)) + '.py'
                    nkFile = dxExportNuke.createNukePyForUndistortImage(self.req, self.cam, nukePyScriptFile, data, frame, sendTractor=True)
                    nuke_cmd = ['/backstage/dcc/DCC'] + cmds + [nkFile]

                    task = author.Task(title='frame %s' % str(frame+self.seqAttr[0]-1))
                    task.addCommand(author.Command(argv=nuke_cmd, service='nuke'))
                    rootTask.addChild(task)
                else:
                    cont = tde4.updateProgressRequester(frame, "Undistorting, Frame %d..." % frame)
                    nkFile = dxExportNuke.createNukePyForUndistortImage(self.req, self.cam, nukePyScriptFile, data, frame)
                    nuke_cmd = [os.environ['DCCPROC']] + cmds + [nkFile]
                    nuke_cmd = ' '.join(nuke_cmd)
                    if not cont: break

                    run = subprocess.Popen(nuke_cmd, shell=True)
                    run.wait()
            else:   # onlyScript
                nkFile = dxExportNuke.createNukePyForUndistortImage(self.req, self.cam, nukePyScriptFile, data, frame, onlyScript=True)
                tde4.postQuestionRequester('Export Nuke...', '%s\nExport Only Script success!' %  nkFile,'Ok')
                break


        if sendTractor == 1:
            author.setEngineClientParam(hostname=TRACTOR_IP, port=PORT, user=getpass.getuser(), debug=True)
            job.spool()
            author.closeEngineClient()

        # TODO: ADD DATABASE RECORD
        if db_publish == 1 and onlyScript == 0:
            record = {}
            record['files'] = {}
            record['task_publish'] = {}

            record['show'] = self.show
            record['sequence'] = self.seq
            record['shot'] = self.shot
            record['data_type'] = 'imageplane'
            record['task'] = 'matchmove'
            record['tags'] = tag_parser.run(data['jpgFile'])
            record['artist'] = getpass.getuser()
            record['time'] = datetime.datetime.now().isoformat()

            record['files']['path'] = [data['jpgFile']]
            record['task_publish']['keying'] = data['jpgFile'].endswith('png')
            record['task_publish']['plateType'] = self.plateType

            if data['jpgSize'] == 1:
                record['task_publish']['hi_lo'] = 'hi'
            else:
                record['task_publish']['hi_lo'] = 'lo'

            if data['overscan'] == 1:
                record['task_publish']['overscan'] = True
                record['task_publish']['overscan_value'] = data['overscanValue']
                record['task_publish']['render_width'] = data['overscanWidth']
                record['task_publish']['render_height'] = data['overscanHeight']

            else:
                record['task_publish']['overscan'] = False
                record['task_publish']['render_width'] = self.seqWidth
                record['task_publish']['render_height'] = self.seqHeight

            record['task_publish']['start_frame'] = self.seqAttr[0]
            record['task_publish']['end_frame'] = self.seqAttr[1]

            record['version'] = insertDB.getLatestPubVersion(show= record['show'],
                                                             seq=record['sequence'],
                                                             shot=record['shot'],
                                                             data_type='imageplane',
                                                             plateType=record['task_publish']['plateType']
                                                             ) + 1
            client = MongoClient(DB_IP)
            db = client[DB_NAME]
            coll = db[record['show']]
            result = coll.insert_one(record)
            #print(result)

        tde4.unpostProgressRequester()
