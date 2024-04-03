# -*- coding: utf-8 -*-
from PySide2 import QtWidgets, QtCore

import nuke
import nukescripts
import os
import re
import subprocess
import threading
import time, datetime
import getpass
from dxConfig import dxConfig

from dxname import tag_parser

_gRenderDialogState = nukescripts.DialogState()
_gFlipbookDialogState = nukescripts.DialogState()

try:
    import pymongo
    from pymongo import MongoClient
    DB_IP = dxConfig.getConf('DB_IP')
    DB_NAME = 'PIPE_PUB'
except:
    import sys
    sys.path.insert(0, '/backstage/lib/python_lib_2.6')
    import pymongo
    from pymongo import MongoClient
    DB_IP = dxConfig.getConf('DB_IP')
    DB_NAME = 'PIPE_PUB'

def getLatestPubVersion(show, seq, shot, data_type,plateType=None):
    client = MongoClient(DB_IP)
    db = client[DB_NAME]
    coll = db[show]
    if plateType:
        recentDoc = coll.find_one({'show': show,
                                   'shot': shot,
                                   'data_type': data_type,
                                   'task_publish.plateType':plateType},
                                  sort=[('version', pymongo.DESCENDING)])
    else:
        recentDoc = coll.find_one({'show': show,
                                   'shot': shot,
                                   'data_type': data_type},
                                  sort=[('version', pymongo.DESCENDING)])

    if recentDoc:
        return recentDoc['version']
    else:
        return 0

def getTop(node):
    if node.input(0):
        node = getTop(node.input(0))
    else:
        print("no input")
    return node

def checkIntegrity(filePath, seq, shot, plate):
    ele = re.split('[.]|_', os.path.basename(filePath))
    file_seq = ele[0]
    file_shot = '_'.join([seq, ele[1]])
    file_plate = ele[2]
    return (seq == file_seq) & (shot == file_shot) & (plate == file_plate)


def showRenderDialog(nodesToRender, exceptOnError = True, allowFrameServer=True):
    """Present a dialog that renders the given list of nodes."""
    # TODO : allowFrameServer must implement when Nuke 11 is ready to use.
    groupContext = nuke.root()
    d = HslthRenderDialog(_gRenderDialogState, groupContext, nodesToRender, exceptOnError)
    if d.showModalDialog() == True:
        d.run()

class FfmpegThread(threading.Thread):
    def __init__(self, command):
        threading.Thread.__init__(self)
        self.command = command.split(' ')

    def run(self):
        print("starting command: ", self.command)
        subprocess.Popen(self.command).wait()


def getMovMetadata(renderNkFile, fullPath, writeNodeName, rawPath):
    movMetadata = '{\"renderNK\":\"%s\"\,\"saveNK\":\"%s\"\,\"writeNode\":\"%s\"\,\"wFilePath\":\"%s\"}' % (
    renderNkFile, fullPath, writeNodeName, rawPath)
    return movMetadata

class HslthRenderDialog(nukescripts.RenderDialog):
    def __init__(self, dialogState, groupContext, nodeSelection = [], exceptOnError = True):
        super(HslthRenderDialog, self).__init__(dialogState, groupContext, nodeSelection = [], exceptOnError = True)

        print("hslth!!")
        self._nodeSelection = nodeSelection
        """
        if nuke.selectedNode():
            self._nodeSelection = nuke.selectedNodes()

        else:
            self._nodeSelection = [nuke.thisNode()]
        """
        # CREATE FOLDER IF NOT EXISTS
        for i in self._nodeSelection:
            if os.path.exists(os.path.dirname(i['file'].getEvaluatedValue())):
                pass
            else:
                os.makedirs(os.path.dirname(i['file'].getEvaluatedValue()))

        self.makeMovknob = nuke.Boolean_Knob('makeMov', 'Make MOV?')

        self.makeMovknob.setFlag(nuke.STARTLINE)
        self.addKnob(self.makeMovknob)

        self.impPublish = None
        if 'ImagePlane' in nuke.thisNode().name():
            # SHOW SEQ SHOT PLATE CHECK BASED FILE PATH AND KNOB.
            # SOME ARTISTS COPT WRITE NODE AND MODIFY ONLY FILEPATH.
            wf = nuke.thisNode()['file'].getEvaluatedValue()
            seq = nuke.thisNode().knob('seq').value()
            shot = nuke.thisNode().knob('shot').value()
            plate = nuke.thisNode().knob('plate').value()
            if checkIntegrity(filePath=wf, seq=seq, shot=shot, plate=plate):
                self.impPublish = nuke.Boolean_Knob('ImagePlane', 'ImagePlane Publish?')
                self.impPublish.setFlag(nuke.STARTLINE)
                self.addKnob(self.impPublish)
            else:
                nuke.message('SEQ, SHOT, PLATE NOT MATCHING.')
                self.impPublish = nuke.Boolean_Knob('ImagePlane', 'ImagePlane Publish?')
                self.impPublish.setFlag(nuke.STARTLINE)
                self.addKnob(self.impPublish)

        self.selectedPath = self._nodeSelection[0]['file'].getEvaluatedValue()

        #self.defaultPathList = [os.path.dirname(self.selectedPath), self.setDefaultPaht(self._nodeSelection[0]['file'])]
        self.defaultPathList = [os.path.dirname(os.path.dirname(self.selectedPath))]
        self.defaultPathList.append("...")

        if self.ismovKnob():
            print("get value from userknob")
            self.usermovKnob = self._nodeSelection[0]['usermov']
            self.defaultPathList = self._nodeSelection[0]['usermov'].values() + self.defaultPathList

        self.movPathKnob = nuke.Enumeration_Knob('movPath', 'Mov path : ', self.defaultPathList)
        self.addKnob(self.movPathKnob)
        self.movPathKnob.setVisible(False)

    def setDefaultPath(self, writePath):
        evaluatedPath = writePath.getEvaluatedValue()
        pad = re.search('(\d+)\.\w+$', evaluatedPath)
        padNum = str(pad.group(1))
        jpgFilepath = evaluatedPath.split(padNum)[0] + '%04d' + evaluatedPath.split(padNum)[-1]
        return jpgFilepath

    def ismovKnob(self):
        if self._nodeSelection[0].knob('usermov'):
            return True
        else:
            return False

    def makeUsermovKnob(self):
        self.usermovKnob = nuke.Enumeration_Knob('usermov', 'USERMOV', [])
        self.usermovKnob.setFlag(nuke.INVISIBLE)

        self._nodeSelection[0].addKnob(self.usermovKnob)
        self._nodeSelection[0].knob('User').setFlag(nuke.INVISIBLE)
        self._nodeSelection[0]['file'].setFlag(0)

    def _titleString(self):
        return "Dexter Comp Render"

    def knobChanged( self, knob ):
        nukescripts.ExecuteDialog.knobChanged(self, knob)
        if (knob == self._bgRender):
            self._numThreads.setVisible(self._bgRender.value())
            self._maxMem.setVisible(self._bgRender.value())
        if (knob == self.makeMovknob):
            self.movPathKnob.setVisible(self.makeMovknob.value())

        if (knob == self.movPathKnob):
            if self.movPathKnob.value() == "...":
                if self.ismovKnob():
                    print("usermov exists")
                else:
                    self.makeUsermovKnob()

                selectedPath = nuke.getFilename("SEO file dialog")
                if not(selectedPath == None):
                    if selectedPath in self.defaultPathList:
                        # move selectedPath first
                        self.defaultPathList.remove(selectedPath)

                    self.defaultPathList.insert(0,selectedPath)
                    self.movPathKnob.setValues(self.defaultPathList)
                    self.movPathKnob.setValue(0)

                    self.usermovKnob.setValues(list(set([selectedPath]) | set(self.usermovKnob.values())))
                    #self.usermovKnob.setValues([selectedPath] + self.usermovKnob.values())
                else:
                    self.movPathKnob.setValue(0)

    def run(self):
        renderNkFile = nuke.root().name()
        frame_ranges = nuke.FrameRanges(self._frameRange.value().split(','))
        views = self._selectedViews()
        rootProxyMode = nuke.root().proxy()
        isExr = False
        nuke.root().setModified(True)

        # SSSS metadata parsing
        try:
            projectName = nuke.value("root.name").split('/')[-8]
            findMeta = nuke.selectedNode().metadata()
            if projectName == 'ssss' and not 'input/timecode' in findMeta:
                mBox = QtWidgets.QMessageBox(QtWidgets.QApplication.activeWindow())
                mBox.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
                mBox.setWindowTitle("WARNING!!!")
                mBox.setText(u'타임코드 데이타가 없습니다.\n\n타임코드 없이 렌더를 진행할까요?')
                answer = mBox.exec_()

                if answer == QtWidgets.QMessageBox.No:
                    return
        except:
            pass

        # if nuke.root().name() == 'Root':
        #     dupName = 'Untitled_' + time.strftime("%y%m%d.%H%M%S") + '.nk'
        # else:
        #     baseFileName = os.path.splitext(os.path.basename(fullPath))[0]
        #     dupName = baseFileName + '_' + time.strftime("%y%m%d.%H%M%S") + '.nk'
        """
        if (nuke.thisNode().Class() == 'Write') and (nuke.thisNode()['file_type'].value() == 'exr'):
            isExr = True
            print("render exr!!!")

            rsPath = '/netapp/dexter/render_script/LOCAL/'
            if not(os.path.exists(rsPath)):
                os.makedirs(rsPath)
            fullPath = nuke.root().name()
            writeNode = nuke.thisNode()

            if nuke.root().name() == 'Root':
                dupName = 'Untitled_' + time.strftime("%y%m%d.%H%M%S") + '.nk'
            else:
                baseFileName = os.path.splitext(os.path.basename(fullPath))[0]
                dupName = baseFileName + '_' + time.strftime("%y%m%d.%H%M%S") + '.nk'


            renderNkFile = rsPath + dupName
            nuke.scriptSave(renderNkFile)

            rawPath = writeNode['file'].getEvaluatedValue()

            #rmNode = nuke.createNode('ModifyMetaData')
            rmNode = nuke.nodes.ModifyMetaData()

            rmNode.setInput(0, writeNode.input(0))
            rmMetaList = []

            rmMetaList.append('{remove exr/nuke/renderNK ""}')
            rmMetaList.append('{remove exr/nuke/saveNK ""}')
            rmMetaList.append('{remove exr/nuke/wFilePath ""}')
            rmMetaList.append('{remove exr/nuke/writeNode ""}')
            rmMetaList.append('{remove exr/nuke/artist ""}')
            rmNode['metadata'].fromScript('\n'.join(rmMetaList))

            #mmNode = nuke.createNode('ModifyMetaData')
            mmNode = nuke.nodes.ModifyMetaData()
            mmNode.setInput(0, rmNode)

            writeNode.setInput(0, mmNode)
            metaList = []

            metaList.append('{set %s %s}' % ('renderNK', renderNkFile))
            metaList.append('{set %s %s}' % ('saveNK', fullPath))
            metaList.append('{set %s %s}' % ('writeNode', writeNode.name()))
            metaList.append('{set %s %s}' % ('wFilePath', rawPath))
            metaList.append('{set %s %s}' % ('artist', getpass.getuser()))
            mmNode['metadata'].fromScript('\n'.join(metaList))
        """
        try:
            nuke.Undo().disable()
            nuke.root().setProxy(self._useProxy.value())
            if (self.isBackgrounded()):
                print("backgrounded!!")
                print(self._nodeSelection, frame_ranges, views, self._continueOnError.value())
                nuke.executeBackgroundNuke(nuke.EXE_PATH, self._nodeSelection,
                                           frame_ranges, views, self._getBackgroundLimits(), continueOnError = self._continueOnError.value())
            else:
                print("just render!!")
                print(self._nodeSelection, frame_ranges, views, self._continueOnError.value())
                nuke.executeMultiple(self._nodeSelection, frame_ranges, views, continueOnError = self._continueOnError.value())
        except RuntimeError as e:
            if self._exceptOnError or e.args[0][0:9] != "Cancelled":   # TO DO: change this to an exception type
                raise
        finally:
            nuke.root().setProxy(rootProxyMode)
            nuke.Undo().enable()
            print("final script!!!!")
            if isExr:
                nuke.delete(rmNode)
                nuke.delete(mmNode)

        if self.makeMovknob.value():
            fullPath = nuke.value("root.name")
            if fullPath.startswith('/netapp/dexter'):
                fullPath = fullPath.replace('/netapp/dexter', '')

            pathElement = fullPath.split('/')
            # /mach/show/slc/works/CMP/MET/MET_0160/comp/MET_0160_comp_v001.nk
            try:
                projectName = pathElement[pathElement.index('show')+1]
                sequenceName = pathElement[pathElement.index('CMP')+1]
                shotName = pathElement[pathElement.index(sequenceName)+1]
                fileName = pathElement[-1][:-3]
                process = pathElement[pathElement.index(shotName)+1]
                context = pathElement[pathElement.index(process)+1]
                if '.nk' in context:
                    context = ''
            except:
                projectName = pathElement[pathElement.index('show')+1]
                sequenceName = pathElement[pathElement.index('CMP')+1]
                shotName = pathElement[pathElement.index(sequenceName)+1]
                fileName = pathElement[-1][:-3]
                process = ''
                context = ''

            if process == 'comp':
                pathElement.insert(-1, 'render_script')
                pathElement.pop()
                pathElement.append(fileName)
                renderNkFile = '/'.join(pathElement) + '_' + time.strftime("%y%m%d.%H%M%S") + '.nk'

                if not (os.path.exists(os.path.dirname(renderNkFile))):
                    os.makedirs(os.path.dirname(renderNkFile))
                nuke.scriptSave(renderNkFile)

            print("NODE NAME", nuke.thisNode().name())
            rawPath = nuke.thisNode()['file'].getEvaluatedValue()
            elements = os.path.basename(rawPath).split('.')
            elements.pop(-2)
            elements.insert(-1, '%04d')
            paddfilename = '.'.join(elements)
            rawPath = os.path.join(os.path.dirname(rawPath), paddfilename)

            fps = nuke.root()['fps'].value()
            viewList = nuke.thisNode()['views'].value().split(' ')
            if len(viewList) > 1:
                isStereo = True
            else:
                isStereo = False
            threadList = []

            codec = 'proresProxy'
            configData = comm.getDxConfig()
            if configData:
                codec = configData['deliveryMOV']['codec']

            for i in viewList:
                if "%V" in rawPath:
                    jpgFilepath = rawPath.replace("%V", i)
                else:
                    jpgFilepath = rawPath

                outputPath = os.path.join(self.movPathKnob.value(),
                                          jpgFilepath.split('/')[-1].split('.%04d.')[0]) + '.mov'

                startNum = self._frameRange.value().split(',')[0].split('-')[0]
                endNum = self._frameRange.value().split(',')[0].split('-')[-1]

                if nuke.thisNode().name().startswith('out_mov'):
                    ffmpegCommand = os.environ['DCCPROC'] + ' rez-env ffmpeg_toolkit -- ffmpeg_converter -r {FPS}'.format(FPS=fps)
                    ffmpegCommand += ' -i {INPUTFILE} -o {OUTPUTFILE} -c {CODEC}'.format(INPUTFILE=jpgFilepath, OUTPUTFILE=outputPath,
                                                                                 CODEC=codec)

                else:
                    movMetadata = getMovMetadata(renderNkFile, fullPath, nuke.thisNode().name(), rawPath)
                    ffmpegCommand = os.environ['DCCPROC'] + ' rez-env ffmpeg_toolkit -- ffmpeg_converter -r {FPS}'.format(FPS=fps)
                    ffmpegCommand += ' -i {INPUTFILE} -o {OUTPUTFILE} -c {CODEC}'.format(INPUTFILE=jpgFilepath, OUTPUTFILE=outputPath,
                                                                                 CODEC=codec)
                    ffmpegCommand += ' -metadata nukeInfo=\'' + movMetadata + '\''

            print('ffmpeg_cmd:', ffmpegCommand)
            threadObject = FfmpegThread(ffmpegCommand)
            threadObject.start()
            threadList.append(threadObject)

            for i in threadList:
                i.join()

            # DB RECORD
            dbdata = {}
            dbdata['platform'] = 'Nuke'
            dbdata['show'] = projectName
            dbdata['shot'] = shotName
            dbdata['process'] = process
            dbdata['context'] = context
            dbdata['artist'] = getpass.getuser()


            dbdata['files'] = {'render_path': [rawPath],
                               'mov': outputPath,
                               'render_nk': renderNkFile,
                               'save_nk': fullPath
                               }

            dbdata['version'] = fileName.split('_')[-1]
            dbdata['time'] = datetime.datetime.now().isoformat()
            dbdata['is_stereo'] = isStereo
            dbdata['is_publish'] = False
            dbdata['start_frame'] = int(startNum)
            dbdata['end_frame'] = int(endNum)
            dbdata['write_node'] = nuke.thisParent().name()

            dbdata['ext'] = os.path.splitext(rawPath)[-1][1:]
            dbdata['render_from'] = 'local'

            nuke.thisParent().end()

            if nuke.allNodes('OFXcom.revisionfx.twixtor_v5'):
                dbdata['service'] = 'twixtor'
                print("twixtor!!!!")
            elif nuke.allNodes('OFXcom.revisionfx.twixtorpro_v5'):
                dbdata['service'] = 'twixtor'
                print("twixtor!!!!")
            elif nuke.allNodes('OFXcom.revisionfx.twixtorvectorsin_v5'):
                dbdata['service'] = 'twixtor'
                print("twixtor!!!!")
            elif nuke.allNodes('OFXcom.revisionfx.rsmb_v3'):
                dbdata['service'] = 'rsmb'
                print("rsmb!!!!")
            else:
                dbdata['service'] = 'nuke'
                print("nuke!!!!")


            client = MongoClient(DB_IP)
            db = client.RENDER
            coll = db[projectName]
            print(dbdata)

            coll.insert_one(dbdata)


            # DB RECORD DONE

            nuke.message("Mov file Done")
            print("thread finished")


        if self.impPublish:
            if self.impPublish.value():
                # PUBLISH MONGODB
                print("publish image plane DB")

                # GET SHOW SHOT INFO FROM KNOB
                if nuke.thisNode().knob('show'):
                    imgSeq = nuke.thisNode()['file'].getEvaluatedValue()
                    readNode = getTop(nuke.thisNode())

                    record = {}
                    record['files'] = {}
                    record['task_publish'] = {}

                    record['show'] = nuke.thisNode().knob('show').value()
                    record['sequence'] = nuke.thisNode().knob('seq').value()
                    record['shot'] = nuke.thisNode().knob('shot').value()
                    record['data_type'] = 'imageplane'
                    record['task'] = 'matchmove'
                    record['tags'] = tag_parser.run(imgSeq)
                    record['artist'] = getpass.getuser()
                    record['time'] = datetime.datetime.now().isoformat()

                    record['files']['path'] = [imgSeq]
                    record['task_publish']['keying'] = imgSeq.endswith('png')
                    record['task_publish']['plateType'] = nuke.thisNode().knob('plate').value()
                    record['task_publish']['start_frame'] = nuke.thisNode().firstFrame()
                    record['task_publish']['end_frame'] = nuke.thisNode().lastFrame()
                    # WIDTH? HEIGHT?
                    rh = readNode.height()
                    rw = readNode.width()
                    wh = nuke.thisNode().height()
                    ww = nuke.thisNode().width()

                    record['task_publish']['render_width'] = ww
                    record['task_publish']['render_height'] = wh

                    # OVERSCAN CHECK BY RESOLUTION COMPARE
                    # HIGH or LOW CHECK BY RESOLUTION COMPARE

                    if ww > rw:
                        overscan = True
                        hi_lo = 'hi'
                    else:
                        if ww < rw:
                            # IF WRITE WIDTH < READ WIDTH:
                            hi_lo = 'lo'

                        else:
                            # IF WRITE WIDTH == READ WIDTH:
                            hi_lo = 'hi'
                        overscan = False
                    record['task_publish']['overscan'] = overscan
                    record['task_publish']['hi_lo'] = hi_lo

                    record['version'] = getLatestPubVersion(show= record['show'],
                                                            seq=record['sequence'],
                                                            shot=record['shot'],
                                                            data_type='imageplane',
                                                            plateType=record['task_publish']['plateType']
                                                            ) + 1
                    client = MongoClient(DB_IP)
                    db = client[DB_NAME]
                    coll = db[record['show']]
                    result = coll.insert_one(record)
                    print(result)

                else:
                    # IF NOT GET INFO FROM FILE PATH
                    pass
