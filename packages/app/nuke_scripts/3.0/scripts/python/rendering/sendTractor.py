# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#
#   Dexter CG Supervisor
#
#		daeseok.chae		daeseok.chae@dexterstudios.com
#
#	2020.10.29
#-------------------------------------------------------------------------------


from PySide2 import QtWidgets, QtGui, QtCore
from .ui_sendTractor import Ui_Dialog

import sys
import os
import time, datetime
import getpass
import json
import socket

import nuke, nukescripts, _nuke

import nukeCommon as comm

# if sys.platform == 'linux2':
#     sys.path.append('/backstage/apps/Tractor/applications/linux/Tractor-2.3/lib/python2.7/site-packages')

import tractor.api.author as author

def getIpAddress():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))  # GOOGLE PUBLIC DNS SERVER
    ip_addr = s.getsockname()[0]
    return ip_addr

class SendTractor(QtWidgets.QDialog):
    def __init__(self, parent=None):
        #UI setup
        QtWidgets.QDialog.__init__(self, parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.setWindowTitle('Send Tractor By Tae Hyung Lee, Dexter Digital')
        self.resize(650, 500)

        self.isqcOK = True
        self.isStereo = False

        self.writeNodeList = []

        self.authorizedUser = ["wonrak.son", "jinhee.lee", "hyebin.lee"]

        print("!!!!!")
        self.initSetting()
        self.setWriteNode()

        self.ui.checkBox.stateChanged.connect(self.inputChecked)
        self.ui.treeWidget.itemDoubleClicked.connect(self.writeDoubleClicked)
        self.ui.buttonBox.accepted.connect(self.makeAlf)
        self.ui.buttonBox.rejected.connect(self.reject)


    def getConnectedNodes(self, node=None):
        allDeps = set()

        if node:
            if type(node) == list:
                depsList = node
            else:
                depsList = [node]
        elif nuke.selectedNode():
            depsList = [nuke.selectedNode()]
        else:
            return

        evaluateAll = True
        while depsList:
            deps = nuke.dependencies(depsList, _nuke.INPUTS | _nuke.HIDDEN_INPUTS)
            deps += nuke.dependentNodes(_nuke.INPUTS | _nuke.HIDDEN_INPUTS, depsList, evaluateAll)
            evaluateAll = False
            depsList = [i for i in deps if i not in allDeps and not allDeps.add(i)]
        return allDeps

    def setWriteNode(self):
        jpgItem = None
        etcItem = None

        #------------------------------------------------------------------------------
        frameRange =  None
        if frameRange:
            scriptFrameRange = str(frameRange[0]) + ' - ' + str(frameRange[1])
        else:
            scriptFrameRange = None
        #------------------------------------------------------------------------------
        fullPath = nuke.value("root.name")
        if fullPath.startswith('/mach/show'):
            fullPath = fullPath.replace('/mach/show', '/show')

        writeNodes = [i for i in nuke.selectedNodes('Write') if not(i['disable'].value())]
        writeNodes += [i for i in nuke.selectedNodes('DeepWrite') if not(i['disable'].value())]

        writeNodes = sorted(writeNodes, key=lambda x: x['render_order'].value(), reverse=False)

        for node in writeNodes:
            if (os.path.splitext(node['file'].getEvaluatedValue())[-1]).lower() == '.jpg':
                if not jpgItem:
                    jpgItem = TopLevelItem(self.ui.treeWidget, 'JPG')
                writeItem = WriteItem(jpgItem, isJpg=True)
            else:
                if not etcItem:
                    etcItem = TopLevelItem(self.ui.treeWidget, 'ETC')
                writeItem = WriteItem(etcItem, isJpg=False)

            writeItem.setText(0, node.name())
            writeItem.setRenderOrder(node['render_order'].value())

            #------------------------------------------------------------------------------
            nodeFrameRange = str(node.firstFrame()) + ' - ' + str(node.lastFrame())
            writeItem.setText(2, nodeFrameRange)
            writeItem.setTextAlignment(2, QtCore.Qt.AlignCenter)
            if not(nodeFrameRange == scriptFrameRange):
                writeItem.setBackground(2, QtGui.QBrush(QtCore.Qt.darkRed))
                self.isqcOK = False
            #------------------------------------------------------------------------------
            nodeFormat = '%dx%d' %(node.format().width(), node.format().height())
            writeItem.setText(3, nodeFormat)
            writeItem.setTextAlignment(3, QtCore.Qt.AlignCenter)
            #------------------------------------------------------------------------------
            writeItem.setTextAlignment(4, QtCore.Qt.AlignCenter)
            channels = node.channels()
            layers = list( set([c.split('.')[0] for c in channels]) )
            layers.sort()
            writeItem.setText(4, str(len(layers)))
            #------------------------------------------------------------------------------
            try:
                nodeVersion = os.path.basename(node['file'].getEvaluatedValue()).split('.')[0].split('_')[-1]
                scVersion = os.path.splitext(os.path.basename(fullPath))[0].split('_')[-1]
                writeItem.setText(5, str(scVersion == nodeVersion))
                writeItem.setTextAlignment(5, QtCore.Qt.AlignCenter)

                if not(scVersion == nodeVersion):
                     writeItem.setBackground(5, QtGui.QBrush(QtCore.Qt.darkRed))
                     self.isqcOK = False
            except:

                writeItem.setText(5, 'Unknown')
            self.writeNodeList.append(writeItem)

        if jpgItem:
            jpgItem.setExpanded(True)
        if etcItem:
            etcItem.setExpanded(True)

        self.ui.treeWidget.sortItems(0, QtCore.Qt.AscendingOrder)
        #self.ui.treeWidget.sortItems(0, QtCore.Qt.DescendingOrder)
        #print(self.hasEqualRenderOrder(self.writeNodeList))


    def hasEqualRenderOrder(self, writeList):
        orderList = [i.getRenderOrder() for i in writeList]
        print(orderList)
        return orderList.count(orderList[0]) == len(orderList)

    def initSetting(self):
        #------------------------------------------------------------------------------
        self.widgetFont = self.font()
        self.widgetFont.setPointSize(10)
        self.setFont(self.widgetFont)
        #------------------------------------------------------------------------------

        # init
        self.ui.groupBox_2.setStyleSheet("""
        QGroupBox{
        margin-top: 8px;
        border: 2px solid green;
        }
        QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top center;
        }

        """)

        self.ui.groupBox_3.setStyleSheet("""
        QGroupBox{
        margin-top: 8px;
        border: 2px solid rgb(182,73,38);
        }
        QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top center;
        }
        """)

        serviceList = ['nuke', 'rsmb', 'twixtor', 'pg']
        self.ui.comboBox_2.addItems(serviceList)

        svc, serviceCount = self.checkService()
        self.ui.comboBox_2.setCurrentIndex(serviceList.index(svc))

        text = '<p style= "color:rgb(239,193,100)">rsmb : ' + str(serviceCount[0]) + '</p>\n'
        text += '<p style= "color:rgb(243,89,85)">twixtor : ' + str(serviceCount[1]) + '</p>\n'
        text += '<p style= "color:rgb(60,120,117)">pgbokeh : ' + str(serviceCount[2]) + '</p>\n'

        self.ui.serviceLabel.setText(text)

        self.ui.treeWidget.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)

        self.ui.treeWidget.setColumnCount(6)
        self.ui.treeWidget.headerItem().setText(0, 'Node Name')
        self.ui.treeWidget.headerItem().setText(1, 'Mov?')
        self.ui.treeWidget.headerItem().setText(2, 'Range')
        self.ui.treeWidget.headerItem().setText(3, 'Format')
        self.ui.treeWidget.headerItem().setText(4, 'Layers')
        self.ui.treeWidget.headerItem().setText(5, 'Version Sync')

        self.ui.treeWidget.header().resizeSection(0, 220)
        self.ui.treeWidget.header().resizeSection(1, 60)
        self.ui.treeWidget.header().resizeSection(2, 90)
        self.ui.treeWidget.header().resizeSection(3, 90)
        self.ui.treeWidget.header().resizeSection(4, 50)
        self.ui.treeWidget.header().resizeSection(5, 50)

        self.ui.treeWidget.setSortingEnabled(True)

        self.ui.spinBox.setMaximum(50000)
        self.ui.spinBox_2.setMaximum(50000)
        self.ui.spinBox.setValue(int(nuke.knob("first_frame")))
        self.ui.spinBox_2.setValue(int(nuke.knob("last_frame")))

        nkName = (nuke.root().name())

        if int(nuke.knob("last_frame")) - int(nuke.knob("first_frame")) > 100:
            self.ui.spinBox_3.setValue(4)
        elif 'fx' in nkName :
            self.ui.spinBox_3.setValue(1)
        elif 'lgt' in nkName :
            self.ui.spinBox_3.setValue(2)
        else:
            self.ui.spinBox_3.setValue(2)

        self.ui.spinBox_4.setValue(1)

        if not(getpass.getuser() in self.authorizedUser):
            self.ui.spinBox_4.setEnabled(False)

        # AUTO PLAY SETTING
        self.ui.checkBox_2.setChecked(True)

        #REAL FPS Setting
        print ("HI")
        try:
            fullPath = nuke.root().name()
            if fullPath.startswith('/mach/'):
                fullPath = fullPath.replace('/mach', '')

            splitFullPath = fullPath.split('/')
            showIndex = splitFullPath.index('show')
            self.prjName = splitFullPath[showIndex + 1]

            self.prjData = comm.getDxConfig()
            if self.prjData:
                fps = self.prjData['previewMOV']['fps']
                fpsIndex = self.ui.comboBox_3.findText(str(fps))
                if  -1 != fpsIndex:
                    self.ui.comboBox_3.setCurrentIndex(fpsIndex)
                else:
                    self.ui.comboBox_3.addItem(str(fps))
                    self.ui.comboBox_3.setCurrentText(str(fps))
        except Exception as e:
            print("# ERROR :", e.message)
            self.prjData = None
        print("OK")


    def checkService(self):
        #print(self.sese())
        #['rsmb', 'twixtor', 'pg']
        serviceCount = [0,0,0]
        svc = 'nuke'
        warning = False

        for i in self.getConnectedNodes(nuke.selectedNodes('Write') + nuke.selectedNodes('DeepWrite')):
            isDisabled = False
            if i.knob('disable'):
                isDisabled = i['disable'].value()

            if not(isDisabled):
                if i.Class() in ['OFXcom.revisionfx.clamptime_v1',
                                 'OFXcom.revisionfx.motionvectorscreate_v2',
                                 'OFXcom.revisionfx.twixtor_v5',
                                 'OFXcom.revisionfx.twixtorpro_v5',
                                 'OFXcom.revisionfx.twixtorvectorsin_v5']:
                    priority = 1
                    serviceCount[1] += 1

                elif i.Class() in ['OFXcom.revisionfx.rsmb_v3',
                                   'OFXcom.revisionfx.rsmb_vectors_v3'
                                   ]:
                    priority = '1'
                    serviceCount[0] += 1

                elif i.Class() in ['pgBokeh',
                                   'OFXuk.co.thefoundry.furnace.f_deflicker2_v403',
                                   'ZDefocus2',
                                   'ZBlur',
                                   'MotionBlur',
                                   'deep_to_depth'
                                   ]:
                    serviceCount[2] += 1

        if warning:
            nuke.message('please delete stamp node.')

        if serviceCount[1] > 0:
            svc = 'twixtor'
        elif serviceCount[0] >0:
            svc = 'rsmb'
#        elif serviceCount[2] >0:
#            svc = 'pg'

        return svc, serviceCount

    def makeAlf(self):
        engine = self.ui.comboBox_4.currentText()

        # priority = 500
        # tier = 'comp'
        # tags = ['2d']
        # service = 'nuke'
        # projects = ['comp']

        # if engine == '10.0.0.106':
        #     if 'mask' in os.path.basename(nuke.value("root.name")):
        #         priority = 0
        #     elif 'Mask' in os.path.basename(nuke.value("root.name")):
        #         priority = 0
        #     else:
        #         priority = int(self.ui.spinBox_4.value())
        #     tier = 'COMP'
        #     tags = ['team']
        #     envkey = ['nuke']
        #     service = str(self.ui.comboBox_2.currentText())
        #     if service == 'saph':
        #         priority += 1
        #     projects = ['comp']

        priority = 500
        if str(self.ui.comboBox_2.currentText()) == 'twixtor':
            tags = ['twixtor']
            tier = ''
        else:
            tags = ['2d']
            tier = 'comp'
        service = 'nuke'
        projects = ['comp']

        maxactive = 50
        configData = comm.getDxConfig()
        if configData:
            if 'tractor' in configData:
                if 'maxactive' in configData['tractor']:
                    maxactive = int(configData['tractor']['maxactive'])

        # original script
        nuke.scriptSave(nuke.root().name())
        print("make alf")
        # ALF FILE SETTING

        nukeDir = 'Nuke' + nuke.NUKE_VERSION_STRING
        nukeexec = nukeDir.split('v')[0]

        #-----------------------------------------------------------------------
        fullPath = nuke.value("root.name")
        if fullPath.startswith('/mach/show'):
            fullPath = fullPath.replace('/mach/show', '/show')

        nkScriptDir = os.path.dirname(fullPath)
        nkScriptFileName = os.path.basename(fullPath)
        fileName, extension = os.path.splitext(nkScriptFileName)
        now = datetime.datetime.now()
        stamp = now.strftime('%m%d%y_%H%M_%S')
        nkScriptFileName = '{FILENAME}--{ARTIST}-{TIME}{EXT}'.format(FILENAME=fileName, ARTIST=getpass.getuser(),
                                                                      TIME=stamp, EXT=extension)

        renderNkFile = os.path.join(nkScriptDir, 'Renderfarm_Submissions', nkScriptFileName)
        fps = str(self.ui.comboBox_3.currentText())

        if not(os.path.exists(os.path.dirname(renderNkFile))):
            os.umask(0)
            os.makedirs(os.path.dirname(renderNkFile))

        # render script
        nuke.scriptSave(renderNkFile)

        #
        deleteNodeList = []

        job = author.Job(title=os.path.basename(fullPath),
                         priority=priority,
                         maxactive=maxactive,
                         service=service,
                         tags=tags,
                         tier=tier,
                         projects=projects,
                         comment='RenderFile: %s' % nkScriptFileName
                         )

        jobMsgCmd = ['/backstage/dcc/DCC', 'rez-env', 'rocketchattoolkit', '--', 'TrBotMsg']
        job.newPostscript(argv=jobMsgCmd + ['-b', 'BadBot'], when='error')
        job.newPostscript(argv=jobMsgCmd + ['-b', 'GoodBot'], when='done')

        if not(self.hasEqualRenderOrder(self.writeNodeList)):
            emptyRootTask = author.Task(title='Empty for Serial')
            emptyRootTask.serialsubtasks = 1
            job.addChild(emptyRootTask)

            wnroDic = {}
            for wn in self.writeNodeList:
                writeNode = nuke.toNode(str(wn.text(0)))
                wnro = writeNode['render_order'].value()
                if wnroDic.get(wnro):
                    pass
                else:
                    roTask = author.Task(title='Render Order' + str(wnro))
                    emptyRootTask.addChild(roTask)
                    wnroDic[wnro] = roTask

        for item in self.writeNodeList:
            #slot
            if item.parent().checkBox.isChecked():
                slot = 1
            elif item.parent().checkBox2.isChecked():
                slot = 3

            writeNode = nuke.toNode(str(item.text(0)))
            isProres = False

            try:
                rawPath = writeNode['file'].value()

                if '[' in rawPath:
                    rawPath = writeNode['file'].getEvaluatedValue()
                    elements = os.path.basename(rawPath).split('.')
                    elements.pop(-2)
                    elements.insert(-1, '%04d')
                    paddfilename = '.'.join(elements)

                    rawPath = os.path.join(os.path.dirname(rawPath), paddfilename)
            except:
                # WRITE_ASD EXCEPTION
                outType = writeNode['out_type'].value()
                rawPath = writeNode[outType + '_path'].getEvaluatedValue()
                if outType == 'out_mov':
                    isProres = True

                elements = os.path.basename(rawPath).split('.')
                elements.pop(-2)
                elements.insert(-1, '%04d')
                paddfilename = '.'.join(elements)

                rawPath = os.path.join(os.path.dirname(rawPath), paddfilename)


            if rawPath.startswith('/netapp/dexter'):
                rawPath = rawPath.replace('/netapp/dexter', '')

            # MAKE DIR IF DIRECTORY NOT EXIST
            if not(os.path.exists(os.path.dirname(rawPath))):
                os.makedirs(os.path.dirname(rawPath))
            # if not(os.path.exists(os.path.dirname(writeNode['file'].value()))):
            #     os.makedirs(os.path.dirname(writeNode['file'].value()))

            if self.ui.checkBox.isChecked():
                startFrame = writeNode.firstFrame()
                endFrame = writeNode.lastFrame()
            else:
                startFrame = self.ui.spinBox.value()
                endFrame = self.ui.spinBox_2.value()

            framePerTask = self.ui.spinBox_3.value()

            # ADD CLEAN UP TASK
            try:
                viewList = writeNode['views'].value().split(' ')
            except:
                viewList = ['main']

            if len(viewList) > 1:
                self.isStereo = True
            else:
                self.isStereo = False
            outputPath = ''
            if item.isJpg:
                if item.movCheck.isChecked():
                    #viewList = writeNode['views'].value().split(' ')
                    rootJob = author.Task(title='FFMPEG MOV')
                    for j in viewList:
                        if "%V" in rawPath:
                            jpgFilepath = rawPath.replace("%V", j)
                        #------------------------------------------------------------------------------
                        elif "%v" in rawPath:
                            jpgFilepath = rawPath.replace("%v", j[0])
                        #------------------------------------------------------------------------------
                        else:
                            jpgFilepath = rawPath
                        print(jpgFilepath)

                        outputPath = os.path.join(os.path.dirname(os.path.dirname(jpgFilepath)),
                                                  '.'.join(os.path.basename(jpgFilepath).split('.')[:-2])
                                                  ) + '.mov'

                        movMetadata = self.getMovMetadata(renderNkFile, fullPath, writeNode.name(), rawPath)                            #-timecode 00:00:00:00
                        if isinstance(self.prjData, dict):
                            jobArg = self.movSetting(fps, jpgFilepath, movMetadata, outputPath, self.prjData['previewMOV']['codec'])
                        else:
                            jobArg = self.movSetting(fps, jpgFilepath, movMetadata, outputPath, 'proresProxy')

                        movCommand = author.Command(argv=jobArg)
                        rootJob.addCommand(movCommand)

                        # if auto play checked!
                        if self.ui.checkBox_2.isChecked():
                            ip = getIpAddress()
                            cmd = '/backstage/apps/bladeControl/nc_to_exe/rv_command.sh '
                            cmd += '%s %s' % (outputPath, ip)
                            ncCmd  = author.Command(argv=cmd)
                            rootJob.addCommand(ncCmd)
                else:
                    rootJob = author.Task(title='DONE')
            else:
                # exr case exr metadata add
                if rawPath.endswith('exr'):
                    if writeNode['metadata'].value() == 'no metadata':
                        pass
                    else:
                        writeNode['metadata'].setValue('all metadata')

                        #rmNode = nuke.createNode('ModifyMetaData')
                        rmNode = nuke.nodes.ModifyMetaData()
                        deleteNodeList.append(rmNode)
                        rmNode.setInput(0, writeNode.input(0))
                        rmMetaList = []

                        rmMetaList.append('{remove exr/nuke/renderNK ""}')
                        rmMetaList.append('{remove exr/nuke/saveNK ""}')
                        rmMetaList.append('{remove exr/nuke/wFilePath ""}')
                        rmMetaList.append('{remove exr/nuke/writeNode ""}')
                        rmMetaList.append('{remove exr/nuke/artist ""}')
                        rmNode['metadata'].fromScript('\n'.join(rmMetaList))

                        mmNode = nuke.nodes.ModifyMetaData()

                        deleteNodeList.append(mmNode)
                        mmNode.setInput(0, rmNode)

                        writeNode.setInput(0, mmNode)
                        metaList = []

                        metaList.append('{set %s %s}' % ('renderNK', renderNkFile))
                        metaList.append('{set %s %s}' % ('saveNK', fullPath))
                        metaList.append('{set %s %s}' % ('writeNode', writeNode.name()))
                        metaList.append('{set %s %s}' % ('wFilePath', rawPath))
                        metaList.append('{set %s %s}' % ('artist', getpass.getuser()))
                        mmNode['metadata'].fromScript('\n'.join(metaList))

                rootJob = author.Task(title='DONE')

                if item.screeningCheck.isChecked():
                    screeningPath = '/show/%s/screening/_to_screen/comp' % self.prjName
                    if not(os.path.exists(screeningPath)):
                        os.makedirs(screeningPath)

                    screeningCmd = author.Command(argv='cp -vrf %s %s' % (os.path.dirname(rawPath),
                                                                          screeningPath
                                                                          ))
                    screeningCmd.service = 'rsmb'
                    rootJob.addCommand(screeningCmd)

            for j in range(startFrame, endFrame+1,framePerTask):
                firstJobFrame = j
                if j+framePerTask > endFrame:
                    lastJobFrame = endFrame
                else:
                    lastJobFrame = j+framePerTask-1

                wname = writeNode.name()
                subTask = author.Task(title="%s, %s" % (firstJobFrame, lastJobFrame))

                command = ['/backstage/dcc/DCC', 'rez-env']

                # temporarily when using both: cent7, 8
                for package in os.environ['REZ_USED_RESOLVE'].split():
                    if 'centos' not in package:
                        command.append(package)
                # command += os.getenv('REZ_USED_REQUEST').split(' ')
                command += ['--show', self.prjName]
                command += ['--', 'nukeX', '-i', '-t']
                command += ['-F', '{START},{END}'.format(START=firstJobFrame, END=lastJobFrame)]
                command += ['-X', wname, renderNkFile]

                subCmd = author.Command(argv=command, service=service, atleast=slot, atmost=slot)
                subTask.addCommand(subCmd)

                rootJob.addChild(subTask)

            #------------------------------------------------------------------------------
            if self.hasEqualRenderOrder(self.writeNodeList):
                job.addChild(rootJob)
            else:
                #emptyRootTask.addChild(rootJob)
                roTask = wnroDic[writeNode['render_order'].value()]
                roTask.addChild(rootJob)

            # ADD COMMAND TO rootJob FOR INSERTING RENDER DB
            # FOR EACH WRITENODE
            # ONLY WORKS IF SCRIPT FILE IS UNDER /show/
            if isinstance(self.prjData, dict):
                if '/asset/' in fullPath:
                    splitFileName = fileName.split('_')
                    shotName = splitFileName[0]
                    context = splitFileName[1]
                    version = splitFileName[2]
                else:
                    splitFileName = fileName.split('_')
                    try:
                        shotName = splitFileName[0] + '_' + splitFileName[1]
                        context = splitFileName[2]
                        version = splitFileName[3]
                    except:
                        shotName = fileName.split('.')[0]
                        context = fileName.split('.')[0]
                        version = fileName.split('.')[0]


                dbdata = {}
                dbdata['platform'] = 'Nuke'
                dbdata['show'] = self.prjName
                dbdata['shot'] = shotName
                if '/CMP/' in fullPath or '/comp/' in fullPath:
                    process = 'comp'
                elif '/LNR/' in fullPath or '/lighting/' in fullPath:
                    process = 'lighting'
                elif '/PFX/' in fullPath or '/fx/' in fullPath:
                    process = 'fx'
                else:
                    process = 'unknown'

                dbdata['process'] = process
                dbdata['context'] = context
                dbdata['artist'] = getpass.getuser()

                if item.isJpg:
                    dbdata['files'] = {'render_path':[rawPath],
                                       'mov':outputPath,
                                       'render_nk':renderNkFile,
                                       'save_nk':fullPath
                                       }
                else:
                    dbdata['files'] = {'render_path':[rawPath],
                                       'mov':None,
                                       'render_nk':renderNkFile,
                                       'save_nk':fullPath
                                       }

                dbdata['version'] = version
                dbdata['time'] = datetime.datetime.now().isoformat()
                dbdata['is_stereo'] = self.isStereo
                dbdata['is_publish'] = False

                # dbdata['start_frame'] = int(writeNode.firstFrame())
                # dbdata['end_frame'] = int(writeNode.lastFrame())
                dbdata['start_frame'] = startFrame
                dbdata['end_frame'] = endFrame

                dbdata['write_node'] = writeNode.name()
                dbdata['ext'] = os.path.splitext(rawPath)[-1][1:]
                dbdata['render_from'] = 'farm'
                dbdata['service'] = service


                dbctx = json.dumps(dbdata,indent=4, separators=(',',' : '))
                dbJsonFile = os.path.splitext(renderNkFile)[0] + '_%s.json' % writeNode.name()

                f = open( dbJsonFile, 'w')
                f.write(dbctx)
                f.close()

                cmdArg = '/backstage/bin/DCC python-2 -- python /others/backstage/libs/python_lib/render_to_mongo.py '
                cmdArg += dbJsonFile
                dbInsertCmd = author.Command(argv=cmdArg)
                rootJob.addCommand(dbInsertCmd)

                jsonFileDeleteCmd = author.Command(argv='rm -f %s' % dbJsonFile)
                rootJob.addCommand(jsonFileDeleteCmd)

            #------------------------------------------------------------------------------

        nuke.scriptSave(renderNkFile)

        for delNode in deleteNodeList:
            nuke.delete(delNode)

        print(engine, '!!!!')
        job.spool(hostname=engine, port=80)

        self.accept()
        nuke.message('JOB HAS BEEN SENT')

    def getMovMetadata(self, renderNkFile, fullPath, writeNodeName, rawPath):
        movMetadata = '{\"renderNK\":\"%s\"\,\"saveNK\":\"%s\"\,\"writeNode\":\"%s\"\,\"wFilePath\":\"%s\"}' % (
        renderNkFile, fullPath, writeNodeName, rawPath)
        return movMetadata

    def movSetting(self, fps, renderPath, movMetadata, outputPath, codec):
        jobArg = '/backstage/dcc/DCC rez-env ffmpeg_toolkit -- ffmpeg_converter -r {FPS}'.format(FPS=fps)
        jobArg += ' -i {INPUTFILE} -o {OUTPUTFILE} -c {CODEC} -u {USER}'.format(INPUTFILE=renderPath, OUTPUTFILE=outputPath,
                                                                     CODEC=codec, USER=getpass.getuser())
        jobArg += ' -metadata nukeInfo=\'' + movMetadata + '\''
        return jobArg


    def inputChecked(self, state):
        if state == 2:
            self.ui.spinBox.setEnabled(False)
            self.ui.spinBox_2.setEnabled(False)
        else:
            self.ui.spinBox.setEnabled(True)
            self.ui.spinBox_2.setEnabled(True)

    def writeDoubleClicked(self, item, column):

        targetNode = nuke.toNode(str(item.text(0)))
        xCenter = targetNode.xpos() + targetNode.screenWidth()/2
        yCenter = targetNode.ypos() + targetNode.screenHeight()/2
        nuke.zoom( 3, [ xCenter, yCenter ])
        targetNode['selected'].setValue(True)


class WriteItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent=None, isJpg = False):
        super(WriteItem, self).__init__(parent)
        self.setTextAlignment(2, QtCore.Qt.AlignHCenter)

        self.isJpg = isJpg
        self.renderOrder = 1

        if self.isJpg:
            self.movCheck = QtWidgets.QCheckBox()
            self.movCheck.setText('Mov?')
            self.movCheck.setStyleSheet("""
            QCheckBox:checked {
            color: rgb(0,255,0);
            }
            """)
            self.treeWidget().setItemWidget( self, 1, self.movCheck)
            self.movCheck.setChecked(True)
        else:
            self.screeningCheck = QtWidgets.QCheckBox()
            self.screeningCheck.setText('Screen?')
            self.screeningCheck.setStyleSheet("""
            QCheckBox:checked {
            color: rgb(0,255,0);
            }
            """)
            self.treeWidget().setItemWidget( self, 1, self.screeningCheck)
            self.screeningCheck.setChecked(False)

    def getRenderOrder(self):
        return self.renderOrder

    def setRenderOrder(self, order):
        self.renderOrder = order


class TopLevelItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent=None, name = None):
        super(TopLevelItem, self).__init__(parent)

        bigFont = QtGui.QFont()
        bigFont.setPointSize(16)
        bigFont.setBold(True)
        self.setFont(0, bigFont)

        self.container = QtWidgets.QWidget(self.treeWidget())
        self.gridLayout = QtWidgets.QGridLayout(self.container)

        self.nameText = QtWidgets.QLabel(self.container)
        self.nameText.setFont(bigFont)
        self.nameText.setText(name)
        self.gridLayout.addWidget(self.nameText, 0, 0, 1, 1)

        self.checkBox = QtWidgets.QRadioButton(self.container)
        self.checkBox.setText('Light')
        self.gridLayout.addWidget(self.checkBox, 0, 1, 1, 1)

        self.checkBox2 = QtWidgets.QRadioButton(self.container)
        self.checkBox2.setText('Heavy')
        self.gridLayout.addWidget(self.checkBox2, 0, 2, 1, 1)

        buttonStyleGreen = """
        QRadioButton:checked {
        color: rgb(0,255,0);
        }
        """

        self.checkBox.setStyleSheet(buttonStyleGreen)
        self.checkBox2.setStyleSheet(buttonStyleGreen)
        self.checkBox.setChecked(True)

        self.treeWidget().setItemWidget( self, 0, self.container )

    def __lt__(self, otherItem):
        column = self.treeWidget().sortColumn()
        if column == 0:
            itemName = self.nameText.text()
            otherName = otherItem.nameText.text()
            if str(itemName) == 'JPG':
                return True
            else:
                return itemName > otherName

def sendtoTractor():
    print("sendtoTractor")
    fullPath = nuke.value("root.name")
    if fullPath.startswith('/mach/show'):
        fullPath = fullPath.replace('/mach/show', '/show')

    if nuke.root()['proxy'].value():
        if nuke.ask("Disable Proxy Mode??"):
            nuke.root()['proxy'].setValue(False)
        else:
            pass

    # opticalFlares error messangeBox
    if nuke.allNodes('OpticalFlares'):
    # if [i for i in nuke.allNodes('OpticalFlares') if i['disable'].value() == False]:
        mBox = QtWidgets.QMessageBox(QtWidgets.QApplication.activeWindow())
        mBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
        mBox.setWindowTitle("WARNING!!!")
        # mBox.setText(u'작업노드에 옵티컬 플레어가 포함되어 있습니다.\n\n옵티컬 플레어를 디스에이블 하고 다시 시도하거나\n\n로컬로 렌더를 걸어주세요')
        mBox.setText(u'작업노드에 옵티컬 플레어가 포함되어 있습니다.\n\n옵티컬 플레어를 삭제하고 다시 시도하거나\n\n로컬로 렌더를 걸어주세요')
        answer = mBox.exec_()

        if answer == QtWidgets.QMessageBox.Ok:
            return False

    try:
        findMeta = nuke.selectedNode().metadata()
        if not 'input/timecode' in findMeta:
            mBox = QtWidgets.QMessageBox(QtWidgets.QApplication.activeWindow())
            mBox.setStandardButtons(QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No)
            mBox.setWindowTitle("WARNING!!!")
            mBox.setText(u'타임코드 데이타가 없습니다.\n\n타임코드 없이 렌더를 진행할까요?')
            answer = mBox.exec_()

            if answer == QtWidgets.QMessageBox.No:
                return
    except:
        pass

    """
    if frameRange:
        frame_in = str(frameRange[0])
        frame_out = str(frameRange[1])

        isFrameMatch = True
        infoText = u'<font size=8 color="Green">Tactic : %s ~ %s </font><br/>' % (frame_in, frame_out)

        for node in nuke.selectedNodes():
            nodeIn = str(node.firstFrame())
            nodeOut = str(node.lastFrame())
            if (frame_in == nodeIn) and (frame_out == nodeOut):
                infoText += u'<font size=6 color="Green">%s : %s ~ %s </font><br/>' % (node.name(), nodeIn, nodeOut)
            else:
                isFrameMatch = False
                infoText += u'<font size=6 color="Red">Duration Not Match</font><br/>'
                infoText += u'<font size=6 color="Red">%s : %s ~ %s </font><br/>' % (node.name(), nodeIn, nodeOut)


        if isFrameMatch:
            pass
        else:
            mBox = QtGui.QMessageBox(QtGui.QApplication.activeWindow())
            #mBox.setStandardButtons(QtGui.QMessageBox.Yes|QtGui.QMessageBox.No)
            mBox.setText(infoText)
            answer = mBox.exec_()
    """
    global send_window
    send_window = SendTractor(QtWidgets.QApplication.activeWindow())
    send_window.show()
    """
    if nuke.root()['proxy'].value():
        if nuke.ask("Disable Proxy Mode??"):
            nuke.root()['proxy'].setValue(False)
        else:
            pass
    global send_window

    send_window = SendTractor(QtGui.QApplication.activeWindow())
    send_window.show()
    """
