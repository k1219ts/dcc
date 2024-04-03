# -*- coding: utf-8 -*-
import sys, os, time, webbrowser

from rv import commands, rvtypes, runtime, qtutils
from pymu import MuSymbol

from PySide2 import QtWidgets, QtCore

import dxVersioning_api as api
import dxTacticCommon
import getpass
import dxConfig
from dxstats import inc_tool_by_user as log
import pprint


CONFLUENCE_IP = '10.10.10.117:8090'

# define hotKey
# ---------------------------------
KEY_TOGGLE_VERSION_VIEW  = 'V'
KEY_TASK_CHANGE_BACKWARD = 'shift--left'
KEY_TASK_CHANGE_FORWARD  = 'shift--right'
KEY_VERSION_CHANGE_UP    = 'shift--up'
KEY_VERSION_CHANGE_DOWN  = 'shift--down'
KEY_MULTISHOT_EXTENSION  = '"'
KEY_RELOAD_NEW_SNAPSHOT  = '`'
SAVE_MOV_LIST            = 'S'
# ---------------------------------


class VersioningMode(rvtypes.MinorMode):
    def __init__(self):
        rvtypes.MinorMode.__init__(self)

        self.taskList = ['edit',
                         'matchmove', 'animation',
                         'lighting', 'fx', 'comp']
        self.show = ''
        self.current = 0
        self.last = 0

        self.eldOrder = [0]
        self.widgets = qtutils.sessionWindow()

        self.init("versioning-mode",
        [
            ("key-down--%s" % KEY_VERSION_CHANGE_UP, self.changeVersion, "change version backward"),
            ("key-down--%s" % KEY_VERSION_CHANGE_DOWN, self.changeVersion, "change version forward"),
            ("key-down--%s" % KEY_TOGGLE_VERSION_VIEW, self.toggleDxSelector, "tactic data query"),
            ("key-down--%s" % KEY_TASK_CHANGE_BACKWARD, self.changeTask, "change task backward"),
            ("key-down--%s" % KEY_TASK_CHANGE_FORWARD, self.changeTask, "change task forward"),
            ("key-down--%s" % KEY_MULTISHOT_EXTENSION, self.multiShotExtension, "multiShot Extension"),
            ("key-down--%s" % KEY_RELOAD_NEW_SNAPSHOT, self.reloadSnapShot, "Reload New Snapshot"),
            ("key-down--%s" % SAVE_MOV_LIST, self.saveMovList, "save mov List"),
            ("play-start", self.SelectorHide, 'selector hide'),
            ],
            None,
	    [("Dexter", [
                        ("Dexter RV User's Guide.", self.openDexterGuide, "", None),
                        ("select Snapshot Ver.", self.toggleDxSelector, "", None),
                        ("_", None, "", None),
                        ])
	    ])

        self._syncActivated = False

    def saveMovList(self, event):
        if '.rv' not in commands.sessionFileName():
            self.MessagePopup('지금 열려 있는 파일은 rv session File이 아닙니다.')
            return

        filepath = commands.sessionFileName().replace('.rv', '.txt')
        fj = open(filepath, "w")
        for i in commands.nodesInGroup('defaultSequence'):
            if 'defaultSequence_a_' in i:
                node = i.replace('defaultSequence_a_', '') + '_source'
                print(i, node)
                info = commands.sourceMediaInfo(node)
                # pprint.pprint(info)

                tmp = info['file'].split('/')
                shotName = tmp[tmp.index('shot')+1]
                movFile = os.path.basename(info['file'])
                print(shotName)
                fj.write(shotName + '\t' + movFile + '\n')
        fj.close()
        self.MessagePopup('샷 네임이 저장되었습니다.\n%s' % filepath)

    def reloadSnapShot(self, event):
        print('reload new snapshot!!')
        if not self.chkMedia():
            return

        result = self.MessagePopupOkCancel('최신 snapshot으로 update 하시겠습니까?')
        if result != QtWidgets.QMessageBox.Ok:
            return

        changedShot = ''
        for i in commands.nodesInGroup('defaultSequence'):
            if 'defaultSequence_a_' in i:
                node = i.replace('defaultSequence_a_', '') + '_source'
                info = commands.sourceMediaInfo(node)

                mediaPath = os.path.dirname(info['file'])
                filesList = []

                for file in os.listdir(mediaPath):
                    if '.mov' in file:
                        time = os.path.getctime(os.path.join(mediaPath, file))
                        filesList.append((file, time))
                filesList = sorted(filesList, key=lambda x: x[1], reverse=True)
                latestMediaFile = os.path.join(mediaPath, filesList[0][0])

                if info['file'] != latestMediaFile:
                    # print 'node:', node, '---------------------'
                    # print 'now:', info['file']
                    # print 'latest:', latestMediaFile

                    commands.setSourceMedia(node, [latestMediaFile])

                    tmp = mediaPath.split('/')
                    shot = tmp[tmp.index('shot')+1]
                    task = tmp[tmp.index('shot')+2]

                    changedShot += '\n%s, %s' % (shot, task)

        if changedShot:
            self.MessagePopup('최신 snapshot으로 update 되었습니다.\n'+changedShot)
        else:
            self.MessagePopup('모두 최신 snapshot입니다.')


    def toggleDxSelector(self, event):
        # CHECK FOR MEDIA
        if not self.chkMedia():
            return

        isActive = MuSymbol('dxTask_selector.selectorIsActive')
        if not isActive():
            MuSymbol('dxTask_selector.toggleSelector')()
            self.doIt()
        elif isActive() and self.chkVersioning() == False:
            MuSymbol('dxTask_selector.toggleSelector')()
            MuSymbol('dxTask_selector.toggleSelector')()
            self.doIt()
        else:
            MuSymbol('dxTask_selector.toggleSelector')()


    def render(self, event):
        if (self._syncActivated):
            self._syncActivated = False
            code = """
            {
                rvtypes.State s = commands.data();
                sync.RemoteSync mode = s.sync;
                mode.addSendPattern(regex("#RVFileSource\\.versioning\\..*"));
                mode.addSendPattern(regex("#RVFileSource\\.group\\..*"));
            }"""
            runtime.eval(code, ["sync", "rvtypes", "commands"])


    def chkMedia(self):
        media = commands.sources()
        if not media:
            self.MessagePopup('열려있는 미디어가 없습니다.')
            return False
        else:
            currentFrame = commands.frame()
            node = commands.sourcesAtFrame(currentFrame)
            info = commands.sourceMediaInfo(node[0])

            # CHECK TACTIC SNAPSHOT
            if not 'show' in info['file']:
                self.MessagePopup('현재 미디어가 tactic snapshot이 아닙니다.')
                return False
            elif '/asset/' in info['file']:
                self.MessagePopup('현재 미디어가 shot mov가 아닙니다.')
                return False
            else:
                return True


    def chkVersioning(self):
        currentFrame = commands.frame()
        node = commands.sourcesAtFrame(currentFrame)
        prefix = node[0] + ".versioning.media"
        if commands.propertyExists(prefix):
            # print 'TRUE'
            return True
        else:
            # print 'FALSE'
            return False


    def getShotInfo(self):
        currentFrame = commands.frame()
        node = commands.sourcesAtFrame(currentFrame)
        info = commands.sourceMediaInfo(node[0])
        info['file'] = info['file'].replace('/Volumes', '')
        fileName = os.path.basename(info['file'])

        # pprint.pprint(info['file'])

        tmp = info['file'].split('/')
        show = tmp[tmp.index('shot')-1]
        shotName = tmp[tmp.index('shot')+1]
        task = ''

        # print tmp, show, shotName

        self.show = show
        self.taskList = dxTacticCommon.getTaskList(show, shotName)

        prefix = node[0] + ".versioning"
        if commands.propertyExists(prefix + '.currentIndex'):
            self.current = commands.getIntProperty(prefix + '.currentIndex')[0]
            self.last = commands.getIntProperty(prefix + '.lastIndex')[0]
        else:
            self.current = 0
            self.last = 0

        # CHECK TASK
        if commands.propertyExists(prefix + '.currentTask'):
            idx = commands.getIntProperty(prefix + '.currentTask')[0]
            task = self.taskList[idx]
        else:
            if 'edit' in fileName:
                task = 'edit'
            else:
                for i in self.taskList:
                    if i in fileName:
                        task = i

        print(node, self.show, shotName, self.taskList, task)
        return node, show, shotName, task


    def backwardIdx(self, list, idx):
        if idx-1 < 0:
            idx = len(list)-1
        else:
            idx = idx - 1
        return idx


    def forwardIdx(self, list, idx):
        if idx+1 > len(list)-1:
            idx = 0
        else:
            idx = idx + 1
        return idx


    def changeVersion(self, event):
        # CHECK FOR MEDIA
        if not self.chkMedia():
            return

        node, self.show, shotName, task = self.getShotInfo()

        media = node[0] + ".versioning.media"
        current = node[0] + ".versioning.currentIndex"
        last = node[0] + ".versioning.lastIndex"

        if commands.propertyExists(media):
            files = commands.getStringProperty(media)
        else:
            if task == 'edit':   #BREAKDOWN EDIT
                files, colors = dxTacticCommon.getBreakdown(self.show, shotName)
            else:   #SNAPSHOT
                files, colors = dxTacticCommon.getSnapshot(self.show, shotName, task)

            self.setVersiongData(node[0], files, colors, task)

        key = event.name().replace('key-down--', '')
        if KEY_VERSION_CHANGE_UP == key:
            nextIdx = self.backwardIdx(files, self.current)
        elif KEY_VERSION_CHANGE_DOWN == key:
            nextIdx = self.forwardIdx(files, self.current)

        self.last = self.current
        self.current = nextIdx
        commands.setSourceMedia(node[0], [files[self.current]])
        commands.setIntProperty(current, [self.current], True)
        commands.setIntProperty(last, [self.last], True)

        isActive = MuSymbol('dxTask_selector.selectorIsActive')
        if not (isActive()):
            MuSymbol('dxTask_selector.toggleSelector')()

        # CLEAR ALL ANNOTATION (DRAWING)
        dxTacticCommon.clearAnnotate(True)

        log.run('action.RV.dxVersioning.changeVersion', getpass.getuser())


    def changeTask(self, event):
        # CHECK FOR MEDIA
        if not self.chkMedia():
            return

        node, self.show, shotName, task = self.getShotInfo()
        currentidx = self.taskList.index(task)

        key = event.name().replace('key-down--', '')
        if KEY_TASK_CHANGE_BACKWARD in key:
            nextIdx = self.backwardIdx(self.taskList, currentidx)
        elif KEY_TASK_CHANGE_FORWARD in key:
            nextIdx = self.forwardIdx(self.taskList, currentidx)

        nextTask = self.taskList[nextIdx]
        print(task, currentidx, '->', nextIdx, nextTask)

        if nextTask == 'edit':   #BREAKDOWN EDIT
            files, colors = dxTacticCommon.getBreakdown(self.show, shotName)
        else:   #SNAPSHOT
            files, colors = dxTacticCommon.getSnapshot(self.show, shotName, nextTask)
        currentidx = nextIdx

        # CLEAR SELECT HISTORY
        self.current = 0
        self.last = 0
        current = node[0] + ".versioning.currentIndex"
        last = node[0] + ".versioning.lastIndex"
        if commands.propertyExists(current):
            commands.setIntProperty(current, [self.current], True)
            commands.setIntProperty(last, [self.last], True)

        # RELOAD SNAPSHOT LIST
        currentTask = node[0] + ".versioning.currentTask"
        self.setVersiongData(node[0], files, colors, nextTask)
        commands.setIntProperty(currentTask, [currentidx], True)
        commands.setSourceMedia(node[0], files)

        # CLEAR ALL ANNOTATION (DRAWING)
        dxTacticCommon.clearAnnotate(True)

        # SHOW TASK SELECTOR
        isActive = MuSymbol('dxTask_selector.selectorIsActive')
        if not (isActive()):
            MuSymbol('dxTask_selector.toggleSelector')()

        log.run('action.RV.dxVersioning.changeTask', getpass.getuser())


    def setVersiongData(self, node, files, colors, task):
        names = []
        for i in files:
            names.append(os.path.basename(i))

        setVersioning = api.VersionData()
        setVersioning._show = self.show
        setVersioning._media = files
        setVersioning._name = names
        if colors:
            setVersioning._color = colors
        setVersioning._task = self.taskList
        setVersioning._currentTask = self.taskList.index(task)
        if self.current > len(files) or self.last > len(files):
            self.current = 0
            self.last = 0
        info = commands.sourceMediaInfo(node)
        if info['file'] in files:
            self.current = files.index(info['file'])
            self.last = files.index(info['file'])

        setVersioning._current = self.current
        setVersioning._last = self.last
        setVersioning.setVersionDataOnSource(node)


    def doIt(self):
        node, self.show, shotName, task = self.getShotInfo()

        if task == 'edit':   #BREAKDOWN EDIT
            files, colors = dxTacticCommon.getBreakdown(self.show, shotName)
        else:   #SNAPSHOT
            files, colors = dxTacticCommon.getSnapshot(self.show, shotName, task)

        # pprint.pprint(files)

        # SET VERSIONDATA
        self.setVersiongData(node[0], files, colors, task)

        log.run('action.RV.dxVersioning.setVersionData', getpass.getuser())


    def SelectorHide(self, event):
        isActive = MuSymbol('dxTask_selector.selectorIsActive')
        if (isActive()):
            MuSymbol('dxTask_selector.toggleSelector')()


    def multiShotExtension(self, event):
        # CHECK FOR MEDIA
        if not self.chkMedia():
            return

        currentFrame = commands.frame()
        node = commands.sourcesAtFrame(currentFrame)
        print(node)
        info = commands.sourceMediaInfo(node[0])
        info['file'] = info['file'].replace('/Volumes', '')
        fileName = os.path.basename(info['file'])

        # pprint.pprint(commands.sourceAttributes(node[0]))

        tmp = info['file'].split('/')
        show = tmp[tmp.index('shot')-1]
        shotName = tmp[tmp.index('shot')+1]
        task = ''

        # print tmp, show, shotName

        parentNode = commands.nodeGroup(node[0])
        nodeGrps = commands.nodeConnections(parentNode)
        # print 'nodeGrps:', nodeGrps

        # GET SeqNode
        seqNode = ''
        for i in nodeGrps[1]:
            if 'Sequence' in i:
                seqNode = i
                # print 'seqNode:', i

        nodes = commands.nodes()
        mediaList = []
        for n in nodes:
            if 'RVFileSource' in commands.nodeType(n):
                path = commands.sourceMediaInfo(n)['file']
                tmp = path.split('/')
                media = tmp[tmp.index('shot')+1]
                mediaList.append(media)
        # print 'mediaList:', mediaList

        count = len(mediaList)
        files, sortedShot = dxTacticCommon.getMultiShot(show, shotName, count, task, getOrder=True)
        print('sortedShot:', count, sortedShot)

        start = mediaList[0]
        end = mediaList[-1]

        # SEARCH IN, OUT
        for media in mediaList:
            if sortedShot.index(media) < sortedShot.index(start):
                start = media
            elif sortedShot.index(media) > sortedShot.index(end):
                end = media

        addMediaPath = []
        if start != sortedShot[0]:
            addMediaPath.append(files[sortedShot.index(start)-1])
        if end != sortedShot[-1]:
            addMediaPath.append(files[sortedShot.index(end)+1])
        # print 'addMediaPath:', addMediaPath

        # ADD NEW MEDIA
        if addMediaPath:
            for addMedia in addMediaPath:
                commands.addSource(addMedia)

            # REORDER SOURCE NODE
            nodes = commands.nodes()
            sourceNodes = []
            for n in nodes:
                if 'RVFileSource' in commands.nodeType(n):
                    sourceNodes.append(n)

            orderedNode = []
            for shot in sortedShot:
                for node in sourceNodes:
                    info = commands.sourceMediaInfo(node)
                    if shot in info['file']:
                        orderedNode.append(node.replace('_source', ''))
            # print 'orderedNode:', orderedNode

            commands.setNodeInputs(seqNode, orderedNode)

            log.run('action.RV.dxVersioning.multiShotExtension', getpass.getuser())

    def openDexterGuide(self, event):
        chkplatform = dxTacticCommon.chkPlatform()
        if chkplatform != 'Darwin':
            webbrowser.open('http://%s/display/PIP/RV+User+Guide' % CONFLUENCE_IP, new=1, autoraise=True)
            log.run('action.RV.openDexterGuide', getpass.getuser())
        else:
            self.MessagePopup('해당 기능은 centOS에서만 지원합니다.')

    def MessagePopup(self, msg):
        QtWidgets.QMessageBox.information(self.widgets, 'dxVersioning info', msg,
                                        QtWidgets.QMessageBox.Ok)
    def MessagePopupOkCancel(self, msg):
        result = QtWidgets.QMessageBox.information(self.widgets, 'dxVersioning info',
                        msg, QtWidgets.QMessageBox.Ok |  QtWidgets.QMessageBox.Cancel)
        return result


def createMode():
    return VersioningMode()

