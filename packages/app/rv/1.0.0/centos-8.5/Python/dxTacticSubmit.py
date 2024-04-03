# -*- coding: utf-8 -*-
import sys, os, time
# import ConfigParser

from PySide2 import QtWidgets, QtCore

from rv import rvtypes, commands, qtutils

import dxConfig
from tactic_client_lib import TacticServerStub
from ui_tacticSubmit import Ui_Tacticsubmit
import dxTacticCommon
import getpass
import pprint


# define hotKey
# ---------------------------------
KEY_TACTIC_SUBMIT  = 'Z'
# ---------------------------------


API_KEY = 'c70181f2b648fdc2102714e8b5cb344d'
VELOZ_CONFIG_PATH = os.path.join(os.path.expanduser('~'),
                                 '.config/DEXTER_DIGITAL/Veloz.conf')


class dxTacticSubmit(rvtypes.MinorMode):
    def __init__(self):
        rvtypes.MinorMode.__init__(self)
        # self.init("dxTacticSubmit", None, None)
        self.init("dxTacticSubmit-mode",
        [
            ("source-media-set", self.setInfo, "media change"),
            ("frame-changed", self.setInfo, "frame change"),
            ], None)

        self.taskList = ['edit',
                         'matchmove', 'animation',
                         'lighting', 'fx', 'comp']
        self.currentShot = ''
        self.context = ''

        self.widgets = QtWidgets.QWidget()
        self.widgets.resize(802, 762)
        self.ui = Ui_Tacticsubmit()
        self.ui.setupUi(self.widgets)

        rvSessionQObject = qtutils.sessionWindow()
        self.dialog = QtWidgets.QDockWidget('TACTIC SUBMIT', rvSessionQObject)
        rvSessionQObject.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.dialog)
        self.dialog.setWidget(self.widgets)

        self.ui.submit_pushButton.clicked.connect(self.tacticSubmit)


    def chkMedia(self):
        media = commands.sources()
        if not media:
            self.ui.info_label.clear()
            self.ui.comment_textEdit.clear()
            self.MessagePopup('열려있는 미디어가 없습니다.')
            return False
        else:
            currentFrame = commands.frame()
            node = commands.sourcesAtFrame(currentFrame)
            info = commands.sourceMediaInfo(node[0])

            # CHECK TACTIC SNAPSHOT
            if not 'tactic' in info['file']:
                self.ui.info_label.clear()
                self.ui.comment_textEdit.clear()
                self.MessagePopup('현재 열려있는 미디어가 tactic snapshot이 아닙니다.')
                return False
            else:
                return True


    def test(self, event):
        print('asdfkjasdlfkjsad')


    def tacticSubmit(self):
        # CHECK FOR MEDIA
        if not self.chkMedia():
            return

        rvtypes.MinorMode.deactivate(self)
        self.dialog.hide()

        node, info = dxTacticCommon.getSourceMediaInfo()
        fileName = os.path.basename(info['file'])
        mediaFile = fileName.split('.')
        tmp = mediaFile[0].split('_')
        show = info['file'].split('/')[3]
        shotName = '_'.join(tmp[:2])
        task = ''

        # UPDATE TASK
        self.taskList = dxTacticCommon.getTaskList(show, shotName)

        # CHECK TASK
        prefix = node[0] + ".versioning"
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

        description = self.ui.comment_textEdit.toPlainText()
        if not 0 < len(description):
            self.MessagePopup('입력 된 코멘트가 없습니다.')
            return
        else:
            description = '(%s)\n' % fileName + description

        result = self.MessagePopupOkCancel('업로드 하시겠습니까?')
        if result == QtWidgets.QMessageBox.Ok:
            t = time.strftime('.%Y%m%d%H%M%S', time.localtime(time.time()))
            fileName = ''.join(mediaFile[:-1]) + t + '.jpg'
            tmpImagePath = os.path.join('/tmp', fileName)
            commands.exportCurrentFrame(tmpImagePath)

            try:
                setting = QtCore.QSettings('DEXTER_DIGITAL', 'Veloz')
                if setting.contains('login'):
                    login = setting.value('login')
                    password = setting.value('password')
                else:
                    self.MessagePopup('로그인 정보 오류입니다.\nVeloz에서 save login을 하셔야 합니다.')
                    return
            except:
                self.MessagePopup('로그인 정보 오류입니다.\nVeloz에서 save login을 하셔야 합니다.')
                return

            searchType = '%s/shot' % show
            context = 'publish/reference'
            # publish/edit, publish/previz, publish/onset
            # publish/source, publish/reference

            #tactic upload
            tactic = TacticServerStub(login=login, password=password,
                                      server=dxConfig.getConf('TACTIC_IP'), project=show)
            searchKey = tactic.build_search_key(searchType, shotName)

            tactic.start()
            try:
                if os.path.isfile(tmpImagePath):
                    tactic.simple_checkin(searchKey, context, tmpImagePath,
                                        description=description, mode='upload')
                    tactic.insert("sthpw/note",
                                  {'context': task,
                                   'note': description,
                                   'login': tactic.get_login()},
                                  parent_key=searchKey)
            except:
                tactic.abort()
                self.MessagePopup('업로드 에러!!')
            else:
                tactic.finish()
                self.MessagePopup('성공적으로 업로드 되었습니다!!')
                self.ui.comment_textEdit.clear()
                dxTacticCommon.clearAnnotate(False)
        else:
            rvtypes.MinorMode.activate(self)
            self.dialog.show()


    def shotInfoUpdate(self):
        if not self.chkMedia():
            return

        node, info = dxTacticCommon.getSourceMediaInfo()
        mediaFile = os.path.basename(info['file']).split('.')
        tmp = mediaFile[0].split('_')
        show = info['file'].split('/')[3]
        shotName = '_'.join(tmp[:2])
        context = tmp[-3:-1]
        if context[0] == context[1]:
            context = context[0]
        else:
            context = '/'.join(context)

        if 'edit' in context:
            self.ui.info_label.setText("")
            return
        elif (self.currentShot == '' or
              shotName != self.currentShot or
              context != self.context):
            self.currentShot = shotName
            self.context = context
            taskInfo = dxTacticCommon.getTaskShot(show, shotName, context)

            if taskInfo:
                infoText = 'artist : %s\n' % taskInfo['assigned']
                infoText += 'task : %s\n' % taskInfo['context']
                infoText += 'status : %s\n' % taskInfo['status']
                infoText += 'milestone : %s' % taskInfo['milestone_code']

            self.ui.info_label.setText(infoText)


    def setInfo(self, event):
        # print event.name()
        self.shotInfoUpdate()


    def activate(self):
        rvtypes.MinorMode.activate(self)
        self.dialog.show()
        print('activate!!')
        self.shotInfoUpdate()


    def deactivate(self):
        rvtypes.MinorMode.deactivate(self)
        self.dialog.hide()
        print('deactivate!!')


    def MessagePopup(self, msg):
        QtWidgets.QMessageBox.information(self.widgets, 'TACTIC SUBMIT info', msg,
                                        QtWidgets.QMessageBox.Ok)


    def MessagePopupOkCancel(self, msg):
        result = QtWidgets.QMessageBox.information(self.widgets, 'TACTIC SUBMIT info',
                        msg, QtWidgets.QMessageBox.Ok |  QtWidgets.QMessageBox.Cancel)
        return result


def createMode():
    return dxTacticSubmit()
