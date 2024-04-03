# -*- coding: utf-8 -*-
import os
import sys
import getpass

STATUS = 'RELEASE'

# 수정
debuggerList = ['daeseok.chae', 'rman.td', 'wonchul.kang', 'kwantae.kim', 'yeojin.lee']
dxRunnerPath = '/backstage/dcc/packages/ext/dxrunner/scripts'
if getpass.getuser() == 'kwantae.kim':
    dxRunnerPath = '/WORK_DATA/develop/dcc/packages/ext/dxrunner/scripts'
# 끝
if os.path.exists(dxRunnerPath):
    if dxRunnerPath not in sys.path:
        sys.path.append(dxRunnerPath)

from PyQt4 import QtCore
from PyQt4 import QtGui

class VDashboardAction(QtCore.QObject):
    def __init__(self, parent=None):
        QtCore.QObject.__init__(self, parent)

        self.parent = parent
        self.actions = []

        # 수정: 액션추가
        # self.dxRunnerAct = QtGui.QAction('dxRunnderMain', self)
        # self.dxRunnerAct.triggered.connect(self.dxRunnerActTriggered)
        # self.actions.append(self.dxRunnerAct)

        self.dxRunnerAct = QtGui.QAction('Start DXRunner', self)
        self.dxRunnerAct.triggered.connect(self.dxRunnerActTriggered)
        if STATUS == 'DEBUG' and getpass.getuser() in debuggerList:
            self.actions.append(self.dxRunnerAct)
        elif STATUS == 'RELEASE':
            self.actions.append(self.dxRunnerAct)

        # 끝

    def dxRunnerActTriggered(self):
        # 선택된 태스크 가지고 오기
        task = self.getTask()
        # 끝

        # # Sample
        # from main import Window
        # win = Window(task, self.parent)
        # win.show()
        """
        Table : TASK

        active_manday
        assigned
        assigned_info
        bd_type
        bd_type2
        bd_type3
        bd_type4
        bid_duration
        bid_end_date
        bid_manday
        bid_start_date
        code
        context
        description
        end_date
        extra_code
        extra_name
        extra_reel_name
        favor
        frame_in
        frame_out
        grade
        id
        keywords
        login
        m_description
        m_name
        milestone_code
        pipeline_code
        priority
        process
        project_code
        s_status
        search_id
        search_type
        start_date
        status
        supervisor
        supervisor_info
        task_info
        timestamp
        """
        from dxRunner import dxRunnerMain
        dxrunner = dxRunnerMain(self.parent, task)
        dxrunner.show()

    # def dxRunnerActTriggered2(self):
    #     # 선택된 태스크 가지고 오기
    #     task = self.getTask()
    #     # 끝
    #
    #     from dialog import Window
    #     win = Window(task, self.parent)
    #     win.show()

    def getActions(self):
        return self.actions

    def getTask(self):
        return self.parent.getTask()
