# -*- coding: utf-8 -*-
import os
import re
import getpass
import pprint

from PySide2 import QtWidgets, QtCore, QtGui

from rv import rvtypes, commands, qtutils, extra_commands

from ui_dxTacticReview import Ui_Form
import dxTacticCommon as dxCommon
import dxTacticWidget as tWidget
from dxstats import inc_tool_by_user as log


API_KEY = 'c70181f2b648fdc2102714e8b5cb344d'


class dxTacticReview(rvtypes.MinorMode):
    def __init__(self):
        rvtypes.MinorMode.__init__(self)
        self.init('dxTacticReview', None, None)
        self.widgets = QtWidgets.QWidget()
        self.ui = Ui_Form()
        self.ui.setupUi(self.widgets)

        self.rvSessionQObject = qtutils.sessionWindow()
        self.dialog = QtWidgets.QDockWidget('dxTacticReview', self.rvSessionQObject)
        self.rvSessionQObject.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.dialog)
        self.dialog.setWidget(self.widgets)

        width = self.rvSessionQObject.width()
        self.rvSessionQObject.resize(width+620, self.rvSessionQObject.height())

        # init
        self.currentShowIndex = 0
        self.ShowIndex = [self.ui.comboBox_snapShow,
                          self.ui.comboBox_reSupShow,
                          self.ui.comboBox_mileShow,
                          self.ui.comboBox_feedShow]
        self.configData = dxCommon.ReviewConfig()
        self.setShowList()
        tWidget.setCustomWidget(self.ui)

        # connect
        self.ui.tabWidget_main.currentChanged.connect(self.tabChanged)
        self.ui.comboBox_snapShow.currentIndexChanged.connect(self.setSeqList)
        self.ui.lineEdit_snapShot.returnPressed.connect(self.serchShot)
        self.ui.pushButton_snapSearch.clicked.connect(self.getSourceSnapthot)

        self.ui.comboBox_reSupShow.currentIndexChanged.connect(self.setReSupList)
        self.ui.comboBox_reviewSup.currentIndexChanged.connect(self.serchReview)
        self.ui.pushButton_resupSearch.clicked.connect(self.reSupRefresh)

        self.ui.comboBox_mileShow.currentIndexChanged.connect(self.setMilestoneList)
        self.ui.comboBox_milestone.currentIndexChanged.connect(self.serchMilestone)
        self.ui.checkBox_mileAll.stateChanged.connect(self.mileAll)
        self.ui.pushButton_mileSearch.clicked.connect(self.mileRefresh)

        self.ui.comboBox_feedShow.currentIndexChanged.connect(self.setTopicList)
        self.ui.comboBox_feedback.currentIndexChanged.connect(self.getFeedbackTopic)
        self.ui.pushButton_feedSearch.clicked.connect(self.feedRefresh)

    def tabChanged(self, index):
        combo = self.ShowIndex[index]
        combo.setCurrentIndex(self.currentShowIndex)

        if self.ui.tabWidget_main.currentIndex() == 0:
            self.getSourceSnapthot()

    def setShowList(self):
        showList = dxCommon.getShowList()
        for comboBox in self.ShowIndex:
            comboBox.clear()
            comboBox.addItem('select', None)
            for prj in sorted(showList.keys()):
                title = showList[prj]['title']
                enTitle = title.split(' ')[-1]
                korTitle = title.split(' ')[0]
                comboBox.addItem(enTitle[1:-1] + ' ' + korTitle, showList[prj])


    # snapShot tab -------------------------------------------------------------
    def setSeqList(self, index):
        self.currentShowIndex = index

        self.ui.comboBox_snapSeq.clear()
        index = self.ui.comboBox_snapShow.currentIndex()
        prjData = self.ui.comboBox_snapShow.itemData(index)
        infos = dxCommon.getSeqList(prjData['code'])
        self.ui.comboBox_snapSeq.addItems(sorted(infos))

    def getSourceSnapthot(self):
        sources = commands.sourcesAtFrame(commands.frame())
        filepath = None
        if sources:
            info = commands.sourceMediaInfo(sources[0], None)
            if '/shot/' in info['file']:
                filepath = info['file'].split('/')
                showCode = filepath[filepath.index('assets')+1]
                showName = self.configData.revPrjDic[showCode]['code']
                shot = filepath[filepath.index('shot')+1]
                seq = shot.split('_')[0]
                shotNumber = shot.split('_')[-1]

        if self.ui.tabWidget_main.currentIndex() == 0 and filepath:
            self.ui.snapTree.clear()

            for i in range(self.ui.comboBox_snapShow.count()):
                data = self.ui.comboBox_snapShow.itemData(i)
                if not(data):
                    continue
                if data['code'] == showCode:
                    self.ui.comboBox_snapShow.setCurrentIndex(i)
                    break

            for i in range(self.ui.comboBox_snapSeq.count()):
                if self.ui.comboBox_snapSeq.itemText(i) == seq:
                    self.ui.comboBox_snapSeq.setCurrentIndex(i)
                    break

            self.ui.lineEdit_snapShot.setText(shot.split('_')[-1])
            self.serchShot()

            for i in range(self.ui.snapTree.columnCount()):
                self.ui.snapTree.resizeColumnToContents(i)

    def serchShot(self):
        self.ui.snapTree.clear()
        self.ui.brkdownTree.clear()
        prjData = self.ui.comboBox_snapShow.itemData(self.ui.comboBox_snapShow.currentIndex())

        showCode = prjData['code']
        seq = unicode(self.ui.comboBox_snapSeq.currentText())
        shot = seq + '_' + unicode(self.ui.lineEdit_snapShot.text())

        if seq and (self.ui.lineEdit_snapShot.text()):
            # get snapShot
            for i in dxCommon.getSnapshot(showCode, shot):
                item = tWidget.MovItem(self.ui.snapTree)
                item.tacticData = i

                version_reO = re.search('v[0-9]+', os.path.basename(i['path']))
                if version_reO:
                    version = 'v' + str(int(version_reO.group()[1:]))
                else:
                    version = '-'

                item.setText(0, i['task_name'])
                item.setText(1, version)
                item.setText(2, i['process'])
                item.setForeground(2, self.configData.teamColor[i['process']])
                if '/' in i['context']:
                    item.setText(3, i['context'].split('/')[-1])
                else:
                    item.setText(3, i['context'])
                item.setText(4, i['task_status'])
                item.setForeground(4, QtGui.QBrush(self.configData.colorScheme[i['task_status']]))
                if i['base_type'] == 'file':
                    item.setText(5, os.path.splitext(i['path'])[1][1:])
                else:
                    item.setText(5, i['base_type'])
                item.setText(6, i['login'])
                item.setText(7, i['timestamp'].split('.')[0])

            # get breakdown
            for i in dxCommon.getBreakdown(showCode, shot, True):
                item = tWidget.MovItem(self.ui.brkdownTree)
                item.tacticData = i

                version_reO = re.search('v[0-9]+', os.path.basename(i['path']))
                if version_reO:
                    version = 'v' + str(int(version_reO.group()[1:]))
                else:
                    version = '-'

                item.setText(0, shot)
                item.setText(1, version)
                item.setText(2, i['process'])
                item.setForeground(2, self.configData.teamColor[i['process']])
                if '/' in i['context']:
                    item.setText(3, i['context'].split('/')[-1])
                else:
                    item.setText(3, i['context'])
                if i['base_type'] == 'file':
                    item.setText(4, os.path.splitext(i['path'])[1][1:])
                else:
                    item.setText(4, i['base_type'])
                item.setText(5, i['login'])
                item.setText(6, i['timestamp'].split('.')[0])

            self.ui.snapTree.sortItems(8, QtCore.Qt.DescendingOrder)
            self.ui.brkdownTree.sortItems(8, QtCore.Qt.DescendingOrder)

            for i in range(self.ui.snapTree.columnCount()):
                self.ui.snapTree.resizeColumnToContents(i)

            for i in range(self.ui.brkdownTree.columnCount()):
                self.ui.brkdownTree.resizeColumnToContents(i)


    # review supervisor tab ----------------------------------------------------
    def setReSupList(self, index):
        self.currentShowIndex = index

        self.ui.comboBox_reviewSup.clear()
        self.ui.comboBox_reviewSup.addItem('None')

        prjData = self.ui.comboBox_reSupShow.itemData(index)
        show = prjData['code']

        today = QtCore.QDate.currentDate()
        startDate = today.addDays(-3)
        endDate = today.addDays(2)

        for i in dxCommon.getReviewSupervisor(show, startDate, endDate):
            reviewName = i['pubdate'].split(' ')[0] + ' ' + i['title']
            self.ui.comboBox_reviewSup.addItem(reviewName, i)

    def serchReview(self):
        self.ui.reSupTree.clear()
        prjData = self.ui.comboBox_reSupShow.itemData(self.ui.comboBox_reSupShow.currentIndex())
        showcode = prjData['code']
        reviewCode = None

        try:
            reviewCode = self.ui.comboBox_reviewSup.itemData(self.ui.comboBox_reviewSup.currentIndex())['code']
        except:
            pass

        if reviewCode:
            infos = dxCommon.getReviewSnapshot(showcode, reviewCode)

            if not infos:
                QtWidgets.QMessageBox.information(self.widgets, "No Result", "No Result")
                return

            for i in infos:
                item = tWidget.ReviewItem(self.ui.reSupTree)
                item.tacticData = i

                version_reO = re.search('v[0-9]+', os.path.basename(i['tactic_path']))
                if version_reO:
                    version = 'v' + str(int(version_reO.group()[1:]))
                else:
                    version = '-'

                item.setText(0, i['relative_dir'].split('/')[2])
                item.setText(1, version)
                item.setText(2, i['process'])
                item.setForeground(2, self.configData.teamColor[i['process']])
                if '/' in i['context']:
                    item.setText(3, i['context'].split('/')[-1])
                else:
                    item.setText(3, i['context'])
                item.setText(4, i['task_status'])
                item.setForeground(4, QtGui.QBrush(self.configData.colorScheme[i['task_status']]))
                item.setText(5, i['login'])
                item.setText(6, i['timestamp'].split('.')[0])

            self.ui.reSupTree.sortItems(0, QtCore.Qt.AscendingOrder)

            for i in range(self.ui.reSupTree.columnCount()):
                self.ui.reSupTree.resizeColumnToContents(i)

    def reSupRefresh(self):
        reviewIndex = self.ui.comboBox_reviewSup.currentIndex()
        self.setReSupList(self.ui.comboBox_reSupShow.currentIndex())
        self.ui.comboBox_reviewSup.setCurrentIndex(reviewIndex)
        self.serchReview()


    # milestone tab ------------------------------------------------------------
    def setMilestoneList(self, index):
        self.currentShowIndex = index
        self.ui.comboBox_milestone.clear()
        self.ui.comboBox_milestone.addItem('None')

        prjData = self.ui.comboBox_mileShow.itemData(index)
        project = prjData['code']

        today = QtCore.QDate.currentDate()
        offset = today.dayOfWeek() - 1
        startDate = today.addDays(-offset)
        endDate = today.addDays(offset + 7)

        infos = dxCommon.getMilestoneName(project)
        for i in infos:
            itemDate = QtCore.QDateTime.fromString(i['due_date'], 'yyyy-MM-dd hh:mm:ss').date()

            if not self.ui.checkBox_mileAll.isChecked():
                if (startDate <= itemDate) and (endDate >= itemDate):
                    self.ui.comboBox_milestone.addItem(i['name'], i)
            else:
                self.ui.comboBox_milestone.addItem(i['name'], i)

    def serchMilestone(self):
        self.ui.mileTree.clear()
        prjData = self.ui.comboBox_mileShow.itemData(self.ui.comboBox_mileShow.currentIndex())
        showCode = prjData['code']
        mileCode = None

        try:
            mileCode = self.ui.comboBox_milestone.itemData(self.ui.comboBox_milestone.currentIndex())['code']
        except:
            pass

        if mileCode:
            infos = dxCommon.getMilestoneSnapshotAll(showCode, mileCode)

            if not infos:
                QtWidgets.QMessageBox.information(self.widgets, "No Result", "No Result")
                return

            for i in infos:
                item = tWidget.ReviewItem(self.ui.mileTree)
                item.tacticData = i

                version_reO = re.search('v[0-9]+', os.path.basename(i['tactic_path']))
                if version_reO:
                    version = 'v' + str(int(version_reO.group()[1:]))
                else:
                    version = '-'

                item.setText(0, i['task_extra_code'])
                item.setText(1, version)
                item.setText(2, i['task_process'])
                item.setForeground(2, self.configData.teamColor[i['task_process']])
                if '/' in i['task_context']:
                    item.setText(3, i['task_context'].split('/')[-1])
                else:
                    item.setText(3, i['task_context'])
                item.setText(4, i['task_status'])
                item.setForeground(4, QtGui.QBrush(self.configData.colorScheme[i['task_status']]))
                item.setText(5, i['task_assigned'])
                item.setText(6, i['timestamp'].split('.')[0])

            self.ui.mileTree.sortItems(0, QtCore.Qt.AscendingOrder)

            for i in range(self.ui.mileTree.columnCount()):
                self.ui.mileTree.resizeColumnToContents(i)

    def mileAll(self):
        self.setMilestoneList(self.ui.comboBox_mileShow.currentIndex())
        self.serchMilestone()

    def mileRefresh(self):
        mileIndex = self.ui.comboBox_milestone.currentIndex()
        self.setMilestoneList(self.ui.comboBox_mileShow.currentIndex())
        self.ui.comboBox_milestone.setCurrentIndex(mileIndex)
        self.serchMilestone()


    # Feedback Topic tab -------------------------------------------------------
    def setTopicList(self, index):
        self.currentShowIndex = index
        self.ui.comboBox_feedback.clear()
        self.ui.comboBox_feedback.addItem('None', None)

        if unicode(self.ui.comboBox_feedShow.currentText()) == 'None':
            pass
        else:
            prjData = self.ui.comboBox_feedShow.itemData(index)
            project = prjData['code']
            infos = dxCommon.getFeedbackTopicName(project)
            for i in infos:
                self.ui.comboBox_feedback.addItem(i['title'], i)

    def getFeedbackTopic(self):
        self.ui.feedTree.clear()

        prjData = self.ui.comboBox_feedShow.itemData(self.ui.comboBox_feedShow.currentIndex())
        showCode = prjData['code']
        feedCode = None

        try:
            feedCode = self.ui.comboBox_feedback.itemData(self.ui.comboBox_feedback.currentIndex())['code']
        except:
            pass

        if feedCode:
            infos = dxCommon.getFeedbackTopic(showCode, feedCode)

            if not infos:
                QtWidgets.QMessageBox.information(self.widgets, "No Result", "No Result")
                return

            for i in infos:
                item = tWidget.ReviewItem(self.ui.feedTree)
                item.tacticData = i

                version_reO = re.search('v[0-9]+', os.path.basename(i['tactic_path']))
                if version_reO:
                    version = 'v' + str(int(version_reO.group()[1:]))
                else:
                    version = '-'
                    brkInfo = dxCommon.getBreakdown(showCode, i['code'], True)
                    if brkInfo:
                        # i['author'] = 'edit'
                        for brk in brkInfo:
                            if '.mov' in brk['path']:
                                item.tacticData['tactic_path'] = brk['path']
                                break

                item.setText(0, i['code'])
                item.setText(1, version)
                item.setText(2, i['process'])
                item.setForeground(2, self.configData.teamColor[i['process']])
                if '/' in i['context']:
                    item.setText(3, i['context'].split('/')[-1])
                else:
                    item.setText(3, i['context'])
                item.setText(4, i['status'])
                item.setForeground(4, QtGui.QBrush(self.configData.colorScheme[i['status']]))

                item.setText(5, i['author'])
                item.setText(6, i['pubDate'].split('.')[0])

            self.ui.feedTree.sortItems(0, QtCore.Qt.AscendingOrder)

            for i in range(self.ui.feedTree.columnCount()):
                self.ui.feedTree.resizeColumnToContents(i)

    def feedRefresh(self):
        feedIndex = self.ui.comboBox_feedback.currentIndex()
        self.setTopicList(self.ui.comboBox_feedShow.currentIndex())
        self.ui.comboBox_feedback.setCurrentIndex(feedIndex)
        self.getFeedbackTopic()


    def MessagePopup(self, msg):
        QtWidgets.QMessageBox.information(self.widgets, 'dxTacticReview', msg,
                                        QtWidgets.QMessageBox.Ok)

    def activate(self):
        # rvtypes.MinorMode.activate(self)
        self.getSourceSnapthot()
        self.dialog.show()
        print 'activate!!'

    def deactivate(self):
        # rvtypes.MinorMode.deactivate(self)
        # self.dialog.hide()
        print 'deactivate!!'
        # return commands.UncheckedMenuState


def createMode():
    return dxTacticReview()
