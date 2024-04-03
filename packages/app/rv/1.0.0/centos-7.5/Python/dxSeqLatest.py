# -*- coding: utf-8 -*-
import os
import getpass

from PySide2 import QtWidgets, QtCore

from rv import rvtypes, commands, qtutils, extra_commands
from ui_dxSeqLatest import Ui_Form
import dxTacticCommon
from dxstats import inc_tool_by_user as log


API_KEY = 'c70181f2b648fdc2102714e8b5cb344d'


class dxSeqLatest(rvtypes.MinorMode):
    def __init__(self):
        rvtypes.MinorMode.__init__(self)
        self.init('dxSeqLatest', None, None)
        self.widgets = QtWidgets.QWidget()
        self.ui = Ui_Form()
        self.ui.setupUi(self.widgets)

        self.rvSessionQObject = qtutils.sessionWindow()
        self.dialog = QtWidgets.QDockWidget('dxSeqLatest', self.rvSessionQObject)
        self.rvSessionQObject.addDockWidget(QtCore.Qt.TopDockWidgetArea, self.dialog)

        self.dialog.setWidget(self.widgets)

        tasks = ['ALL', 'matchmove', 'animation', 'lighting', 'fx', 'comp', 'model', 'edit']
        self.ui.task_comboBox.addItems(tasks)

        # Latest init
        self.setShowList()
        self.setListWidgetSize()
        self.ui.show_comboBox.currentIndexChanged.connect(self.setSeqList)
        self.ui.seq_comboBox.currentIndexChanged.connect(self.setSelectedSeq)
        self.ui.seq_listWidget.itemClicked.connect(self.deleteSeqItem)
        self.ui.import_pushButton.clicked.connect(self.getLatest)


    def setShowList(self):
        showList = dxTacticCommon.getShowList()
        self.ui.show_comboBox.clear()
        self.ui.show_comboBox.addItem('select', None)
        for prj in sorted(showList.keys()):
            title = showList[prj]['title']
            enTitle = title.split(' ')[-1]
            korTitle = title.split(' ')[0]

            self.ui.show_comboBox.addItem(enTitle[1:-1] + ' ' + korTitle,
                                                        showList[prj])


    def setSeqList(self):
        self.ui.seq_comboBox.clear()
        self.ui.seq_listWidget.clear()
        index = self.ui.show_comboBox.currentIndex()
        prjData = self.ui.show_comboBox.itemData(index)
        infos = dxTacticCommon.getSeqList(prjData['code'])
        self.ui.seq_comboBox.blockSignals(True)
        self.ui.seq_comboBox.addItem('select')
        self.ui.seq_comboBox.addItems(sorted(infos))
        self.ui.seq_comboBox.blockSignals(False)

    def setSelectedSeq(self, index):
        if index:
            seq = self.ui.seq_comboBox.itemText(index)

            for i in range(self.ui.seq_listWidget.count()):
                if seq in self.ui.seq_listWidget.item(i).text():
                    return

            seqItem = QtWidgets.QListWidgetItem()
            seqItem.setText(seq)
            seqItem.setTextAlignment(QtCore.Qt.AlignHCenter)
            self.ui.seq_listWidget.addItem(seqItem)
            self.setListWidgetSize()

    def deleteSeqItem(self, item):
        rowIdx = self.ui.seq_listWidget.row(item)
        self.ui.seq_listWidget.takeItem(rowIdx)
        self.setListWidgetSize()

    def setListWidgetSize(self):
        width = self.ui.seq_listWidget.width()
        tmp = self.ui.seq_listWidget.sizeHintForColumn(0) * self.ui.seq_listWidget.count()
        if tmp >= self.ui.seq_listWidget.width()-10:
            width = self.ui.seq_listWidget.width()+ (self.ui.seq_listWidget.sizeHintForColumn(0))
        height = 25
        self.ui.seq_listWidget.setFixedSize(width, height)

    def getLatest(self):
        commands.clearSession()

        try:
            showIndex = self.ui.show_comboBox.currentIndex()
            show = self.ui.show_comboBox.itemData(showIndex)['code']
        except:
            self.MessagePopup('검색 옵션을 다시 체크 해 주세요.')
            return

        task = self.ui.task_comboBox.currentText()

        files = []
        if self.ui.seq_listWidget.count():
            seqList = []
            for i in range(self.ui.seq_listWidget.count()):
                seqList.append(self.ui.seq_listWidget.item(i).text())
            seq = '|'.join(seqList)
            files = dxTacticCommon.getMultiShot(showCode=show, process=task, seqCode=seq)
            commands.addSources(files)

            self.ui.import_pushButton.clearFocus()
            self.dialog.hide()
        else:
            self.MessagePopup('선택된 시퀀스가 없습니다.')
        log.run('action.RV.dxSeqLatest.importLatest', getpass.getuser())


    def MessagePopup(self, msg):
        QtWidgets.QMessageBox.information(self.widgets, 'dxSeqLatest info', msg,
                                        QtWidgets.QMessageBox.Ok)


    def activate(self):
        # rvtypes.MinorMode.activate(self)
        self.dialog.show()
        print('activate!!')

    def deactivate(self):
        # rvtypes.MinorMode.deactivate(self)
        # self.dialog.hide()
        print('deactivate!!')
        # return commands.UncheckedMenuState


def createMode():
    return dxSeqLatest()
