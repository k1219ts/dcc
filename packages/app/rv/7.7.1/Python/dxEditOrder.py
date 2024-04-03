# -*- coding: utf-8 -*-
import os
import getpass

from PySide2 import QtWidgets, QtCore, QtGui

from rv import rvtypes, commands, qtutils, extra_commands
from ui_dxEditOrder import Ui_Form
import dxTacticCommon as comm
from dxstats import inc_tool_by_user as log


API_KEY = 'c70181f2b648fdc2102714e8b5cb344d'


class dxEditOrder(rvtypes.MinorMode):
    def __init__(self):
        rvtypes.MinorMode.__init__(self)
        self.init('dxEditOrder', None, None)
        self.widgets = QtWidgets.QWidget()
        self.ui = Ui_Form()
        self.ui.setupUi(self.widgets)

        self.rvSessionQObject = qtutils.sessionWindow()
        self.dialog = QtWidgets.QDockWidget('dxEditOrder', self.rvSessionQObject)
        self.rvSessionQObject.addDockWidget(QtCore.Qt.TopDockWidgetArea, self.dialog)

        self.dialog.setWidget(self.widgets)

        tasks = ['ALL', 'matchmove', 'animation', 'lighting', 'fx', 'comp', 'edit']
        self.ui.task_comboBox.addItems(tasks)

        # Latest init
        self.setShowList()
        regExp = QtCore.QRegExp('[0-9]{5}')
        self.ui.editStart_lineEdit.setValidator(QtGui.QRegExpValidator(regExp))
        self.ui.editEnd_lineEdit.setValidator(QtGui.QRegExpValidator(regExp))
        self.ui.import_pushButton.clicked.connect(self.getLatest)


    def setShowList(self):
        showList = comm.getShowList()
        self.ui.show_comboBox.clear()
        self.ui.show_comboBox.addItem('select', None)
        for prj in sorted(showList.keys()):
            title = showList[prj]['title']
            enTitle = title.split(' ')[-1]
            korTitle = title.split(' ')[0]

            self.ui.show_comboBox.addItem(enTitle[1:-1] + ' ' + korTitle,
                                                        showList[prj])


    def getLatest(self):
        commands.clearSession()
        print 'clearSession'

        try:
            showIndex = self.ui.show_comboBox.currentIndex()
            show = self.ui.show_comboBox.itemData(showIndex)['code']
            task = self.ui.task_comboBox.currentText()
            start = str(self.ui.editStart_lineEdit.text())
            end = str(self.ui.editEnd_lineEdit.text())
            editOrder = '%s-%s' % (start, end)
        except:
            self.MessagePopup(u'검색 옵션을 다시 체크 해 주세요.')
            return

        if start and end:
            files = comm.getMultiShotEditOrder(showCode=show, editOrder=editOrder,
                                               process=task)
            commands.addSources(files)

            self.ui.import_pushButton.clearFocus()
            self.dialog.hide()
        else:
            self.MessagePopup(u'에딧오더 범위를 다시 체크 해 주세요.')
        log.run('action.RV.dxEditOrder.importLatest', getpass.getuser())


    def MessagePopup(self, msg):
        QtWidgets.QMessageBox.information(self.widgets, 'dxEditOrder info', msg,
                                        QtWidgets.QMessageBox.Ok)


    def activate(self):
        self.dialog.show()
        print 'activate!!'

    def deactivate(self):
        print 'deactivate!!'


def createMode():
    return dxEditOrder()
