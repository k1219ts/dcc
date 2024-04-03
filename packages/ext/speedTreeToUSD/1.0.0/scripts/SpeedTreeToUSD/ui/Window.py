#coding:utf-8
from __future__ import print_function

import getpass
from PySide2 import QtWidgets, QtCore, QtGui

import DXRulebook.Interface as rb
import DXUSD.Utils as utl

from .speedTreeToUSDUI import Ui_stu_win
from SpeedTreeToUSD.uiFn import Clip
import SpeedTreeToUSD.Vars as var


class Win(QtWidgets.QWidget):
    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent)
        self.ui = Ui_stu_win()
        self.ui.setupUi(self)

        self.arg = rb.Flags(dcc='USD')


        self.InitializeUI()


    def InitializeUI(self):
        # open dir button image
        icon = QtGui.QIcon(QtGui.QPixmap(utl.SJoin(var.ICON, 'folder.png')))
        self.ui.stu_abcFile_openDir_pushButton.setIcon(icon)
        self.ui.stu_xmlFile_openDir_pushButton.setIcon(icon)

        # font to white style
        self.ui.stu_machineType_comboBox.setStyleSheet(var.STYLE.WHITE)
        self.ui.stu_roots_comboBox.setStyleSheet(var.STYLE.WHITE)
        self.ui.stu_show_comboBox.setStyleSheet(var.STYLE.WHITE)
        self.ui.stu_ver_lineEdit.setStyleSheet(var.STYLE.WHITE)
        self.ui.stu_texVer_lineEdit.setStyleSheet(var.STYLE.WHITE)
        self.ui.stu_asset_lineEdit.setStyleSheet(var.STYLE.WHITE)

        # set default values
        self.ui.stu_user_name_label.setText(getpass.getuser())

        self.ui.stu_clip_checkBox.setCheckState(QtCore.Qt.Unchecked)
        Clip.ActiveUI(self)


    def SetEnabled(self, ui, enable, uiEnabled=True):
        ui.setStyleSheet(var.STYLE.WHITE if enable else var.STYLE.GRAY)
        if uiEnabled:
            ui.setEnabled(enable)


    def Message(self, text='', type=var.MSG.RESULT):
        style = [var.STYLE.GREEN, var.STYLE.WARNING, var.STYLE.ERROR][type]
        text  = ['RESULT : %s', 'ERROR : %s', 'ERROR : %s'][type]%text

        self.ui.stu_message_label.setStyleSheet(style)
        self.ui.stu_message_label.setText(text)






#
