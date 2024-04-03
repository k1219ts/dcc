#encoding=utf-8

import os
import platform
from Qt_Ani.Qt import QtCore, QtGui, QtWidgets, QtCompat

HOSTNAME = platform.node().split("-")[0]
CURRENT_FILE = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_FILE)
#UIFILE = os.sep.join([CURRENT_DIR, 'resource', '{uifile}'])

REDSHIFT_SCRIP_PATH = '%s/apps/redshift2/scripts/tractorSpool' % os.getenv('BACKSTAGE_PATH')

def setup_ui(ui, base_instance=None):
    for member in dir(ui):
        if not member.startswith('__') and member is not 'staticMetObject':
            setattr(base_instance, member, getattr(ui, member))


class RedshiftWindow(QtWidgets.QWidget):
    uifile = '%s/resource/redshiftOptionUi.ui'%CURRENT_DIR

    def __init__(self, parent=None):
        super(RedshiftWindow, self).__init__(parent)
        ui = QtCompat.loadUi(self.uifile, self)
        setup_ui(ui, self)

        self.setWindowTitle('Redshift Render')
        self.move(700, 300)

        redshiftTractorIcon = QtGui.QPixmap(
            "{}/icons/redshift_tractorSubmit32.png".format(REDSHIFT_SCRIP_PATH)
        )
        self.redshift_icon_label.setPixmap(redshiftTractorIcon)
        cameraHeader = self.camera_tableWidget.horizontalHeader()
        cameraHeader.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.camera_tableWidget.setColumnWidth(2, 60)
        self.camera_tableWidget.setColumnWidth(3, 60)
        self.splitter.setStretchFactor(0, 10)
        self.splitter.setSizes([20, 180])
        sequencerHeader = self.sequencer_tableWidget.horizontalHeader()
        sequencerHeader.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.sequencer_tableWidget.setColumnWidth(0, 150)
        self.sequencer_tableWidget.setColumnWidth(2, 60)
        self.sequencer_tableWidget.setColumnWidth(3, 60)

        if HOSTNAME == "ani":
            self.cache_checkBox.setChecked(True)
            self.gpu_checkBox.setChecked(True)
            self.user_checkBox.setChecked(True)
        else:
            self.gpu_checkBox.setChecked(True)

        self.render_Btn.setStyleSheet('QPushButton {background-color: rgb(140, 40, 40)}')
        self.set_Btn.setStyleSheet('QPushButton {background-color: rgb(99, 33, 66)}')
        self.cancel_Btn.setStyleSheet('QPushButton {background-color: rgb(42, 42, 42)}')

        self.byFrame_lineEdit.setText('1')
