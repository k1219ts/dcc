# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'setup.ui'
#
# Created: Fri Mar  8 15:34:16 2019
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtWidgets, QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtWidgets.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtWidgets.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtWidgets.QApplication.translate(context, text, disambig)

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName(_fromUtf8("Form"))
        # Form.resize(1550, 698)
        Form.resize(1650, 700)
        #######################
        bigFont = QtGui.QFont()
        bigFont.setPointSize(10)
        # bigFont.setBold(True)
        #######################
        Form.setFont(bigFont)

        self.gridLayout_2 = QtWidgets.QGridLayout(Form)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.setup_GRP = QtWidgets.QGroupBox(Form)
        self.setup_GRP.setObjectName(_fromUtf8("setup_GRP"))
        self.setup_gridLayout = QtWidgets.QGridLayout(self.setup_GRP)
        self.setup_gridLayout.setObjectName(_fromUtf8("setup_gridLayout"))
        self.seq_listWidget = QtWidgets.QListWidget(self.setup_GRP)
        self.seq_listWidget.setObjectName(_fromUtf8("seq_listWidget"))
        self.setup_gridLayout.addWidget(self.seq_listWidget, 4, 0, 1, 1)
        self.shot_listWidget = QtWidgets.QListWidget(self.setup_GRP)
        self.shot_listWidget.setObjectName(_fromUtf8("shot_listWidget"))
        self.setup_gridLayout.addWidget(self.shot_listWidget, 4, 1, 1, 1)

        self.ctx_lineEdit = QtWidgets.QLineEdit(self.setup_GRP)
        self.ctx_lineEdit.setEnabled(False)
        self.ctx_lineEdit.setObjectName(_fromUtf8("ctx_lineEdit"))
        self.setup_gridLayout.addWidget(self.ctx_lineEdit, 3, 0, 1, 2)

        self.ctx_comboBox = QtWidgets.QComboBox(self.setup_GRP)
        self.ctx_comboBox.setObjectName(_fromUtf8("ctx_comboBox"))
        self.setup_gridLayout.addWidget(self.ctx_comboBox, 2, 0, 1, 2)
        self.prj_comboBox = QtWidgets.QComboBox(self.setup_GRP)
        self.prj_comboBox.setObjectName(_fromUtf8("prj_comboBox"))
        self.setup_gridLayout.addWidget(self.prj_comboBox, 1, 0, 1, 2)
        self.setup_gridLayout.setColumnStretch(0, 1)
        self.setup_gridLayout.setColumnStretch(1, 1)
        self.gridLayout_2.addWidget(self.setup_GRP, 1, 0, 2, 1)

        self.plate_GRP = QtWidgets.QGroupBox(Form)
        self.plate_GRP.setObjectName(_fromUtf8("plate_GRP"))
        self.plate_verticalLayout = QtWidgets.QVBoxLayout(self.plate_GRP)
        self.plate_verticalLayout.setObjectName(_fromUtf8("plate_verticalLayout"))
        self.plate_treeWidget = QtWidgets.QTreeWidget(self.plate_GRP)
        self.plate_treeWidget.setObjectName(_fromUtf8("plate_treeWidget"))
        self.plate_treeWidget.headerItem().setText(0, _fromUtf8("1"))
        self.plate_verticalLayout.addWidget(self.plate_treeWidget)
        self.gridLayout_2.addWidget(self.plate_GRP, 2, 1, 1, 2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.cancel_button = QtWidgets.QPushButton(Form)
        self.cancel_button.setObjectName(_fromUtf8("cancel_button"))
        self.horizontalLayout_3.addWidget(self.cancel_button)
        self.ok_button = QtWidgets.QPushButton(Form)
        self.ok_button.setObjectName(_fromUtf8("ok_button"))
        self.horizontalLayout_3.addWidget(self.ok_button)
        self.gridLayout_2.addLayout(self.horizontalLayout_3, 3, 2, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.filter_job_checkBox = QtWidgets.QCheckBox(Form)
        self.filter_job_checkBox.setObjectName(_fromUtf8("filter_job_checkBox"))
        self.horizontalLayout.addWidget(self.filter_job_checkBox)
        self.filter_exist_checkBox = QtWidgets.QCheckBox(Form)
        self.filter_exist_checkBox.setObjectName(_fromUtf8("filter_exist_checkBox"))
        self.horizontalLayout.addWidget(self.filter_exist_checkBox)
        self.gridLayout_2.addLayout(self.horizontalLayout, 0, 2, 1, 1)
        self.mmv_GRP = QtWidgets.QGroupBox(Form)
        self.mmv_GRP.setTitle(_fromUtf8("Matchmove"))
        self.mmv_GRP.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.mmv_GRP.setObjectName(_fromUtf8("mmv_GRP"))
        self.mmv_verticalLayout = QtWidgets.QVBoxLayout(self.mmv_GRP)
        self.mmv_verticalLayout.setObjectName(_fromUtf8("mmv_verticalLayout"))
        self.cam_treeWidget = QtWidgets.QTreeWidget(self.mmv_GRP)
        self.cam_treeWidget.setHeaderHidden(False)
        self.cam_treeWidget.setObjectName(_fromUtf8("cam_treeWidget"))
        self.cam_treeWidget.headerItem().setText(0, _fromUtf8("1"))
        self.mmv_verticalLayout.addWidget(self.cam_treeWidget)
        self.dist_treeWidget = QtWidgets.QTreeWidget(self.mmv_GRP)
        self.dist_treeWidget.setObjectName(_fromUtf8("dist_treeWidget"))
        self.dist_treeWidget.headerItem().setText(0, _fromUtf8("1"))
        self.mmv_verticalLayout.addWidget(self.dist_treeWidget)
        self.gridLayout_2.addWidget(self.mmv_GRP, 1, 1, 1, 2)
        # self.gridLayout_2.setColumnStretch(0, 1)
        self.gridLayout_2.setColumnStretch(0, 2)
        self.gridLayout_2.setColumnStretch(1, 2)
        self.gridLayout_2.setColumnStretch(2, 1)
        self.gridLayout_2.setRowStretch(0, 1)
        self.gridLayout_2.setRowStretch(1, 8)
        self.gridLayout_2.setRowStretch(2, 8)
        self.gridLayout_2.setRowStretch(3, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

        self.cam_treeWidget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.dist_treeWidget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)

        self.cam_treeWidget.setSelectionMode(QtWidgets.QAbstractItemView.ContiguousSelection)
        self.dist_treeWidget.setSelectionMode(QtWidgets.QAbstractItemView.ContiguousSelection)
        self.plate_treeWidget.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)

        #############################################
        # set CamTreeWidget
        self.cam_treeWidget.setColumnCount(5)
        self.cam_treeWidget.headerItem().setText(0, 'Camera File')
        self.cam_treeWidget.headerItem().setText(1, 'Type')
        self.cam_treeWidget.headerItem().setText(2, 'Team')
        self.cam_treeWidget.headerItem().setText(3, 'Version')
        self.cam_treeWidget.headerItem().setText(4, 'Over Scan')
        self.cam_treeWidget.headerItem().setText(5, 'Stereo')
        self.cam_treeWidget.headerItem().setText(6, 'Time')

        # self.cam_treeWidget.header().setMovable(False)
        self.cam_treeWidget.header().resizeSection(0, 300)
        self.cam_treeWidget.header().resizeSection(1, 80)
        self.cam_treeWidget.header().resizeSection(2, 80)
        self.cam_treeWidget.header().resizeSection(3, 80)
        self.cam_treeWidget.header().resizeSection(4, 80)
        self.cam_treeWidget.header().resizeSection(5, 80)
        self.cam_treeWidget.header().resizeSection(6, 100)

        self.cam_treeWidget.setSortingEnabled(True)
        self.cam_treeWidget.sortItems(0, QtCore.Qt.AscendingOrder)

        ########################################################
        self.dist_treeWidget.setColumnCount(4)
        self.dist_treeWidget.headerItem().setText(0, 'Distortion File')
        self.dist_treeWidget.headerItem().setText(1, 'Version')
        self.dist_treeWidget.headerItem().setText(2, 'Plate')
        self.dist_treeWidget.headerItem().setText(3, 'Time')

        # self.dist_treeWidget.header().setMovable(False)
        self.dist_treeWidget.header().resizeSection(0, 300)
        self.dist_treeWidget.header().resizeSection(1, 80)
        self.dist_treeWidget.header().resizeSection(2, 80)
        self.dist_treeWidget.header().resizeSection(3, 100)

        self.dist_treeWidget.setSortingEnabled(True)
        self.dist_treeWidget.sortItems(0, QtCore.Qt.AscendingOrder)

        #####################################################
        self.plate_treeWidget.setColumnCount(3)

        self.plateCheck = QtWidgets.QCheckBox(self.plate_treeWidget.header())
        self.plateCheck.setText('Plate')
        self.plateCheck.setChecked(True)

        self.plate_treeWidget.headerItem().setText(1, 'Version')
        self.plate_treeWidget.headerItem().setText(2, 'Time')
        # self.plate_treeWidget.header().setMovable(False)
        self.plate_treeWidget.header().resizeSection(0, 200)
        self.plate_treeWidget.header().resizeSection(1, 160)
        self.plate_treeWidget.header().resizeSection(2, 100)
        self.plate_treeWidget.setSortingEnabled(True)
        self.plate_treeWidget.sortItems(0, QtCore.Qt.AscendingOrder)

        #######################################################

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Form", None))
        self.setup_GRP.setTitle(_translate("Form", "Shot Setup", None))
        self.plate_GRP.setTitle(_translate("Form", "Plate", None))
        self.cancel_button.setText(_translate("Form", "Cancel", None))
        self.ok_button.setText(_translate("Form", "OK", None))
        self.filter_job_checkBox.setText(_translate("Form", "My Jobs", None))
        self.filter_exist_checkBox.setText(_translate("Form", "No Exists", None))
