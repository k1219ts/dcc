# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_SceneSetup.ui'
#
# Created: Thu Nov 16 19:57:42 2017
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

import pymodule.Qt as Qt
import pymodule.Qt.QtWidgets as QtGui
from pymodule.Qt import QtCore


try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName(_fromUtf8("Form"))
        Form.resize(1200, 800)
        Form.setMinimumSize(QtCore.QSize(1200, 800))
        self.gridLayout_8 = QtGui.QGridLayout(Form)
        self.gridLayout_8.setObjectName(_fromUtf8("gridLayout_8"))
        self.gridLayout_6 = QtGui.QGridLayout()
        self.gridLayout_6.setObjectName(_fromUtf8("gridLayout_6"))
        self.frame = QtGui.QFrame(Form)
        self.frame.setMinimumSize(QtCore.QSize(0, 70))
        self.frame.setMaximumSize(QtCore.QSize(16777215, 70))
        self.frame.setObjectName(_fromUtf8("frame"))
        self.gridLayout_10 = QtGui.QGridLayout(self.frame)
        self.gridLayout_10.setObjectName(_fromUtf8("gridLayout_10"))
        spacerItem = QtGui.QSpacerItem(267, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout_10.addItem(spacerItem, 0, 2, 1, 1)
        self.label = QtGui.QLabel(self.frame)
        self.label.setMinimumSize(QtCore.QSize(0, 50))
        font = Qt.QtGui.QFont()
        font.setFamily(_fromUtf8("Cantarell"))
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setMargin(10)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout_10.addWidget(self.label, 0, 5, 1, 1)
        self.logo_label = QtGui.QLabel(self.frame)
        self.logo_label.setMinimumSize(QtCore.QSize(50, 50))
        self.logo_label.setMaximumSize(QtCore.QSize(50, 50))
        self.logo_label.setText(_fromUtf8(""))
        self.logo_label.setScaledContents(True)
        self.logo_label.setObjectName(_fromUtf8("logo_label"))
        self.gridLayout_10.addWidget(self.logo_label, 0, 4, 1, 1)
        self.show_comboBox = QtGui.QComboBox(self.frame)
        self.show_comboBox.setMinimumSize(QtCore.QSize(150, 25))
        self.show_comboBox.setMaximumSize(QtCore.QSize(16777215, 40))
        font = Qt.QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.show_comboBox.setFont(font)
        self.show_comboBox.setStyleSheet(_fromUtf8("Qcombobox {\'qproperty-alignment: \'AlignBottom|AlignRight\';\'}"))
        self.show_comboBox.setObjectName(_fromUtf8("show_comboBox"))
        self.gridLayout_10.addWidget(self.show_comboBox, 0, 1, 1, 1)
        self.label_2 = QtGui.QLabel(self.frame)
        self.label_2.setMinimumSize(QtCore.QSize(70, 30))
        self.label_2.setMaximumSize(QtCore.QSize(16777215, 50))
        font = Qt.QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout_10.addWidget(self.label_2, 0, 0, 1, 1)
        spacerItem1 = QtGui.QSpacerItem(10, 10, QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Minimum)
        self.gridLayout_10.addItem(spacerItem1, 0, 6, 1, 1)
        self.gridLayout_6.addWidget(self.frame, 0, 0, 1, 1)
        self.gridLayout_8.addLayout(self.gridLayout_6, 0, 0, 1, 1)
        spacerItem2 = QtGui.QSpacerItem(478, 13, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        self.gridLayout_8.addItem(spacerItem2, 1, 0, 1, 1)
        self.splitter_2 = QtGui.QSplitter(Form)
        self.splitter_2.setOrientation(QtCore.Qt.Vertical)
        self.splitter_2.setObjectName(_fromUtf8("splitter_2"))
        self.splitter = QtGui.QSplitter(self.splitter_2)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName(_fromUtf8("splitter"))
        self.tabWidget = QtGui.QTabWidget(self.splitter)
        self.tabWidget.setMinimumSize(QtCore.QSize(0, 359))
        font = Qt.QtGui.QFont()
        font.setFamily(_fromUtf8("PT Sans"))
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.tabWidget.setFont(font)
        self.tabWidget.setStyleSheet(_fromUtf8(""))
        self.tabWidget.setObjectName(_fromUtf8("tabWidget"))
        self.shot_tab = QtGui.QWidget()
        self.shot_tab.setObjectName(_fromUtf8("shot_tab"))
        self.gridLayout_14 = QtGui.QGridLayout(self.shot_tab)
        self.gridLayout_14.setObjectName(_fromUtf8("gridLayout_14"))
        self.gridLayout_3 = QtGui.QGridLayout()
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.gridLayout_9 = QtGui.QGridLayout()
        self.gridLayout_9.setContentsMargins(-1, 6, -1, 6)
        self.gridLayout_9.setObjectName(_fromUtf8("gridLayout_9"))
        self.shot_comboBox = QtGui.QComboBox(self.shot_tab)
        self.shot_comboBox.setMinimumSize(QtCore.QSize(120, 30))
        self.shot_comboBox.setMaximumSize(QtCore.QSize(120, 30))
        font = Qt.QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.shot_comboBox.setFont(font)
        self.shot_comboBox.setObjectName(_fromUtf8("shot_comboBox"))
        self.gridLayout_9.addWidget(self.shot_comboBox, 0, 5, 1, 1)
        self.seq_comboBox = QtGui.QComboBox(self.shot_tab)
        self.seq_comboBox.setMinimumSize(QtCore.QSize(100, 30))
        self.seq_comboBox.setMaximumSize(QtCore.QSize(100, 30))
        font = Qt.QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.seq_comboBox.setFont(font)
        self.seq_comboBox.setObjectName(_fromUtf8("seq_comboBox"))
        self.gridLayout_9.addWidget(self.seq_comboBox, 0, 3, 1, 1)
        self.findShot_lineEdit = QtGui.QLineEdit(self.shot_tab)
        self.findShot_lineEdit.setMinimumSize(QtCore.QSize(100, 30))
        self.findShot_lineEdit.setMaximumSize(QtCore.QSize(100, 30))
        font = Qt.QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.findShot_lineEdit.setFont(font)
        self.findShot_lineEdit.setObjectName(_fromUtf8("findShot_lineEdit"))
        self.gridLayout_9.addWidget(self.findShot_lineEdit, 0, 6, 1, 1)
        self.label_3 = QtGui.QLabel(self.shot_tab)
        self.label_3.setMinimumSize(QtCore.QSize(80, 30))
        self.label_3.setMaximumSize(QtCore.QSize(80, 50))
        font = Qt.QtGui.QFont()
        font.setFamily(_fromUtf8("PT Sans"))
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout_9.addWidget(self.label_3, 0, 2, 1, 1)
        self.label_4 = QtGui.QLabel(self.shot_tab)
        self.label_4.setMinimumSize(QtCore.QSize(80, 30))
        self.label_4.setMaximumSize(QtCore.QSize(80, 50))
        font = Qt.QtGui.QFont()
        font.setFamily(_fromUtf8("PT Sans"))
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.gridLayout_9.addWidget(self.label_4, 0, 4, 1, 1)
        spacerItem3 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout_9.addItem(spacerItem3, 0, 7, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout_9, 0, 0, 1, 2)
        self.dataTypeList_treeWidget = QtGui.QTreeWidget(self.shot_tab)
        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        self.dataTypeList_treeWidget.setFont(font)
        self.dataTypeList_treeWidget.setStyleSheet(_fromUtf8("QTableWidget::item { padding: 5 px; margin: 5px; border: 0 px}\n"
"QTableWidget::item:selected{background-color: grey;}"))
        self.dataTypeList_treeWidget.setProperty("showDropIndicator", True)
        self.dataTypeList_treeWidget.setAlternatingRowColors(False)
        self.dataTypeList_treeWidget.setSelectionMode(QtGui.QAbstractItemView.ContiguousSelection)
        self.dataTypeList_treeWidget.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.dataTypeList_treeWidget.setRootIsDecorated(False)
        self.dataTypeList_treeWidget.setObjectName(_fromUtf8("dataTypeList_treeWidget"))
        self.dataTypeList_treeWidget.header().setDefaultSectionSize(150)
        self.dataTypeList_treeWidget.header().setStretchLastSection(False)
        self.gridLayout_3.addWidget(self.dataTypeList_treeWidget, 1, 1, 1, 1)
        self.gridLayout_5 = QtGui.QGridLayout()
        self.gridLayout_5.setObjectName(_fromUtf8("gridLayout_5"))
        self.delQuick_pushButton = QtGui.QPushButton(self.shot_tab)
        self.delQuick_pushButton.setMinimumSize(QtCore.QSize(80, 0))
        self.delQuick_pushButton.setMaximumSize(QtCore.QSize(80, 16777215))
        font = Qt.QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.delQuick_pushButton.setFont(font)
        self.delQuick_pushButton.setObjectName(_fromUtf8("delQuick_pushButton"))
        self.gridLayout_5.addWidget(self.delQuick_pushButton, 3, 0, 1, 1)
        self.addQuick_pushButton = QtGui.QPushButton(self.shot_tab)
        self.addQuick_pushButton.setMinimumSize(QtCore.QSize(80, 0))
        self.addQuick_pushButton.setMaximumSize(QtCore.QSize(80, 16777215))
        font = Qt.QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.addQuick_pushButton.setFont(font)
        self.addQuick_pushButton.setObjectName(_fromUtf8("addQuick_pushButton"))
        self.gridLayout_5.addWidget(self.addQuick_pushButton, 2, 0, 1, 1)
        self.quick_listWidget = QtGui.QListWidget(self.shot_tab)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.quick_listWidget.sizePolicy().hasHeightForWidth())
        self.quick_listWidget.setSizePolicy(sizePolicy)
        self.quick_listWidget.setMinimumSize(QtCore.QSize(80, 0))
        self.quick_listWidget.setMaximumSize(QtCore.QSize(80, 16777215))
        font = Qt.QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.quick_listWidget.setFont(font)
        self.quick_listWidget.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.quick_listWidget.setObjectName(_fromUtf8("quick_listWidget"))
        self.gridLayout_5.addWidget(self.quick_listWidget, 0, 0, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout_5, 1, 0, 1, 1)
        self.gridLayout_14.addLayout(self.gridLayout_3, 0, 0, 1, 1)
        self.tabWidget.addTab(self.shot_tab, _fromUtf8(""))
        self.asset_tab = QtGui.QWidget()
        self.asset_tab.setObjectName(_fromUtf8("asset_tab"))
        self.gridLayout_17 = QtGui.QGridLayout(self.asset_tab)
        self.gridLayout_17.setObjectName(_fromUtf8("gridLayout_17"))
        self.gridLayout_16 = QtGui.QGridLayout()
        self.gridLayout_16.setObjectName(_fromUtf8("gridLayout_16"))
        self.gridLayout_15 = QtGui.QGridLayout()
        self.gridLayout_15.setContentsMargins(-1, 6, -1, 6)
        self.gridLayout_15.setObjectName(_fromUtf8("gridLayout_15"))
        self.lineEdit = QtGui.QLineEdit(self.asset_tab)
        self.lineEdit.setMinimumSize(QtCore.QSize(0, 30))
        self.lineEdit.setMaximumSize(QtCore.QSize(100, 30))
        font = Qt.QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.lineEdit.setFont(font)
        self.lineEdit.setObjectName(_fromUtf8("lineEdit"))
        self.gridLayout_15.addWidget(self.lineEdit, 0, 4, 1, 1)
        spacerItem4 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout_15.addItem(spacerItem4, 0, 5, 1, 1)
        self.label_8 = QtGui.QLabel(self.asset_tab)
        self.label_8.setMinimumSize(QtCore.QSize(100, 0))
        self.label_8.setMaximumSize(QtCore.QSize(100, 16777215))
        font = Qt.QtGui.QFont()
        font.setFamily(_fromUtf8("PT Sans"))
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.gridLayout_15.addWidget(self.label_8, 0, 2, 1, 1)
        self.assetName_comboBox = QtGui.QComboBox(self.asset_tab)
        self.assetName_comboBox.setMinimumSize(QtCore.QSize(180, 30))
        self.assetName_comboBox.setMaximumSize(QtCore.QSize(180, 30))
        font = Qt.QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.assetName_comboBox.setFont(font)
        self.assetName_comboBox.setObjectName(_fromUtf8("assetName_comboBox"))
        self.gridLayout_15.addWidget(self.assetName_comboBox, 0, 3, 1, 1)
        self.label_9 = QtGui.QLabel(self.asset_tab)
        self.label_9.setMinimumSize(QtCore.QSize(100, 0))
        self.label_9.setMaximumSize(QtCore.QSize(100, 16777215))
        font = Qt.QtGui.QFont()
        font.setFamily(_fromUtf8("PT Sans"))
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_9.setFont(font)
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName(_fromUtf8("label_9"))
        self.gridLayout_15.addWidget(self.label_9, 0, 0, 1, 1)
        self.assetType_comboBox = QtGui.QComboBox(self.asset_tab)
        self.assetType_comboBox.setMinimumSize(QtCore.QSize(100, 30))
        self.assetType_comboBox.setMaximumSize(QtCore.QSize(100, 30))
        font = Qt.QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.assetType_comboBox.setFont(font)
        self.assetType_comboBox.setObjectName(_fromUtf8("assetType_comboBox"))
        self.gridLayout_15.addWidget(self.assetType_comboBox, 0, 1, 1, 1)
        self.gridLayout_16.addLayout(self.gridLayout_15, 0, 0, 1, 2)
        self.asset_dataTypeList_treeWidget = QtGui.QTreeWidget(self.asset_tab)
        self.asset_dataTypeList_treeWidget.setMinimumSize(QtCore.QSize(0, 262))
        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        self.asset_dataTypeList_treeWidget.setFont(font)
        self.asset_dataTypeList_treeWidget.setStyleSheet(_fromUtf8("QTableWidget::item { padding: 5 px; margin: 5px; border: 0 px}\n"
"QTableWidget::item:selected{background-color: grey;}"))
        self.asset_dataTypeList_treeWidget.setProperty("showDropIndicator", False)
        self.asset_dataTypeList_treeWidget.setAlternatingRowColors(False)
        self.asset_dataTypeList_treeWidget.setSelectionMode(QtGui.QAbstractItemView.ContiguousSelection)
        self.asset_dataTypeList_treeWidget.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.asset_dataTypeList_treeWidget.setRootIsDecorated(False)
        self.asset_dataTypeList_treeWidget.setItemsExpandable(False)
        self.asset_dataTypeList_treeWidget.setAllColumnsShowFocus(False)
        self.asset_dataTypeList_treeWidget.setHeaderHidden(False)
        self.asset_dataTypeList_treeWidget.setExpandsOnDoubleClick(False)
        self.asset_dataTypeList_treeWidget.setObjectName(_fromUtf8("asset_dataTypeList_treeWidget"))
        self.asset_dataTypeList_treeWidget.header().setDefaultSectionSize(150)
        self.asset_dataTypeList_treeWidget.header().setStretchLastSection(False)
        self.gridLayout_16.addWidget(self.asset_dataTypeList_treeWidget, 1, 1, 1, 1)
        self.gridLayout_17.addLayout(self.gridLayout_16, 0, 0, 1, 1)
        self.tabWidget.addTab(self.asset_tab, _fromUtf8(""))
        self.layoutWidget = QtGui.QWidget(self.splitter)
        self.layoutWidget.setObjectName(_fromUtf8("layoutWidget"))
        self.gridLayout_4 = QtGui.QGridLayout(self.layoutWidget)
        self.gridLayout_4.setMargin(0)
        self.gridLayout_4.setObjectName(_fromUtf8("gridLayout_4"))
        self.DBdatatype_label = QtGui.QLabel(self.layoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.DBdatatype_label.sizePolicy().hasHeightForWidth())
        self.DBdatatype_label.setSizePolicy(sizePolicy)
        font = Qt.QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.DBdatatype_label.setFont(font)
        self.DBdatatype_label.setStyleSheet(_fromUtf8("QLabel { padding: 2 10 2 10 px;}"))
        self.DBdatatype_label.setObjectName(_fromUtf8("DBdatatype_label"))
        self.gridLayout_4.addWidget(self.DBdatatype_label, 0, 0, 1, 1)
        self.db_treeWidget = QtGui.QTreeWidget(self.layoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.db_treeWidget.sizePolicy().hasHeightForWidth())
        self.db_treeWidget.setSizePolicy(sizePolicy)
        font = Qt.QtGui.QFont()
        font.setFamily(_fromUtf8("Cantarell"))
        font.setPointSize(10)
        self.db_treeWidget.setFont(font)
        self.db_treeWidget.setRootIsDecorated(False)
        self.db_treeWidget.setObjectName(_fromUtf8("db_treeWidget"))
        self.db_treeWidget.header().setDefaultSectionSize(80)
        self.gridLayout_4.addWidget(self.db_treeWidget, 1, 0, 1, 1)
        self.ReadMoreDB_pushButton = QtGui.QPushButton(self.layoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ReadMoreDB_pushButton.sizePolicy().hasHeightForWidth())
        self.ReadMoreDB_pushButton.setSizePolicy(sizePolicy)
        self.ReadMoreDB_pushButton.setObjectName(_fromUtf8("ReadMoreDB_pushButton"))
        self.gridLayout_4.addWidget(self.ReadMoreDB_pushButton, 2, 0, 1, 1)
        self.frame1 = QtGui.QFrame(self.splitter_2)
        self.frame1.setMinimumSize(QtCore.QSize(0, 0))
        self.frame1.setMaximumSize(QtCore.QSize(16777215, 200))
        self.frame1.setObjectName(_fromUtf8("frame1"))
        self.gridLayout_12 = QtGui.QGridLayout(self.frame1)
        self.gridLayout_12.setObjectName(_fromUtf8("gridLayout_12"))
        self.groupBox_2 = QtGui.QGroupBox(self.frame1)
        self.groupBox_2.setMinimumSize(QtCore.QSize(0, 190))
        self.groupBox_2.setMaximumSize(QtCore.QSize(16777215, 190))
        font = Qt.QtGui.QFont()
        font.setFamily(_fromUtf8("PT Sans"))
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName(_fromUtf8("groupBox_2"))
        self.gridLayout_13 = QtGui.QGridLayout(self.groupBox_2)
        self.gridLayout_13.setContentsMargins(20, -1, 20, 25)
        self.gridLayout_13.setObjectName(_fromUtf8("gridLayout_13"))
        self.groupBox = QtGui.QGroupBox(self.groupBox_2)
        self.groupBox.setMinimumSize(QtCore.QSize(0, 50))
        font = Qt.QtGui.QFont()
        font.setFamily(_fromUtf8("PT Sans"))
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.gridLayout = QtGui.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.GPU_radioButton = QtGui.QRadioButton(self.groupBox)
        self.GPU_radioButton.setMinimumSize(QtCore.QSize(100, 0))
        font = Qt.QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.GPU_radioButton.setFont(font)
        self.GPU_radioButton.setStyleSheet(_fromUtf8("QRadioButton::indicator{height:15px;width:15px;}"))
        self.GPU_radioButton.setObjectName(_fromUtf8("GPU_radioButton"))
        self.gridLayout.addWidget(self.GPU_radioButton, 0, 0, 1, 1)
        self.Mesh_radioButton = QtGui.QRadioButton(self.groupBox)
        self.Mesh_radioButton.setMinimumSize(QtCore.QSize(100, 20))
        font = Qt.QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.Mesh_radioButton.setFont(font)
        self.Mesh_radioButton.setStyleSheet(_fromUtf8("QRadioButton::indicator{height:15px;width:15px;}"))
        self.Mesh_radioButton.setObjectName(_fromUtf8("Mesh_radioButton"))
        self.gridLayout.addWidget(self.Mesh_radioButton, 0, 1, 1, 1)
        self.gridLayout_13.addWidget(self.groupBox, 0, 0, 1, 1)
        self.groupBox1 = QtGui.QGroupBox(self.groupBox_2)
        self.groupBox1.setMinimumSize(QtCore.QSize(0, 50))
        font = Qt.QtGui.QFont()
        font.setFamily(_fromUtf8("PT Sans"))
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox1.setFont(font)
        self.groupBox1.setObjectName(_fromUtf8("groupBox1"))
        self.gridLayout_11 = QtGui.QGridLayout(self.groupBox1)
        self.gridLayout_11.setObjectName(_fromUtf8("gridLayout_11"))
        self.None_radioButton = QtGui.QRadioButton(self.groupBox1)
        self.None_radioButton.setMinimumSize(QtCore.QSize(100, 20))
        font = Qt.QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.None_radioButton.setFont(font)
        self.None_radioButton.setStyleSheet(_fromUtf8("QRadioButton::indicator{height:15px;width:15px;}"))
        self.None_radioButton.setObjectName(_fromUtf8("None_radioButton"))
        self.gridLayout_11.addWidget(self.None_radioButton, 0, 3, 1, 1)
        self.seperate_radioButton = QtGui.QRadioButton(self.groupBox1)
        self.seperate_radioButton.setMinimumSize(QtCore.QSize(100, 20))
        font = Qt.QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.seperate_radioButton.setFont(font)
        self.seperate_radioButton.setStyleSheet(_fromUtf8("QRadioButton::indicator{height:15px;width:15px;}"))
        self.seperate_radioButton.setObjectName(_fromUtf8("seperate_radioButton"))
        self.gridLayout_11.addWidget(self.seperate_radioButton, 0, 1, 1, 1)
        self.baked_radioButton = QtGui.QRadioButton(self.groupBox1)
        self.baked_radioButton.setMinimumSize(QtCore.QSize(100, 20))
        font = Qt.QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.baked_radioButton.setFont(font)
        self.baked_radioButton.setStyleSheet(_fromUtf8("QRadioButton::indicator{height:15px;width:15px;}"))
        self.baked_radioButton.setObjectName(_fromUtf8("baked_radioButton"))
        self.gridLayout_11.addWidget(self.baked_radioButton, 0, 0, 1, 1)
        self.gridLayout_13.addWidget(self.groupBox1, 2, 0, 1, 2)
        self.groupBox2 = QtGui.QGroupBox(self.groupBox_2)
        self.groupBox2.setMinimumSize(QtCore.QSize(0, 50))
        font = Qt.QtGui.QFont()
        font.setFamily(_fromUtf8("PT Sans"))
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox2.setFont(font)
        self.groupBox2.setObjectName(_fromUtf8("groupBox2"))
        self.gridLayout_2 = QtGui.QGridLayout(self.groupBox2)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.static_radioButton = QtGui.QRadioButton(self.groupBox2)
        self.static_radioButton.setMinimumSize(QtCore.QSize(100, 20))
        font = Qt.QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.static_radioButton.setFont(font)
        self.static_radioButton.setStyleSheet(_fromUtf8("QRadioButton::indicator{height:15px;width:15px;}"))
        self.static_radioButton.setObjectName(_fromUtf8("static_radioButton"))
        self.gridLayout_2.addWidget(self.static_radioButton, 0, 0, 1, 1)
        self.simulation_radioButton = QtGui.QRadioButton(self.groupBox2)
        self.simulation_radioButton.setMinimumSize(QtCore.QSize(100, 20))
        font = Qt.QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.simulation_radioButton.setFont(font)
        self.simulation_radioButton.setStyleSheet(_fromUtf8("QRadioButton::indicator{height:15px;width:15px;}"))
        self.simulation_radioButton.setObjectName(_fromUtf8("simulation_radioButton"))
        self.gridLayout_2.addWidget(self.simulation_radioButton, 0, 1, 1, 1)
        self.auto_radioButton = QtGui.QRadioButton(self.groupBox2)
        self.auto_radioButton.setMinimumSize(QtCore.QSize(100, 20))
        font = Qt.QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.auto_radioButton.setFont(font)
        self.auto_radioButton.setStyleSheet(_fromUtf8("QRadioButton::indicator{height:15px;width:15px;}"))
        self.auto_radioButton.setObjectName(_fromUtf8("auto_radioButton"))
        self.gridLayout_2.addWidget(self.auto_radioButton, 0, 2, 1, 1)
        self.gridLayout_13.addWidget(self.groupBox2, 1, 0, 1, 2)
        self.gridLayout_12.addWidget(self.groupBox_2, 0, 0, 1, 1)
        self.gridLayout_7 = QtGui.QGridLayout()
        self.gridLayout_7.setContentsMargins(-1, -1, 25, -1)
        self.gridLayout_7.setObjectName(_fromUtf8("gridLayout_7"))
        spacerItem5 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Minimum)
        self.gridLayout_7.addItem(spacerItem5, 0, 0, 1, 1)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        spacerItem6 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem6)
        self.spanner2_pushButton = QtGui.QPushButton(self.frame1)
        self.spanner2_pushButton.setMinimumSize(QtCore.QSize(35, 35))
        self.spanner2_pushButton.setMaximumSize(QtCore.QSize(35, 35))
        self.spanner2_pushButton.setText(_fromUtf8(""))
        self.spanner2_pushButton.setIconSize(QtCore.QSize(30, 30))
        self.spanner2_pushButton.setAutoRepeat(False)
        self.spanner2_pushButton.setAutoExclusive(False)
        self.spanner2_pushButton.setAutoDefault(False)
        self.spanner2_pushButton.setDefault(False)
        self.spanner2_pushButton.setFlat(False)
        self.spanner2_pushButton.setObjectName(_fromUtf8("spanner2_pushButton"))
        self.horizontalLayout.addWidget(self.spanner2_pushButton)
        self.renderSpool_pushButton = QtGui.QPushButton(self.frame1)
        self.renderSpool_pushButton.setMinimumSize(QtCore.QSize(35, 35))
        self.renderSpool_pushButton.setMaximumSize(QtCore.QSize(35, 35))
        self.renderSpool_pushButton.setText(_fromUtf8(""))
        self.renderSpool_pushButton.setIconSize(QtCore.QSize(30, 30))
        self.renderSpool_pushButton.setObjectName(_fromUtf8("renderSpool_pushButton"))
        self.horizontalLayout.addWidget(self.renderSpool_pushButton)
        self.gridLayout_7.addLayout(self.horizontalLayout, 0, 1, 1, 1)
        spacerItem7 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout_7.addItem(spacerItem7, 1, 1, 1, 1)
        spacerItem8 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout_7.addItem(spacerItem8, 2, 0, 1, 1)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        spacerItem9 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem9)
        self.import_pushButton = QtGui.QPushButton(self.frame1)
        self.import_pushButton.setMinimumSize(QtCore.QSize(150, 35))
        self.import_pushButton.setMaximumSize(QtCore.QSize(120, 16777215))
        font = Qt.QtGui.QFont()
        font.setFamily(_fromUtf8("Cantarell"))
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.import_pushButton.setFont(font)
        self.import_pushButton.setObjectName(_fromUtf8("import_pushButton"))
        self.horizontalLayout_2.addWidget(self.import_pushButton)
        self.cancel_pushButton = QtGui.QPushButton(self.frame1)
        self.cancel_pushButton.setMinimumSize(QtCore.QSize(150, 35))
        self.cancel_pushButton.setMaximumSize(QtCore.QSize(120, 16777215))
        font = Qt.QtGui.QFont()
        font.setFamily(_fromUtf8("Cantarell"))
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.cancel_pushButton.setFont(font)
        self.cancel_pushButton.setObjectName(_fromUtf8("cancel_pushButton"))
        self.horizontalLayout_2.addWidget(self.cancel_pushButton)
        self.gridLayout_7.addLayout(self.horizontalLayout_2, 2, 1, 1, 1)
        self.gridLayout_12.addLayout(self.gridLayout_7, 0, 1, 1, 1)
        self.gridLayout_8.addWidget(self.splitter_2, 2, 0, 1, 1)

        self.retranslateUi(Form)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Form", None))
        self.label.setText(_translate("Form", "SCENE SETUP MANAGER", None))
        self.label_2.setText(_translate("Form", "SHOW", None))
        self.label_3.setText(_translate("Form", "SEQUENCE", None))
        self.label_4.setText(_translate("Form", "SHOT NUM.", None))
        self.dataTypeList_treeWidget.headerItem().setText(1, _translate("Form", "geo", None))
        self.dataTypeList_treeWidget.headerItem().setText(2, _translate("Form", "zenn", None))
        self.dataTypeList_treeWidget.headerItem().setText(3, _translate("Form", "Element", None))
        self.dataTypeList_treeWidget.headerItem().setText(4, _translate("Form", "Type", None))
        self.dataTypeList_treeWidget.headerItem().setText(5, _translate("Form", "Version", None))
        self.dataTypeList_treeWidget.headerItem().setText(6, _translate("Form", "Type", None))
        self.dataTypeList_treeWidget.headerItem().setText(7, _translate("Form", "Version", None))
        self.delQuick_pushButton.setText(_translate("Form", "Delete", None))
        self.addQuick_pushButton.setText(_translate("Form", "+ Add", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.shot_tab), _translate("Form", "SHOT", None))
        self.label_8.setText(_translate("Form", "ASSET NAME", None))
        self.label_9.setText(_translate("Form", "ASSET TYPE", None))
        self.asset_dataTypeList_treeWidget.headerItem().setText(1, _translate("Form", "Type", None))
        self.asset_dataTypeList_treeWidget.headerItem().setText(2, _translate("Form", "Version", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.asset_tab), _translate("Form", "ASSET", None))
        self.DBdatatype_label.setText(_translate("Form", "DB viewer", None))
        self.db_treeWidget.headerItem().setText(0, _translate("Form", "Type", None))
        self.db_treeWidget.headerItem().setText(1, _translate("Form", "Ver", None))
        self.db_treeWidget.headerItem().setText(2, _translate("Form", "Artist", None))
        self.db_treeWidget.headerItem().setText(3, _translate("Form", "Time", None))
        self.db_treeWidget.headerItem().setText(4, _translate("Form", "File", None))
        self.ReadMoreDB_pushButton.setText(_translate("Form", "+ Read More ...", None))
        self.groupBox_2.setTitle(_translate("Form", "Settings", None))
        self.groupBox.setTitle(_translate("Form", "Mesh", None))
        self.GPU_radioButton.setText(_translate("Form", "GPU", None))
        self.Mesh_radioButton.setText(_translate("Form", "Mesh", None))
        self.groupBox1.setTitle(_translate("Form", "World", None))
        self.None_radioButton.setText(_translate("Form", "None", None))
        self.seperate_radioButton.setText(_translate("Form", "Seperate", None))
        self.baked_radioButton.setText(_translate("Form", "Baked", None))
        self.groupBox2.setTitle(_translate("Form", "Zenn", None))
        self.static_radioButton.setText(_translate("Form", "Static", None))
        self.simulation_radioButton.setText(_translate("Form", "Simulation", None))
        self.auto_radioButton.setText(_translate("Form", "Auto", None))
        self.import_pushButton.setText(_translate("Form", "IMPORT", None))
        self.cancel_pushButton.setText(_translate("Form", "CANCEL", None))

