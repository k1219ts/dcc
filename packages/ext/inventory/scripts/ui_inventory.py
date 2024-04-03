# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'inventory.ui'
#
# Created: Thu Jan 12 10:34:02 2017
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

#from PyQt4 import QtCore, QtWidgets
from pymodule import Qt
from pymodule.Qt import QtGui
from pymodule.Qt import QtWidgets
from pymodule.Qt import QtCore
from viewer import BaseList, BookmarkList, CategoryTree

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
        Form.resize(1800, 1000)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.gridLayout.setSpacing(0)
        self.gridLayout.setMargin(0)
        self.splitter = QtWidgets.QSplitter(Form)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName(_fromUtf8("splitter"))
        self.menuTab = QtWidgets.QTabWidget(self.splitter)
        self.menuTab.setMaximumSize(QtCore.QSize(420, 16777215))
        self.menuTab.setFocusPolicy(QtCore.Qt.NoFocus)
        self.menuTab.setObjectName(_fromUtf8("menuTab"))
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName(_fromUtf8("tab"))
        self.gridLayout_2 = QtWidgets.QGridLayout(self.tab)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        #self.categoryTree = QtWidgets.QTreeWidget(self.tab)
        self.categoryTree = CategoryTree(self.tab)
        # self.categoryTree.setFocusPolicy(QtCore.Qt.NoFocus)
        # self.categoryTree.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        # self.categoryTree.setObjectName(_fromUtf8("categoryTree"))
        # self.categoryTree.headerItem().setText(0, _fromUtf8("1"))
        # self.categoryTree.header().setVisible(False)
        self.gridLayout_2.addWidget(self.categoryTree, 0, 0, 1, 1)
        self.menuTab.addTab(self.tab, _fromUtf8(""))
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName(_fromUtf8("tab_2"))
        self.gridLayout_3 = QtWidgets.QGridLayout(self.tab_2)
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.categoryCombo = QtWidgets.QComboBox(self.tab_2)
        self.categoryCombo.setObjectName(_fromUtf8("categoryCombo"))
        self.gridLayout_3.addWidget(self.categoryCombo, 1, 0, 1, 1)
        self.matchAllRadio = QtWidgets.QRadioButton(self.tab_2)
        self.matchAllRadio.setObjectName(_fromUtf8("matchAllRadio"))
        self.matchAllRadio.setChecked(True)
        self.gridLayout_3.addWidget(self.matchAllRadio, 1, 1, 1, 1)
        self.matchAnyRadio = QtWidgets.QRadioButton(self.tab_2)
        self.matchAnyRadio.setObjectName(_fromUtf8("matchAnyRadio"))
        self.gridLayout_3.addWidget(self.matchAnyRadio, 1, 2, 1, 1)
        self.searchButton = QtWidgets.QPushButton(self.tab_2)
        self.searchButton.setObjectName(_fromUtf8("searchButton"))
        self.gridLayout_3.addWidget(self.searchButton, 1, 3, 1, 1)
        self.tagLine = QtWidgets.QLineEdit(self.tab_2)
        self.tagLine.setObjectName(_fromUtf8("tagLine"))
        self.gridLayout_3.addWidget(self.tagLine, 0, 0, 1, 4)
        self.tagList = QtWidgets.QListWidget(self.tab_2)
        self.tagList.setFocusPolicy(QtCore.Qt.NoFocus)
        self.tagList.setResizeMode(QtWidgets.QListView.Adjust)
        self.tagList.setObjectName(_fromUtf8("tagList"))
        self.gridLayout_3.addWidget(self.tagList, 3, 0, 1, 4)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.uploadTimeCheck = QtWidgets.QCheckBox(self.tab_2)
        self.uploadTimeCheck.setObjectName(_fromUtf8("uploadCheck"))
        self.horizontalLayout.addWidget(self.uploadTimeCheck)
        self.fromDate = QtWidgets.QDateEdit(self.tab_2)
        self.fromDate.setObjectName(_fromUtf8("fromDate"))
        self.fromDate.setDisplayFormat('yyyy/MM/dd')
        self.fromDate.setCalendarPopup(True)
        self.horizontalLayout.addWidget(self.fromDate)
        self.dateLabel = QtWidgets.QLabel(self.tab_2)
        self.dateLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.dateLabel.setObjectName(_fromUtf8("dateLabel"))
        self.horizontalLayout.addWidget(self.dateLabel)
        self.toDate = QtWidgets.QDateEdit(self.tab_2)
        self.toDate.setObjectName(_fromUtf8("toDate"))
        self.toDate.setDisplayFormat('yyyy/MM/dd')
        self.toDate.setCalendarPopup(True)
        self.horizontalLayout.addWidget(self.toDate)
        self.gridLayout_3.addLayout(self.horizontalLayout, 2, 0, 1, 4)
        self.menuTab.addTab(self.tab_2, _fromUtf8(""))
        #
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName(_fromUtf8("tab_4"))
        self.gridLayout_4 = QtWidgets.QGridLayout(self.tab_4)
        self.gridLayout_4.setObjectName(_fromUtf8("gridLayout_4"))

        self.playSpeedSpin = QtWidgets.QDoubleSpinBox(self.tab_4)
        self.playSpeedSpin.setObjectName(_fromUtf8("playSpeedSpin"))
        self.gridLayout_4.addWidget(self.playSpeedSpin, 4, 1, 1, 1)
        self.playSpeedLabel = QtWidgets.QLabel(self.tab_4)
        self.playSpeedLabel.setObjectName(_fromUtf8("playSpeedLabel"))
        self.gridLayout_4.addWidget(self.playSpeedLabel, 4, 0, 1, 1)

        self.itemPerClickLabel = QtWidgets.QLabel(self.tab_4)
        self.itemPerClickLabel.setObjectName(_fromUtf8("itemPerClickLabel"))
        self.gridLayout_4.addWidget(self.itemPerClickLabel, 3, 0, 1, 1)
        self.itemPerClickSpin = QtWidgets.QSpinBox(self.tab_4)
        self.itemPerClickSpin.setMinimum(10)
        self.itemPerClickSpin.setMaximum(100)
        self.itemPerClickSpin.setProperty("value", 20)
        self.itemPerClickSpin.setObjectName(_fromUtf8("itemPerClickSpin"))
        self.gridLayout_4.addWidget(self.itemPerClickSpin, 3, 1, 1, 1)
        self.mouseOverToPlayButton = QtWidgets.QRadioButton(self.tab_4)
        self.mouseOverToPlayButton.setAutoExclusive(True)
        self.mouseOverToPlayButton.setObjectName(_fromUtf8("mouseOverToPlayButton"))
        self.gridLayout_4.addWidget(self.mouseOverToPlayButton, 1, 1, 1, 1)
        self.clickToPlayButton = QtWidgets.QRadioButton(self.tab_4)
        self.clickToPlayButton.setChecked(True)
        self.clickToPlayButton.setAutoExclusive(True)
        self.clickToPlayButton.setObjectName(_fromUtf8("clickToPlayButton"))
        self.gridLayout_4.addWidget(self.clickToPlayButton, 1, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 610, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_4.addItem(spacerItem, 5, 0, 1, 1)
        self.autoReplayCheck = QtWidgets.QCheckBox(self.tab_4)
        self.autoReplayCheck.setObjectName(_fromUtf8("autoReplayCheck"))
        self.gridLayout_4.addWidget(self.autoReplayCheck, 2, 0, 1, 2)
        self.configLabel = QtWidgets.QLabel(self.tab_4)
        self.configLabel.setObjectName(_fromUtf8("configLabel"))
        self.gridLayout_4.addWidget(self.configLabel, 0, 0, 1, 2)
        self.menuTab.addTab(self.tab_4, _fromUtf8(""))
        #
        # self.tab_5 = QtWidgets.QWidget()
        # self.menuTab.addTab(self.tab_5, "test")
        self.layoutWidget_2 = QtWidgets.QWidget(self.splitter)
        self.layoutWidget_2.setObjectName(_fromUtf8("layoutWidget_2"))
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.layoutWidget_2)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setMargin(0)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.mainTab = QtWidgets.QTabWidget(self.layoutWidget_2)
        self.mainTab.setFocusPolicy(QtCore.Qt.NoFocus)
        self.mainTab.setTabsClosable(True)
        self.mainTab.setObjectName(_fromUtf8("mainTab"))
        # self.tab_3 = QtWidgets.QWidget()
        # self.tab_3.setObjectName(_fromUtf8("tab_3"))
        # self.mainTab.addTab(self.tab_3, _fromUtf8(""))
        self.verticalLayout_2.addWidget(self.mainTab)
        #
        #self.itemTagList = QtWidgets.QListWidget(Form)
        self.itemTagList = QtWidgets.QListWidget(Form)
        self.itemTagList.setMaximumHeight(100)
        self.itemTagList.setViewMode(QtWidgets.QListView.IconMode)
        self.itemTagList.setFocusPolicy(QtCore.Qt.NoFocus)

        self.verticalLayout_2.addWidget(self.itemTagList)
        #
        self.layoutWidget = QtWidgets.QWidget(self.splitter)
        self.layoutWidget.setObjectName(_fromUtf8("layoutWidget"))
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setMargin(0)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.bookmarkLabel = QtWidgets.QLabel(self.layoutWidget)
        self.bookmarkLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.bookmarkLabel.setText('BOOKMARK')
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setWeight(75)
        font.setBold(True)
        self.bookmarkLabel.setFont(font)
        self.verticalLayout.addWidget(self.bookmarkLabel)

        self.bookmarkList = BookmarkList()
        self.verticalLayout.addWidget(self.bookmarkList)

        self.gridLayout.addWidget(self.splitter, 0, 1, 1, 1)

        self.retranslateUi(Form)
        self.menuTab.setCurrentIndex(0)

        QtCore.QMetaObject.connectSlotsByName(Form)
        #self.splitter.setSizes([100, 1100, 180])

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Form", None))
        self.menuTab.setTabText(self.menuTab.indexOf(self.tab), _translate("Form", "Category", None))
        self.matchAllRadio.setText(_translate("Form", "Match All", None))
        self.matchAnyRadio.setText(_translate("Form", "Match Any", None))
        self.searchButton.setText(_translate("Form", "Search", None))
        self.uploadTimeCheck.setText(_translate("Form", "Upload Time", None))
        self.dateLabel.setText(_translate("Form", "~", None))
        self.menuTab.setTabText(self.menuTab.indexOf(self.tab_2), _translate("Form", "Search", None))
        self.clickToPlayButton.setText(_translate("Form", "Clicked to play", None))
        self.mouseOverToPlayButton.setText(
            _translate("Form", "Mouse over to play", None))
        self.autoReplayCheck.setText(_translate("Form", "Auto Repeat", None))
        self.itemPerClickLabel.setText(_translate("Form", "Show items per click", None))
        self.configLabel.setText(_translate("Form", "Config Data from User : ", None))
        self.playSpeedLabel.setText(_translate("Form", "Play Speed", None))
        self.menuTab.setTabText(self.menuTab.indexOf(self.tab_4),
                                _translate("Form", "Config", None))

