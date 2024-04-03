# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Resources/MainFormUI.ui'
#
# Created: Tue Nov 12 14:53:50 2019
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

# from PyQt4 import QtCore, QtGui
from PySide2 import QtWidgets
from PySide2 import QtCore
from PySide2 import QtGui

from customWidget.CategoryTreeWidget import categoryTreeWidget
from customWidget.Item import ItemListWidget
from customWidget.Bookmark import BookmarkListWidget

import getpass

import os

CURRENT_DIR = os.path.dirname(__file__)

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
        Form.setWindowModality(QtCore.Qt.NonModal)
        Form.resize(1920, 1080)
        Form.setWindowOpacity(1.0)
        Form.setStyleSheet(_fromUtf8("background-color:rgb(72, 72, 72)"))
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.titleLabel = QtWidgets.QLabel(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.titleLabel.sizePolicy().hasHeightForWidth())
        self.titleLabel.setSizePolicy(sizePolicy)
        self.titleLabel.setMaximumSize(QtCore.QSize(16777215, 40))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.titleLabel.setFont(font)
        self.titleLabel.setObjectName(_fromUtf8("titleLabel"))
        self.horizontalLayout_2.addWidget(self.titleLabel)
        spacerItem = QtWidgets.QSpacerItem(40, 40, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.userNameLabel = QtWidgets.QLabel(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.userNameLabel.sizePolicy().hasHeightForWidth())
        self.userNameLabel.setSizePolicy(sizePolicy)
        self.userNameLabel.setMaximumSize(QtCore.QSize(150, 40))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.userNameLabel.setFont(font)
        self.userNameLabel.setStyleSheet(_fromUtf8("color:white; font:bold 20px;"))
        self.userNameLabel.setObjectName(_fromUtf8("userNameLabel"))
        self.horizontalLayout_2.addWidget(self.userNameLabel)
        self.searchEdit = QtWidgets.QLineEdit(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.searchEdit.sizePolicy().hasHeightForWidth())
        self.searchEdit.setSizePolicy(sizePolicy)
        self.searchEdit.setMaximumSize(QtCore.QSize(200, 40))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.searchEdit.setFont(font)
        self.searchEdit.setStyleSheet(_fromUtf8("color:white; background-color:#383838"))
        self.searchEdit.setText(_fromUtf8(""))
        self.searchEdit.setObjectName(_fromUtf8("searchEdit"))
        self.horizontalLayout_2.addWidget(self.searchEdit)
        # self.helpBtn = QtWidgets.QPushButton(Form)
        # sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        # sizePolicy.setHorizontalStretch(0)
        # sizePolicy.setVerticalStretch(0)
        # sizePolicy.setHeightForWidth(self.helpBtn.sizePolicy().hasHeightForWidth())
        # self.helpBtn.setSizePolicy(sizePolicy)
        # self.helpBtn.setMinimumSize(QtCore.QSize(48, 48))
        # self.helpBtn.setMaximumSize(QtCore.QSize(48, 48))
        # self.helpBtn.setText(_fromUtf8(""))
        # self.helpBtn.setObjectName(_fromUtf8("helpBtn"))
        # self.horizontalLayout_2.addWidget(self.helpBtn)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.line = QtWidgets.QFrame(Form)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName(_fromUtf8("line"))
        self.verticalLayout_2.addWidget(self.line)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.widget = QtWidgets.QWidget(Form)
        self.widget.setMaximumSize(QtCore.QSize(200, 16777215))
        self.widget.setObjectName(_fromUtf8("widget"))
        self.gridLayout_2 = QtWidgets.QGridLayout(self.widget)
        self.gridLayout_2.setMargin(0)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.categoryWidget = categoryTreeWidget(self.widget)
        self.categoryWidget.setMaximumSize(QtCore.QSize(300, 16777215))
        self.categoryWidget.setStyleSheet(_fromUtf8("QTreeView {color: #C2C2C2; font: bold 15px; width: 100px; background-color: #383838; border: 1px solid #3d3d3d;}\n"
"QTreeView::item:selected {background-color: #53728e;color: #C2C2C2;}\n"
"QTreeView::item:hover {border: 1px solid #53728e;}\n"
"QTreeView:verticalScrollBar {background-color: #FFA91D;alternate-background-color: #FF0000;}\n"
""))
        self.categoryWidget.setHeaderHidden(True)
        self.categoryWidget.setObjectName(_fromUtf8("categoryWidget"))
        #self.categoryWidget.headerItem().setText(0, _fromUtf8("1"))
        #item_0 = QtWidgets.QTreeWidgetItem(self.categoryWidget)
        self.gridLayout_2.addWidget(self.categoryWidget, 0, 0, 1, 1)
        self.line_2 = QtWidgets.QFrame(self.widget)
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName(_fromUtf8("line_2"))
        self.gridLayout_2.addWidget(self.line_2, 0, 1, 1, 1)
        self.horizontalLayout.addWidget(self.widget)
        self.widget_3 = QtWidgets.QWidget(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_3.sizePolicy().hasHeightForWidth())
        self.widget_3.setSizePolicy(sizePolicy)
        self.widget_3.setObjectName(_fromUtf8("widget_3"))
        self.gridLayout = QtWidgets.QGridLayout(self.widget_3)
        self.gridLayout.setMargin(0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        self.widget_4 = QtWidgets.QWidget(self.widget_3)
        self.widget_4.setMinimumSize(QtCore.QSize(100, 0))
        self.widget_4.setObjectName(_fromUtf8("widget_4"))
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.widget_4)
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setMargin(0)
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))

#         self.listModeRadioBtn = QtWidgets.QRadioButton(self.widget_4)
#         self.listModeRadioBtn.setMinimumSize(QtCore.QSize(40, 40))
#         self.listModeRadioBtn.setMaximumSize(QtCore.QSize(40, 40))
#         self.listModeRadioBtn.setStyleSheet(_fromUtf8("QRadioButton::indicator{width:40px; height:40px}\n"
# "QRadioButton::indicator:unchecked{image: url(%s/Resources/icon/unchecked_listMode.png)}\n"
# "QRadioButton::indicator:checked{image: url(%s/Resources/icon/checked_listMode.png)}" % (CURRENT_DIR, CURRENT_DIR)))
#         self.listModeRadioBtn.setText(_fromUtf8(""))
#         self.listModeRadioBtn.setIconSize(QtCore.QSize(24, 24))
#         self.listModeRadioBtn.setObjectName(_fromUtf8("listModeRadioBtn"))
#         self.horizontalLayout_4.addWidget(self.listModeRadioBtn)
#         self.iconModeRadioBtn = QtWidgets.QRadioButton(self.widget_4)
#         self.iconModeRadioBtn.setMinimumSize(QtCore.QSize(40, 40))
#         self.iconModeRadioBtn.setMaximumSize(QtCore.QSize(40, 40))
#         self.iconModeRadioBtn.setStyleSheet(_fromUtf8("QRadioButton::indicator{width:40px; height:40px}\n"
# "QRadioButton::indicator:unchecked{image: url(%s/Resources/icon/unchecked_iconMode.png)}\n"
# "QRadioButton::indicator:checked{image: url(%s/Resources/icon/checked_iconMode.png)}" % (CURRENT_DIR, CURRENT_DIR)))
#         self.iconModeRadioBtn.setText(_fromUtf8(""))
#         self.iconModeRadioBtn.setChecked(True)
#         self.iconModeRadioBtn.setObjectName(_fromUtf8("iconModeRadioBtn"))
#         self.horizontalLayout_4.addWidget(self.iconModeRadioBtn)
#         self.horizontalLayout_3.addWidget(self.widget_4)
        self.gridLayout.addLayout(self.horizontalLayout_3, 0, 0, 1, 2)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem2, 3, 0, 1, 1)
        self.itemTreeWidget = ItemListWidget(self.widget_3)
        self.itemTreeWidget.setStyleSheet(_fromUtf8("QListView {color: #C2C2C2; font: bold 15px; width: 100px; background-color: #383838; border: 1px solid #3d3d3d;}\n"
"QListView::item:selected {background-color: #53728e;color: #C2C2C2;}\n"
"QListView::item:hover {border: 1px solid #53728e;}\n"
"QListView:verticalScrollBar {background-color: #FFA91D;alternate-background-color: #FF0000;}"))
        self.itemTreeWidget.setObjectName(_fromUtf8("itemTreeWidget"))
        self.gridLayout.addWidget(self.itemTreeWidget, 1, 1, 3, 1)
        self.line_3 = QtWidgets.QFrame(self.widget_3)
        self.line_3.setMinimumSize(QtCore.QSize(10, 0))
        self.line_3.setBaseSize(QtCore.QSize(20, 0))
        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName(_fromUtf8("line_3"))
        self.gridLayout.addWidget(self.line_3, 0, 2, 4, 1)
#         self.widget_5 = QtWidgets.QWidget(self.widget_3)
#         self.widget_5.setMinimumSize(QtCore.QSize(0, 0))
#         self.widget_5.setObjectName(_fromUtf8("widget_5"))
#         self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.widget_5)
#         self.verticalLayout_3.setSpacing(0)
#         self.verticalLayout_3.setContentsMargins(4, 0, 4, 0)
#         self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
#         self.importRadioBtn = QtWidgets.QRadioButton(self.widget_5)
#         self.importRadioBtn.setMinimumSize(QtCore.QSize(40, 40))
#         self.importRadioBtn.setMaximumSize(QtCore.QSize(40, 40))
#         self.importRadioBtn.setStyleSheet(_fromUtf8("QRadioButton::indicator{width:40px; height:40px}\n"
# "QRadioButton::indicator:unchecked{image: url(%s/Resources/icon/unchecked_iconMode.png)}\n"
# "QRadioButton::indicator:checked{image: url(%s/Resources/icon/checked_iconMode.png)}" % (CURRENT_DIR, CURRENT_DIR)))
#         self.importRadioBtn.setText(_fromUtf8(""))
#         self.importRadioBtn.setChecked(True)
#         self.importRadioBtn.setObjectName(_fromUtf8("importRadioBtn"))
#         self.verticalLayout_3.addWidget(self.importRadioBtn)
#         self.mergeRadioBtn = QtWidgets.QRadioButton(self.widget_5)
#         self.mergeRadioBtn.setMinimumSize(QtCore.QSize(40, 40))
#         self.mergeRadioBtn.setMaximumSize(QtCore.QSize(40, 40))
#         self.mergeRadioBtn.setStyleSheet(_fromUtf8("QRadioButton::indicator{width:40px; height:40px}\n"
# "QRadioButton::indicator:unchecked{image: url(%s/Resources/icon/unchecked_iconMode.png)}\n"
# "QRadioButton::indicator:checked{image: url(%s/Resources/icon/checked_iconMode.png)}" % (CURRENT_DIR, CURRENT_DIR)))
#         self.mergeRadioBtn.setText(_fromUtf8(""))
#         self.mergeRadioBtn.setObjectName(_fromUtf8("mergeRadioBtn"))
#         self.verticalLayout_3.addWidget(self.mergeRadioBtn)
#         self.replaceRadioBtn = QtWidgets.QRadioButton(self.widget_5)
#         self.replaceRadioBtn.setMinimumSize(QtCore.QSize(40, 40))
#         self.replaceRadioBtn.setMaximumSize(QtCore.QSize(40, 40))
#         self.replaceRadioBtn.setStyleSheet(_fromUtf8("QRadioButton::indicator{width:40px; height:40px}\n"
# "QRadioButton::indicator:unchecked{image: url(%s/Resources/icon/unchecked_iconMode.png)}\n"
# "QRadioButton::indicator:checked{image: url(%s/Resources/icon/checked_iconMode.png)}" % (CURRENT_DIR, CURRENT_DIR)))
#         self.replaceRadioBtn.setText(_fromUtf8(""))
#         self.replaceRadioBtn.setObjectName(_fromUtf8("replaceRadioBtn"))
#         self.verticalLayout_3.addWidget(self.replaceRadioBtn)
#         self.gridLayout.addWidget(self.widget_5, 2, 0, 1, 1)
        self.horizontalLayout.addWidget(self.widget_3)

        self.widget_2 = QtWidgets.QWidget(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_2.sizePolicy().hasHeightForWidth())
        self.widget_2.setSizePolicy(sizePolicy)
        self.widget_2.setMaximumSize(QtCore.QSize(320, 16777215))
        self.widget_2.setObjectName(_fromUtf8("widget_2"))
        # self.gridLayout_4 = QtWidgets.QGridLayout(self.widget_2)
        # self.gridLayout_4.setMargin(0)
        # self.gridLayout_4.setObjectName(_fromUtf8("gridLayout_4"))
        self.gridLayout_3 = QtWidgets.QGridLayout(self.widget_2)
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.gridLayout_3.setSpacing(0)
        self.gridLayout_3.setMargin(0)

        self.listWidget = BookmarkListWidget()
        self.listWidget.setStyleSheet(_fromUtf8(
            "color: #C2C2C2; font: bold 10px; width: 100px; background-color: #383838; border: 1px solid #3d3d3d;}\n"
            "QTreeView::item:selected {background-color: #53728e;color: #C2C2C2;}\n"
            "QTreeView::item:hover {border: 1px solid #53728e;}\n"
            "QTreeView:verticalScrollBar {background-color: #FFA91D;alternate-background-color: #FF0000;"))
        self.listWidget.setObjectName(_fromUtf8("listWidget"))
        self.gridLayout_3.addWidget(self.listWidget, 3, 0, 1, 1)

        self.previewLabel = QtWidgets.QLabel(self.widget_2)
        self.previewLabel.setMinimumSize(QtCore.QSize(320, 240))
        self.previewLabel.setMaximumSize(QtCore.QSize(320, 240))
        self.previewLabel.setObjectName(_fromUtf8("previewLabel"))
        self.gridLayout_3.addWidget(self.previewLabel, 0, 0, 1, 1)
        # self.verticalLayout.addWidget(self.previewLabel)

        self.commentLabel = QtWidgets.QLabel(self.widget_2)
        self.commentLabel.setMinimumSize(QtCore.QSize(320, 120))
        self.commentLabel.setMaximumSize(QtCore.QSize(320, 300))
        self.commentLabel.setStyleSheet(_fromUtf8("background-color:#383838; color:#FFFFFF;"))
        # self.commentLabel.setStyleSheet(_fromUtf8("background-color:#383838; color:#FFFFFF; font:bold 15px;"))
        self.commentLabel.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.commentLabel.setObjectName(_fromUtf8("commentLabel"))
        self.gridLayout_3.addWidget(self.commentLabel, 1, 0, 1, 1)


        # self.commentLabel2 = QtWidgets.QLabel(self.widget_2)
        # self.commentLabel2.setMinimumSize(QtCore.QSize(320, 10))
        # self.commentLabel2.setMaximumSize(QtCore.QSize(320, 50))
        # self.commentLabel2.setStyleSheet(_fromUtf8("background-color:#383838; color:#FFFFFF;"))
        # # self.commentLabel2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        # self.commentLabel2.setObjectName(_fromUtf8("commentLabel2"))
        # self.gridLayout_3.addWidget(self.commentLabel2, 2, 0, 1, 1)


        self.label = QtWidgets.QLabel(self.widget_2)
        # self.label.setMinimumSize(QtCore.QSize(320, 300))
        # self.label.setMaximumSize(QtCore.QSize(320, 500))
        self.label.setStyleSheet(_fromUtf8("background-color:#383838; color:#FFFFFF;"))
        self.label.setObjectName(_fromUtf8("label"))

        self.gridLayout_3.addWidget(self.label, 2, 0, 1, 1)
        # self.gridLayout_4.addLayout(self.gridLayout_3, 0, 0, 1, 1)

        self.horizontalLayout.addWidget(self.widget_2)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.statusLabel = QtWidgets.QLabel(Form)
        self.statusLabel.setStyleSheet(_fromUtf8("color:white; font:10px;"))
        self.statusLabel.setObjectName(_fromUtf8("statusLabel"))

        self.itemTreeWidget.itemConnect(self.statusLabel,self.previewLabel,self.commentLabel,self.categoryWidget)
        self.listWidget.itemConnect(self.statusLabel, self.previewLabel, self.commentLabel,self.categoryWidget)

        self.verticalLayout_2.addWidget(self.statusLabel)
        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Form", None))
        self.titleLabel.setText(_translate("Form", "<img src=\'%s/Resources/USDattr.png\' width=\'45\' height=\'45\' align=\'left\'  <p><font color=\'#1F91D0\'><font size=\'5\'> ASSETLIB BROWSER</p>" % CURRENT_DIR, None))
        userName = getpass.getuser()
        self.userNameLabel.setText(_translate("Form", userName, None))
        __sortingEnabled = self.categoryWidget.isSortingEnabled()
        self.categoryWidget.setSortingEnabled(__sortingEnabled)

        # self.itemTreeWidget.headerItem().setText(0, _translate("Form", "Name", None))
        # self.itemTreeWidget.headerItem().setText(1, _translate("Form", "Date modified", None))
        # self.itemTreeWidget.headerItem().setText(2, _translate("Form", "Type", None))
        # self.itemTreeWidget.headerItem().setText(3, _translate("Form", "Location", None))
        # self.itemTreeWidget.headerItem().setText(4, _translate("Form", "Size", None))
        self.previewLabel.setText(_translate("Form", "Preview", None))
        self.commentLabel.setText(_translate("Form", "File Summary", None))
        self.statusLabel.setText(_translate("Form", "progress....", None))
        self.label.setText(_translate("Form", "Bookmark", None))

