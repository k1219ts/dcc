# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'bakeRolling.ui'
#
# Created: Thu Aug 30 18:53:50 2018
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

# for changing ui
from pymodule.Qt import QtCore
from pymodule.Qt import QtWidgets as QtGui
import pymodule.Qt as Qt

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

class Ui_bkr_win(object):
    def setupUi(self, bkr_win):
        bkr_win.setObjectName(_fromUtf8("bkr_win"))
        bkr_win.resize(426, 413)
        self.verticalLayout = QtGui.QVBoxLayout(bkr_win)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.bkr_grpbox_3 = QtGui.QGroupBox(bkr_win)
        self.bkr_grpbox_3.setObjectName(_fromUtf8("bkr_grpbox_3"))
        self.horizontalLayout_4 = QtGui.QHBoxLayout(self.bkr_grpbox_3)
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.bkr_selection_combox = QtGui.QComboBox(self.bkr_grpbox_3)
        self.bkr_selection_combox.setObjectName(_fromUtf8("bkr_selection_combox"))
        self.horizontalLayout_4.addWidget(self.bkr_selection_combox)
        self.bkr_update_btn = QtGui.QPushButton(self.bkr_grpbox_3)
        self.bkr_update_btn.setObjectName(_fromUtf8("bkr_update_btn"))
        self.horizontalLayout_4.addWidget(self.bkr_update_btn)
        self.bkr_pickSelected_btn = QtGui.QPushButton(self.bkr_grpbox_3)
        self.bkr_pickSelected_btn.setObjectName(_fromUtf8("bkr_pickSelected_btn"))
        self.horizontalLayout_4.addWidget(self.bkr_pickSelected_btn)
        self.bkr_remove_btn = QtGui.QPushButton(self.bkr_grpbox_3)
        self.bkr_remove_btn.setObjectName(_fromUtf8("bkr_remove_btn"))
        self.horizontalLayout_4.addWidget(self.bkr_remove_btn)
        self.horizontalLayout_4.setStretch(0, 1)
        self.verticalLayout.addWidget(self.bkr_grpbox_3)
        self.bkr_grpbox_1 = QtGui.QGroupBox(bkr_win)
        self.bkr_grpbox_1.setObjectName(_fromUtf8("bkr_grpbox_1"))
        self.horizontalLayout_3 = QtGui.QHBoxLayout(self.bkr_grpbox_1)
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.bkr_groundCtr_add_btn = QtGui.QPushButton(self.bkr_grpbox_1)
        self.bkr_groundCtr_add_btn.setMinimumSize(QtCore.QSize(0, 0))
        self.bkr_groundCtr_add_btn.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.bkr_groundCtr_add_btn.setObjectName(_fromUtf8("bkr_groundCtr_add_btn"))
        self.verticalLayout_2.addWidget(self.bkr_groundCtr_add_btn)
        self.bkr_fitCustomCtrAxis_btn = QtGui.QPushButton(self.bkr_grpbox_1)
        self.bkr_fitCustomCtrAxis_btn.setObjectName(_fromUtf8("bkr_fitCustomCtrAxis_btn"))
        self.verticalLayout_2.addWidget(self.bkr_fitCustomCtrAxis_btn)
        self.horizontalLayout_3.addLayout(self.verticalLayout_2)
        self.bkr_prebake_btn = QtGui.QPushButton(self.bkr_grpbox_1)
        self.bkr_prebake_btn.setMinimumSize(QtCore.QSize(100, 50))
        self.bkr_prebake_btn.setObjectName(_fromUtf8("bkr_prebake_btn"))
        self.horizontalLayout_3.addWidget(self.bkr_prebake_btn)
        self.bkr_prebakeRemove_btn = QtGui.QPushButton(self.bkr_grpbox_1)
        self.bkr_prebakeRemove_btn.setMinimumSize(QtCore.QSize(0, 50))
        self.bkr_prebakeRemove_btn.setObjectName(_fromUtf8("bkr_prebakeRemove_btn"))
        self.horizontalLayout_3.addWidget(self.bkr_prebakeRemove_btn)
        self.horizontalLayout_3.setStretch(1, 1)
        self.verticalLayout.addWidget(self.bkr_grpbox_1)
        self.bkr_grpbox_2 = QtGui.QGroupBox(bkr_win)
        self.bkr_grpbox_2.setObjectName(_fromUtf8("bkr_grpbox_2"))
        self.horizontalLayout_5 = QtGui.QHBoxLayout(self.bkr_grpbox_2)
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        self.bkr_finalBake_btn = QtGui.QPushButton(self.bkr_grpbox_2)
        self.bkr_finalBake_btn.setMinimumSize(QtCore.QSize(100, 50))
        self.bkr_finalBake_btn.setObjectName(_fromUtf8("bkr_finalBake_btn"))
        self.horizontalLayout_5.addWidget(self.bkr_finalBake_btn)
        self.bkr_finalbakeRemove_btn = QtGui.QPushButton(self.bkr_grpbox_2)
        self.bkr_finalbakeRemove_btn.setMinimumSize(QtCore.QSize(0, 50))
        self.bkr_finalbakeRemove_btn.setObjectName(_fromUtf8("bkr_finalbakeRemove_btn"))
        self.horizontalLayout_5.addWidget(self.bkr_finalbakeRemove_btn)
        self.horizontalLayout_5.setStretch(0, 1)
        self.verticalLayout.addWidget(self.bkr_grpbox_2)
        self.bkr_grpbox_4 = QtGui.QGroupBox(bkr_win)
        self.bkr_grpbox_4.setObjectName(_fromUtf8("bkr_grpbox_4"))
        self.verticalLayout_4 = QtGui.QVBoxLayout(self.bkr_grpbox_4)
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.bkr_layout_6 = QtGui.QHBoxLayout()
        self.bkr_layout_6.setObjectName(_fromUtf8("bkr_layout_6"))
        self.bkr_addTag_btn = QtGui.QPushButton(self.bkr_grpbox_4)
        self.bkr_addTag_btn.setObjectName(_fromUtf8("bkr_addTag_btn"))
        self.bkr_layout_6.addWidget(self.bkr_addTag_btn)
        self.bkr_removeTag_btn = QtGui.QPushButton(self.bkr_grpbox_4)
        self.bkr_removeTag_btn.setMinimumSize(QtCore.QSize(100, 0))
        self.bkr_removeTag_btn.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.bkr_removeTag_btn.setObjectName(_fromUtf8("bkr_removeTag_btn"))
        self.bkr_layout_6.addWidget(self.bkr_removeTag_btn)
        self.bkr_layout_6.setStretch(0, 1)
        self.verticalLayout_4.addLayout(self.bkr_layout_6)
        self.bkr_layout_7 = QtGui.QHBoxLayout()
        self.bkr_layout_7.setObjectName(_fromUtf8("bkr_layout_7"))
        self.bkr_selectTag_btn = QtGui.QPushButton(self.bkr_grpbox_4)
        self.bkr_selectTag_btn.setObjectName(_fromUtf8("bkr_selectTag_btn"))
        self.bkr_layout_7.addWidget(self.bkr_selectTag_btn)
        self.bkr_setRadiusTag_btn = QtGui.QPushButton(self.bkr_grpbox_4)
        self.bkr_setRadiusTag_btn.setMinimumSize(QtCore.QSize(100, 0))
        self.bkr_setRadiusTag_btn.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.bkr_setRadiusTag_btn.setObjectName(_fromUtf8("bkr_setRadiusTag_btn"))
        self.bkr_layout_7.addWidget(self.bkr_setRadiusTag_btn)
        self.bkr_connectScale_btn = QtGui.QPushButton(self.bkr_grpbox_4)
        self.bkr_connectScale_btn.setObjectName(_fromUtf8("bkr_connectScale_btn"))
        self.bkr_layout_7.addWidget(self.bkr_connectScale_btn)
        self.bkr_layout_7.setStretch(0, 1)
        self.verticalLayout_4.addLayout(self.bkr_layout_7)
        self.verticalLayout.addWidget(self.bkr_grpbox_4)
        spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)

        self.retranslateUi(bkr_win)
        QtCore.QMetaObject.connectSlotsByName(bkr_win)

    def retranslateUi(self, bkr_win):
        bkr_win.setWindowTitle(_translate("bkr_win", "Dx_bakeRolling", None))
        self.bkr_grpbox_3.setTitle(_translate("bkr_win", "Selection", None))
        self.bkr_update_btn.setText(_translate("bkr_win", "Update", None))
        self.bkr_pickSelected_btn.setText(_translate("bkr_win", ">", None))
        self.bkr_remove_btn.setText(_translate("bkr_win", "Remove", None))
        self.bkr_grpbox_1.setTitle(_translate("bkr_win", "Pre-baking", None))
        self.bkr_groundCtr_add_btn.setText(_translate("bkr_win", "Add Ground Ctr.", None))
        self.bkr_fitCustomCtrAxis_btn.setText(_translate("bkr_win", "Fit Custom Ctr Axis", None))
        self.bkr_prebake_btn.setText(_translate("bkr_win", "Pre-bake", None))
        self.bkr_prebakeRemove_btn.setText(_translate("bkr_win", "Remove", None))
        self.bkr_grpbox_2.setTitle(_translate("bkr_win", "Final-baking", None))
        self.bkr_finalBake_btn.setText(_translate("bkr_win", "Final Bake", None))
        self.bkr_finalbakeRemove_btn.setText(_translate("bkr_win", "Remove", None))
        self.bkr_grpbox_4.setTitle(_translate("bkr_win", "Rigging", None))
        self.bkr_addTag_btn.setText(_translate("bkr_win", "Add tag for bakeRolling", None))
        self.bkr_removeTag_btn.setText(_translate("bkr_win", "Remove", None))
        self.bkr_selectTag_btn.setText(_translate("bkr_win", "Select Tagged Node", None))
        self.bkr_setRadiusTag_btn.setText(_translate("bkr_win", "Set Radius", None))
        self.bkr_connectScale_btn.setText(_translate("bkr_win", "Connect Scale", None))
