# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'shotsub_anim_widget.ui'
#
# Created: Sat Apr 29 19:03:03 2017
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

#from PyQt4 import QtCore, QtGui
from PySide2 import QtCore, QtGui, QtWidgets

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

class Ui_shotsub_anim_Dialog(object):
    def setupUi(self, shotsub_anim_Dialog):
        shotsub_anim_Dialog.setObjectName(_fromUtf8("shotsub_anim_Dialog"))
        shotsub_anim_Dialog.resize(464, 531)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(shotsub_anim_Dialog)
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.header_layout = QtWidgets.QHBoxLayout()
        self.header_layout.setObjectName(_fromUtf8("header_layout"))
        self.header = QtWidgets.QLabel(shotsub_anim_Dialog)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.header.setFont(font)
        self.header.setObjectName(_fromUtf8("header"))
        self.header_layout.addWidget(self.header)
        self.headerVer = QtWidgets.QLabel(shotsub_anim_Dialog)
        self.headerVer.setObjectName(_fromUtf8("headerVer"))
        self.header_layout.addWidget(self.headerVer)
        spacerItem = QtWidgets.QSpacerItem(40, 20,
                                           QtWidgets.QSizePolicy.Expanding,
                                           QtWidgets.QSizePolicy.Minimum)
        self.header_layout.addItem(spacerItem)
        self.verticalLayout_3.addLayout(self.header_layout)
        self.importFrom_layout = QtWidgets.QHBoxLayout()
        self.importFrom_layout.setObjectName(_fromUtf8("importFrom_layout"))
        self.importFrom = QtWidgets.QLabel(shotsub_anim_Dialog)
        self.importFrom.setMinimumSize(QtCore.QSize(100, 0))
        self.importFrom.setMaximumSize(QtCore.QSize(100, 16777215))
        self.importFrom.setObjectName(_fromUtf8("importFrom"))
        self.importFrom_layout.addWidget(self.importFrom)
        self.showUI = QtWidgets.QComboBox(shotsub_anim_Dialog)
        self.showUI.setMinimumSize(QtCore.QSize(100, 0))
        self.showUI.setObjectName(_fromUtf8("showUI"))
        self.showUI.addItem(_fromUtf8(""))
        self.importFrom_layout.addWidget(self.showUI)
        self.seqUI = QtWidgets.QComboBox(shotsub_anim_Dialog)
        self.seqUI.setMinimumSize(QtCore.QSize(100, 0))
        self.seqUI.setObjectName(_fromUtf8("seqUI"))
        self.seqUI.addItem(_fromUtf8(""))
        self.importFrom_layout.addWidget(self.seqUI)
        self.shotUI = QtWidgets.QComboBox(shotsub_anim_Dialog)
        self.shotUI.setMinimumSize(QtCore.QSize(120, 0))
        self.shotUI.setObjectName(_fromUtf8("shotUI"))
        self.shotUI.addItem(_fromUtf8(""))
        self.importFrom_layout.addWidget(self.shotUI)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20,
                                            QtWidgets.QSizePolicy.Expanding,
                                            QtWidgets.QSizePolicy.Minimum)
        self.importFrom_layout.addItem(spacerItem1)
        self.verticalLayout_3.addLayout(self.importFrom_layout)
        self.plateType_layout = QtWidgets.QHBoxLayout()
        self.plateType_layout.setObjectName(_fromUtf8("plateType_layout"))
        self.plateType = QtWidgets.QLabel(shotsub_anim_Dialog)
        self.plateType.setMinimumSize(QtCore.QSize(100, 0))
        self.plateType.setMaximumSize(QtCore.QSize(100, 16777215))
        self.plateType.setObjectName(_fromUtf8("plateType"))
        self.plateType_layout.addWidget(self.plateType)
        self.plateTypeUI = QtWidgets.QComboBox(shotsub_anim_Dialog)
        self.plateTypeUI.setMinimumSize(QtCore.QSize(100, 0))
        self.plateTypeUI.setObjectName(_fromUtf8("plateTypeUI"))
        self.plateTypeUI.addItem(_fromUtf8(""))
        self.plateType_layout.addWidget(self.plateTypeUI)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20,
                                            QtWidgets.QSizePolicy.Expanding,
                                            QtWidgets.QSizePolicy.Minimum)
        self.plateType_layout.addItem(spacerItem2)
        self.verticalLayout_3.addLayout(self.plateType_layout)
        self.scene_layout = QtWidgets.QHBoxLayout()
        self.scene_layout.setObjectName(_fromUtf8("scene_layout"))
        self.scene = QtWidgets.QLabel(shotsub_anim_Dialog)
        self.scene.setMinimumSize(QtCore.QSize(100, 0))
        self.scene.setMaximumSize(QtCore.QSize(100, 16777215))
        self.scene.setObjectName(_fromUtf8("scene"))
        self.scene_layout.addWidget(self.scene)
        self.sceneUI = QtWidgets.QComboBox(shotsub_anim_Dialog)
        self.sceneUI.setObjectName(_fromUtf8("sceneUI"))
        self.sceneUI.addItem(_fromUtf8(""))
        self.scene_layout.addWidget(self.sceneUI)
        self.verticalLayout_3.addLayout(self.scene_layout)
        self.sceneInfo = QtWidgets.QGroupBox(shotsub_anim_Dialog)
        self.sceneInfo.setObjectName(_fromUtf8("sceneInfo"))
        self.verticalLayout = QtWidgets.QVBoxLayout(self.sceneInfo)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.sceneInfo_layout = QtWidgets.QGridLayout()
        self.sceneInfo_layout.setObjectName(_fromUtf8("sceneInfo_layout"))
        self.publisher = QtWidgets.QLabel(self.sceneInfo)
        self.publisher.setMinimumSize(QtCore.QSize(80, 0))
        self.publisher.setObjectName(_fromUtf8("publisher"))
        self.sceneInfo_layout.addWidget(self.publisher, 0, 0, 1, 1)
        self.publisherUI = QtWidgets.QLineEdit(self.sceneInfo)
        self.publisherUI.setEnabled(True)
        self.publisherUI.setReadOnly(True)
        self.publisherUI.setObjectName(_fromUtf8("publisherUI"))
        self.sceneInfo_layout.addWidget(self.publisherUI, 0, 1, 1, 1)
        self.ver = QtWidgets.QLabel(self.sceneInfo)
        self.ver.setObjectName(_fromUtf8("ver"))
        self.sceneInfo_layout.addWidget(self.ver, 1, 0, 1, 1)
        self.verUI = QtWidgets.QLineEdit(self.sceneInfo)
        self.verUI.setEnabled(True)
        self.verUI.setReadOnly(True)
        self.verUI.setObjectName(_fromUtf8("verUI"))
        self.sceneInfo_layout.addWidget(self.verUI, 1, 1, 1, 1)
        self.date = QtWidgets.QLabel(self.sceneInfo)
        self.date.setObjectName(_fromUtf8("date"))
        self.sceneInfo_layout.addWidget(self.date, 2, 0, 1, 1)
        self.dateUI = QtWidgets.QLineEdit(self.sceneInfo)
        self.dateUI.setEnabled(True)
        self.dateUI.setReadOnly(True)
        self.dateUI.setObjectName(_fromUtf8("dateUI"))
        self.sceneInfo_layout.addWidget(self.dateUI, 2, 1, 1, 1)
        self.comment = QtWidgets.QLabel(self.sceneInfo)
        self.comment.setObjectName(_fromUtf8("comment"))
        self.sceneInfo_layout.addWidget(self.comment, 3, 0, 1, 1)
        self.commentUI = QtWidgets.QLineEdit(self.sceneInfo)
        self.commentUI.setEnabled(True)
        self.commentUI.setReadOnly(True)
        self.commentUI.setObjectName(_fromUtf8("commentUI"))
        self.sceneInfo_layout.addWidget(self.commentUI, 3, 1, 1, 1)
        self.verticalLayout.addLayout(self.sceneInfo_layout)
        self.sceneInfo_layout_2 = QtWidgets.QGridLayout()
        self.sceneInfo_layout_2.setObjectName(_fromUtf8("sceneInfo_layout_2"))
        self.frameRange = QtWidgets.QLabel(self.sceneInfo)
        self.frameRange.setMinimumSize(QtCore.QSize(80, 0))
        self.frameRange.setObjectName(_fromUtf8("frameRange"))
        self.sceneInfo_layout_2.addWidget(self.frameRange, 0, 0, 1, 1)
        self.startFrameUI = QtWidgets.QLineEdit(self.sceneInfo)
        self.startFrameUI.setEnabled(True)
        self.startFrameUI.setMinimumSize(QtCore.QSize(70, 0))
        self.startFrameUI.setMaximumSize(QtCore.QSize(70, 16777215))
        self.startFrameUI.setReadOnly(True)
        self.startFrameUI.setObjectName(_fromUtf8("startFrameUI"))
        self.sceneInfo_layout_2.addWidget(self.startFrameUI, 0, 1, 1, 1)
        self.endFrameUI = QtWidgets.QLineEdit(self.sceneInfo)
        self.endFrameUI.setEnabled(True)
        self.endFrameUI.setMinimumSize(QtCore.QSize(70, 0))
        self.endFrameUI.setMaximumSize(QtCore.QSize(70, 16777215))
        self.endFrameUI.setReadOnly(True)
        self.endFrameUI.setObjectName(_fromUtf8("endFrameUI"))
        self.sceneInfo_layout_2.addWidget(self.endFrameUI, 0, 2, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20,
                                            QtWidgets.QSizePolicy.Expanding,
                                            QtWidgets.QSizePolicy.Minimum)
        self.sceneInfo_layout_2.addItem(spacerItem3, 0, 3, 1, 1)
        self.renderRes = QtWidgets.QLabel(self.sceneInfo)
        self.renderRes.setMinimumSize(QtCore.QSize(0, 0))
        self.renderRes.setObjectName(_fromUtf8("renderRes"))
        self.sceneInfo_layout_2.addWidget(self.renderRes, 1, 0, 1, 1)
        self.renderWidthUI = QtWidgets.QLineEdit(self.sceneInfo)
        self.renderWidthUI.setEnabled(True)
        self.renderWidthUI.setMinimumSize(QtCore.QSize(70, 0))
        self.renderWidthUI.setMaximumSize(QtCore.QSize(70, 16777215))
        self.renderWidthUI.setReadOnly(True)
        self.renderWidthUI.setObjectName(_fromUtf8("renderWidthUI"))
        self.sceneInfo_layout_2.addWidget(self.renderWidthUI, 1, 1, 1, 1)
        self.renderHeightUI = QtWidgets.QLineEdit(self.sceneInfo)
        self.renderHeightUI.setEnabled(True)
        self.renderHeightUI.setMinimumSize(QtCore.QSize(70, 0))
        self.renderHeightUI.setMaximumSize(QtCore.QSize(70, 16777215))
        self.renderHeightUI.setReadOnly(True)
        self.renderHeightUI.setObjectName(_fromUtf8("renderHeightUI"))
        self.sceneInfo_layout_2.addWidget(self.renderHeightUI, 1, 2, 1, 1)
        self.verticalLayout.addLayout(self.sceneInfo_layout_2)
        self.sceneInfo_layout_3 = QtWidgets.QGridLayout()
        self.sceneInfo_layout_3.setObjectName(_fromUtf8("sceneInfo_layout_3"))
        self.overscanUI = QtWidgets.QCheckBox(self.sceneInfo)
        self.overscanUI.setEnabled(False)
        self.overscanUI.setCheckable(False)
        self.overscanUI.setObjectName(_fromUtf8("overscanUI"))
        self.sceneInfo_layout_3.addWidget(self.overscanUI, 0, 0, 1, 1)
        self.stereoUI = QtWidgets.QCheckBox(self.sceneInfo)
        self.stereoUI.setEnabled(False)
        self.stereoUI.setCheckable(False)
        self.stereoUI.setObjectName(_fromUtf8("stereoUI"))
        self.sceneInfo_layout_3.addWidget(self.stereoUI, 0, 1, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20,
                                            QtWidgets.QSizePolicy.Expanding,
                                            QtWidgets.QSizePolicy.Minimum)
        self.sceneInfo_layout_3.addItem(spacerItem4, 0, 2, 1, 1)
        self.verticalLayout.addLayout(self.sceneInfo_layout_3)
        self.verticalLayout_3.addWidget(self.sceneInfo)
        self.options = QtWidgets.QGroupBox(shotsub_anim_Dialog)
        self.options.setObjectName(_fromUtf8("options"))
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.options)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.options_layout = QtWidgets.QGridLayout()
        self.options_layout.setObjectName(_fromUtf8("options_layout"))
        self.layoutCameraUI = QtWidgets.QCheckBox(self.options)
        self.layoutCameraUI.setCheckable(True)
        self.layoutCameraUI.setChecked(False)
        self.layoutCameraUI.setObjectName(_fromUtf8("layoutCameraUI"))
        self.options_layout.addWidget(self.layoutCameraUI, 0, 0, 1, 1)
        self.verticalLayout_2.addLayout(self.options_layout)
        self.verticalLayout_3.addWidget(self.options)
        self.do_layout = QtWidgets.QHBoxLayout()
        self.do_layout.setObjectName(_fromUtf8("do_layout"))
        spacerItem5 = QtWidgets.QSpacerItem(40, 20,
                                            QtWidgets.QSizePolicy.Expanding,
                                            QtWidgets.QSizePolicy.Minimum)
        self.do_layout.addItem(spacerItem5)
        self.doIt = QtWidgets.QPushButton(shotsub_anim_Dialog)
        self.doIt.setObjectName(_fromUtf8("doIt"))
        self.do_layout.addWidget(self.doIt)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20,
                                            QtWidgets.QSizePolicy.Expanding,
                                            QtWidgets.QSizePolicy.Minimum)
        self.do_layout.addItem(spacerItem6)
        self.verticalLayout_3.addLayout(self.do_layout)

        self.retranslateUi(shotsub_anim_Dialog)
        QtCore.QMetaObject.connectSlotsByName(shotsub_anim_Dialog)

    def retranslateUi(self, shotsub_anim_Dialog):
        shotsub_anim_Dialog.setWindowTitle(_translate("shotsub_anim_Dialog", "ShotSub - anim", None))
        self.header.setText(_translate("shotsub_anim_Dialog", "shotsub - anim", None))
        self.headerVer.setText(_translate("shotsub_anim_Dialog", "v1.0", None))
        self.importFrom.setText(_translate("shotsub_anim_Dialog", "Import from", None))
        self.showUI.setItemText(0, _translate("shotsub_anim_Dialog", "select show", None))
        self.seqUI.setItemText(0, _translate("shotsub_anim_Dialog", "select seq", None))
        self.shotUI.setItemText(0, _translate("shotsub_anim_Dialog", "select shot", None))
        self.plateType.setText(_translate("shotsub_anim_Dialog", "Plate Type", None))
        self.plateTypeUI.setItemText(0, _translate("shotsub_anim_Dialog", "select shot", None))
        self.scene.setText(_translate("shotsub_anim_Dialog", "Matchmove", None))
        self.sceneUI.setItemText(0, _translate("shotsub_anim_Dialog", "select plate type", None))
        self.sceneInfo.setTitle(_translate("shotsub_anim_Dialog", "Scene Info", None))
        self.publisher.setText(_translate("shotsub_anim_Dialog", "Publisher", None))
        self.ver.setText(_translate("shotsub_anim_Dialog", "Version", None))
        self.date.setText(_translate("shotsub_anim_Dialog", "Date", None))
        self.comment.setText(_translate("shotsub_anim_Dialog", "Comment", None))
        self.frameRange.setText(_translate("shotsub_anim_Dialog", "Frame Range", None))
        self.renderRes.setText(_translate("shotsub_anim_Dialog", "Render Res", None))
        self.overscanUI.setText(_translate("shotsub_anim_Dialog", "Overscan", None))
        self.stereoUI.setText(_translate("shotsub_anim_Dialog", "Stereo", None))
        self.options.setTitle(_translate("shotsub_anim_Dialog", "Import Options", None))
        self.layoutCameraUI.setText(_translate("shotsub_anim_Dialog", "Create Layout Camera", None))
        self.doIt.setText(_translate("shotsub_anim_Dialog", "Import", None))

