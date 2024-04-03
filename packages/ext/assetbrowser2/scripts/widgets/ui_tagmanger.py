# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'tagmanager.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from pymodule.Qt.QtCore import *
from pymodule.Qt.QtGui import *
from pymodule.Qt.QtWidgets import *

from libs.clickableQLabel import ClickableQLabel

import resources_rc

class Ui_TagManager(object):
    def setupUi(self, TagManager):
        if not TagManager.objectName():
            TagManager.setObjectName(u"TagManager")
        TagManager.resize(402, 646)
        TagManager.setStyleSheet(u"background-color: #484848;")
        self.verticalLayout = QVBoxLayout(TagManager)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(-1, 0, -1, 0)
        self.label = QLabel(TagManager)
        self.label.setObjectName(u"label")
        font = QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setStyleSheet(u"color: #cccccc;")

        self.verticalLayout.addWidget(self.label)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.result_label = QLabel(TagManager)
        self.result_label.setObjectName(u"result_label")
        self.result_label.setStyleSheet(u"color: #e2c08d;")

        self.horizontalLayout.addWidget(self.result_label)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)

        self.icon_mode_label = ClickableQLabel(TagManager)
        self.icon_mode_label.setObjectName(u"icon_mode_label")
        self.icon_mode_label.setPixmap(QPixmap(u":/Resources/icon/tag-5-16.png"))

        self.horizontalLayout.addWidget(self.icon_mode_label)

        self.list_mode_label = ClickableQLabel(TagManager)
        self.list_mode_label.setObjectName(u"list_mode_label")
        self.list_mode_label.setPixmap(QPixmap(u":/Resources/icon/list-2-16.png"))

        self.horizontalLayout.addWidget(self.list_mode_label)

        self.update_label = ClickableQLabel(TagManager)
        self.update_label.setObjectName(u"update_label")
        self.update_label.setPixmap(QPixmap(u":/Resources/icon/reload.png"))

        self.horizontalLayout.addWidget(self.update_label)

        self.add_label = ClickableQLabel(TagManager)
        self.add_label.setObjectName(u"add_label")
        self.add_label.setPixmap(QPixmap(u":/Resources/icon/plus-2-16.png"))

        self.horizontalLayout.addWidget(self.add_label)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.filter_line_edit = QLineEdit(TagManager)
        self.filter_line_edit.setObjectName(u"filter_line_edit")
        self.filter_line_edit.setMinimumSize(QSize(0, 24))
        self.filter_line_edit.setStyleSheet(u"color: #cccccc; background-color: #383838;")
        self.filter_line_edit.setFrame(False)

        self.verticalLayout.addWidget(self.filter_line_edit)

        self.list_view = QListView(TagManager)
        self.list_view.setObjectName(u"list_view")
        self.list_view.setStyleSheet(u"color: #cccccc; background-color: #383838;")

        self.verticalLayout.addWidget(self.list_view)


        self.retranslateUi(TagManager)

        QMetaObject.connectSlotsByName(TagManager)
    # setupUi

    def retranslateUi(self, TagManager):
        TagManager.setWindowTitle(QCoreApplication.translate("TagManager", u"Form", None))
        self.label.setText(QCoreApplication.translate("TagManager", u"TAG MANAGER", None))
        self.result_label.setText(QCoreApplication.translate("TagManager", u"TextLabel", None))
        self.icon_mode_label.setText("")
        self.list_mode_label.setText("")
        self.update_label.setText("")
#if QT_CONFIG(statustip)
        self.add_label.setStatusTip(QCoreApplication.translate("TagManager", u"Add Tag", None))
#endif // QT_CONFIG(statustip)
        self.add_label.setText("")
        self.filter_line_edit.setPlaceholderText(QCoreApplication.translate("TagManager", u"Filter", None))
    # retranslateUi

