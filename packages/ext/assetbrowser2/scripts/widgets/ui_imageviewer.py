# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'imageviewer.ui'
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

class Ui_ImageViewer(object):
    def setupUi(self, ImageViewer):
        if not ImageViewer.objectName():
            ImageViewer.setObjectName(u"ImageViewer")
        ImageViewer.resize(884, 616)
        ImageViewer.setStyleSheet(u"background-color: black;")
        self.verticalLayout = QVBoxLayout(ImageViewer)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.left_lbl = ClickableQLabel(ImageViewer)
        self.left_lbl.setObjectName(u"left_lbl")
        self.left_lbl.setPixmap(QPixmap(u":/Resources/icon/arrow-left.png"))

        self.horizontalLayout.addWidget(self.left_lbl)

        self.scroll_area = QScrollArea(ImageViewer)
        self.scroll_area.setObjectName(u"scroll_area")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 734, 612))
        self.scroll_area.setWidget(self.scrollAreaWidgetContents)

        self.horizontalLayout.addWidget(self.scroll_area)

        self.right_lbl = ClickableQLabel(ImageViewer)
        self.right_lbl.setObjectName(u"right_lbl")
        self.right_lbl.setPixmap(QPixmap(u":/Resources/icon/arrow-right.png"))

        self.horizontalLayout.addWidget(self.right_lbl)


        self.verticalLayout.addLayout(self.horizontalLayout)


        self.retranslateUi(ImageViewer)

        QMetaObject.connectSlotsByName(ImageViewer)
    # setupUi

    def retranslateUi(self, ImageViewer):
        ImageViewer.setWindowTitle(QCoreApplication.translate("ImageViewer", u"Dialog", None))
        self.left_lbl.setText("")
        self.right_lbl.setText("")
    # retranslateUi

