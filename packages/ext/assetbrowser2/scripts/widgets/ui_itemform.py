# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'itemform.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from pymodule.Qt.QtCore import *
from pymodule.Qt.QtGui import *
from pymodule.Qt.QtWidgets import *

from libs.imageQListWidget import ImageQListWidget
from widgets.tagmanager import TagManager


class Ui_ItemForm(object):
    def setupUi(self, ItemForm):
        if not ItemForm.objectName():
            ItemForm.setObjectName(u"ItemForm")
        ItemForm.resize(1039, 798)
        ItemForm.setStyleSheet(u"background-color: #484848; font: 12px;")
        self.verticalLayout_2 = QVBoxLayout(ItemForm)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.splitter = QSplitter(ItemForm)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Horizontal)
        self.layoutWidget = QWidget(self.splitter)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.verticalLayout = QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.storage_file_path_edit = QLineEdit(self.layoutWidget)
        self.storage_file_path_edit.setObjectName(u"storage_file_path_edit")
        self.storage_file_path_edit.setMinimumSize(QSize(0, 24))
        self.storage_file_path_edit.setStyleSheet(u"color: #cccccc; background-color: #383838;")
        self.storage_file_path_edit.setFrame(False)

        self.gridLayout.addWidget(self.storage_file_path_edit, 5, 1, 1, 1)

        self.label_11 = QLabel(self.layoutWidget)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setStyleSheet(u"color: #e2c08d;")
        self.label_11.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.label_11, 11, 0, 1, 1)

        self.label_5 = QLabel(self.layoutWidget)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setStyleSheet(u"color: #e2c08d;")
        self.label_5.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.label_5, 3, 0, 1, 1)

        self.label_4 = QLabel(self.layoutWidget)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setStyleSheet(u"color: #e2c08d;")
        self.label_4.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.label_4, 2, 0, 1, 1)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.set_preview_btn = QPushButton(self.layoutWidget)
        self.set_preview_btn.setObjectName(u"set_preview_btn")
        self.set_preview_btn.setStyleSheet(u"color: #81b88b;")

        self.horizontalLayout_2.addWidget(self.set_preview_btn)

        self.remove_image_btn = QPushButton(self.layoutWidget)
        self.remove_image_btn.setObjectName(u"remove_image_btn")
        self.remove_image_btn.setStyleSheet(u"color: #f88070;")

        self.horizontalLayout_2.addWidget(self.remove_image_btn)


        self.gridLayout.addLayout(self.horizontalLayout_2, 12, 1, 1, 2)

        self.tag_edit = QLineEdit(self.layoutWidget)
        self.tag_edit.setObjectName(u"tag_edit")
        self.tag_edit.setMinimumSize(QSize(0, 24))
        self.tag_edit.setStyleSheet(u"color: #cccccc; background-color: #383838;")
        self.tag_edit.setFrame(False)

        self.gridLayout.addWidget(self.tag_edit, 8, 1, 1, 2)

        self.label_6 = QLabel(self.layoutWidget)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setStyleSheet(u"color: #e2c08d;")
        self.label_6.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.label_6, 6, 0, 1, 1)

        self.comment_edit = QPlainTextEdit(self.layoutWidget)
        self.comment_edit.setObjectName(u"comment_edit")
        self.comment_edit.setStyleSheet(u"color: #cccccc; background-color: #383838;")
        self.comment_edit.setFrameShape(QFrame.NoFrame)

        self.gridLayout.addWidget(self.comment_edit, 10, 1, 1, 2)

        self.label = QLabel(self.layoutWidget)
        self.label.setObjectName(u"label")
        self.label.setStyleSheet(u"color: #e2c08d;")
        self.label.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.label_3 = QLabel(self.layoutWidget)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setStyleSheet(u"color: #e2c08d;")
        self.label_3.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.label_3, 5, 0, 1, 1)

        self.files_usdfile_edit = QLineEdit(self.layoutWidget)
        self.files_usdfile_edit.setObjectName(u"files_usdfile_edit")
        self.files_usdfile_edit.setMinimumSize(QSize(0, 24))
        self.files_usdfile_edit.setStyleSheet(u"color: #cccccc; background-color: #383838;")
        self.files_usdfile_edit.setFrame(False)

        self.gridLayout.addWidget(self.files_usdfile_edit, 4, 1, 1, 1)

        self.label_10 = QLabel(self.layoutWidget)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setStyleSheet(u"color: #f88070;")
        self.label_10.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.label_10, 8, 0, 1, 1)

        self.files_preview_btn = QToolButton(self.layoutWidget)
        self.files_preview_btn.setObjectName(u"files_preview_btn")
        self.files_preview_btn.setStyleSheet(u"color: #e2c08d;")

        self.gridLayout.addWidget(self.files_preview_btn, 3, 2, 1, 1)

        self.label_2 = QLabel(self.layoutWidget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setStyleSheet(u"color: #e2c08d;")
        self.label_2.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)

        self.files_preview_edit = QLineEdit(self.layoutWidget)
        self.files_preview_edit.setObjectName(u"files_preview_edit")
        self.files_preview_edit.setMinimumSize(QSize(0, 24))
        self.files_preview_edit.setStyleSheet(u"color: #cccccc; background-color: #383838;")
        self.files_preview_edit.setFrame(False)

        self.gridLayout.addWidget(self.files_preview_edit, 3, 1, 1, 1)

        self.storage_file_path_btn = QToolButton(self.layoutWidget)
        self.storage_file_path_btn.setObjectName(u"storage_file_path_btn")
        self.storage_file_path_btn.setStyleSheet(u"color: #e2c08d;")

        self.gridLayout.addWidget(self.storage_file_path_btn, 5, 2, 1, 1)

        self.image_list = ImageQListWidget(self.layoutWidget)
        self.image_list.setObjectName(u"image_list")
        self.image_list.setAcceptDrops(True)
        self.image_list.setStyleSheet(u"color: #cccccc; background-color: #383838;")
        self.image_list.setFrameShape(QFrame.NoFrame)
        self.image_list.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.gridLayout.addWidget(self.image_list, 11, 1, 1, 2)

        self.files_usdfile_label = QLabel(self.layoutWidget)
        self.files_usdfile_label.setObjectName(u"files_usdfile_label")
        self.files_usdfile_label.setStyleSheet(u"color: #e2c08d;")
        self.files_usdfile_label.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.files_usdfile_label, 4, 0, 1, 1)

        self.files_path_btn = QToolButton(self.layoutWidget)
        self.files_path_btn.setObjectName(u"files_path_btn")
        self.files_path_btn.setStyleSheet(u"color: #e2c08d;")

        self.gridLayout.addWidget(self.files_path_btn, 4, 2, 1, 1)

        self.label_7 = QLabel(self.layoutWidget)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setStyleSheet(u"color: #e2c08d;")
        self.label_7.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.label_7, 7, 0, 1, 1)

        self.label_8 = QLabel(self.layoutWidget)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setStyleSheet(u"color: #e2c08d;")
        self.label_8.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.label_8, 10, 0, 1, 1)

        self.storage_file_size_edit = QLineEdit(self.layoutWidget)
        self.storage_file_size_edit.setObjectName(u"storage_file_size_edit")
        self.storage_file_size_edit.setMinimumSize(QSize(0, 24))
        self.storage_file_size_edit.setStyleSheet(u"color: #cccccc; background-color: #383838;")
        self.storage_file_size_edit.setFrame(False)

        self.gridLayout.addWidget(self.storage_file_size_edit, 6, 1, 1, 2)

        self.name_edit = QLineEdit(self.layoutWidget)
        self.name_edit.setObjectName(u"name_edit")
        self.name_edit.setMinimumSize(QSize(0, 24))
        self.name_edit.setStyleSheet(u"color: #cccccc; background-color: #383838;")
        self.name_edit.setFrame(False)

        self.gridLayout.addWidget(self.name_edit, 2, 1, 1, 2)

        self.tag_list_widget = QListWidget(self.layoutWidget)
        self.tag_list_widget.setObjectName(u"tag_list_widget")
        self.tag_list_widget.setStyleSheet(u"QListWidget {\n"
"	padding: 5px;\n"
"	background-color: #383838;\n"
"}\n"
"\n"
"QListWidget::item {\n"
"	border: 1px solid #518e55;\n"
"	border-radius: 3px;\n"
"	background-color: #3b5045;\n"
"	margin: 5px;\n"
"	color: white;\n"
"}\n"
"\n"
"QListWidget::item::hover {\n"
"	background-color: #4d8453;\n"
"}")
        self.tag_list_widget.setFrameShape(QFrame.NoFrame)
        self.tag_list_widget.setDragDropMode(QAbstractItemView.NoDragDrop)
        self.tag_list_widget.setMovement(QListView.Static)
        self.tag_list_widget.setViewMode(QListView.IconMode)

        self.gridLayout.addWidget(self.tag_list_widget, 9, 1, 1, 2)

        self.label_9 = QLabel(self.layoutWidget)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setStyleSheet(u"color: #e2c08d;")
        self.label_9.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.label_9, 9, 0, 1, 1)

        self.status_combo = QComboBox(self.layoutWidget)
        self.status_combo.setObjectName(u"status_combo")
        self.status_combo.setMinimumSize(QSize(0, 24))
        self.status_combo.setStyleSheet(u"QComboBox {color:#cccccc;}\n"
"QComboBox::item:selected {\n"
"	background-color: #2a2d2e;\n"
"}")

        self.gridLayout.addWidget(self.status_combo, 7, 1, 1, 2)

        self.category_combo = QComboBox(self.layoutWidget)
        self.category_combo.setObjectName(u"category_combo")
        self.category_combo.setMinimumSize(QSize(0, 24))
        self.category_combo.setStyleSheet(u"QComboBox {color:#cccccc;}\n"
"QComboBox::item:selected {\n"
"	background-color: #2a2d2e;\n"
"}")

        self.gridLayout.addWidget(self.category_combo, 0, 1, 1, 2)

        self.sub_category_combo = QComboBox(self.layoutWidget)
        self.sub_category_combo.setObjectName(u"sub_category_combo")
        self.sub_category_combo.setMinimumSize(QSize(0, 24))
        self.sub_category_combo.setStyleSheet(u"QComboBox {color:#cccccc;}\n"
"QComboBox::item:selected {\n"
"	background-color: #2a2d2e;\n"
"}")

        self.gridLayout.addWidget(self.sub_category_combo, 1, 1, 1, 2)


        self.verticalLayout.addLayout(self.gridLayout)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)

        self.cancel_btn = QPushButton(self.layoutWidget)
        self.cancel_btn.setObjectName(u"cancel_btn")
        self.cancel_btn.setStyleSheet(u"color: #f88070;")

        self.horizontalLayout.addWidget(self.cancel_btn)

        self.ok_btn = QPushButton(self.layoutWidget)
        self.ok_btn.setObjectName(u"ok_btn")
        self.ok_btn.setStyleSheet(u"color: #81b88b;")

        self.horizontalLayout.addWidget(self.ok_btn)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.splitter.addWidget(self.layoutWidget)
        self.tag_manager_widget = TagManager(self.splitter)
        self.tag_manager_widget.setObjectName(u"tag_manager_widget")
        self.splitter.addWidget(self.tag_manager_widget)

        self.verticalLayout_2.addWidget(self.splitter)

        QWidget.setTabOrder(self.category_combo, self.sub_category_combo)
        QWidget.setTabOrder(self.sub_category_combo, self.name_edit)
        QWidget.setTabOrder(self.name_edit, self.files_preview_edit)
        QWidget.setTabOrder(self.files_preview_edit, self.files_preview_btn)
        QWidget.setTabOrder(self.files_preview_btn, self.files_usdfile_edit)
        QWidget.setTabOrder(self.files_usdfile_edit, self.files_path_btn)
        QWidget.setTabOrder(self.files_path_btn, self.storage_file_path_edit)
        QWidget.setTabOrder(self.storage_file_path_edit, self.storage_file_path_btn)
        QWidget.setTabOrder(self.storage_file_path_btn, self.storage_file_size_edit)
        QWidget.setTabOrder(self.storage_file_size_edit, self.status_combo)
        QWidget.setTabOrder(self.status_combo, self.tag_edit)
        QWidget.setTabOrder(self.tag_edit, self.tag_list_widget)
        QWidget.setTabOrder(self.tag_list_widget, self.comment_edit)
        QWidget.setTabOrder(self.comment_edit, self.image_list)
        QWidget.setTabOrder(self.image_list, self.set_preview_btn)
        QWidget.setTabOrder(self.set_preview_btn, self.remove_image_btn)
        QWidget.setTabOrder(self.remove_image_btn, self.cancel_btn)
        QWidget.setTabOrder(self.cancel_btn, self.ok_btn)

        self.retranslateUi(ItemForm)

        QMetaObject.connectSlotsByName(ItemForm)
    # setupUi

    def retranslateUi(self, ItemForm):
        ItemForm.setWindowTitle(QCoreApplication.translate("ItemForm", u"Edit Item", None))
        self.label_11.setText(QCoreApplication.translate("ItemForm", u"(n) images", None))
        self.label_5.setText(QCoreApplication.translate("ItemForm", u"files.preview", None))
        self.label_4.setText(QCoreApplication.translate("ItemForm", u"* name", None))
        self.set_preview_btn.setText(QCoreApplication.translate("ItemForm", u"Set Preview", None))
        self.remove_image_btn.setText(QCoreApplication.translate("ItemForm", u"Remove", None))
        self.label_6.setText(QCoreApplication.translate("ItemForm", u"(n) storageFileSize", None))
        self.label.setText(QCoreApplication.translate("ItemForm", u"* category", None))
        self.label_3.setText(QCoreApplication.translate("ItemForm", u"(n) storageFilePath", None))
        self.label_10.setText(QCoreApplication.translate("ItemForm", u"(deprecated) tag", None))
        self.files_preview_btn.setText(QCoreApplication.translate("ItemForm", u"...", None))
        self.label_2.setText(QCoreApplication.translate("ItemForm", u"* subCategory", None))
        self.storage_file_path_btn.setText(QCoreApplication.translate("ItemForm", u"...", None))
        self.files_usdfile_label.setText(QCoreApplication.translate("ItemForm", u"files.usdfile", None))
        self.files_path_btn.setText(QCoreApplication.translate("ItemForm", u"...", None))
        self.label_7.setText(QCoreApplication.translate("ItemForm", u"(n) status", None))
        self.label_8.setText(QCoreApplication.translate("ItemForm", u"comment", None))
        self.name_edit.setPlaceholderText(QCoreApplication.translate("ItemForm", u"name is required!", None))
        self.label_9.setText(QCoreApplication.translate("ItemForm", u"(n) tags", None))
        self.cancel_btn.setText(QCoreApplication.translate("ItemForm", u"Cancel", None))
        self.ok_btn.setText(QCoreApplication.translate("ItemForm", u"OK", None))
    # retranslateUi

