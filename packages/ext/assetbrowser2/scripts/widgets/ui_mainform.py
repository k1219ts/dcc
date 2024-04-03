# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainform.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from pymodule.Qt.QtCore import *
from pymodule.Qt.QtGui import *
from pymodule.Qt.QtWidgets import *

from libs.clickableQLabel import ClickableQLabel
from customWidget.Category import CategoryTreeWidget
from customWidget.Item import ItemView
from customWidget.Bookmark import BookmarkView

import resources_rc

class Ui_MainForm(object):
    def setupUi(self, MainForm):
        if not MainForm.objectName():
            MainForm.setObjectName(u"MainForm")
        MainForm.setWindowModality(Qt.NonModal)
        MainForm.resize(1280, 887)
        MainForm.setWindowOpacity(1.000000000000000)
        MainForm.setStyleSheet(u"background-color:rgb(72, 72, 72)")
        self.verticalLayout_2 = QVBoxLayout(MainForm)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 5, 0, 0)
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.titleLabel = QLabel(MainForm)
        self.titleLabel.setObjectName(u"titleLabel")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.titleLabel.sizePolicy().hasHeightForWidth())
        self.titleLabel.setSizePolicy(sizePolicy)
        self.titleLabel.setMinimumSize(QSize(285, 40))
        self.titleLabel.setMaximumSize(QSize(285, 40))
        font = QFont()
        font.setPointSize(15)
        self.titleLabel.setFont(font)

        self.horizontalLayout_2.addWidget(self.titleLabel)

        self.horizontalSpacer = QSpacerItem(40, 40, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer)

        self.db1RadioButton = QRadioButton(MainForm)
        self.db1RadioButton.setObjectName(u"db1RadioButton")
        self.db1RadioButton.setStyleSheet(u"color:white; font:bold 14px;")
        self.db1RadioButton.setChecked(True)

        self.horizontalLayout_2.addWidget(self.db1RadioButton)

        self.db2RadioButton = QRadioButton(MainForm)
        self.db2RadioButton.setObjectName(u"db2RadioButton")
        self.db2RadioButton.setStyleSheet(u"color:white; font:bold 14px;")

        self.horizontalLayout_2.addWidget(self.db2RadioButton)

        self.horizontalSpacer_2 = QSpacerItem(40, 40, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_2)

        self.userNameLabel = QLabel(MainForm)
        self.userNameLabel.setObjectName(u"userNameLabel")
        sizePolicy.setHeightForWidth(self.userNameLabel.sizePolicy().hasHeightForWidth())
        self.userNameLabel.setSizePolicy(sizePolicy)
        self.userNameLabel.setMaximumSize(QSize(150, 40))
        self.userNameLabel.setStyleSheet(u"color:white; font:bold 20px;")

        self.horizontalLayout_2.addWidget(self.userNameLabel)

        self.search_combobox = QComboBox(MainForm)
        self.search_combobox.setObjectName(u"search_combobox")
        self.search_combobox.setMinimumSize(QSize(100, 40))
        self.search_combobox.setStyleSheet(u"QComboBox {color:#cccccc;}\n"
"QComboBox::item:selected {\n"
"	background-color: #2a2d2e;\n"
"}")

        self.horizontalLayout_2.addWidget(self.search_combobox)

        self.searchEdit = QLineEdit(MainForm)
        self.searchEdit.setObjectName(u"searchEdit")
        sizePolicy.setHeightForWidth(self.searchEdit.sizePolicy().hasHeightForWidth())
        self.searchEdit.setSizePolicy(sizePolicy)
        self.searchEdit.setMaximumSize(QSize(200, 40))
        font1 = QFont()
        font1.setPointSize(14)
        font1.setBold(True)
        self.searchEdit.setFont(font1)
        self.searchEdit.setStyleSheet(u"color:white; background-color:#383838")
        self.searchEdit.setFrame(False)

        self.horizontalLayout_2.addWidget(self.searchEdit)


        self.verticalLayout_2.addLayout(self.horizontalLayout_2)

        self.line = QFrame(MainForm)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.verticalLayout_2.addWidget(self.line)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(-1, 12, -1, -1)
        self.widget_category = QWidget(MainForm)
        self.widget_category.setObjectName(u"widget_category")
        self.widget_category.setMaximumSize(QSize(200, 16777215))
        self.verticalLayout_4 = QVBoxLayout(self.widget_category)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_2 = QLabel(self.widget_category)
        self.label_2.setObjectName(u"label_2")
        font2 = QFont()
        font2.setBold(True)
        self.label_2.setFont(font2)
        self.label_2.setStyleSheet(u"color: #cccccc; padding-left: 5;")

        self.horizontalLayout_3.addWidget(self.label_2)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_4)

        self.update_category_lbl = ClickableQLabel(self.widget_category)
        self.update_category_lbl.setObjectName(u"update_category_lbl")
        sizePolicy1 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.update_category_lbl.sizePolicy().hasHeightForWidth())
        self.update_category_lbl.setSizePolicy(sizePolicy1)
        self.update_category_lbl.setPixmap(QPixmap(u":/Resources/icon/reload.png"))

        self.horizontalLayout_3.addWidget(self.update_category_lbl)

        self.add_category_lbl = ClickableQLabel(self.widget_category)
        self.add_category_lbl.setObjectName(u"add_category_lbl")
        sizePolicy1.setHeightForWidth(self.add_category_lbl.sizePolicy().hasHeightForWidth())
        self.add_category_lbl.setSizePolicy(sizePolicy1)
        self.add_category_lbl.setPixmap(QPixmap(u":/Resources/icon/plus-2-16.png"))

        self.horizontalLayout_3.addWidget(self.add_category_lbl)


        self.verticalLayout_4.addLayout(self.horizontalLayout_3)

        self.category_tree_widget = CategoryTreeWidget(self.widget_category)
        __qtreewidgetitem = QTreeWidgetItem()
        __qtreewidgetitem.setText(0, u"1");
        self.category_tree_widget.setHeaderItem(__qtreewidgetitem)
        self.category_tree_widget.setObjectName(u"category_tree_widget")
        self.category_tree_widget.setMaximumSize(QSize(300, 16777215))
        self.category_tree_widget.setStyleSheet(u"QTreeView {color: #C2C2C2; font: bold 14px; width: 100px; background-color: #383838; border: 1px solid #3d3d3d;}\n"
"QTreeView::item:selected {background-color: #53728e;color: #C2C2C2;}\n"
"QTreeView::item:hover {border: 1px solid #53728e;}\n"
"QTreeView:verticalScrollBar {background-color: #FFA91D;alternate-background-color: #FF0000;}")
        self.category_tree_widget.setFrameShape(QFrame.NoFrame)
        self.category_tree_widget.setHeaderHidden(True)

        self.verticalLayout_4.addWidget(self.category_tree_widget)


        self.horizontalLayout.addWidget(self.widget_category)

        self.widget_item = QWidget(MainForm)
        self.widget_item.setObjectName(u"widget_item")
        sizePolicy.setHeightForWidth(self.widget_item.sizePolicy().hasHeightForWidth())
        self.widget_item.setSizePolicy(sizePolicy)
        self.verticalLayout_3 = QVBoxLayout(self.widget_item)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(12, 0, 12, 0)
        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.label_4 = QLabel(self.widget_item)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setFont(font2)
        self.label_4.setStyleSheet(u"color: #cccccc; padding-left: 5;")

        self.horizontalLayout_5.addWidget(self.label_4)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_3)

        self.asset_count_lbl = QLabel(self.widget_item)
        self.asset_count_lbl.setObjectName(u"asset_count_lbl")
        sizePolicy2 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.asset_count_lbl.sizePolicy().hasHeightForWidth())
        self.asset_count_lbl.setSizePolicy(sizePolicy2)
        self.asset_count_lbl.setMinimumSize(QSize(0, 23))
        self.asset_count_lbl.setStyleSheet(u"QLabel {color:#FFFFFF; padding-right: 15}")

        self.horizontalLayout_5.addWidget(self.asset_count_lbl)

        self.add_item_lbl = ClickableQLabel(self.widget_item)
        self.add_item_lbl.setObjectName(u"add_item_lbl")
        sizePolicy1.setHeightForWidth(self.add_item_lbl.sizePolicy().hasHeightForWidth())
        self.add_item_lbl.setSizePolicy(sizePolicy1)
        self.add_item_lbl.setPixmap(QPixmap(u":/Resources/icon/plus-2-16.png"))

        self.horizontalLayout_5.addWidget(self.add_item_lbl)


        self.verticalLayout_3.addLayout(self.horizontalLayout_5)

        self.item_list_view = ItemView(self.widget_item)
        self.item_list_view.setObjectName(u"item_list_view")
        self.item_list_view.setStyleSheet(u"QListView {color: #C2C2C2; font: bold 14px; width: 100px; background-color: #383838; border: 1px solid #3d3d3d;}\n"
"QListView::item:selected {background-color: #53728e;color: #C2C2C2;}\n"
"QListView:verticalScrollBar {background-color: #FFA91D;alternate-background-color: #FF0000;}")

        self.verticalLayout_3.addWidget(self.item_list_view)


        self.horizontalLayout.addWidget(self.widget_item)

        self.widget_preview = QWidget(MainForm)
        self.widget_preview.setObjectName(u"widget_preview")
        self.widget_preview.setMaximumSize(QSize(320, 16777215))
        self.verticalLayout = QVBoxLayout(self.widget_preview)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.label_3 = QLabel(self.widget_preview)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setFont(font2)
        self.label_3.setStyleSheet(u"color:#cccccc;")
        self.label_3.setMargin(5)

        self.verticalLayout.addWidget(self.label_3)

        self.thumbnail_list_widget = QListWidget(self.widget_preview)
        self.thumbnail_list_widget.setObjectName(u"thumbnail_list_widget")
        sizePolicy1.setHeightForWidth(self.thumbnail_list_widget.sizePolicy().hasHeightForWidth())
        self.thumbnail_list_widget.setSizePolicy(sizePolicy1)
        self.thumbnail_list_widget.setMinimumSize(QSize(320, 360))
        self.thumbnail_list_widget.setMaximumSize(QSize(320, 360))
        self.thumbnail_list_widget.setStyleSheet(u"background-color: #383838;")
        self.thumbnail_list_widget.setFrameShape(QFrame.NoFrame)
        self.thumbnail_list_widget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.thumbnail_list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.thumbnail_list_widget.setDragDropMode(QAbstractItemView.NoDragDrop)
        self.thumbnail_list_widget.setMovement(QListView.Static)
        self.thumbnail_list_widget.setViewMode(QListView.IconMode)

        self.verticalLayout.addWidget(self.thumbnail_list_widget)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.horizontalLayout_6.setContentsMargins(-1, 5, -1, -1)
        self.fileSummaryLabel = QLabel(self.widget_preview)
        self.fileSummaryLabel.setObjectName(u"fileSummaryLabel")
        self.fileSummaryLabel.setFont(font2)
        self.fileSummaryLabel.setStyleSheet(u"color:#cccccc;")
        self.fileSummaryLabel.setMargin(5)

        self.horizontalLayout_6.addWidget(self.fileSummaryLabel)


        self.verticalLayout.addLayout(self.horizontalLayout_6)

        self.fileSummaryGridLayout = QGridLayout()
        self.fileSummaryGridLayout.setObjectName(u"fileSummaryGridLayout")
        self.tagLabel = QLabel(self.widget_preview)
        self.tagLabel.setObjectName(u"tagLabel")
        self.tagLabel.setStyleSheet(u"color:#cccccc; background-color: #383838;")

        self.fileSummaryGridLayout.addWidget(self.tagLabel, 5, 0, 1, 2)

        self.filePathLabel = QLabel(self.widget_preview)
        self.filePathLabel.setObjectName(u"filePathLabel")
        self.filePathLabel.setStyleSheet(u"color:#cccccc; background-color: #383838;")
        self.filePathLabel.setIndent(16)

        self.fileSummaryGridLayout.addWidget(self.filePathLabel, 2, 0, 1, 2)

        self.commentText = QPlainTextEdit(self.widget_preview)
        self.commentText.setObjectName(u"commentText")
        sizePolicy3 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.commentText.sizePolicy().hasHeightForWidth())
        self.commentText.setSizePolicy(sizePolicy3)
        self.commentText.setMaximumSize(QSize(16777215, 100))
        self.commentText.setStyleSheet(u"color:#cccccc; background-color: #383838;")
        self.commentText.setFrameShape(QFrame.NoFrame)
        self.commentText.setReadOnly(True)

        self.fileSummaryGridLayout.addWidget(self.commentText, 7, 0, 1, 2)

        self.assetNameLabel = QLabel(self.widget_preview)
        self.assetNameLabel.setObjectName(u"assetNameLabel")
        self.assetNameLabel.setStyleSheet(u"color:#cccccc; background-color: #383838;")

        self.fileSummaryGridLayout.addWidget(self.assetNameLabel, 1, 0, 1, 2)

        self.uploadUserLabel = QLabel(self.widget_preview)
        self.uploadUserLabel.setObjectName(u"uploadUserLabel")
        self.uploadUserLabel.setStyleSheet(u"color:#cccccc; background-color: #383838;")

        self.fileSummaryGridLayout.addWidget(self.uploadUserLabel, 8, 0, 1, 1)

        self.uploadTimeLabel = QLabel(self.widget_preview)
        self.uploadTimeLabel.setObjectName(u"uploadTimeLabel")
        self.uploadTimeLabel.setStyleSheet(u"color:#cccccc; background-color: #383838;")
        self.uploadTimeLabel.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.fileSummaryGridLayout.addWidget(self.uploadTimeLabel, 8, 1, 1, 1)

        self.tag_list_widget = QListWidget(self.widget_preview)
        self.tag_list_widget.setObjectName(u"tag_list_widget")
        sizePolicy3.setHeightForWidth(self.tag_list_widget.sizePolicy().hasHeightForWidth())
        self.tag_list_widget.setSizePolicy(sizePolicy3)
        self.tag_list_widget.setMaximumSize(QSize(16777215, 80))
        self.tag_list_widget.setStyleSheet(u"QListWidget {\n"
"	padding: 2px;\n"
"	background-color: #383838;\n"
"}\n"
"\n"
"QListWidget::item {\n"
"	border: 1px solid #518e55;\n"
"	border-radius: 3px;\n"
"	background-color: #3b5045;\n"
"	margin: 2px;\n"
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

        self.fileSummaryGridLayout.addWidget(self.tag_list_widget, 6, 0, 1, 2)

        self.fileSizeLabel = QLabel(self.widget_preview)
        self.fileSizeLabel.setObjectName(u"fileSizeLabel")
        self.fileSizeLabel.setStyleSheet(u"color:#cccccc; background-color: #383838;")

        self.fileSummaryGridLayout.addWidget(self.fileSizeLabel, 3, 0, 1, 1)

        self.statusLabel_2 = QLabel(self.widget_preview)
        self.statusLabel_2.setObjectName(u"statusLabel_2")
        self.statusLabel_2.setStyleSheet(u"color:#cccccc; background-color: #383838;")

        self.fileSummaryGridLayout.addWidget(self.statusLabel_2, 3, 1, 1, 1)


        self.verticalLayout.addLayout(self.fileSummaryGridLayout)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(-1, 5, -1, -1)
        self.label = QLabel(self.widget_preview)
        self.label.setObjectName(u"label")
        self.label.setFont(font2)
        self.label.setStyleSheet(u"color:#cccccc;")
        self.label.setMargin(5)

        self.horizontalLayout_4.addWidget(self.label)

        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_5)

        self.reload_bookmark_lbl = ClickableQLabel(self.widget_preview)
        self.reload_bookmark_lbl.setObjectName(u"reload_bookmark_lbl")
        sizePolicy1.setHeightForWidth(self.reload_bookmark_lbl.sizePolicy().hasHeightForWidth())
        self.reload_bookmark_lbl.setSizePolicy(sizePolicy1)
        self.reload_bookmark_lbl.setPixmap(QPixmap(u":/Resources/icon/reload.png"))

        self.horizontalLayout_4.addWidget(self.reload_bookmark_lbl)


        self.verticalLayout.addLayout(self.horizontalLayout_4)

        self.bookmark_list_view = BookmarkView(self.widget_preview)
        self.bookmark_list_view.setObjectName(u"bookmark_list_view")
        self.bookmark_list_view.setStyleSheet(u"color: #C2C2C2; font: 12px; background-color: #383838")

        self.verticalLayout.addWidget(self.bookmark_list_view)


        self.horizontalLayout.addWidget(self.widget_preview)


        self.verticalLayout_2.addLayout(self.horizontalLayout)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.category_count_lbl = QLabel(MainForm)
        self.category_count_lbl.setObjectName(u"category_count_lbl")
        self.category_count_lbl.setStyleSheet(u"color:white; font:10px;")
        self.category_count_lbl.setMargin(5)

        self.horizontalLayout_7.addWidget(self.category_count_lbl)

        self.horizontalSpacer_6 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_7.addItem(self.horizontalSpacer_6)

        self.statusLabel = QLabel(MainForm)
        self.statusLabel.setObjectName(u"statusLabel")
        self.statusLabel.setStyleSheet(u"color:white; font:10px;")
        self.statusLabel.setMargin(5)

        self.horizontalLayout_7.addWidget(self.statusLabel)


        self.verticalLayout_2.addLayout(self.horizontalLayout_7)

        QWidget.setTabOrder(self.db1RadioButton, self.db2RadioButton)
        QWidget.setTabOrder(self.db2RadioButton, self.search_combobox)
        QWidget.setTabOrder(self.search_combobox, self.searchEdit)
        QWidget.setTabOrder(self.searchEdit, self.category_tree_widget)
        QWidget.setTabOrder(self.category_tree_widget, self.item_list_view)
        QWidget.setTabOrder(self.item_list_view, self.thumbnail_list_widget)
        QWidget.setTabOrder(self.thumbnail_list_widget, self.tag_list_widget)
        QWidget.setTabOrder(self.tag_list_widget, self.commentText)
        QWidget.setTabOrder(self.commentText, self.bookmark_list_view)

        self.retranslateUi(MainForm)

        QMetaObject.connectSlotsByName(MainForm)
    # setupUi

    def retranslateUi(self, MainForm):
        MainForm.setWindowTitle(QCoreApplication.translate("MainForm", u"Form", None))
        self.titleLabel.setText(QCoreApplication.translate("MainForm", u"<html><head/><body><p><img src=\":/Resources/USDattr.png\" width=\"45\" height=\"45\" style=\"float: left;\"/><font color=\"#1F91D0\"><font size=\"5\">ASSETLIB BROWSER</span></p></body></html>", None))
        self.db1RadioButton.setText(QCoreApplication.translate("MainForm", u"ASSETLIB", None))
        self.db2RadioButton.setText(QCoreApplication.translate("MainForm", u"ASSETLIB2", None))
        self.userNameLabel.setText(QCoreApplication.translate("MainForm", u"daeseok.chae", None))
        self.searchEdit.setText("")
        self.searchEdit.setPlaceholderText(QCoreApplication.translate("MainForm", u"Search Assets", None))
        self.label_2.setText(QCoreApplication.translate("MainForm", u"CATEGORY", None))
#if QT_CONFIG(tooltip)
        self.update_category_lbl.setToolTip(QCoreApplication.translate("MainForm", u"Reload", None))
#endif // QT_CONFIG(tooltip)
        self.update_category_lbl.setText("")
#if QT_CONFIG(tooltip)
        self.add_category_lbl.setToolTip(QCoreApplication.translate("MainForm", u"Add Category", None))
#endif // QT_CONFIG(tooltip)
        self.add_category_lbl.setText("")
        self.label_4.setText(QCoreApplication.translate("MainForm", u"ITEM", None))
        self.asset_count_lbl.setText(QCoreApplication.translate("MainForm", u"TextLabel", None))
#if QT_CONFIG(tooltip)
        self.add_item_lbl.setToolTip(QCoreApplication.translate("MainForm", u"Add Item", None))
#endif // QT_CONFIG(tooltip)
        self.add_item_lbl.setText("")
        self.label_3.setText(QCoreApplication.translate("MainForm", u"PREVIEW", None))
        self.fileSummaryLabel.setText(QCoreApplication.translate("MainForm", u"FILE SUMMARY", None))
        self.tagLabel.setText(QCoreApplication.translate("MainForm", u"Tag:", None))
        self.filePathLabel.setText(QCoreApplication.translate("MainForm", u"/136arch/bds/asset/env/factory", None))
        self.commentText.setPlainText(QCoreApplication.translate("MainForm", u"PEA /\n"
"\uc548\uac1c\uc18d \uba40\ub9ac \ubcf4\uc774\ub294 \ud3d0\uacf5\uc7a5 \uc804\uacbd\n"
"(\ubaa8\ud329)", None))
        self.assetNameLabel.setText(QCoreApplication.translate("MainForm", u"factory", None))
        self.uploadUserLabel.setText(QCoreApplication.translate("MainForm", u"Uploaded by rndvfx", None))
        self.uploadTimeLabel.setText(QCoreApplication.translate("MainForm", u"2021-02-25 10:46:43", None))
        self.fileSizeLabel.setText(QCoreApplication.translate("MainForm", u"Size: 2.2G", None))
        self.statusLabel_2.setText(QCoreApplication.translate("MainForm", u"Status: Approved", None))
        self.label.setText(QCoreApplication.translate("MainForm", u"BOOKMARK", None))
#if QT_CONFIG(tooltip)
        self.reload_bookmark_lbl.setToolTip(QCoreApplication.translate("MainForm", u"Reload", None))
#endif // QT_CONFIG(tooltip)
        self.reload_bookmark_lbl.setText("")
        self.category_count_lbl.setText(QCoreApplication.translate("MainForm", u"progress....", None))
        self.statusLabel.setText(QCoreApplication.translate("MainForm", u"progress....", None))
    # retranslateUi

