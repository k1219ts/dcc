# -*- coding: utf-8 -*-
import getpass
import sys
sys.path.append("/backstage/apps/Maya/versions/2018/global/linux/scripts/xbUtils")

from pymodule.Qt import QtCore
from pymodule.Qt import QtGui
from pymodule.Qt import QtWidgets

import resources_rc

from core import Database
from libs.hashableQListWidgetItem import HashableQListWidgetItem
from libs.ustr import ustr
from libs.utils import get_grid_size
from libs.utils import resize_pixmap
from widgets.imageviewer import ImageViewer
from widgets.itemform import ItemForm
from widgets.ui_mainform import Ui_MainForm

class MainForm(QtWidgets.QWidget):
    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent)

        # ui_path = os.path.dirname(os.path.abspath(__file__))
        # self.ui = QtCompat.loadUi(os.path.join(ui_path, "Resources", "MainFormUi.ui"), self)
        self.ui = Ui_MainForm()
        self.ui.setupUi(self)

        # instance variables
        self.thumbnails = [] # for self.ui.thumbnail_list_widget

        self.setWindowFlags(QtCore.Qt.Window)
        self.setWindowTitle("ASSETLIB BROWSER v2.1.1")

        self.ui.item_list_view.setDragEnabled(True)
        self.ui.search_combobox.addItems(["All", "Comment", "Status"])

        self.ui.thumbnail_list_widget.itemDoubleClicked.connect(self.show_image_viewer)
        # category widget
        self.ui.category_tree_widget.clicked.connect(self.category_tree_widget_clicked)
        self.ui.category_tree_widget.changed.connect(self.category_tree_widget_changed)
        self.ui.category_tree_widget.updated.connect(self.category_tree_widget_updated)
        # item view
        self.ui.item_list_view.changed.connect(self.list_view_changed)
        self.ui.item_list_view.itemConnect(self.ui.statusLabel)
        self.ui.item_list_view.item_changed.connect(self.list_view_item_changed)
        self.ui.item_list_view.item_clicked.connect(self.list_view_item_clicked)
        # bookmark view
        self.ui.bookmark_list_view.changed.connect(self.list_view_changed)
        self.ui.bookmark_list_view.itemConnect(self.ui.statusLabel)
        self.ui.bookmark_list_view.item_changed.connect(self.list_view_item_changed)
        self.ui.bookmark_list_view.item_clicked.connect(self.list_view_item_clicked)

        self.ui.tag_list_widget.itemClicked.connect(self.tag_item_clicked)

        self.ui.reload_bookmark_lbl.clicked.connect(self.reload_bookmark_list)
        self.ui.update_category_lbl.clicked.connect(self.reload_category_list)
        self.ui.add_category_lbl.clicked.connect(self.add_category)
        self.ui.add_item_lbl.clicked.connect(self.show_add_item_window)
        self.ui.db1RadioButton.clicked.connect(self.db1_clicked)
        self.ui.db2RadioButton.clicked.connect(self.db2_clicked)
        self.ui.searchEdit.returnPressed.connect(self.search_return_pressed)

        self.ui.userNameLabel.setText(getpass.getuser())
        self.clear_file_summary()

        self.ui.db2RadioButton.setChecked(True)
        self.db2_clicked()

        self.image_viewer = ImageViewer(self)

        # self.ui.bookmark_list_view.reload_list()

    def category_tree_widget_clicked(self):
        self.category_clicked()

    def category_tree_widget_changed(self):
        self.category_clicked()
        self.ui.bookmark_list_view.reload_list()

    def category_tree_widget_updated(self):
        self.reload_category_list()

    def list_view_changed(self):
        self.category_clicked()
        self.ui.bookmark_list_view.reload_list()

    def list_view_item_clicked(self, document):
        self.asset_clicked(document)

    def list_view_item_changed(self, document):
        self.asset_clicked(document)
        self.ui.bookmark_list_view.reload_list()

    def print_category_count(self):
        """하단에 Asset / Texture 의 전체 갯수를 출력하는 메소드"""
        self.ui.statusLabel.clear()
        text = "Asset Total: {},     Texture Total: {}".format(
            str(Database.gDB.item.find().count()),
            str(Database.gDB.source.find().count()))
        self.ui.category_count_lbl.setText(text)

    def show_add_item_window(self):
        """Asset / Texture 의 아이템을 생성하는 메소드"""
        category = "Default"
        sub_category = "unknown"

        find_item = self.ui.category_tree_widget.currentItem()
        if find_item:
            parent = find_item.parent()
            if parent:
                category = parent.text(0)
                sub_category = find_item.text(0)
            else:
                category = find_item.text(0)

        dialog = ItemForm(self, document=None)
        dialog.tag_changed.connect(self.category_clicked)
        dialog.update_form_for_add(category, sub_category)
        dialog.show()

    def tag_item_clicked(self, item):
        """태그를 클릭하면 DB를 검색해서 리스트에 출력하는 메소드"""
        object_id = item.data(QtCore.Qt.UserRole)
        collections = Database.TagSearch(object_id) # return [ CommandCursor1, CommandCursor2 ]

        if not collections:
            return

        self.ui.item_list_view.clear()
        self.ui.item_list_view.scrollToTop()
        self.clear_file_summary()

        items = []
        for collection in collections:
            for c in collection:
                items.append(c)

        if items:
            count = self.ui.item_list_view.add_items(items, return_count=True)
            self.ui.asset_count_lbl.setText("{} results.".format(count))

    # TODO : 우측 메뉴 비활성화를 위해서 잠시 사용
    def set_advanced_mode(self, dbname):
        """v1 DB의 기능을 제한하는 메소드"""
        if sys.platform == "darwin":
            mode = False
        elif sys.platform == "win32":
            mode = False
        else:
            mode = True

        mode = True
        if dbname == "ASSETLIB":
            mode = False

        self.ui.category_tree_widget.set_advanced_mode(mode)
        self.ui.item_list_view.set_advanced_mode(mode)
        self.ui.bookmark_list_view.set_advanced_mode(mode)

    def show_image_viewer(self, item):
        """이미지 뷰어를 보여주는 메소드"""
        if not self.thumbnails:
            return

        current_index = 0
        for row in range(self.ui.thumbnail_list_widget.count()):
            current_item = self.ui.thumbnail_list_widget.item(row)
            if current_item == item:
                current_index = row
                break

        if not self.image_viewer.isVisible():
            self.image_viewer.show()

        self.image_viewer.add_images_and_loadfile(self.thumbnails, current_index)

    def reload_bookmark_list(self):
        """북마크를 갱신하는 메소드"""
        self.ui.bookmark_list_view.reload_list()
        self.clear_file_summary()

    def reload_category_list(self):
        """카테고리를 갱신하는 메소드"""
        self.ui.category_tree_widget.reload_list()
        self.ui.item_list_view.clear()
        self.clear_file_summary()
        self.print_category_count()

    def add_category(self):
        self.ui.category_tree_widget.makeCategory()

    def asset_clicked(self, document):
        self.thumbnails = []
        self.ui.thumbnail_list_widget.clear()
        self.update_file_summary(document)

        if document.preview_path:
            images = [document.preview_path] + document.images
        else:
            images = document.images

        if not images:
            return

        grid_size = get_grid_size(len(images))
        self.ui.thumbnail_list_widget.setIconSize(grid_size)
        for image in images:
            itm = HashableQListWidgetItem()
            pixmap = resize_pixmap(image, document.status, grid_size.width()*0.95,
                                    grid_size.height()*0.95, QtCore.Qt.SmoothTransformation)
            itm.setIcon(QtGui.QIcon(pixmap))
            itm.setSizeHint(grid_size)
            self.ui.thumbnail_list_widget.addItem(itm)

        self.thumbnails = images

    def update_file_summary(self, document):
        """FILE SUMMARY 의 내용을 업데이트하는 메소드"""
        self.ui.tag_list_widget.clear()
        self.ui.assetNameLabel.setText(document.name)
        self.ui.filePathLabel.setText(document.dir_path)
        self.ui.fileSizeLabel.setText(document.storage_file_size)
        self.ui.statusLabel_2.setText(document.status)
        self.ui.tagLabel.setText(u"Tag: {}".format(",".join(document.tag)))
        for tag_object in document.tag_objects:
            item = QtWidgets.QListWidgetItem()
            item.setText(tag_object["name"])
            item.setData(QtCore.Qt.UserRole, tag_object["_id"])
            self.ui.tag_list_widget.addItem(item)
        self.ui.commentText.setPlainText(document.comment)
        self.ui.uploadUserLabel.setText(document.user)
        self.ui.uploadTimeLabel.setText(document.timestamp)

    def clear_file_summary(self):
        """FILE SUMMARY 의 내용을 지우는 메소드"""
        self.ui.asset_count_lbl.clear()
        self.ui.thumbnail_list_widget.clear()
        self.ui.assetNameLabel.clear()
        self.ui.filePathLabel.clear()
        self.ui.fileSizeLabel.clear()
        self.ui.statusLabel_2.clear()
        self.ui.tagLabel.clear()
        self.ui.tag_list_widget.clear()
        self.ui.commentText.clear()
        self.ui.uploadUserLabel.clear()
        self.ui.uploadTimeLabel.clear()

    def db1_clicked(self):
        database = "ASSETLIB"
        self.ui.db1RadioButton.setFocus()
        Database.set_database(database)
        self.initial_widget()
        self.set_advanced_mode(database)

    def db2_clicked(self):
        database = "ASSETLIB2"
        self.ui.db2RadioButton.setFocus()
        Database.set_database(database)
        self.initial_widget()
        self.set_advanced_mode(database)

    def initial_widget(self):
        self.ui.category_tree_widget.setFocus()

        self.ui.searchEdit.clear()
        self.ui.item_list_view.clear()
        self.ui.category_tree_widget.clear()
        self.clear_file_summary()
        self.print_category_count()

        self.ui.category_tree_widget.reload_list()
        self.ui.bookmark_list_view.reload_list()

    def search_return_pressed(self):
        """검색어를 통해 Asset / Texture 의 아이템을 검색하는 메소드"""
        inputText = self.ui.searchEdit.text()
        inputText = inputText.lower()

        if not inputText:
            return

        self.ui.item_list_view.clear()
        self.ui.item_list_view.scrollToTop()
        self.clear_file_summary()

        itemList = []
        # SCcursor = Database.gDB.source.find({'name': {'$exists': True}})
        # cursor = Database.gDB.item.find({'name': {'$exists': True}})
        # cursorList = [SCcursor, cursor]
        # for i in cursorList:
        #     for itr in i:
        #         if inputText in itr['name']:
        #             if not itr in itemList:
        #                 itemList.append(itr)

        #         elif inputText in itr['name'].lower():
        #             if not itr in itemList:
        #                 itemList.append(itr)

        #         elif itr['tag']:
        #             for tag in itr['tag']:
        #                 if inputText in tag.lower():
        #                     if not itr in itemList:
        #                         itemList.append(itr)
        #         else:
        #             pass

        #         if itr['comment']:
        #             if itr['comment'].lower().find(inputText) != -1:
        #                 if not itr in itemList:
        #                     itemList.append(itr)

        field = ustr(self.ui.search_combobox.currentText()).lower()
        if field == "all":
            collections = Database.Search(inputText)
        else:
            collections = Database.SimpleSearch(field, inputText)

        for collection in collections:
            for item in collection:
                if field == "all":
                    current_item = item["matches"][0]
                else:
                    current_item = item
                itemList.append(current_item)

        if itemList:
            count = self.ui.item_list_view.add_items(itemList, return_count=True)
            self.ui.asset_count_lbl.setText("{} results.".format(count))

    # TODO: 메소드 분리
    def category_clicked(self):
        """카테고리를 클릭하면 어셋(Asset, Texture)을 갱신합니다.

        """
        self.ui.item_list_view.clear()
        self.ui.item_list_view.scrollToTop()
        self.ui.searchEdit.clear()
        self.clear_file_summary()

        find_item = self.ui.category_tree_widget.currentItem()
        if find_item is None:
            return

        items = []
        if find_item.parent() is None: #Main category
            pass
        elif find_item.parent(): #sub_category:
            current = self.ui.category_tree_widget.currentItem()
            sub = current.text(0)
            main = current.parent().text(0)
            items = Database.GetItems(main, sub)
            count = self.ui.item_list_view.add_items(items, return_count=True)
            self.ui.asset_count_lbl.setText("{} results.".format(count))

        # elif find_item.parent().parent(): #child_category
        #     current =self.ui.category_tree_widget.currentItem()
        #     sub = current.text(0)
        #     main = current.parent().text(0)

        else:
            pass
